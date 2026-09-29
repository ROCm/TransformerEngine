/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "ck_grouped_gemm_common.h"

namespace transformer_engine {
namespace grouped_gemm {

using mx_grouped_gemm_kargs = ck_tile::MxGroupedGemmHostArgs<>;

static constexpr ck_tile::index_t ScaleBlockSize = 32;

enum struct MxGemmPipelineType {
  CompTDMV1,
  CompTDMV2,
  CompAsync
};

template <MxGemmPipelineType PT, typename Problem>
struct MxGemmPipelineTypeSelector;

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV1, Problem> {
  using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
  using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;
  static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV1"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV2, Problem> {
  using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
  using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;
  static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV2"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompAsync, Problem> {
  using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompAsync<Problem>;
  using pipeline = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;
  static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsync"; }
};

// The TDM pipelines drive the epilogue directly; the async pipeline writes C
// through LDS and needs the CShuffle epilogue.
template <MxGemmPipelineType PT, typename Problem>
struct MxGemmEpilogueTypeSelector;

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompTDMV1, Problem> {
  using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompTDMV2, Problem> {
  using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompAsync, Problem> {
  using epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

// gfx1250 scale preshuffle.
//
// Unlike the existing MXFP8 GEMM scale swizzle defined in:
//   transformer_engine/common/swizzle/swizzle.cu
//
// CK gfx1250 WMMA kernels expect scales in the layout below:
//
// Input scales are logically [MN, KScale]
//
// The output layout groups KScale into tiles of 4 (= 128 / ScaleBlockSize)
// and additionally blocks M into chunks of 32 rows:
//
//  [MN, KScale]
//    -> [MN/32, KScale/4, 32, 4]
//
// For A scales, rows=M and output_rows is M padded to M_Warp_Tile.
// For B scales, rows=N and output_rows is currently N.
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
__global__ void preshuffle_scale_gfx1250_kernel(const ScaleType *__restrict__ src,
                                                ScaleType *__restrict__ dst,
                                                int actual_rows,
                                                int output_rows,
                                                int KScale) {
  static_assert(ScaleBlockSize == 32 && sizeof(ScaleType) == 1,
                "gfx1250 scale preshuffle only supports 8-bit scale with ScaleBlockSize=32");
  constexpr int MPerXdlops = 16;
  constexpr int KPerXdlops = 128;
  constexpr int MNPack = 2;
  constexpr int KPack  = 1;
  constexpr int MNStep = MPerXdlops;                  // 16
  constexpr int KStep  = KPerXdlops / ScaleBlockSize; // 4
  const int K0 = KScale / (KPack * KStep);
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total  = output_rows * KScale;
  if (linear >= total) {
    return;
  }
  const int mn = linear / KScale;
  const int k  = linear % KScale;
  const int iMNRepeat = mn / (MNStep * MNPack);
  const int tempmn    = mn % (MNStep * MNPack);
  const int iKRepeat = k / (KStep * KPack);
  const int tempk    = k % (KStep * KPack);
  const int outputIndex =
      (iMNRepeat * MNPack * MNStep) * (KStep * KPack * K0) +
      (iKRepeat * KStep * KPack) * (MNStep * MNPack) +
      tempmn * (KStep * KPack) +
      tempk;
  ScaleType value{};
  if (mn < actual_rows) {
    if constexpr (KStride) {
      value = src[mn * KScale + k];
    } else {
      value = src[k * actual_rows + mn];
    }
  }
  dst[outputIndex] = value;
}

template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType *src,
                                   ScaleType *dst,
                                   int actual_rows,
                                   int output_rows,
                                   int KScale,
                                   hipStream_t stream) {
  constexpr int KPerXdlops = 128;
  constexpr int KStep      = KPerXdlops / ScaleBlockSize; // 4
  if (KScale % KStep != 0) {
    NVTE_ERROR("preshuffle_scale_gfx1250: KScale must be a multiple of 4, "
               "i.e. original K must be a multiple of 128 for ScaleBlockSize=32.");
  }
  const int total = output_rows * KScale;
  constexpr int block_size = 256;
  const int grid_size      = (total + block_size - 1) / block_size;
  hipLaunchKernelGGL((preshuffle_scale_gfx1250_kernel<ScaleType, ScaleBlockSize, KStride>),
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     stream,
                     src,
                     dst,
                     actual_rows,
                     output_rows,
                     KScale);
  NVTE_CHECK_CUDA(hipGetLastError());
}

// gfx950 scale preshuffle.
//
// Device port of ck_tile::preShuffleScaleBuffer_gfx950 (ck_tile/host/mx_processing.hpp),
// which is a host-only loop over host memory. The index math below is copied from
// it verbatim; keep the two in sync. One thread handles one (packed_mn, packed_k)
// pair and writes the MNPack * KPack scales belonging to it.
//
// KStride selects how the source is read, matching the gfx1250 kernel:
//   true  -> src is [MN, KScale]  (TE rowwise scale_inv)
//   false -> src is [KScale, MN]  (TE columnwise_scale_inv)
template <typename ScaleType, ck_tile::index_t MNPack, ck_tile::index_t KPack,
          ck_tile::index_t XdlMNThread, ck_tile::index_t XdlKThread, bool KStride>
__global__ void preshuffle_scale_gfx950_kernel(const ScaleType *__restrict__ src,
                                               ScaleType *__restrict__ dst,
                                               int actual_rows,
                                               int output_rows,
                                               int KScale) {
  static_assert(sizeof(ScaleType) == 1,
                "gfx950 scale preshuffle only supports 8-bit scale types");
  constexpr int NumScalesPerDword = 4 / sizeof(ScaleType);
  const int MN_packed = output_rows / MNPack;
  const int K_packed  = KScale / KPack;
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= MN_packed * K_packed) {
    return;
  }
  const int packed_mn = linear / K_packed;
  const int packed_k  = linear % K_packed;
  const int mn_lane  = packed_mn % XdlMNThread;
  const int mn_group = packed_mn / XdlMNThread;
  const int k_lane   = packed_k % XdlKThread;
  const int k_group  = packed_k / XdlKThread;
  for (int ik = 0; ik < KPack; ik++) {
    for (int imn = 0; imn < MNPack; imn++) {
      const int byteIdx = ik * MNPack + imn;
      const int orig_mn = mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
      const int orig_k  = k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;
      ScaleType value{};
      if (orig_mn < actual_rows) {
        if constexpr (KStride) {
          value = src[orig_k + static_cast<int64_t>(orig_mn) * KScale];
        } else {
          value = src[orig_mn + static_cast<int64_t>(orig_k) * actual_rows];
        }
      }
      const int64_t outputIndex =
          byteIdx + static_cast<int64_t>(mn_lane) * NumScalesPerDword +
          static_cast<int64_t>(packed_k) * XdlMNThread * NumScalesPerDword +
          static_cast<int64_t>(mn_group) * XdlMNThread * NumScalesPerDword * K_packed;
      dst[outputIndex] = value;
    }
  }
}

template <typename ScaleType, ck_tile::index_t MNPack, ck_tile::index_t KPack,
          ck_tile::index_t XdlMNThread, ck_tile::index_t XdlKThread, bool KStride>
void preShuffleScaleBuffer_gfx950(const ScaleType *src,
                                  ScaleType *dst,
                                  int actual_rows,
                                  int output_rows,
                                  int KScale,
                                  hipStream_t stream) {
  if (output_rows % MNPack != 0 || KScale % KPack != 0) {
    NVTE_ERROR("preshuffle_scale_gfx950: output_rows must be a multiple of ", MNPack,
               " and KScale a multiple of ", KPack, ".");
  }
  const int total = (output_rows / MNPack) * (KScale / KPack);
  constexpr int block_size = 256;
  const int grid_size      = (total + block_size - 1) / block_size;
  hipLaunchKernelGGL((preshuffle_scale_gfx950_kernel<ScaleType, MNPack, KPack,
                                                     XdlMNThread, XdlKThread, KStride>),
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     stream,
                     src,
                     dst,
                     actual_rows,
                     output_rows,
                     KScale);
  NVTE_CHECK_CUDA(hipGetLastError());
}

template <GPUArch Arch> struct MxTileCfg;

template <> struct MxTileCfg<GPUArch::GFX1250> {
  static const bool kPadM = false;
  static const bool kPadN = false;
  static const bool kPadK = false;
  static const int kBlockPerCu         = 1;
  static const ck_tile::index_t M_Tile = 64;
  static const ck_tile::index_t N_Tile = 64;
  static const ck_tile::index_t K_Tile = 128;
  static const ck_tile::index_t M_Warp = 2;
  static const ck_tile::index_t N_Warp = 2;
  static const ck_tile::index_t K_Warp = 1;
  static const ck_tile::index_t M_Warp_Tile     = 32;
  static const ck_tile::index_t N_Warp_Tile     = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 128;
  static constexpr MxGemmPipelineType PipelineType = MxGemmPipelineType::CompTDMV1;
  // WMMA writes C transposed when the warp tile is square and C is RowMajor.
  static constexpr bool TransposeC = true;
  // A scales are padded to the warp tile; the WMMA swizzle blocks M by 32 rows.
  static constexpr ck_tile::index_t ScalePadM = M_Warp_Tile;
};

// gfx950 MFMA. Shape and pipeline follow KernelTypesMxGemmCompAsync in CK's
// test/ck_tile/grouped_gemm_mx/test_mx_grouped_gemm_pipeline_kernel_types.hpp;
// K_Warp_Tile is what get_k_warp_tile() there yields for a 16-wide MFMA warp tile.
template <> struct MxTileCfg<GPUArch::GFX950> {
  static const bool kPadM = false;
  static const bool kPadN = false;
  static const bool kPadK = false;
  static const int kBlockPerCu         = 1;
  static const ck_tile::index_t M_Tile = 64;
  static const ck_tile::index_t N_Tile = 64;
  static const ck_tile::index_t K_Tile = 256;
  static const ck_tile::index_t M_Warp = 2;
  static const ck_tile::index_t N_Warp = 2;
  static const ck_tile::index_t K_Warp = 1;
  static const ck_tile::index_t M_Warp_Tile     = 16;
  static const ck_tile::index_t N_Warp_Tile     = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 128;
  static constexpr MxGemmPipelineType PipelineType = MxGemmPipelineType::CompAsync;
  static constexpr bool TransposeC = false;
  // CK's test pads A scales to the block tile before the MFMA swizzle.
  static constexpr ck_tile::index_t ScalePadM = M_Tile;
};

// XdlPack factors for the gfx950 scale swizzle, derived exactly as CK's test does:
// pack two iterations together whenever the per-warp iteration count is even.
template <ck_tile::index_t IterPerWarp>
inline constexpr ck_tile::index_t MxXdlPackEff =
    (IterPerWarp >= 2 && IterPerWarp % 2 == 0) ? 2 : 1;

template <GPUArch Arch,
          typename AType,
          typename BType,
          typename CType,
          typename AScaleType,
          typename BScaleType,
          typename AccType = float>
bool invoke_mx_grouped_gemm(const std::vector<mx_grouped_gemm_kargs> &descs,
                            const GroupedGemmRunContext &ctx,
                            const ck_tile::stream_config &stream_cfg,
                            bool warn_fallback) {

  using Cfg = MxTileCfg<Arch>;

  // Check hardware WMMA support for the warp tile. gfx950 uses MFMA, not WMMA,
  // so the gfx125 traits do not apply there.
  if constexpr (Arch == GPUArch::GFX1250) {
    static constexpr bool has_wmma_support =
        ck_tile::has_wmma_traits_v<ck_tile::gfx125_t,
                                   AType,
                                   BType,
                                   AccType,
                                   Cfg::M_Warp_Tile,
                                   Cfg::N_Warp_Tile,
                                   Cfg::K_Warp_Tile>;

    NVTE_CHECK(has_wmma_support,
               "ck_tile_mx_grouped_gemm: unsupported gfx125 WMMA traits for "
               "AType/BType/AccType with warp tile shape ",
               Cfg::M_Warp_Tile, "x",
               Cfg::N_Warp_Tile, "x",
               Cfg::K_Warp_Tile);
  }

  using CLayout = RowMajor;
  constexpr bool preshuffle       = false;
  // Both the TDM and async pipelines static_assert on this being true.
  constexpr bool DoubleSmemBuffer = true;
  constexpr bool TransposeC = Cfg::TransposeC;
  static constexpr bool StructuredSparsity = false;
  static constexpr bool NumWaveGroup       = 1;
  constexpr ck_tile::index_t TileParitionerGroupNum = 8;
  constexpr ck_tile::index_t TileParitionerM01      = 4;
  using GemmShape =
      ck_tile::TileGemmShape<ck_tile::sequence<Cfg::M_Tile,
                                               Cfg::N_Tile,
                                               Cfg::K_Tile>,
                             ck_tile::sequence<Cfg::M_Warp,
                                               Cfg::N_Warp,
                                               Cfg::K_Warp>,
                             ck_tile::sequence<Cfg::M_Warp_Tile,
                                               Cfg::N_Warp_Tile,
                                               Cfg::K_Warp_Tile>>;
  using TilePartitioner = ck_tile::
      GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transA, kTransA, {
    using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;
    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transB, kTransB, {
      using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;
      using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<Cfg::kPadM,
                                                                   Cfg::kPadN,
                                                                   Cfg::kPadK,
                                                                   DoubleSmemBuffer,
                                                                   ALayout,
                                                                   BLayout,
                                                                   CLayout,
                                                                   TransposeC,
                                                                   StructuredSparsity,
                                                                   false,  // Persistent
                                                                   NumWaveGroup,
                                                                   preshuffle>;
      using UniversalGemmProblem =
        ck_tile::MxGemmPipelineProblem<AType,
                                       BType,
                                       float,
                                       GemmShape,
                                       GemmUniversalTraits,
                                       ck_tile::GemmPipelineScheduler::Intrawave,
                                       ck_tile::element_wise::PassThrough,
                                       ck_tile::element_wise::PassThrough,
                                       AType,
                                       BType,
                                       AScaleType,
                                       BScaleType>;
      /* Make pipeline selective. */
      using GemmPipeline =
        typename MxGemmPipelineTypeSelector<
          Cfg::PipelineType,
          UniversalGemmProblem>::pipeline;

      using GemmEpilogueProblem =
        ck_tile::CShuffleEpilogueProblem<AType,
                                         BType,
                                         ck_tile::tuple<>,  // DsDataType
                                         float,
                                         CType,
                                         ck_tile::tuple<>,  // DsLayout
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         Cfg::M_Warp,
                                         Cfg::N_Warp,
                                         Cfg::M_Warp_Tile,
                                         Cfg::N_Warp_Tile,
                                         Cfg::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC,
                                         1,                /* kNumWaveGroups_ */
                                         false,            /* FixedVectorSize_ */
                                         1,                /* VectorSizeC_ */
                                         1,                /* BlockedXDLN_PerWarp_ */
                                         DoubleSmemBuffer, /* DoubleSmemBuffer */
                                         AType,            /* AType_ */
                                         BType,            /* BType_ */
                                         // TilesPacked_: the block GEMM emits contiguous
                                         // MRepeat/NRepeat because the scales are packed.
                                         // Only CShuffleEpilogue reads this; TdmEpilogue
                                         // ignores it, so gfx1250 is unaffected.
                                         !preshuffle>;
      using GemmEpilogue =
        typename MxGemmEpilogueTypeSelector<Cfg::PipelineType, GemmEpilogueProblem>::epilogue;
      using Kernel = ck_tile::MxGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

      if (!has_sufficient_workspace<Kernel>(ctx)) {
        return false;
      }

      auto kargs = Kernel::MakeKargs(descs);
      if (!Kernel::IsSupportedArgument(kargs)) {
        if (warn_fallback) {
          NVTE_WARN("ck_tile_mx_grouped_gemm: CK_Tile kernel arguments not supported for this config. "
                    "Falling back.");
        }
        return false;
      }
      const dim3 blocks = Kernel::BlockSize();
      const dim3 grids  = Kernel::GridSize(kargs);
      NVTE_CHECK_CUDA(hipMemcpyAsync(ctx.workspace,
                                     kargs.data(),
                                     kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                     hipMemcpyHostToDevice,
                                     ctx.stream));
      ck_tile::ignore = ck_tile::launch_kernel(
        stream_cfg, ck_tile::make_kernel<Cfg::kBlockPerCu>(
                      Kernel{}, grids, blocks, 0,
                      ck_tile::cast_pointer_to_constant_address_space(ctx.workspace),
                      kargs.size()));
      return true;
    });
  });
  return false;
}

template <GPUArch Arch>
bool ck_tile_mx_grouped_gemm_impl(DType a_dtype, DType b_dtype, DType d_dtype,
                                  const GroupedGemmRunContext& ctx_in) {
  using Cfg = MxTileCfg<Arch>;

  using AScaleType = ck_tile::e8m0_t;
  using BScaleType = ck_tile::e8m0_t;

  // gfx950 scale swizzle parameters (unused on gfx1250, whose swizzle is fixed).
  constexpr ck_tile::index_t MIterPerWarp = Cfg::M_Tile / (Cfg::M_Warp * Cfg::M_Warp_Tile);
  constexpr ck_tile::index_t NIterPerWarp = Cfg::N_Tile / (Cfg::N_Warp * Cfg::N_Warp_Tile);
  constexpr ck_tile::index_t KIterPerWarp = Cfg::K_Tile / Cfg::K_Warp_Tile;
  constexpr ck_tile::index_t MXdlPack = MxXdlPackEff<MIterPerWarp>;
  constexpr ck_tile::index_t NXdlPack = MxXdlPackEff<NIterPerWarp>;
  constexpr ck_tile::index_t KXdlPack = MxXdlPackEff<KIterPerWarp>;
  constexpr ck_tile::index_t XdlMNThread = Cfg::M_Warp_Tile;
  constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

  // The scale workspace is carved out of the tail of the caller's workspace, so
  // this routine shrinks workspace_bytes for the kernel launch below.
  GroupedGemmRunContext ctx = ctx_in;

  const bool warn_fallback =
    getenv<bool>("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", false);

  const ck_tile::stream_config s{ctx.stream};

  std::vector<mx_grouped_gemm_kargs> descs;
  descs.reserve(ctx.group_num);

  NVTE_CHECK(ctx.workspace != nullptr,
             "ck_tile_mx_grouped_gemm: workspace is required for shuffled MXFP8 scales.");

  // Carve regions from the end of the workspace for mxfp8 scales.
  // Layout: [CK kargs workspace ... | a_scales (i) | b_scales (i) | ... | a_scales (group_num-1) | b_scales (group_num-1)]
  constexpr size_t kScaleWorkspaceAlign = 256;
  uint8_t *scale_workspace_base = reinterpret_cast<uint8_t *>(ctx.workspace);
  size_t scale_workspace_end =
    (ctx.workspace_bytes / kScaleWorkspaceAlign) * kScaleWorkspaceAlign;

  for (int i = 0; i < ctx.group_num; i++) {
    const transformer_engine::Tensor *const A_te =
      transformer_engine::convertNVTETensorCheck(ctx.A[i]);
    const transformer_engine::Tensor *const B_te =
      transformer_engine::convertNVTETensorCheck(ctx.B[i]);
    transformer_engine::Tensor *D_te =
      transformer_engine::convertNVTETensorCheck(ctx.D[i]);

    const auto &a = ctx.use_a_columnwise_data ? A_te->columnwise_data : A_te->data;
    const auto &b = ctx.use_b_columnwise_data ? B_te->columnwise_data : B_te->data;
    const auto &d = D_te->data;
    const auto &a_scales =
        ctx.use_a_columnwise_data ? A_te->columnwise_scale_inv : A_te->scale_inv;
    const auto &b_scales =
        ctx.use_b_columnwise_data ? B_te->columnwise_scale_inv : B_te->scale_inv;

    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;

    if (!get_flat_2d_dims(*A_te, Ad0, Ad1)) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for normalized A in group ", i);
    }

    if (!get_flat_2d_dims(*B_te, Bd0, Bd1)) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for normalized B in group ", i);
    }

    if (!get_flat_2d_dims(*D_te, Dd0, Dd1)) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: expected rank>=2 for normalized D in group ", i);
    }
    if (a.dptr == nullptr || b.dptr == nullptr || a_scales.dptr == nullptr ||
        b_scales.dptr == nullptr) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: effective A/B data or scale_inv is missing.");
    }
    if (a_scales.shape.size() != 2 || b_scales.shape.size() != 2) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: expected effective A/B scale_inv tensors to be rank-2.");
    }

    const size_t M = ctx.transA ? Ad1 : Ad0;
    const size_t K = ctx.transA ? Ad0 : Ad1;
    const size_t N = ctx.transB ? Bd0 : Bd1;
    const size_t Kb = ctx.transB ? Bd1 : Bd0;
    if (K % ScaleBlockSize != 0) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: K must be a multiple of ScaleBlockSize for MX GEMM", i);
    }
    const int KScale = static_cast<int>(K / ScaleBlockSize);
    if (Kb != K) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: K mismatch between A and B in group ", i,
                  ". op(A)=", M, "x", K, ", op(B)=", Kb, "x", N);
    }
    if (Dd0 != M || Dd1 != N) {
      NVTE_ERROR("ck_tile_mx_grouped_gemm: D shape mismatch in group ", i,
                  ". D=", Dd0, "x", Dd1, ", expected=", M, "x", N);
    }

    const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
    const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
    const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);

    // Pre-shuffle scale buffers for the hardware.
    const int a_scale_actual_rows = static_cast<int>(M);
    const int a_scale_output_rows =
      ck_tile::integer_least_multiple(
        static_cast<ck_tile::index_t>(M),
        static_cast<ck_tile::index_t>(Cfg::ScalePadM));
    const int b_scale_actual_rows = static_cast<int>(N);
    const int b_scale_output_rows = static_cast<int>(N);
    // gfx1250 writes one scale per input element; the gfx950 swizzle packs
    // MNPack x KPack scales per dword group, so its buffer is sized in packs.
    const auto shuffled_scale_elems = [&](int rows, ck_tile::index_t mn_pack) -> size_t {
      if constexpr (Arch == GPUArch::GFX1250) {
        return static_cast<size_t>(rows) * static_cast<size_t>(KScale);
      } else {
        return static_cast<size_t>(rows / mn_pack * 2) *
               static_cast<size_t>(KScale / KXdlPack * 2);
      }
    };
    const size_t a_scale_shuffled_bytes =
      shuffled_scale_elems(a_scale_output_rows, MXdlPack) * sizeof(AScaleType);
    const size_t b_scale_shuffled_bytes =
      shuffled_scale_elems(b_scale_output_rows, NXdlPack) * sizeof(BScaleType);
    const size_t scale_pair_bytes =
      a_scale_shuffled_bytes + b_scale_shuffled_bytes;
    scale_workspace_end =
      (scale_workspace_end / kScaleWorkspaceAlign) * kScaleWorkspaceAlign;

    NVTE_CHECK(scale_workspace_end >= scale_pair_bytes,
               "ck_tile_mx_grouped_gemm: insufficient workspace for shuffled MXFP8 scales. "
               "Need current group scale bytes=", scale_pair_bytes,
               ", available workspace bytes=", scale_workspace_end,
               ". Increase the grouped GEMM workspace size.");

    scale_workspace_end -= scale_pair_bytes;
    uint8_t *scale_pair_ptr = scale_workspace_base + scale_workspace_end;

    void *a_scale_shuffled_ptr = scale_pair_ptr;
    void *b_scale_shuffled_ptr = scale_pair_ptr + a_scale_shuffled_bytes;

    // CK expects canonical pre-shuffled scale buffers laid out as
    // A: [M, KScale] and B: [N, KScale], independent of A/B data layouts.
    // TE rowwise MXFP8 scale_inv is [rows, KScale] and can be read with
    // KStride=true. TE columnwise_scale_inv is [KScale, rows] and must be
    // read with KStride=false before writing CK's canonical shuffled layout.
    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.use_a_columnwise_data, kAColwise, {
      if constexpr (Arch == GPUArch::GFX1250) {
        preShuffleScaleBuffer_gfx1250<AScaleType, ScaleBlockSize, !kAColwise>(
          reinterpret_cast<const AScaleType *>(a_scales.dptr),
          reinterpret_cast<AScaleType *>(a_scale_shuffled_ptr),
          a_scale_actual_rows, a_scale_output_rows, KScale, ctx.stream);
      } else {
        preShuffleScaleBuffer_gfx950<AScaleType, MXdlPack, KXdlPack,
                                     XdlMNThread, XdlKThread, !kAColwise>(
          reinterpret_cast<const AScaleType *>(a_scales.dptr),
          reinterpret_cast<AScaleType *>(a_scale_shuffled_ptr),
          a_scale_actual_rows, a_scale_output_rows, KScale, ctx.stream);
      }
    });

    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.use_b_columnwise_data, kBColwise, {
      if constexpr (Arch == GPUArch::GFX1250) {
        preShuffleScaleBuffer_gfx1250<BScaleType, ScaleBlockSize, !kBColwise>(
          reinterpret_cast<const BScaleType *>(b_scales.dptr),
          reinterpret_cast<BScaleType *>(b_scale_shuffled_ptr),
          b_scale_actual_rows, b_scale_output_rows, KScale, ctx.stream);
      } else {
        preShuffleScaleBuffer_gfx950<BScaleType, NXdlPack, KXdlPack,
                                     XdlMNThread, XdlKThread, !kBColwise>(
          reinterpret_cast<const BScaleType *>(b_scales.dptr),
          reinterpret_cast<BScaleType *>(b_scale_shuffled_ptr),
          b_scale_actual_rows, b_scale_output_rows, KScale, ctx.stream);
      }
    });
    descs.emplace_back(mx_grouped_gemm_kargs(
      a.dptr, a_scale_shuffled_ptr, b.dptr, b_scale_shuffled_ptr,
      {/*ds_ptr*/}, d.dptr, 1,  // kbatch
      M, N, K, stride_A, stride_B, {/*stride_Ds*/}, stride_E));
  }
  ctx.workspace_bytes = scale_workspace_end;

  // Invoke the GEMM.
  bool ok = false;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_te_type, {
    using AType = typename TETypeToCKType<a_te_type>::type;
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_te_type, {
      using BType = typename TETypeToCKType<b_te_type>::type;
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;
        ok = invoke_mx_grouped_gemm<Arch,
                                    AType, BType, CType,
                                    AScaleType, BScaleType>(descs, ctx, s, warn_fallback);
      });  // NOLINT(*)
    });  // NOLINT(*)
  });  // NOLINT(*)
  return ok;
}

// Per-architecture dispatch function signature.
// Each architecture file implements one of these. The TDM pipeline is gfx1250
// only, so each file is built for its own architecture and the shared template
// above is never instantiated for an architecture it does not support.
bool ck_tile_mx_grouped_gemm_dispatch_gfx1250(DType a_dtype, DType b_dtype, DType d_dtype,
                                              const GroupedGemmRunContext& ctx);
bool ck_tile_mx_grouped_gemm_dispatch_gfx950(DType a_dtype, DType b_dtype, DType d_dtype,
                                             const GroupedGemmRunContext& ctx);

}  // namespace grouped_gemm
}  // namespace transformer_engine