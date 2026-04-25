/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include "../../common.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_grouped_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace transformer_engine {
namespace mx_grouped_gemm {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using mx_grouped_gemm_kargs = ck_tile::MxGroupedGemmHostArgs<>;

template <typename TEScalar> struct TETypeToCKType;
template <> struct TETypeToCKType<transformer_engine::fp8e4m3> { using type = ck_tile::fp8_t; };
template <> struct TETypeToCKType<transformer_engine::fp8e5m2> { using type = ck_tile::bf8_t; };
template <> struct TETypeToCKType<transformer_engine::fp16>    { using type = ck_tile::half_t; };
template <> struct TETypeToCKType<transformer_engine::bf16>    { using type = ck_tile::bfloat16_t; };
template <> struct TETypeToCKType<transformer_engine::fp32>    { using type = float; };

struct GroupedGemmRunContext {
    const NVTETensor* A = nullptr;
    const NVTETensor* B = nullptr;
    NVTETensor* D = nullptr;

    int group_num = 0;
    bool transA = false;
    bool transB = false;

    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    hipStream_t stream = nullptr;

};

// Treat TE tensors as generalized 2D matrices by flattening:
// (D1, D2, ..., Dn) -> (D1*...*D(n-1), Dn), consistent with TE Tensor::flat_*_dim.
static inline bool get_flat_2d_dims(const transformer_engine::Tensor& t,
                                    int64_t& d0, int64_t& d1) {
  if (t.shape().size() < 2) {
    return false;
  }
  d0 = static_cast<int64_t>(t.flat_first_dim());
  d1 = static_cast<int64_t>(t.flat_last_dim());
  return true;
}

static constexpr ck_tile::index_t ScaleBlockSize = 32;

enum struct MxGemmPipelineType
{
    CompTDMV1,
    CompTDMV2
};

template <MxGemmPipelineType PT, typename Problem>
struct MxGemmPipelineTypeSelector;
template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV1, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;
    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV1"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV2, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;
    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV2"; }
};

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data;
}

static inline const transformer_engine::SimpleTensor& scale_inv_view(const transformer_engine::Tensor& t) {
  return t.scale_inv;
}

template <typename Kernel>
static inline bool has_sufficient_workspace(const GroupedGemmRunContext& ctx) {
  const size_t needed = Kernel::GetWorkSpaceSize(ctx.group_num);
  if (!ctx.workspace || ctx.workspace_bytes < needed) {
    NVTE_WARN("ck_tile_grouped_gemm: insufficient workspace for CK path. Needed bytes=", needed,
              ", available bytes=", ctx.workspace_bytes, ". Falling back.");
    return false;
  }
  return true;
}

struct GroupedGemKernelParam_Wmma
{
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
};

template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
__global__ void preshuffle_scale_gfx1250_kernel(const ScaleType* __restrict__ src,
                                                ScaleType* __restrict__ dst,
                                                int actual_rows,
                                                int output_rows,
                                                int KScale)
{
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
    if(linear >= total)
        return;
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
    if(mn < actual_rows)
    {
        if constexpr(KStride)
            value = src[mn * KScale + k];
        else
            value = src[k * actual_rows + mn];
    }
    dst[outputIndex] = value;
}

template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                                 ScaleType* dst,
                                                 int actual_rows,
                                                 int output_rows,
                                                 int KScale,
                                                 hipStream_t stream)
{
    constexpr int KPerXdlops = 128;
    constexpr int KStep      = KPerXdlops / ScaleBlockSize; // 4
    if(KScale % KStep != 0)
    {
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

template <typename MXFP8GemmConfig, typename AType, typename BType, typename CType, typename AScaleType, typename BScaleType, typename AccType = float>
bool invoke_mx_grouped_gemm(const std::vector<mx_grouped_gemm_kargs>& descs, const GroupedGemmRunContext& ctx, const ck_tile::stream_config& stream_cfg)
{
  // check hardware WMMA support for the warp tile
  static constexpr bool has_wmma_support =
      ck_tile::has_wmma_traits_v<ck_tile::gfx125_t,
                               AType,
                               BType,
                               AccType,
                               MXFP8GemmConfig::M_Warp_Tile,
                               MXFP8GemmConfig::N_Warp_Tile,
                               MXFP8GemmConfig::K_Warp_Tile>;

  NVTE_CHECK(has_wmma_support,
           "ck_tile_mx_grouped_gemm: unsupported gfx125 WMMA traits for "
           "AType/BType/AccType with warp tile shape ",
           MXFP8GemmConfig::M_Warp_Tile, "x",
           MXFP8GemmConfig::N_Warp_Tile, "x",
           MXFP8GemmConfig::K_Warp_Tile);

  using CLayout = RowMajor;
  constexpr bool preshuffle       = false;
  constexpr bool DoubleSmemBuffer = true; // TDM pipeline requires double smem buffer
  constexpr bool TransposeC =
      std::is_same_v<CLayout, RowMajor> &&
      MXFP8GemmConfig::M_Warp_Tile == MXFP8GemmConfig::N_Warp_Tile;
  static constexpr bool StructuredSparsity = false;
  static constexpr bool NumWaveGroup       = 1;
  constexpr ck_tile::index_t TileParitionerGroupNum = 8;
  constexpr ck_tile::index_t TileParitionerM01      = 4;
  using GemmShape =
      ck_tile::TileGemmShape<ck_tile::sequence<MXFP8GemmConfig::M_Tile,
                                               MXFP8GemmConfig::N_Tile,
                                               MXFP8GemmConfig::K_Tile>,
                             ck_tile::sequence<MXFP8GemmConfig::M_Warp,
                                               MXFP8GemmConfig::N_Warp,
                                               MXFP8GemmConfig::K_Warp>,
                             ck_tile::sequence<MXFP8GemmConfig::M_Warp_Tile,
                                               MXFP8GemmConfig::N_Warp_Tile,
                                               MXFP8GemmConfig::K_Warp_Tile>>;
  using TilePartitioner = ck_tile::
      GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transA, kTransA, {
    using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;
    TRANSFORMER_ENGINE_SWITCH_CONDITION(ctx.transB, kTransB, {
      using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;
      using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<MXFP8GemmConfig::kPadM,
                                                                   MXFP8GemmConfig::kPadN,
                                                                   MXFP8GemmConfig::kPadK,
                                                                   DoubleSmemBuffer,
                                                                   ALayout,
                                                                   BLayout,
                                                                   CLayout,
                                                                   TransposeC,
                                                                   StructuredSparsity,
                                                                   false,//Persistent
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
        /* make pipeline selective */
      using GemmPipeline =
          typename MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV1,
                                                UniversalGemmProblem>::pipeline;
      using GemmEpilogue = ck_tile::TdmEpilogue<
          ck_tile::CShuffleEpilogueProblem<AType,
                                           BType,
                                           ck_tile::tuple<>,//DsDataType
                                           float,
                                           CType,
                                           ck_tile::tuple<>,//DsLayout
                                           CLayout,
                                           ck_tile::element_wise::PassThrough,
                                           TilePartitioner::MPerBlock,
                                           TilePartitioner::NPerBlock,
                                           MXFP8GemmConfig::M_Warp,
                                           MXFP8GemmConfig::N_Warp,
                                           MXFP8GemmConfig::M_Warp_Tile,
                                           MXFP8GemmConfig::N_Warp_Tile,
                                           MXFP8GemmConfig::K_Warp_Tile,
                                           UniversalGemmProblem::TransposeC,
                                           1,                /*kNumWaveGroups_*/
                                           false,            /*FixedVectorSize_*/
                                           1,                /*VectorSizeC_*/
                                           false,            /*TiledMMAPermuteN_*/
                                           1,                /*BlockedXDLN_PerWarp_*/
                                           DoubleSmemBuffer, /*DoubleSmemBuffer*/
                                           AType, /*AType_*/
                                           BType /*BType_*/>>;
      using Kernel = ck_tile::MxGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

      if (!has_sufficient_workspace<Kernel>(ctx)) {
        return false;
      }

      auto kargs = Kernel::MakeKargs(descs);
      if(!Kernel::IsSupportedArgument(kargs))
      {
        NVTE_WARN("ck_tile_mx_grouped_gemm: CK_Tile kernel arguments not supported for this config. "
                "Falling back.");
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
          stream_cfg, ck_tile::make_kernel<MXFP8GemmConfig::kBlockPerCu>(
                         Kernel{}, grids, blocks, 0,
                         ck_tile::cast_pointer_to_constant_address_space(ctx.workspace),
                         kargs.size()));
      return true;
    });
  });
  return false;
}

bool ck_tile_mx_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,//ignored for now
                          hipStream_t stream) {
  if (group_num <= 0) {
    return true;
  }

  // Normalize input mats
  // I.e., swap A and B, as well as transa and transb.
  const NVTETensor* A_use = B;
  const NVTETensor* B_use = A;
  bool transA_use = transB;
  bool transB_use = transA;

  // Validate scale type / data type combination
  // Expected input data format: fp8/bf8 (e4m3/e5m2)
  // Expected scale data format: e8m0
  const auto* A0 = convertNVTETensorCheck(A_use[0]);
  const auto* B0 = convertNVTETensorCheck(B_use[0]);
  const auto* D0 = convertNVTETensorCheck(D[0]);
  NVTE_CHECK(A0->scale_inv.dptr != nullptr,
            "ck_tile_mx_grouped_gemm: A[0] scale_inv is not initialized");
  NVTE_CHECK(B0->scale_inv.dptr != nullptr,
            "ck_tile_mx_grouped_gemm: B[0] scale_inv is not initialized");

  const auto a_scale_dtype = A0->scale_inv.dtype;
  const auto b_scale_dtype = B0->scale_inv.dtype;
  NVTE_CHECK(a_scale_dtype == DType::kFloat8E8M0,
        "ck_tile_mx_grouped_gemm: A scale_inv dtype must be Float8E8M0, got ",
        static_cast<int>(a_scale_dtype));
  
  NVTE_CHECK(b_scale_dtype == DType::kFloat8E8M0,
        "ck_tile_mx_grouped_gemm: B scale_inv dtype must be Float8E8M0, got ",
        static_cast<int>(b_scale_dtype));
  
  const auto a_dtype = A0->dtype();
  const auto b_dtype = B0->dtype();
  const auto d_dtype = D0->dtype();
  NVTE_CHECK(is_fp8_dtype(a_dtype), "ck_tile_mx_grouped_gemm: A dtype must be FP8");
  NVTE_CHECK(is_fp8_dtype(b_dtype), "ck_tile_mx_grouped_gemm: B dtype must be FP8");

  using AScaleType = ck_tile::e8m0_t;
  using BScaleType = ck_tile::e8m0_t;

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }
  
  GroupedGemmRunContext ctx = {
      A_use,
      B_use,
      D,
      group_num,
      transA_use,
      transB_use,
      ws_ptr,
      ws_bytes,
      stream};

  const ck_tile::stream_config s{ctx.stream};

  std::vector<mx_grouped_gemm_kargs> descs;
  descs.reserve(group_num);

  std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_scale_shuffled_bufs;
  std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_scale_shuffled_bufs;
  a_scale_shuffled_bufs.reserve(group_num);
  b_scale_shuffled_bufs.reserve(group_num);

  for (int i = 0; i < group_num; i++) {
      const transformer_engine::Tensor* const A_te =
          transformer_engine::convertNVTETensorCheck(ctx.A[i]);
      const transformer_engine::Tensor* const B_te =
          transformer_engine::convertNVTETensorCheck(ctx.B[i]);
      transformer_engine::Tensor* D_te =
          transformer_engine::convertNVTETensorCheck(ctx.D[i]);
      const auto& a = data_view(*A_te);
      const auto& b = data_view(*B_te);
      const auto& d = data_view(*D_te);
      int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;
      if (!get_flat_2d_dims(*A_te, Ad0, Ad1) ||
          !get_flat_2d_dims(*B_te, Bd0, Bd1) ||
          !get_flat_2d_dims(*D_te, Dd0, Dd1)) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected all groups to be rank>=2.");
      }
      const auto& a_scales = scale_inv_view(*A_te);
      const auto& b_scales = scale_inv_view(*B_te);
      if (a_scales.shape.size() != 2 || b_scales.shape.size() != 2) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: expected A/B scale_inv tensors to be rank-2.");
      }
      const int64_t M = ctx.transA ? Ad1 : Ad0;
      const int64_t K = ctx.transA ? Ad0 : Ad1;
      const int64_t N = ctx.transB ? Bd0 : Bd1;
      const int64_t Kb = ctx.transB ? Bd1 : Bd0;
      if (K % ScaleBlockSize != 0) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: K must be a multiple of ScaleBlockSize for MX GEMM", i);
      }
      const int KScale = static_cast<int>(K / ScaleBlockSize);
      if (Kb != K) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: K mismatch between A and B in group ", i);
      }
      if (Dd0 != M || Dd1 != N) {
        NVTE_ERROR("ck_tile_mx_grouped_gemm: D shape mismatch in group ", i);
      }
      const ck_tile::index_t stride_A = static_cast<ck_tile::index_t>(Ad1);
      const ck_tile::index_t stride_B = static_cast<ck_tile::index_t>(Bd1);
      const ck_tile::index_t stride_E = static_cast<ck_tile::index_t>(Dd1);
      // Pre-shuffle scale buffers for the hardware
      const int a_scale_actual_rows = static_cast<int>(M);
      const int a_scale_output_rows =
      ck_tile::integer_least_multiple(
          static_cast<ck_tile::index_t>(M),
          static_cast<ck_tile::index_t>(GroupedGemKernelParam_Wmma::M_Warp_Tile));
      const int b_scale_actual_rows = static_cast<int>(N);
      const int b_scale_output_rows = static_cast<int>(N);
      const size_t a_scale_shuffled_bytes =
          static_cast<size_t>(a_scale_output_rows) *
          static_cast<size_t>(KScale) *
          sizeof(AScaleType);
      const size_t b_scale_shuffled_bytes =
          static_cast<size_t>(b_scale_output_rows) *
          static_cast<size_t>(KScale) *
          sizeof(BScaleType);
      a_scale_shuffled_bufs.push_back(
          std::make_unique<ck_tile::DeviceMem>(a_scale_shuffled_bytes));
      b_scale_shuffled_bufs.push_back(
          std::make_unique<ck_tile::DeviceMem>(b_scale_shuffled_bytes));
      void* a_scale_shuffled_ptr = a_scale_shuffled_bufs.back()->GetDeviceBuffer();
      void* b_scale_shuffled_ptr = b_scale_shuffled_bufs.back()->GetDeviceBuffer();
      preShuffleScaleBuffer_gfx1250<AScaleType, ScaleBlockSize, true>(
          reinterpret_cast<const AScaleType*>(a_scales.dptr),
          reinterpret_cast<AScaleType*>(a_scale_shuffled_ptr),
          a_scale_actual_rows,
          a_scale_output_rows,
          KScale,
          stream);
      preShuffleScaleBuffer_gfx1250<BScaleType, ScaleBlockSize, true>(
          reinterpret_cast<const BScaleType*>(b_scales.dptr),
          reinterpret_cast<BScaleType*>(b_scale_shuffled_ptr),
          b_scale_actual_rows,
          b_scale_output_rows,
          KScale,
          stream);
      descs.emplace_back(mx_grouped_gemm_kargs(
                         a.dptr,
                         a_scale_shuffled_ptr,
                         b.dptr,
                         b_scale_shuffled_ptr,
                         {/*ds_ptr*/},
                         d.dptr,
                         1,//kbatch
                         M,
                         N,
                         K,
                         stride_A,
                         stride_B,
                         {/*stride_Ds*/},
                         stride_E));
  }
  // invoke gemm
  bool ok = false;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(a_dtype, a_te_type, {
    using AType = typename TETypeToCKType<a_te_type>::type;
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(b_dtype, b_te_type, {
      using BType = typename TETypeToCKType<b_te_type>::type;
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(d_dtype, d_te_type, {
        using CType = typename TETypeToCKType<d_te_type>::type;
        ok = invoke_mx_grouped_gemm<GroupedGemKernelParam_Wmma,
                                    AType, BType, CType,
                                    AScaleType, BScaleType>(descs,ctx,s);
      });
    });
  });
  return ok;
}

}  // namespace mx_grouped_gemm
}  // namespace transformer_engine