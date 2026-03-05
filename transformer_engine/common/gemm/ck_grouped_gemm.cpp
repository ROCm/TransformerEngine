/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <hip/hip_runtime.h>

#include <transformer_engine/transformer_engine.h>
#include "../common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace transformer_engine {
namespace grouped_gemm {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TEScalar> struct TETypeToCKType;
template <> struct TETypeToCKType<transformer_engine::fp16> { using type = ck_tile::half_t; };
template <> struct TETypeToCKType<transformer_engine::bf16> { using type = ck_tile::bfloat16_t; };

// Treat TE tensors as generalized 2D matrices by flattening:
// (D1, D2, ..., Dn) -> (D1*...*D(n-1), Dn), consistent with TE Tensor::flat_*_dim.
static inline bool get_flat_2d_dims(const transformer_engine::Tensor& t,
                                   int64_t& d0, int64_t& d1) {
  // Require at least a matrix (rank >= 2). Higher ranks are flattened.
  if (t.shape().size() < 2)
    return false;
  d0 = static_cast<int64_t>(t.flat_first_dim());
  d1 = static_cast<int64_t>(t.flat_last_dim());
  return true;
}

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data; // rowwise data view
}

// Primus-Turbo-like FP16/BF16 tile configs
// Selection rule:
//   if (N % 256 == 0) use 256x256x64
//   else if (N % 128 == 0) use 256x128x64
//   else use 256x128x64 with N padding enabled
struct TileCfg_256x256x64 {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 256;
  static constexpr ck_tile::index_t K_Tile = 64;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = false;
  static constexpr bool kPadN = false;
  static constexpr bool kPadK = false;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01      = 4;
};

struct TileCfg_256x128x64 : TileCfg_256x256x64 {
  static constexpr ck_tile::index_t N_Tile = 128;
};

struct TileCfg_256x128x64_padding : TileCfg_256x128x64 {
  static constexpr bool kPadN = true;
};

// This class instantiates CK_Tile's grouped GEMM pipeline.
// See e.g. https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/03_gemm/universal_gemm_invoker.hpp for reference.
template <typename AType, typename BType, typename CType,
          typename ALayout, typename BLayout, typename CLayout,
          typename TileCfg, ck_tile::memory_operation_enum MemOp,
          typename AccType = float>
struct Runner{
  using GemmShape = ck_tile::TileGemmShape<
      ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
      ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
      ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

  using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
      GemmShape, TileCfg::TilePartitionerGroupNum, TileCfg::TilePartitionerM01>;

  using UniversalTraits = ck_tile::PersistentTileGemmUniversalTraits<
      TileCfg::kPadM, TileCfg::kPadN, TileCfg::kPadK,
      TileCfg::DoubleSmemBuffer, ALayout, BLayout, CLayout>;

  static constexpr ck_tile::GemmPipelineScheduler Scheduler =
      ck_tile::GemmPipelineScheduler::Intrawave;

  using Problem = ck_tile::UniversalGemmPipelineProblem<
      AType, BType, AccType, GemmShape, UniversalTraits, Scheduler>;

  using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

  using Epilogue = ck_tile::CShuffleEpilogue<
      ck_tile::CShuffleEpilogueProblem<
          AType, BType, ck_tile::tuple<>, AccType,
          CType, ck_tile::tuple<>, CLayout,
          ck_tile::element_wise::PassThrough,
          Partitioner::MPerBlock, Partitioner::NPerBlock,
          TileCfg::M_Warp, TileCfg::N_Warp,
          TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile,
        Problem::TransposeC>>;

  using Kernel = ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>;
};

template <typename T, typename ALayout, typename BLayout, typename CLayout,
          ck_tile::memory_operation_enum MemOp, typename TileCfg>
static bool run_grouped_impl(const NVTETensor* A_use,
                             const NVTETensor* B_use,
                             NVTETensor* D,
                             int group_num,
                             bool transA_use,
                             bool transB_use,
                             void* workspace,
                             size_t workspace_bytes,
                             hipStream_t stream)
{
  using Kernel = typename Runner<T, T, T, ALayout, BLayout, CLayout, TileCfg, MemOp>::Kernel;

  const size_t needed = Kernel::GetWorkSpaceSize(group_num);
  if (!workspace || workspace_bytes < needed) {
    NVTE_ERROR("ck_tile_grouped_gemm: insufficient workspace. Needed bytes=", needed);
    return false;
  }

  thread_local std::vector<ck_tile::GroupedGemmHostArgs<0>> descs;
  descs.clear();
  descs.reserve(group_num);

  for (int i = 0; i < group_num; ++i) {
    const transformer_engine::Tensor* const A_te =
        transformer_engine::convertNVTETensorCheck(A_use[i]);
    const transformer_engine::Tensor* const B_te =
        transformer_engine::convertNVTETensorCheck(B_use[i]);
    transformer_engine::Tensor* D_te =
        transformer_engine::convertNVTETensorCheck(D[i]);

    const auto& a = data_view(*A_te);
    const auto& b = data_view(*B_te);
    const auto& d = data_view(*D_te);

    int64_t Ad0 = 0, Ad1 = 0, Bd0 = 0, Bd1 = 0, Dd0 = 0, Dd1 = 0;
    if (!get_flat_2d_dims(*A_te, Ad0, Ad1) ||
        !get_flat_2d_dims(*B_te, Bd0, Bd1) ||
        !get_flat_2d_dims(*D_te, Dd0, Dd1)) {
      NVTE_ERROR("ck_tile_grouped_gemm: expected all groups to be rank>=2 (2D or higher).");
      return false;
    }

    const int64_t M  = transA_use ? Ad1 : Ad0;
    const int64_t K  = transA_use ? Ad0 : Ad1;
    const int64_t N  = transB_use ? Bd0 : Bd1;
    const int64_t Kb = transB_use ? Bd1 : Bd0;

    if (Kb != K) {
      NVTE_ERROR("ck_tile_grouped_gemm: K mismatch between A and B in group ", i);
      return false;
    }

    if (Dd0 != M || Dd1 != N) {
      NVTE_ERROR("ck_tile_grouped_gemm: D shape mismatch in group ", i);
      return false;
    }

    // Leading dimensions under the flattened-contiguous interpretation
    const ck_tile::index_t stride_A = Ad1;
    const ck_tile::index_t stride_B = Bd1;
    const ck_tile::index_t stride_E = Dd1;

    descs.emplace_back(
        a.dptr,
        b.dptr,
        std::array<const void*, 0>{},
        d.dptr,
        1,
        M,
        N,
        K,
        stride_A,
        stride_B,
        std::array<ck_tile::index_t, 0>{},
        stride_E);
  }

  const dim3 grids = Kernel::GridSize(descs);
  auto kargs = Kernel::MakeKargs(descs);
  if (!Kernel::IsSupportedArgument(kargs)) {
    NVTE_ERROR("ck_tile_grouped_gemm: CK_Tile kernel arguments not supported for this config.");
    return false;
  }

  HIP_CHECK_ERROR(hipMemcpyAsync(workspace,
                                kargs.data(),
                                kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                hipMemcpyHostToDevice,
                                stream));

  const ck_tile::stream_config s{stream};
  const dim3 blocks = Kernel::BlockSize();

  ck_tile::launch_kernel(
      s,
      ck_tile::make_kernel<1>(
          Kernel{}, grids, blocks, 0,
          ck_tile::cast_pointer_to_constant_address_space(workspace),
          group_num));
  return true;
}

}  // namespace grouped_gemm
}  // namespace transformer_engine

bool ck_tile_grouped_gemm(const NVTETensor* A,
                          const NVTETensor* B,
                          NVTETensor* D,
                          int group_num,
                          bool transA,
                          bool transB,
                          NVTETensor* workspace,
                          bool accumulate,
                          hipStream_t stream)
{
  if (group_num <= 0)
    return true;

  using namespace transformer_engine;
  using namespace transformer_engine::grouped_gemm;

  // Workspace pointer + bytes
  void*  ws_ptr   = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = convertNVTETensorCheck(*workspace);
    ws_ptr   = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() * typeToSize(ws_te->data.dtype);
  }

  // Normalize similar to upstream
  // See https://github.com/NVIDIA/TransformerEngine/blob/59f6f3876767d07045152bfae07b5dd4c54e1725/transformer_engine/common/gemm/cutlass_grouped_gemm.cu#L54-L68
  // I.e., swap A and B, as well as transa and transb.
  const NVTETensor* A_use = B;
  const NVTETensor* B_use = A;
  const bool transA_use = transB;
  const bool transB_use = transA;

  const auto a_dtype = convertNVTETensorCheck(A_use[0])->dtype();

  // Get N from D[0] (assume uniform N across groups)
  int64_t ref_d0 = 0, ref_d1 = 0;
  Tensor* D0_te = convertNVTETensorCheck(D[0]);
  if (!get_flat_2d_dims(*D0_te, ref_d0, ref_d1)) {
    NVTE_ERROR("ck_tile_grouped_gemm: expected rank>=2 for D[0]");
    return false;
  }
  const ck_tile::index_t N = static_cast<ck_tile::index_t>(ref_d1);

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(a_dtype, te_type, {
    using T = typename TETypeToCKType<te_type>::type;

    auto run_with_tilecfg = [&](auto tile_tag) -> bool {
      using TileCfgSel = decltype(tile_tag);

      TRANSFORMER_ENGINE_SWITCH_CONDITION(transA_use, kTransA, {
        using ALayout = std::conditional_t<kTransA, ColMajor, RowMajor>;

        TRANSFORMER_ENGINE_SWITCH_CONDITION(transB_use, kTransB, {
          using BLayout = std::conditional_t<kTransB, ColMajor, RowMajor>;

          if (accumulate) {
            return run_grouped_impl<T, ALayout, BLayout, RowMajor,
                                  ck_tile::memory_operation_enum::atomic_add, TileCfgSel>(
                A_use, B_use, D, group_num, kTransA, kTransB, ws_ptr, ws_bytes, stream);
          } else {
            return run_grouped_impl<T, ALayout, BLayout, RowMajor,
                                  ck_tile::memory_operation_enum::set, TileCfgSel>(
                A_use, B_use, D, group_num, kTransA, kTransB, ws_ptr, ws_bytes, stream);
          }
        });
      });
    };

    // Select tile config like Primus-Turbo for FP16/BF16:
    //   N%256 -> 256x256x64
    //   N%128 -> 256x128x64
    //   else  -> 256x128x64 padding
    // NOTE: We assume N is uniform across groups.
    if ((N % 256) == 0) {
      return run_with_tilecfg(TileCfg_256x256x64{});
    } else if ((N % 128) == 0) {
      return run_with_tilecfg(TileCfg_256x128x64{});
    } else {
      return run_with_tilecfg(TileCfg_256x128x64_padding{});
    }
  });
}
