/* Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved. */

#include <hip/hip_runtime.h>

#include <transformer_engine/transformer_engine.h>
#include "../common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TeScalar> struct TeTypeToCkType;
template <> struct TeTypeToCkType<transformer_engine::fp16> { using type = ck_tile::half_t; };
template <> struct TeTypeToCkType<transformer_engine::bf16> { using type = ck_tile::bfloat16_t; };


static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data; // rowwise data view
}

struct TileCfg_basic {
  static constexpr ck_tile::index_t M_Tile = 256;
  static constexpr ck_tile::index_t N_Tile = 128;
  static constexpr ck_tile::index_t K_Tile = 64;

  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;

  static constexpr ck_tile::index_t M_Warp_Tile = 32;
  static constexpr ck_tile::index_t N_Warp_Tile = 32;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;

  static constexpr bool kPadM = true;
  static constexpr bool kPadN = true;
  static constexpr bool kPadK = true;

  static constexpr bool DoubleSmemBuffer = false;

  static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
  static constexpr ck_tile::index_t TilePartitionerM01      = 1;
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
          Problem::TransposeC, MemOp>>;

  using Kernel = ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>;
};

template <typename T, typename ALayout, typename BLayout, typename CLayout,
          ck_tile::memory_operation_enum MemOp>
static bool run_grouped_impl(const transformer_engine::Tensor* const* A_use,
                             const transformer_engine::Tensor* const* B_use,
                             transformer_engine::Tensor* const* D,
                             int group_num,
                             bool transA_use,
                             bool transB_use,
                             void* workspace,
                             size_t workspace_bytes,
                             hipStream_t stream)
{
  using Kernel = typename Runner<T, T, T, ALayout, BLayout, CLayout, TileCfg_basic, MemOp>::Kernel;

  const size_t needed = Kernel::GetWorkSpaceSize(group_num);
  if (!workspace || workspace_bytes < needed) {
    NVTE_ERROR("grouped_gemm_ck_tile: insufficient workspace. Needed bytes=", needed);
    return false;
  }

  std::vector<ck_tile::GroupedGemmHostArgs<0>> descs;
  descs.reserve(group_num);

  for (int i = 0; i < group_num; ++i) {
    const auto& a = data_view(*A_use[i]);
    const auto& b = data_view(*B_use[i]);
    const auto& d = data_view(*D[i]);

    if (a.shape.size() != 2 || b.shape.size() != 2 || d.shape.size() != 2) {
      NVTE_ERROR("grouped_gemm_ck_tile: expected all groups to be 2D.");
      return false;
    }

    const int64_t Ad0 = a.shape[0];
    const int64_t Ad1 = a.shape[1];
    const int64_t Bd0 = b.shape[0];
    const int64_t Bd1 = b.shape[1];

    const int64_t M  = transA_use ? Ad1 : Ad0;
    const int64_t K  = transA_use ? Ad0 : Ad1;
    const int64_t N  = transB_use ? Bd0 : Bd1;
    const int64_t Kb = transB_use ? Bd1 : Bd0;

    if (Kb != K) {
      NVTE_ERROR("grouped_gemm_ck_tile: K mismatch between A and B in group ", i);
      return false;
    }

    if (d.shape[0] != M || d.shape[1] != N) {
      NVTE_ERROR("grouped_gemm_ck_tile: D shape mismatch in group ", i);
      return false;
    }

    const ck_tile::index_t stride_A = a.shape[1];
    const ck_tile::index_t stride_B = b.shape[1];
    const ck_tile::index_t stride_E = d.shape[1];

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
    NVTE_ERROR("grouped_gemm_ck_tile: CK_Tile kernel arguments not supported for this config.");
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

static inline bool infer_gemm_mode_group0(const transformer_engine::Tensor* const* A,
                                          const transformer_engine::Tensor* const* B,
                                          transformer_engine::Tensor* const* D,
                                          int group_num,
                                          const transformer_engine::Tensor* const*&  A_use,
                                          const transformer_engine::Tensor* const*&  B_use,
                                          bool& transA_use,
                                          bool& transB_use)
{
  A_use = A;
  B_use = B;
  transA_use = false;
  transB_use = false;

  if (group_num <= 0)
    return true;

  const auto& a0 = data_view(*A[0]);
  const auto& b0 = data_view(*B[0]);
  const auto& d0 = data_view(*D[0]);

  if (a0.shape.size() != 2 || b0.shape.size() != 2 || d0.shape.size() != 2) {
    return false;
  }

  const int64_t Ad0 = a0.shape[0];
  const int64_t Ad1 = a0.shape[1];
  const int64_t Bd0 = b0.shape[0];
  const int64_t Bd1 = b0.shape[1];
  const int64_t Dm  = d0.shape[0];
  const int64_t Dn  = d0.shape[1];

  auto check = [&](bool do_swap, bool ta, bool tb) -> bool {
    const int64_t A0d0 = do_swap ? Bd0 : Ad0;
    const int64_t A0d1 = do_swap ? Bd1 : Ad1;
    const int64_t B0d0 = do_swap ? Ad0 : Bd0;
    const int64_t B0d1 = do_swap ? Ad1 : Bd1;

    const int64_t M  = ta ? A0d1 : A0d0;
    const int64_t K  = ta ? A0d0 : A0d1;
    const int64_t N  = tb ? B0d0 : B0d1;
    const int64_t Kb = tb ? B0d1 : B0d0;

    return (M == Dm) && (N == Dn) && (K == Kb);
  };

  // Try all candidates; prefer "no swap" first, then swap
  for (bool do_swap : {false, true}) {
    for (bool ta : {false, true}) {
      for (bool tb : {false, true}) {
        if (check(do_swap, ta, tb)) {
          A_use = do_swap ? B : A;
          B_use = do_swap ? A : B;
          transA_use = ta;
          transB_use = tb;
          return true;
        }
      }
    }
  }

  // Nothing matched D = op(A) * op(B)
  return false;
}

template <typename T, typename CLayout, ck_tile::memory_operation_enum MemOp>
static inline bool dispatch_grouped(bool transA_use,
                                    bool transB_use,
                                    const transformer_engine::Tensor* const* A_use,
                                    const transformer_engine::Tensor* const* B_use,
                                    transformer_engine::Tensor* const* D,
                                    int group_num,
                                    void* workspace,
                                    size_t workspace_bytes,
                                    hipStream_t stream) {

// FIXME: This could be a templated lambda function in C++20.
#define CALL(ALayout_, BLayout_, ta_, tb_)                                    \
  return run_grouped_impl<T, ALayout_, BLayout_, CLayout, MemOp>(             \
      A_use, B_use, D, group_num, (ta_), (tb_), workspace, workspace_bytes, stream)
  
  if (!transA_use && !transB_use) { CALL(RowMajor, RowMajor, false, false); }
  if (!transA_use &&  transB_use) { CALL(RowMajor, ColMajor, false, true ); }
  if ( transA_use && !transB_use) { CALL(ColMajor, RowMajor, true,  false); }
  /* transA_use && transB_use */  { CALL(ColMajor, ColMajor, true,  true ); }

#undef CALL
}

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

  // Convert A/B/D arrays into TE Tensor arrays
  std::vector<const transformer_engine::Tensor*> A_te(group_num);
  std::vector<const transformer_engine::Tensor*> B_te(group_num);
  std::vector<transformer_engine::Tensor*>       D_te(group_num);

  for (int i = 0; i < group_num; ++i) {
    A_te[i] = transformer_engine::convertNVTETensorCheck(A[i]);
    B_te[i] = transformer_engine::convertNVTETensorCheck(B[i]);
    D_te[i] = transformer_engine::convertNVTETensorCheck(D[i]);
  }

  // Workspace pointer + bytes
  void*  ws_ptr   = nullptr;
  size_t ws_bytes = 0;
  if (workspace) {
    auto* ws_te = transformer_engine::convertNVTETensorCheck(*workspace);
    ws_ptr   = ws_te->data.dptr;
    ws_bytes = ws_te->data.numel() *
               transformer_engine::typeToSize(ws_te->data.dtype);
  }

  const transformer_engine::Tensor* const* A_use = A_te.data();
  const transformer_engine::Tensor* const* B_use = B_te.data();
  bool transA_use = transA;
  bool transB_use = transB;

  // If TE's flags disagree with storage, infer the correct mode from shapes.
  if (!infer_gemm_mode_group0(A_te.data(), B_te.data(), D_te.data(),
                              group_num, A_use, B_use, transA_use, transB_use)) {
    const auto& a0 = data_view(*A_te[0]);
    const auto& b0 = data_view(*B_te[0]);
    const auto& d0 = data_view(*D_te[0]);
    NVTE_ERROR("grouped_gemm_ck_tile: could not infer a consistent GEMM mode from shapes. ",
              "A0=[", a0.shape[0], ",", a0.shape[1], "] ",
              "B0=[", b0.shape[0], ",", b0.shape[1], "] ",
              "D0=[", d0.shape[0], ",", d0.shape[1], "] ",
              "given flags transA=", transA, " transB=", transB);
    return false;
  }

  const auto a_dtype = A_use[0]->dtype();

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(a_dtype, te_type, {
    using T = typename TeTypeToCkType<te_type>::type;

    if (accumulate)
      return dispatch_grouped<T, RowMajor, ck_tile::memory_operation_enum::atomic_add>(transA_use, transB_use,
                                     A_use, B_use, D_te.data(), group_num,
                                     ws_ptr, ws_bytes, stream);
    else
      return dispatch_grouped<T, RowMajor, ck_tile::memory_operation_enum::set>(transA_use, transB_use,
                                     A_use, B_use, D_te.data(), group_num,
                                     ws_ptr, ws_bytes, stream);
  });
}
