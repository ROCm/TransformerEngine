/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <type_traits>
#include <vector>
#include <memory>

#include <transformer_engine/transformer_engine.h>
#include "../../common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace transformer_engine {
namespace grouped_gemm {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TileCfg>
using GroupedGemmShape = ck_tile::TileGemmShape<
    ck_tile::sequence<TileCfg::M_Tile, TileCfg::N_Tile, TileCfg::K_Tile>,
    ck_tile::sequence<TileCfg::M_Warp, TileCfg::N_Warp, TileCfg::K_Warp>,
    ck_tile::sequence<TileCfg::M_Warp_Tile, TileCfg::N_Warp_Tile, TileCfg::K_Warp_Tile>>;

template <typename TileCfg>
using GroupedGemmPartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
    GroupedGemmShape<TileCfg>,
    TileCfg::TilePartitionerGroupNum,
    TileCfg::TilePartitionerM01>;

template <typename TEScalar> struct TETypeToCKType;
template <> struct TETypeToCKType<transformer_engine::fp32>    { using type = float; };
template <> struct TETypeToCKType<transformer_engine::fp8e4m3> { using type = ck_tile::fp8_t; };
template <> struct TETypeToCKType<transformer_engine::fp8e5m2> { using type = ck_tile::bf8_t; };
template <> struct TETypeToCKType<transformer_engine::fp16>    { using type = ck_tile::half_t; };
template <> struct TETypeToCKType<transformer_engine::bf16>    { using type = ck_tile::bfloat16_t; };

// Selects epilogue traits based on whether we are accumulating (D += A*B) or not (D = A*B).
// For accumulate=true, the existing D buffer is passed as a MultiD input tensor and combined
// via element_wise::Add. For accumulate=false, no extra input is needed and PassThrough is used.
template <typename CType, typename CLayout, bool Accumulate>
struct EpilogueTraits {
  using DsDataType = ck_tile::tuple<>;
  using DsLayout   = ck_tile::tuple<>;
  using ElemOp     = ck_tile::element_wise::PassThrough;
};

template <typename CType, typename CLayout>
struct EpilogueTraits<CType, CLayout, true> {
  using DsDataType = ck_tile::tuple<CType>;
  using DsLayout   = ck_tile::tuple<CLayout>;
  using ElemOp     = ck_tile::element_wise::Add;
};

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data;
}

static inline const transformer_engine::SimpleTensor& scale_inv_view(const transformer_engine::Tensor& t) {
  return t.scale_inv;
}

struct GroupedGemmRunContext {
    const NVTETensor* A = nullptr;
    const NVTETensor* B = nullptr;
    NVTETensor* D = nullptr;
    int64_t N = 0;

    int group_num = 0;
    bool transA = false;
    bool transB = false;

    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    hipStream_t stream = nullptr;

    bool use_a_columnwise_data = false;
    bool use_b_columnwise_data = false;
    bool accumulate = false;
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

// Extract GEMM dims from columnwise storage.
// This path expects columnwise_data to already be normalized to a 2D layout.
static inline bool get_columnwise_storage_2d_dims(
    const transformer_engine::SimpleTensor& t,
    int64_t& d0,
    int64_t& d1) {

  if (t.shape.size() != 2) {
    return false;
  }

  d0 = static_cast<int64_t>(t.shape[0]);
  d1 = static_cast<int64_t>(t.shape[1]);
  return true;
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

template <typename Kernel, typename DescContainer>
static inline bool launch_grouped_gemm_kernel(const DescContainer& descs,
                                              const GroupedGemmRunContext& ctx,
                                              const ck_tile::stream_config& stream_cfg) {
  constexpr int kBlockPerCu = 1;

  const dim3 blocks = Kernel::BlockSize();
  const dim3 grids  = Kernel::GridSize(descs);
  auto kargs = Kernel::MakeKargs(descs);

  if (!Kernel::IsSupportedArgument(kargs)) {
    NVTE_WARN("ck_tile_grouped_gemm: CK_Tile kernel arguments not supported for this config. "
              "transA=", ctx.transA, " transB=", ctx.transB,
              " accumulate=", ctx.accumulate, " groups=", ctx.group_num,
              ". Falling back. "
              "CK_Tile constraints for bf16/fp16: "
              "contiguous dim of A and B must be dword-aligned (even).");
    for (size_t i = 0; i < descs.size(); ++i) {
      NVTE_WARN("  group ", i, ": M=", descs[i].M, " N=", descs[i].N, " K=", descs[i].K,
                " stride_A=", descs[i].stride_A, " stride_B=", descs[i].stride_B,
                " stride_E=", descs[i].stride_E);
    }
    return false;
  }

  NVTE_CHECK_CUDA(hipMemcpyAsync(ctx.workspace,
                                  kargs.data(),
                                  kargs.size() * sizeof(typename decltype(kargs)::value_type),
                                  hipMemcpyHostToDevice,
                                  ctx.stream));

  ck_tile::launch_kernel(
      stream_cfg, ck_tile::make_kernel<kBlockPerCu>(
                      Kernel{}, grids, blocks, 0,
                      ck_tile::cast_pointer_to_constant_address_space(ctx.workspace),
                      ctx.group_num));
  return true;
}

class RunnerInterface {
public:
    virtual ~RunnerInterface() = default;
    virtual bool run(const ck_tile::stream_config& stream_cfg,
                     const GroupedGemmRunContext& ctx) = 0;
};

}  // namespace grouped_gemm
}  // namespace transformer_engine
