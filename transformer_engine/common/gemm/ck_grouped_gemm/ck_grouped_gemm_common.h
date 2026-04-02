/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <type_traits>
#include <vector>
#include <memory>

#include <transformer_engine/transformer_engine.h>
#include "../../common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace transformer_engine {
namespace grouped_gemm {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

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
    cudaStream_t stream = nullptr;

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

bool ck_tile_grouped_gemm_fp16_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx);

bool ck_tile_grouped_gemm_fp8_dispatch(DType a_dtype,
                                       DType b_dtype,
                                       DType d_dtype,
                                       const GroupedGemmRunContext& ctx);

class RunnerInterface {
public:
    virtual ~RunnerInterface() = default;
    virtual bool run(const ck_tile::stream_config& stream_cfg,
                const GroupedGemmRunContext& ctx) = 0;
};
                                                 
}  // namespace grouped_gemm
}  // namespace transformer_engine
