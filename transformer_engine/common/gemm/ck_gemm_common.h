/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include "common/util/cuda_runtime.h"

#include "../common.h"


#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace transformer_engine {

using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TEScalar> struct TETypeToCKType;
template <> struct TETypeToCKType<transformer_engine::fp32>    { using type = float; };
template <> struct TETypeToCKType<transformer_engine::fp8e4m3> { using type = ck_tile::fp8_t; };
template <> struct TETypeToCKType<transformer_engine::fp8e5m2> { using type = ck_tile::bf8_t; };
template <> struct TETypeToCKType<transformer_engine::fp16>    { using type = ck_tile::half_t; };
template <> struct TETypeToCKType<transformer_engine::bf16>    { using type = ck_tile::bfloat16_t; };

struct NormalizedGemmInputs {
  const NVTETensor* A;
  const NVTETensor* B;
  bool transA;
  bool transB;
};

struct CkFp8NtPresentation {
  bool transA;
  bool transB;
  bool use_a_colwise_data;
  bool use_b_colwise_data;
};

enum class GPUArch {
  GFX942,
  GFX950,
  GFX1250,
  UNKNOWN
};

static inline GPUArch detect_gpu_arch() {
  int arch = cuda::sm_arch(0);

  if (arch == 94) {
    return GPUArch::GFX942;
  }
  if (arch == 95) {
    return GPUArch::GFX950;
  }
  if (arch == 125 || arch == 1250) {
    return GPUArch::GFX1250;
  }
  return GPUArch::UNKNOWN;
}

struct CKGemmRunContext {
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

template <typename Kernel>
static inline bool has_sufficient_workspace(const CKGemmRunContext& ctx) {
  const size_t needed = Kernel::GetWorkSpaceSize(ctx.group_num);
  if (!ctx.workspace || ctx.workspace_bytes < needed) {
    NVTE_WARN("ck_tile_grouped_gemm: insufficient workspace for CK path. Needed bytes=", needed,
              ", available bytes=", ctx.workspace_bytes, ". Falling back.");
    return false;
  }
  return true;
}

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

static inline const transformer_engine::SimpleTensor& data_view(const transformer_engine::Tensor& t) {
  return t.data;
}

static inline const transformer_engine::SimpleTensor& scale_inv_view(const transformer_engine::Tensor& t) {
  return t.scale_inv;
}

// Normalize similar to upstream
// See https://github.com/NVIDIA/TransformerEngine/blob/59f6f3876767d07045152bfae07b5dd4c54e1725/transformer_engine/common/gemm/cutlass_grouped_gemm.cu#L54-L68
// I.e., swap A and B, as well as transa and transb.
inline NormalizedGemmInputs normalize_gemm_inputs(
    const NVTETensor* A,
    const NVTETensor* B,
    bool transA,
    bool transB) {
  return NormalizedGemmInputs{
    B,
    A,
    transB,
    transA,
  };
}

// FP8 special handling.
//
// A_use/B_use and transA_use/transB_use have already gone through the
// upstream-style grouped GEMM normalization above. This block only rewrites
// that normalized presentation into the CK FP8 preferred NT presentation by selecting
// `columnwise_data` when needed.
//
// CK FP8 target presentation:
//   A_use: N
//   B_use: T
//
// The outer condition checks whether this NT presentation is possible:
//   - A_use is already N, or can be made N using columnwise_data
//   - B_use is already T, or can be made T using columnwise_data
//
// Then each operand is rewritten independently only if needed:
//   NN -> rewrite B only
//   TN -> rewrite A and B
//   NT -> already in target form
//   TT -> rewrite A only
//
// This preserves the intended math and only changes the physical
// storage/transpose-flag encoding seen by CK.
inline CkFp8NtPresentation select_ck_fp8_nt_presentation(
    bool is_8bit_float,
    bool transA,
    bool transB,
    bool has_a_colwise_data,
    bool has_b_colwise_data) {
  CkFp8NtPresentation out{
      transA,
      transB,
      false,
      false,
  };

  if (!is_8bit_float) {
    return out;
  }

  const bool can_make_a_nt = !out.transA || has_a_colwise_data;
  const bool can_make_b_nt = out.transB || has_b_colwise_data;

  if (!can_make_a_nt || !can_make_b_nt) {
    NVTE_ERROR("ck_tile_grouped_gemm: FP8 grouped GEMM requires NT presentation. "
              "Missing required columnwise_data for layout rewrite.",0);
  }

  if (out.transA) {
    out.use_a_colwise_data = true;
    out.transA = false;
  }

  if (!out.transB) {
    out.use_b_colwise_data = true;
    out.transB = true;
  }

  return out;
} 
}  // namespace transformer_engine
