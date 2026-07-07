/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "aiter_gemm/aiter_gemm.hpp"

#include <cstdio>
#include <exception>
#include <string>

#include "aiter_tensor.h"  // aiter_tensor_t, AiterDtype
#include "qola_gemm_a4w4_asm.h"
#include "qola_gemm_a4w4_blockscale.h"

namespace aiter_gemm {

namespace {

AiterDtype to_aiter_dtype(DType dt) {
  switch (dt) {
    case DType::fp4x2:
      return AITER_DTYPE_fp4x2;
    case DType::e8m0:
      return AITER_DTYPE_fp8_e8m0;
    case DType::bf16:
      return AITER_DTYPE_bf16;
    case DType::fp16:
      return AITER_DTYPE_fp16;
    case DType::fp32:
      return AITER_DTYPE_fp32;
    case DType::u8:
      return AITER_DTYPE_u8;
    case DType::i8:
      return AITER_DTYPE_i8;
  }
  return AITER_DTYPE_u8;
}

// Build an aiter_tensor_t POD from a TE-side descriptor.  Shares the caller's
// device pointer; no ownership is transferred.
aiter_tensor_t to_aiter_tensor(const TensorDesc& d) {
  aiter_tensor_t t{};
  t.ptr = d.ptr;
  t.ndim = d.ndim;
  size_t numel = (d.ndim > 0) ? 1 : 0;
  for (int i = 0; i < d.ndim; ++i) {
    t.shape[i] = d.shape[i];
    t.strides[i] = d.strides[i];
    numel *= static_cast<size_t>(d.shape[i]);
  }
  t.numel_ = numel;
  t.dtype_ = to_aiter_dtype(d.dtype);
  t.device_id = d.device_id;
  return t;
}

}  // namespace

hipError_t gemm_a4w4_blockscale(const TensorDesc& XQ,
                                const TensorDesc& WQ,
                                const TensorDesc& x_scale,
                                const TensorDesc& w_scale,
                                const TensorDesc& Y,
                                int split_k,
                                const char* kernel_name,
                                hipStream_t stream) {
  try {
    aiter_tensor_t a_xq = to_aiter_tensor(XQ);
    aiter_tensor_t a_wq = to_aiter_tensor(WQ);
    aiter_tensor_t a_xs = to_aiter_tensor(x_scale);
    aiter_tensor_t a_ws = to_aiter_tensor(w_scale);
    aiter_tensor_t a_y = to_aiter_tensor(Y);
    QOLA_NS(gemm_a4w4_blockscale)
    (a_xq, a_wq, a_xs, a_ws, a_y, split_k, stream,
     kernel_name ? std::string(kernel_name) : std::string());
    return hipSuccess;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[aiter_gemm] gemm_a4w4_blockscale failed: %s\n", e.what());
    return hipErrorUnknown;
  }
}

hipError_t gemm_a4w4_asm(const TensorDesc& A,
                         const TensorDesc& B,
                         const TensorDesc& a_scale,
                         const TensorDesc& b_scale,
                         const TensorDesc& out,
                         const TensorDesc* bias,
                         const char* kernel_name,
                         float alpha,
                         float beta,
                         int bpreshuffle,
                         int log2_k_split,
                         hipStream_t stream) {
  try {
    aiter_tensor_t a_a = to_aiter_tensor(A);
    aiter_tensor_t a_b = to_aiter_tensor(B);
    aiter_tensor_t a_as = to_aiter_tensor(a_scale);
    aiter_tensor_t a_bs = to_aiter_tensor(b_scale);
    aiter_tensor_t a_out = to_aiter_tensor(out);
    aiter_tensor_t a_bias;
    aiter_tensor_t* a_bias_ptr = nullptr;
    if (bias != nullptr && bias->ptr != nullptr) {
      a_bias = to_aiter_tensor(*bias);
      a_bias_ptr = &a_bias;
    }
    QOLA_NS(gemm_a4w4_asm)
    (&a_a, &a_b, &a_as, &a_bs, &a_out, kernel_name ? kernel_name : "", a_bias_ptr, alpha, beta,
     bpreshuffle, log2_k_split, stream);
    return hipSuccess;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[aiter_gemm] gemm_a4w4_asm failed: %s\n", e.what());
    return hipErrorUnknown;
  }
}

}  // namespace aiter_gemm
