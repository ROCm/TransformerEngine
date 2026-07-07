/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVTE C API for AITER's a4w4 (FP4) GEMM.  Translates the C descriptor into
// the aiter_gemm wrapper library's C++ types.  When TE is built without the
// AITER a4w4 backend (USE_AITER_GEMM undefined -- e.g. non-gfx950), the
// symbols still exist but report failure so the framework layer degrades
// gracefully instead of failing to link.

#include "transformer_engine/aiter_gemm.h"

#ifdef USE_AITER_GEMM

#include <hip/hip_runtime.h>

#include "aiter_gemm/aiter_gemm.hpp"

namespace {

aiter_gemm::TensorDesc to_desc(const NVTEAiterGemmTensor *t) {
  aiter_gemm::TensorDesc d;
  d.ptr = t->ptr;
  d.ndim = t->ndim;
  for (int i = 0; i < t->ndim && i < 8; ++i) {
    d.shape[i] = t->shape[i];
    d.strides[i] = t->strides[i];
  }
  d.dtype = static_cast<aiter_gemm::DType>(t->dtype);
  d.device_id = t->device_id;
  return d;
}

}  // namespace

extern "C" int nvte_aiter_gemm_a4w4_blockscale(const NVTEAiterGemmTensor *XQ,
                                               const NVTEAiterGemmTensor *WQ,
                                               const NVTEAiterGemmTensor *x_scale,
                                               const NVTEAiterGemmTensor *w_scale,
                                               const NVTEAiterGemmTensor *Y, int split_k,
                                               const char *kernel_name, void *stream) {
  aiter_gemm::TensorDesc xq = to_desc(XQ);
  aiter_gemm::TensorDesc wq = to_desc(WQ);
  aiter_gemm::TensorDesc xs = to_desc(x_scale);
  aiter_gemm::TensorDesc ws = to_desc(w_scale);
  aiter_gemm::TensorDesc y = to_desc(Y);
  hipError_t err = aiter_gemm::gemm_a4w4_blockscale(
      xq, wq, xs, ws, y, split_k, kernel_name, static_cast<hipStream_t>(stream));
  return err == hipSuccess ? 0 : 1;
}

extern "C" int nvte_aiter_gemm_a4w4_asm(const NVTEAiterGemmTensor *A, const NVTEAiterGemmTensor *B,
                                        const NVTEAiterGemmTensor *a_scale,
                                        const NVTEAiterGemmTensor *b_scale,
                                        const NVTEAiterGemmTensor *out,
                                        const NVTEAiterGemmTensor *bias, const char *kernel_name,
                                        float alpha, float beta, int bpreshuffle, int log2_k_split,
                                        void *stream) {
  aiter_gemm::TensorDesc a = to_desc(A);
  aiter_gemm::TensorDesc b = to_desc(B);
  aiter_gemm::TensorDesc as = to_desc(a_scale);
  aiter_gemm::TensorDesc bs = to_desc(b_scale);
  aiter_gemm::TensorDesc o = to_desc(out);
  aiter_gemm::TensorDesc bias_desc;
  const aiter_gemm::TensorDesc *bias_ptr = nullptr;
  if (bias != nullptr && bias->ptr != nullptr) {
    bias_desc = to_desc(bias);
    bias_ptr = &bias_desc;
  }
  hipError_t err = aiter_gemm::gemm_a4w4_asm(a, b, as, bs, o, bias_ptr, kernel_name, alpha, beta,
                                             bpreshuffle, log2_k_split,
                                             static_cast<hipStream_t>(stream));
  return err == hipSuccess ? 0 : 1;
}

#else  // !USE_AITER_GEMM

extern "C" int nvte_aiter_gemm_a4w4_blockscale(const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *, int, const char *, void *) {
  return -1;
}

extern "C" int nvte_aiter_gemm_a4w4_asm(const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const char *, float, float, int, int, void *) {
  return -1;
}

#endif  // USE_AITER_GEMM
