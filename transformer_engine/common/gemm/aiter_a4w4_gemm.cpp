/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVTE C API for AITER's a4w4 (FP4) GEMM.  QoLA exports a C ABI over a
// descriptor that is layout-identical to NVTEAiterGemmTensor, so these are
// pure forwards -- the static_asserts below are what keep that true.
//
// When TE is built without the AITER a4w4 backend (USE_AITER_GEMM undefined
// -- e.g. non-gfx950), the symbols still exist but report failure so the
// framework layer degrades gracefully instead of failing to link.

#include "transformer_engine/aiter_gemm.h"

#ifdef USE_AITER_GEMM

#include <hip/hip_runtime.h>

#include <cstddef>

#include "qola_gemm_a4w4.h"

namespace {

static_assert(sizeof(NVTEAiterGemmTensor) == sizeof(qola_tensor_t),
              "NVTEAiterGemmTensor must mirror qola_tensor_t");
static_assert(offsetof(NVTEAiterGemmTensor, ptr) == offsetof(qola_tensor_t, ptr),
              "NVTEAiterGemmTensor::ptr must mirror qola_tensor_t::ptr");
static_assert(offsetof(NVTEAiterGemmTensor, ndim) == offsetof(qola_tensor_t, ndim),
              "NVTEAiterGemmTensor::ndim must mirror qola_tensor_t::ndim");
static_assert(offsetof(NVTEAiterGemmTensor, dtype) == offsetof(qola_tensor_t, dtype),
              "NVTEAiterGemmTensor::dtype must mirror qola_tensor_t::dtype");
static_assert(offsetof(NVTEAiterGemmTensor, device_id) == offsetof(qola_tensor_t, device_id),
              "NVTEAiterGemmTensor::device_id must mirror qola_tensor_t::device_id");
static_assert(offsetof(NVTEAiterGemmTensor, shape) == offsetof(qola_tensor_t, shape),
              "NVTEAiterGemmTensor::shape must mirror qola_tensor_t::shape");
static_assert(offsetof(NVTEAiterGemmTensor, strides) == offsetof(qola_tensor_t, strides),
              "NVTEAiterGemmTensor::strides must mirror qola_tensor_t::strides");

// The dtype enumerators are part of the shared ABI too.  Compared as ints
// because the two enums are deliberately distinct types.
static_assert(static_cast<int>(kNVTEAiterGemmFP4x2) == static_cast<int>(QOLA_DTYPE_FP4X2),
              "a4w4 dtype enum drift");
static_assert(static_cast<int>(kNVTEAiterGemmE8M0) == static_cast<int>(QOLA_DTYPE_E8M0),
              "a4w4 dtype enum drift");
static_assert(static_cast<int>(kNVTEAiterGemmBF16) == static_cast<int>(QOLA_DTYPE_BF16),
              "a4w4 dtype enum drift");
static_assert(static_cast<int>(kNVTEAiterGemmFP16) == static_cast<int>(QOLA_DTYPE_FP16),
              "a4w4 dtype enum drift");
static_assert(static_cast<int>(kNVTEAiterGemmFP32) == static_cast<int>(QOLA_DTYPE_FP32),
              "a4w4 dtype enum drift");

inline const qola_tensor_t *as_qola(const NVTEAiterGemmTensor *t) {
  return reinterpret_cast<const qola_tensor_t *>(t);
}

}  // namespace

extern "C" int nvte_aiter_gemm_a4w4_blockscale(const NVTEAiterGemmTensor *XQ,
                                               const NVTEAiterGemmTensor *WQ,
                                               const NVTEAiterGemmTensor *x_scale,
                                               const NVTEAiterGemmTensor *w_scale,
                                               const NVTEAiterGemmTensor *Y, int split_k,
                                               const char *kernel_name, void *stream, char *err_buf,
                                               size_t err_buf_size) {
  return QOLA_C(gemm_a4w4_blockscale)(as_qola(XQ), as_qola(WQ), as_qola(x_scale), as_qola(w_scale),
                                      as_qola(Y), split_k, kernel_name,
                                      static_cast<hipStream_t>(stream), err_buf, err_buf_size);
}

extern "C" int nvte_aiter_gemm_a4w4_asm(const NVTEAiterGemmTensor *A, const NVTEAiterGemmTensor *B,
                                        const NVTEAiterGemmTensor *a_scale,
                                        const NVTEAiterGemmTensor *b_scale,
                                        const NVTEAiterGemmTensor *out,
                                        const NVTEAiterGemmTensor *bias, const char *kernel_name,
                                        float alpha, float beta, int bpreshuffle, int log2_k_split,
                                        void *stream, char *err_buf, size_t err_buf_size) {
  return QOLA_C(gemm_a4w4_asm)(as_qola(A), as_qola(B), as_qola(a_scale), as_qola(b_scale),
                               as_qola(out), as_qola(bias), kernel_name, alpha, beta, bpreshuffle,
                               log2_k_split, static_cast<hipStream_t>(stream), err_buf,
                               err_buf_size);
}

#else  // !USE_AITER_GEMM

#include <cstring>

namespace {

void report_disabled(char *err_buf, size_t err_buf_size) {
  const char *msg = "TransformerEngine was built without the AITER a4w4 GEMM backend";
  if (err_buf != nullptr && err_buf_size > 0) {
    std::strncpy(err_buf, msg, err_buf_size - 1);
    err_buf[err_buf_size - 1] = '\0';
  }
}

}  // namespace

extern "C" int nvte_aiter_gemm_a4w4_blockscale(const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *,
                                               const NVTEAiterGemmTensor *, int, const char *,
                                               void *, char *err_buf, size_t err_buf_size) {
  report_disabled(err_buf, err_buf_size);
  return -1;
}

extern "C" int nvte_aiter_gemm_a4w4_asm(const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const NVTEAiterGemmTensor *, const NVTEAiterGemmTensor *,
                                        const char *, float, float, int, int, void *, char *err_buf,
                                        size_t err_buf_size) {
  report_disabled(err_buf, err_buf_size);
  return -1;
}

#endif  // USE_AITER_GEMM
