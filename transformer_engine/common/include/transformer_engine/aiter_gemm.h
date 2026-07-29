/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file aiter_gemm.h
 *  \brief C API for AITER a4w4 (FP4 x FP4) GEMM (ROCm only).
 *
 *  Thin executor entry points: kernel selection (tuned-CSV lookup) and
 *  weight/scale pre-shuffling happen in the framework layer; these functions
 *  take a resolved kernel name and already-shuffled inputs.
 *
 *  The kernels themselves live in QoLA-built shared objects. This header
 *  mirrors QoLA's public C ABI so the two descriptors are layout-identical
 *  and the call needs no translation; the mirroring is asserted at compile
 *  time in the implementation. It is declared independently (rather than
 *  including QoLA's header) so that this API stays available on builds where
 *  the AITER a4w4 backend is disabled.
 */

#ifndef TRANSFORMER_ENGINE_AITER_GEMM_H_
#define TRANSFORMER_ENGINE_AITER_GEMM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Element type of an a4w4 GEMM operand. Values match QoLA's
 *         qola_dtype_t. */
typedef enum {
  kNVTEAiterGemmFP4x2 = 0, /*!< two packed FP4 (E2M1) values per byte */
  kNVTEAiterGemmE8M0 = 1,  /*!< 8-bit exponent-only microscale (1 byte) */
  kNVTEAiterGemmBF16 = 2,
  kNVTEAiterGemmFP16 = 3,
  kNVTEAiterGemmFP32 = 4,
  kNVTEAiterGemmU8 = 5,
  kNVTEAiterGemmI8 = 6,
} NVTEAiterGemmDType;

/*! \brief Lightweight device-tensor descriptor (raw pointer + layout).
 *         Layout-identical to QoLA's qola_tensor_t. */
typedef struct {
  void *ptr;
  int32_t ndim;
  int32_t dtype; /*!< one of NVTEAiterGemmDType */
  int32_t device_id;
  int32_t reserved;
  int64_t shape[8];
  int64_t strides[8];
} NVTEAiterGemmTensor;

/*! \brief CK blockscale a4w4 GEMM: Y = XQ @ WQ^T with per-1x32 microscaling.
 *
 *  \param[out] err_buf       Optional buffer receiving a failure message.
 *  \param[in]  err_buf_size  Size of \p err_buf in bytes; 0 to discard.
 *  \return 0 on success, nonzero on failure (or if TE was built without the
 *          AITER a4w4 backend).
 */
int nvte_aiter_gemm_a4w4_blockscale(const NVTEAiterGemmTensor *XQ, const NVTEAiterGemmTensor *WQ,
                                    const NVTEAiterGemmTensor *x_scale,
                                    const NVTEAiterGemmTensor *w_scale, const NVTEAiterGemmTensor *Y,
                                    int split_k, const char *kernel_name, void *stream,
                                    char *err_buf, size_t err_buf_size);

/*! \brief ASM (f4gemm) a4w4 GEMM: D = alpha*A*B + beta*C. `bias` may be NULL.
 *  \return 0 on success, nonzero on failure (or if TE was built without the
 *          AITER a4w4 backend).
 */
int nvte_aiter_gemm_a4w4_asm(const NVTEAiterGemmTensor *A, const NVTEAiterGemmTensor *B,
                             const NVTEAiterGemmTensor *a_scale, const NVTEAiterGemmTensor *b_scale,
                             const NVTEAiterGemmTensor *out, const NVTEAiterGemmTensor *bias,
                             const char *kernel_name, float alpha, float beta, int bpreshuffle,
                             int log2_k_split, void *stream, char *err_buf, size_t err_buf_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_AITER_GEMM_H_
