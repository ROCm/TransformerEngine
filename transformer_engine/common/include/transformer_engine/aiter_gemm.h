/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file aiter_gemm.h
 *  \brief AITER a4w4 (MXFP4 x MXFP4) GEMM (ROCm only).
 */

#ifndef TRANSFORMER_ENGINE_AITER_GEMM_H_
#define TRANSFORMER_ENGINE_AITER_GEMM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute D = A * B^T with AITER's a4w4 kernels.
 *
 *  The backend (CK blockscale or ASM f4gemm) is chosen from \p kernel_name the
 *  same way AITER's own dispatcher does: a non-empty, non-mangled name selects
 *  CK, anything else selects ASM (an empty name lets ASM pick heuristically).
 *
 *  \param[in]     A            MXFP4 tensor of shape [M, K] with row-wise data and scale_inv.
 *  \param[in]     B            MXFP4 tensor of shape [N, K] with row-wise data and scale_inv.
 *                              Data must already be shuffled into AITER's (16, 16) layout.
 *  \param[in,out] D            BF16 or FP16 output of shape [M_pad, N], M_pad = ceil(M / 32) * 32.
 *  \param[in]     kernel_name  Tuned kernel name, or empty/NULL for the backend heuristic.
 *  \param[in]     split_k      Split-K value from the tuned config, passed as-is to the
 *                              selected backend.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Throws if TE was built without the AITER a4w4 backend.
 */
void nvte_aiter_gemm_a4w4(const NVTETensor A, const NVTETensor B, NVTETensor D,
                          const char *kernel_name, int split_k, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_AITER_GEMM_H_
