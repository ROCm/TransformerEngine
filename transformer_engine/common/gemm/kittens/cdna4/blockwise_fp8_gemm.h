/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

#ifndef KITTENS_DTYPE_ENUM_DEFINED
#define KITTENS_DTYPE_ENUM_DEFINED
enum KittensDType {
    KITTENS_FLOAT32  = 4,
    KITTENS_FLOAT16  = 5,
    KITTENS_BFLOAT16 = 6,
    KITTENS_FP8E4M3  = 7,
    KITTENS_FP8E5M2  = 8,
};
#endif  // KITTENS_DTYPE_ENUM_DEFINED

#ifndef KITTENS_SCALING_MODE_DEFINED
#define KITTENS_SCALING_MODE_DEFINED
enum KittensScalingMode {
    KITTENS_BLOCK_SCALING_1D = 2,
    KITTENS_BLOCK_SCALING_2D = 3,
};
#endif  // KITTENS_SCALING_MODE_DEFINED

// gfx950 (CDNA4) blockwise FP8 GEMM (TN). C[M,N] = A[M,K] x B[N,K]^T.
// 1Dx2D scaling only: scale_A = per-row 1D [K/128, M], scale_B = per-tile 2D
// [N/128, K/128]; e4m3 x e4m3; bf16 output; M/N/K multiples of 128; no epilogue.
// Returns true if dispatched, false for any unsupported case (caller must NOT
// fall back to the gfx942-only cdna3 kernel on gfx950).
bool kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    bool has_bias, bool has_gelu, bool has_beta,
    hipStream_t stream);
