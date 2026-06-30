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

// gfx950 (CDNA4) blockwise FP8 GEMM launcher.
// DeepSeek-style TN GEMM: A[M,K] x B[N,K] -> C[N,M] bf16 (column-major).
// e4m3 inputs only; scale_A [M/128, K/128], scale_B [N/128, K/128].
// Returns true if the shape was dispatched, false if unsupported (caller
// should fall back). M, N, K must be multiples of 128.
bool kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    hipStream_t stream);
