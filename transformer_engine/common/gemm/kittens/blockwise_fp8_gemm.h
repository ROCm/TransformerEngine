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

void kittens_blockwise_fp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const void *c_in, float beta,
    hipStream_t stream);
