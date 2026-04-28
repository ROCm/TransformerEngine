/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

// dtype codes match NVTEDType values:
//   4 = float32, 5 = float16, 6 = bfloat16, 7 = fp8e4m3, 8 = fp8e5m2

bool kittens_mxfp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    const void *bias, int bias_dtype,
    void *aux_gelu, int out_dtype, int aux_dtype,
    void *workspace, size_t workspace_size,
    hipStream_t stream);
