/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

// Values match NVTEDType in transformer_engine.h
enum KittensDType {
    KITTENS_FLOAT32  = 4,
    KITTENS_FLOAT16  = 5,
    KITTENS_BFLOAT16 = 6,
    KITTENS_FP8E4M3  = 7,
    KITTENS_FP8E5M2  = 8,
};

// Workspace sizing (all sub-allocations 256-byte aligned):
//   k_iters = K / 128,  scale_K = K / 32
//   TN:  k_iters*M*4  + k_iters*N*4
//   NN:  M*K + M*scale_K + k_iters*M*4 + k_iters*N*4
//   NT:  M*K + N*K + M*scale_K + N*scale_K + k_iters*M*4 + k_iters*N*4
// Returns false if workspace_size is insufficient.

size_t kittens_mxfp8_workspace_bytes(int M, int N, int K, bool transa, bool transb);

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
