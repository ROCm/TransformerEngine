/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

#include "kittens_gemm_enums.h"

struct BlockwiseGemmArgs {
    const void *A;
    const void *B;
    void *C;
    const void *scale_A;
    const void *scale_B;
    int M, N, K;
    int a_dtype, b_dtype;
    int a_scaling_mode, b_scaling_mode;
    int out_dtype;
    const void *bias;
    int bias_dtype;
    const void *gelu_aux;
    int gelu_aux_dtype;
    const void *c_in;
    float beta;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

class BlockwiseGemmBackend {
 public:
    virtual ~BlockwiseGemmBackend() = default;
    virtual void run(const BlockwiseGemmArgs &args) = 0;
};

BlockwiseGemmBackend *get_blockwise_backend_cdna3();
BlockwiseGemmBackend *get_blockwise_backend_cdna4();
