/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

namespace te_mxfp4 {

extern "C" void launch_cast_transpose_mxfp4(
    const void* input,
    void* rowwise_fp4,
    void* rowwise_scale,
    void* colwise_fp4,
    void* colwise_scale,
    int M, int N,
    bool use_rowwise,
    bool use_colwise,
    bool shuffle_scales,
    bool use_hadamard,
    bool shuffle_rowwise_fp4,
    bool shuffle_colwise_fp4,
    int rowwise_scale_stride,
    int colwise_scale_stride,
    int rowwise_scale_N,
    int rowwise_scale_M_pad,
    int rowwise_scale_N_pad,
    int colwise_scale_M,
    int colwise_scale_N,
    int colwise_scale_M_pad,
    int colwise_scale_N_pad,
    hipStream_t stream);

}  // namespace te_mxfp4
