/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include "blockwise_fp8_gemm.h"
#include "cdna3/blockwise_fp8_gemm.h"
#include "cdna4/blockwise_fp8_gemm.h"
#include "../../util/hip_runtime.h"

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
    hipStream_t stream) {
    const int arch = transformer_engine::cuda::sm_arch();
    if (arch != 94 && arch != 95) {
        throw std::runtime_error(
            "kittens_blockwise_fp8_gemm: not implemented for this GPU arch (sm_arch=" +
            std::to_string(arch) + "). Only gfx942 and gfx950 are supported.");
    }

    const bool a_fp8 = (a_dtype == KITTENS_FP8E4M3 || a_dtype == KITTENS_FP8E5M2);
    const bool b_fp8 = (b_dtype == KITTENS_FP8E4M3 || b_dtype == KITTENS_FP8E5M2);
    const bool out_ok = (out_dtype == KITTENS_BFLOAT16 || out_dtype == KITTENS_FLOAT16 ||
                         out_dtype == KITTENS_FLOAT32);
    const bool scaling_ok = (b_scaling_mode == KITTENS_BLOCK_SCALING_1D) &&
                            (a_scaling_mode == KITTENS_BLOCK_SCALING_1D ||
                             a_scaling_mode == KITTENS_BLOCK_SCALING_2D);
    if (!a_fp8 || !b_fp8 || !out_ok || !scaling_ok || (transa && transb)) {
        throw std::runtime_error(
            "kittens_blockwise_fp8_gemm: unsupported config");
    }

    if (arch == 95) {
        blockwise_gfx950::kittens_blockwise_fp8_gemm_impl_cdna4(
            A, B, C, scale_A, scale_B, M, N, K,
            a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
            bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta, stream);
    } else {
        blockwise_gfx942::kittens_blockwise_fp8_gemm_impl_cdna3(
            A, B, C, scale_A, scale_B, M, N, K, transa, transb,
            a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
            bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta, stream);
    }
}
