/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include "blockwise_fp8_gemm.h"
#include "blockwise_fp8_gemm_backend.h"
#include "../../util/hip_runtime.h"

void kittens_blockwise_fp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const void *c_in, float beta,
    void *workspace, size_t workspace_size,
    hipStream_t stream) {
    const int arch = transformer_engine::cuda::sm_arch();

    BlockwiseGemmBackend *backend = nullptr;
    if (arch == 95) {
        backend = get_blockwise_backend_cdna4();
    } else if (arch == 94) {
        backend = get_blockwise_backend_cdna3();
    }
    if (backend == nullptr) {
        throw std::runtime_error(
            "kittens_blockwise_fp8_gemm: not implemented for this GPU arch (sm_arch=" +
            std::to_string(arch) + "). Only gfx942 and gfx950 are supported.");
    }

    BlockwiseGemmArgs args{
        A, B, C, scale_A, scale_B, M, N, K,
        a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
        bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta,
        workspace, workspace_size, stream};
    backend->run(args);
}
