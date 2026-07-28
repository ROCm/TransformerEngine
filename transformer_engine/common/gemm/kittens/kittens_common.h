/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "../../util/hip_runtime.h"

// Values match NVTEDType in transformer_engine.h
enum KittensDType {
    KITTENS_FLOAT32  = 4,
    KITTENS_FLOAT16  = 5,
    KITTENS_BFLOAT16 = 6,
    KITTENS_FP8E4M3  = 7,
    KITTENS_FP8E5M2  = 8,
};

// Values match NVTEScalingMode in transformer_engine.h
enum KittensScalingMode {
    KITTENS_BLOCK_SCALING_1D = 2,
    KITTENS_BLOCK_SCALING_2D = 3,
};

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

 private:
    static BlockwiseGemmBackend *get_cdna3();
    static BlockwiseGemmBackend *get_cdna4();

    friend inline void kittens_blockwise_fp8_gemm(
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
        hipStream_t stream);
};

inline void kittens_blockwise_fp8_gemm(
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
#ifdef KITTENS_HAVE_CDNA4
    if (arch == 95) {
        backend = BlockwiseGemmBackend::get_cdna4();
    }
#endif
#ifdef KITTENS_HAVE_CDNA3
    if (arch == 94) {
        backend = BlockwiseGemmBackend::get_cdna3();
    }
#endif
    if (backend == nullptr) {
        throw std::runtime_error(
            "kittens_blockwise_fp8_gemm: not built for this GPU arch (sm_arch=" +
            std::to_string(arch) + "). This build includes only the HipKittens "
            "backends compiled into it; rebuild with the matching gfx target.");
    }

    BlockwiseGemmArgs args{
        A, B, C, scale_A, scale_B, M, N, K,
        a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
        bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta,
        workspace, workspace_size, stream};
    backend->run(args);
}
