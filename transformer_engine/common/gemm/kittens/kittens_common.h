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

    static BlockwiseGemmBackend *get() {
        const int arch = transformer_engine::cuda::sm_arch();
#ifdef KITTENS_HAVE_CDNA4
        if (arch == 95) {
            return get_cdna4();
        }
#endif
#ifdef KITTENS_HAVE_CDNA3
        if (arch == 94) {
            return get_cdna3();
        }
#endif
        static_cast<void>(arch);
        return nullptr;
    }

 private:
    static BlockwiseGemmBackend *get_cdna3();
    static BlockwiseGemmBackend *get_cdna4();
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
    BlockwiseGemmBackend *backend = BlockwiseGemmBackend::get();
    if (backend == nullptr) {
        throw std::runtime_error(
            "kittens_blockwise_fp8_gemm: not implemented for this GPU arch (sm_arch=" +
            std::to_string(transformer_engine::cuda::sm_arch()) + "). Only gfx942 and gfx950 are supported.");
    }

    BlockwiseGemmArgs args{
        A, B, C, scale_A, scale_B, M, N, K,
        a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
        bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta,
        workspace, workspace_size, stream};
    backend->run(args);
}

struct MXFP8GemmArgs {
    const void *A;
    const void *B;
    void *C;
    const void *scale_A;
    const void *scale_B;
    int M, N, K;
    bool transa, transb;
    int a_dtype, b_dtype;
    const void *bias;
    int bias_dtype;
    void *aux_gelu;
    int out_dtype, aux_dtype;
    float beta;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

struct MXFP8GroupedGemmArgs {
    const void *const *A_array;
    const void *const *B_array;
    void *const *C_array;
    const void *const *scale_A_array;
    const void *const *scale_B_array;
    int M;
    const int *N_array;
    int K;
    int num_experts;
    bool transa, transb;
    int a_dtype, b_dtype, out_dtype;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

struct MXFP8WgradArgs {
    const void *const *A_array;
    const void *const *B_array;
    void *const *D_array;
    const void *const *scale_A_array;
    const void *const *scale_B_array;
    int N, K;
    const int *M_array;
    int num_experts;
    int a_dtype, b_dtype, out_dtype;
    bool accumulate;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

class MXFP8GemmBackend {
 public:
    virtual ~MXFP8GemmBackend() = default;
    virtual bool gemm(const MXFP8GemmArgs &args) = 0;
    virtual bool grouped_gemm(const MXFP8GroupedGemmArgs &args) = 0;
    virtual bool grouped_wgrad(const MXFP8WgradArgs &args) = 0;

    static MXFP8GemmBackend *get() {
#ifdef KITTENS_HAVE_CDNA4
        if (transformer_engine::cuda::sm_arch() == 95) {
            return get_cdna4();
        }
#endif
        return nullptr;
    }

 private:
    static MXFP8GemmBackend *get_cdna4();
};

inline bool kittens_mxfp8_supported() { return MXFP8GemmBackend::get() != nullptr; }

inline bool kittens_mxfp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    const void *bias, int bias_dtype,
    void *aux_gelu, int out_dtype, int aux_dtype,
    float beta, void *workspace, size_t workspace_size,
    hipStream_t stream) {
    MXFP8GemmBackend *backend = MXFP8GemmBackend::get();
    if (backend == nullptr) {
        return false;
    }

    MXFP8GemmArgs args{
        A, B, C, scale_A, scale_B, M, N, K, transa, transb,
        a_dtype, b_dtype, bias, bias_dtype, aux_gelu, out_dtype, aux_dtype,
        beta, workspace, workspace_size, stream};
    return backend->gemm(args);
}

inline bool kittens_grouped_mxfp8_gemm(
    const void *const *A_array, const void *const *B_array, void *const *C_array,
    const void *const *scale_A_array, const void *const *scale_B_array,
    int M, const int *N_array, int K, int num_experts,
    bool transa, bool transb, int a_dtype, int b_dtype, int out_dtype,
    void *workspace, size_t workspace_size, hipStream_t stream) {
    MXFP8GemmBackend *backend = MXFP8GemmBackend::get();
    if (backend == nullptr) {
        return false;
    }

    MXFP8GroupedGemmArgs args{
        A_array, B_array, C_array, scale_A_array, scale_B_array,
        M, N_array, K, num_experts, transa, transb,
        a_dtype, b_dtype, out_dtype, workspace, workspace_size, stream};
    return backend->grouped_gemm(args);
}

inline bool kittens_grouped_mxfp8_wgrad(
    const void *const *A_array, const void *const *B_array, void *const *D_array,
    const void *const *scale_A_array, const void *const *scale_B_array,
    int N, int K, const int *M_array, int num_experts,
    int a_dtype, int b_dtype, int out_dtype,
    bool accumulate,
    void *workspace, size_t workspace_size, hipStream_t stream) {
    MXFP8GemmBackend *backend = MXFP8GemmBackend::get();
    if (backend == nullptr) {
        return false;
    }

    MXFP8WgradArgs args{
        A_array, B_array, D_array, scale_A_array, scale_B_array,
        N, K, M_array, num_experts, a_dtype, b_dtype, out_dtype, accumulate,
        workspace, workspace_size, stream};
    return backend->grouped_wgrad(args);
}
