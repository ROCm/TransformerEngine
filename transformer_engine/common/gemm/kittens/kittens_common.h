/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
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

inline bool kittens_is_fp8(int dtype) {
    return dtype == KITTENS_FP8E4M3 || dtype == KITTENS_FP8E5M2;
}

// MFMA cbsz/blgp operand format code, fixed by v_mfma_scale_f32_16x16x128_f8f6f4.
inline int fp8_code(KittensDType dt) {
    switch (dt) {
    case KITTENS_FP8E4M3: return 0;
    case KITTENS_FP8E5M2: return 1;
    default: assert(0 && "unexpected FP8 dtype"); return -1;
    }
}

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
        return get_for_arch(transformer_engine::cuda::sm_arch());
    }

 private:
    static BlockwiseGemmBackend *get_for_arch(int sm_arch);
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

// D = alpha * op(A) op(B) + beta * D plus the epilogue, column-major (cdna4 ignores alpha, grad).
struct GemmArgs {
    const void *A;
    const void *B;
    void *C;
    const void *scale_A;
    const void *scale_B;
    int M, N, K;
    bool transa, transb;
    int a_dtype, b_dtype;
    bool mxfp8_compact;    // A and B are MXFP8 and neither is tagged GEMM-swizzled
    const void *bias;
    int bias_dtype;
    void *aux_gelu;
    int out_dtype, aux_dtype;
    bool grad;             // bias is the dbias output (BGRADB), aux_gelu an input (DGELU)
    float alpha, beta;
    int splitk;            // NVTE_HK_SPLITK: 0 auto, 1 off, S > 1 forced
    int cu_count;
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

/* BF16/FP16 grouped GEMM, D_e = alpha op(A_e) op(B_e) (+ beta D_e) (+ bias_e) for every expert e with
 * m_splits[e] tokens. Operands as TE passes them (row-major, contiguous rows), weight dims N, K uniform:
 *   TN fprop: A_e = W_e [N][K], B_e = X_e [m_e][K],  D_e [m_e][N], bias_e [N] input (bias_dtype)
 *   NN dgrad: A_e = W_e [N][K], B_e = dY_e [m_e][N], D_e [m_e][K]
 *   NT wgrad: A_e = X_e [m_e][K], B_e = dY_e [m_e][N], D_e [N][K], grad: bias_e = dbias_e [N] output
 * Experts with m_e == 0 may have null operands; an NT one still gets D_e zeroed (unless accumulating)
 * and its dbias zeroed. */
struct Grouped16GemmArgs {
    const void *const *A;
    const void *const *B;
    void *const *D;
    const void *const *bias;    // null: no bias
    const int64_t *m_splits;    // host
    int num_gemms;
    int N, K;
    bool transa, transb;
    int in_dtype;               // A and B
    int out_dtype;
    int bias_dtype;
    bool grad;
    bool accumulate;
    float alpha, beta;
    int cu_count;               // workgroup cap; 0: every CU
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

// A backend returns false, before any launch, when it does not take the GEMM.
class GemmBackend {
 public:
    virtual ~GemmBackend() = default;
    virtual bool gemm(const GemmArgs &args) = 0;
    virtual bool grouped_gemm(const MXFP8GroupedGemmArgs &args) = 0;
    virtual bool grouped_wgrad(const MXFP8WgradArgs &args) = 0;
    // On a decline *why gets the reason.
    virtual bool grouped16_gemm(const Grouped16GemmArgs &args, const char **why) = 0;

    static GemmBackend *get() {
        return get_for_arch(transformer_engine::cuda::sm_arch());
    }

 private:
    static GemmBackend *get_for_arch(int sm_arch);
    static GemmBackend *get_cdna4();
    static GemmBackend *get_gfx1250();
};

inline bool kittens_gemm_supported() { return GemmBackend::get() != nullptr; }

inline bool kittens_gemm(const GemmArgs &args) {
    GemmBackend *backend = GemmBackend::get();
    if (backend == nullptr) {
        return false;
    }
    return backend->gemm(args);
}

inline bool kittens_grouped_mxfp8_gemm(
    const void *const *A_array, const void *const *B_array, void *const *C_array,
    const void *const *scale_A_array, const void *const *scale_B_array,
    int M, const int *N_array, int K, int num_experts,
    bool transa, bool transb, int a_dtype, int b_dtype, int out_dtype,
    void *workspace, size_t workspace_size, hipStream_t stream) {
    GemmBackend *backend = GemmBackend::get();
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
    GemmBackend *backend = GemmBackend::get();
    if (backend == nullptr) {
        return false;
    }

    MXFP8WgradArgs args{
        A_array, B_array, D_array, scale_A_array, scale_B_array,
        N, K, M_array, num_experts, a_dtype, b_dtype, out_dtype, accumulate,
        workspace, workspace_size, stream};
    return backend->grouped_wgrad(args);
}

inline bool kittens_grouped16_gemm(const Grouped16GemmArgs &args, const char **why) {
    GemmBackend *backend = GemmBackend::get();
    if (backend == nullptr) {
        *why = "no HipKittens backend for this GPU";
        return false;
    }
    return backend->grouped16_gemm(args, why);
}
