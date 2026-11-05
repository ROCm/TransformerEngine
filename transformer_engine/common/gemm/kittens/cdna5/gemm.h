/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include <stdexcept>
#include <string>
#include <type_traits>
#include "../kittens_common.h"

namespace te_kittens::cdna5 {

// TE BLAS layouts: TN forward, NN dgrad, NT wgrad.
enum Layout { TN, NN, NT };

// Epilogue of a dense kernel, fixed at compile time. SPLIT writes raw fp32 partials [splits][N][M].
enum Epi { NONE, BIAS, GELU_AUX, GELU_AUX_BIAS, DGELU, BGRADB, SPLIT };

constexpr int BLOCK_M       = 256;
constexpr int BLOCK_N       = 256;
constexpr int NUM_THREADS   = 256;
constexpr int BF16_BLOCK_K  = 128;
constexpr int MXFP8_BLOCK_K = 256;
constexpr int LDS_MAX_BYTES = 320 * 1024;

typedef int v4i __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// TDM cluster multicast runs on B0 only; A0 reports gfx1250-strict. A build with
// NVTE_HK_GFX1250_MCAST=0/1 fixes the choice.
#ifdef NVTE_HK_GFX1250_MCAST
inline bool use_mcast() { return NVTE_HK_GFX1250_MCAST != 0; }
#else
inline bool use_mcast() {
    return transformer_engine::cuda::sm_arch_name().find("-strict") == std::string::npos;
}
#endif

// Device code built for gfx1250-strict (A0) never multicasts, whichever kernel runs.
#ifdef __gfx1250_strict__
constexpr bool ARCH_MCAST = false;
#else
constexpr bool ARCH_MCAST = true;
#endif

// D = alpha * op(A) op(B) (+ beta * D) then the epilogue; false, before any launch, when no kernel matches.
template <Layout L>
bool launch_mxfp8(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template <Layout L>
bool launch_bf16(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);

// D = alpha * (sum of the splits fp32 partials in workspace, in split order) (+ beta * D).
void splitk_reduce(const GemmArgs &args, const void *workspace, int splits);

inline void kittens_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

// Bias and aux share one dtype.
inline int side_dtype(const GemmArgs &args) {
    if (args.bias != nullptr) {
        return args.bias_dtype;
    }
    return args.aux_gelu != nullptr ? args.aux_dtype : args.out_dtype;
}

template <typename T>
constexpr int dtype_of = std::is_same_v<T, float>      ? KITTENS_FLOAT32
                         : std::is_same_v<T, _Float16> ? KITTENS_FLOAT16
                                                       : KITTENS_BFLOAT16;

}  // namespace te_kittens::cdna5
