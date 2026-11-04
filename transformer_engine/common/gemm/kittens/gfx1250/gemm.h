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

namespace te_kittens::gfx1250 {

// TE BLAS layouts: TN forward, NN dgrad, NT wgrad.
enum Layout { TN, NN, NT };

// Epilogue of a dense kernel, fixed at compile time. SPLIT writes raw fp32 partials [S][N][M].
enum Epi { NONE, BIAS, GELU_AUX, GELU_AUX_BIAS, DGELU, BGRADB, SPLIT };

// D = alpha * op(A) op(B) (+ beta * D with acc), column-major M x N, then the epilogue; BGRADB
// writes bias[n] = sum_k B(n, k). D is out_dtype, bias and aux side_dtype.
struct Launch {
    const void *A;
    const void *B;
    const void *scale_A;
    const void *scale_B;
    void *D;
    const void *bias;      // dbias output for BGRADB
    void *aux;
    void *workspace;       // fp32 partials [splits][N][M] when splits > 1
    float alpha, beta;
    int M, N, K, splits;
    int a_dtype, b_dtype, out_dtype, side_dtype;
    Epi epi;
    bool acc;
    bool mcast;            // cluster kernels with TDM multicast, else no cluster
    hipStream_t stream;
};

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

// false (before any launch) when no instance matches; launch failures throw.
template <Layout L> bool launch_mxfp8(const Launch &l);
template <Layout L> bool launch_bf16(const Launch &l);

// D = alpha * (sum of the l.splits partials in split order) (+ beta * D).
void splitk_reduce(const Launch &l);

inline void kittens_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

// One instantiated epilogue variant of a dense kernel family; bias and aux hold SideT.
template <Epi E_, bool ACC_, typename OutT_, typename SideT_ = OutT_>
struct Variant {
    static constexpr Epi E = E_;
    static constexpr bool ACC = ACC_;
    using OutT  = OutT_;
    using SideT = SideT_;
};

template <typename... T>
struct List {};

template <typename... T, typename F>
bool any_of(List<T...>, F &&f) {
    return (f(T{}) || ...);
}

template <typename T>
constexpr int dtype_of = std::is_same_v<T, float>      ? KITTENS_FLOAT32
                         : std::is_same_v<T, _Float16> ? KITTENS_FLOAT16
                                                       : KITTENS_BFLOAT16;

}  // namespace te_kittens::gfx1250
