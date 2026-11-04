/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

#define KITTENS_BOOL_SWITCH(val, NAME, ...) \
    if (val) { constexpr bool NAME = true; __VA_ARGS__ } \
    else { constexpr bool NAME = false; __VA_ARGS__ }

static inline size_t kittens_align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

namespace te_kittens::blockwise {

// dtype codes are NVTEDType: 6 = bfloat16, 5 = float16, anything else = float32
__device__ inline float read_elem(const void *p, int dtype, int idx) {
    if (dtype == 6) return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(p)[idx]);
    if (dtype == 5) return __half2float(reinterpret_cast<const __half *>(p)[idx]);
    return reinterpret_cast<const float *>(p)[idx];
}

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    BETA,
    BIAS_BETA,
    GELU_AUX_BETA,
};

__host__ __device__ inline constexpr bool epilogue_has_bias(GemmEpilogue e) {
    return e == GemmEpilogue::BIAS || e == GemmEpilogue::BIAS_BETA;
}
__host__ __device__ inline constexpr bool epilogue_has_gelu(GemmEpilogue e) {
    return e == GemmEpilogue::GELU_AUX || e == GemmEpilogue::GELU_AUX_BETA;
}
__host__ __device__ inline constexpr bool epilogue_has_beta(GemmEpilogue e) {
    return e == GemmEpilogue::BETA || e == GemmEpilogue::BIAS_BETA
        || e == GemmEpilogue::GELU_AUX_BETA;
}

inline GemmEpilogue select_epilogue(bool has_bias, bool has_gelu, bool has_beta) {
    if (has_gelu) return has_beta ? GemmEpilogue::GELU_AUX_BETA : GemmEpilogue::GELU_AUX;
    if (has_bias) return has_beta ? GemmEpilogue::BIAS_BETA     : GemmEpilogue::BIAS;
    return has_beta ? GemmEpilogue::BETA : GemmEpilogue::DEFAULT;
}

}  // namespace te_kittens::blockwise

namespace te_kittens {

typedef float v8f __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));

constexpr int REDUCE_THREADS = 256;

template <typename T>
__device__ __forceinline__ __attribute__((address_space(3))) T *lds_ptr(uint32_t addr) {
    return reinterpret_cast<__attribute__((address_space(3))) T *>(static_cast<uintptr_t>(addr));
}

template <typename T>
__device__ __forceinline__ v8f load8(const T *p, int n, bool vec) {
    if (vec) {
        return __builtin_convertvector(*reinterpret_cast<const vec8<T> *>(p), v8f);
    }
    v8f v{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        if (e < n) {
            v[e] = static_cast<float>(p[e]);
        }
    }
    return v;
}

template <typename T, typename V>
__device__ __forceinline__ void store8(T *p, const V &v, int n, bool vec) {
    if (vec) {
        *reinterpret_cast<vec8<T> *>(p) = __builtin_convertvector(v, vec8<T>);
        return;
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
        if (e < n) {
            p[e] = static_cast<T>(v[e]);
        }
    }
}

template <typename OutT, bool ACC>
__global__ void __launch_bounds__(REDUCE_THREADS)
splitk_reduce_kernel(const float *__restrict__ ws, OutT *__restrict__ D, float alpha, float beta,
                     size_t mn, int splits) {
    for (size_t i = (size_t(blockIdx.x) * REDUCE_THREADS + threadIdx.x) * 8; i < mn;
         i += size_t(gridDim.x) * REDUCE_THREADS * 8) {
        v8f v = *reinterpret_cast<const v8f *>(ws + i);
        for (int z = 1; z < splits; z++) {
            v += *reinterpret_cast<const v8f *>(ws + z * mn + i);
        }
        v *= alpha;
        if constexpr (ACC) {
            v += beta * __builtin_convertvector(*reinterpret_cast<const vec8<OutT> *>(D + i), v8f);
        }
        *reinterpret_cast<vec8<OutT> *>(D + i) = __builtin_convertvector(v, vec8<OutT>);
    }
}

}  // namespace te_kittens
