/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kittens.cuh"
#include "gemm.h"

namespace te_kittens::cdna5 {

template <typename T>
__device__ __forceinline__ __attribute__((address_space(3))) T *lds_ptr(uint32_t addr) {
    return reinterpret_cast<__attribute__((address_space(3))) T *>(static_cast<uintptr_t>(addr));
}

// Bytes per column of the C tile staged column-major in LDS.
constexpr int staged_stride(int elem_bytes) { return BLOCK_M * elem_bytes + 16; }

// Cluster bits (x fastest) of the workgroups in one column of a cx x cy cluster.
constexpr uint32_t cluster_col_bits(int cx, int cy) {
    uint32_t bits = 0;
    for (int y = 0; y < cy; y++) {
        bits |= 1u << (y * cx);
    }
    return bits;
}

// The first n of the 8 elements at p, one vector access when vec (then n is 8).
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

// Copy-out of the C tile staged in LDS: D = alpha * C (+ beta * D) (+ bias), then GELU (pre-GELU to aux) or
// times dgelu(aux). EDGE stores only D's rows and columns, element-wise where 8 rows do not fit.
template <typename StT, Epi E, bool ACC, typename OutT, typename SideT, bool PLAIN_ALPHA, bool EDGE>
__device__ __forceinline__ void copy_out(uint32_t lds0, OutT *__restrict__ d, const SideT *__restrict__ bias,
                                         SideT *__restrict__ aux, float alpha, float beta, int M, int N, int tile_m,
                                         int tile_n) {
    constexpr bool PLAIN    = !std::is_same_v<StT, float>;
    constexpr bool ADD_BIAS = E == BIAS || E == GELU_AUX_BIAS;
    constexpr bool GELU     = E == GELU_AUX || E == GELU_AUX_BIAS;
    constexpr int STRIDE     = staged_stride(sizeof(StT));
    constexpr int ROW_CHUNKS = BLOCK_M / 8;
    constexpr int UNROLL     = GELU || E == DGELU || EDGE ? 1 : 4;
    static_assert(NUM_THREADS % ROW_CHUNKS == 0, "a thread's rows are the same in every column");
    const int m8 = threadIdx.x % ROW_CHUNKS;
    const size_t tile = size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M + m8 * 8;
    const int rows = EDGE ? std::clamp(M - tile_m * BLOCK_M - m8 * 8, 0, 8) : 8;
    const int cols = EDGE ? std::min(N - tile_n * BLOCK_N, BLOCK_N) : BLOCK_N;
    const bool vec = !EDGE || (rows == 8 && M % 8 == 0);
    v8f b8{};
    if constexpr (ADD_BIAS) {
        b8 = load8(bias + tile_m * BLOCK_M + m8 * 8, rows, vec);
    }
#pragma unroll UNROLL
    for (int col = threadIdx.x / ROW_CHUNKS; col < cols; col += NUM_THREADS / ROW_CHUNKS) {
        const size_t off = tile + size_t(col) * M;
        const uint32_t src = lds0 + col * STRIDE + m8 * 8 * sizeof(StT);
        if constexpr (PLAIN) {
            if (vec && (!PLAIN_ALPHA || alpha == 1.f)) {
                *reinterpret_cast<v4i *>(d + off) = *lds_ptr<v4i>(src);
            } else if constexpr (PLAIN_ALPHA) {
                store8(d + off, alpha * __builtin_convertvector(*lds_ptr<vec8<StT>>(src), v8f), rows, vec);
            } else {
                store8(d + off, *lds_ptr<vec8<StT>>(src), rows, false);
            }
        } else {
            v8f v = alpha * *lds_ptr<v8f>(src);
            if constexpr (ACC) {
                v += beta * load8(d + off, rows, vec);
            }
            if constexpr (ADD_BIAS) {
                v += b8;
            }
            if constexpr (GELU) {
                store8(aux + off, v, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] = kittens::base_ops::gelu::op<float>(v[e]);
                }
            }
            if constexpr (E == DGELU) {
                const v8f g = load8(aux + off, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] *= kittens::base_ops::dgelu::op<float>(g[e]);
                }
            }
            store8(d + off, v, rows, vec);
        }
    }
}

}  // namespace te_kittens::cdna5
