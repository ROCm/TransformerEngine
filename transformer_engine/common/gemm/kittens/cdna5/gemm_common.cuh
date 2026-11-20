/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Shared workgroup mapping, epilogue and host planning of the gfx1250 GEMM kernels.

#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include "kittens.cuh"
#include "../kittens_common.h"
#include "../kittens_kernel_common.cuh"

namespace te_kittens::cdna5 {

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    GELU_AUX_BIAS,
    DGELU,
    BGRADB,
    SPLIT_K,
};

constexpr int CLUSTER_DIM       = 4;
constexpr int REDUCE_MAX_BLOCKS = 65536;
constexpr int MAX_SPLITS        = 8;
constexpr int WS_ALIGN          = 256;
constexpr int NUM_XCDS          = 8;
constexpr int XCD_CHUNK         = 32;
constexpr int WGM               = 8;
constexpr int STAGGER_U         = 64;
constexpr int STAGGER_STEP      = 2;

// 128x128 runs 4 waves of 64x64 and two workgroups per CU; the other tiles run 8 waves, one workgroup per CU.
constexpr int wg_threads(int bm, int bn) { return bm * bn <= 128 * 128 ? 128 : 256; }

typedef int v4i __attribute__((ext_vector_type(4)));

#ifdef __gfx1250_strict__
constexpr bool ARCH_MCAST = false;
#else
constexpr bool ARCH_MCAST = true;
#endif

// A function rather than inline at the call site: the kernels' codegen depends on it.
__device__ __forceinline__ int warpid_uniform() { return __builtin_amdgcn_readfirstlane(threadIdx.x >> 5); }

struct TileCoord {
    int m;
    int n;
    int z;
};

// XCD remap of the dispatch id (runs of XCD_CHUNK ids per XCD share its L2), split-major, then a raster grouped by
// WGM tile columns. A bijection of the grid.
__device__ __forceinline__ TileCoord workgroup_tile() {
    constexpr int BLOCK = NUM_XCDS * XCD_CHUNK;
    const int wg    = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int gx    = gridDim.x;
    const int gy    = gridDim.y;
    const int gz    = gridDim.z;
    const int plane = gx * gy;
    const int nwg   = plane * gz;
    const int full  = nwg / BLOCK * BLOCK;
    const int xcd   = wg % NUM_XCDS;
    int id;
    if (wg < full) {
        id = wg / BLOCK * BLOCK + xcd * XCD_CHUNK + wg / NUM_XCDS % XCD_CHUNK;
    } else {
        const int rest = nwg - full;
        const int lo   = rest % NUM_XCDS;
        id = full + xcd * (rest / NUM_XCDS) + (xcd < lo ? xcd : lo) + (wg - full) / NUM_XCDS;
    }
    const int p     = id % plane;
    const int group = WGM * gx;
    const int n0    = p / group * WGM;
    const int r     = p % group;
    const int w     = gy - n0 < WGM ? gy - n0 : WGM;
    return TileCoord{r / w, n0 + r % w, id / plane};
}

// StaggerU: physical K block of logical block it >= 1 (block 0, the partial head, never moves).
__device__ __forceinline__ int stagger_k(int it, int shift, int k_iters) {
    const int p = it + shift;
    return p < k_iters ? p : p - (k_iters - 1);
}

template <int BLOCK_ROW, int BLOCK_COL, typename StT, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT,
          typename AuxT, bool PLAIN_ALPHA, bool EDGE>
__device__ __forceinline__ void gemm_epilogue(uint32_t lds0, OutT *__restrict__ d, const AuxT *__restrict__ bias,
                                              AuxT *__restrict__ aux, float alpha, float beta, int M, int N,
                                              int m_tile, int n_tile) {
    constexpr bool PLAIN    = !std::is_same_v<StT, float>;
    constexpr bool HAS_BIAS = EPILOGUE == GemmEpilogue::BIAS || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr bool HAS_GELU = EPILOGUE == GemmEpilogue::GELU_AUX || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr int STRIDE      = BLOCK_ROW * sizeof(StT) + 16;
    constexpr int ROW_CHUNKS  = BLOCK_ROW / 8;
    constexpr int UNROLL      = HAS_GELU || EPILOGUE == GemmEpilogue::DGELU || EDGE ? 1 : 4;
    constexpr int NUM_THREADS = wg_threads(BLOCK_ROW, BLOCK_COL);
    const int m8 = threadIdx.x % ROW_CHUNKS;
    const size_t tile = size_t(n_tile) * BLOCK_COL * M + size_t(m_tile) * BLOCK_ROW + m8 * 8;
    const int rows = EDGE ? std::clamp(M - m_tile * BLOCK_ROW - m8 * 8, 0, 8) : 8;
    const int cols = EDGE ? std::min(N - n_tile * BLOCK_COL, BLOCK_COL) : BLOCK_COL;
    const bool vec = !EDGE || (rows == 8 && M % 8 == 0);
    v8f b8{};
    if constexpr (HAS_BIAS) {
        b8 = load8(bias + m_tile * BLOCK_ROW + m8 * 8, rows, vec);
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
            if constexpr (ACCUMULATE) {
                v += beta * load8(d + off, rows, vec);
            }
            if constexpr (HAS_BIAS) {
                v += b8;
            }
            if constexpr (HAS_GELU) {
                store8(aux + off, v, rows, vec);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    v[e] = kittens::base_ops::gelu::op<float>(v[e]);
                }
            }
            if constexpr (EPILOGUE == GemmEpilogue::DGELU) {
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

// All partials of an element are loaded before the first add; fixed z order keeps the sum deterministic.
template <typename OutT, bool ACC>
__global__ void __launch_bounds__(REDUCE_THREADS)
splitk_reduce_kernel(const float *__restrict__ ws, OutT *__restrict__ D, float alpha, float beta, size_t mn,
                     int splits) {
    for (size_t i = (size_t(blockIdx.x) * REDUCE_THREADS + threadIdx.x) * 8; i < mn;
         i += size_t(gridDim.x) * REDUCE_THREADS * 8) {
        v8f p[MAX_SPLITS];
#pragma unroll
        for (int z = 0; z < MAX_SPLITS; z++) {
            if (z < splits) {
                p[z] = *reinterpret_cast<const v8f *>(ws + z * mn + i);
            }
        }
        v8f v = p[0];
#pragma unroll
        for (int z = 1; z < MAX_SPLITS; z++) {
            if (z < splits) {
                v += p[z];
            }
        }
        v *= alpha;
        if constexpr (ACC) {
            v += beta * __builtin_convertvector(*reinterpret_cast<const vec8<OutT> *>(D + i), v8f);
        }
        *reinterpret_cast<vec8<OutT> *>(D + i) = __builtin_convertvector(v, vec8<OutT>);
    }
}

template <typename OutT, typename Args>
static void launch_reduce(const Args &args, const void *workspace, int splits) {
    const size_t mn     = size_t(args.M) * args.N;
    const size_t blocks = std::min<size_t>((mn / 8 + REDUCE_THREADS - 1) / REDUCE_THREADS, REDUCE_MAX_BLOCKS);
    const auto kernel = args.beta != 0.f ? splitk_reduce_kernel<OutT, true> : splitk_reduce_kernel<OutT, false>;
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(REDUCE_THREADS), 0, args.stream,
                       static_cast<const float *>(workspace), static_cast<OutT *>(args.C), args.alpha, args.beta,
                       mn, splits);
}

struct TileModel {
    double e_rect;
    double e_lone;
    double c0;
    double a;
    double b;
};

struct GemmPlan {
    GemmEpilogue epilogue;
    int splits;
    int bm;
    int bn;
    bool mcast;
    void *workspace;
};

template <typename Args>
static bool plan_gemm(const Args &args, const TileModel &model, int block_k, GemmPlan *plan) {
    const bool bias = args.bias != nullptr;
    const bool aux  = args.aux_gelu != nullptr;
    if ((bias && aux && args.bias_dtype != args.aux_dtype) || (args.grad && bias && aux)) {
        return false;
    }
    GemmEpilogue epilogue = GemmEpilogue::DEFAULT;
    if (args.grad) {
        epilogue = bias ? GemmEpilogue::BGRADB : aux ? GemmEpilogue::DGELU : GemmEpilogue::DEFAULT;
    } else if (bias || aux) {
        epilogue = !aux ? GemmEpilogue::BIAS : bias ? GemmEpilogue::GELU_AUX_BIAS : GemmEpilogue::GELU_AUX;
    }
    if (args.transa && epilogue == GemmEpilogue::DEFAULT && args.beta == 0.f && args.out_dtype != KITTENS_FLOAT32 &&
        args.alpha != 1.f) {
        return false;
    }

    const uintptr_t ws_raw = reinterpret_cast<uintptr_t>(args.workspace);
    const uintptr_t ws     = (ws_raw + WS_ALIGN - 1) & ~uintptr_t(WS_ALIGN - 1);
    const size_t pad         = ws - ws_raw;
    const size_t ws_size     = args.workspace_size > pad ? args.workspace_size - pad : 0;
    const size_t split_bytes = size_t(args.M) * args.N * sizeof(float);
    plan->epilogue = epilogue;
    plan->splits   = 1;
    plan->bm       = 256;
    plan->bn       = 256;
    // DEFAULT: NVTE_HK_TILE / NVTE_HK_SPLITK force the tile / split (an invalid split falls back to 1), else the
    // tile x split minimizing busiest-CU runs x per-tile time plus the split-K reduction (model).
    if (epilogue == GemmEpilogue::DEFAULT) {
        const int max_splits = args.M % 8 == 0 ? int(std::min<size_t>(ws_size / split_bytes, MAX_SPLITS)) : 1;
        const int cu         = std::max(args.cu_count, 1);
        if (args.splitk != 0) {
            plan->splits = args.splitk <= args.K / block_k && args.splitk <= max_splits ? args.splitk : 1;
        }
        if (args.tile_m != 0) {
            plan->bm = args.tile_m;
            plan->bn = args.tile_n;
        }
        if (args.tile_m == 0 && args.splitk == 0) {
            const int tm[4] = {256, 128, 256, 128};
            const int tn[4] = {256, 256, 128, 128};
            const int ts[6] = {1, 2, 3, 4, 6, 8};
            const int nb    = kittens::ceil_div(args.K, block_k);
            double best = 0;
            for (int t = 0; t < 4; t++) {
                for (int s : ts) {
                    if (s > 1 && (s > args.K / block_k || s > max_splits)) {
                        continue;
                    }
                    const int64_t wg =
                        int64_t(kittens::ceil_div(args.M, tm[t])) * kittens::ceil_div(args.N, tn[t]) * s;
                    const int64_t busy = (wg + cu - 1) / cu;
                    double runs = t == 0 ? double(busy) : double(busy) / model.e_rect;
                    if (t == 3) {
                        runs = double(busy - busy % 2) / model.e_rect + double(busy % 2) / model.e_lone;
                    }
                    double time = runs * tm[t] * tn[t] / 65536.0 * (kittens::ceil_div(nb, s) + model.c0);
                    if (s > 1) {
                        time += model.a * s * double(args.M) * args.N / cu / 65536.0 + model.b;
                    }
                    if (best == 0 || time < best) {
                        best         = time;
                        plan->bm     = tm[t];
                        plan->bn     = tn[t];
                        plan->splits = s;
                    }
                }
            }
        }
    }
    plan->workspace = reinterpret_cast<void *>(ws);
#ifdef NVTE_HK_GFX1250_MCAST
    plan->mcast = NVTE_HK_GFX1250_MCAST != 0;
#else
    plan->mcast = transformer_engine::cuda::sm_arch_name().find("-strict") == std::string::npos;
#endif
    return true;
}

}  // namespace te_kittens::cdna5
