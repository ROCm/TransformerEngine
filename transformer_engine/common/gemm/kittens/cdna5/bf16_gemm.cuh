/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// gfx1250 BF16/FP16 GEMM, column-major D (M x N), 256x256 tiles, 4x4 clusters with multicast or unclustered.
// Any M, N, K: the TDM zero-fills past D's edge and edge tiles store masked.

#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "kittens.cuh"
#include "gemm.h"

namespace te_kittens::cdna5 {
namespace bf16_gemm {

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
constexpr int WARPS_N     = 2;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int S           = 2;
constexpr int CLUSTER_DIM = 4;
constexpr int A_FRAGS     = WARP_M / 16;
constexpr int B_FRAGS     = WARP_N / 16;
constexpr int DS_SUBSTEP  = 2 * (A_FRAGS + B_FRAGS);      // ds_load_b128 per K step
constexpr int WMMA_STEP   = A_FRAGS * B_FRAGS;

// Interleave groups of PIN_WMMA matrix ops and PIN_DS reads, PIN_REP per K step.
constexpr int PIN_WMMA = 8;
constexpr int PIN_REP  = WMMA_STEP / PIN_WMMA;
constexpr int PIN_DS   = DS_SUBSTEP / PIN_REP;
// Matrix ops of the last K step issued before the stage flip; they read registers loaded earlier.
constexpr int PRE_BAR  = 12;

static_assert(K_STEP * 4 == BLOCK_K && WMMA_STEP == 32 && A_FRAGS == 4 && B_FRAGS == 8);
static_assert((WMMA_STEP - PRE_BAR) % PIN_REP == 0);

typedef int   v4i __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));

template <typename T, int R, int C>
using st_e = kittens::st<T, R, C, kittens::ducks::st_shape::st_16x32_padded<>>;
template <typename T, int R, int C>
using rt_e = kittens::rt<T, R, C, kittens::ducks::rt_layout::row,
                         kittens::ducks::rt_shape::rt_16x32>;
template <typename T>
using gl_e  = kittens::gl<T, -1, -1, -1, -1>;
using C_acc = kittens::rt_fl<WARP_M, WARP_N, kittens::col_l, kittens::rt_16x16_s>;

// The clang element type of an input type (HK's bf16 and half are structs).
template <typename T>
using native_t = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;

template <int N, typename F>
__device__ __forceinline__ void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

template <typename T>
__device__ __forceinline__ __attribute__((address_space(3))) T *lds_ptr(uint32_t addr) {
    return reinterpret_cast<__attribute__((address_space(3))) T *>(static_cast<uintptr_t>(addr));
}

// An M/N-contiguous operand's stage keeps its global orientation, K-major.
template <typename T, Layout L>
struct Tiles {
    static constexpr bool TRANS_A = L != TN;
    static constexpr bool TRANS_B = L == NT;
    using A_deep = std::conditional_t<TRANS_A, st_e<T, BLOCK_K, BLOCK_M>,
                                      st_e<T, BLOCK_M, BLOCK_K>>;
    using B_deep = std::conditional_t<TRANS_B, st_e<T, BLOCK_K, BLOCK_N>,
                                      st_e<T, BLOCK_N, BLOCK_K>>;
    using A_reg  = rt_e<T, WARP_M, K_STEP>;
    using B_reg  = rt_e<T, WARP_N, K_STEP>;

    static constexpr int LDS_BYTES = S * (sizeof(A_deep) + sizeof(B_deep));

    static_assert(A_deep::shape::pad_interval % WARP_M == 0 &&
                      B_deep::shape::pad_interval % WARP_N == 0,
                  "a warp's operand rows must not straddle a pad interval of a K-major stage");
    static_assert(BLOCK_N * (BLOCK_M * 4 + 16) <= LDS_BYTES, "fp32 C staging fits the operand LDS");
    static_assert(LDS_BYTES <= 327680, "LDS is 320 KiB per workgroup");
};

/* Issue order of ND reads then NW matrix ops, PIN_REP times (SchedGroup masks: DS read 1<<8, matrix
 * 1<<3). Kept in a function: written inline four times it spilled. */
template <int NW = PIN_WMMA, int ND = PIN_DS>
__device__ __forceinline__ void pin_interleave() {
#pragma unroll
    for (int i = 0; i < PIN_REP; i++) {
        __builtin_amdgcn_sched_group_barrier(0x100, ND, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NW, 0);
    }
}

// K-major stage [BLOCK_K][C]: (k, c) is at k * tr_row_elems + c + c / pad_interval * pad_amount.
template <typename ST>
constexpr int tr_row_elems = ST::cols + ST::cols / ST::shape::pad_interval * ST::shape::pad_amount;

// Byte offset of a lane's first ds_load_tr16_b128 for a warp whose operand rows start at w0;
// lane half h gets K 16h + 0..7 of row L % 16, the order the K-contiguous load produces.
template <typename ST>
__device__ __forceinline__ uint32_t tr_lane_offset(int w0) {
    using Shape = typename ST::shape;
    const int L   = kittens::laneid();
    const int g   = L / 8;
    const int col = w0 + 8 * (g % 2);
    return 2 * ((L % 8 + 16 * (g / 2)) * tr_row_elems<ST> + col +
                col / Shape::pad_interval * Shape::pad_amount);
}

// Sub-step KS of a warp's fragments from a K-major stage: two ds_load_tr16_b128 per 16x32 fragment.
template <int KS, typename RT, typename ST>
__device__ __forceinline__ void load_tr(RT &dst, const ST &src, uint32_t lane_off) {
    static_assert(std::is_same_v<typename RT::shape, kittens::ducks::rt_shape::rt_16x32> &&
                  RT::cols == K_STEP);
    static_assert(ST::rows == BLOCK_K && ST::cols % ST::shape::pad_interval == 0);
    using v8s      = short __attribute__((ext_vector_type(8)));
    using lds_v8s  = v8s __attribute__((address_space(3)));
    using packed_t = std::remove_cvref_t<decltype(dst.tiles[0][0].data[0])>;
    constexpr int ROW_B = 2 * tr_row_elems<ST>;

    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data)) + lane_off;
#pragma unroll
    for (int ti = 0; ti < RT::height; ti++) {
#pragma unroll
        for (int x = 0; x < 2; x++) {
            const uint32_t addr = base + (KS * K_STEP + 8 * x) * ROW_B + ti * 16 * 2;
            v8s v = __builtin_amdgcn_ds_load_tr16_b128_v8i16(
                reinterpret_cast<lds_v8s *>(static_cast<uintptr_t>(addr)));
            const packed_t *p = reinterpret_cast<const packed_t *>(&v);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                dst.tiles[ti][0].data[4 * x + i] = p[i];
            }
        }
    }
}

// Sub-step KS of a warp's operand from stage st: transposing reads for a K-major stage.
template <bool TRANS, int KS, typename RT, typename ST>
__device__ __forceinline__ void load_step(RT &dst, const ST &st, uint32_t warp_off) {
    if constexpr (TRANS) {
        load_tr<KS>(dst, st, warp_off);
    } else {
        kittens::load(dst, st, int(warp_off) + KS * K_STEP);
    }
}

/* Matrix ops T0..T1-1 of a K step in zigzag order: op t uses A fragment t / 8 and B fragment t % 8,
 * reversed on odd rows, and reuses A except at the start of a row. */
template <int T0, int T1, typename RA, typename RB>
__device__ __forceinline__ void mma_zigzag(C_acc &c, const RA &a, const RB &b) {
    static_for<T1 - T0>([&](auto x) {
        constexpr int t = T0 + decltype(x)::value, i = t / B_FRAGS;
        constexpr int j = i % 2 ? B_FRAGS - 1 - t % B_FRAGS : t % B_FRAGS;
        kittens::mma_ABt_base<t % B_FRAGS != 0, false>(c.tiles[i][j], a.tiles[i][0], b.tiles[j][0],
                                                       c.tiles[i][j]);
    });
}

// Adds the K values this lane holds of each B fragment row (one dbias column per row) to s.
template <typename RT>
__device__ __forceinline__ void colsum(float *s, const RT &b) {
    using base = std::remove_cvref_t<decltype(b.tiles[0][0])>;
    static_for<RT::height>([&](auto t) {
        static_for<base::packed_per_thread>([&](auto d) {
            const float2 f = kittens::base_types::convertor<float2, typename base::dtype>::convert(
                b.tiles[t][0].data[d]);
            s[t] += f.x + f.y;
        });
    });
}

// C is staged in LDS as D itself for a D of the input type without epilogue, else as fp32.
template <typename T, Epi E, bool ACC, typename OutT>
using staged_t = std::conditional_t<(E == NONE || E == BGRADB) && !ACC &&
                                        std::is_same_v<OutT, native_t<T>>,
                                    OutT, float>;

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
__device__ __forceinline__ void copy_out(uint32_t lds0, OutT *__restrict__ d,
                                         const SideT *__restrict__ bias, SideT *__restrict__ aux,
                                         float alpha, float beta, int M, int N, int tile_m,
                                         int tile_n) {
    constexpr bool PLAIN = !std::is_same_v<StT, float>;
    constexpr int SSTR = BLOCK_M * sizeof(StT) + 16;
    constexpr int CH   = BLOCK_M / 8;
    constexpr bool ADD_BIAS = E == BIAS || E == GELU_AUX_BIAS;
    constexpr bool GELU     = E == GELU_AUX || E == GELU_AUX_BIAS;
    constexpr int UNROLL = GELU || E == DGELU || EDGE ? 1 : 4;
    static_assert(NUM_THREADS % CH == 0, "a thread's rows are the same in every column");
    const int m8 = threadIdx.x % CH;
    const size_t tile = size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M + m8 * 8;
    const int rows = EDGE ? std::clamp(M - tile_m * BLOCK_M - m8 * 8, 0, 8) : 8;
    const int cols = EDGE ? std::min(N - tile_n * BLOCK_N, BLOCK_N) : BLOCK_N;
    const bool vec = !EDGE || (rows == 8 && M % 8 == 0);
    v8f b8{};
    if constexpr (ADD_BIAS) {
        b8 = load8(bias + tile_m * BLOCK_M + m8 * 8, rows, vec);
    }
#pragma unroll UNROLL
    for (int col = threadIdx.x / CH; col < cols; col += NUM_THREADS / CH) {
        const size_t off = tile + size_t(col) * M;
        const uint32_t src = lds0 + col * SSTR + m8 * 8 * sizeof(StT);
        if constexpr (PLAIN) {
            if (vec && (!PLAIN_ALPHA || alpha == 1.f)) {
                *reinterpret_cast<v4i *>(d + off) = *lds_ptr<v4i>(src);
            } else if constexpr (PLAIN_ALPHA) {
                store8(d + off, alpha * __builtin_convertvector(*lds_ptr<vec8<StT>>(src), v8f),
                       rows, vec);
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

// SPLIT: workgroup z writes its fp32 partial to D + z M N; BGRADB: tile_m = 0 also writes bias[n] = sum_k B(n, k).
// Both kernels include one body: a shared __device__ function changed the cluster kernel's K-loop schedule.
template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __cluster_dims__(CLUSTER_DIM, CLUSTER_DIM, 1) __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel(const gl_e<T> ga, const gl_e<T> gb, OutT *__restrict__ D,
                      SideT *__restrict__ bias, SideT *__restrict__ aux, float alpha, float beta,
                      int M, int N, int K) {
    constexpr bool MC = ARCH_MCAST;
#include "bf16_gemm_body.inc"
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel_nomc(const gl_e<T> ga, const gl_e<T> gb, OutT *__restrict__ D,
                           SideT *__restrict__ bias, SideT *__restrict__ aux, float alpha,
                           float beta, int M, int N, int K) {
    constexpr bool MC = false;
#include "bf16_gemm_body.inc"
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
void launch_kernel(const Launch &l, void *d, int splits) {
    using Tl = Tiles<T, L>;
    const auto kernel = l.mcast ? bf16_gemm_kernel<L, T, E, ACC, OutT, SideT>
                                : bf16_gemm_kernel_nomc<L, T, E, ACC, OutT, SideT>;
    kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, Tl::LDS_BYTES),
                  "gfx1250 bf16_gemm_kernel LDS");
    T *A = static_cast<T *>(const_cast<void *>(l.A));
    T *B = static_cast<T *>(const_cast<void *>(l.B));
    const gl_e<T> ga(A, 1, 1, Tl::TRANS_A ? l.K : l.M, Tl::TRANS_A ? l.M : l.K);
    const gl_e<T> gb(B, 1, 1, Tl::TRANS_B ? l.K : l.N, Tl::TRANS_B ? l.N : l.K);
    const int side = l.mcast ? CLUSTER_DIM : 1;
    auto clusters = [&](int n, int tile) {
        return ((n + tile - 1) / tile + side - 1) / side * side;
    };
    const dim3 grid(clusters(l.M, BLOCK_M), clusters(l.N, BLOCK_N), splits);
    hipLaunchKernelGGL(kernel, grid, dim3(NUM_THREADS), Tl::LDS_BYTES, l.stream, ga, gb,
                       static_cast<OutT *>(d), static_cast<SideT *>(const_cast<void *>(l.bias)),
                       static_cast<SideT *>(l.aux), l.alpha, l.beta, l.M, l.N, l.K);
    kittens_check(hipGetLastError(), "gfx1250 bf16_gemm_kernel");
}

// Instantiated epilogue variants of layout L, defined in bf16_<layout>.cpp; In: the input type.
struct In {};
template <Layout L>
struct Instances;

}  // namespace bf16_gemm

template <Layout L>
bool launch_bf16(const Launch &l) {
    using namespace bf16_gemm;
    auto run = [&](auto t) {
        using T = typename decltype(t)::type;
        auto resolve = [](auto x) {
            using X = typename decltype(x)::type;
            return std::type_identity<std::conditional_t<std::is_same_v<X, In>, native_t<T>, X>>{};
        };
        return any_of(typename Instances<L>::variants{}, [&](auto v) {
            using V     = decltype(v);
            using OutT  = typename decltype(resolve(std::type_identity<typename V::OutT>{}))::type;
            using SideT = typename decltype(resolve(std::type_identity<typename V::SideT>{}))::type;
            if (l.epi != V::E || l.acc != V::ACC || l.out_dtype != dtype_of<OutT> ||
                l.side_dtype != dtype_of<SideT>) {
                return false;
            }
            if constexpr (V::E == NONE) {
                if (l.splits > 1) {
                    launch_kernel<L, T, SPLIT, false, float, float>(l, l.workspace, l.splits);
                    splitk_reduce(l);
                    return true;
                }
            }
            launch_kernel<L, T, V::E, V::ACC, OutT, SideT>(l, l.D, 1);
            return true;
        });
    };
    return l.a_dtype == KITTENS_FLOAT16 ? run(std::type_identity<kittens::half>{})
                                        : run(std::type_identity<kittens::bf16>{});
}

}  // namespace te_kittens::cdna5
