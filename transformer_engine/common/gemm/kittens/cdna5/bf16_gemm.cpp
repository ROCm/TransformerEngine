/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include "kittens.cuh"
#include "../kittens_common.h"
#include "../kittens_kernel_common.cuh"

namespace te_kittens::cdna5::bf16 {

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    GELU_AUX_BIAS,
    DGELU,
    BGRADB,
    SPLIT_K,
};

constexpr int BLOCK_ROW         = 256;
constexpr int BLOCK_COL         = 256;
constexpr int BLOCK_K           = 128;
constexpr int NUM_THREADS       = 256;
constexpr int K_STEP            = 32;
constexpr int WARPS_ROW         = 4;
constexpr int WARPS_COL         = 2;
constexpr int REG_M             = BLOCK_ROW / WARPS_ROW;
constexpr int REG_N             = BLOCK_COL / WARPS_COL;
constexpr int CLUSTER_DIM       = 4;
constexpr int SCHED_DS_READ     = 0x100;
constexpr int SCHED_WMMA        = 0x008;
constexpr int LDS_BYTES         = 4 * sizeof(kittens::st_bf<BLOCK_K, BLOCK_ROW>);
constexpr int REDUCE_MAX_BLOCKS = 65536;
constexpr int WS_ALIGN          = 256;

static_assert(BLOCK_COL * (BLOCK_ROW * sizeof(float) + 16) <= LDS_BYTES && LDS_BYTES <= 320 * 1024);

typedef int v4i __attribute__((ext_vector_type(4)));

#ifdef __gfx1250_strict__
constexpr bool ARCH_MCAST = false;
#else
constexpr bool ARCH_MCAST = true;
#endif

template <typename StT, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT, typename AuxT, bool PLAIN_ALPHA,
          bool EDGE>
__device__ __forceinline__ void gemm_epilogue(uint32_t lds0, OutT *__restrict__ d, const AuxT *__restrict__ bias,
                                              AuxT *__restrict__ aux, float alpha, float beta, int M, int N,
                                              int m_tile, int n_tile) {
    constexpr bool PLAIN    = !std::is_same_v<StT, float>;
    constexpr bool HAS_BIAS = EPILOGUE == GemmEpilogue::BIAS || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr bool HAS_GELU = EPILOGUE == GemmEpilogue::GELU_AUX || EPILOGUE == GemmEpilogue::GELU_AUX_BIAS;
    constexpr int STRIDE     = BLOCK_ROW * sizeof(StT) + 16;
    constexpr int ROW_CHUNKS = BLOCK_ROW / 8;
    constexpr int UNROLL     = HAS_GELU || EPILOGUE == GemmEpilogue::DGELU || EDGE ? 1 : 4;
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

template <bool TRANS, int KS, typename RT, typename ST>
__device__ __forceinline__ void load_k_step(RT &dst, const ST &src, uint32_t off) {
    if constexpr (TRANS) {
        typedef short v8s __attribute__((ext_vector_type(8)));
        const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data)) + off;
#pragma unroll
        for (int ti = 0; ti < RT::height; ti++) {
#pragma unroll
            for (int x = 0; x < 2; x++) {
                *reinterpret_cast<v8s *>(&dst.tiles[ti][0].data[4 * x]) = __builtin_amdgcn_ds_load_tr16_b128_v8i16(lds_ptr<v8s>(base + 2 * ST::idx(KS * K_STEP + 8 * x, 16 * ti)));
            }
        }
    } else {
        kittens::load(dst, src, int(off) + KS * K_STEP);
    }
}

template <bool TRANSA, bool TRANSB, bool MC, typename T, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT,
          typename AuxT>
__global__ __cluster_dims__(MC ? CLUSTER_DIM : 0, MC ? CLUSTER_DIM : 0, MC ?: 0) __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel(const kittens::gl<T, -1, -1, -1, -1> ga, const kittens::gl<T, -1, -1, -1, -1> gb,
                      OutT *__restrict__ D, AuxT *__restrict__ bias, AuxT *__restrict__ aux, float alpha, float beta,
                      int M, int N, int K) {
    using ST_A = kittens::st<T, TRANSA ? BLOCK_ROW : BLOCK_K, TRANSA ? BLOCK_K : BLOCK_ROW,
                             kittens::ducks::st_shape::st_16x32_padded<>>;
    using ST_B = kittens::st<T, TRANSB ? BLOCK_K : BLOCK_COL, TRANSB ? BLOCK_COL : BLOCK_K,
                             kittens::ducks::st_shape::st_16x32_padded<>>;
    using RT_A = kittens::rt<T, REG_M, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_B = kittens::rt<T, REG_N, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using IT   = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;
    using StT  = std::conditional_t<(EPILOGUE == GemmEpilogue::DEFAULT || EPILOGUE == GemmEpilogue::BGRADB) &&
                                        !ACCUMULATE && std::is_same_v<OutT, IT>, OutT, float>;
    typedef IT v16in __attribute__((ext_vector_type(16)));
    typedef float v16f __attribute__((ext_vector_type(16)));
    constexpr bool CLUSTER = MC && ARCH_MCAST;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al(reinterpret_cast<int *>(&__shm[0]));
    ST_A &as0 = al.allocate_in<kittens::segment<0>, ST_A>();
    ST_B &bs0 = al.allocate_in<kittens::segment<0>, ST_B>();
    ST_A &as1 = al.allocate_in<kittens::segment<0>, ST_A>();
    ST_B &bs1 = al.allocate_in<kittens::segment<0>, ST_B>();
    ST_A *As[2] = {&as0, &as1};
    ST_B *Bs[2] = {&bs0, &bs1};

    kittens::sched::lock_simd();

    RT_C c;
    kittens::zero(c);
    float bsum[EPILOGUE == GemmEpilogue::BGRADB ? RT_C::width : 1] = {};

    const int m_tile  = blockIdx.x;
    const int n_tile  = blockIdx.y;
    const int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    const int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    const int wid     = kittens::warpid();
    const int warp_m  = wid / WARPS_COL;
    const int warp_n  = wid % WARPS_COL;
    const int num_kb  = kittens::ceil_div(K, BLOCK_K);
    int kb0     = 0;
    int k_iters = num_kb;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        const int splits = gridDim.z;
        const int z      = blockIdx.z;
        kb0     = num_kb * z / splits;
        k_iters = num_kb * (z + 1) / splits - kb0;
    }
    if (k_iters <= 0) {
        return;
    }
    // A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (m_tile >= tiles_M || n_tile >= tiles_N) {
        return;
    }

    const int lane = kittens::laneid();
    const int g    = lane / 8;
    const uint32_t a_off = TRANSA ? warp_m * REG_M * BLOCK_K
                                  : 2 * ST_A::idx(lane % 8 + 16 * (g / 2), warp_m * REG_M + 8 * (g % 2));
    const uint32_t b_off = TRANSB ? 2 * ST_B::idx(lane % 8 + 16 * (g / 2), warp_n * REG_N + 8 * (g % 2))
                                  : warp_n * REG_N * BLOCK_K;

    uint32_t a_mask = 0;
    uint32_t b_mask = 0;
    if constexpr (CLUSTER) {
        const uint32_t cx   = blockIdx.x % CLUSTER_DIM;
        const uint32_t cy   = blockIdx.y % CLUSTER_DIM;
        const uint32_t in_x = min(tiles_M - int(blockIdx.x - cx), CLUSTER_DIM);
        const uint32_t in_y = min(tiles_N - int(blockIdx.y - cy), CLUSTER_DIM);
        a_mask = (0x1111u << cx) & ((1u << (CLUSTER_DIM * in_y)) - 1u);
        b_mask = ((1u << in_x) - 1u) << (CLUSTER_DIM * cy);
    }

    const int m_rem  = M - m_tile * BLOCK_ROW;
    const int n_rem  = N - n_tile * BLOCK_COL;
    const int k_head = K % BLOCK_K;
    const int back   = k_head ? BLOCK_K - k_head : 0;
    auto ga_k = ga;
    auto gb_k = gb;
    ga_k.raw_ptr -= size_t(back) * (TRANSA ? 1 : M);
    gb_k.raw_ptr -= size_t(back) * (TRANSB ? N : 1);
    const auto &ga0 = kb0 == 0 ? ga : ga_k;
    const auto &gb0 = kb0 == 0 ? gb : gb_k;
    const int kx0 = kb0 == 0 && k_head ? k_head : K;
    if (wid == 0) {
        kittens::tdm::load_async(*As[0], ga0, {0, 0, TRANSA ? m_tile : kb0, TRANSA ? kb0 : m_tile},
                                 TRANSA ? m_rem : kx0, TRANSA ? kx0 : m_rem, TRANSA ? K : M, a_mask, 1);
    }
    if (wid == 1) {
        kittens::tdm::load_async(*Bs[0], gb0, {0, 0, TRANSB ? kb0 : n_tile, TRANSB ? n_tile : kb0},
                                 TRANSB ? kx0 : n_rem, TRANSB ? n_rem : kx0, TRANSB ? N : K, b_mask, 1);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    if constexpr (CLUSTER) {
        kittens::cluster::sync();
    } else {
        kittens::sync::arrive();
        kittens::sync::wait();
    }
    kittens::sched::compiler_fence();

    RT_A a0, a1;
    RT_B b0, b1;
    load_k_step<!TRANSA, 0>(a0, *As[0], a_off);
    load_k_step<TRANSB, 0>(b0, *Bs[0], b_off);

    for (int kb = 0; kb < k_iters; kb++) {
        const int    cur = kb % 2;
        const int    nxt = (kb + 1) % 2;
        const int fill_k = kb0 + (kb + 1 < k_iters ? kb + 1 : 0);

        const uint32_t count = kb + 1 < k_iters ?: 0u;
        if (wid == 0) {
            kittens::tdm::load_async(*As[nxt], ga_k, {0, 0, TRANSA ? m_tile : fill_k, TRANSA ? fill_k : m_tile},
                                     TRANSA ? m_rem : K, TRANSA ? K : m_rem, TRANSA ? K : M, a_mask, count);
        }
        if (wid == 1) {
            kittens::tdm::load_async(*Bs[nxt], gb_k, {0, 0, TRANSB ? fill_k : n_tile, TRANSB ? n_tile : fill_k},
                                     TRANSB ? K : n_rem, TRANSB ? n_rem : K, TRANSB ? N : K, b_mask, count);
        }
        kittens::sched::compiler_fence();

        load_k_step<!TRANSA, 1>(a1, *As[cur], a_off);
        load_k_step<TRANSB, 1>(b1, *Bs[cur], b_off);
        kittens::mma_ABt(c, a0, b0, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (EPILOGUE == GemmEpilogue::BGRADB) {
            if (m_tile == 0 && warp_m == 0) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b0.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        load_k_step<!TRANSA, 2>(a0, *As[cur], a_off);
        load_k_step<TRANSB, 2>(b0, *Bs[cur], b_off);
        kittens::mma_ABt(c, a1, b1, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (EPILOGUE == GemmEpilogue::BGRADB) {
            if (m_tile == 0 && warp_m == 1) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b1.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        load_k_step<!TRANSA, 3>(a1, *As[cur], a_off);
        load_k_step<TRANSB, 3>(b1, *Bs[cur], b_off);
        kittens::mma_ABt(c, a0, b0, c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 8, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (EPILOGUE == GemmEpilogue::BGRADB) {
            if (m_tile == 0 && warp_m == 2) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b0.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }

        kittens::mma_ABt_base<false,false>(c.tiles[0][0], a1.tiles[0][0], b1.tiles[0][0], c.tiles[0][0]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][1], a1.tiles[0][0], b1.tiles[1][0], c.tiles[0][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][2], a1.tiles[0][0], b1.tiles[2][0], c.tiles[0][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][3], a1.tiles[0][0], b1.tiles[3][0], c.tiles[0][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][4], a1.tiles[0][0], b1.tiles[4][0], c.tiles[0][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][5], a1.tiles[0][0], b1.tiles[5][0], c.tiles[0][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][6], a1.tiles[0][0], b1.tiles[6][0], c.tiles[0][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[0][7], a1.tiles[0][0], b1.tiles[7][0], c.tiles[0][7]);
        kittens::mma_ABt_base<false,false>(c.tiles[1][7], a1.tiles[1][0], b1.tiles[7][0], c.tiles[1][7]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][6], a1.tiles[1][0], b1.tiles[6][0], c.tiles[1][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][5], a1.tiles[1][0], b1.tiles[5][0], c.tiles[1][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][4], a1.tiles[1][0], b1.tiles[4][0], c.tiles[1][4]);
        kittens::sched::compiler_fence();

        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();
        if (CLUSTER && wid == 0) {
            kittens::cluster::arrive();
        }
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::sched::compiler_fence();

        load_k_step<!TRANSA, 0>(a0, *As[nxt], a_off);
        load_k_step<TRANSB, 0>(b0, *Bs[nxt], b_off);
        kittens::mma_ABt_base<true, false>(c.tiles[1][3], a1.tiles[1][0], b1.tiles[3][0], c.tiles[1][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][2], a1.tiles[1][0], b1.tiles[2][0], c.tiles[1][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][1], a1.tiles[1][0], b1.tiles[1][0], c.tiles[1][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[1][0], a1.tiles[1][0], b1.tiles[0][0], c.tiles[1][0]);
        kittens::mma_ABt_base<false,false>(c.tiles[2][0], a1.tiles[2][0], b1.tiles[0][0], c.tiles[2][0]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][1], a1.tiles[2][0], b1.tiles[1][0], c.tiles[2][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][2], a1.tiles[2][0], b1.tiles[2][0], c.tiles[2][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][3], a1.tiles[2][0], b1.tiles[3][0], c.tiles[2][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][4], a1.tiles[2][0], b1.tiles[4][0], c.tiles[2][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][5], a1.tiles[2][0], b1.tiles[5][0], c.tiles[2][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][6], a1.tiles[2][0], b1.tiles[6][0], c.tiles[2][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[2][7], a1.tiles[2][0], b1.tiles[7][0], c.tiles[2][7]);
        kittens::mma_ABt_base<false,false>(c.tiles[3][7], a1.tiles[3][0], b1.tiles[7][0], c.tiles[3][7]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][6], a1.tiles[3][0], b1.tiles[6][0], c.tiles[3][6]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][5], a1.tiles[3][0], b1.tiles[5][0], c.tiles[3][5]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][4], a1.tiles[3][0], b1.tiles[4][0], c.tiles[3][4]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][3], a1.tiles[3][0], b1.tiles[3][0], c.tiles[3][3]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][2], a1.tiles[3][0], b1.tiles[2][0], c.tiles[3][2]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][1], a1.tiles[3][0], b1.tiles[1][0], c.tiles[3][1]);
        kittens::mma_ABt_base<true, false>(c.tiles[3][0], a1.tiles[3][0], b1.tiles[0][0], c.tiles[3][0]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 6, 0);
            __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, 5, 0);
        }
        kittens::sched::compiler_fence();
        if constexpr (CLUSTER) {
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
        if constexpr (EPILOGUE == GemmEpilogue::BGRADB) {
            if (m_tile == 0 && warp_m == 3) {
#pragma unroll
                for (int t = 0; t < RT_B::height; t++) {
                    const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b1.tiles[t][0].data), v16f);
#pragma unroll
                    for (int e = 0; e < 16; e += 2) {
                        bsum[t] += f[e] + f[e + 1];
                    }
                }
            }
        }
    }

    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();

    const int row0 = warp_m * REG_M + 8 * (lane / 16);
    const int col0 = warp_n * REG_N + lane % 16;
    const bool edge = (m_tile + 1) * BLOCK_ROW > M || (n_tile + 1) * BLOCK_COL > N || M % 8 != 0;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        float *ws = reinterpret_cast<float *>(D) + size_t(blockIdx.z) * M * N + size_t(n_tile) * BLOCK_COL * M +
                    size_t(m_tile) * BLOCK_ROW;
#pragma unroll
        for (int i = 0; i < RT_C::height; i++) {
#pragma unroll
            for (int j = 0; j < RT_C::width; j++) {
                const auto &d = c.tiles[i][j].data;
                const int row = row0 + 16 * i;
                const int col = col0 + 16 * j;
                if (!edge || (row < m_rem && col < n_rem)) {
                    *reinterpret_cast<v8f *>(ws + size_t(col) * M + row) =
                        v8f{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
                }
            }
        }
        return;
    }

    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&__shm[0]));
    if constexpr (EPILOGUE == GemmEpilogue::BGRADB) {
        if (m_tile == 0) {
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
#pragma unroll
            for (int t = 0; t < RT_C::width; t++) {
                const int p = (warp_m * 2 + lane / 16) * BLOCK_COL + col0 + t * 16;
                *lds_ptr<float>(lds0 + p * sizeof(float)) = bsum[t];
            }
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
            float sum = 0.f;
#pragma unroll
            for (int p = 0; p < WARPS_ROW * 2; p++) {
                sum += *lds_ptr<float>(lds0 + (p * BLOCK_COL + threadIdx.x) * sizeof(float));
            }
            if (int(threadIdx.x) < n_rem) {
                bias[n_tile * BLOCK_COL + threadIdx.x] = static_cast<AuxT>(sum);
            }
        }
    }

    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
#pragma unroll
    for (int i = 0; i < RT_C::height; i++) {
#pragma unroll
        for (int j = 0; j < RT_C::width; j++) {
            const auto &d = c.tiles[i][j].data;
            const v8f v{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
            const uint32_t addr =
                lds0 + (col0 + 16 * j) * (BLOCK_ROW * sizeof(StT) + 16) + (row0 + 16 * i) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(v, vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    if (edge) {
        gemm_epilogue<StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, true>(lds0, D, bias, aux, alpha, beta, M, N,
                                                                            m_tile, n_tile);
    } else {
        gemm_epilogue<StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, false>(lds0, D, bias, aux, alpha, beta, M, N,
                                                                             m_tile, n_tile);
    }
}

template <bool TRANSA, bool TRANSB, typename T, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT, typename AuxT>
static void launch_gemm(const Bf16GemmArgs &args, void *d, int splits, bool mcast) {
    const auto kernel = mcast ? bf16_gemm_kernel<TRANSA, TRANSB, true, T, EPILOGUE, ACCUMULATE, OutT, AuxT>
                              : bf16_gemm_kernel<TRANSA, TRANSB, false, T, EPILOGUE, ACCUMULATE, OutT, AuxT>;
    hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
    const kittens::gl<T, -1, -1, -1, -1> ga(static_cast<T *>(const_cast<void *>(args.A)), 1, 1,
                                            TRANSA ? args.M : args.K, TRANSA ? args.K : args.M);
    const kittens::gl<T, -1, -1, -1, -1> gb(static_cast<T *>(const_cast<void *>(args.B)), 1, 1,
                                            TRANSB ? args.K : args.N, TRANSB ? args.N : args.K);
    const int cluster = mcast ? CLUSTER_DIM : 1;
    const dim3 grid(kittens::ceil_div(kittens::ceil_div(args.M, BLOCK_ROW), cluster) * cluster,
                    kittens::ceil_div(kittens::ceil_div(args.N, BLOCK_COL), cluster) * cluster, splits);
    hipLaunchKernelGGL(kernel, grid, dim3(NUM_THREADS), LDS_BYTES, args.stream, ga, gb, static_cast<OutT *>(d),
                       static_cast<AuxT *>(const_cast<void *>(args.bias)), static_cast<AuxT *>(args.aux_gelu),
                       args.alpha, args.beta, args.M, args.N, args.K);
}

template <typename OutT>
static void launch_reduce(const Bf16GemmArgs &args, const void *workspace, int splits) {
    const size_t mn     = size_t(args.M) * args.N;
    const size_t blocks = std::min<size_t>((mn / 8 + REDUCE_THREADS - 1) / REDUCE_THREADS, REDUCE_MAX_BLOCKS);
    const auto kernel = args.beta != 0.f ? splitk_reduce_kernel<OutT, true> : splitk_reduce_kernel<OutT, false>;
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(REDUCE_THREADS), 0, args.stream,
                       static_cast<const float *>(workspace), static_cast<OutT *>(args.C), args.alpha, args.beta,
                       mn, splits);
}

template <bool TRANSA, bool TRANSB, typename T>
static bool dispatch_gemm(const Bf16GemmArgs &args, void *workspace, int splits, GemmEpilogue epilogue,
                          bool mcast) {
    using IT = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;
    const int in_dtype = std::is_same_v<T, kittens::half> ? KITTENS_FLOAT16 : KITTENS_BFLOAT16;
    const bool accumulate = args.beta != 0.f;
    const bool f32        = args.out_dtype == KITTENS_FLOAT32;
    if (!f32 && args.out_dtype != in_dtype) {
        return false;
    }
    switch (epilogue) {
    case GemmEpilogue::DEFAULT:
        if (splits > 1) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::SPLIT_K, false, float, float>(args, workspace, splits, mcast);
            if (f32) {
                launch_reduce<float>(args, workspace, splits);
            } else {
                launch_reduce<IT>(args, workspace, splits);
            }
            return true;
        }
        KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
            KITTENS_BOOL_SWITCH(f32, F32,
                using OutT = std::conditional_t<F32, float, IT>;
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::DEFAULT, ACCUMULATE, OutT, OutT>(args, args.C, 1, mcast);
        )) // NOLINT(*)
        return true;
    case GemmEpilogue::BIAS:
        if (args.bias_dtype != in_dtype || (accumulate && (f32 || !TRANSA))) {
            return false;
        }
        if (f32) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, false, float, IT>(args, args.C, 1, mcast);
        } else if (!accumulate) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, false, IT, IT>(args, args.C, 1, mcast);
        } else if constexpr (TRANSA) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, true, IT, IT>(args, args.C, 1, mcast);
        }
        return true;
    case GemmEpilogue::GELU_AUX:
        if constexpr (TRANSA) {
            if (args.aux_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::GELU_AUX, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case GemmEpilogue::GELU_AUX_BIAS:
        if constexpr (TRANSA) {
            if (args.bias_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::GELU_AUX_BIAS, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case GemmEpilogue::DGELU:
        if constexpr (!TRANSA && !TRANSB) {
            if (args.aux_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::DGELU, false, IT, IT>(args, args.C, 1, mcast);
                return true;
            }
        }
        return false;
    case GemmEpilogue::BGRADB:
        if constexpr (TRANSB) {
            if (args.bias_dtype != args.out_dtype) {
                return false;
            }
            KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
                KITTENS_BOOL_SWITCH(f32, F32,
                    using OutT = std::conditional_t<F32, float, IT>;
                    launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BGRADB, ACCUMULATE, OutT, OutT>(args, args.C, 1, mcast);
            )) // NOLINT(*)
            return true;
        }
        return false;
    default:
        return false;
    }
}

class Bf16GemmCdna5 final : public Bf16GemmBackend {
 public:
    bool gemm(const Bf16GemmArgs &args) override {
        if (args.a_dtype != args.b_dtype || (args.a_dtype != KITTENS_BFLOAT16 && args.a_dtype != KITTENS_FLOAT16) ||
            (args.transa && args.transb) || args.M <= 0 || args.N <= 0 || args.K <= 0) {
            return false;
        }
        const uintptr_t ptrs = reinterpret_cast<uintptr_t>(args.A) | reinterpret_cast<uintptr_t>(args.B) |
                               reinterpret_cast<uintptr_t>(args.C) | reinterpret_cast<uintptr_t>(args.bias) |
                               reinterpret_cast<uintptr_t>(args.aux_gelu);
        if (ptrs % 16 != 0) {
            return false;
        }

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
        if (args.transa && epilogue == GemmEpilogue::DEFAULT && args.beta == 0.f &&
            args.out_dtype != KITTENS_FLOAT32 && args.alpha != 1.f) {
            return false;
        }

        const uintptr_t ws_raw = reinterpret_cast<uintptr_t>(args.workspace);
        const uintptr_t ws     = (ws_raw + WS_ALIGN - 1) & ~uintptr_t(WS_ALIGN - 1);
        const size_t pad     = ws - ws_raw;
        const size_t ws_size = args.workspace_size > pad ? args.workspace_size - pad : 0;
        const int64_t tiles = int64_t(kittens::ceil_div(args.M, BLOCK_ROW)) * kittens::ceil_div(args.N, BLOCK_COL);
        const size_t split_bytes = size_t(args.M) * args.N * sizeof(float);
        int splits = 1;
        if (epilogue == GemmEpilogue::DEFAULT && args.M % 8 == 0) {
            if (args.splitk > 1 && args.splitk <= args.K / BLOCK_K && split_bytes * args.splitk <= ws_size) {
                splits = args.splitk;
            }
            double best_t = args.K;
            for (int s = 2; args.splitk == 0 && s <= 16 && tiles * s <= args.cu_count; s *= 2) {
                const double t = double(args.K) / s + 512 + 2048 * double(tiles * s) / args.cu_count;
                if (s <= args.K / BLOCK_K && split_bytes * s <= ws_size && t < best_t) {
                    splits = s;
                    best_t = t;
                }
            }
        }

        void *workspace = reinterpret_cast<void *>(ws);
#ifdef NVTE_HK_GFX1250_MCAST
        const bool mcast = NVTE_HK_GFX1250_MCAST != 0;
#else
        const bool mcast = transformer_engine::cuda::sm_arch_name().find("-strict") == std::string::npos;
#endif
        const bool fp16 = args.a_dtype == KITTENS_FLOAT16;
        if (args.transa && !args.transb) {
            return fp16 ? dispatch_gemm<true, false, kittens::half>(args, workspace, splits, epilogue, mcast)
                        : dispatch_gemm<true, false, kittens::bf16>(args, workspace, splits, epilogue, mcast);
        }
        if (!args.transa && args.transb) {
            return fp16 ? dispatch_gemm<false, true, kittens::half>(args, workspace, splits, epilogue, mcast)
                        : dispatch_gemm<false, true, kittens::bf16>(args, workspace, splits, epilogue, mcast);
        }
        return fp16 ? dispatch_gemm<false, false, kittens::half>(args, workspace, splits, epilogue, mcast)
                    : dispatch_gemm<false, false, kittens::bf16>(args, workspace, splits, epilogue, mcast);
    }
};

}  // namespace te_kittens::cdna5::bf16

Bf16GemmBackend *Bf16GemmBackend::get_cdna5() {
    static te_kittens::cdna5::bf16::Bf16GemmCdna5 impl;
    return &impl;
}
