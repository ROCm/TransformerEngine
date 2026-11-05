/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// gfx1250 BF16/FP16 GEMM: 256x256 tiles, 4x4 multicast clusters or none, any M, N, K.

#pragma once

#include "hip/hip_runtime.h"
#include <cstdint>
#include <type_traits>
#include "common.cuh"

namespace te_kittens::cdna5 {
namespace bf16_gemm {

constexpr int BLOCK_K     = BF16_BLOCK_K;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
constexpr int WARPS_N     = 2;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int STAGES      = 2;
constexpr int CLUSTER_DIM = 4;
constexpr int A_FRAGS     = WARP_M / 16;
constexpr int B_FRAGS     = WARP_N / 16;
constexpr int DS_STEP     = 2 * (A_FRAGS + B_FRAGS);
constexpr int WMMA_STEP   = A_FRAGS * B_FRAGS;
constexpr int PIN_WMMA    = 8;
constexpr int PIN_REP     = WMMA_STEP / PIN_WMMA;
constexpr int PIN_DS      = DS_STEP / PIN_REP;
// WMMAs of the last K step issued before the stage barrier.
constexpr int PRE_BAR     = 12;

// SchedGroup masks.
constexpr int SCHED_DS_READ = 0x100;
constexpr int SCHED_WMMA    = 0x008;

static_assert(NUM_WARPS * kittens::WARP_THREADS == NUM_THREADS);
static_assert(K_STEP * 4 == BLOCK_K && A_FRAGS == 4 && B_FRAGS == 8);
static_assert((WMMA_STEP - PRE_BAR) % PIN_REP == 0);

template <typename T, int R, int C>
using st_e = kittens::st<T, R, C, kittens::ducks::st_shape::st_16x32_padded<>>;
template <typename T, int R, int C>
using rt_e = kittens::rt<T, R, C, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
template <typename T>
using gl_e = kittens::gl<T, -1, -1, -1, -1>;
using C_acc = kittens::rt_fl<WARP_M, WARP_N, kittens::col_l, kittens::rt_16x16_s>;

// The clang element type of an input type (HK's bf16 and half are structs).
template <typename T>
using native_t = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;

// TRANS: the operand is M/N-contiguous and its LDS stage is [BLOCK_K][rows].
template <typename T, Layout L>
struct Tiles {
    static constexpr bool TRANS_A = L != TN;
    static constexpr bool TRANS_B = L == NT;
    using A_stage = std::conditional_t<TRANS_A, st_e<T, BLOCK_K, BLOCK_M>, st_e<T, BLOCK_M, BLOCK_K>>;
    using B_stage = std::conditional_t<TRANS_B, st_e<T, BLOCK_K, BLOCK_N>, st_e<T, BLOCK_N, BLOCK_K>>;
    using A_reg = rt_e<T, WARP_M, K_STEP>;
    using B_reg = rt_e<T, WARP_N, K_STEP>;

    static constexpr int LDS_BYTES = STAGES * (sizeof(A_stage) + sizeof(B_stage));

    static_assert(A_stage::shape::pad_interval % WARP_M == 0 && B_stage::shape::pad_interval % WARP_N == 0,
                  "a warp's operand rows must not straddle a pad interval of a K-major stage");
    static_assert(BLOCK_N * staged_stride(sizeof(float)) <= LDS_BYTES, "fp32 C staging fits the operand LDS");
    static_assert(LDS_BYTES <= LDS_MAX_BYTES);
};

// Kept in a function: written inline four times it spilled.
template <int NW = PIN_WMMA>
__device__ __forceinline__ void pin_interleave() {
#pragma unroll
    for (int i = 0; i < PIN_REP; i++) {
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, PIN_DS, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_WMMA, NW, 0);
    }
}

template <typename ST>
constexpr int tr_row_elems = ST::cols + ST::cols / ST::shape::pad_interval * ST::shape::pad_amount;

// Byte offset of a lane's first ds_load_tr16_b128 for warp rows from w0: lane half h gets K 16h + 0..7 of
// row L % 16, the order the K-contiguous load produces.
template <typename ST>
__device__ __forceinline__ uint32_t tr_lane_offset(int w0) {
    using Shape = typename ST::shape;
    const int L   = kittens::laneid();
    const int g   = L / 8;
    const int col = w0 + 8 * (g % 2);
    return 2 * ((L % 8 + 16 * (g / 2)) * tr_row_elems<ST> + col + col / Shape::pad_interval * Shape::pad_amount);
}

// Two ds_load_tr16_b128 per 16x32 fragment.
template <int KS, typename RT, typename ST>
__device__ __forceinline__ void load_tr(RT &dst, const ST &src, uint32_t lane_off) {
    static_assert(std::is_same_v<typename RT::shape, kittens::ducks::rt_shape::rt_16x32> && RT::cols == K_STEP);
    static_assert(ST::rows == BLOCK_K && ST::cols % ST::shape::pad_interval == 0);
    typedef short v8s __attribute__((ext_vector_type(8)));
    using packed_t = std::remove_cvref_t<decltype(dst.tiles[0][0].data[0])>;
    constexpr int ROW_BYTES = 2 * tr_row_elems<ST>;

    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data)) + lane_off;
#pragma unroll
    for (int ti = 0; ti < RT::height; ti++) {
#pragma unroll
        for (int x = 0; x < 2; x++) {
            const uint32_t addr = base + (KS * K_STEP + 8 * x) * ROW_BYTES + ti * 16 * 2;
            const v8s v = __builtin_amdgcn_ds_load_tr16_b128_v8i16(lds_ptr<v8s>(addr));
            const packed_t *p = reinterpret_cast<const packed_t *>(&v);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                dst.tiles[ti][0].data[4 * x + i] = p[i];
            }
        }
    }
}

template <bool TRANS, int KS, typename RT, typename ST>
__device__ __forceinline__ void load_step(RT &dst, const ST &st, uint32_t warp_off) {
    if constexpr (TRANS) {
        load_tr<KS>(dst, st, warp_off);
    } else {
        kittens::load(dst, st, int(warp_off) + KS * K_STEP);
    }
}

// WMMAs T..T1-1 of a K step in zigzag order: op t uses A fragment t / 8 and B fragment t % 8, reversed on odd
// rows, and reuses A except at the start of a row.
template <int T, int T1, typename RA, typename RB>
__device__ __forceinline__ void mma_zigzag(C_acc &c, const RA &a, const RB &b) {
    constexpr int i = T / B_FRAGS;
    constexpr int j = i % 2 ? B_FRAGS - 1 - T % B_FRAGS : T % B_FRAGS;
    kittens::mma_ABt_base<T % B_FRAGS != 0, false>(c.tiles[i][j], a.tiles[i][0], b.tiles[j][0], c.tiles[i][j]);
    if constexpr (T + 1 < T1) {
        mma_zigzag<T + 1, T1>(c, a, b);
    }
}

// BGRAD: adds this lane's K values of each B fragment row (one dbias column per row) to s.
template <bool BGRAD, typename RT>
__device__ __forceinline__ void colsum(float *s, const RT &b, bool active) {
    using base = std::remove_cvref_t<decltype(b.tiles[0][0])>;
    if constexpr (BGRAD) {
        if (active) {
#pragma unroll
            for (int t = 0; t < RT::height; t++) {
#pragma unroll
                for (int d = 0; d < base::packed_per_thread; d++) {
                    const float2 f = kittens::base_types::convertor<float2, typename base::dtype>::convert(
                        b.tiles[t][0].data[d]);
                    s[t] += f.x + f.y;
                }
            }
        }
    }
}

// Wave 0 fills A, wave 1 B: one K-block of the operand from tile origin (tile, kblock), kx K and rem rows.
template <bool TRANS, typename ST, typename T>
__device__ __forceinline__ void fill(ST &dst, const gl_e<T> &src, int tile, int kblock, int kx, int rem, int rows,
                                     int K, uint32_t mask, uint32_t count) {
    if constexpr (TRANS) {
        kittens::tdm::load_async(dst, src, {0, 0, kblock, tile}, kx, rem, rows, mask, count);
    } else {
        kittens::tdm::load_async(dst, src, {0, 0, tile, kblock}, rem, kx, K, mask, count);
    }
}

// C is staged in LDS as D itself for a D of the input type without epilogue, else as fp32.
template <typename T, Epi E, bool ACC, typename OutT>
using staged_t =
    std::conditional_t<(E == NONE || E == BGRADB) && !ACC && std::is_same_v<OutT, native_t<T>>, OutT, float>;

// The SPLIT partial straight from the accumulators; EDGE stores only rows < m_rem and columns < n_rem.
template <bool EDGE>
__device__ __forceinline__ void store_split(float *ws, const C_acc &C, int row0, int col0, int m_rem, int n_rem,
                                            int M) {
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const auto &d = C.tiles[i][j].data;
            const v8f v{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
            const int row = row0 + 16 * i;
            const int col = col0 + 16 * j;
            if (!EDGE || (row < m_rem && col < n_rem)) {
                *reinterpret_cast<v8f *>(ws + size_t(col) * M + row) = v;
            }
        }
    }
}

// MC: cluster with multicast fills.
template <bool MC, Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__device__ __forceinline__ void bf16_gemm_body(const gl_e<T> &ga, const gl_e<T> &gb, OutT *__restrict__ D,
                                               SideT *__restrict__ bias, SideT *__restrict__ aux, float alpha,
                                               float beta, int M, int N, int K) {
    using Tl = Tiles<T, L>;
    using A_stage = typename Tl::A_stage;
    using B_stage = typename Tl::B_stage;
    using A_reg = typename Tl::A_reg;
    using B_reg = typename Tl::B_reg;
    constexpr bool BGRAD = E == BGRADB;
    static_assert(!BGRAD || Tl::TRANS_B, "BGRADB is the NT (wgrad) epilogue");

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al(reinterpret_cast<int *>(&__shm[0]));
    A_stage &a0 = al.allocate_in<kittens::segment<0>, A_stage>();
    B_stage &b0 = al.allocate_in<kittens::segment<0>, B_stage>();
    A_stage &a1 = al.allocate_in<kittens::segment<0>, A_stage>();
    B_stage &b1 = al.allocate_in<kittens::segment<0>, B_stage>();
    A_stage *A_st[STAGES] = {&a0, &a1};
    B_stage *B_st[STAGES] = {&b0, &b1};

    kittens::sched::lock_simd();

    C_acc C;
    kittens::zero(C);
    float bsum[BGRAD ? B_FRAGS : 1] = {};

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int tiles_m = ceil_div(M, BLOCK_M);
    const int tiles_n = ceil_div(N, BLOCK_N);
    const int wid     = kittens::warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int num_kb  = ceil_div(K, BLOCK_K);
    int kb0      = 0;
    int k_blocks = num_kb;
    if constexpr (E == SPLIT) {
        const int splits = gridDim.z;
        const int z      = blockIdx.z;
        kb0      = num_kb * z / splits;
        k_blocks = num_kb * (z + 1) / splits - kb0;
    }
    // Never taken (the host keeps K > 0 and splits <= num_kb), but without it the accumulator zeroing grows.
    if (k_blocks <= 0) {
        return;
    }
    // A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (tile_m >= tiles_m || tile_n >= tiles_n) {
        return;
    }

    // The wave's slice of each operand tile: an element index (K-contiguous) or a lane byte offset.
    const uint32_t warp_off_a = Tl::TRANS_A ? tr_lane_offset<A_stage>(warp_r * WARP_M)
                                            : uint32_t(warp_r * WARP_M * BLOCK_K);
    const uint32_t warp_off_b = Tl::TRANS_B ? tr_lane_offset<B_stage>(warp_c * WARP_N)
                                            : uint32_t(warp_c * WARP_N * BLOCK_K);

    // Multicast sets: WGs sharing tile_m receive A, those sharing tile_n receive B (ISA 10.7).
    uint32_t a_mask = 0;
    uint32_t b_mask = 0;
    if constexpr (MC) {
        const uint32_t cx   = blockIdx.x % CLUSTER_DIM;
        const uint32_t cy   = blockIdx.y % CLUSTER_DIM;
        const uint32_t in_x = min(tiles_m - int(blockIdx.x - cx), CLUSTER_DIM);
        const uint32_t in_y = min(tiles_n - int(blockIdx.y - cy), CLUSTER_DIM);
        a_mask = (cluster_col_bits(CLUSTER_DIM, CLUSTER_DIM) << cx) & ((1u << (CLUSTER_DIM * in_y)) - 1u);
        b_mask = ((1u << in_x) - 1u) << (CLUSTER_DIM * cy);
    }

    // Wave 0 fills A, wave 1 B. The K % BLOCK_K head is block 0; later blocks read through ga_k and gb_k,
    // moved back by back elements.
    const int m_rem  = M - tile_m * BLOCK_M;
    const int n_rem  = N - tile_n * BLOCK_N;
    const int k_head = K % BLOCK_K;
    const int back   = k_head ? BLOCK_K - k_head : 0;
    gl_e<T> ga_k = ga;
    gl_e<T> gb_k = gb;
    ga_k.raw_ptr -= size_t(back) * (Tl::TRANS_A ? M : 1);
    gb_k.raw_ptr -= size_t(back) * (Tl::TRANS_B ? N : 1);
    // Block 0 is the head (or this slice's first block).
    const gl_e<T> &ga0 = kb0 == 0 ? ga : ga_k;
    const gl_e<T> &gb0 = kb0 == 0 ? gb : gb_k;
    const int kx0 = kb0 == 0 && k_head ? k_head : K;
    if (wid == 0) {
        fill<Tl::TRANS_A>(*A_st[0], ga0, tile_m, kb0, kx0, m_rem, M, K, a_mask, 1);
    }
    if (wid == 1) {
        fill<Tl::TRANS_B>(*B_st[0], gb0, tile_n, kb0, kx0, n_rem, N, K, b_mask, 1);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    if constexpr (MC) {
        kittens::cluster::sync();
    } else {
        // The empty branch keeps the cluster kernel's block structure: the two K loops differ only in the barrier.
        kittens::sync::arrive();
        if (wid == 0) {
            asm volatile("");
        }
        kittens::sync::wait();
    }
    kittens::sched::compiler_fence();

    // Two register slots, named rather than indexed so they stay in registers; K steps alternate.
    struct operand_slot {
        A_reg a;
        B_reg b;
    };
    operand_slot slot0;
    operand_slot slot1;
    // Column sums of K step ks, after its WMMAs. Kept a lambda: a direct colsum call costs the bf16 BGRADB K loop
    // 14% more instructions (16-bit repacking of the B fragments).
    auto bgrad = [&](int ks, const B_reg &b) {
        colsum<BGRAD>(bsum, b, tile_m == 0 && warp_r == ks);
    };
    load_step<Tl::TRANS_A, 0>(slot0.a, *A_st[0], warp_off_a);
    load_step<Tl::TRANS_B, 0>(slot0.b, *B_st[0], warp_off_b);

    for (int kb = 0; kb < k_blocks; kb++) {
        const int cur = kb % STAGES;
        const int nxt = (kb + 1) % STAGES;

        // A count of zero skips the tail transfer without a branch (TDM cannot be EXEC-masked).
        const int fill_k = kb0 + ((kb + 1 < k_blocks) ? kb + 1 : 0);
        const uint32_t count = (kb + 1 < k_blocks) ? 1u : 0u;
        if (wid == 0) {
            fill<Tl::TRANS_A>(*A_st[nxt], ga_k, tile_m, fill_k, K, m_rem, M, K, a_mask, count);
        }
        if (wid == 1) {
            fill<Tl::TRANS_B>(*B_st[nxt], gb_k, tile_n, fill_k, K, n_rem, N, K, b_mask, count);
        }
        kittens::sched::compiler_fence();

        load_step<Tl::TRANS_A, 1>(slot1.a, *A_st[cur], warp_off_a);
        load_step<Tl::TRANS_B, 1>(slot1.b, *B_st[cur], warp_off_b);
        kittens::mma_ABt(C, slot0.a, slot0.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(0, slot0.b);

        load_step<Tl::TRANS_A, 2>(slot0.a, *A_st[cur], warp_off_a);
        load_step<Tl::TRANS_B, 2>(slot0.b, *B_st[cur], warp_off_b);
        kittens::mma_ABt(C, slot1.a, slot1.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(1, slot1.b);

        load_step<Tl::TRANS_A, 3>(slot1.a, *A_st[cur], warp_off_a);
        load_step<Tl::TRANS_B, 3>(slot1.b, *B_st[cur], warp_off_b);
        kittens::mma_ABt(C, slot0.a, slot0.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(2, slot0.b);

        // Last K step: PRE_BAR WMMAs overlap the wait and barrier, the rest the next reads.
        mma_zigzag<0, PRE_BAR>(C, slot1.a, slot1.b);
        kittens::sched::compiler_fence();

        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();
        if (wid == 0) {
            if constexpr (MC) {
                kittens::cluster::arrive();
            } else {
                asm volatile("");
            }
        }
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::sched::compiler_fence();

        load_step<Tl::TRANS_A, 0>(slot0.a, *A_st[nxt], warp_off_a);
        load_step<Tl::TRANS_B, 0>(slot0.b, *B_st[nxt], warp_off_b);
        mma_zigzag<PRE_BAR, WMMA_STEP>(C, slot1.a, slot1.b);

        pin_interleave<(WMMA_STEP - PRE_BAR) / PIN_REP>();
        kittens::sched::compiler_fence();
        if constexpr (MC) {
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
        bgrad(3, slot1.b);
    }

    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();

    // Accumulator lane L holds column L % 16 and rows 8 (L / 16) + 0..7 of each 16x16 tile.
    using StT = staged_t<T, E, ACC, OutT>;
    int lane = kittens::laneid();
    if constexpr (L == NT && E == NONE && !std::is_same_v<StT, float>) {
        // Else the 16-bit staging reuses the transposed reads' lane terms past the K loop (+1 VGPR).
        asm volatile("" : "+v"(lane));
    }
    const int row0 = warp_r * WARP_M + 8 * (lane / 16);
    const int col0 = warp_c * WARP_N + lane % 16;
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == SPLIT) {
        float *ws = reinterpret_cast<float *>(D) + size_t(blockIdx.z) * M * N + size_t(tile_n) * BLOCK_N * M +
                    size_t(tile_m) * BLOCK_M;
        if (edge) {
            store_split<true>(ws, C, row0, col0, m_rem, n_rem, M);
        } else {
            store_split<false>(ws, C, row0, col0, m_rem, n_rem, M);
        }
        return;
    }

    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&__shm[0]));
    if constexpr (BGRAD) {
        // Partials [warp_r][lane half][column] in LDS; thread n adds column n's 8 in that order.
        if (tile_m == 0) {
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
#pragma unroll
            for (int t = 0; t < B_FRAGS; t++) {
                const int p = (warp_r * 2 + lane / 16) * BLOCK_N + col0 + t * 16;
                *lds_ptr<float>(lds0 + p * sizeof(float)) = bsum[t];
            }
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
            float sum = 0.f;
#pragma unroll
            for (int p = 0; p < WARPS_M * 2; p++) {
                sum += *lds_ptr<float>(lds0 + (p * BLOCK_N + threadIdx.x) * sizeof(float));
            }
            if (int(threadIdx.x) < n_rem) {
                bias[tile_n * BLOCK_N + threadIdx.x] = static_cast<SideT>(sum);
            }
        }
    }

    constexpr int STRIDE = staged_stride(sizeof(StT));
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
#pragma unroll
    for (int i = 0; i < A_FRAGS; i++) {
#pragma unroll
        for (int j = 0; j < B_FRAGS; j++) {
            const auto &d = C.tiles[i][j].data;
            const v8f v{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
            const uint32_t addr = lds0 + (col0 + 16 * j) * STRIDE + (row0 + 16 * i) * sizeof(StT);
            *lds_ptr<vec8<StT>>(addr) = __builtin_convertvector(v, vec8<StT>);
        }
    }
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    if (edge) {
        copy_out<StT, E, ACC, OutT, SideT, L != TN, true>(lds0, D, bias, aux, alpha, beta, M, N, tile_m, tile_n);
    } else {
        copy_out<StT, E, ACC, OutT, SideT, L != TN, false>(lds0, D, bias, aux, alpha, beta, M, N, tile_m, tile_n);
    }
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __cluster_dims__(CLUSTER_DIM, CLUSTER_DIM, 1) __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel(const gl_e<T> ga, const gl_e<T> gb, OutT *__restrict__ D, SideT *__restrict__ bias,
                      SideT *__restrict__ aux, float alpha, float beta, int M, int N, int K) {
    bf16_gemm_body<ARCH_MCAST, L, T, E, ACC, OutT, SideT>(ga, gb, D, bias, aux, alpha, beta, M, N, K);
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
__global__ __launch_bounds__(NUM_THREADS, 1)
void bf16_gemm_kernel_nomc(const gl_e<T> ga, const gl_e<T> gb, OutT *__restrict__ D, SideT *__restrict__ bias,
                           SideT *__restrict__ aux, float alpha, float beta, int M, int N, int K) {
    bf16_gemm_body<false, L, T, E, ACC, OutT, SideT>(ga, gb, D, bias, aux, alpha, beta, M, N, K);
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
void launch_kernel(const GemmArgs &args, void *d, int splits, bool mcast) {
    using Tl = Tiles<T, L>;
    const auto kernel = mcast ? bf16_gemm_kernel<L, T, E, ACC, OutT, SideT>
                              : bf16_gemm_kernel_nomc<L, T, E, ACC, OutT, SideT>;
    kittens_check(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, Tl::LDS_BYTES),
                  "gfx1250 bf16_gemm_kernel LDS");
    T *A = static_cast<T *>(const_cast<void *>(args.A));
    T *B = static_cast<T *>(const_cast<void *>(args.B));
    const gl_e<T> ga(A, 1, 1, Tl::TRANS_A ? args.K : args.M, Tl::TRANS_A ? args.M : args.K);
    const gl_e<T> gb(B, 1, 1, Tl::TRANS_B ? args.K : args.N, Tl::TRANS_B ? args.N : args.K);
    const int side = mcast ? CLUSTER_DIM : 1;
    const dim3 grid(ceil_div(ceil_div(args.M, BLOCK_M), side) * side, ceil_div(ceil_div(args.N, BLOCK_N), side) * side,
                    splits);
    hipLaunchKernelGGL(kernel, grid, dim3(NUM_THREADS), Tl::LDS_BYTES, args.stream, ga, gb, static_cast<OutT *>(d),
                       static_cast<SideT *>(const_cast<void *>(args.bias)), static_cast<SideT *>(args.aux_gelu),
                       args.alpha, args.beta, args.M, args.N, args.K);
    kittens_check(hipGetLastError(), "gfx1250 bf16_gemm_kernel");
}

// The compiled variants; in is the input dtype. Any other combination launches nothing.
constexpr bool supported(Layout L, Epi E, bool acc, int in, int out, int side) {
    const bool in_or_f32 = out == in || out == KITTENS_FLOAT32;
    switch (E) {
    case NONE:          return side == out && in_or_f32;
    case BIAS:          return side == in && in_or_f32 && (!acc || (L == TN && out == in));
    case GELU_AUX:      return L == TN && !acc && out == in && side == in;
    case GELU_AUX_BIAS: return L == TN && !acc && out == in && side == in;
    case DGELU:         return L == NN && !acc && out == in && side == in;
    case BGRADB:        return L == NT && side == out && in_or_f32;
    default:            return false;
    }
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT, typename SideT>
bool launch_variant(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    if constexpr (!supported(L, E, ACC, dtype_of<native_t<T>>, dtype_of<OutT>, dtype_of<SideT>)) {
        return false;
    } else {
        if constexpr (E == NONE) {
            if (splits > 1) {
                launch_kernel<L, T, SPLIT, false, float, float>(args, workspace, splits, mcast);
                splitk_reduce(args, workspace, splits);
                return true;
            }
        }
        launch_kernel<L, T, E, ACC, OutT, SideT>(args, args.C, 1, mcast);
        return true;
    }
}

template <Layout L, typename T, Epi E, bool ACC, typename OutT>
bool switch_side(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    switch (side_dtype(args)) {
    case KITTENS_FLOAT32:  return launch_variant<L, T, E, ACC, OutT, float>(args, workspace, splits, mcast);
    case KITTENS_FLOAT16:  return launch_variant<L, T, E, ACC, OutT, _Float16>(args, workspace, splits, mcast);
    case KITTENS_BFLOAT16: return launch_variant<L, T, E, ACC, OutT, __bf16>(args, workspace, splits, mcast);
    default:               return false;
    }
}

template <Layout L, typename T, Epi E, bool ACC>
bool switch_out(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    switch (args.out_dtype) {
    case KITTENS_FLOAT32:  return switch_side<L, T, E, ACC, float>(args, workspace, splits, mcast);
    case KITTENS_FLOAT16:  return switch_side<L, T, E, ACC, _Float16>(args, workspace, splits, mcast);
    case KITTENS_BFLOAT16: return switch_side<L, T, E, ACC, __bf16>(args, workspace, splits, mcast);
    default:               return false;
    }
}

template <Layout L, typename T, Epi E>
bool switch_acc(const GemmArgs &args, void *workspace, int splits, bool mcast) {
    if (args.beta != 0.f) {
        return switch_out<L, T, E, true>(args, workspace, splits, mcast);
    }
    return switch_out<L, T, E, false>(args, workspace, splits, mcast);
}

template <Layout L, typename T>
bool switch_epi(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    switch (epi) {
    case NONE:          return switch_acc<L, T, NONE>(args, workspace, splits, mcast);
    case BIAS:          return switch_acc<L, T, BIAS>(args, workspace, splits, mcast);
    case GELU_AUX:      return switch_acc<L, T, GELU_AUX>(args, workspace, splits, mcast);
    case GELU_AUX_BIAS: return switch_acc<L, T, GELU_AUX_BIAS>(args, workspace, splits, mcast);
    case DGELU:         return switch_acc<L, T, DGELU>(args, workspace, splits, mcast);
    case BGRADB:        return switch_acc<L, T, BGRADB>(args, workspace, splits, mcast);
    default:            return false;
    }
}

}  // namespace bf16_gemm

template <Layout L>
bool launch_bf16(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast) {
    if (args.a_dtype == KITTENS_FLOAT16) {
        return bf16_gemm::switch_epi<L, kittens::half>(args, workspace, splits, epi, mcast);
    }
    return bf16_gemm::switch_epi<L, kittens::bf16>(args, workspace, splits, epi, mcast);
}

}  // namespace te_kittens::cdna5
