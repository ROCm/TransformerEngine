/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// BF16 / FP16 GEMM for gfx1250: TDM-filled LDS double buffer, 16x16x32 WMMA.

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include "kittens.cuh"
#include "../kittens_common.h"
#include "gemm_common.cuh"

namespace te_kittens::cdna5::bf16 {

constexpr int BLOCK_K = 128;
constexpr int K_STEP  = 32;
constexpr int SG_WMMA = 0x008;
constexpr int SG_DS   = 0x100;

constexpr int lds_bytes(int bm, int bn) { return 2 * BLOCK_K * (bm + 16 + bn + 16) * 2; }

template <typename T>
using in_t = std::conditional_t<std::is_same_v<T, kittens::half>, _Float16, __bf16>;

// An MN-major 128-wide operand pads 32 B per 256 B row: 16 B leaves ds_load_tr16_b128 2-way bank-conflicted.
template <bool MN, int WIDTH>
using op_shape = std::conditional_t<MN && WIDTH == 128, kittens::ducks::st_shape::st_16x32_padded<128, 16>,
                                    kittens::ducks::st_shape::st_16x32_padded<>>;

// K-step KS of an operand from LDS; base is this lane's address in the stage, so every address is base + immediate.
template <bool TRANS, int KS, typename ST, typename RT>
__device__ __forceinline__ void load_k_step(RT &dst, uint32_t base) {
    if constexpr (TRANS) {
        typedef short v8s __attribute__((ext_vector_type(8)));
#pragma unroll
        for (int ti = 0; ti < RT::height; ti++) {
#pragma unroll
            for (int x = 0; x < 2; x++) {
                *reinterpret_cast<v8s *>(&dst.tiles[ti][0].data[4 * x]) = __builtin_amdgcn_ds_load_tr16_b128_v8i16(lds_ptr<v8s>(base + 2 * ST::idx(KS * K_STEP + 8 * x, 16 * ti)));
            }
        }
    } else {
        // The lane's 16 columns at K-step KS stay inside one padding interval (one row).
        static_assert(ST::cols == ST::shape::pad_interval && (K_STEP * (KS + 1)) <= ST::cols);
        const uint32_t addr = base + 2 * KS * K_STEP;
#pragma unroll
        for (int ti = 0; ti < RT::height; ti++) {
            const kittens::u32x4 lo = *lds_ptr<kittens::u32x4>(addr + ti * 2 * ST::idx(16, 0));
            const kittens::u32x4 hi = *lds_ptr<kittens::u32x4>(addr + ti * 2 * ST::idx(16, 0) + 16);
            kittens::u32x4 *d = reinterpret_cast<kittens::u32x4 *>(dst.tiles[ti][0].data);
            d[0] = lo;
            d[1] = hi;
        }
    }
}

// Row ROW of a K-step, zig-zag (odd rows right to left) so A is reused after the first WMMA of the row.
template <int ROW, typename RTC, typename RTA, typename RTB, int... K>
__device__ __forceinline__ void mma_row(RTC &c, const RTA &a, const RTB &b, std::integer_sequence<int, K...>) {
    constexpr int W = RTC::width;
    (kittens::mma_ABt_base<K != 0, false>(c.tiles[ROW][ROW % 2 ? W - 1 - K : K], a.tiles[ROW][0], b.tiles[ROW % 2 ? W - 1 - K : K][0], c.tiles[ROW][ROW % 2 ? W - 1 - K : K]), ...);
}

struct FillArgs {
    int m_tile;
    int n_tile;
    int m_rem;
    int n_rem;
    int M;
    int N;
    int K;
    int wid;
    int kb0;
    int su;
    int k_iters;
    uint32_t a_mask;
    uint32_t b_mask;
};

// TDM load of row band `part` of PARTS of dst. A band wholly beyond tensor_rows is the NULL descriptor and keeps
// stale LDS rows, so split only along M / N (rows outside the output), never along K.
template <int PARTS, typename T, int ROWS, int COLS, typename Shape, typename GL>
__device__ inline void tdm_load_part(kittens::st<T, ROWS, COLS, Shape> &dst, const GL &src, const kittens::coord<> &idx,
                                     int tensor_rows, int tensor_cols, int row_stride, int part, uint32_t cluster_mask,
                                     uint32_t count) {
    using ST = kittens::st<T, ROWS, COLS, Shape>;
    constexpr int R = ROWS / PARTS;
    static_assert(ROWS % PARTS == 0 && (R * COLS) % Shape::pad_interval == 0);
    const int r0        = part * R;
    const int64_t row   = (int64_t(idx.b) * src.depth() + idx.d) * src.rows() + idx.r * ROWS + r0;
    const T *base       = src.raw_ptr + (row * src.cols() + idx.c * COLS);
    const int rows_left = tensor_rows - r0;
    kittens::v4u32 g0;
    kittens::v8u32 g1;
    kittens::build_tdm_descriptor_2d<Shape, R, COLS, T, false>(g0, g1, base, &dst.data[ST::idx(r0, 0)], rows_left > 0 ? rows_left : 0, tensor_cols, row_stride, cluster_mask, 0, rows_left > 0 ? count : 0u);
    kittens::v4u32 g2 = {0, 0, 0, 0};
    kittens::v4u32 g3 = {0, 0, 0, 0};
    kittens::emit_tensor_load_to_lds(g0, g1, g2, g3);
}

// Full K block of logical index i (NULL descriptor for i >= k_iters) into stage (as, bs); wave w moves row band w.
template <bool TRANSA, bool TRANSB, int NUM_WARPS, typename ST_A, typename ST_B, typename GL>
__device__ __forceinline__ void fill(ST_A &as, ST_B &bs, const GL &ga, const GL &gb, const FillArgs &f, int i) {
    const int kblk       = f.kb0 + (i < f.k_iters ? stagger_k(i, f.su, f.k_iters) : 0);
    const uint32_t count = i < f.k_iters;
    tdm_load_part<NUM_WARPS>(as, ga, {0, 0, TRANSA ? f.m_tile : kblk, TRANSA ? kblk : f.m_tile}, TRANSA ? f.m_rem : f.K, TRANSA ? f.K : f.m_rem, TRANSA ? f.K : f.M, f.wid, f.a_mask, count);
    tdm_load_part<NUM_WARPS>(bs, gb, {0, 0, TRANSB ? kblk : f.n_tile, TRANSB ? f.n_tile : kblk}, TRANSB ? f.K : f.n_rem, TRANSB ? f.n_rem : f.K, TRANSB ? f.N : f.K, f.wid, f.b_mask, count);
}

template <bool BGRADB, typename IT, typename RT_B>
__device__ __forceinline__ void bias_sum(float *bsum, const RT_B &b, bool add) {
    typedef IT v16in __attribute__((ext_vector_type(16)));
    typedef float v16f __attribute__((ext_vector_type(16)));
    if constexpr (BGRADB) {
        if (add) {
#pragma unroll
            for (int t = 0; t < RT_B::height; t++) {
                const v16f f = __builtin_convertvector(*reinterpret_cast<const v16in *>(b.tiles[t][0].data), v16f);
#pragma unroll
                for (int e = 0; e < 16; e += 2) {
                    bsum[t] += f[e] + f[e + 1];
                }
            }
        }
    }
}

// K block kb from stage CUR, ending with K-step 0 of kb + 1 loaded from CUR ^ 1. CLUSTER fills kb + 1 into CUR ^ 1
// at the start, else kb + 2 is filled into CUR once every wave has read it.
template <int CUR, bool TRANSA, bool TRANSB, bool BIG, bool CLUSTER, bool BGRADB, int NUM_WARPS, typename IT,
          typename RT_C, typename RT_A, typename RT_B, typename ST_A, typename ST_B, typename GL>
__device__ __forceinline__ void kblock(RT_C &c, RT_A &a0, RT_A &a1, RT_B &b0, RT_B &b1, const uint32_t *a_base,
                                       const uint32_t *b_base, ST_A *const *As, ST_B *const *Bs, const GL &ga,
                                       const GL &gb, const FillArgs &f, float *bsum, int warp_m, int kb) {
    constexpr int TM    = RT_C::height;
    constexpr int TN    = RT_C::width;
    constexpr int SG_LD = (RT_A::height + RT_B::height) / 2;
    constexpr int SG_W0 = BIG ? 4 : TM * TN / 4;
    constexpr int SG_W1 = BIG ? 2 : TM * TN / 4;
    constexpr int SG_W2 = BIG ? 6 : TM * TN / 4;
    constexpr int SG_W3 = BIG ? 8 : TM * TN / 4;
    constexpr int SG_W4 = TM * TN - SG_W0 - SG_W1 - SG_W2 - SG_W3;
    constexpr int SG_L0 = TN / 2;
    constexpr int SG_L4 = 2 * TN - SG_L0 - 3;
    static_assert(TM == 4 && 4 * SG_LD == 2 * (RT_A::height + RT_B::height));
    using COLS = std::make_integer_sequence<int, TN>;

    if constexpr (CLUSTER) {
        fill<TRANSA, TRANSB, NUM_WARPS>(*As[CUR ^ 1], *Bs[CUR ^ 1], ga, gb, f, kb + 1);
    }
    kittens::sched::compiler_fence();

    load_k_step<TRANSB, 1, ST_B>(b1, b_base[CUR]);
    load_k_step<!TRANSA, 1, ST_A>(a1, a_base[CUR]);
    kittens::mma_ABt(c, a0, b0, c);
    // 256x256: loads after WMMAs 3, 5, 11, 19, each >= 4 WMMAs after its register's last reader, so the
    // expert-mode WAR wait (s_wait_alu depctr_va_vdst) is partial.
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W0, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W2, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W3, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W4, 0);
    kittens::sched::compiler_fence();
    bias_sum<BGRADB, IT>(bsum, b0, f.m_tile == 0 && warp_m == 0);

    load_k_step<TRANSB, 2, ST_B>(b0, b_base[CUR]);
    load_k_step<!TRANSA, 2, ST_A>(a0, a_base[CUR]);
    kittens::mma_ABt(c, a1, b1, c);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W0, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W2, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W3, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W4, 0);
    kittens::sched::compiler_fence();
    bias_sum<BGRADB, IT>(bsum, b1, f.m_tile == 0 && warp_m == 1);

    load_k_step<TRANSB, 3, ST_B>(b1, b_base[CUR]);
    load_k_step<!TRANSA, 3, ST_A>(a1, a_base[CUR]);
    kittens::mma_ABt(c, a0, b0, c);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W0, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W2, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W3, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_W4, 0);
    kittens::sched::compiler_fence();
    bias_sum<BGRADB, IT>(bsum, b0, f.m_tile == 0 && warp_m == 2);

    mma_row<0>(c, a1, b1, COLS{});
    kittens::sched::compiler_fence();

    kittens::sync::wait_ds<0>();
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    kittens::sync::arrive();
    kittens::sched::compiler_fence();
    mma_row<1>(c, a1, b1, COLS{});
    kittens::sched::compiler_fence();
    kittens::sync::wait();
    if constexpr (CLUSTER) {
        if (f.wid == 0) {
            kittens::cluster::arrive();
        }
    } else {
        // Every wave's reads of stage CUR completed before the arrive.
        fill<TRANSA, TRANSB, NUM_WARPS>(*As[CUR], *Bs[CUR], ga, gb, f, kb + 2);
    }
    kittens::sched::compiler_fence();

    load_k_step<TRANSB, 0, ST_B>(b0, b_base[CUR ^ 1]);
    load_k_step<!TRANSA, 0, ST_A>(a0, a_base[CUR ^ 1]);
    mma_row<2>(c, a1, b1, COLS{});
    mma_row<3>(c, a1, b1, COLS{});
    // A1 tile 1 is freed at WMMA 15 and tile 2 at 23: the next block's loads go after WMMAs 19-22.
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_L0, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, 1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, 1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, 1, 0);
    __builtin_amdgcn_sched_group_barrier(SG_DS, SG_LD, 0);
    __builtin_amdgcn_sched_group_barrier(SG_WMMA, SG_L4, 0);
    kittens::sched::compiler_fence();
    if constexpr (CLUSTER) {
        kittens::cluster::wait();
    }
    kittens::sched::compiler_fence();
    bias_sum<BGRADB, IT>(bsum, b1, f.m_tile == 0 && warp_m == 3);
}

template <bool TRANSA, bool TRANSB, bool MC, typename T, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT,
          typename AuxT, int BLOCK_ROW, int BLOCK_COL>
__global__ __cluster_dims__(MC ? CLUSTER_DIM : 0, MC ? CLUSTER_DIM : 0, MC ?: 0)
__launch_bounds__(wg_threads(BLOCK_ROW, BLOCK_COL), 256 / wg_threads(BLOCK_ROW, BLOCK_COL))
void bf16_gemm_kernel(const kittens::gl<T, -1, -1, -1, -1> ga, const kittens::gl<T, -1, -1, -1, -1> gb,
                      OutT *__restrict__ D, AuxT *__restrict__ bias, AuxT *__restrict__ aux, float alpha, float beta,
                      int M, int N, int K) {
    constexpr bool BIG      = BLOCK_ROW == 256 && BLOCK_COL == 256;
    constexpr int NUM_WARPS = wg_threads(BLOCK_ROW, BLOCK_COL) / kittens::WARP_THREADS;
    constexpr int WARPS_ROW = NUM_WARPS == 8 && BLOCK_ROW >= BLOCK_COL ? 4 : 2;
    constexpr int WARPS_COL = NUM_WARPS / WARPS_ROW;
    constexpr int REG_M     = BLOCK_ROW / WARPS_ROW;
    constexpr int REG_N     = BLOCK_COL / WARPS_COL;
    using ST_A = kittens::st<T, TRANSA ? BLOCK_ROW : BLOCK_K, TRANSA ? BLOCK_K : BLOCK_ROW,
                             op_shape<!TRANSA, BLOCK_ROW>>;
    using ST_B = kittens::st<T, TRANSB ? BLOCK_K : BLOCK_COL, TRANSB ? BLOCK_COL : BLOCK_K,
                             op_shape<TRANSB, BLOCK_COL>>;
    static_assert(2 * (sizeof(ST_A) + sizeof(ST_B)) <= lds_bytes(BLOCK_ROW, BLOCK_COL) &&
                  BLOCK_COL * (BLOCK_ROW * sizeof(float) + 16) <= lds_bytes(BLOCK_ROW, BLOCK_COL) &&
                  lds_bytes(BLOCK_ROW, BLOCK_COL) * (256 / wg_threads(BLOCK_ROW, BLOCK_COL)) <= 320 * 1024);
    static_assert(!MC || BIG);
    using RT_A = kittens::rt<T, REG_M, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_B = kittens::rt<T, REG_N, K_STEP, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using StT  = std::conditional_t<(EPILOGUE == GemmEpilogue::DEFAULT || EPILOGUE == GemmEpilogue::BGRADB) &&
                                        !ACCUMULATE && std::is_same_v<OutT, in_t<T>>, OutT, float>;
    constexpr bool CLUSTER = MC && ARCH_MCAST;
    constexpr bool BGRADB  = EPILOGUE == GemmEpilogue::BGRADB;

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
    float bsum[BGRADB ? RT_C::width : 1] = {};

    // MC keeps blockIdx order: the cluster masks and the shared K order of a cluster depend on it.
    TileCoord tc = {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)};
    if constexpr (!MC) {
        tc = workgroup_tile();
    }
    const int m_tile  = tc.m;
    const int n_tile  = tc.n;
    const int tiles_M = kittens::ceil_div(M, BLOCK_ROW);
    const int tiles_N = kittens::ceil_div(N, BLOCK_COL);
    const int wid     = warpid_uniform();
    const int warp_m  = wid / WARPS_COL;
    const int warp_n  = wid % WARPS_COL;
    const int num_kb  = kittens::ceil_div(K, BLOCK_K);
    int kb0     = 0;
    int k_iters = num_kb;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        const int splits = gridDim.z;
        const int z      = tc.z;
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

    const int su   = MC || k_iters < 3 ? 0 : m_tile % STAGGER_U * STAGGER_STEP % (k_iters - 1);
    const int lane = kittens::laneid();
    const int g    = lane / 8;
    // Own VGPRs for the lane's LDS bases: no address VALU may write a WMMA source VGPR (ISA 7.12.1).
    uint32_t a_base[2];
    uint32_t b_base[2];
#pragma unroll
    for (int st = 0; st < 2; st++) {
        a_base[st] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(As[st]->data)) +
                     2 * (TRANSA ? ST_A::idx(warp_m * REG_M + lane % 16, 16 * (lane / 16))
                                 : ST_A::idx(lane % 8 + 16 * (g / 2), warp_m * REG_M + 8 * (g % 2)));
        b_base[st] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(Bs[st]->data)) +
                     2 * (TRANSB ? ST_B::idx(lane % 8 + 16 * (g / 2), warp_n * REG_N + 8 * (g % 2))
                                 : ST_B::idx(warp_n * REG_N + lane % 16, 16 * (lane / 16)));
        asm volatile("" : "+v"(a_base[st]));
        asm volatile("" : "+v"(b_base[st]));
    }

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
    const int kx0 = kb0 == 0 && k_head ? k_head : K;
    const FillArgs fa = {m_tile, n_tile, m_rem, n_rem, M, N, K, wid, kb0, su, k_iters, a_mask, b_mask};
    if (wid == 0) {
        kittens::tdm::load_async(*As[0], kb0 == 0 ? ga : ga_k, {0, 0, TRANSA ? m_tile : kb0, TRANSA ? kb0 : m_tile}, TRANSA ? m_rem : kx0, TRANSA ? kx0 : m_rem, TRANSA ? K : M, a_mask, 1);
    }
    if (wid == 1) {
        kittens::tdm::load_async(*Bs[0], kb0 == 0 ? gb : gb_k, {0, 0, TRANSB ? kb0 : n_tile, TRANSB ? n_tile : kb0}, TRANSB ? kx0 : n_rem, TRANSB ? n_rem : kx0, TRANSB ? N : K, b_mask, 1);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    if constexpr (CLUSTER) {
        kittens::cluster::sync();
    } else {
        kittens::sync::arrive();
        kittens::sync::wait();
        fill<TRANSA, TRANSB, NUM_WARPS>(*As[1], *Bs[1], ga_k, gb_k, fa, 1);
    }
    kittens::sched::compiler_fence();

    RT_A a0, a1;
    RT_B b0, b1;
    load_k_step<!TRANSA, 0, ST_A>(a0, a_base[0]);
    load_k_step<TRANSB, 0, ST_B>(b0, b_base[0]);

    // Unrolled by 2 so every LDS address is loop-invariant: address VALU between the WMMAs would cost a full
    // va_vdst drain under expert scheduling.
    for (int kb2 = 0; kb2 < k_iters; kb2 += 2) {
        kblock<0, TRANSA, TRANSB, BIG, CLUSTER, BGRADB, NUM_WARPS, in_t<T>>(c, a0, a1, b0, b1, a_base, b_base, As, Bs,
                                                                            ga_k, gb_k, fa, bsum, warp_m, kb2);
        if (kb2 + 1 < k_iters) {
            kblock<1, TRANSA, TRANSB, BIG, CLUSTER, BGRADB, NUM_WARPS, in_t<T>>(c, a0, a1, b0, b1, a_base, b_base, As,
                                                                                Bs, ga_k, gb_k, fa, bsum, warp_m,
                                                                                kb2 + 1);
        }
    }

    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();

    const int row0 = warp_m * REG_M + 8 * (lane / 16);
    const int col0 = warp_n * REG_N + lane % 16;
    const bool edge = (m_tile + 1) * BLOCK_ROW > M || (n_tile + 1) * BLOCK_COL > N || M % 8 != 0;
    if constexpr (EPILOGUE == GemmEpilogue::SPLIT_K) {
        float *ws = D + size_t(tc.z) * M * N + size_t(n_tile) * BLOCK_COL * M + size_t(m_tile) * BLOCK_ROW;
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
        gemm_epilogue<BLOCK_ROW, BLOCK_COL, StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, true>(
            lds0, D, bias, aux, alpha, beta, M, N, m_tile, n_tile);
    } else {
        gemm_epilogue<BLOCK_ROW, BLOCK_COL, StT, EPILOGUE, ACCUMULATE, OutT, AuxT, !TRANSA, false>(
            lds0, D, bias, aux, alpha, beta, M, N, m_tile, n_tile);
    }
}

template <bool TRANSA, bool TRANSB, typename T, GemmEpilogue EPILOGUE, bool ACCUMULATE, typename OutT, typename AuxT,
          int BLOCK_ROW = 256, int BLOCK_COL = 256>
static void launch_gemm(const Bf16GemmArgs &args, void *d, int splits, bool mcast) {
    constexpr bool BIG = BLOCK_ROW == 256 && BLOCK_COL == 256;
    constexpr int LDS = lds_bytes(BLOCK_ROW, BLOCK_COL);
    const bool mc = BIG && mcast;
    const auto kernel = mc ? bf16_gemm_kernel<TRANSA, TRANSB, BIG, T, EPILOGUE, ACCUMULATE, OutT, AuxT, BLOCK_ROW,
                                              BLOCK_COL>
                           : bf16_gemm_kernel<TRANSA, TRANSB, false, T, EPILOGUE, ACCUMULATE, OutT, AuxT, BLOCK_ROW,
                                              BLOCK_COL>;
    hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, LDS);
    const kittens::gl<T, -1, -1, -1, -1> ga(static_cast<T *>(const_cast<void *>(args.A)), 1, 1,
                                            TRANSA ? args.M : args.K, TRANSA ? args.K : args.M);
    const kittens::gl<T, -1, -1, -1, -1> gb(static_cast<T *>(const_cast<void *>(args.B)), 1, 1,
                                            TRANSB ? args.K : args.N, TRANSB ? args.N : args.K);
    const int cluster = mc ? CLUSTER_DIM : 1;
    const dim3 grid(kittens::ceil_div(kittens::ceil_div(args.M, BLOCK_ROW), cluster) * cluster,
                    kittens::ceil_div(kittens::ceil_div(args.N, BLOCK_COL), cluster) * cluster, splits);
    hipLaunchKernelGGL(kernel, grid, dim3(wg_threads(BLOCK_ROW, BLOCK_COL)), LDS, args.stream, ga, gb,
                       static_cast<OutT *>(d), static_cast<AuxT *>(const_cast<void *>(args.bias)),
                       static_cast<AuxT *>(args.aux_gelu), args.alpha, args.beta, args.M, args.N, args.K);
}

template <bool TRANSA, bool TRANSB, typename T, int BM, int BN>
static void dispatch_default(const Bf16GemmArgs &args, const GemmPlan &plan) {
    const bool accumulate = args.beta != 0.f;
    const bool f32        = args.out_dtype == KITTENS_FLOAT32;
    if (plan.splits > 1) {
        launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::SPLIT_K, false, float, float, BM, BN>(
            args, plan.workspace, plan.splits, plan.mcast);
        if (f32) {
            launch_reduce<float>(args, plan.workspace, plan.splits);
        } else {
            launch_reduce<in_t<T>>(args, plan.workspace, plan.splits);
        }
        return;
    }
    KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
        KITTENS_BOOL_SWITCH(f32, F32,
            using OutT = std::conditional_t<F32, float, in_t<T>>;
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::DEFAULT, ACCUMULATE, OutT, OutT, BM, BN>(
                args, args.C, 1, plan.mcast);
    )) // NOLINT(*)
}

template <bool TRANSA, bool TRANSB, typename T>
static bool dispatch_gemm(const Bf16GemmArgs &args, const GemmPlan &plan) {
    const int in_dtype = std::is_same_v<T, kittens::half> ? KITTENS_FLOAT16 : KITTENS_BFLOAT16;
    const bool accumulate = args.beta != 0.f;
    const bool f32        = args.out_dtype == KITTENS_FLOAT32;
    if (!f32 && args.out_dtype != in_dtype) {
        return false;
    }
    switch (plan.epilogue) {
    case GemmEpilogue::DEFAULT:
        if (plan.bm == 128 && plan.bn == 128) {
            dispatch_default<TRANSA, TRANSB, T, 128, 128>(args, plan);
        } else if (plan.bm == 128) {
            dispatch_default<TRANSA, TRANSB, T, 128, 256>(args, plan);
        } else if (plan.bn == 128) {
            dispatch_default<TRANSA, TRANSB, T, 256, 128>(args, plan);
        } else {
            dispatch_default<TRANSA, TRANSB, T, 256, 256>(args, plan);
        }
        return true;
    case GemmEpilogue::BIAS:
        if (args.bias_dtype != in_dtype || (accumulate && (f32 || !TRANSA))) {
            return false;
        }
        if (f32) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, false, float, in_t<T>>(args, args.C, 1, plan.mcast);
        } else if (!accumulate) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, false, in_t<T>, in_t<T>>(args, args.C, 1, plan.mcast);
        } else if constexpr (TRANSA) {
            launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BIAS, true, in_t<T>, in_t<T>>(args, args.C, 1, plan.mcast);
        }
        return true;
    case GemmEpilogue::GELU_AUX:
        if constexpr (TRANSA) {
            if (args.aux_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::GELU_AUX, false, in_t<T>, in_t<T>>(
                    args, args.C, 1, plan.mcast);
                return true;
            }
        }
        return false;
    case GemmEpilogue::GELU_AUX_BIAS:
        if constexpr (TRANSA) {
            if (args.bias_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::GELU_AUX_BIAS, false, in_t<T>, in_t<T>>(
                    args, args.C, 1, plan.mcast);
                return true;
            }
        }
        return false;
    case GemmEpilogue::DGELU:
        if constexpr (!TRANSA && !TRANSB) {
            if (args.aux_dtype == in_dtype && !accumulate && !f32) {
                launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::DGELU, false, in_t<T>, in_t<T>>(
                    args, args.C, 1, plan.mcast);
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
                    using OutT = std::conditional_t<F32, float, in_t<T>>;
                    launch_gemm<TRANSA, TRANSB, T, GemmEpilogue::BGRADB, ACCUMULATE, OutT, OutT>(
                        args, args.C, 1, plan.mcast);
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

        GemmPlan plan;
        if (!plan_gemm(args, TileModel{0.6, 0.34, 5.0, 21.0, 2.0}, BLOCK_K, &plan)) {
            return false;
        }
        const bool fp16 = args.a_dtype == KITTENS_FLOAT16;
        if (args.transa && !args.transb) {
            return fp16 ? dispatch_gemm<true, false, kittens::half>(args, plan)
                        : dispatch_gemm<true, false, kittens::bf16>(args, plan);
        }
        if (!args.transa && args.transb) {
            return fp16 ? dispatch_gemm<false, true, kittens::half>(args, plan)
                        : dispatch_gemm<false, true, kittens::bf16>(args, plan);
        }
        return fp16 ? dispatch_gemm<false, false, kittens::half>(args, plan)
                    : dispatch_gemm<false, false, kittens::bf16>(args, plan);
    }
};

}  // namespace te_kittens::cdna5::bf16

Bf16GemmBackend *Bf16GemmBackend::get_cdna5() {
    static te_kittens::cdna5::bf16::Bf16GemmCdna5 impl;
    return &impl;
}
