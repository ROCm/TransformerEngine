/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/

// Grouped TN (fprop): D_e [m_e][N] = alpha X_e W_e^T (+ beta D_e) (+ bias_e), X_e [m_e][K], W_e [N][K].

#include "hip/hip_runtime.h"
#include "grouped16_common.cuh"

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

template <typename T, typename O, bool BIAS, bool ACC>
__global__ __launch_bounds__(NUM_THREADS, 2)
void tn_kernel(const Table t, int ne, int N, int K, float alpha, float beta, int *ctr) {
    using ST_A = st<T, HALF_ROW, K_STEP, st_16x32_s>;
    using ST_B = st<T, HALF_COL, K_STEP, st_16x32_s>;
    using RT_A = rt<T, REG_M, K_STEP, row_l, rt_16x32_s>;
    using RT_B = rt<T, REG_N, K_STEP, row_l, rt_16x32_s>;
    using RT_C = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
    using R    = typename Elem<T>::raw;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ int s_unit;
    constexpr int SZA = sizeof(ST_A);  // As[p][q] = As[0][0] + (2p + q) SZA: the fragment loads' immediate
    constexpr int SZB = sizeof(ST_B);
    static_assert(std::is_same_v<ST_A, ST_B>, "B's lane offsets are A's");

    const int wid = warpid() % NUM_WARPS;
    const uintptr_t lds_warp_off = wid * 16 * kittens::WARP_THREADS;
    const uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    const uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    const uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + lds_warp_off));
    const uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + lds_warp_off));
    const uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + lds_warp_off));
    const uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + lds_warp_off));
    uint32_t sw_A[2], sw_A1[2], sw_B[2];

    const int units   = t.unit_off[ne];
    const int w_tiles = (N + BLOCK_COL - 1) / BLOCK_COL;
#pragma unroll 1
    for (;;) {
        const int u = claim(ctr, units, &s_unit);
        if (u >= units) {
            break;
        }
        const int e    = grouped_sched::unit_expert<MAX_EXPERTS>(t.unit_off, ne, u);
        const int rows = t.rows[e];
        int block_col, block_row;
        grouped_sched::unit_tile(u - t.unit_off[e], w_tiles, (rows + BLOCK_ROW - 1) / BLOCK_ROW, block_col, block_row);
        const int rows_left = min(rows - block_row * BLOCK_ROW, BLOCK_ROW);
        // The pipeline retires k-steps in pairs; for an odd k_real the extra step loads data past K (never read)
        // and the last drain phase keeps only its barriers.
        const int k_real    = K / K_STEP;
        const int k_tiles   = k_real + (k_real & 1);
        const uint32_t row_bytes = (uint32_t)K * sizeof(T);
        const R *a_base = fs_uniform_ptr((const R *)t.b[e] + (size_t)block_row * BLOCK_ROW * K);
        const R *b_base = fs_uniform_ptr((const R *)t.a[e]);
        const i32x4 a_srd = fs_srd(a_base, (uint32_t)rows_left * row_bytes);
        const i32x4 b_srd = fs_srd(b_base, (uint32_t)N * row_bytes);

        const int ft     = fs_tid();
        const int warp_m = ft / (kittens::WARP_THREADS * WARPS_COL), warp_n = ft / kittens::WARP_THREADS % WARPS_COL;
        const int lane   = ft % kittens::WARP_THREADS;
        tile_lane_offsets<ST_A>(lane, ft / kittens::WARP_THREADS, row_bytes, sw_A);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            sw_B[i]  = sw_A[i];
            sw_A1[i] = sw_A[i] + HALF_ROW * row_bytes;
        }
#define TN_A(KT) ((uint32_t)(KT) * (K_STEP * (int)sizeof(T)))
#define TN_B(J, KT) ((uint32_t)((block_col * 2 + (J)) * HALF_COL) * row_bytes + TN_A(KT))

        const auto sa0 = subtile_inplace<REG_M, K_STEP>(As[0][0], {warp_m, 0});
        const auto sb0 = subtile_inplace<REG_N, K_STEP>(Bs[0][0], {warp_n, 0});
        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        zero(c00);
        zero(c01);
        zero(c10);
        zero(c11);

        dma2(sw_B, b_srd, TN_B(0, 0), b_lds_00);
        dma2(sw_A, a_srd, TN_A(0), a_lds_00);
        dma2(sw_B, b_srd, TN_B(1, 0), b_lds_01);
        dma2(sw_A1, a_srd, TN_A(0), a_lds_01);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        dma2(sw_B, b_srd, TN_B(0, 1), b_lds_10);
        dma2(sw_A, a_srd, TN_A(1), a_lds_10);
        dma2(sw_B, b_srd, TN_B(1, 1), b_lds_11);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        load_frag_rows<0>(b_tile_0, sb0, lane);

#pragma unroll 1
        for (int tile = 0; tile < k_tiles - 2; tile += 2) {
            load_frag_rows<0>(a_tile, sa0, lane);
            dma2(sw_A1, a_srd, TN_A(tile + 1), a_lds_11);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c00, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<SZB>(b_tile_1, sb0, lane);
            dma2(sw_B, b_srd, TN_B(0, tile + 2), b_lds_00);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c01, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZA>(a_tile, sa0, lane);
            dma2(sw_A, a_srd, TN_A(tile + 2), a_lds_00);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c10, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<2 * SZB>(b_tile_0, sb0, lane);
            dma2(sw_B, b_srd, TN_B(1, tile + 2), b_lds_01);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c11, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<2 * SZA>(a_tile, sa0, lane);
            dma2(sw_A1, a_srd, TN_A(tile + 2), a_lds_01);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c00, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<3 * SZB>(b_tile_1, sb0, lane);
            dma2(sw_B, b_srd, TN_B(0, tile + 3), b_lds_10);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c01, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZA>(a_tile, sa0, lane);
            dma2(sw_A, a_srd, TN_A(tile + 3), a_lds_10);
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");  // P3's Bs[0][0] DMA, read in P15
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c10, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<0>(b_tile_0, sb0, lane);
            dma2(sw_B, b_srd, TN_B(1, tile + 3), b_lds_11);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c11, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            const int tile = k_tiles - 2;
            load_frag_rows<0>(b_tile_0, sb0, lane);
            load_frag_rows<0>(a_tile, sa0, lane);
            dma2(sw_A1, a_srd, TN_A(tile + 1), a_lds_11);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c00, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZB>(b_tile_1, sb0, lane);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c01, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZA>(a_tile, sa0, lane);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c10, a_tile, b_tile_0);
            mma_ABt_T(c11, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (k_real == k_tiles) {
            load_frag_rows<2 * SZB>(b_tile_0, sb0, lane);
            load_frag_rows<2 * SZA>(a_tile, sa0, lane);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c00, a_tile, b_tile_0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZB>(b_tile_1, sb0, lane);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c01, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZA>(a_tile, sa0, lane);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_T(c10, a_tile, b_tile_0);
            mma_ABt_T(c11, a_tile, b_tile_1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        } else {
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
        }
#undef TN_A
#undef TN_B

        const int fe = fs_tid();
        const int we = fe / kittens::WARP_THREADS;
        if (we / WARPS_COL == 0) {
            __builtin_amdgcn_s_barrier();
        }
        const int rf0 = __builtin_amdgcn_readfirstlane(we / WARPS_COL);
        const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + we / WARPS_COL);
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + we % WARPS_COL);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + we % WARPS_COL);
        int lane_epi = fe % kittens::WARP_THREADS;
        asm volatile("" : "+v"(lane_epi));
        const O *d_base = fs_uniform_ptr((const O *)t.d[e] + (size_t)block_row * BLOCK_ROW * N);
        const i32x4 d_srd    = fs_srd(d_base, (uint32_t)rows_left * (uint32_t)N * sizeof(O));
        const i32x4 bias_srd = BIAS ? fs_srd(t.bias[e], (uint32_t)N * sizeof(R)) : i32x4{};
        epi_t16<O, BIAS, ACC, R>(d_srd, c00, rf0 * REG_M, cf0 * REG_N, N, lane_epi, alpha, beta, bias_srd);
        epi_t16<O, BIAS, ACC, R>(d_srd, c01, rf0 * REG_M, cf1 * REG_N, N, lane_epi, alpha, beta, bias_srd);
        epi_t16<O, BIAS, ACC, R>(d_srd, c10, rf1 * REG_M, cf0 * REG_N, N, lane_epi, alpha, beta, bias_srd);
        epi_t16<O, BIAS, ACC, R>(d_srd, c11, rf1 * REG_M, cf1 * REG_N, N, lane_epi, alpha, beta, bias_srd);
    }
}

template <typename T, typename O, bool BIAS, bool ACC>
static void run(const Launch &l) {
    const auto kernel = tn_kernel<T, O, BIAS, ACC>;
    for (int i = 0; i < l.num_tables; i++) {
        const int units = l.tables[i].unit_off[l.experts_per_table[i]];
        hk_check(hipMemsetAsync(l.ctr, 0, XCDS * sizeof(int), l.stream), "grouped16 tn counters");
        hipLaunchKernelGGL(kernel, dim3(min(units, l.grid_cap)), dim3(NUM_THREADS), 0, l.stream, l.tables[i],
                           l.experts_per_table[i], l.N, l.K, l.alpha, l.beta, l.ctr);
        hk_check(hipGetLastError(), "grouped16 tn_kernel");
    }
}

template <typename T>
static bool run_t(const Launch &l, bool f32, bool bias, bool acc) {
    using R = typename Elem<T>::raw;
    if (f32) {
        bias ? (acc ? run<T, float, true, true>(l) : run<T, float, true, false>(l))
             : (acc ? run<T, float, false, true>(l) : run<T, float, false, false>(l));
    } else {
        bias ? (acc ? run<T, R, true, true>(l) : run<T, R, true, false>(l))
             : (acc ? run<T, R, false, true>(l) : run<T, R, false, false>(l));
    }
    return true;
}

bool launch_tn(const Launch &l, int in_dtype, int out_dtype, bool bias, bool acc) {
    const bool f32 = out_dtype == KITTENS_FLOAT32;
    return in_dtype == KITTENS_BFLOAT16 ? run_t<bf16>(l, f32, bias, acc) : run_t<half>(l, f32, bias, acc);
}

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
