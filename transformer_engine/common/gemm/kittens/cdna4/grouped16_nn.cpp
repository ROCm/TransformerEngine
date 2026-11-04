/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/

// Grouped NN (dgrad): D_e [m_e][K] = alpha dY_e W_e (+ beta D_e), dY_e [m_e][N], W_e [N][K].

#include "hip/hip_runtime.h"
#include "grouped16_common.cuh"

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

template <typename T, typename O, bool ACC>
__global__ __launch_bounds__(NUM_THREADS, 2)
void nn_kernel(const Table t, int ne, int Nw, int Kw, float alpha, float beta, int *ctr) {
    using ST_A = st<T, HALF_ROW, K_STEP, st_16x32_s>;
    using RT_A = rt<T, REG_M, K_STEP, row_l, rt_16x32_s>;
    using RT_B = rt<T, REG_N, K_STEP, col_l, rt_16x32_s>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
    using R    = typename Elem<T>::raw;
    constexpr int B_HALF = HALF_COL * K_STEP;
    __shared__ alignas(16) unsigned char s_tiles[LDS_TILES];
    __shared__ int s_unit;
    static_assert(4 * sizeof(ST_A) + 4 * B_HALF * sizeof(T) == LDS_TILES, "NN tiles must fill the region");
    ST_A(&As)[2][2] = *reinterpret_cast<ST_A(*)[2][2]>(s_tiles);
    T(&Bs)[2][2][B_HALF] = *reinterpret_cast<T(*)[2][2][B_HALF]>(s_tiles + 4 * sizeof(ST_A));
    constexpr int B_INSTS       = (HALF_COL * K_STEP * (int)sizeof(T)) / (16 * NUM_THREADS);
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;

    const int wid0 = warpid() % NUM_WARPS;
    const uintptr_t lds_warp_off = wid0 * elem_per_warp * sizeof(T);
    const uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    const uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    uint32_t b_lds_wo = __builtin_amdgcn_readfirstlane((uint32_t)lds_warp_off);
    asm volatile("" : "+s"(b_lds_wo));
    uint32_t sw_A[2], sw_A1[2];

    const int units   = t.unit_off[ne];
    const int w_tiles = (Kw + BLOCK_COL - 1) / BLOCK_COL;
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
        const int n_cols    = Kw;
        // The pipeline retires k-steps in pairs; for an odd k_real the extra step loads data past N (never read)
        // and the last drain phase keeps only its barriers.
        const int k_real    = Nw / K_STEP;
        const int k_tiles   = k_real + (k_real & 1);
        const uint32_t a_row_bytes = (uint32_t)Nw * sizeof(T);
        const R *a_base = fs_uniform_ptr((const R *)t.b[e] + (size_t)block_row * BLOCK_ROW * Nw);
        const R *b_base = fs_uniform_ptr((const R *)t.a[e]);
        const i32x4 a_srd = fs_srd(a_base, (uint32_t)rows_left * a_row_bytes);
        const i32x4 b_srd = fs_srd(b_base, (uint32_t)Nw * (uint32_t)Kw * sizeof(T));

        int lane_o = (int)threadIdx.x & 63;
        asm volatile("" : "+v"(lane_o));
        int tid_o = (int)threadIdx.x;
        asm volatile("" : "+v"(tid_o));
        const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
        const int warp_m = wv_o / WARPS_COL;
        const int warp_n = wv_o % WARPS_COL;
        const int wid    = wv_o % NUM_WARPS;
        tile_lane_offsets<ST_A>(lane_o, wv_o, a_row_bytes, sw_A);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            sw_A1[i] = sw_A[i] + HALF_ROW * a_row_bytes;
        }
        const int lane_b = lane_o;
        uint32_t b_lane_off[B_INSTS];
#pragma unroll
        for (int i = 0; i < B_INSTS; i++) {
            const int q  = (wid + i * NUM_WARPS) * 64 + lane_o;
            const int W  = q / 32, within = q % 32;
            const int nb = W / 4, w = W % 4;
            const int r  = within / 2, c0 = (within % 2) * 8;
            const int n  = nb * 16 + c0;
            const int kk = 32 * (w / 2) + 8 * (r / 4) + (r % 4) + 4 * (w % 2);
            b_lane_off[i] = (uint32_t)((kk * n_cols + n) * (int)sizeof(T));
        }
#include "grouped16_nn_kloop.inc"
        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(warp_n);
        int lane_epi = kittens::laneid();
        asm volatile("" : "+v"(lane_epi));
        const O *d_base =
            fs_uniform_ptr((const O *)t.d[e] + (size_t)block_row * BLOCK_ROW * Kw + block_col * BLOCK_COL);
        const i32x4 d_srd = fs_srd(d_base, (uint32_t)rows_left * (uint32_t)Kw * sizeof(O));
        epi_staged<O, ACC>(s_tiles, c00, c01, c10, c11, rf0, cf0, lane_epi, d_srd, Kw, Kw - block_col * BLOCK_COL,
                           alpha, beta);
    }
}

template <typename T, typename O, bool ACC>
static void run(const Launch &l) {
    const auto kernel = nn_kernel<T, O, ACC>;
    for (int i = 0; i < l.num_tables; i++) {
        const int units = l.tables[i].unit_off[l.experts_per_table[i]];
        hk_check(hipMemsetAsync(l.ctr, 0, XCDS * sizeof(int), l.stream), "grouped16 nn counters");
        hipLaunchKernelGGL(kernel, dim3(min(units, l.grid_cap)), dim3(NUM_THREADS), 0, l.stream, l.tables[i],
                           l.experts_per_table[i], l.N, l.K, l.alpha, l.beta, l.ctr);
        hk_check(hipGetLastError(), "grouped16 nn_kernel");
    }
}

template <typename T>
static bool run_t(const Launch &l, bool f32, bool acc) {
    using R = typename Elem<T>::raw;
    if (f32) {
        acc ? run<T, float, true>(l) : run<T, float, false>(l);
    } else {
        acc ? run<T, R, true>(l) : run<T, R, false>(l);
    }
    return true;
}

bool launch_nn(const Launch &l, int in_dtype, int out_dtype, bool acc) {
    const bool f32 = out_dtype == KITTENS_FLOAT32;
    return in_dtype == KITTENS_BFLOAT16 ? run_t<bf16>(l, f32, acc) : run_t<half>(l, f32, acc);
}

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
