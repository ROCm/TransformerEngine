/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// gfx950 (CDNA4) blockwise FP8 GEMM (TN), 1Dx2D scaling.
// Based on the HipKittens main-branch 8-wave native FP8 GEMM
// (kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu): 256x256 tile, 8 warps (2x4),
// 4 accumulators, double-buffered ping-pong. Each K-block is one K=128 MFMA per
// output, so blockwise scale is applied per K-block: compute an unscaled partial
// (mma_ABt_scaled with unit e8m0) then acc += partial * (scale_A[m] * scale_B[tile]).
//
// Scale roles match the cdna3 kernel's 1Dx2D path: scale_A is per-row over the
// output M dimension (one FP32 per row), scale_B is a per-(M/N-tile,k) scalar.

#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"

using namespace kittens;

namespace gfx950_blockwise {

constexpr int NUM_WARPS  = 8;
constexpr int WARPS_COL  = 4;
constexpr int WARPS_ROW  = 2;
constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;   // 64
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;   // 32
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int SCALE_BLOCK = 128;

using gl_fp8  = gl<fp8e4m3, 1, 1, -1, -1>;
using gl_bf16 = gl<bf16,    1, 1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#include "blockwise_fp8_gemm_device.cuh"

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_blockwise_gemm_kernel(
    const gl_fp8 A, const gl_fp8 B, const gl_bf16 C,
    const float *__restrict__ scale_A, const float *__restrict__ scale_B,
    int M, int N, int K) {
#if defined(__gfx950__)
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / SCALE_BLOCK;
    const int blocks_per_col = (N + BLOCK_COL - 1) / BLOCK_COL;

    using ST_A = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ float smem_sa[2][BLOCK_ROW];  // activation per-row scale, double-buffered
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    RT_C p;

    const int global_block_id = blockIdx.x;
    const int block_row = global_block_id / blocks_per_col;
    const int block_col = global_block_id % blocks_per_col;

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
    const int tid = threadIdx.x;

    int tic = 0, toc = 1;

    using T = fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int memcpy_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int memcpy_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[memcpy_A], sw_B[memcpy_B];
    G::prefill_swizzled_offsets(As[tic][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, sw_B);

    // Full-matrix SRDs so out-of-bounds rows (partial M/N edge tiles) auto-zero
    // on the hardware buffer load. K stays a multiple of BLOCK_K.
    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
            b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        }

    const fp8e8m0_4 unit = 0x7F7F7F7Fu;

    // scale_A = per-row over M (one FP32 per output row); scale_B = per-tile
    // scalar per N-block. Read from global directly per K-block. N-block index is
    // clamped to the last valid block for partial-N tiles (those outputs are
    // masked out at store, so the clamped scale is harmless).
    const int n_scale_blocks = (N + SCALE_BLOCK - 1) / SCALE_BLOCK;
    const int nb0 = min(block_col * 2 + 0, n_scale_blocks - 1);
    const int nb1 = min(block_col * 2 + 1, n_scale_blocks - 1);
    const float *sa_row = scale_A + block_row * BLOCK_ROW;
    const float *sb0 = scale_B + nb0 * scale_K;
    const float *sb1 = scale_B + nb1 * scale_K;
    const int local_m0 = warp_m * REG_M;
    const int local_m1 = HALF_ROW + warp_m * REG_M;
    const int m_valid = M - block_row * BLOCK_ROW;  // valid local rows in this tile

    zero(cA); zero(cB); zero(cC); zero(cD);

    // Scale staging is register-hop (global->VGPR->ds_write) like CDNA3, not a
    // direct global->LDS DMA. The VGPR loads sit where the old scale DMAs were so
    // the tuned A/B vmcnt(4)/vmcnt(6) waits keep their meaning; the ds_write into
    // LDS rides lgkmcnt and never perturbs the A/B tile vmcnt accounting. Loads
    // are issued on all 8 waves uniformly (tid%BLOCK_ROW) to avoid wave imbalance.
    const int sa_tid_p = tid % BLOCK_ROW;

    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A, a_srd, a_base, a_lds[tic][0]);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds[tic][1]);
    const float sa0_reg = sa_tid_p < m_valid ? sa_row[0 * M + sa_tid_p] : 0.f;

    if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
    }

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A, a_srd, a_base, a_lds[toc][0]);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
    const float sa1_reg = (k_iters > 1 && sa_tid_p < m_valid) ? sa_row[1 * M + sa_tid_p] : 0.f;

    asm volatile("s_waitcnt vmcnt(6)");
    // At vmcnt(6) the small scale VGPR loads have long since landed. Commit them
    // to both LDS slots via ds_write.
    if (tid < BLOCK_ROW) {
        smem_sa[tic][tid] = sa0_reg;
        if (k_iters > 1) smem_sa[toc][tid] = sa1_reg;
    }
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        const float sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
        const float sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));

        // Prefetch NEXT K-block's (k+1) activation scale (global->VGPR, all 8
        // waves, vmcnt). ds_write lands in the OTHER slot (toc) — the one iter k+1
        // will read — so it never overwrites the slot being read this iter. This
        // mirrors CDNA3: read slot != write slot, which is required because the
        // 8-wave ping-pong s_barrier is a per-group (offset) barrier, not a full
        // block barrier, so same-slot write-after-read would race the other group.
        const int sa_tid = tid % BLOCK_ROW;
        const float sa_next = sa_tid < m_valid ? sa_row[(k + 1) * M + sa_tid] : 0.f;

        auto bs0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs0)>(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as0)>(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        // Read this K-block's scale (staged in the previous iter) into registers.
        const auto rs0 = load_row_scale_lds<RT_C>(smem_sa[tic], local_m0);
        const auto rs1 = load_row_scale_lds<RT_C>(smem_sa[tic], local_m1);
        // Stage k+1 into the toc slot (different from the tic slot read above).
        if (tid < BLOCK_ROW) smem_sa[toc][tid] = sa_next;
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs1)>(b1, bs1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as1)>(a, as1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cC, p, rs1, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cD, p, rs1, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        // stage k+2 activation scale into this (tic) slot only after all four
        __builtin_amdgcn_s_barrier();
    }

    {
        const int k = k_iters - 2;
        const float sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
        const float sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        const float *sa_k = sa_row + k * M;
        const auto rs0 = load_row_scale<RT_C>(sa_k, local_m0, m_valid);
        const auto rs1 = load_row_scale<RT_C>(sa_k, local_m1, m_valid);

        auto bs0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs0)>(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as0)>(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs1)>(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as1)>(a, as1);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cC, p, rs1, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs0)>(b0, bs0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cD, p, rs1, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1;
    }

    {
        const int k = k_iters - 1;
        const float sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
        const float sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        const float *sa_k = sa_row + k * M;
        const auto rs0 = load_row_scale<RT_C>(sa_k, local_m0, m_valid);
        const auto rs1 = load_row_scale<RT_C>(sa_k, local_m1, m_valid);

        auto as0 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as0)>(a, as0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs1)>(b1, bs1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as1)>(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b0, p, &unit, &unit);
        scale_accumulate(cC, p, rs1, sb0_k);
        zero(p); mma_ABt_scaled<0, 0>(p, a, b1, p, &unit, &unit);
        scale_accumulate(cD, p, rs1, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
    }

    apply_rtne_bias(cA); apply_rtne_bias(cB); apply_rtne_bias(cC); apply_rtne_bias(cD);
    // Global element origins for each accumulator. cA/cB = first M-half, cC/cD =
    // second (+128 rows); cA/cC = first N-half, cB/cD = second (+128 cols).
    const int m_off0 = block_row * BLOCK_ROW + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_ROW + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_COL + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_COL + HALF_COL + warp_n * REG_N;
    bf16 *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;
    // Full 256x256 output tile in-bounds: use the fast library store. Edge tiles
    // (partial M/N) fall back to a bounds-checked store.
    const bool full = (block_row + 1) * BLOCK_ROW <= M && (block_col + 1) * BLOCK_COL <= N;
    if (full) {
        store(C, cA, {0, 0, ca, cn0});
        store(C, cB, {0, 0, ca, cn1});
        store(C, cC, {0, 0, cc, cn0});
        store(C, cD, {0, 0, cc, cn1});
    } else {
        store_masked(c_ptr, cA, m_off0, n_off0, M, N);
        store_masked(c_ptr, cB, m_off0, n_off1, M, N);
        store_masked(c_ptr, cC, m_off1, n_off0, M, N);
        store_masked(c_ptr, cD, m_off1, n_off1, M, N);
    }
#endif  // __gfx950__
}

}  // namespace gfx950_blockwise

bool kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    bool has_bias, bool has_gelu, bool has_beta,
    hipStream_t stream) {
    // Dispatch (rocm_gemm.cu, gfx950) now passes canonical A/B/M/N without the
    // cuBLAS swap: A = weight (2D per-tile), B = activation (1D per-row), M/N =
    // user M/N. The kernel body is written for the swapped convention
    // (kernel-A = activation/1D, kernel-B = weight/2D, kernel-M = N_user), so we
    // swap the operands back here: (A,B)->(B,A), (M,N)->(N,M), scales/modes/dtypes
    // likewise. This keeps the verified kernel body unchanged while the dispatch
    // stays mxfp8-style.
    const void *kA = B,        *kB = A;
    const void *ksa = scale_B,  *ksb = scale_A;
    const int   kM = N,         kN = M;
    const int   ka_mode = b_scaling_mode, kb_mode = a_scaling_mode;
    const int   ka_dtype = b_dtype,       kb_dtype = a_dtype;

    const bool is_1d2d = (ka_mode == KITTENS_BLOCK_SCALING_1D) &&
                         (kb_mode == KITTENS_BLOCK_SCALING_2D);
    if (!is_1d2d) return false;
    if (ka_dtype != KITTENS_FP8E4M3 || kb_dtype != KITTENS_FP8E4M3) return false;
    if (out_dtype != KITTENS_BFLOAT16) return false;
    if (has_bias || has_gelu || has_beta) return false;
    using gfx950_blockwise::BLOCK_ROW;
    using gfx950_blockwise::BLOCK_COL;
    using gfx950_blockwise::BLOCK_K;
    if (K % BLOCK_K != 0) return false;  // partial-K not yet supported
    if (K < 2 * BLOCK_K) return false;
    // M/N may be arbitrary (partial edge tiles handled via SRD-zeroed loads +
    // masked store).

    const float *sa = reinterpret_cast<const float *>(ksa);
    const float *sb = reinterpret_cast<const float *>(ksb);
    gfx950_blockwise::gl_fp8 A_gl((fp8e4m3 *)const_cast<void *>(kA), nullptr, nullptr, kM, K);
    gfx950_blockwise::gl_fp8 B_gl((fp8e4m3 *)const_cast<void *>(kB), nullptr, nullptr, kN, K);
    gfx950_blockwise::gl_bf16 C_gl((bf16 *)C, nullptr, nullptr, kM, kN);

    const int grid = ((kM + BLOCK_ROW - 1) / BLOCK_ROW) * ((kN + BLOCK_COL - 1) / BLOCK_COL);
    gfx950_blockwise::fp8_blockwise_gemm_kernel<<<grid, gfx950_blockwise::NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, sa, sb, kM, kN, K);
    return true;
}
