/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"


namespace blockwise_gfx950 {

#include "blockwise_fp8_gemm_device.cuh"

constexpr int NUM_WARPS  = 8;
constexpr int WARPS_COL  = 4;
constexpr int WARPS_ROW  = 2;
constexpr int BLOCK_ROW  = 128;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;   // 64
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;   // 32
constexpr int SCALE_BLOCK = 128;
constexpr int MFMA_K      = 128;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

using gl_fp8  = kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1>;
using gl_bf16 = kittens::gl<kittens::bf16,    1, 1, -1, -1>;

using G = kittens::group<NUM_WARPS>;


__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_blockwise_gemm_kernel(
    const gl_fp8 A, const gl_fp8 B, const gl_bf16 C,
    const float *__restrict__ scale_A, const float *__restrict__ scale_B,
    int M, int N, int K) {
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / SCALE_BLOCK;
    const int blocks_per_col = (N + BLOCK_COL - 1) / BLOCK_COL;

    using ST_A = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A = kittens::rt_fp8e4m3<REG_M, MFMA_K>;
    using RT_B = kittens::rt_fp8e4m3<REG_N, MFMA_K>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ float smem_sa[2][BLOCK_ROW];
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    RT_C p;

    const int global_block_id = blockIdx.x;
    const int block_row = global_block_id / blocks_per_col;
    const int block_col = global_block_id % blocks_per_col;

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int tid = threadIdx.x;

    int tic = 0, toc = 1;

    using T = kittens::fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int memcpy_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int memcpy_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[memcpy_A], sw_B[memcpy_B];
    G::prefill_swizzled_offsets(As[tic][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, M * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
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

    const kittens::fp8e8m0_4 unit = 0x7F7F7F7Fu;

    const int n_scale_blocks = (N + SCALE_BLOCK - 1) / SCALE_BLOCK;
    const int nb0 = min(block_col * 2 + 0, n_scale_blocks - 1);
    const int nb1 = min(block_col * 2 + 1, n_scale_blocks - 1);
    const float *sa_row = scale_A + block_row * BLOCK_ROW;
    const float *sb0 = scale_B + nb0 * scale_K;
    const float *sb1 = scale_B + nb1 * scale_K;
    const int local_m0 = warp_m * REG_M;
    const int local_m1 = HALF_ROW + warp_m * REG_M;
    const int m_valid = M - block_row * BLOCK_ROW;

    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

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
    if (tid < BLOCK_ROW) {
        smem_sa[tic][tid] = sa0_reg;
        if (k_iters > 1) smem_sa[toc][tid] = sa1_reg;
    }
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        const float sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
        const float sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));

        const int sa_tid = tid % BLOCK_ROW;
        const float sa_next = sa_tid < m_valid ? sa_row[(k + 1) * M + sa_tid] : 0.f;

        auto bs0 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][0], {warp_n, 0});
        kittens::load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][0], {warp_m, 0});
        kittens::load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();


        const auto rs0 = load_row_scale_lds<RT_C>(smem_sa[tic], local_m0);
        const auto rs1 = load_row_scale_lds<RT_C>(smem_sa[tic], local_m1);

        if (tid < BLOCK_ROW) smem_sa[toc][tid] = sa_next;
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b1, p);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cC, p, rs1, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b1, p);
        scale_accumulate(cD, p, rs1, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        const int k = k_iters - 2;
        const float sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
        const float sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        const float *sa_k = sa_row + k * M;
        const auto rs0 = load_row_scale<RT_C>(sa_k, local_m0, m_valid);
        const auto rs1 = load_row_scale<RT_C>(sa_k, local_m1, m_valid);

        auto bs0 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][0], {warp_n, 0});
        kittens::load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][0], {warp_m, 0});
        kittens::load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b1, p);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cC, p, rs1, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs0 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[toc][0], {warp_n, 0});
        kittens::load(b0, bs0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b1, p);
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

        auto as0 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][0], {warp_m, 0});
        kittens::load(a, as0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cA, p, rs0, sb0_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b1, p);
        scale_accumulate(cB, p, rs0, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt(p, a, b0, p);
        scale_accumulate(cC, p, rs1, sb0_k);
        kittens::zero(p); mma_ABt(p, a, b1, p);
        scale_accumulate(cD, p, rs1, sb1_k);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
    }

    apply_rtne_bias(cA); apply_rtne_bias(cB); apply_rtne_bias(cC); apply_rtne_bias(cD);

    const int m_off0 = block_row * BLOCK_ROW + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_ROW + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_COL + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_COL + HALF_COL + warp_n * REG_N;
    kittens::bf16 *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;
    
    const bool full = (block_row + 1) * BLOCK_ROW <= M && (block_col + 1) * BLOCK_COL <= N;
    if (full) {
        kittens::store(C, cA, {0, 0, ca, cn0});
        kittens::store(C, cB, {0, 0, ca, cn1});
        kittens::store(C, cC, {0, 0, cc, cn0});
        kittens::store(C, cD, {0, 0, cc, cn1});
    } else {
        store_masked(c_ptr, cA, m_off0, n_off0, M, N);
        store_masked(c_ptr, cB, m_off0, n_off1, M, N);
        store_masked(c_ptr, cC, m_off1, n_off0, M, N);
        store_masked(c_ptr, cD, m_off1, n_off1, M, N);
    }
}

}  // namespace blockwise_gfx950

bool kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    bool has_bias, bool has_gelu, bool has_beta,
    hipStream_t stream) {

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
    using blockwise_gfx950::BLOCK_ROW;
    using blockwise_gfx950::BLOCK_COL;
    using blockwise_gfx950::BLOCK_K;
    if (K % BLOCK_K != 0) return false;  // partial-K not yet supported
    if (K < 2 * BLOCK_K) return false;
    // M/N may be arbitrary (partial edge tiles handled via SRD-zeroed loads +
    // masked kittens::store).

    const float *sa = reinterpret_cast<const float *>(ksa);
    const float *sb = reinterpret_cast<const float *>(ksb);
    blockwise_gfx950::gl_fp8 A_gl((kittens::fp8e4m3 *)const_cast<void *>(kA), nullptr, nullptr, kM, K);
    blockwise_gfx950::gl_fp8 B_gl((kittens::fp8e4m3 *)const_cast<void *>(kB), nullptr, nullptr, kN, K);
    blockwise_gfx950::gl_bf16 C_gl((kittens::bf16 *)C, nullptr, nullptr, kM, kN);

    const int grid = ((kM + BLOCK_ROW - 1) / BLOCK_ROW) * ((kN + BLOCK_COL - 1) / BLOCK_COL);
    blockwise_gfx950::fp8_blockwise_gemm_kernel<<<grid, blockwise_gfx950::NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, sa, sb, kM, kN, K);
    return true;
}
