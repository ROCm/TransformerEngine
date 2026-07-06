/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <type_traits>
#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"


namespace blockwise_gfx950 {

#include "blockwise_fp8_gemm_device.cuh"

constexpr int NUM_WARPS   = 8;
constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int HALF_ROW    = BLOCK_M / 2;
constexpr int HALF_COL    = BLOCK_N / 2;
constexpr int REG_M       = BLOCK_M / WARPS_ROW / 2;
constexpr int REG_N       = BLOCK_N / WARPS_COL / 2;
constexpr int MFMA_K      = 128;
constexpr int SCALE_BLOCK = 128;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

template <typename T> using _gl_A_t = kittens::gl<T, -1, -1, -1, -1>;
template <typename T> using _gl_B_t = kittens::gl<T, -1, -1, -1, -1>;
template <typename OType> using _gl_C_t = kittens::gl<OType, -1, -1, -1, -1>;
using _gl_SA = kittens::gl<float, -1, -1, -1, -1>;
using _gl_SB = kittens::gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template <typename AType, typename BType, typename OType>
struct micro_globals {
    _gl_A_t<AType> a;
    _gl_B_t<BType> b;
    _gl_C_t<OType> c;
    _gl_SA scale_a;
    _gl_SB scale_b;
    const void *bias;
    int bias_dtype;
    const void *gelu_aux;
    int gelu_aux_dtype;
    const OType *c_in;
    float beta;
    hipStream_t stream;
    int M() const { return (int)c.rows(); }
    int N() const { return (int)c.cols(); }
    int K() const { return (int)a.cols(); }
    dim3 grid()  { return dim3(((M() + BLOCK_M - 1) / BLOCK_M) * ((N() + BLOCK_N - 1) / BLOCK_N)); }
    dim3 block() { return dim3(NUM_THREADS); }
};

template <typename OType, int CBSZ, int BLGP, bool IS_1D2D,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__device__ void micro_tk_body(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
    const auto A = g.a;
    const auto B = g.b;
    const auto C = g.c;
    const float *scale_A = g.scale_a.raw_ptr;
    const float *scale_B = g.scale_b.raw_ptr;
    const int M = (int)g.c.rows(), N = (int)g.c.cols(), K = (int)g.a.cols();
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / SCALE_BLOCK;
    const int blocks_per_col = (N + BLOCK_N - 1) / BLOCK_N;

    using ST_A = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A = kittens::rt_fp8e4m3<REG_M, MFMA_K>;
    using RT_B = kittens::rt_fp8e4m3<REG_N, MFMA_K>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ float smem_sa[2][BLOCK_M];
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
    const float *sa_row = scale_A + block_row * BLOCK_M;
    const float *sb0 = scale_B + nb0 * scale_K;
    const float *sb1 = scale_B + nb1 * scale_K;
    const int local_m0 = warp_m * REG_M;
    const int local_m1 = HALF_ROW + warp_m * REG_M;
    const int m_valid = M - block_row * BLOCK_M;

    const int local_n0 = block_col * BLOCK_N + warp_n * REG_N;
    const int local_n1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    const int sa_tid_p = tid % BLOCK_M;

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
    if (tid < BLOCK_M) {
        smem_sa[tic][tid] = sa0_reg;
        if (k_iters > 1) smem_sa[toc][tid] = sa1_reg;
    }
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        float sb0_k, sb1_k;
        ColScale<RT_C::width> cs0, cs1;
        if constexpr (IS_1D2D) {
            sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
            sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        } else {
            const float *sb_col = scale_B + k * N;
            cs0 = load_col_scale<RT_C>(sb_col, local_n0, N);
            cs1 = load_col_scale<RT_C>(sb_col, local_n1, N);
        }

        const int sa_tid = tid % BLOCK_M;
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

        if (tid < BLOCK_M) smem_sa[toc][tid] = sa_next;
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cA, p, rs0, sb0_k);
        else                   scale_accumulate_1d1d(cA, p, rs0, cs0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cB, p, rs0, sb1_k);
        else                   scale_accumulate_1d1d(cB, p, rs0, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cC, p, rs1, sb0_k);
        else                   scale_accumulate_1d1d(cC, p, rs1, cs0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cD, p, rs1, sb1_k);
        else                   scale_accumulate_1d1d(cD, p, rs1, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        const int k = k_iters - 2;
        float sb0_k, sb1_k;
        ColScale<RT_C::width> cs0, cs1;
        if constexpr (IS_1D2D) {
            sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
            sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        } else {
            const float *sb_col = scale_B + k * N;
            cs0 = load_col_scale<RT_C>(sb_col, local_n0, N);
            cs1 = load_col_scale<RT_C>(sb_col, local_n1, N);
        }
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
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cA, p, rs0, sb0_k);
        else                   scale_accumulate_1d1d(cA, p, rs0, cs0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cB, p, rs0, sb1_k);
        else                   scale_accumulate_1d1d(cB, p, rs0, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cC, p, rs1, sb0_k);
        else                   scale_accumulate_1d1d(cC, p, rs1, cs0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs0 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[toc][0], {warp_n, 0});
        kittens::load(b0, bs0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cD, p, rs1, sb1_k);
        else                   scale_accumulate_1d1d(cD, p, rs1, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1;
    }

    {
        const int k = k_iters - 1;
        float sb0_k, sb1_k;
        ColScale<RT_C::width> cs0, cs1;
        if constexpr (IS_1D2D) {
            sb0_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb0[k])));
            sb1_k = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, sb1[k])));
        } else {
            const float *sb_col = scale_B + k * N;
            cs0 = load_col_scale<RT_C>(sb_col, local_n0, N);
            cs1 = load_col_scale<RT_C>(sb_col, local_n1, N);
        }
        const float *sa_k = sa_row + k * M;
        const auto rs0 = load_row_scale<RT_C>(sa_k, local_m0, m_valid);
        const auto rs1 = load_row_scale<RT_C>(sa_k, local_m1, m_valid);

        auto as0 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][0], {warp_m, 0});
        kittens::load(a, as0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cA, p, rs0, sb0_k);
        else                   scale_accumulate_1d1d(cA, p, rs0, cs0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[tic][1], {warp_n, 0});
        kittens::load(b1, bs1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cB, p, rs0, sb1_k);
        else                   scale_accumulate_1d1d(cB, p, rs0, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[tic][1], {warp_m, 0});
        kittens::load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cC, p, rs1, sb0_k);
        else                   scale_accumulate_1d1d(cC, p, rs1, cs0);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cD, p, rs1, sb1_k);
        else                   scale_accumulate_1d1d(cD, p, rs1, cs1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
    }

    const int m_off0 = block_row * BLOCK_M + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_M + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_N + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA) {
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cA, m_off0, n_off0, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cB, m_off0, n_off1, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cC, m_off1, n_off0, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cD, m_off1, n_off1, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
    }

    if constexpr (std::is_same_v<OType, kittens::bf16>) {
        apply_rtne_bias(cA); apply_rtne_bias(cB); apply_rtne_bias(cC); apply_rtne_bias(cD);
    }

    OType *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;

    const bool full = (block_row + 1) * BLOCK_M <= M && (block_col + 1) * BLOCK_N <= N;
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

template <typename OType, int CBSZ, int BLGP,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d2d(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
    micro_tk_body<OType, CBSZ, BLGP, true, HAS_BIAS, HAS_GELU, HAS_BETA>(g);
}

template <typename OType, int CBSZ, int BLGP,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d1d(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
    micro_tk_body<OType, CBSZ, BLGP, false, HAS_BIAS, HAS_GELU, HAS_BETA>(g);
}

template <typename OType, int CBSZ, int BLGP, bool IS_1D2D,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_smallk(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
    const auto A = g.a;
    const auto B = g.b;
    const auto C = g.c;
    const float *scale_A = g.scale_a.raw_ptr;
    const float *scale_B = g.scale_b.raw_ptr;
    const int M = (int)g.c.rows(), N = (int)g.c.cols(), K = (int)g.a.cols();
    const int k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    const int scale_K = k_blocks;
    const int blocks_per_col = (N + BLOCK_N - 1) / BLOCK_N;

    using ST_A = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A = kittens::rt_fp8e4m3<REG_M, MFMA_K>;
    using RT_B = kittens::rt_fp8e4m3<REG_N, MFMA_K>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];

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

    using T = kittens::fp8e4m3;
    const kittens::fp8e8m0_4 unit = 0x7F7F7F7Fu;

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>();
    const int b_row_stride = B.template stride<2>();

    const int n_scale_blocks = (N + SCALE_BLOCK - 1) / SCALE_BLOCK;
    const int nb0 = min(block_col * 2 + 0, n_scale_blocks - 1);
    const int nb1 = min(block_col * 2 + 1, n_scale_blocks - 1);
    const float *sa_row = scale_A + block_row * BLOCK_M;
    const float *sb0 = scale_B + nb0 * scale_K;
    const float *sb1 = scale_B + nb1 * scale_K;
    const int local_m0 = warp_m * REG_M;
    const int local_m1 = HALF_ROW + warp_m * REG_M;
    const int m_valid = M - block_row * BLOCK_M;
    const int local_n0 = block_col * BLOCK_N + warp_n * REG_N;
    const int local_n1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    for (int k = 0; k < k_blocks; k++) {
        __builtin_amdgcn_s_barrier();
        load_tile_masked<NUM_THREADS>(As[0], a_base, a_row_stride, block_row * 2 + 0, k, M, K);
        load_tile_masked<NUM_THREADS>(As[1], a_base, a_row_stride, block_row * 2 + 1, k, M, K);
        load_tile_masked<NUM_THREADS>(Bs[0], b_base, b_row_stride, block_col * 2 + 0, k, N, K);
        load_tile_masked<NUM_THREADS>(Bs[1], b_base, b_row_stride, block_col * 2 + 1, k, N, K);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        const auto rs0 = load_row_scale<RT_C>(sa_row + k * M, local_m0, m_valid);
        const auto rs1 = load_row_scale<RT_C>(sa_row + k * M, local_m1, m_valid);
        float sb0_k, sb1_k;
        ColScale<RT_C::width> cs0, cs1;
        if constexpr (IS_1D2D) {
            sb0_k = sb0[k];
            sb1_k = sb1[k];
        } else {
            const float *sb_col = scale_B + k * N;
            cs0 = load_col_scale<RT_C>(sb_col, local_n0, N);
            cs1 = load_col_scale<RT_C>(sb_col, local_n1, N);
        }

        auto as0 = kittens::subtile_inplace<REG_M, MFMA_K>(As[0], {warp_m, 0});
        kittens::load(a, as0);
        auto bs0 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[0], {warp_n, 0});
        kittens::load(b0, bs0);
        auto bs1 = kittens::subtile_inplace<REG_N, MFMA_K>(Bs[1], {warp_n, 0});
        kittens::load(b1, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cA, p, rs0, sb0_k);
        else                   scale_accumulate_1d1d(cA, p, rs0, cs0);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cB, p, rs0, sb1_k);
        else                   scale_accumulate_1d1d(cB, p, rs0, cs1);
        __builtin_amdgcn_sched_barrier(0);

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[1], {warp_m, 0});
        kittens::load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b0, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cC, p, rs1, sb0_k);
        else                   scale_accumulate_1d1d(cC, p, rs1, cs0);
        kittens::zero(p); mma_ABt_scaled<CBSZ, BLGP>(p, a, b1, p, &unit, &unit);
        if constexpr (IS_1D2D) scale_accumulate(cD, p, rs1, sb1_k);
        else                   scale_accumulate_1d1d(cD, p, rs1, cs1);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    const int m_off0 = block_row * BLOCK_M + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_M + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_N + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA) {
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cA, m_off0, n_off0, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cB, m_off0, n_off1, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cC, m_off1, n_off0, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(cD, m_off1, n_off1, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
    }

    if constexpr (std::is_same_v<OType, kittens::bf16>) {
        apply_rtne_bias(cA); apply_rtne_bias(cB); apply_rtne_bias(cC); apply_rtne_bias(cD);
    }

    OType *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;

    const bool full = (block_row + 1) * BLOCK_M <= M && (block_col + 1) * BLOCK_N <= N;
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

template <typename OType>
using micro_globals_fp8 = micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType>;

template <typename OType, int CBSZ, int BLGP, bool IS_1D2D,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
static void dispatch_micro_kernel(micro_globals_fp8<OType> g) {
    const int K = g.K();
    if (K < 2 * BLOCK_K || K % BLOCK_K != 0) {
        micro_tk_smallk<OType, CBSZ, BLGP, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else if constexpr (IS_1D2D) {
        micro_tk_1d2d<OType, CBSZ, BLGP, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else {
        micro_tk_1d1d<OType, CBSZ, BLGP, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    }
}

template <typename OType, int CBSZ, int BLGP, bool IS_1D2D>
static void dispatch_micro_epilogue(bool has_bias, bool has_gelu, bool has_beta, micro_globals_fp8<OType> g) {
    if (has_gelu) {
        if (has_beta) dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, false, true, true >(g);
        else          dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, false, true, false>(g);
    } else if (has_bias) {
        if (has_beta) dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, true, false, true >(g);
        else          dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, true, false, false>(g);
    } else {
        if (has_beta) dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, false, false, true >(g);
        else          dispatch_micro_kernel<OType, CBSZ, BLGP, IS_1D2D, false, false, false>(g);
    }
}

template <typename OType, bool IS_1D2D>
static void dispatch_micro_fp8(int cbsz, int blgp, bool has_bias, bool has_gelu, bool has_beta,
                                 micro_globals_fp8<OType> g) {
    if      (cbsz == 0 && blgp == 0) dispatch_micro_epilogue<OType, 0, 0, IS_1D2D>(has_bias, has_gelu, has_beta, g);
    else if (cbsz == 0 && blgp == 1) dispatch_micro_epilogue<OType, 0, 1, IS_1D2D>(has_bias, has_gelu, has_beta, g);
    else if (cbsz == 1 && blgp == 0) dispatch_micro_epilogue<OType, 1, 0, IS_1D2D>(has_bias, has_gelu, has_beta, g);
    else                             dispatch_micro_epilogue<OType, 1, 1, IS_1D2D>(has_bias, has_gelu, has_beta, g);
}

template <typename OType>
static void dispatch_micro(bool is_1d2d, int cbsz, int blgp, bool has_bias, bool has_gelu, bool has_beta,
                                  micro_globals_fp8<OType> g) {
    if (is_1d2d) dispatch_micro_fp8<OType, true>(cbsz, blgp, has_bias, has_gelu, has_beta, g);
    else         dispatch_micro_fp8<OType, false>(cbsz, blgp, has_bias, has_gelu, has_beta, g);
}

void kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const void *c_in, float beta,
    hipStream_t stream) {
    const bool has_bias = (bias != nullptr);
    const bool has_gelu = (gelu_aux != nullptr);
    const bool has_beta = (c_in != nullptr);

    const void *kA = B,         *kB = A;
    const void *ksa = scale_B,  *ksb = scale_A;
    const int   kM = N,         kN = M;
    const int   ka_mode = b_scaling_mode, kb_mode = a_scaling_mode;
    const int   ka_dtype = b_dtype,       kb_dtype = a_dtype;

    const bool is_1d2d = (kb_mode == KITTENS_BLOCK_SCALING_2D);
    const int cbsz = (ka_dtype == KITTENS_FP8E5M2) ? 1 : 0;
    const int blgp = (kb_dtype == KITTENS_FP8E5M2) ? 1 : 0;
    float *sa = reinterpret_cast<float *>(const_cast<void *>(ksa));
    float *sb = reinterpret_cast<float *>(const_cast<void *>(ksb));

    auto run = [&]<typename OType>() {
        micro_globals_fp8<OType> g{
            _gl_A_t<kittens::fp8e4m3>((kittens::fp8e4m3 *)const_cast<void *>(kA), 1, 1, kM, K),
            _gl_B_t<kittens::fp8e4m3>((kittens::fp8e4m3 *)const_cast<void *>(kB), 1, 1, kN, K),
            _gl_C_t<OType>((OType *)C, 1, 1, kM, kN),
            _gl_SA(sa, 1, 1, 1, kM * K),
            _gl_SB(sb, 1, 1, 1, kN * K),
            bias, bias_dtype, gelu_aux, gelu_aux_dtype,
            reinterpret_cast<const OType *>(c_in), beta, stream};
        dispatch_micro<OType>(is_1d2d, cbsz, blgp, has_bias, has_gelu, has_beta, g);
    };

    if      (out_dtype == KITTENS_FLOAT32) run.template operator()<float>();
    else if (out_dtype == KITTENS_FLOAT16) run.template operator()<kittens::half>();
    else                                   run.template operator()<kittens::bf16>();
}

}  // namespace blockwise_gfx950
