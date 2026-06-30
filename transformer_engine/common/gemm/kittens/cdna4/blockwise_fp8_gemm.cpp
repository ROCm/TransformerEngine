/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// gfx950 (CDNA4) blockwise FP8 GEMM kernel.
// DeepSeek-style: unscaled FP8 MFMA + rescale-and-accumulate per K-block.
//   128x128 tile, 8 warps (2x4). Scale: 1 FP32 per 128 elems per dim.
//   A scale [M/128, K/128], B scale [N/128, K/128]. TN layout, bf16 out (col-major).
// Uses HipKittens *main* branch (CDNA4) headers — distinct from the gfx942
// blockwise kernel in blockwise_fp8_gemm.cpp (which uses the cdna3 fork).
// Ported from standalone fp8_blockwise_gemm_tn.cu. Host launcher / runtime-shape
// dispatch are wired up separately (kernel-only for now).

#include "kittens.cuh"
#include "blockwise_fp8_gemm.h"

using namespace kittens;

namespace gfx950_blockwise {

constexpr int NUM_WARPS   = 8;
constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int BLOCK_ROW   = 128;
constexpr int BLOCK_COL   = 128;
constexpr int BLOCK_K     = 128;
constexpr int REG_M       = BLOCK_ROW / WARPS_ROW;   // 64
constexpr int REG_N       = BLOCK_COL / WARPS_COL;   // 32
constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;
constexpr int SCALE_BLOCK  = 128;

using G = kittens::group<NUM_WARPS>;

template <int M, int N, int K>
__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_blockwise_gemm_kernel(
    const gl<fp8e4m3, 1, 1, M, K> A,
    const gl<fp8e4m3, 1, 1, N, K> B,
    const gl<bf16,    1, 1, N, M>  C,
    const float *__restrict__ scale_A,
    const float *__restrict__ scale_B) {
#if defined(__gfx950__)

    constexpr int k_iters = K / BLOCK_K;
    constexpr int scale_K = K / SCALE_BLOCK;

    using ST_A   = st_fp8e4m3<BLOCK_ROW, BLOCK_K, st_16x128_s>;
    using ST_B   = st_fp8e4m3<BLOCK_COL, BLOCK_K, st_16x128_s>;
    using RT_A   = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B   = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C   = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
    using RT_C_T = rt_fl<REG_N, REG_M, row_l, rt_16x16_s>;

    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];

    RT_A a;
    RT_B b;
    RT_C acc;
    zero(acc);

    constexpr int tiles_M = M / BLOCK_ROW;
    constexpr int tiles_N = N / BLOCK_COL;
    const int NUM_XCDS    = 8;
    const int WGM         = 8;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    int sa_idx = block_row;
    int sb_idx = block_col;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = BLOCK_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(T)));

    uint32_t a_lds[2] = {a_lds_0, a_lds_1};
    uint32_t b_lds[2] = {b_lds_0, b_lds_1};

    int tic = 0, toc = 1;

    // Prologue: load first two K-tiles
    G::load(As[tic], A, {0, 0, block_row, 0}, sw_A, a_srd, a_base, a_lds[tic]);
    G::load(Bs[tic], B, {0, 0, block_col, 0}, sw_B, b_srd, b_base, b_lds[tic]);

    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc], A, {0, 0, block_row, 1}, sw_A, a_srd, a_base, a_lds[toc]);
    G::load(Bs[toc], B, {0, 0, block_col, 1}, sw_B, b_srd, b_base, b_lds[toc]);
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    float prev_scale = 1.0f;

    // Main loop
    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        float curr_scale = scale_A[sa_idx * scale_K + k] * scale_B[sb_idx * scale_K + k];

        auto bs = subtile_inplace<REG_N, BLOCK_K>(Bs[tic], {warp_n, 0});
        load(b, bs);
        auto as = subtile_inplace<REG_M, BLOCK_K>(As[tic], {warp_m, 0});
        load(a, as);

        mul(acc, acc, prev_scale / curr_scale);

        G::load(As[tic], A, {0, 0, block_row, k + 2}, sw_A, a_srd, a_base, a_lds[tic]);
        G::load(Bs[tic], B, {0, 0, block_col, k + 2}, sw_B, b_srd, b_base, b_lds[tic]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt(acc, a, b, acc);
        __builtin_amdgcn_s_setprio(0);

        prev_scale = curr_scale;

        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue k = k_iters - 2
    {
        int k = k_iters - 2;
        float curr_scale = scale_A[sa_idx * scale_K + k] * scale_B[sb_idx * scale_K + k];

        auto bs = subtile_inplace<REG_N, BLOCK_K>(Bs[tic], {warp_n, 0});
        load(b, bs);
        auto as = subtile_inplace<REG_M, BLOCK_K>(As[tic], {warp_m, 0});
        load(a, as);

        mul(acc, acc, prev_scale / curr_scale);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt(acc, a, b, acc);
        __builtin_amdgcn_s_setprio(0);

        prev_scale = curr_scale;

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        tic ^= 1; toc ^= 1;
    }

    // Final epilogue k = k_iters - 1
    {
        int k = k_iters - 1;
        float curr_scale = scale_A[sa_idx * scale_K + k] * scale_B[sb_idx * scale_K + k];

        auto bs = subtile_inplace<REG_N, BLOCK_K>(Bs[tic], {warp_n, 0});
        load(b, bs);
        auto as = subtile_inplace<REG_M, BLOCK_K>(As[tic], {warp_m, 0});
        load(a, as);

        mul(acc, acc, prev_scale / curr_scale);

        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(2);
        mma_ABt(acc, a, b, acc);
        __builtin_amdgcn_s_setprio(0);

        prev_scale = curr_scale;
    }

    mul(acc, acc, prev_scale);

    // Column-major BF16 output via transpose
    RT_C_T out;
    transpose(out, acc);
    store(C, out, {0, 0, block_col * WARPS_COL + warp_n, block_row * WARPS_ROW + warp_m});
#endif  // __gfx950__
}

// Host-side launch for a compile-time shape.
template <int M, int N, int K>
static inline void launch_one(const void *A, const void *B, void *C,
                              const float *scale_A, const float *scale_B,
                              hipStream_t stream) {
    constexpr int grid = (M / BLOCK_ROW) * (N / BLOCK_COL);
    gl<fp8e4m3, 1, 1, M, K> A_gl((fp8e4m3 *)const_cast<void *>(A), nullptr, nullptr, nullptr, nullptr);
    gl<fp8e4m3, 1, 1, N, K> B_gl((fp8e4m3 *)const_cast<void *>(B), nullptr, nullptr, nullptr, nullptr);
    gl<bf16,    1, 1, N, M> C_gl((bf16 *)C, nullptr, nullptr, nullptr, nullptr);
    fp8_blockwise_gemm_kernel<M, N, K><<<grid, NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, scale_A, scale_B);
}

}  // namespace gfx950_blockwise

// Runtime-shape dispatch. Compile-time M/N/K template instances are limited to
// a fixed set of common shapes (square 1024..16384). Returns false for any
// other shape so the caller can fall back. TODO: generalize to runtime shapes.
bool kittens_blockwise_fp8_gemm_impl_cdna4(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    hipStream_t stream) {
    const float *sa = reinterpret_cast<const float *>(scale_A);
    const float *sb = reinterpret_cast<const float *>(scale_B);
#define KK_CASE(m, n, k) \
    if (M == (m) && N == (n) && K == (k)) { \
        gfx950_blockwise::launch_one<(m), (n), (k)>(A, B, C, sa, sb, stream); \
        return true; \
    }
    KK_CASE( 1024,  1024,  1024)
    KK_CASE( 2048,  2048,  2048)
    KK_CASE( 4096,  4096,  4096)
    KK_CASE( 8192,  8192,  8192)
    KK_CASE(16384, 16384, 16384)
#undef KK_CASE
    return false;  // unsupported shape -> caller falls back
}
