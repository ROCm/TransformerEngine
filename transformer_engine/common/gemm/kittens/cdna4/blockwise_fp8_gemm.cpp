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
constexpr int BLOCK_M     = 256;
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

// Per-row scale (running-rescale ratio) held in registers between the LDS read and the multiply,
// so the read can be hoisted to an earlier cluster and reused across accumulators sharing a row.
template <int HEIGHT>
struct RowRatio { float s[HEIGHT][4]; };

// (1) Read the per-row ratios for one m-half from LDS into registers (vectorizes to ds_read_b128).
template <typename RT_C>
__device__ __forceinline__ RowRatio<RT_C::height>
load_row_ratio(const float *smem_ratios, int warp_m_offset) {
    int lane = kittens::laneid();
    int row_off = RT_C::base_tile_stride * (lane / RT_C::base_tile_cols);
    RowRatio<RT_C::height> rr;
    #pragma unroll
    for (int i = 0; i < RT_C::height; i++) {
        int base_m = warp_m_offset + i * 16 + row_off;
        #pragma unroll
        for (int r = 0; r < 4; r++) rr.s[i][r] = smem_ratios[base_m + r];
    }
    return rr;
}

// (2) Multiply the accumulator by pre-loaded per-row ratios (pure VALU, no LDS access).
template <typename RT_C>
__device__ __forceinline__ void apply_row_ratio(RT_C &acc, const RowRatio<RT_C::height> &rr) {
    #pragma unroll
    for (int i = 0; i < acc.height; i++)
        #pragma unroll
        for (int j = 0; j < acc.width; j++)
            #pragma unroll
            for (int kk = 0; kk < acc.base_tile_num_strides; kk++)
                #pragma unroll
                for (int l = 0; l < acc.base_tile_stride / 2; l++) {
                    int idx = l + kk * acc.base_tile_stride / 2;
                    acc.tiles[i][j].data[idx].x *= rr.s[i][l * 2];
                    acc.tiles[i][j].data[idx].y *= rr.s[i][l * 2 + 1];
                }
}

// (2b) Fused row-ratio * col-scale(sb): fold sb into the per-row ratio (height*4 scalar muls),
// then sweep the accumulator ONCE. Replaces apply_row_ratio + a full-tile kittens::mul(sb) pass.
template <typename RT_C>
__device__ __forceinline__ void apply_row_ratio_sb(RT_C &acc, const RowRatio<RT_C::height> &rr, float sb) {
    float s[RT_C::height][4];
    #pragma unroll
    for (int i = 0; i < RT_C::height; i++)
        #pragma unroll
        for (int r = 0; r < 4; r++) s[i][r] = rr.s[i][r] * sb;
    #pragma unroll
    for (int i = 0; i < acc.height; i++)
        #pragma unroll
        for (int j = 0; j < acc.width; j++)
            #pragma unroll
            for (int kk = 0; kk < acc.base_tile_num_strides; kk++)
                #pragma unroll
                for (int l = 0; l < acc.base_tile_stride / 2; l++) {
                    int idx = l + kk * acc.base_tile_stride / 2;
                    acc.tiles[i][j].data[idx].x *= s[i][l * 2];
                    acc.tiles[i][j].data[idx].y *= s[i][l * 2 + 1];
                }
}

// (2c) Running-rescale for 1Dx1D: weight scale is a per-N vector (ColScale, one scalar per output
// col-tile j), so the rescale is a 2D outer product acc[i][j] *= row_ratio[i] * col_ratio[j].
// Unlike apply_row_ratio_sb (scalar sb pre-folded into rr), row and col vary independently, so the
// col factor cr.v[j] multiplies per (i,j). rr and cr are prev/curr ratios (running-rescale).
template <typename RT_C>
__device__ __forceinline__ void apply_row_col_ratio(RT_C &acc, const RowRatio<RT_C::height> &rr,
                                                    const ColScale<RT_C::width> &cr) {
    #pragma unroll
    for (int i = 0; i < acc.height; i++)
        #pragma unroll
        for (int j = 0; j < acc.width; j++) {
            const float cj = cr.v[j];
            #pragma unroll
            for (int kk = 0; kk < acc.base_tile_num_strides; kk++)
                #pragma unroll
                for (int l = 0; l < acc.base_tile_stride / 2; l++) {
                    int idx = l + kk * acc.base_tile_stride / 2;
                    acc.tiles[i][j].data[idx].x *= rr.s[i][l * 2]     * cj;
                    acc.tiles[i][j].data[idx].y *= rr.s[i][l * 2 + 1] * cj;
                }
        }
}

// Convenience: original fused form (read + multiply) for call sites that don't hoist the read.
template <typename RT_C>
__device__ __forceinline__ void mul_row_prescaled(RT_C &acc, const float *smem_ratios, int warp_m_offset) {
    apply_row_ratio(acc, load_row_ratio<RT_C>(smem_ratios, warp_m_offset));
}

// One output base-tile MMA-accumulate (acc += a*b^T), selecting the input format via CBSZ/BLGP.
// CBSZ/BLGP==0 (e4m3 both) uses the plain fast path; any e5m2 operand uses the scaled MFMA with a
// unit E8M0 scale (0x7F..=2^0 per byte) so the hardware decodes the e5m2 bits without scaling
// (running-rescale scales separately). unit is a function-local so &unit is a real VGPR address.
template <int CBSZ, int BLGP, typename DT, typename AT, typename BT>
__device__ __forceinline__ void mma_accum_base(DT &d, const AT &a, const BT &b) {
    if constexpr (CBSZ == 0 && BLGP == 0) {
        kittens::mma_ABt_base(d, a, b, d);
    } else {
        const kittens::fp8e8m0_4 unit = 0x7F7F7F7Fu;
        kittens::mma_ABt_base_scaled<0, 0, CBSZ, BLGP>(d, a, b, d, &unit, &unit);
    }
}

// Whole-accumulator MMA-accumulate (acc += a*b^T) with the same CBSZ/BLGP format select. Used by
// the batched (non-fused) MMA sites; e4m3 takes the plain fast path, e5m2 the unit-scaled MFMA.
// One base-tile MMA-accumulate carrying the fp8 input format via cbsz/blgp. Uses the CDNA4
// mfma_scale builtin with an IMMEDIATE 0 scale exponent (== unit, no block scaling -- running-
// rescale scales separately). Passing the scale as an immediate is essential: kittens'
// mma_ABt_scaled dereferences a *scale pointer, which the builtin mis-handles here and yields
// nondeterministic nan (the scale operand wants an immediate/constant, not a loaded value).
template <int CBSZ, int BLGP>
__device__ __forceinline__ void mfma_fmt(float2 (&D)[2], const kittens::fp8e4m3_4 (&A)[8],
                                         const kittens::fp8e4m3_4 (&B)[8], const float2 (&C)[2]) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    *(floatx4_t*)D = {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(intx8_t*)A, *(intx8_t*)B, *(floatx4_t*)C, CBSZ, BLGP, 0, 0, 0, 0)};
}

template <int CBSZ, int BLGP, typename RT_C, typename RT_A, typename RT_B>
__device__ __forceinline__ void mma_accum(RT_C &acc, const RT_A &a, const RT_B &b) {
    if constexpr (CBSZ == 0 && BLGP == 0) {
        kittens::mma_ABt(acc, a, b, acc);   // e4m3: plain fast path
    } else {
        #pragma unroll
        for (int n = 0; n < acc.height; n++)
            #pragma unroll
            for (int m = 0; m < acc.width; m++)
                mfma_fmt<CBSZ, BLGP>(acc.tiles[n][m].data, a.tiles[n][0].data,
                                     b.tiles[m][0].data, acc.tiles[n][m].data);
    }
}

// Fine-grained per-tile fusion of running-rescale (apply_row_ratio_sb) with the MMA. For each
// output base tile [n][m]: rescale that tile (VALU), then immediately accumulate a*b^T into it
// (MFMA). Because tiles are independent, tile[n][m]'s MFMA can overlap tile[n'][m']'s VALU
// rescale on separate units -- unlike the batched form (rescale ALL tiles, then MMA ALL tiles)
// where the VALU block finishes before any MFMA starts. Same accumulator/a/b liveness as before
// (no extra VGPR); same math (acc = acc*ratio + a*b per tile).
template <int CBSZ, int BLGP, typename RT_C, typename RT_A, typename RT_B>
__device__ __forceinline__ void mma_ABt_rescaled(
        RT_C &acc, const RowRatio<RT_C::height> &rr, float sb, const RT_A &a, const RT_B &b) {
    // No hoisted s[height][4] temporary: fold sb into the row-ratio on the fly so only 4 scalars
    // (the current row n) are live at once, not all height*4 -- keeps register pressure down so
    // the interleaved MFMA doesn't spill.
    #pragma unroll
    for (int n = 0; n < acc.height; n++) {
        float sr[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) sr[r] = rr.s[n][r] * sb;
        #pragma unroll
        for (int m = 0; m < acc.width; m++) {
            #pragma unroll
            for (int kk = 0; kk < acc.base_tile_num_strides; kk++)
                #pragma unroll
                for (int l = 0; l < acc.base_tile_stride / 2; l++) {
                    int idx = l + kk * acc.base_tile_stride / 2;
                    acc.tiles[n][m].data[idx].x *= sr[l * 2];
                    acc.tiles[n][m].data[idx].y *= sr[l * 2 + 1];
                }
            mma_accum_base<CBSZ, BLGP>(acc.tiles[n][m], a.tiles[n][0], b.tiles[m][0]);
        }
    }
}

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

template <typename OType>
using micro_globals_fp8 = micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType>;

template <typename OType, int CBSZ, int BLGP,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d2d(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
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

    // Double-buffer LDS + running-rescale (ported from blockwise_fp8_gemm_ref.cpp, 256-tile).
    // Register pressure pushed to LDS: single RT_A `a` reused, row-scale read from LDS (not regs),
    // accum in-place (no partial tile). scale_A read k-major [scale_K, M] (our layout).
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ alignas(16) float smem_sa_prev[BLOCK_M];
    __shared__ alignas(16) float smem_sa_curr[BLOCK_M];
    __shared__ alignas(16) float smem_a_ratio[2][BLOCK_M];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    // XCD-aware block swizzle + WGM group ordering (ref 256-tile: WGM=8, tiles over 256 blocks).
    const int tiles_M = (M + BLOCK_M - 1) / BLOCK_M;
    const int tiles_N = (N + BLOCK_N - 1) / BLOCK_N;
    constexpr int WGM = 8;
    int wgid = kittens::chiplet_transform_chunked((int)blockIdx.x, (int)gridDim.x, kittens::NUM_XCDS, WGM * WGM);
    const int num_wgid_in_group = WGM * tiles_N;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(tiles_M - first_pid_m, WGM);
    const int block_row = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int block_col = (wgid % num_wgid_in_group) / group_size_m;
    const int block_m = block_row * BLOCK_M;
    const int block_n = block_col * BLOCK_N;

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int tid = threadIdx.x;

    const int sb_h0_idx = block_n / SCALE_BLOCK;
    const int sb_h1_idx = sb_h0_idx + 1;
    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;

    using T = kittens::fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, M * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
            b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        }

    // scale_A is k-major [scale_K, M] (our layout): element (k, m) at k*M + m.
    // scale_B moved out of this lambda: it is uniform per block, so read via scalar_load (SGPR)
    // instead of a vmem load broadcast through LDS -- keeps scale_B off the vmcnt(6) drain that
    // gated it every iteration (it was bunched with scale_A's 256 vmem loads).
    const float *sb0_p = scale_B + sb_h0_idx * scale_K;
    const float *sb1_p = scale_B + sb_h1_idx * scale_K;
    // scale_A direct global->LDS DMA (no register round-trip => no forced early vmcnt(0) at the
    // register->LDS store). SRD spans the whole scale_A [scale_K, M] buffer; per-lane 4-byte load,
    // m0 base is wave-uniform per warp and the DMA scatters lane L to base + L*4.
    kittens::i32x4 sa_srd = kittens::make_srsrc((const void *)scale_A, (uint32_t)(scale_K * M * (int)sizeof(float)));
    const int sa_warp = tid / kittens::WARP_THREADS;          // 0..7
    const int sa_lane = tid % kittens::WARP_THREADS;          // 0..63
    const uint32_t sa_curr_lds_warp = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&smem_sa_curr[sa_warp * kittens::WARP_THREADS])));
    auto load_scales_to_curr = [&](int k) {
        if (tid < BLOCK_M)
            load_scale_to_lds(sa_srd, sa_curr_lds_warp, (k * M + block_m + sa_warp * kittens::WARP_THREADS + sa_lane) * 4);
    };
    // Split the BLOCK_M ratios across BOTH warp_m groups (tid>>8 == warp_m): group 0 owns m=0..127,
    // group 1 owns m=128..255, each via float2 on its first 64 lanes (ds_read/write_b64). Both halves
    // of the ping-pong pair do equal division work instead of piling it all on warp_m=0, which is the
    // fastest of the variants tried (per-element-on-warp0 and wave-0-only-float4 both unbalanced it).
    auto compute_a_ratios_and_promote = [&](int dst) {
        int g = tid >> 8;             // warp_m group (0 or 1)
        int lt = tid & 255;           // lane within group
        if (lt < BLOCK_M / 4) {
            int e = g * (BLOCK_M / 4) + lt;
            float2 p = reinterpret_cast<const float2 *>(smem_sa_prev)[e];
            float2 c = reinterpret_cast<const float2 *>(smem_sa_curr)[e];
            float2 r = {p.x / c.x, p.y / c.y};
            reinterpret_cast<float2 *>(smem_a_ratio[dst])[e] = r;
            reinterpret_cast<float2 *>(smem_sa_prev)[e] = c;
        }
    };

    int tic = 0, toc = 1;
    int rtic = 0, rtoc = 1;

    // ---- Prologue: load k=0 (tic) + k=1 (toc), init scales ----
    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A, a_srd, a_base, a_lds[tic][0]);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds[tic][1]);

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A, a_srd, a_base, a_lds[toc][0]);
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    if (tid < BLOCK_M) smem_sa_prev[tid] = 1.0f;
    load_scales_to_curr(0);
    float curr_sb_h0 = scalar_load_scale(sb0_p, 0);   // scale_B(k=0) via SGPR; drained by lgkmcnt(0) below
    float curr_sb_h1 = scalar_load_scale(sb1_p, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    compute_a_ratios_and_promote(rtic);
    __builtin_amdgcn_s_barrier();

    float prev_sb_h0 = 1.0f, prev_sb_h1 = 1.0f;

    // ---- Main loop (8-phase, unroll 2, A reused, running-rescale) ----
    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, rtic ^= 1, rtoc ^= 1) {
        float sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;   // used cluster 1 (cA); h1 division deferred to c2

        // rr_h0 read FIRST (ratio was written by prev-iter cluster 6, already visible): its
        // ds_read latency then overlaps the cluster-0 a/b LDS reads below before cluster-1 uses it.
        auto rr_h0 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0);
        load_scales_to_curr(k + 1);
        // next-k scale_B via SGPR (drained by Phase 3/5 lgkmcnt(0) before promotion at Phase 8)
        float nxt_sb_h0 = scalar_load_scale(sb0_p, k + 1);
        float nxt_sb_h1 = scalar_load_scale(sb1_p, k + 1);

        // cluster 0: load b0, a(h0), issue G::load(A[toc][h1], k+1)
        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        // cluster 1: rescale + MMA cA
        apply_row_ratio_sb(cA, rr_h0, sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 2: load b1, issue G::load(A[tic][h0], k+2); h1 col-ratio division (deferred from c0)
        float sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // cluster 3: rescale + MMA cB (reuses h0 ratio)
        apply_row_ratio_sb(cB, rr_h0, sb_ratio_h1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // cluster 4: load a(h1), issue G::load(B[tic][h0], k+2); load h1 row-ratio once (reused cC/cD)
        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        auto rr_h1 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1);
        __builtin_amdgcn_s_barrier();
        
        // cluster 5: rescale + MMA cC
        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cC, rr_h1, sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 6: issue G::load(B[tic][h1], k+2), wait; compute next-iter row ratios here
        // (smem_sa_curr's scale_A DMA has drained by this point; ratio write is made visible to
        //  next iter's cluster 0 by this cluster's barrier).
        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        compute_a_ratios_and_promote(rtoc);
        __builtin_amdgcn_s_barrier();

        // cluster 7: rescale + MMA cD (reuses h1 ratio)
        apply_row_ratio_sb(cD, rr_h1, sb_ratio_h1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);

        prev_sb_h0 = curr_sb_h0; prev_sb_h1 = curr_sb_h1;
        curr_sb_h0 = nxt_sb_h0; curr_sb_h1 = nxt_sb_h1;
        __builtin_amdgcn_s_barrier();
    }

    // ---- Epilogue k = k_iters - 2 ----
    {
        int k = k_iters - 2;
        float sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;
        float sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        load_scales_to_curr(k + 1);
        float nxt_sb_h0 = scalar_load_scale(sb0_p, k + 1);
        float nxt_sb_h1 = scalar_load_scale(sb1_p, k + 1);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h1);
        compute_a_ratios_and_promote(rtoc);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);

        prev_sb_h0 = curr_sb_h0; prev_sb_h1 = curr_sb_h1;
        curr_sb_h0 = nxt_sb_h0; curr_sb_h1 = nxt_sb_h1;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1; rtic ^= 1; rtoc ^= 1;
    }

    // ---- Final epilogue k = k_iters - 1 ----
    {
        float sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;
        float sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_ratio_sb(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);

        apply_row_ratio_sb(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h1);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_barrier();
    }

    // Apply final absolute scale + store per-accumulator (sequential) so only one cX is live at a
    // time through scale/bias/store. Cuts the 4-accumulator peak that caused the epilogue VGPR spill.
    const int m_off0 = block_row * BLOCK_M + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_M + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_N + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    OType *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;
    const bool full = (block_row + 1) * BLOCK_M <= M && (block_col + 1) * BLOCK_N <= N;

    auto finish = [&](RT_C &c, int a_row, float sb, int m_off, int n_off, int crow, int ccol) {
        apply_row_ratio_sb(c, load_row_ratio<RT_C>(smem_sa_prev, a_row), sb);
        if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA)
            apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(c, m_off, n_off, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        if constexpr (std::is_same_v<OType, kittens::bf16>) apply_rtne_bias(c);
        if (full) kittens::store(C, c, {0, 0, crow, ccol});
        else      store_masked(c_ptr, c, m_off, n_off, M, N);
    };
    finish(cA, a_row_h0, curr_sb_h0, m_off0, n_off0, ca, cn0);
    finish(cB, a_row_h0, curr_sb_h1, m_off0, n_off1, ca, cn1);
    finish(cC, a_row_h1, curr_sb_h0, m_off1, n_off0, cc, cn0);
    finish(cD, a_row_h1, curr_sb_h1, m_off1, n_off1, cc, cn1);
}

#ifndef NVTE_SQ_ONLY
template <typename OType, int CBSZ, int BLGP,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk_1d1d(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
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

    // Double-buffer LDS + running-rescale (cloned from micro_tk_1d2d, 256-tile). Only difference:
    // the WEIGHT scale is a per-N vector (ColScale, one scalar per output col-tile) instead of a
    // scalar per column-half, so the rescale is a 2D outer product (apply_row_col_ratio).
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ alignas(16) float smem_sa_prev[BLOCK_M];
    __shared__ alignas(16) float smem_sa_curr[BLOCK_M];
    __shared__ alignas(16) float smem_a_ratio[2][BLOCK_M];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    // XCD-aware block swizzle + WGM group ordering (ref 256-tile: WGM=8, tiles over 256 blocks).
    const int tiles_M = (M + BLOCK_M - 1) / BLOCK_M;
    const int tiles_N = (N + BLOCK_N - 1) / BLOCK_N;
    constexpr int WGM = 8;
    int wgid = kittens::chiplet_transform_chunked((int)blockIdx.x, (int)gridDim.x, kittens::NUM_XCDS, WGM * WGM);
    const int num_wgid_in_group = WGM * tiles_N;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(tiles_M - first_pid_m, WGM);
    const int block_row = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int block_col = (wgid % num_wgid_in_group) / group_size_m;
    const int block_m = block_row * BLOCK_M;
    const int block_n = block_col * BLOCK_N;

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int tid = threadIdx.x;

    // Weight per-N col-scale bases for the two N-halves (mirrors 1d2d's sb_h0/h1 scalar indices).
    const int local_n0 = block_n + warp_n * REG_N;
    const int local_n1 = block_n + HALF_COL + warp_n * REG_N;
    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;

    using T = kittens::fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, M * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
            b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        }

    // scale_A is k-major [scale_K, M] (our layout): element (k, m) at k*M + m.
    // scale_B (1d1d weight) is [scale_K, N]: element (k, n) at k*N + n; loaded as a per-N ColScale.
    // scale_A direct global->LDS DMA (no register round-trip => no forced early vmcnt(0) at the
    // register->LDS store). SRD spans the whole scale_A [scale_K, M] buffer; per-lane 4-byte load,
    // m0 base is wave-uniform per warp and the DMA scatters lane L to base + L*4.
    kittens::i32x4 sa_srd = kittens::make_srsrc((const void *)scale_A, (uint32_t)(scale_K * M * (int)sizeof(float)));
    const int sa_warp = tid / kittens::WARP_THREADS;          // 0..7
    const int sa_lane = tid % kittens::WARP_THREADS;          // 0..63
    const uint32_t sa_curr_lds_warp = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&smem_sa_curr[sa_warp * kittens::WARP_THREADS])));
    auto load_scales_to_curr = [&](int k) {
        if (tid < BLOCK_M)
            load_scale_to_lds(sa_srd, sa_curr_lds_warp, (k * M + block_m + sa_warp * kittens::WARP_THREADS + sa_lane) * 4);
    };
    // Split the BLOCK_M ratios across BOTH warp_m groups (tid>>8 == warp_m): group 0 owns m=0..127,
    // group 1 owns m=128..255, each via float2 on its first 64 lanes (ds_read/write_b64). Both halves
    // of the ping-pong pair do equal division work instead of piling it all on warp_m=0, which is the
    // fastest of the variants tried (per-element-on-warp0 and wave-0-only-float4 both unbalanced it).
    auto compute_a_ratios_and_promote = [&](int dst) {
        int g = tid >> 8;             // warp_m group (0 or 1)
        int lt = tid & 255;           // lane within group
        if (lt < BLOCK_M / 4) {
            int e = g * (BLOCK_M / 4) + lt;
            float2 p = reinterpret_cast<const float2 *>(smem_sa_prev)[e];
            float2 c = reinterpret_cast<const float2 *>(smem_sa_curr)[e];
            float2 r = {p.x / c.x, p.y / c.y};
            reinterpret_cast<float2 *>(smem_a_ratio[dst])[e] = r;
            reinterpret_cast<float2 *>(smem_sa_prev)[e] = c;
        }
    };

    // Elementwise weight col-ratio prev/curr (per output col-tile j) for one N-half.
    auto col_ratio = [](const ColScale<RT_C::width> &p, const ColScale<RT_C::width> &c) {
        ColScale<RT_C::width> r;
        #pragma unroll
        for (int j = 0; j < RT_C::width; j++) r.v[j] = p.v[j] / c.v[j];
        return r;
    };

    int tic = 0, toc = 1;
    int rtic = 0, rtoc = 1;

    // ---- Prologue: load k=0 (tic) + k=1 (toc), init scales ----
    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A, a_srd, a_base, a_lds[tic][0]);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds[tic][1]);

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A, a_srd, a_base, a_lds[toc][0]);
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    if (tid < BLOCK_M) smem_sa_prev[tid] = 1.0f;
    load_scales_to_curr(0);
    ColScale<RT_C::width> curr_cs0 = load_col_scale<RT_C>(scale_B + 0 * N, local_n0, N);
    ColScale<RT_C::width> curr_cs1 = load_col_scale<RT_C>(scale_B + 0 * N, local_n1, N);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    compute_a_ratios_and_promote(rtic);
    __builtin_amdgcn_s_barrier();

    ColScale<RT_C::width> prev_cs0, prev_cs1;
    #pragma unroll
    for (int j = 0; j < RT_C::width; j++) { prev_cs0.v[j] = 1.0f; prev_cs1.v[j] = 1.0f; }

    // ---- Main loop (8-phase, unroll 2, A reused, running-rescale) ----
    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, rtic ^= 1, rtoc ^= 1) {
        ColScale<RT_C::width> cr0 = col_ratio(prev_cs0, curr_cs0);   // used cluster 1 (cA); h1 ratio deferred to c2

        // rr_h0 read FIRST (ratio was written by prev-iter cluster 6, already visible): its
        // ds_read latency then overlaps the cluster-0 a/b LDS reads below before cluster-1 uses it.
        auto rr_h0 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0);
        load_scales_to_curr(k + 1);
        // next-k weight col-scale (per-N vector); advanced into prev/curr at cluster 7.
        ColScale<RT_C::width> nxt_cs0 = load_col_scale<RT_C>(scale_B + (k + 1) * N, local_n0, N);
        ColScale<RT_C::width> nxt_cs1 = load_col_scale<RT_C>(scale_B + (k + 1) * N, local_n1, N);

        // cluster 0: load b0, a(h0), issue G::load(A[toc][h1], k+1)
        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        // cluster 1: rescale + MMA cA
        apply_row_col_ratio(cA, rr_h0, cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 2: load b1, issue G::load(A[tic][h0], k+2); h1 col-ratio (deferred from c0)
        ColScale<RT_C::width> cr1 = col_ratio(prev_cs1, curr_cs1);
        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // cluster 3: rescale + MMA cB (reuses h0 row ratio)
        apply_row_col_ratio(cB, rr_h0, cr1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // cluster 4: load a(h1), issue G::load(B[tic][h0], k+2); load h1 row-ratio once (reused cC/cD)
        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        auto rr_h1 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1);
        __builtin_amdgcn_s_barrier();

        // cluster 5: rescale + MMA cC
        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cC, rr_h1, cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 6: issue G::load(B[tic][h1], k+2), wait; compute next-iter row ratios here
        // (smem_sa_curr's scale_A DMA has drained by this point; ratio write is made visible to
        //  next iter's cluster 0 by this cluster's barrier).
        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        compute_a_ratios_and_promote(rtoc);
        __builtin_amdgcn_s_barrier();

        // cluster 7: rescale + MMA cD (reuses h1 row ratio)
        apply_row_col_ratio(cD, rr_h1, cr1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);

        prev_cs0 = curr_cs0; prev_cs1 = curr_cs1;
        curr_cs0 = nxt_cs0; curr_cs1 = nxt_cs1;
        __builtin_amdgcn_s_barrier();
    }

    // ---- Epilogue k = k_iters - 2 ----
    {
        int k = k_iters - 2;
        ColScale<RT_C::width> cr0 = col_ratio(prev_cs0, curr_cs0);
        ColScale<RT_C::width> cr1 = col_ratio(prev_cs1, curr_cs1);
        load_scales_to_curr(k + 1);
        ColScale<RT_C::width> nxt_cs0 = load_col_scale<RT_C>(scale_B + (k + 1) * N, local_n0, N);
        ColScale<RT_C::width> nxt_cs1 = load_col_scale<RT_C>(scale_B + (k + 1) * N, local_n1, N);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr1);
        compute_a_ratios_and_promote(rtoc);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);

        prev_cs0 = curr_cs0; prev_cs1 = curr_cs1;
        curr_cs0 = nxt_cs0; curr_cs1 = nxt_cs1;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1; rtic ^= 1; rtoc ^= 1;
    }

    // ---- Final epilogue k = k_iters - 1 ----
    {
        ColScale<RT_C::width> cr0 = col_ratio(prev_cs0, curr_cs0);
        ColScale<RT_C::width> cr1 = col_ratio(prev_cs1, curr_cs1);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr1);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        apply_row_col_ratio(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr0);
        __builtin_amdgcn_s_setprio(2);
        mma_accum<CBSZ, BLGP>(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);

        apply_row_col_ratio(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr1);
        mma_accum<CBSZ, BLGP>(cD, a, b1);
        __builtin_amdgcn_s_barrier();
    }

    // Apply final absolute scale + store per-accumulator (sequential) so only one cX is live at a
    // time through scale/bias/store. Cuts the 4-accumulator peak that caused the epilogue VGPR spill.
    const int m_off0 = block_row * BLOCK_M + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_M + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_N + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    OType *c_ptr = C.raw_ptr;
    const int ca = block_row * WARPS_ROW * 2 + warp_m;
    const int cc = block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m;
    const int cn0 = block_col * WARPS_COL * 2 + warp_n;
    const int cn1 = block_col * WARPS_COL * 2 + WARPS_COL + warp_n;
    const bool full = (block_row + 1) * BLOCK_M <= M && (block_col + 1) * BLOCK_N <= N;

    auto finish = [&](RT_C &c, int a_row, const ColScale<RT_C::width> &cs, int m_off, int n_off, int crow, int ccol) {
        apply_row_col_ratio(c, load_row_ratio<RT_C>(smem_sa_prev, a_row), cs);
        if constexpr (HAS_BIAS || HAS_GELU || HAS_BETA)
            apply_epilogue<OType, HAS_BIAS, HAS_GELU, HAS_BETA>(c, m_off, n_off, M, N, g.bias, g.bias_dtype, g.gelu_aux, g.gelu_aux_dtype, g.c_in, g.beta);
        if constexpr (std::is_same_v<OType, kittens::bf16>) apply_rtne_bias(c);
        if (full) kittens::store(C, c, {0, 0, crow, ccol});
        else      store_masked(c_ptr, c, m_off, n_off, M, N);
    };
    finish(cA, a_row_h0, curr_cs0, m_off0, n_off0, ca, cn0);
    finish(cB, a_row_h0, curr_cs1, m_off0, n_off1, ca, cn1);
    finish(cC, a_row_h1, curr_cs0, m_off1, n_off0, cc, cn0);
    finish(cD, a_row_h1, curr_cs1, m_off1, n_off1, cc, cn1);
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

        kittens::zero(p); mma_accum<CBSZ, BLGP>(p, a, b0);
        if constexpr (IS_1D2D) scale_accumulate(cA, p, rs0, sb0_k);
        else                   scale_accumulate_1d1d(cA, p, rs0, cs0);
        kittens::zero(p); mma_accum<CBSZ, BLGP>(p, a, b1);
        if constexpr (IS_1D2D) scale_accumulate(cB, p, rs0, sb1_k);
        else                   scale_accumulate_1d1d(cB, p, rs0, cs1);
        __builtin_amdgcn_sched_barrier(0);

        auto as1 = kittens::subtile_inplace<REG_M, MFMA_K>(As[1], {warp_m, 0});
        kittens::load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        kittens::zero(p); mma_accum<CBSZ, BLGP>(p, a, b0);
        if constexpr (IS_1D2D) scale_accumulate(cC, p, rs1, sb0_k);
        else                   scale_accumulate_1d1d(cC, p, rs1, cs0);
        kittens::zero(p); mma_accum<CBSZ, BLGP>(p, a, b1);
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


template <typename OType, int CBSZ, int BLGP, bool IS_1D2D,
          bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA, bool IS_PARTIAL_K>
static void dispatch_micro_kernel(micro_globals_fp8<OType> g) {
    if constexpr (IS_PARTIAL_K) {
        micro_tk_smallk<OType, CBSZ, BLGP, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else if constexpr (IS_1D2D) {
        micro_tk_1d2d<OType, CBSZ, BLGP, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else {
        micro_tk_1d1d<OType, CBSZ, BLGP, HAS_BIAS, HAS_GELU, HAS_BETA><<<g.grid(), g.block(), 0, g.stream>>>(g);
    }
}

template <typename OType, bool IS_1D2D, bool IS_PARTIAL_K, bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA>
static void dispatch_micro_dtype(int cbsz, int blgp, micro_globals_fp8<OType> g) {
    if      (cbsz == 0 && blgp == 0) dispatch_micro_kernel<OType, 0, 0, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>(g);
    else if (cbsz == 0 && blgp == 1) dispatch_micro_kernel<OType, 0, 1, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>(g);
    else if (cbsz == 1 && blgp == 0) dispatch_micro_kernel<OType, 1, 0, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>(g);
    else                             dispatch_micro_kernel<OType, 1, 1, IS_1D2D, HAS_BIAS, HAS_GELU, HAS_BETA, IS_PARTIAL_K>(g);
}

template <typename OType, bool IS_1D2D, bool IS_PARTIAL_K>
static void dispatch_micro_epilogue(int cbsz, int blgp, bool has_bias, bool has_gelu, bool has_beta,
                                    micro_globals_fp8<OType> g) {
    if (has_gelu) {
        if (has_beta) dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, false, true, true >(cbsz, blgp, g);
        else          dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, false, true, false>(cbsz, blgp, g);
    } else if (has_bias) {
        if (has_beta) dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, true, false, true >(cbsz, blgp, g);
        else          dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, true, false, false>(cbsz, blgp, g);
    } else {
        if (has_beta) dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, false, false, true >(cbsz, blgp, g);
        else          dispatch_micro_dtype<OType, IS_1D2D, IS_PARTIAL_K, false, false, false>(cbsz, blgp, g);
    }
}

template <typename OType, bool IS_1D2D>
static void dispatch_micro_k(int cbsz, int blgp, bool has_bias, bool has_gelu, bool has_beta,
                             bool has_partial_k, micro_globals_fp8<OType> g) {
    if (has_partial_k) dispatch_micro_epilogue<OType, IS_1D2D, true >(cbsz, blgp, has_bias, has_gelu, has_beta, g);
    else               dispatch_micro_epilogue<OType, IS_1D2D, false>(cbsz, blgp, has_bias, has_gelu, has_beta, g);
}

template <typename OType>
static void dispatch_micro(bool is_1d2d, int cbsz, int blgp, bool has_bias, bool has_gelu, bool has_beta,
                           bool has_partial_k, micro_globals_fp8<OType> g) {
    if (is_1d2d) dispatch_micro_k<OType, true >(cbsz, blgp, has_bias, has_gelu, has_beta, has_partial_k, g);
    else         dispatch_micro_k<OType, false>(cbsz, blgp, has_bias, has_gelu, has_beta, has_partial_k, g);
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
    const bool has_partial_k = (K < 2 * BLOCK_K || K % BLOCK_K != 0);

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
        dispatch_micro<OType>(is_1d2d, cbsz, blgp, has_bias, has_gelu, has_beta, has_partial_k, g);
    };

    if      (out_dtype == KITTENS_FLOAT32) run.template operator()<float>();
    else if (out_dtype == KITTENS_FLOAT16) run.template operator()<kittens::half>();
    else                                   run.template operator()<kittens::bf16>();
}

#endif  // NVTE_SQ_ONLY
}  // namespace blockwise_gfx950
