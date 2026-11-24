/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MXFP8 backward tiles: the NN (dgrad) and NT (wgrad) tiles and dgrad2's SwiGLU-backward epilogue with its dG copies.
#pragma once

namespace hk_moe {
// dG staging per round: bf16 [row][264] (the pad makes a 32-lane subtile row write conflict-free) in one of two
// MX_DG_BUF-byte buffers (the K loop's tiles use both).
constexpr int MX_DG_LD = 528, MX_DG_BUF = 70 * 1024;
static_assert(128 * MX_DG_LD <= MX_DG_BUF, "one round's staging fits a buffer");

// G (gate | up, bf16) of one half round: per lane 4 gate and 4 up 8-byte words.
struct MxGHalf {
  uint64_t gw[2][2], uw[2][2];
};

// g8 is the uniform base of the round (row rt, column n0c of G) and lo the lane's byte offset; the up words are uo
// bytes after the gate words.
template <int BB>
__device__ __forceinline__ void mx_g_load(MxGHalf &gh, const __attribute__((address_space(1))) uint8_t *g8, uint32_t lo,
                                          int N2, uint32_t uo = 256) {
  typedef __attribute__((address_space(1))) uint64_t g64;
#pragma unroll
  for (int m = 0; m < 2; m++) {
    const __attribute__((address_space(1))) uint8_t *b = g8 + (uint32_t)(16 * (2 * BB + m) * N2) * 2;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      gh.gw[h][m] = *(const g64 *)(b + lo + 32 * h);
      gh.uw[h][m] = *(const g64 *)(b + lo + 32 * h + uo);
    }
  }
}

// SwiGLU backward of subtiles 2 BB, 2 BB + 1 of ct, rounded to bf16 into the staging rows r0.. (dgate at bf16 column
// c0, dup at c0 + 128). GATE: dA scaled by the rows' gates sgl[sr + row], and each row's sum of dA act over the lane
// group's columns into sgl[256 (1 + cu) + sr + row].
template <int BB, typename ACT, bool GATE = false>
__device__ __forceinline__ void mx_dg_half(__attribute__((address_space(3))) uint8_t *l, const MxGHalf &gh,
                                           const MxRtCT &ct, int r0, int c0, int lane,
                                           __attribute__((address_space(3))) float *sgl = nullptr, int sr = 0,
                                           int cu = 0) {
  typedef __attribute__((address_space(3))) uint64_t l64;
  typedef __bf16 bf2 __attribute__((ext_vector_type(2)));
  typedef float f2 __attribute__((ext_vector_type(2)));
  const int lc = 2 * (c0 + 4 * (lane >> 4));
  float lo, hi;
  glu_bounds<ACT>(lo, hi);
  float gsum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int h = 0; h < 2; h++) {
#pragma unroll
    for (int m = 0; m < 2; m++) {
      uint64_t pg = 0, pu = 0;
      const int rw = r0 + 16 * (2 * BB + m) + (lane & 15);
      const float sgv = GATE ? sgl[sr + rw] : 1.0f;
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const uint32_t gx = (uint32_t)(gh.gw[h][m] >> (32 * j)), ux = (uint32_t)(gh.uw[h][m] >> (32 * j));
        const auto &d = ct.tiles[h][2 * BB + m].data[j];
        float g0, u0, g1, u1, a0, a1;
        swiglu_bwd_elem<ACT>(GATE ? d.x * sgv : d.x, __builtin_bit_cast(float, gx << 16),
                             __builtin_bit_cast(float, ux << 16), lo, hi, g0, u0, a0);
        swiglu_bwd_elem<ACT>(GATE ? d.y * sgv : d.y, __builtin_bit_cast(float, gx & 0xffff0000u),
                             __builtin_bit_cast(float, ux & 0xffff0000u), lo, hi, g1, u1, a1);
        if constexpr (GATE) {
          gsum[m] = fmaf(d.x, a0, fmaf(d.y, a1, gsum[m]));
        }
        pg |= (uint64_t)__builtin_bit_cast(uint32_t, __builtin_convertvector((f2){g0, g1}, bf2)) << (32 * j);
        pu |= (uint64_t)__builtin_bit_cast(uint32_t, __builtin_convertvector((f2){u0, u1}, bf2)) << (32 * j);
      }
      const int o = (r0 + 16 * (2 * BB + m) + (lane & 15)) * MX_DG_LD + lc + 32 * h;
      *(l64 *)(l + o) = pg;
      *(l64 *)(l + o + 256) = pu;
    }
  }
  if constexpr (GATE) {
#pragma unroll
    for (int m = 0; m < 2; m++) {
      const float v = mx_sum_x16_x32(gsum[m]);
      if (lane < 16) {
        sgl[BLOCK_ROW * (1 + cu) + sr + r0 + 16 * (2 * BB + m) + lane] = v;
      }
    }
  }
}

// dG of one round (128 rows by 128 columns of I) from ct into the staging buffer l as bf16; gh holds this round's G
// and receives the next's (rows rn, column nn; up uo bytes after gate). GATE: as mx_dg_half's, the round's first row sr
// of the tile and its column half ch.
template <bool NEXT, typename ACT, bool GATE = false>
__device__ __forceinline__ void mx_dg_round(__attribute__((address_space(3))) uint8_t *l,
                                            const __attribute__((address_space(1))) uint8_t *g8, const MxRtCT &ct,
                                            MxGHalf (&gh)[2], int wm, int wn, int N2, int lane, uint32_t lo, int rn,
                                            int nn, uint32_t uo = 256,
                                            __attribute__((address_space(3))) float *sgl = nullptr, int sr = 0,
                                            int ch = 0) {
  mx_dg_half<0, ACT, GATE>(l, gh[0], ct, wm * REG_M, wn * REG_N, lane, sgl, sr, ch * WARPS_COL + wn);
  mx_dg_half<1, ACT, GATE>(l, gh[1], ct, wm * REG_M, wn * REG_N, lane, sgl, sr, ch * WARPS_COL + wn);
  if constexpr (NEXT) {
    const __attribute__((address_space(1))) uint8_t *gn = g8 + (uint32_t)(rn * N2 + nn) * 2;
    mx_g_load<0>(gh[0], gn, lo, N2, uo);
    mx_g_load<1>(gh[1], gn, lo, N2, uo);
  }
}

// mx_qput<true, true> of a staged dG block with every maximum taken before any conversion, so the |bf16| copies are
// dead by the time the scales and the converted words are live.
__device__ __forceinline__ void mx_dg_qput(const mx_u4 (&w)[8], int sb, int K, int ldq, i32x4 rs, i32x4 cs, i32x4 ss,
                                           int c, int rg) {
  asm volatile("" : "+s"(K), "+s"(ldq));
  uint32_t pl = 0x05040100u, ph = 0x07060302u, pe = 0x06040200u, po = 0x07050301u;
  asm volatile("" : "+v"(pl), "+v"(ph), "+v"(pe), "+v"(po));
  uint32_t sc[4], sr[4];
  {
    uint32_t a[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        a[i][q] = w[i][q] & 0x7fff7fffu;
      }
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint32_t m = a[0][q];
#pragma unroll
      for (int i = 1; i < 8; i++) {
        m = mx_hmax(m, a[i][q]);
      }
      sc[q] = mx_sc2(__builtin_bit_cast(uint32_t, mx_max_x16_x32(__builtin_bit_cast(mx_h2, m))));
    }
#pragma unroll
    for (int p = 0; p < 4; p++) {
      uint32_t m[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        m[h] = mx_hmax(mx_hmax(a[2 * p + h][0], a[2 * p + h][1]), mx_hmax(a[2 * p + h][2], a[2 * p + h][3]));
      }
      const mx_h2 x = __builtin_bit_cast(mx_h2, mx_hmax(__builtin_amdgcn_perm(m[1], m[0], pl),
                                                   __builtin_amdgcn_perm(m[1], m[0], ph)));
      sr[p] = mx_sc2(__builtin_bit_cast(uint32_t, mx_dpp_max<0x4E>(mx_dpp_max<0xB1>(x))));
    }
  }
  float f[8];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    f[2 * q] = mx_scf(sc[q] & 0xffu);
    f[2 * q + 1] = mx_scf(sc[q] >> 16);
  }
#pragma unroll
  for (int p = 0; p < 4; p++) {
    uint32_t d[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint32_t lo = __builtin_amdgcn_perm(w[2 * p + 1][q], w[2 * p][q], pl);
      const uint32_t hi = __builtin_amdgcn_perm(w[2 * p + 1][q], w[2 * p][q], ph);
      d[q] = mx_cvt4(lo, hi, f[2 * q], f[2 * q + 1]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t sel = h ? po : pe;
      const uint64_t o = (uint64_t)__builtin_amdgcn_perm(d[1], d[0], sel) |
                         (uint64_t)__builtin_amdgcn_perm(d[3], d[2], sel) << 32;
      llvm_amdgcn_raw_buffer_store_b64(o, cs, (uint32_t)(rg * K + c), (32 * sb + 2 * p + h) * K, 0);
    }
  }
  const uint64_t sw = (uint64_t)__builtin_amdgcn_perm(sc[1], sc[0], pe) |
                      (uint64_t)__builtin_amdgcn_perm(sc[3], sc[2], pe) << 32;
  llvm_amdgcn_raw_buffer_store_b64(sw, ss, (uint32_t)c, sb * K, 0);
#pragma unroll
  for (int p = 0; p < 4; p++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int i = 2 * p + h;
      const uint32_t s = h ? sr[p] >> 16 : sr[p] & 0xffu;
      const float fr = mx_scf(s);
      const uint64_t o =
          (uint64_t)mx_cvt4(w[i][0], w[i][1], fr, fr) | (uint64_t)mx_cvt4(w[i][2], w[i][3], fr, fr) << 32;
      const int so = (32 * sb + i) * ldq;
      llvm_amdgcn_raw_buffer_store_b64(o, rs, (uint32_t)(rg * ldq + c), so, 0);
      llvm_amdgcn_raw_buffer_store_b8((uint8_t)s, rs, (uint32_t)(rg * ldq + K + c / 32), so, 0);
    }
  }
}

// A staged round (rows rt, columns n0c of dG) row-wise into dq (2I bytes then 2I/32 scales, 4I per row) and
// column-wise into cq, cs in one pass: each wave quantizes 32 rows by 128 columns, the up half's us columns further.
__device__ __forceinline__ void mx_dg_quant(const __attribute__((address_space(3))) uint8_t *l, uint8_t *dq,
                                            uint8_t *cq, uint8_t *cs, int rt, int n0c, int N2, int lane, int us = 0) {
  typedef const __attribute__((address_space(3))) mx_u4 l128;
  const int w = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS);
  const int sb = w >> 1, c = (w & 1) * 128 + (lane & 15) * 8, rg = 8 * (lane >> 4);
  mx_u4 v[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[i] = *(l128 *)(l + (32 * sb + rg + i) * MX_DG_LD + 2 * c);
  }
  mx_dg_qput(v, sb, N2, 2 * N2, fs_srd(dq + (size_t)rt * 2 * N2, 0xffffffffu, 0),
             fs_srd(cq + (size_t)rt * N2, 0xffffffffu, 0), fs_srd(cs + (size_t)rt / 32 * N2, 0xffffffffu, 0),
             n0c + c + (w & 1) * us, rg);
  __builtin_amdgcn_sched_barrier(0);
}

// dgrad2's SwiGLU backward over the 256 x 256 tile (cA, cB: rows r0, columns n0c and n0c + 256 of G; cC, cD: rows
// r0 + 128). Rounds are staged two at a time into the two buffers in lds (MX_DG_BUF bytes each), then quantized and
// stored; each round's G is loaded during the round before it. MEG: G and dG in Megatron's [gate; up] column order
// (the shared expert; n0c a gate column, cB's 128 after it, up I further). GATE: mx_dg_half's, sgl the gates in LDS.
template <typename ACT, bool MEG = false, bool GATE = false>
__device__ __forceinline__ void mx_swiglu_bwd(unsigned char *lds, const bf16 *g, uint8_t *dq, uint8_t *cq, uint8_t *cs,
                                              MxRtCT &cA, MxRtCT &cB, MxRtCT &cC, MxRtCT &cD, int r0, int wm, int n0c,
                                              int wn, int I, int lane,
                                              __attribute__((address_space(3))) float *sl = nullptr) {
  const int N2 = 2 * I, r1 = r0 + WARPS_ROW * REG_M, nb = MEG ? HALF_COL : 256;
  const uint32_t uo = MEG ? 2 * I : 256;
  const int us = MEG ? I - HALF_COL : 0;
  __attribute__((address_space(3))) uint8_t *l = (__attribute__((address_space(3))) uint8_t *)lds;
  const __attribute__((address_space(1))) uint8_t *g8 = (const __attribute__((address_space(1))) uint8_t *)fs_g(g);
  const uint32_t lo = (uint32_t)((wm * REG_M + (lane & 15)) * N2 + wn * REG_N + 4 * (lane >> 4)) * 2;
  MxGHalf gh[2];
  mx_g_load<0>(gh[0], g8 + (uint32_t)(r0 * N2 + n0c) * 2, lo, N2, uo);
  mx_g_load<1>(gh[1], g8 + (uint32_t)(r0 * N2 + n0c) * 2, lo, N2, uo);
  __syncthreads();
  mx_dg_round<true, ACT, GATE>(l, g8, cA, gh, wm, wn, N2, lane, lo, r0, n0c + nb, uo, sl, 0, 0);
  mx_dg_round<false, ACT, GATE>(l + MX_DG_BUF, g8, cB, gh, wm, wn, N2, lane, lo, 0, 0, uo, sl, 0, 1);
  __syncthreads();
  mx_dg_quant(l, dq, cq, cs, r0, n0c, N2, lane, us);
  mx_g_load<0>(gh[0], g8 + (uint32_t)(r1 * N2 + n0c) * 2, lo, N2, uo);
  mx_dg_quant(l + MX_DG_BUF, dq, cq, cs, r0, n0c + nb, N2, lane, us);
  mx_g_load<1>(gh[1], g8 + (uint32_t)(r1 * N2 + n0c) * 2, lo, N2, uo);
  __syncthreads();
  mx_dg_round<true, ACT, GATE>(l, g8, cC, gh, wm, wn, N2, lane, lo, r1, n0c + nb, uo, sl, HALF_ROW, 0);
  mx_dg_round<false, ACT, GATE>(l + MX_DG_BUF, g8, cD, gh, wm, wn, N2, lane, lo, 0, 0, uo, sl, HALF_ROW, 1);
  __syncthreads();
  mx_dg_quant(l, dq, cq, cs, r1, n0c, N2, lane, us);
  mx_dg_quant(l + MX_DG_BUF, dq, cq, cs, r1, n0c + nb, N2, lane, us);
}

__device__ __forceinline__ void mx_nn_mma(MxRtC &c, const MxRtA &a, const MxRtBK &b, fp8e8m0_4 sa, fp8e8m0_4 sb) {
  mma_ABt_scaled(c, a, b, c, &sa, &sb);
}

__device__ __forceinline__ void mx_nn_mma(MxRtCT &c, const MxRtA &a, const MxRtBK &b, fp8e8m0_4 sa, fp8e8m0_4 sb) {
  mx_mma_T(c, a, b, sa, sb);
}

// A: row-wise fp8 rows (a_ld bytes: K, then K / 32 scales); B: fp8 [K][N] with column-wise scales [K / 32][N].
template <typename CT>
__device__ __forceinline__ void mx_nn_tile(unsigned char *lds, const uint8_t *a_base, uint32_t a_rows, int a_ld, int K,
                                           int block_row, const uint8_t *b_base, const uint8_t *b_sc, int N,
                                           int block_col, CT &cA, CT &cB, CT &cC, CT &cD) {
  constexpr uint32_t SZ8 = sizeof(MxSt);
  constexpr int mx_copies = (int)SZ8 / (16 * NUM_THREADS);
  MxSt (&A8)[2][2] = *reinterpret_cast<MxSt (*)[2][2]>(lds);
  MxStK (&B8)[2][2] = *reinterpret_cast<MxStK (*)[2][2]>(lds + 4 * SZ8);
  const auto [ft, warp_m, warp_n, wv, lane] = mx_warp();
  const uint32_t a_lds_00 = mx_dma_base(&A8[0][0].data[0], wv), b_lds_00 = mx_dma_base(&B8[0][0].data[0], wv);
  a_base = fs_uniform_ptr(a_base), b_base = fs_uniform_ptr(b_base), b_sc = fs_uniform_ptr(b_sc);
  asm volatile("" : "+s"(a_base), "+s"(b_base), "+s"(b_sc));
  const i32x4 a_srd = fs_srd(a_base, a_rows * (uint32_t)a_ld, 0);
  const i32x4 b_srd = fs_srd(b_base, (uint32_t)K * N, N);
  uint32_t mx_swa[mx_copies], mx_swa1[mx_copies], mx_swb[mx_copies];
  {
    int ga_row[mx_copies], ga_col[mx_copies];
    gather_parts<MxSt>(lane, wv, ga_row, ga_col);
    const int gbase = block_row * BLOCK_ROW;
#pragma unroll
    for (int i = 0; i < mx_copies; i++) {
      mx_swa[i]  = MOE_SW_OFF(gbase + ga_row[i], a_ld, ga_col[i]);
      mx_swa1[i] = MOE_SW_OFF(gbase + HALF_ROW + ga_row[i], a_ld, ga_col[i]);
    }
    gather_parts<MxStK, true>(lane, wv, ga_row, ga_col);
#pragma unroll
    for (int i = 0; i < mx_copies; i++) {
      mx_swb[i] = MOE_SW_OFF(ga_row[i], N, ga_col[i]);
    }
  }
  const uint32_t b_so0 = (uint32_t)block_col * BLOCK_COL;
  const int q = ft % BLOCK_ROW;
  const uint32_t sc_voff = wv < NUM_WARPS / 2 ? (uint32_t)(block_row * BLOCK_ROW + q) * a_ld + K
                                              : (uint32_t)(q / 64) * N + 4 * (q % 64);
  const uint8_t *b_scb = b_sc + block_col * BLOCK_COL;
  uint32_t (&scl)[2][2 * BLOCK_ROW] = mx_scale_lds();
  const int k_iters = K / K_STEP_128;
  zero(cA);
  zero(cB);
  zero(cC);
  zero(cD);
  constexpr bool a_col = false, b_col = true;
#define MX_DMA_A(P, H, KS) mx_dma((KS) * K_STEP_128, (H) ? mx_swa1 : mx_swa, a_srd, a_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_DMA_B(P, H, KS) \
  mx_dma(b_so0 + (H) * HALF_COL + (KS) * K_STEP_128 * N, mx_swb, b_srd, b_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_SCALE(KS, P) \
  mx_scale_step(a_base, a_rows * (uint32_t)a_ld, 4 * (KS), b_scb, (uint32_t)(K / 32) * N, 4 * (KS) * N, sc_voff, P)
#define MX_MMA(C, B, SA, SB) mx_nn_mma(C, a, B, SA, SB)
#include "moe_mx_sched.inc"
}

// The forward's GEMM1 tile (moe_mx_kloop.inc's TN schedule, so G is bit-identical) on contiguous rows: A row-wise fp8
// rows (2K bytes: K values, then K / 32 scales), B = W1 [N][K] with row-wise scales [N][K / 32].
__device__ __forceinline__ void mx_g1_tile(unsigned char *lds, const uint8_t *a_base, uint32_t a_rows, int K, int N,
                                           int block_row, const uint8_t *b_base, const uint8_t *s_base, int block_col,
                                           MxRtCT &cA, MxRtCT &cB, MxRtCT &cC, MxRtCT &cD) {
  constexpr uint32_t SZ8 = sizeof(MxSt);
  constexpr int mx_copies = (int)SZ8 / (16 * NUM_THREADS);
  MxSt (&A8)[2][2] = *reinterpret_cast<MxSt (*)[2][2]>(lds);
  MxSt (&B8)[2][2] = *reinterpret_cast<MxSt (*)[2][2]>(lds + 4 * SZ8);
  const auto [ft, warp_m, warp_n, wv, lane] = mx_warp();
  const uint32_t a_lds_00 = mx_dma_base(&A8[0][0].data[0], wv), b_lds_00 = mx_dma_base(&B8[0][0].data[0], wv);
  const int a_ld = 2 * K, b_ld = K;
  a_base = fs_uniform_ptr(a_base), b_base = fs_uniform_ptr(b_base), s_base = fs_uniform_ptr(s_base);
  asm volatile("" : "+s"(a_base), "+s"(b_base), "+s"(s_base));
  const i32x4 a_srd = fs_srd(a_base, a_rows * (uint32_t)a_ld, 0);
  const i32x4 b_srd = fs_srd(b_base, (uint32_t)N * b_ld, b_ld);
  const uint32_t b_so0 = (uint32_t)block_col * BLOCK_COL * b_ld;
  uint32_t mx_swa[mx_copies], mx_swa1[mx_copies], mx_swb[mx_copies];
  {
    int ga_row[mx_copies], ga_col[mx_copies];
    gather_parts<MxSt>(lane, wv, ga_row, ga_col);
    const int gbase = block_row * BLOCK_ROW;
#pragma unroll
    for (int i = 0; i < mx_copies; i++) {
      mx_swa[i]  = MOE_SW_OFF(gbase + ga_row[i], a_ld, ga_col[i]);
      mx_swa1[i] = MOE_SW_OFF(gbase + HALF_ROW + ga_row[i], a_ld, ga_col[i]);
      mx_swb[i]  = MOE_SW_OFF(ga_row[i], b_ld, ga_col[i]);
    }
  }
  const uint8_t *sc_b = s_base + (size_t)block_col * BLOCK_COL * (K / 32);
  const uint32_t sc_voff = wv < NUM_WARPS / 2 ? (uint32_t)(block_row * BLOCK_ROW + ft) * a_ld + K
                                              : (uint32_t)(ft - BLOCK_ROW) * (uint32_t)(K / 32);
  uint32_t (&scl)[2][2 * BLOCK_ROW] = mx_scale_lds();
  const int k_iters = K / K_STEP_128;
  zero(cA);
  zero(cB);
  zero(cC);
  zero(cD);
  constexpr bool a_col = false, b_col = false;
#define MX_DMA_A(P, H, KS) mx_dma((KS) * K_STEP_128, (H) ? mx_swa1 : mx_swa, a_srd, a_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_DMA_B(P, H, KS) \
  mx_dma(b_so0 + (H) * HALF_COL * b_ld + (KS) * K_STEP_128, mx_swb, b_srd, b_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_SCALE(KS, P) mx_scale_step(a_srd, sc_b, K, sc_voff, 4 * (KS), P, BLOCK_COL)
#define MX_MMA(C, B, SA, SB) mx_mma_T(C, a, B, SA, SB)
#include "moe_mx_sched.inc"
}

// The tile cA..cD (MxRtC, as store_c_staged's) added to the bf16 tile at tb (row stride ld): each 128-row half staged
// in lds as fp32, added to the stored values in fp32 and rounded to nearest even once.
__device__ __forceinline__ void mx_tile_add(unsigned char *lds, const MxRtC &cA, const MxRtC &cB, const MxRtC &cC,
                                            const MxRtC &cD, int rf0, int cf0, int cf1, int lane, bf16 *tb, int ld) {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef unsigned u2 __attribute__((ext_vector_type(2)));
  typedef __attribute__((address_space(1))) u2 gu2;
  __attribute__((address_space(3))) float *st =
      (__attribute__((address_space(3))) float *)(__attribute__((address_space(3))) unsigned char *)lds;
  const __attribute__((address_space(3))) v4f *sv = (const __attribute__((address_space(3))) v4f *)st;
  int tid_c = fs_tid();
  asm volatile("" : "+v"(tid_c));
#pragma unroll
  for (int h = 0; h < 2; h++) {
    __syncthreads();
    sh_lds_f32(st, h ? cC : cA, rf0, cf0, BLOCK_COL, lane);
    sh_lds_f32(st, h ? cD : cB, rf0, cf1, BLOCK_COL, lane);
    __syncthreads();
    gu2 *d = (gu2 *)fs_g(tb + (size_t)h * HALF_ROW * ld);
#pragma unroll 4
    for (int k = 0; k < HALF_ROW * BLOCK_COL / 4 / NUM_THREADS; k++) {
      const int q = tid_c + k * NUM_THREADS;
      gu2 *p = (gu2 *)((__attribute__((address_space(1))) uint16_t *)d + (size_t)(q >> 6) * ld + (q & 63) * 4);
      const v4f v = sv[q];
      const u2 o = *p;
      const uint32_t w[2] = {o.x, o.y};
      uint32_t r[2];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const unsigned ul = __builtin_bit_cast(unsigned, v[2 * i] + __builtin_bit_cast(float, w[i] << 16));
        const unsigned uh = __builtin_bit_cast(unsigned, v[2 * i + 1] + __builtin_bit_cast(float, w[i] & 0xffff0000u));
        r[i] = ((ul + 0x7fffu + ((ul >> 16) & 1u)) >> 16) | ((uh + 0x7fffu + ((uh >> 16) & 1u)) & 0xffff0000u);
      }
      *p = u2{r[0], r[1]};
    }
  }
}

// A, B: fp8 [rows][Ca, Cb] with column-wise scales [rows / 32][Ca, Cb]; k_iters 0 or >= 2.
__device__ __forceinline__ void mx_nt_tile(unsigned char *lds, const uint8_t *a_base, const uint8_t *a_sc, int Ca,
                                           const uint8_t *b_base, const uint8_t *b_sc, int Cb, int row0, int k_iters,
                                           int nblk, int mblk, MxRtC &cA, MxRtC &cB, MxRtC &cC, MxRtC &cD) {
  constexpr uint32_t SZ8 = sizeof(MxStK);
  constexpr int mx_copies = (int)SZ8 / (16 * NUM_THREADS);
  MxStK (&A8)[2][2] = *reinterpret_cast<MxStK (*)[2][2]>(lds);
  MxStK (&B8)[2][2] = *reinterpret_cast<MxStK (*)[2][2]>(lds + 4 * SZ8);
  zero(cA);
  zero(cB);
  zero(cC);
  zero(cD);
  if (k_iters == 0) {
    return;
  }
  const auto [ft, warp_m, warp_n, wv, lane] = mx_warp();
  const uint32_t a_lds_00 = mx_dma_base(&A8[0][0].data[0], wv), b_lds_00 = mx_dma_base(&B8[0][0].data[0], wv);
  const uint32_t a_rows = (uint32_t)(row0 + K_STEP_128 * k_iters);
  a_base = fs_uniform_ptr(a_base), b_base = fs_uniform_ptr(b_base);
  asm volatile("" : "+s"(a_base), "+s"(b_base));
  const i32x4 a_srd = fs_srd(a_base, a_rows * (uint32_t)Ca, Ca);
  const i32x4 b_srd = fs_srd(b_base, a_rows * (uint32_t)Cb, Cb);
  uint32_t mx_swa[mx_copies], mx_swb[mx_copies];
  {
    int ga_row[mx_copies], ga_col[mx_copies];
    gather_parts<MxStK, true>(lane, wv, ga_row, ga_col);
#pragma unroll
    for (int i = 0; i < mx_copies; i++) {
      mx_swa[i] = MOE_SW_OFF(ga_row[i], Ca, ga_col[i]);
      mx_swb[i] = MOE_SW_OFF(ga_row[i], Cb, ga_col[i]);
    }
  }
  const uint32_t a_so0 = (uint32_t)row0 * Ca + (uint32_t)nblk * BLOCK_ROW;
  const uint32_t b_so0 = (uint32_t)row0 * Cb + (uint32_t)mblk * BLOCK_COL;
  const int q = ft % BLOCK_ROW;
  const uint32_t sc_voff = (uint32_t)(q / 64) * (wv < NUM_WARPS / 2 ? Ca : Cb) + 4 * (q % 64);
  const uint8_t *a_scb = a_sc + (size_t)(row0 / 32) * Ca + nblk * BLOCK_ROW;
  const uint8_t *b_scb = b_sc + (size_t)(row0 / 32) * Cb + mblk * BLOCK_COL;
  const uint32_t sc_rows = (uint32_t)(4 * k_iters);
  uint32_t (&scl)[2][2 * BLOCK_ROW] = mx_scale_lds();
  constexpr bool a_col = true, b_col = true;
#define MX_DMA_A(P, H, KS) \
  mx_dma(a_so0 + (H) * HALF_ROW + (KS) * K_STEP_128 * Ca, mx_swa, a_srd, a_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_DMA_B(P, H, KS) \
  mx_dma(b_so0 + (H) * HALF_COL + (KS) * K_STEP_128 * Cb, mx_swb, b_srd, b_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_SCALE(KS, P) \
  mx_scale_step(a_scb, sc_rows * Ca, 4 * (KS) * Ca, b_scb, sc_rows * Cb, 4 * (KS) * Cb, sc_voff, P)
#define MX_MMA(C, B, SA, SB) mma_ABt_scaled(C, a, B, C, &SA, &SB)
#include "moe_mx_sched.inc"
}

}
