/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MXFP8 tile setup and K-loop (the TN schedule) of the includer's K, N, a_base, b_base, s_base, a_gather; a shared
// expert tile (SH && sh) reads local rows, and with b_meg its B halves are fc1's [gate; up] rows, I rows apart.
      const int a_ld = 2 * K, b_ld = K;
      a_base = fs_uniform_ptr(a_base), b_base = fs_uniform_ptr(b_base), s_base = fs_uniform_ptr(s_base);
      asm volatile("" : "+s"(a_base), "+s"(b_base), "+s"(s_base));
      const i32x4 a_srd = fs_srd(a_base, (uint32_t)((size_t)(SH && sh ? ap->T : A0.rows()) * a_ld), 0);
      const i32x4 b_srd = fs_srd(b_base, (uint32_t)((size_t)N * b_ld), b_ld);
      const uint32_t b_so0 = (uint32_t)block_col * (b_meg ? HALF_COL : BLOCK_COL) * b_ld;
      const auto [ft, warp_m, warp_n, wv, lane] = mx_warp();
      static_assert(sizeof(MxSt) == sizeof(As[0][0]) && sizeof(MxSt) == sizeof(Bs[0][0]), "fp8 tiles reuse As, Bs");
      static_assert(MxSt::underlying_subtile_bytes_per_thread == 16, "16-byte LDS-DMA");
      MxSt (&A8)[2][2] = *reinterpret_cast<MxSt (*)[2][2]>(&As[0][0]);
      MxSt (&B8)[2][2] = *reinterpret_cast<MxSt (*)[2][2]>(&Bs[0][0]);
      constexpr uint32_t SZ8 = sizeof(MxSt);
      constexpr int mx_copies = (int)SZ8 / (16 * NUM_THREADS);
      uint32_t mx_swa[mx_copies], mx_swa1[mx_copies], mx_swb[mx_copies];
      {
        int ga_row[mx_copies], ga_col[mx_copies];
        gather_parts<MxSt>(lane, wv, ga_row, ga_col);
        const int gbase = block_row * BLOCK_ROW;
#pragma unroll
        for (int i = 0; i < mx_copies; i++) {
          const int r0 = SH && sh ? gbase + ga_row[i] : a_gather[gbase + ga_row[i]];
          const int r1 = SH && sh ? gbase + HALF_ROW + ga_row[i] : a_gather[gbase + HALF_ROW + ga_row[i]];
          mx_swa[i]  = MOE_SW_OFF(r0, a_ld, ga_col[i]);
          mx_swa1[i] = MOE_SW_OFF(r1, a_ld, ga_col[i]);
          mx_swb[i]  = MOE_SW_OFF(ga_row[i], b_ld, ga_col[i]);
        }
      }
      const bool sc_a = wv < NUM_WARPS / 2;
      const uint8_t *sc_b = s_base + (size_t)block_col * (b_meg ? HALF_COL : BLOCK_COL) * (K / 32);
      const int b_sr = ft - BLOCK_ROW + (b_meg && ft >= BLOCK_ROW + HALF_COL ? I - HALF_COL : 0);
      const uint32_t sc_voff =
          sc_a ? (uint32_t)(SH && sh ? block_row * BLOCK_ROW + ft : a_gather[block_row * BLOCK_ROW + ft]) * a_ld + K
               : (uint32_t)b_sr * (uint32_t)(K / 32);
      uint32_t (&scl)[2][2 * BLOCK_ROW] = mx_scale_lds();
      const int k_iters = K / K_STEP_128;
      RT_CT cA, cB, cC, cD;
      zero(cA);
      zero(cB);
      zero(cC);
      zero(cD);
      constexpr bool a_col = false, b_col = false;
      const uint32_t b_hoff = (uint32_t)(b_meg ? I : HALF_COL) * b_ld;
      const int b_rows = b_meg ? I + HALF_COL : BLOCK_COL;
#define MX_DMA_A(P, H, KS) mx_dma((KS) * K_STEP_128, (H) ? mx_swa1 : mx_swa, a_srd, a_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_DMA_B(P, H, KS) \
  mx_dma(b_so0 + (H) * b_hoff + (KS) * K_STEP_128, mx_swb, b_srd, b_lds_00 + (2 * (P) + (H)) * SZ8)
#define MX_SCALE(KS, P) mx_scale_step(a_srd, sc_b, K, sc_voff, 4 * (KS), P, b_rows)
#define MX_MMA(C, B, SA, SB) mx_mma_T(C, a, B, SA, SB)
#include "moe_mx_sched.inc"
      const int lane_epi = mx_lane_opaque(ft);
