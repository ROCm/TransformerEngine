/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// MXFP8 GEMM1 + GLU tile (A = gathered fp8 Xp rows, B = W1; a shared expert tile: this rank's quantized X and fc1), its
// MXFP8 A store and publish; textually included.
      using RT_CT = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
      FsArgsC *ap = fs_args(args);
      const int H = ap->H, I = ap->I, K = H, N = 2 * I;
      const bool b_meg = SH && sh;
      const uint8_t *a_base = SH && sh ? (const uint8_t *)fs_uniform_ptr(ap->xq) + (PIN ? (size_t)ap->T * 2 * H : 0)
                                       : mx_ptr(3);
      const uint8_t *b_base = SH && sh ? (const uint8_t *)fs_uniform_ptr(fs_lds(&s_sh)->w1)
                                       : fs_uniform_ptr((const uint8_t *)FKP(const bf16, W1, FK_W1)) +
                                             (size_t)el * N * K;
      const uint8_t *s_base = SH && sh ? fs_uniform_ptr((const uint8_t *)*fs_lds(&s_shs[0]))
                                       : mx_ptr(1) + (size_t)el * N * (K / 32);
      const auto *a_gather = fs_g(ap->gather);
#include "moe_mx_kloop.inc"
      // A call in rounds keeps no routed G or column-wise A (its backward recomputes them).
      if (MXT && (!RND || (SH && sh))) {
        // A shared tile's Gs is in Megatron's [gate; up] column order: its up columns are I after its gate columns.
        bf16 *gbase = fs_uniform_ptr(SH && sh ? fs_lds(&s_sh)->g : (bf16 *)*fs_lds(&s_mxp[0])) +
                      (size_t)block_row * BLOCK_ROW * 2 * I;
        store_T16_bf16_4<false>(gbase, cA, cB, cC, cD, warp_m, (SH && sh ? 1 : 2) * block_col * WARPS_COL + warp_n,
                                2 * I, lane_epi, SH && sh ? I : WARPS_COL * REG_N);
      }
      swiglu_inplace<ACT, true>(cA, cB);
      swiglu_inplace<ACT, true>(cC, cD);
      const float *sg = SH && sh ? fs_uniform_ptr(fs_lds(&s_sh)->sg) : nullptr;
      if (SH && sh && sg) {
        // The shared expert's gate scales the bf16 activation rows, rounded to bf16 again.
        sg += (size_t)block_row * BLOCK_ROW;
        glu_gate_rows<true>(cA, sg + warp_m * REG_M, lane_epi);
        glu_gate_rows<true>(cC, sg + (WARPS_ROW + warp_m) * REG_M, lane_epi);
      }
      {
        uint8_t *abase =
            (uint8_t *)fs_uniform_ptr(SH && sh ? ap->sact : ap->act) + (size_t)block_row * BLOCK_ROW * 2 * I;
        const int n0 = __builtin_amdgcn_readfirstlane((block_col * WARPS_COL + warp_n) * REG_N);
        const int r0 = __builtin_amdgcn_readfirstlane(warp_m * REG_M);
        const int r1 = __builtin_amdgcn_readfirstlane((WARPS_ROW + warp_m) * REG_M);
        store_T16_mx(abase, cA, r0, n0, I + n0 / 32, 2 * I, lane_epi);
        store_T16_mx(abase, cC, r1, n0, I + n0 / 32, 2 * I, lane_epi);
        if (MXT && (!RND || (SH && sh))) {
          uint8_t *cq = fs_uniform_ptr(SH && sh ? ap->sactc : (uint8_t *)*fs_lds(&s_mxp[1])) +
                        (size_t)block_row * BLOCK_ROW * I;
          uint8_t *cs = fs_uniform_ptr(SH && sh ? ap->sactcs : (uint8_t *)*fs_lds(&s_mxp[2])) +
                        (size_t)block_row * (BLOCK_ROW / 32) * I;
          store_T16_mx_col(cq, cs, cA, r0, n0, I, lane_epi);
          store_T16_mx_col(cq, cs, cC, r1, n0, I, lane_epi);
        }
      }
      fs_tile_publish(fs_g(SH && sh ? fs_args(args)->sa_done : fs_args(args)->a_done) + block_row);
