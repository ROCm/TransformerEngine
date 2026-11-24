/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// MXFP8 GEMM2 tile (A = fp8 SwiGLU rows, B = W2; a shared expert tile: its local A rows and fc2), its Y store (MXFP8
// rows if Q8; the shared expert's Ys bf16) and publish; textually included.
      using RT_CT = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
      FsArgsC *ap = fs_args(args);
      const int H = ap->H, I = ap->I, K = I, N = H;
      constexpr bool b_meg = false;
      const uint8_t *a_base = fs_uniform_ptr((const uint8_t *)(SH && sh ? ap->sact : ap->act));
      const uint8_t *b_base = SH && sh ? (const uint8_t *)fs_uniform_ptr(fs_lds(&s_sh)->w2)
                                       : fs_uniform_ptr((const uint8_t *)FKP(const bf16, W2, FK_W2)) +
                                             (size_t)el * N * K;
      const uint8_t *s_base = SH && sh ? fs_uniform_ptr((const uint8_t *)*fs_lds(&s_shs[1]))
                                       : mx_ptr(2) + (size_t)el * N * (K / 32);
      const auto *a_gather = fs_g(ap->gid);
#include "moe_mx_kloop.inc"
      if (Q8 && !(SH && sh)) {
        uint8_t *qbase = (uint8_t *)fs_uniform_ptr(ap->c2) + (size_t)block_row * BLOCK_ROW * 2 * H;
        const int n0 = __builtin_amdgcn_readfirstlane((block_col * WARPS_COL * 2 + warp_n) * REG_N);
        const int n1 = n0 + WARPS_COL * REG_N;
        const int r0 = __builtin_amdgcn_readfirstlane(warp_m * REG_M);
        const int r1 = __builtin_amdgcn_readfirstlane((WARPS_ROW + warp_m) * REG_M);
        store_T16_mx(qbase, cA, r0, n0, H + n0 / 32, 2 * H, lane_epi);
        store_T16_mx(qbase, cB, r0, n1, H + n1 / 32, 2 * H, lane_epi);
        store_T16_mx(qbase, cC, r1, n0, H + n0 / 32, 2 * H, lane_epi);
        store_T16_mx(qbase, cD, r1, n1, H + n1 / 32, 2 * H, lane_epi);
      } else {
        bf16 *sbase = fs_uniform_ptr(SH && sh ? ap->sy : ap->c2) + (size_t)block_row * BLOCK_ROW * H;
        store_T16_bf16_4<true>(sbase, cA, cB, cC, cD, warp_m, block_col * WARPS_COL * 2 + warp_n, H, lane_epi);
      }
      // Every G2 tile MUST add to tile_done (a shared expert tile to sy_done).
      fs_tile_publish(SH && sh ? fs_g(fs_args(args)->sy_done) : fs_g(fs_args(args)->tile_done) + block_row);
