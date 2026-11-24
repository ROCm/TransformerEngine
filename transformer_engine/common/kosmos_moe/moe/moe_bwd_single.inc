/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE backward kernel body, textually included by moe_bwd_single (moe_plain.cu) and moe_bwd_single_router
// (moe_router.cu), which define ROUTER, rcw (the router call's words) and ACT (the GLU activation); the router's score
// function is the translation unit's MoeScore, the shared expert's gate its MoeShGate. MOE_BWD_KS: the small-k kernels;
// MOE_BWD_PIN: with probs on input (k = 1), the kernel also defines pin_x and pin_w (the forward's X and weights).
// MOE_BWD_PART 1-5, the unfused ablation (bench only; router-free, unshared, bf16 combines): one call as five launches,
// dispatch (MXFP8: and the dY quantization), dgrad2 + wgrad2, dgrad1, wgrad1, combine and fold (with the reset); 0 or
// undefined: the one-launch backward.
#define SCALL(T, I) fs_uniform_ptr((T *)s_call[I])
// The shared expert's call words through fs_lds: a plain read hoists its LDS address into a VGPR across the loop.
#define SCL(T, I) fs_uniform_ptr((T *)*fs_lds(&s_call[I]))
#if MOE_BWD_KS
  constexpr bool KS = true;
#else
  constexpr bool KS = false;
#endif
#if MOE_BWD_PIN
  constexpr bool PIN = true;
#else
  constexpr bool PIN = false;
#endif
  static_assert(!PIN || (KS && !ROUTER), "probs on input: the small-k kernels, router-free");
#ifdef MOE_BWD_PART
  constexpr int BPART = MOE_BWD_PART;
#else
  constexpr int BPART = 0;
#endif
  static_assert(BPART == 0 || (BPART <= 5 && !ROUTER && !SH && !Q8),
                "the split backward is router-free and unshared, with bf16 combines");
  static_assert(NUM_THREADS == hk_comm::C_PUSH_THREADS, "the comm roles' row striding assumes 512 threads");
  static_assert(sizeof(SingleCall) == SC_X * sizeof(unsigned long long), "SingleCall is staged as SC_X words");
  static_assert(sizeof(SingleRouterCall) == 4 * sizeof(unsigned long long), "SingleRouterCall is staged as 4 words");
  // Staged, then read back wave-uniform at each use: kernarg values used across the loop would pin SGPRs.
  static_assert(sizeof(SingleShCall) == (SC_NS - SC_N) * sizeof(unsigned long long), "SingleShCall is staged whole");
  __shared__ unsigned long long s_call[SH ? SC_NS : ROUTER ? SC_N : SC_X];
#if MOE_BWD_SH
  {
    // SingleCallSh: the SingleCall words, then the shared words (staged from SC_N, after the router's).
    const int t = fs_tid();
    if (t < SC_X + SC_NS - SC_N) {
      s_call[t < SC_X ? t : t - SC_X + SC_N] = ((const unsigned long long *)&call)[t];
    }
  }
#else
  if (fs_tid() < SC_X) {
    s_call[threadIdx.x] = ((const unsigned long long *)&call)[threadIdx.x];
  }
#endif
  // bf16 training: what the forward kept (s_rg: 0 all, 1 Xp only, 2 nothing); G is the forward's, in the pool, or
  // recomputed into dG's buffer.
  __shared__ int s_rg;
  if constexpr (!MX) {
    SArgsC *pa = fs_args(args);
    if (fs_tid() == SC_G) {
      int rg = 0;
      if (BPART == 0 && pa->pool) {
        gcint *pr = fs_g((const int *)pa->pool) + 2 + 3 * pa->pslot;
        rg = pr[1] < 0 ? 2 : pr[2] && pr[2] == pr[1] ? 1 : 0;
      }
      s_rg = rg;
      if (pa->pool) {
        s_call[SC_G] = (unsigned long long)(rg ? pa->dg : pa->g);
      }
    }
  }
  // MXFP8 training: G, the kept block's (the forward's pool_commit); a forward in rounds kept nothing, and its backward
  // recomputes G, the column-wise A and with I < 1024 the column-wise Xp at the pool's head (mx_head), free until its
  // release.
  if constexpr (MX) {
    SArgsC *pa = fs_args(args);
    if (fs_tid() == SC_G && pa->pool) {
      const int hb = BPART == 0 && pa->nr > 1;
      s_call[SC_G] = hb ? (unsigned long long)(pa->pbase + (size_t)fs_g((const int *)pa->pool)[0] * 2 * pa->H)
                        : (unsigned long long)pa->g;
    }
  }
  if constexpr (ROUTER) {
    if (fs_tid() >= SC_X && fs_tid() < SC_N) {
      s_call[threadIdx.x] = rcw[threadIdx.x - SC_X];
    }
  }
  if constexpr (MX) {
    if (fs_tid() == 0) {
      mx_kp()[0] = (unsigned long long)mxc.x;
      mx_kp()[1] = (unsigned long long)mxc.w1s, mx_kp()[2] = (unsigned long long)mxc.w2s;
      mx_kp()[3] = (unsigned long long)A0.raw_ptr;
    }
  }
#if MOE_BWD_SH
  __shared__ unsigned long long s_mxsh[2];
  if constexpr (MX) {
    if (fs_tid() == 0) {
      mx_sh_scales(s_mxsh, mxc);
    }
  }
#endif
  __syncthreads();
  const int wg = (int)blockIdx.x;
  const auto *wep = fs_g(fs_args(args)->wep);
  const int bcall = __builtin_amdgcn_readfirstlane(wep[0]);
  const int bwait = !__builtin_amdgcn_readfirstlane(wep[1]);
  if (BPART <= 1 && wg < MOE_EP && threadIdx.x == 0) {
    hk_comm::moe_bwd_bfree_store(&args->a.g.push, wg, bcall);
  }
  // bf16, shared and small-k kernels (and MXFP8's rounds, SA0 in moe_bwd_pass.inc): A0 and the tile counters' address
  // (probs on input: also X and the weights) from LDS at each use; held across the loop they cost registers.
  constexpr bool RNDB0 = !MX && BPART == 0;
#if MOE_BWD_SH || MOE_BWD_KS
  constexpr bool SA00 = true;
#else
  constexpr bool SA00 = RNDB0;
#endif
#if MOE_BWD_PIN
  __shared__ unsigned long long s_a0[5];
#else
  __shared__ unsigned long long s_a0[3];
#endif
  if constexpr (RNDB0) {
    if (fs_tid() == 0) {
      s_a0[0] = (unsigned long long)A0.raw_ptr;
      s_a0[1] = (unsigned long long)(unsigned)A0.rows() | (unsigned long long)(unsigned)A0.cols() << 32;
      s_a0[2] = (unsigned long long)tile_counter;
#if MOE_BWD_PIN
      s_a0[3] = (unsigned long long)pin_x, s_a0[4] = (unsigned long long)pin_w;
#endif
    }
    __syncthreads();
  }
#define TCX (SA0 ? fs_uniform_ptr((int *)*fs_lds(&s_a0[2])) : tile_counter)
#define A0S single_a0<SA0>(A0, s_a0)
  // MXFP8: kittens' fp8 column-layout reads XOR the absolute LDS address, so the tiles start 2048-aligned.
  __shared__ alignas(MX ? 2048 : 16) unsigned char s_tiles[MX ? 2 * MX_DG_BUF : LEGA_LDS_TILES];
  // The pass's state (moe_bwd_pass.inc), declared once for both of MXFP8's copies. Over capacity: s_rr the round of
  // s_nrb - 1, s_nrow its rows, s_wm each expert's wgrad mode (1 store, 2 add, 0 none).
  __shared__ int s_rr, s_nrb, s_nrow, s_ngr;
  __shared__ unsigned char s_wm[BPART == 0 ? MOE_BWD_MAX_EXPERTS : 1];
  __shared__ int s_rgt[2];
  __shared__ int s_rb0[MOE_BWD_MAX_EXPERTS + 1];
  __shared__ int s_wkt[MOE_BWD_MAX_EXPERTS];
  __shared__ int s_ord[MOE_BWD_MAX_EXPERTS];
  __shared__ int s_seg[(MX ? 6 : 4) * MOE_BWD_MAX_EXPERTS + (ROUTER ? 3 : 1) + (SH ? 4 : 0)];
  __shared__ short s_segk[(MX ? 6 : 4) * MOE_BWD_MAX_EXPERTS + (ROUTER ? 2 : 0) + (SH ? 4 : 0)];
  // bf16 and MXFP8's rounds: per expert whether every one of its recomputed row blocks has been acquired.
  __shared__ unsigned s_rgok[MOE_BWD_MAX_EXPERTS / 32];
#if MOE_BWD_SH
  __shared__ int s_sdok;
#endif
  __shared__ int s_cap[NUM_XCDS_AFF];
  __shared__ int s_desc[6];
  __shared__ int s_acq_upto;
  __shared__ int s_zok;
  __shared__ unsigned s_dgok[MOE_BWD_MAX_EXPERTS / 32];
  __shared__ int s_arr;
  __shared__ unsigned s_cdone[hk_comm::CCOMB_MASK_W];
  __shared__ int s_csel;
  __shared__ int s_ntiles;
  __shared__ int s_nseg;
  __shared__ int s_nwg;
  constexpr bool SGATE = SH && MoeShGate;
  __shared__ alignas(16) float s_sg[SGATE ? BLOCK_ROW * (1 + BLOCK_COL / REG_N) : 4];
  if constexpr (MX && BPART == 0) {
    // MXFP8 training: a call that fits runs the round-free copy, a call over capacity the rounds' copy.
    if (__builtin_amdgcn_readfirstlane(fs_args(args)->nr) <= 1) {
      constexpr bool RNDB = false;
#define BWD_RMID round_mid_0
#include "moe_bwd_pass.inc"
#undef BWD_RMID
    } else {
      constexpr bool RNDB = true;
#define BWD_RMID round_mid_1
#include "moe_bwd_pass.inc"
#undef BWD_RMID
    }
  } else {
    constexpr bool RNDB = RNDB0;
#define BWD_RMID round_mid
#include "moe_bwd_pass.inc"
#undef BWD_RMID
  }
#undef SCALL
#undef SCL
#undef A0S
#undef TCX
