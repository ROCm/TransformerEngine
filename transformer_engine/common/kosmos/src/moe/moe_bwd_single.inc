/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The training API's backward kernel body, textually included by moe_bwd_single_train_api (moe_train_api.cu;
// MOE_BWD_KS: the small-k kernel, moe_train_api_ks.cu), which defines ACT (the GLU activation), call (its SingleCall)
// and tbc (its TrainApiBwdCall).
#define SCALL(T, I) fs_uniform_ptr((T *)s_call[I])
#if MOE_BWD_KS
  constexpr bool KS = true;
#else
  constexpr bool KS = false;
#endif
  static_assert(NUM_THREADS == hk_comm::C_PUSH_THREADS, "the comm roles' row striding assumes 512 threads");
  static_assert(sizeof(SingleCall) == SC_N * sizeof(unsigned long long), "SingleCall is staged as SC_N words");
  // Staged, then read back wave-uniform at each use: kernarg values used across the loop would pin SGPRs.
  __shared__ unsigned long long s_call[SC_N];
  // Its operands in LDS (s_tbcp: TRAIN_BWD_*; s_rhdr: the route record's header, then acc), read back at each use.
  __shared__ unsigned long long s_tbcp[TRAIN_BWD_N];
  __shared__ int s_rhdr[TRAIN_BWD_ACC + 1];
  if (fs_tid() <= ROUTE_HDR_MAGIC) {
    s_rhdr[fs_tid()] = fs_g((const int *)tbc.route)[fs_tid()];
  }
  if (fs_tid() == 0) {
    s_tbcp[TRAIN_BWD_XP] = (unsigned long long)tbc.xp, s_tbcp[TRAIN_BWD_A] = (unsigned long long)tbc.a;
    s_tbcp[TRAIN_BWD_Y] = (unsigned long long)tbc.y, s_tbcp[TRAIN_BWD_X] = (unsigned long long)tbc.x;
    s_tbcp[TRAIN_BWD_CTR] = (unsigned long long)tbc.ctr, s_rhdr[TRAIN_BWD_ACC] = tbc.acc;
  }
  route_bwd_restore(args, tbc.route, tbc.ctr, tbc.seg, tbc.nseg, tbc.off_slot, tbc.scnt, tbc.sep, (int)gridDim.x);
#define SAVED_XP (*fs_lds(&s_rg) == 2 ? fs_args(args)->xrp[fs_args(args)->a.g.push.my_rank] \
                                      : fs_uniform_ptr((const bf16 *)*fs_lds(&s_tbcp[TRAIN_BWD_XP])))
#define SAVED_A (*fs_lds(&s_rg) ? (const bf16 *)fs_args(args)->ascr \
                                : fs_uniform_ptr((const bf16 *)*fs_lds(&s_tbcp[TRAIN_BWD_A])))
#define SAVED_NROW __builtin_amdgcn_readfirstlane(*fs_lds(&s_rhdr[ROUTE_HDR_NROW]))
  if (fs_tid() < SC_N) {
    s_call[threadIdx.x] = ((const unsigned long long *)&call)[threadIdx.x];
  }
  // What the forward kept (s_rg: 0 all, 1 Xp only, 2 nothing); G is the caller's or recomputed into dG's buffer.
  __shared__ int s_rg;
  {
    SArgsC *pa = fs_args(args);
    if (fs_tid() == SC_G) {
      // The record's kept tier: 2 all (G is the caller's), 1 Xp only, 0 nothing (G recomputed into dG's buffer).
      const int kept = s_rhdr[ROUTE_HDR_KEPT];
      s_rg = kept == 2 ? 0 : kept == 1 ? 1 : 2;
      s_call[SC_G] = (unsigned long long)(kept == 2 ? tbc.g : pa->dg);
    }
  }
  __syncthreads();
  const int wg = (int)blockIdx.x;
  const auto *wep = fs_g(fs_args(args)->wep);
  const int bcall = __builtin_amdgcn_readfirstlane(wep[0]);
  const int bwait = !__builtin_amdgcn_readfirstlane(wep[1]);
  if (wg < MOE_EP && threadIdx.x == 0) {
    hk_comm::moe_bwd_bfree_store(&args->a.g.push, wg, bcall);
  }
  // A0 and the tile counters' address from LDS at each use; held across the loop they cost registers.
  __shared__ unsigned long long s_a0[3];
  if (fs_tid() == 0) {
    s_a0[0] = (unsigned long long)A0.raw_ptr;
    s_a0[1] = (unsigned long long)(unsigned)A0.rows() | (unsigned long long)(unsigned)A0.cols() << 32;
    s_a0[2] = (unsigned long long)tile_counter;
  }
  __syncthreads();
#define TCX (fs_uniform_ptr((int *)*fs_lds(&s_a0[2])))
#define A0S single_a0<true>(A0, s_a0)
  __shared__ alignas(16) unsigned char s_tiles[LEGA_LDS_TILES];
  // The pass's state (moe_bwd_pass.inc). Over capacity: s_rr the round of s_nrb - 1, s_nrow its rows, s_wm each
  // expert's wgrad mode (1 store, 2 add, 0 none).
  __shared__ int s_rr, s_nrb, s_nrow, s_ngr;
  __shared__ unsigned char s_wm[MOE_BWD_MAX_EXPERTS];
  __shared__ int s_rgt[2];
  __shared__ int s_rb0[MOE_BWD_MAX_EXPERTS + 1];
  __shared__ int s_wkt[MOE_BWD_MAX_EXPERTS];
  __shared__ int s_ord[MOE_BWD_MAX_EXPERTS];
  __shared__ int s_seg[4 * MOE_BWD_MAX_EXPERTS + 1];
  __shared__ short s_segk[4 * MOE_BWD_MAX_EXPERTS];
  // Per expert whether every one of its recomputed row blocks has been acquired.
  __shared__ unsigned s_rgok[MOE_BWD_MAX_EXPERTS / 32];
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
  {
#define BWD_RMID round_mid
#include "moe_bwd_pass.inc"
#undef BWD_RMID
  }
#undef SCALL
#undef A0S
#undef TCX
#undef SAVED_XP
#undef SAVED_A
#undef SAVED_NROW
