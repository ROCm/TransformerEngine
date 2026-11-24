/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The one-launch MoE backward kernel: dispatch push, dgrad2/wgrad2, dgrad1/wgrad1, combine push and dX fold.
#pragma once

namespace hk_moe {

constexpr int SK_DG2 = 0, SK_WG2 = 1, SK_DG1 = 2, SK_WG1 = 3, SK_RDG = 4, SK_RWG = 5;
constexpr int SK_SDG2 = 6, SK_SWG2 = 7, SK_SDG1 = 8, SK_SWG1 = 9, SK_QDY = 10, SK_QX = 11, SK_RG = 12;

// An MXFP8 backward's column-wise weight scales (W2c [E][H/32][I], W1c [E][2I/32][H]), the forward's bf16 X, and its
// row-wise W1 and scales (the GEMM1 recompute of a call in rounds).
struct MxBwdCall {
  const uint8_t *w1s, *w2s;
  const bf16 *x;
  const uint8_t *w1f, *w1fs;
};

// With the shared expert: its column-wise fc1 and fc2 scales (Ws1c [2I/32][H], Ws2c [H/32][I]).
struct MxBwdCallSh : MxBwdCall {
  const uint8_t *sw1s, *sw2s;
};

// A call's shared expert scales into s (a call without them has none).
__device__ __forceinline__ void mx_sh_scales(unsigned long long *, const MxBwdCall &) {}

__device__ __forceinline__ void mx_sh_scales(unsigned long long *s, const MxBwdCallSh &m) {
  s[0] = (unsigned long long)m.sw1s, s[1] = (unsigned long long)m.sw2s;
}

struct SingleCall {
  const unsigned short *dout;
  const bf16 *g;
  const bf16 *w1;
  const bf16 *w2;
  bf16 *dx;
  float *dtk;
  bf16 *dw1;
  bf16 *dw2;
  // The router-free backward's dprobs; with the fused router (no dprobs), the logit bias's gradient (null: none).
  float *dprobs;
  // main_grad accumulation of the routed and router weight gradients (and the logit bias's): 0 dw1, dw2, the router's
  // dwr and dlb overwritten (bf16; dlb fp32); 1 added into, fp32; 2 added into, bf16 (each element rounded once).
  long long acc;
};

// The fused router's operands of one backward: X, Wr, dWr and the aux-loss gradient scale (a device fp32 scalar).
struct SingleRouterCall {
  const bf16 *x;
  const bf16 *wr;
  bf16 *dwr;
  const float *aux_scale;
};
constexpr int SC_DOUT = 0, SC_G = 1, SC_W1 = 2, SC_W2 = 3, SC_DX = 4, SC_DTK = 5, SC_DW1 = 6, SC_DW2 = 7, SC_DPR = 8,
              SC_ACC = 9, SC_X = 10, SC_WR = 11, SC_DWR = 12, SC_AUX = 13, SC_N = 14;

// The shared expert's operands of one backward: X, fc1 ([gate; up]) and fc2 weights, the forward's Gs, and the weight
// gradients: bf16 and overwritten (acc 0), fp32 and added into (acc 1), or bf16 and added into (acc 2). sg: the
// forward's per-token gates (null: ungated), dsg: their gradient.
struct SingleShCall {
  const bf16 *x, *w1, *w2, *g;
  void *dw1, *dw2;
  long long acc;
  const float *sg;
  float *dsg;
};
// The shared kernels take one call record (an extra kernel argument costs registers).
struct SingleCallSh {
  SingleCall c;
  SingleShCall s;
};
constexpr int SC_SX = SC_N, SC_SW1 = SC_N + 1, SC_SW2 = SC_N + 2, SC_SG = SC_N + 3, SC_SDW1 = SC_N + 4,
              SC_SDW2 = SC_N + 5, SC_SACC = SC_N + 6, SC_SSG = SC_N + 7, SC_SDSG = SC_N + 8, SC_NS = SC_N + 9;

// fc1 wgrad split-K slices. The wgrad tiles park fp32 partials ([H][I], then [2I][H] per slice); finalize items
// reduce them in a fixed order (fc1: slice 0 + slice 1) into the bf16 or fp32 (accumulated) weight gradient.
constexpr int SH_KSPLIT = 2;
constexpr int SH_FIN_ROWS = 64;

// The shared expert's backward state: As (the forward's, per context), dGs, dXs and the fp32 partials (workspace),
// and the counters: dGs per row block, dXs per row block and in total, parked slices per wgrad tile, finalize claims.
// dsp: the gate gradient's partial sums, [I / 256][T].
struct ShBwd {
  bf16 *act, *dg, *dx;
  float *part, *dsp;
  int *dg_done, *dx_done, *dx_all, *wdone, *fctr;
  int T;
};

struct LegBArgs {
  hk_comm::CCombine c;
  LegAWgrad2 w;
  hk_comm::CSide s;
};

// Operands of a dgrad1-shaped NN tile: dXp = dG W1 (nn1) or the router's dX = dlogits Wr (nnr).
struct NnOps {
  bf16 *a;
  const int *gather;
  bf16 *out;
  int *pub;
  int rows, cols, kb;
};

struct SingleArgs {
  LegAArgs a;
  LegBArgs b;
  bf16 *dg;
  bf16 *dxp;
  const int *row_off;
  const int *cnt;
  const int *a_gather;
  const int *gid;
  int *dg_done;
  int H, I, E, nrow;
  NnOps nn1, nnr;
  RouterArgs r;
  ShBwd sh;
  uint8_t *dyq, *dyc, *dycs, *dgc, *dgcs;
  const uint8_t *actc, *actcs, *xpc, *xpcs;
  int *qdy_done, xc_fwd;
  uint8_t *xcp[MOE_EP], *xcsp[MOE_EP], *xshp[MOE_EP];
  // MXFP8 training's shared expert: dOut row-wise (2H per row) and column-wise, X and dGs column-wise, the forward's
  // column-wise As, and the count of quantized dOut and X blocks.
  uint8_t *sdoq, *sdoc, *sdocs, *sxc, *sxcs, *sdgc, *sdgcs;
  const uint8_t *sactc, *sactcs;
  int *sq_done;
  // MXFP8 with the fused router: dlogits transposed, [Ed][T] (the router dX tiles' NT operand).
  bf16 *rdlt;
  // As MoeFwdArgs: the layer's {epoch, xepoch}, the workspace's {backward count, latest call was a forward}.
  unsigned *ep, *wep;
  // bf16 training: G in the pool (set by the forward), the pool and the context's status word; slot and rows as
  // MoeFwdArgs. A forward that kept Xp only: A recomputed into ascr (G into dg), rg_done per row block, then the
  // recompute's claim count. One that kept nothing: its X (x0, set by the forward) pushed again into every owner's xrp,
  // counted on its xsg as dY on dsig.
  const bf16 *g;
  int *pool;
  int pslot, prows, phead, pres;
  bf16 *ascr;
  int *rg_done;
  const bf16 *x0;
  bf16 *xrp[MOE_EP];
  int *xsg[MOE_EP];
  // A forward over capacity ran nr > 1 passes (nr - 1 rounds): the backward runs the rounds, each from its tables (the
  // forward's plan with fa, the context's forward arguments) and rinfo (RINFO_*).
  const MoeFwdArgs *fa;
  int nr;
  int *rinfo;
  // MXFP8 training in rounds: the landed X rows quantized row-wise (2H bytes per row), and per (row block, 1024
  // columns) item the count of its quantization (row- and column-wise).
  uint8_t *xq8;
  int *qx_done;
  // MXFP8 training: this rank's pool, and the rows of a block at the capacity (its head, where a backward in rounds
  // recomputes G, the column-wise A and with I < 1024 the column-wise Xp).
  uint8_t *pbase;
  int phr;
};

// rinfo: the round's rows (nrow), the start barrier's arrive count, the round barrier's release word, then each
// local expert's wgrad mode (MoeLayout::wm).
constexpr int RINFO_NROW = 0, RINFO_ARR = 1, RINFO_GO = 2, RINFO_WM = 8;

constexpr int SINGLE_FOLD_CTR = NUM_XCDS_AFF + 1;
constexpr int SINGLE_ZERO_CTR = NUM_XCDS_AFF + 2;
constexpr int SINGLE_DISP_CTR = NUM_XCDS_AFF + 3;
constexpr int SINGLE_NCTR     = NUM_XCDS_AFF + 4;

typedef const __attribute__((address_space(4))) SingleArgs SArgsC;
typedef const __attribute__((address_space(1))) int gcint;

// MXFP8 training in rounds: the recompute's operands at the pool's head h (G there): q 0 the column-wise A, 1 its
// scales, 2 the column-wise Xp (I >= 1024: the workspace's), 3 its scales.
__device__ __forceinline__ const uint8_t *mx_head(SArgsC *ka, const bf16 *h, int q) {
  const int R = ka->phr, H = ka->H, I = ka->I;
  const uint8_t *b = (const uint8_t *)h;
  if (q >= 2 && !ka->xc_fwd) {
    return q == 2 ? ka->xpc : ka->xpcs;
  }
  return b + (q == 0 ? pool_mx_ac(R, I) : q == 1 ? pool_mx_acs(R, I) : q == 2 ? pool_mx_xc(R, I) : pool_mx_xcs(R, H, I));
}

template <typename T>
__device__ __forceinline__ T single_opaque(T v);

// make_srsrc's dwords (stride > 0: swizzled) from opaque constants: in the round loop its constants would be hoisted
// out of the loop and held across it.
__device__ __forceinline__ i32x4 single_srd(const void *p, uint32_t range, uint32_t stride) {
  const unsigned long long a = (unsigned long long)p;
  uint32_t flags = 0x110000, sw = stride ? ((stride & 0x3FFF) | 0xC000) << 16 : 0;
  asm volatile("" : "+s"(flags), "+s"(sw));
  return i32x4{__builtin_amdgcn_readfirstlane((int)(uint32_t)a),
               __builtin_amdgcn_readfirstlane((int)((uint32_t)(a >> 32) | sw)),
               __builtin_amdgcn_readfirstlane((int)range), __builtin_amdgcn_readfirstlane((int)flags)};
}

// S: A0 rebuilt from its LDS copy s (pointer, then rows | cols << 32); else A0 itself, by reference.
template <bool S>
__device__ __forceinline__ decltype(auto) single_a0(const gl<bf16, 1, 1, -1, -1> &A0, const unsigned long long *s) {
  if constexpr (S) {
    return gl<bf16, 1, 1, -1, -1>(fs_view(A0, (const bf16 *)*fs_lds(&s[0]), (int)(unsigned)*fs_lds(&s[1]),
                                          (int)(*fs_lds(&s[1]) >> 32)));
  } else {
    return (A0);
  }
}

// The training backward, last workgroup: frees the layer's block of the pool. Its zero operands are opaque: as
// constants they would be hoisted out of the round loop and held across it.
__device__ __forceinline__ void pool_release(const SingleArgs *args, int *scr) {
  const auto *a = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  const int z = single_opaque(0);
  (void)pool_update(fs_g(a->pool), a->pslot, a->prows, a->phead, z, a->pres, z, z, z, scr);
}

// Over capacity, before the rounds: every slot's dY dot as the pushes compute it (one row per wave), for the router's
// prep, which needs them all before the rounds push their rows. Q8: the forward's Y rows are MXFP8.
template <bool Q8 = false>
__device__ __forceinline__ void single_round_dots(SArgsC *ka, const unsigned short *dout, float *dtk, int w) {
  typedef unsigned short v8s __attribute__((ext_vector_type(8)));
  typedef const __attribute__((address_space(1))) v8s gcv8s;
  const int H = ka->H, nv = H / 8, k = ka->b.s.k, TK = ka->b.s.T * k, ft = fs_tid(), lane = ft % 64;
  const int nw = ka->a.g.push.ncomm * NUM_WARPS;
  gcv8s *x0 = (gcv8s *)fs_uniform_ptr(dout), *y0 = (gcv8s *)fs_uniform_ptr(ka->a.g.push.comby);
  for (int s = w * NUM_WARPS + ft / 64; s < TK; s += nw) {
    gcv8s *x = x0 + (size_t)(s / k) * nv, *y = y0 + (size_t)s * nv;
    float dot = 0.0f;
    for (int u = lane; u < nv; u += 64) {
      if constexpr (Q8) {
        const v8s a = x[u];
        float yf[8];
        moe_dq8((const __attribute__((address_space(1))) uint8_t *)y, u, H, yf);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dot = __builtin_fmaf(bits2f(a[e]), yf[e], dot);
        }
      } else {
        const v8s a = x[u], b = __builtin_nontemporal_load(y + u);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dot = __builtin_fmaf(bits2f(a[e]), bits2f(b[e]), dot);
        }
      }
    }
#pragma unroll
    for (int o = 32; o >= 1; o >>= 1) {
      dot += __shfl_xor(dot, o, 64);
    }
    if (lane == 0) {
      fs_g(dtk)[s] = dot;
    }
  }
}

// bf16 training, a forward that kept nothing: this rank's token rows X (PIN: bf16(w x)) pushed again to the experts'
// owners, into their xrp at the forward's rows, each row block's rows then release-added to the owner's xsg.
template <bool PIN>
__device__ __forceinline__ void moe_bwd_xpush(SArgsC *ka, int w) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  typedef const __attribute__((address_space(1))) v4i gcv4i;
  constexpr int RS = NUM_THREADS / kittens::WARP_THREADS;
  const int tid = (int)threadIdx.x, sg = tid / kittens::WARP_THREADS, stl = tid % kittens::WARP_THREADS;
  const int nv = ka->H / 8, k = ka->a.g.push.k, me = ka->a.g.push.my_rank, sigb = ka->a.g.push.sigb;
  const int nsp = ka->a.g.push.ncomm / MOE_EP, gslc = w / MOE_EP;
  const gcint *blks = fs_g((const int *)ka->a.g.push.blks);
  const gcint *send_slot = fs_g((const int *)ka->a.g.push.send_slot);
  const gcv4i *x = (gcv4i *)fs_uniform_ptr(ka->x0);
  for (int j = w % MOE_EP; j < ka->a.g.push.nblks; j += MOE_EP) {
    const int dst = __builtin_amdgcn_readfirstlane(blks[5 * j]), off = __builtin_amdgcn_readfirstlane(blks[5 * j + 2]);
    const int cnt = __builtin_amdgcn_readfirstlane(blks[5 * j + 3]);
    const int dbase = __builtin_amdgcn_readfirstlane(blks[5 * j + 4]);
    const int rlo = cnt * gslc / nsp, rhi = cnt * (gslc + 1) / nsp;
    if (rlo >= rhi) {
      continue;
    }
    gv4i *d = (gv4i *)fs_uniform_ptr(ka->xrp[dst]);
    // Rounds: the rows' slots too, into the owner's slot array (the forward's hold its last round's).
    int *ds = nullptr;
    if (ka->nr > 1) {
      const auto *fa = (const __attribute__((address_space(1))) MoeFwdArgs *)fs_uniform_ptr(ka->fa);
      ds = fs_uniform_ptr(fa->slot[dst]) + __builtin_amdgcn_readfirstlane(fs_g(fa->sbase)[j]);
    }
    for (int i = rlo + sg; i < rhi; i += RS) {
      const int srow = send_slot[off + i] / k;
      if (ds && stl == 0) {
        __builtin_nontemporal_store(send_slot[off + i], fs_g(ds) + i);
      }
      for (int u = stl; u < nv; u += kittens::WARP_THREADS) {
        v4i v = x[(size_t)srow * nv + u];
        if constexpr (PIN) {
          v = __builtin_bit_cast(v4i, moe_pin_scale(__builtin_bit_cast(v8bf, v), fs_g(ka->a.g.push.send_w)[off + i]));
        }
        __builtin_nontemporal_store(v, d + (size_t)(dbase + i) * nv + u);
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    const int blo = (dbase + rlo) / BLOCK_ROW, bhi = (dbase + rhi - 1) / BLOCK_ROW;
    if (tid <= bhi - blo) {
      const int blk = blo + tid, ra = blk * BLOCK_ROW - dbase, rz = ra + BLOCK_ROW;
      (void)__hip_atomic_fetch_add(fs_g(ka->xsg[dst]) + hk_comm::dsig_index(me, blk, sigb),
                                   min(rz, rhi) - max(ra, rlo), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// MXFP8 training in rounds: as moe_bwd_xpush, of x (the backward's X), with every unit's slots, and with padded
// segments each unit's last slice also writes its rows to 32 as zeros, counted as landed.
template <bool PIN>
__device__ __forceinline__ void mx_bwd_xpush(SArgsC *ka, const bf16 *x0, int w) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  typedef const __attribute__((address_space(1))) v4i gcv4i;
  constexpr int RS = NUM_THREADS / kittens::WARP_THREADS;
  const int tid = fs_tid(), sg = tid / kittens::WARP_THREADS, stl = tid % kittens::WARP_THREADS;
  const int nv = ka->H / 8, k = ka->a.g.push.k, me = ka->a.g.push.my_rank, sigb = ka->a.g.push.sigb;
  const int nsp = ka->a.g.push.ncomm / MOE_EP, gslc = w / MOE_EP, pad = ka->xc_fwd;
  const gcint *blks = fs_g((const int *)ka->a.g.push.blks);
  const gcint *send_slot = fs_g((const int *)ka->a.g.push.send_slot);
  const gcv4i *x = (gcv4i *)fs_uniform_ptr(x0);
  const auto *fa = (const __attribute__((address_space(1))) MoeFwdArgs *)fs_uniform_ptr(ka->fa);
  for (int j = w % MOE_EP; j < ka->a.g.push.nblks; j += MOE_EP) {
    const int dst = __builtin_amdgcn_readfirstlane(blks[5 * j]), off = __builtin_amdgcn_readfirstlane(blks[5 * j + 2]);
    const int cnt = __builtin_amdgcn_readfirstlane(blks[5 * j + 3]);
    const int dbase = __builtin_amdgcn_readfirstlane(blks[5 * j + 4]);
    const int rlo = cnt * gslc / nsp, rhi = cnt * (gslc + 1) / nsp;
    const int rend = rhi + (pad && cnt && gslc + 1 == nsp ? plan_pad32(cnt) - cnt : 0);
    if (rlo >= rend) {
      continue;
    }
    gv4i *d = (gv4i *)fs_uniform_ptr(ka->xrp[dst]);
    int *ds = fs_uniform_ptr(fa->slot[dst]) + __builtin_amdgcn_readfirstlane(fs_g(fa->sbase)[j]);
    for (int i = rlo + sg; i < rhi; i += RS) {
      const int srow = send_slot[off + i] / k;
      if (stl == 0) {
        __builtin_nontemporal_store(send_slot[off + i], fs_g(ds) + i);
      }
      for (int u = stl; u < nv; u += kittens::WARP_THREADS) {
        v4i v = x[(size_t)srow * nv + u];
        if constexpr (PIN) {
          v = __builtin_bit_cast(v4i, moe_pin_scale(__builtin_bit_cast(v8bf, v), fs_g(ka->a.g.push.send_w)[off + i]));
        }
        __builtin_nontemporal_store(v, d + (size_t)(dbase + i) * nv + u);
      }
    }
    for (int q = tid; q < (rend - rhi) * nv; q += NUM_THREADS) {
      __builtin_nontemporal_store(v4i{0, 0, 0, 0}, d + (size_t)(dbase + rhi) * nv + q);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    const int blo = (dbase + rlo) / BLOCK_ROW, bhi = (dbase + rend - 1) / BLOCK_ROW;
    if (tid <= bhi - blo) {
      const int blk = blo + tid, ra = blk * BLOCK_ROW - dbase, rz = ra + BLOCK_ROW;
      (void)__hip_atomic_fetch_add(fs_g(ka->xsg[dst]) + hk_comm::dsig_index(me, blk, sigb),
                                   min(rz, rend) - max(ra, rlo), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// The grid size is passed in: reading gridDim.x after the loop would keep the kernarg pointer live across it.
__device__ __forceinline__ bool single_last_wg(int *arrive, int nwg) {
  __shared__ int s_last;
  __syncthreads();
  if (threadIdx.x == 0) {
    __builtin_amdgcn_s_waitcnt(0);
    s_last = __hip_atomic_fetch_add(arrive, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == nwg - 1;
  }
  __syncthreads();
  return s_last;
}

// A uniform value the optimizer must re-derive after this point (not hoist or keep live across the loop).
template <typename T>
__device__ __forceinline__ T single_opaque(T v) {
  asm volatile("" : "+s"(v));
  return v;
}

__device__ __forceinline__ bool single_rows_landed(int x, bool spin, int s, gcint *g_cnt_by_src, gcint *g_row_off_e,
                                                   gcint *g_start_by_src, gcint *g_sig, int g_sigb) {
  if (s >= MOE_EP) {
    return true;
  }
  const int si = sig_index(s, x);
  const int c  = g_cnt_by_src[si];
  if (c <= 0) {
    return true;
  }
  const int st = g_row_off_e[x] + g_start_by_src[si];
  for (int blk = st / BLOCK_ROW; blk * BLOCK_ROW < st + c; blk++) {
    const int ra = blk * BLOCK_ROW, rz = ra + BLOCK_ROW;
    const int need = ((rz < st + c) ? rz : (st + c)) - ((ra > st) ? ra : st);
    const int sb   = hk_comm::dsig_index(s, blk, g_sigb);
    int n = 0;
    while (__hip_atomic_load(g_sig + sb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
      if (!spin) {
        return false;
      }
      spin_backoff(n);
    }
  }
  return true;
}

// The receive-arena rows this call reads but no peer writes (every layer's calls share the arena): each expert's rows
// up to its count rounded to K_STEP_64 (wgrad2 reads them directly) and the zero row that dgrad2 gathers.
// xr (bf16 training, a forward that kept nothing): the same rows of the region its X is pushed again into.
template <bool O = false>
__device__ __forceinline__ void single_zero_pads(const gl<bf16, 1, 1, -1, -1> &A0, SArgsC *ka, int z, int nz,
                                                 int *ctr, bf16 *xr, int nrow, int t = (int)threadIdx.x) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  const int ne = ka->E, nv = ka->H / 8;
  gv4i *base = (gv4i *)fs_uniform_ptr((bf16 *)A0.raw_ptr);
  int z0 = 0;
  if constexpr (O) {
    asm volatile("" : "+v"(z0));  // a constant zero vector would be hoisted out of the round loop
  }
  int g = 0;
  // A 64-bit index: a 32-bit one's zero high half would be hoisted out of the round loop.
  for (std::conditional_t<O, long long, int> e = 0; e <= ne; e++) {
    const int c  = e < ne ? fs_g(ka->cnt)[e] : 0;
    const int lo = e < ne ? fs_g(ka->row_off)[e] + c : nrow - 1;
    const int n  = e < ne ? (K_STEP_64 - c % K_STEP_64) % K_STEP_64 : 1;
    for (int j = ((z - g) % nz + nz) % nz; j < n; j += nz) {
      gv4i *row = base + (size_t)(lo + j) * nv;
      for (int u = t; u < nv; u += NUM_THREADS) {
        row[u] = v4i{z0, z0, z0, z0};
      }
      if (xr) {
        gv4i *xrow = (gv4i *)fs_uniform_ptr(xr) + (size_t)(lo + j) * nv;
        for (int u = t; u < nv; u += NUM_THREADS) {
          xrow[u] = v4i{z0, z0, z0, z0};
        }
      }
    }
    g += n;
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (t == 0) {
    (void)__hip_atomic_fetch_add(ctr, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// Expert e's column-wise Xp rows past its last group up to what wgrad1 reads (its count rounded to K_STEP_128, at
// least 256) and their scale rows, zeroed: other layers' calls write there.
__device__ __forceinline__ void mx_xcol_zero(SArgsC *ka, int e) {
  typedef __attribute__((address_space(1))) mx_u4 gv4i;
  const int c = fs_g(ka->cnt)[e];
  if (!c) {
    return;
  }
  const int ro = fs_g(ka->row_off)[e], H = ka->H, me = ka->a.g.push.my_rank;
  const int lo = ro + plan_pad32(c), hi = ro + K_STEP_128 * max(2, (c + K_STEP_128 - 1) / K_STEP_128);
  gv4i *q = (gv4i *)(fs_g(ka->xcp[me]) + (size_t)lo * H);
  gv4i *s = (gv4i *)(fs_g(ka->xcsp[me]) + (size_t)(lo / 32) * H);
  for (int u = fs_tid(); u < (hi - lo) * (H / 16); u += NUM_THREADS) {
    q[u] = mx_u4{0u, 0u, 0u, 0u};
  }
  for (int u = fs_tid(); u < (hi - lo) / 32 * (H / 16); u += NUM_THREADS) {
    s[u] = mx_u4{0u, 0u, 0u, 0u};
  }
}

// After its dY push: workgroup w's slice of the column-wise Xp groups and of the boundary rows to destination w % 8,
// each unit's groups counted on the destination's (source, expert) xcol word; then this rank's groups no source owns of
// experts w, w + ncomm.., and their zero rows, 1 << 20 on its own word of each expert. PIN: of bf16(w x), w the slot
// weights pw.
template <bool PIN = false>
__device__ __forceinline__ void mx_bwd_xcol(SArgsC *ka, const bf16 *x, int w, const float *pw = nullptr) {
  const int dst = w % MOE_EP, me = ka->a.g.push.my_rank, sigb = ka->a.g.push.sigb;
  (void)mx_xcol_units<false, PIN>(MxUnitsT{fs_g((const int *)ka->a.g.push.blks)}, ka->a.g.push.nblks,
                                  ka->a.g.push.ncomm, w, x, ka->a.g.push.send_slot, ka->a.g.push.k, ka->a.g.push.H, me,
                                  ka->xcp[dst], ka->xcsp[dst], ka->xshp[dst],
                                  ka->a.g.push.peers.sig[dst] + hk_comm::xcol_index(me, 0, sigb), pw);
  const MxSegT seg{fs_g(ka->a.g.start_by_src), fs_g(ka->a.g.cnt_by_src)};
  for (int e = w; e < ka->E; e += ka->a.g.push.ncomm) {
    mx_xcol_recv(seg, e, fs_g(ka->row_off)[e], ka->H, ka->xshp[me], ka->xcp[me], ka->xcsp[me]);
    mx_xcol_zero(ka, e);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      (void)__hip_atomic_fetch_add(fs_g(ka->a.g.push.peers.sig[me]) + hk_comm::xcol_index(me, e, sigb), 1 << 20,
                                   __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// Expert e's column-wise Xp count on source s's word this backward (none when the forward wrote them): its whole
// groups, and for this rank 1 << 20 for the groups no source owns and the zero rows.
__device__ __forceinline__ int mx_xcol_need(SArgsC *ka, int s, int e) {
  if (ka->xc_fwd) {
    return 0;
  }
  const int2 sc = MxSegT{fs_g(ka->a.g.start_by_src), fs_g(ka->a.g.cnt_by_src)}(e, s);
  return (sc.y ? MxSeg(sc.x, sc.y).owned() : 0) + (s == ka->a.g.push.my_rank ? 1 << 20 : 0);
}

// Wave 0: every source's dY rows (sig: X rows) of row block rb of expert e have landed, then acquired.
__device__ __forceinline__ void mx_wait_rows(SArgsC *ka, int lane, int e, int rb, const int *sig = nullptr) {
  if (lane < MOE_EP) {
    const int s = lane, si = sig_index(s, e), sigb = ka->a.g.push.sigb;
    const int c  = fs_g(ka->a.g.cnt_by_src)[si], st = fs_g(ka->a.g.start_by_src)[si];
    const int b0 = rb * BLOCK_ROW - fs_g(ka->row_off)[e], b1 = b0 + BLOCK_ROW;
    if (c > 0 && st < b1 && st + c > b0) {
      const int need = ((b1 < st + c) ? b1 : (st + c)) - ((b0 > st) ? b0 : st);
      gcint *sg = fs_g(sig ? sig : ka->a.g.sig) + hk_comm::dsig_index(s, rb, sigb);
      int n = 0;
      while (__hip_atomic_load(sg, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
        spin_backoff(n);
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

__device__ __forceinline__ void single_wait_rows(gcint *c, int need) {
  if (threadIdx.x == 0) {
    int n = 0;
    while (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < need) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
}

// The fold's inputs are in: every dXp row owed here and every dispatch workgroup's dtopkw and expert ids (wave 0). The
// arrival check's acquire fence also orders the dispatch count read before it.
__device__ __forceinline__ bool single_fold_ready(const SingleArgs *args, int *tc) {
  const int nd = __builtin_amdgcn_readfirstlane(
      __hip_atomic_load(tc + SINGLE_DISP_CTR, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  return nd >= fs_args(args)->a.g.push.ncomm && hk_comm::moe_bwd_combine_arrived(&args->b.c);
}

__device__ __forceinline__ bool single_router_dx_done(SArgsC *ka) {
  return __hip_atomic_load(fs_g(ka->r.bctr) + RC_RDG, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >=
         ka->r.T / BLOCK_ROW * (ka->H / BLOCK_COL);
}

// Dispatch workgroups, after their pushes: once every dispatch workgroup is done (all of dtopkw written; pushes wait
// only for peers to start), they claim the router's dlogits items. Router dX and dWr tiles wait for them. dlb (may be
// null): the logit bias's gradient, by the workgroup of the last item done.
// DT: also the transposed dlogits (rdlt). acc: dlb's main_grad accumulation (moe_router_dlb).
template <int SC, bool DT = false>
__device__ __forceinline__ void single_router_prep(const SingleArgs *args, float aux_g, float *dlb, int acc) {
  __shared__ int s_item;
  __shared__ int s_last;
  __syncthreads();
  if (threadIdx.x == 0) {
    gint *c = fs_g(fs_args(args)->r.bctr);
    __hip_atomic_fetch_add(c + RC_DISP, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    int n = 0;
    while (__hip_atomic_load(c + RC_DISP, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < fs_args(args)->a.g.push.ncomm) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(c + RC_DISP, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
#pragma unroll 1
  while (true) {
    __syncthreads();
    if (threadIdx.x == 0) {
      s_item = __hip_atomic_fetch_add(fs_g(fs_args(args)->r.bctr) + RC_PC, 1, __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    const int i = __builtin_amdgcn_readfirstlane(s_item);
    if (i >= fs_args(args)->r.T / RG_ROWS) {
      break;
    }
    moe_router_prep<SC, DT>(&args->r, i, fs_args(args)->b.s.k, aux_g,
                            __builtin_amdgcn_readfirstlane(fs_g(fs_args(args)->r.seedv)[0]), dlb != nullptr,
                            DT ? fs_args(args)->rdlt : nullptr);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      gint *c = fs_g(fs_args(args)->r.bctr);
      __hip_atomic_fetch_add(fs_g(fs_args(args)->r.rpdone) + i * RG_ROWS / BLOCK_ROW, 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
      s_last = __hip_atomic_fetch_add(c + RC_RP, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) ==
               fs_args(args)->r.T / RG_ROWS - 1;
      if (s_last) {
        (void)__hip_atomic_load(c + RC_RP, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
    if (dlb && s_last) {
      moe_router_dlb(&args->r, dlb, acc);
    }
  }
}

// Token t's shared gate gradient <dOut, Ys> = sum over I of dA act: its fc2 dgrad tiles' partials in a fixed order.
__device__ __forceinline__ void single_sgate_grad(SArgsC *ka, int t, float *dsg) {
  if (!dsg) {
    return;
  }
  const int nch = ka->I / BLOCK_COL, T = ka->sh.T;
  const auto *p = fs_g(ka->sh.dsp) + t;
  float v = 0.0f;
  for (int c = 0; c < nch; c++) {
    v += p[(size_t)c * T];
  }
  fs_g(dsg)[t] = v;
}

__device__ __forceinline__ int sh_fin_items(SArgsC *ka) {
  return 3 * (ka->H / BLOCK_COL) * (ka->I / BLOCK_COL) * (BLOCK_ROW / SH_FIN_ROWS);
}

// Finalize item i: SH_FIN_ROWS rows of one shared wgrad tile, once its slices are parked: fc2 dW = P, fc1 dW = P0 + P1,
// stored bf16 (acc 0) or added into the fp32 (acc 1) or bf16 (acc 2, rounded once) gradient.
__device__ __forceinline__ void sh_finalize_item(SArgsC *ka, int i, void *dw1, void *dw2, int acc) {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef unsigned v2u __attribute__((ext_vector_type(2)));
  typedef __attribute__((address_space(1))) v4f gv4f;
  const int H = ka->H, I = ka->I, n2 = (H / BLOCK_COL) * (I / BLOCK_COL), nb = BLOCK_ROW / SH_FIN_ROWS;
  const int tile = i / nb, band = i - tile * nb;
  const bool f1 = tile >= n2;
  const int t = f1 ? tile - n2 : tile, tn = f1 ? H / BLOCK_COL : I / BLOCK_COL, ld = f1 ? H : I;
  const size_t off = ((size_t)(t / tn) * BLOCK_ROW + band * SH_FIN_ROWS) * ld + (size_t)(t % tn) * BLOCK_COL;
  if (fs_tid() == 0) {
    gint *wd = fs_g(ka->sh.wdone) + tile;
    const int need = f1 ? SH_KSPLIT : 1;
    int n = 0;
    while (__hip_atomic_load(wd, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < need) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(wd, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  const float *p0 = fs_uniform_ptr(ka->sh.part) + (f1 ? (size_t)H * I : 0) + off;
  const size_t sstep = (size_t)2 * H * I;
  int tid_c = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_c));
  for (int q = tid_c; q < SH_FIN_ROWS * BLOCK_COL / 4; q += NUM_THREADS) {
    const size_t g = (size_t)(q >> 6) * ld + (q & 63) * 4;
    v4f v = *(const gv4f *)(fs_g(const_cast<float *>(p0)) + g);
    if (f1) {
      v += *(const gv4f *)(fs_g(const_cast<float *>(p0)) + sstep + g);
    }
    void *dw = f1 ? dw1 : dw2;
    if (acc == 1) {
      gv4f *d = (gv4f *)(fs_g((float *)dw) + off + g);
      *d = *d + v;
    } else {
      __attribute__((address_space(1))) v2u *d = (__attribute__((address_space(1))) v2u *)(fs_g((bf16 *)dw) + off + g);
      if (acc == 2) {
        const v2u m = *d;
        v[0] += __builtin_bit_cast(float, m[0] << 16), v[1] += __builtin_bit_cast(float, m[0] & 0xffff0000u);
        v[2] += __builtin_bit_cast(float, m[1] << 16), v[3] += __builtin_bit_cast(float, m[1] & 0xffff0000u);
      }
      *d = v2u{pk_bf16(v[0], v[1]), pk_bf16(v[2], v[3])};
    }
  }
}

// The shared expert's backward with the sigmoid gate (SingleShCall::sg), a compile-time policy: the shared translation
// units are built once without and once with it (moe_shared_bwdg.cu, moe_shared_router_bwdg.cu).
#ifdef MOE_SHARED_GATE
constexpr bool MoeShGate = true;
#else
constexpr bool MoeShGate = false;
#endif

// The launches of the kernels built once per activation ACT (MoeAct). The router-free, unshared bf16 kernels
// (moe_plain.cu):
template <typename ACT>
void moe_plain_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                          const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                          const MoeFwdArgs *args);
template <typename ACT>
void moe_plain_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                          const SingleCall &call);
// The kernels of moe_plain_launch_bwd and of moe_mx_launch_bwd without q8 (k in {6, 8, 10}), for their attributes.
template <typename ACT>
const void *moe_plain_bwd_kernel();
template <typename ACT>
const void *moe_mx_bwd_kernel();
// The fused router's kernels, once per score function SC (moe_router.cu); with the shared expert,
// moe_shared_router.cu.
template <typename ACT, int SC>
void moe_router_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                           const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                           const MoeFwdArgs *args, const RouterCall &rc);
template <typename ACT, int SC>
void moe_router_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                           const SingleCall &call, const SingleRouterCall &rc);
template <typename ACT, int SC>
void moe_shared_router_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                  const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                  const MoeFwdArgs *args, const RouterCall &rc, const ShCall &sc);
template <typename ACT, int SC, bool G>
void moe_shared_router_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                  const SingleArgs *args, const SingleCallSh &call, const SingleRouterCall &rc);
// The shared-expert kernels without the router (moe_shared.cu); G: the gated backward (moe_shared_bwdg.cu).
template <typename ACT>
void moe_shared_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                           const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                           const MoeFwdArgs *args, const ShCall &sc);
template <typename ACT, bool G>
void moe_shared_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                           const SingleCallSh &call);
// The small-k (k in {1, 2, 4}) kernels' launches, as above (moe_ks.cu, moe_router_ks.cu, moe_shared_ks.cu,
// moe_shared_router_ks.cu).
template <typename ACT>
void moe_ks_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
                       bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args);
template <typename ACT>
void moe_ks_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                       const SingleCall &call);
template <typename ACT, int SC>
void moe_router_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const RouterCall &rc);
template <typename ACT, int SC>
void moe_router_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCall &call, const SingleRouterCall &rc);
template <typename ACT>
void moe_shared_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const ShCall &sc);
template <typename ACT>
void moe_shared_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCallSh &call);
// Probs on input (k = 1; moe_ks_pin.cu, moe_shared_ks_pin.cu): the backward also takes the forward's X and weights.
template <typename ACT>
void moe_ks_pin_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                           const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                           const MoeFwdArgs *args);
template <typename ACT>
void moe_ks_pin_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                           const SingleCall &call, const bf16 *x, const float *w);
template <typename ACT>
void moe_shared_pin_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                  const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                  const MoeFwdArgs *args, const ShCall &sc);
template <typename ACT>
void moe_shared_pin_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                  const SingleArgs *args, const SingleCallSh &call, const bf16 *x, const float *w);
template <typename ACT, int SC>
void moe_shared_router_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                     const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                     const MoeFwdArgs *args, const RouterCall &rc, const ShCall &sc);
template <typename ACT, int SC>
void moe_shared_router_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                     const SingleArgs *args, const SingleCallSh &call, const SingleRouterCall &rc);
// The MXFP8 kernels (moe_mx.cu, moe_mx_q8.cu), once per kernel family V: 0 k in {6, 8, 10}, 1 the small-k kernels
// (moe_mx_ks.cu, moe_mx_q8_ks.cu), 2 probs on input (moe_mx_ks_pin.cu, moe_mx_q8_ks_pin.cu; pw: the slot weights).
template <typename ACT, int V>
void moe_mx_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
                       bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args,
                       const MxCall *mc, const MxTrainCall *mt, bool q8);
template <typename ACT, int V>
void moe_mx_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                       const SingleCall &call, const MxBwdCall &mc, bool q8, const float *pw);
template <typename ACT, int V>
void moe_mx_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                          const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                          const MoeFwdArgs *args, const MxTrainCall &mt);
template <typename ACT, int V>
void moe_mx_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                          const SingleCall &call, const MxBwdCall &mc, const float *pw);
// The MXFP8 training kernels with the shared expert (moe_mx_shared*.cu), once per kernel family V as above; G: the
// gated backward.
template <typename ACT, int V>
void moe_mx_shared_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const MxTrainCall &mt, const ShCallMx &sc, bool q8);
template <typename ACT, int V>
void moe_mx_shared_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const MxTrainCall &mt, const ShCallMx &sc);
template <typename ACT, int V, bool G>
void moe_mx_shared_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCallSh &call, const MxBwdCallSh &mc, bool q8, const float *pw);
template <typename ACT, int V, bool G>
void moe_mx_shared_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCallSh &call, const MxBwdCallSh &mc,
                                 const float *pw);
// The MXFP8 training kernels with the fused router (score SC, kernel family V 0 or 1); S: with the shared expert (sc its
// call, null without), G: its gated backward.
template <typename ACT, int SC, int V, bool S>
void moe_mx_router_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const RouterCall &rc, const MxTrainCall &mt, const ShCallMx *sc,
                              bool q8);
template <typename ACT, int SC, int V, bool S>
void moe_mx_router_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const RouterCall &rc, const MxTrainCall &mt,
                                 const ShCallMx *sc);
template <typename ACT, int SC, int V, bool S, bool G>
void moe_mx_router_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCall &call, const SingleShCall *sc, const SingleRouterCall &rc,
                              const MxBwdCall &mc, bool q8);
template <typename ACT, int SC, int V, bool S, bool G>
void moe_mx_router_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCall &call, const SingleShCall *sc,
                                 const SingleRouterCall &rc, const MxBwdCall &mc);
void moe_mx_launch_quant(const void *w, int nmat, int rows, int cols, bool col, void *q, void *s, hipStream_t st);

}
