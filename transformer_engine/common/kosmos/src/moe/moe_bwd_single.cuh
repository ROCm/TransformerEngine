/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The one-launch MoE backward kernel: dispatch push, dgrad2/wgrad2, dgrad1/wgrad1, combine push and dX fold.
#pragma once

namespace hk_moe {

constexpr int SK_DG2 = 0, SK_WG2 = 1, SK_DG1 = 2, SK_WG1 = 3, SK_RDG = 4, SK_RWG = 5;
constexpr int SK_SDG2 = 6, SK_SWG2 = 7, SK_SDG1 = 8, SK_SWG1 = 9, SK_QDY = 10, SK_QX = 11, SK_RG = 12;

struct SingleCall {
  const unsigned short *dout;
  const bf16 *g;
  const bf16 *w1;
  const bf16 *w2;
  bf16 *dx;
  float *dtk;
  bf16 *dw1;
  bf16 *dw2;
  float *dprobs;
};

// SingleCall's words, staged in LDS (s_call) by the backward.
constexpr int SC_DOUT = 0, SC_G = 1, SC_W1 = 2, SC_W2 = 3, SC_DX = 4, SC_DTK = 5, SC_DW1 = 6, SC_DW2 = 7, SC_DPR = 8,
              SC_N = 9;

// Training backward operands: the forward's saved Xp, G, A, Y, X and record; the instance's counters, record segments
// and the re-plan's count slots and epochs. acc: 0 bf16 overwritten, 1 fp32 added into, 2 bf16 added into.
struct TrainApiBwdCall {
  const bf16 *xp, *g, *a, *y, *x;
  const uint8_t *route;
  int *ctr;
  const RouteSeg *seg;
  int *scnt;
  unsigned *sep;
  int nseg, off_slot, acc;
};
// Its copy in LDS: the pointers xp, a, y, x, ctr (s_tbcp, TRAIN_BWD_*), and the route record's header then acc
// (s_rhdr, acc at TRAIN_BWD_ACC).
constexpr int TRAIN_BWD_XP = 0, TRAIN_BWD_A = 1, TRAIN_BWD_Y = 2, TRAIN_BWD_X = 3, TRAIN_BWD_CTR = 4, TRAIN_BWD_N = 5,
              TRAIN_BWD_ACC = ROUTE_HDR_MAGIC + 1;

// The shared expert's backward state: not built; SingleArgs keeps it, zero, so the kernels' argument layout stays.
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

// Operands of a dgrad1-shaped NN tile: dXp = dG W1 (nn1; nnr, the fused router's dX, is not built).
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
  // r, sh, dyq through rdlt, xq8, qx_done, pbase and phr: the fused router's, the shared expert's and MXFP8's
  // operands, not built; zero.
  RouterArgs r;
  ShBwd sh;
  uint8_t *dyq, *dyc, *dycs, *dgc, *dgcs;
  const uint8_t *actc, *actcs, *xpc, *xpcs;
  int *qdy_done, xc_fwd;
  uint8_t *xcp[MOE_EP], *xcsp[MOE_EP], *xshp[MOE_EP];
  uint8_t *sdoq, *sdoc, *sdocs, *sxc, *sxcs, *sdgc, *sdgcs;
  const uint8_t *sactc, *sactcs;
  int *sq_done;
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
  uint8_t *xq8;
  int *qx_done;
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

// Every workgroup first: the record's segments back into the tables (one pass: also the receive slots), its counts and
// epoch into the re-plan's staging; then a barrier, as no table may be read before all are restored.
__device__ __noinline__ void route_bwd_restore(const SingleArgs *args, const uint8_t *route, int *ctr,
                                               const RouteSeg *seg, int nseg, int off_slot, int *scnt, unsigned *sep,
                                               int nwg) {
  const auto *ka = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  const gcint *h = fs_g((const int *)route);
  const int t = fs_tid(), nr = h[ROUTE_HDR_NR], crows = h[ROUTE_HDR_CROWS], EG = MOE_EP * ka->E;
  const unsigned epoch = (unsigned)h[ROUTE_HDR_EPOCH];
  for (int it = (int)blockIdx.x; it < nseg + 2; it += nwg) {
    if (it < nseg) {
      const RouteSeg sg = seg[it];
      wg_copy16(sg.dev, route + sg.off, 0, sg.bytes / 16);
    } else if (it == nseg) {
      if (nr == 1) {
        wg_copy16((void *)ka->b.c.slot, route + off_slot, 0, ((size_t)crows + 3) / 4);
      }
    } else {
      const gcint *rc = fs_g((const int *)(route + ROUTE_COUNTS_OFF));
      gint *sc = fs_g(scnt) + (size_t)(epoch & 1u) * MOE_EP * EG;
      for (int i = t; i < MOE_EP * EG; i += NUM_THREADS) {
        sc[i] = rc[i];
      }
      if (t < 2) {
        fs_g(sep)[t] = t ? 0u : epoch + 1u;
      }
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (t == 0) {
    (void)__hip_atomic_fetch_add(fs_g(ctr) + INST_CTR_BWD, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    plan_wait(fs_g(ctr) + INST_CTR_BWD, nwg, false);
  }
  __syncthreads();
}

// The training backward, last workgroup: frees the layer's block of the pool. Its zero operands are opaque: as
// constants they would be hoisted out of the round loop and held across it.
__device__ __forceinline__ void pool_release(const SingleArgs *args, int *scr) {
  const auto *a = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  const int z = single_opaque(0);
  (void)pool_update(fs_g(a->pool), a->pslot, a->prows, a->phead, z, a->pres, z, z, z, scr);
}

// bf16 training, a forward that kept nothing: this rank's token rows X pushed again to the experts'
// owners, into their xrp at the forward's rows, each row block's rows then release-added to the owner's xsg. XO: X and
// the passes are xo and nro, not the arguments' x0 and nr.
template <bool XO = false>
__device__ __forceinline__ void moe_bwd_xpush(SArgsC *ka, int w, const bf16 *xo = nullptr, int nro = 0) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  typedef const __attribute__((address_space(1))) v4i gcv4i;
  constexpr int RS = NUM_THREADS / kittens::WARP_THREADS;
  const int tid = (int)threadIdx.x, sg = tid / kittens::WARP_THREADS, stl = tid % kittens::WARP_THREADS;
  const int nv = ka->H / 8, k = ka->a.g.push.k, me = ka->a.g.push.my_rank, sigb = ka->a.g.push.sigb;
  const int nsp = ka->a.g.push.ncomm / MOE_EP, gslc = w / MOE_EP;
  const gcint *blks = fs_g((const int *)ka->a.g.push.blks);
  const gcint *send_slot = fs_g((const int *)ka->a.g.push.send_slot);
  const gcv4i *x = (gcv4i *)fs_uniform_ptr(XO ? xo : ka->x0);
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
    if ((XO ? nro : ka->nr) > 1) {
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

// The fold's inputs are in: every dXp row owed here and every dispatch workgroup's dtopkw and expert ids (wave 0). The
// arrival check's acquire fence also orders the dispatch count read before it.
__device__ __forceinline__ bool single_fold_ready(const SingleArgs *args, int *tc) {
  const int nd = __builtin_amdgcn_readfirstlane(
      __hip_atomic_load(tc + SINGLE_DISP_CTR, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  return nd >= fs_args(args)->a.g.push.ncomm && hk_comm::moe_bwd_combine_arrived(&args->b.c);
}

// The training API's kernels, built once per activation ACT (MoeAct): moe_train_api.cu (k in {6, 8, 10}) and
// moe_train_api_ks.cu (_ks: k in {1, 2, 4}).
template <typename ACT>
void moe_train_api_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const TrainApiCall &v);
template <typename ACT>
void moe_train_api_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                              const SingleArgs *args, const SingleCall &call, const TrainApiBwdCall &v);
template <typename ACT>
void moe_train_api_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const TrainApiCall &v);
template <typename ACT>
void moe_train_api_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCall &call, const TrainApiBwdCall &v);

}
