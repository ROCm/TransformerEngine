/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE backward's comm roles: dispatch push, combine push and dX fold items.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

namespace hk_comm {

using hk_moe::MOE_EP;
using kittens::bf16;

typedef short v8s __attribute__((ext_vector_type(8)));

constexpr double C_DPUSH_WGMS_PER_GB        = 74.2;
constexpr double C_LEGA_GEMM_WGMS_PER_TFLOP = 282.0;
constexpr double C_DNCOMM_MARGIN            = 1.5;
constexpr int C_DNCOMM_MIN = 32;
constexpr int C_DNCOMM_MAX = 64;
__host__ __device__ inline int c_dispatch_ncomm(double push_gb, double gemm_tflop) {
  const double p = C_DNCOMM_MARGIN * push_gb * C_DPUSH_WGMS_PER_GB;
  const double g = gemm_tflop * C_LEGA_GEMM_WGMS_PER_TFLOP;
  const int n = (int)ceil(256.0 * p / (p + g) / hk_moe::NUM_XCDS_AFF) * hk_moe::NUM_XCDS_AFF;
  return n < C_DNCOMM_MIN ? C_DNCOMM_MIN : (n > C_DNCOMM_MAX ? C_DNCOMM_MAX : n);
}
constexpr double C_LEGA_CLAIM_US = 4.0;
// Runs of 2 when claiming a tile would take longer than computing it.
__host__ __device__ inline int c_wgrad2_run(double wg2_tiles, int wgs, double mean_k) {
  const double tile_us = 2.0 * 256.0 * 256.0 * mean_k / 1e12 * C_LEGA_GEMM_WGMS_PER_TFLOP * 1e3;
  const double claim_us = wg2_tiles / wgs * C_LEGA_CLAIM_US;
  return claim_us > tile_us ? 2 : 1;
}

constexpr double C_LEGB_GEMM_WGMS_PER_TFLOP = 229.0;
constexpr double C_CNCOMM_MARGIN            = 0.75;
constexpr int C_CNCOMM_MIN    = 24;
constexpr int C_CNCOMM_MAX    = 128;
__host__ __device__ inline int c_combine_ncomm(double push_gb, double gemm_tflop) {
  const double p = push_gb * C_DPUSH_WGMS_PER_GB, g = gemm_tflop * C_LEGB_GEMM_WGMS_PER_TFLOP;
  const int n = (int)ceil(C_CNCOMM_MARGIN * 256.0 * p / g / hk_moe::NUM_XCDS_AFF) * hk_moe::NUM_XCDS_AFF;
  return n < C_CNCOMM_MIN ? C_CNCOMM_MIN : (n > C_CNCOMM_MAX ? C_CNCOMM_MAX : n);
}

static_assert(C_DNCOMM_MIN % hk_moe::NUM_XCDS_AFF == 0 && C_DNCOMM_MIN % MOE_EP == 0 &&
                  C_DNCOMM_MAX % hk_moe::NUM_XCDS_AFF == 0 && C_DNCOMM_MAX % MOE_EP == 0,
              "dispatch ncomm must be a multiple of the XCD count and of EP (dst-minor affinity)");

constexpr int C_DSLICE_MIN = 64;
constexpr int C_DSLICE_MAX = 2;

constexpr int C_PUSH_THREADS  = 512;

__host__ __device__ __forceinline__ int dsig_index(int src, int blk, int sigb) {
  return src * sigb + blk;
}

// 16 B remote store, L2-bypass (aux 19 = sc0|nt|sc1) via the intrinsic (inline asm hides its vmcnt). dword3 must
// be 0x00020000 (0 drops the writes); row must be wave-uniform or the descriptor gets a waterfall loop.
typedef int cw_v4i __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void cw_pay_store(cw_v4i *row, int u, cw_v4i v) {
  __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void *)row, (short)0, (int)0xffffffffu, 0x00020000);
  __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(__attribute__((ext_vector_type(4))) unsigned int, v),
                                         rs, (unsigned)u * 16u, 0, 19);
}

struct CDispatch {
  const unsigned short *comby;
  const float *send_w;
  const hk_moe::SendBlk *blks;
  const int *send_slot;
  hk_moe::DispPeers peers;
  int nblks, my_rank, ncomm, H, k;
  int sigb;
};

// Struct-read pointers are generic (possibly divergent flat loads): read through global-address-space pointers and
// readfirstlane wave-uniform fields (fs_uniform_ptr), or the payload descriptor lands in VGPRs with a waterfall loop.
#define CW_G __attribute__((address_space(1)))
using hk_moe::bits2f;
using hk_moe::f2bf_bits;
using hk_moe::fs_uniform_ptr;
using hk_moe::moe_prod_burst;
using hk_moe::sig_index;
using hk_moe::spin_backoff;

// One dY row (bf16(w dOut)) pushed, and its partial <dOut, Y> returned.
template <int PB>
__device__ __forceinline__ float cw_push_row(const CW_G v8s *src, const CW_G v8s *ysrc, cw_v4i *dst, float wgt,
                                             int stl, int u_per_row) {
  constexpr int TSG = 64;
  float dot = 0.0f;
  int u = stl;
  for (; u + (PB - 1) * TSG < u_per_row; u += PB * TSG) {
    v8s x[PB], y[PB];
#pragma unroll
    for (int z = 0; z < PB; z++) {
      x[z] = src[u + z * TSG];
    }
#pragma unroll
    for (int z = 0; z < PB; z++) {
      y[z] = __builtin_nontemporal_load(ysrc + u + z * TSG);
    }
#pragma unroll
    for (int z = 0; z < PB; z++) {
      v8s o;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        o[e] = f2bf_bits(wgt * bits2f(x[z][e]));
      }
      cw_pay_store(dst, u + z * TSG, __builtin_bit_cast(cw_v4i, o));
    }
#pragma unroll
    for (int z = 0; z < PB; z++) {
      float yf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        dot = __builtin_fmaf(bits2f(x[z][e]), bits2f(y[z][e]), dot);
      }
    }
  }
  for (; u < u_per_row; u += TSG) {
    const v8s x = src[u];
    v8s y;
    float yf[8];
    y = __builtin_nontemporal_load(ysrc + u);
    v8s o;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      o[e] = f2bf_bits(wgt * bits2f(x[e]));
    }
    cw_pay_store(dst, u, __builtin_bit_cast(cw_v4i, o));
#pragma unroll
    for (int e = 0; e < 8; e++) {
      dot = __builtin_fmaf(bits2f(x[e]), bits2f(y[e]), dot);
    }
  }
  return dot;
}

// BFREE[s] (after the dispatch signals): the backward calls rank s had issued when its latest backward began, i.e.
// rank s no longer reads the receive or combine arena of any earlier call. Stored by s, monotone.
__host__ __device__ __forceinline__ int bfree_index(int src, int sigb) {
  return MOE_EP * sigb + src;
}

__device__ __forceinline__ void moe_bwd_bfree_store(const CDispatch *__restrict__ pp, int w, int bcall) {
  const CW_G CDispatch *p = (const CW_G CDispatch *)pp;
  CW_G int *bf = (CW_G int *)fs_uniform_ptr(p->peers.sig[w]) + bfree_index(p->my_rank, p->sigb);
  __hip_atomic_store(bf, bcall, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// yo: the forward's Y rows by token slot (the caller's saved y).
__device__ __forceinline__ void moe_bwd_dispatch_push_role(const CDispatch *__restrict__ pp, int w,
                                                           const unsigned short *dout, float *dtopkw_out, int bcall,
                                                           int bwait, const unsigned short *yo) {
  const CW_G CDispatch *p = (const CW_G CDispatch *)pp;
  constexpr int NT = C_PUSH_THREADS;
  const int tid = (int)threadIdx.x;
  const int H = p->H, nblks = p->nblks, ncomm = p->ncomm, my_rank = p->my_rank, k = p->k;
  // After a forward the wait is skipped: that forward's count barrier already ordered every peer's previous backward.
  if (tid == 0 && bwait) {
    const CW_G int *bf = (const CW_G int *)fs_uniform_ptr(p->peers.sig[my_rank]) + bfree_index(w % MOE_EP, p->sigb);
    int n = 0;
    while ((int)((unsigned)__hip_atomic_load(bf, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - (unsigned)bcall) < 0) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(bf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  const CW_G unsigned short *dOut = (const CW_G unsigned short *)fs_uniform_ptr(dout);
  const CW_G unsigned short *comby = (const CW_G unsigned short *)fs_uniform_ptr(yo);
  CW_G float *dtopkw = (CW_G float *)fs_uniform_ptr(dtopkw_out);
  const CW_G float *send_w = (const CW_G float *)p->send_w;
  const CW_G int *send_slot = (const CW_G int *)p->send_slot;
  const CW_G int *blks      = (const CW_G int *)p->blks;
  static_assert(sizeof(hk_moe::SendBlk) == 5 * sizeof(int), "SendBlk is read as 5 ints");
  const int u_per_row = H / 8;
  const int pb        = moe_prod_burst(H);
  constexpr int RS  = NT / 64;
  constexpr int TSG = 64;
  const int sg = tid / TSG, stl = tid % TSG;
  const int sigb = p->sigb;
  const int NSP  = ncomm / MOE_EP;
  const int gslc = w / MOE_EP;
  int item0 = 0;
  for (int j = w % MOE_EP; j < nblks; j += MOE_EP) {
    hk_moe::SendBlk b;
    b.dst = __builtin_amdgcn_readfirstlane(blks[5 * j + 0]);
    b.off = __builtin_amdgcn_readfirstlane(blks[5 * j + 2]);
    b.cnt = __builtin_amdgcn_readfirstlane(blks[5 * j + 3]);
    b.dstbase = __builtin_amdgcn_readfirstlane(blks[5 * j + 4]);
    if (b.cnt == 0) {
      continue;
    }
    const int smax = (NSP < C_DSLICE_MAX) ? NSP : C_DSLICE_MAX;
    int nsl = b.cnt / C_DSLICE_MIN;
    nsl = (nsl < 1) ? 1 : ((nsl > smax) ? smax : nsl);
    int sl = gslc - item0 % NSP;
    sl = (sl < 0) ? sl + NSP : sl;
    item0 += nsl;
    if (sl >= nsl) {
      continue;
    }
    const int nrs = (b.cnt + RS - 1) / RS;
    const int rlo = RS * (nrs * sl / nsl);
    const int rhi = (sl + 1 == nsl) ? b.cnt : RS * (nrs * (sl + 1) / nsl);
    if (rlo >= rhi) {
      continue;
    }
    bf16 *dbase = p->peers.recv[b.dst];
    int *dsig = p->peers.sig[b.dst] + dsig_index(my_rank, 0, sigb);
    for (int i0 = rlo; i0 < rhi; i0 += RS) {
      const int i = i0 + sg;
      if (i < rhi) {
        const int slot = __builtin_amdgcn_readfirstlane(send_slot[b.off + i]);
        const int srow = slot / k;
        const float wgt = send_w[b.off + i];
        const CW_G v8s *src = (const CW_G v8s *)(dOut + (size_t)srow * H);
        const CW_G v8s *ysrc = (const CW_G v8s *)(comby + (size_t)slot * H);
        const int drow = __builtin_amdgcn_readfirstlane(b.dstbase + i);
        cw_v4i *dst = fs_uniform_ptr((cw_v4i *)(dbase + (size_t)drow * H));
        float dot;
        switch (pb) {
          case 7: dot = cw_push_row<7>(src, ysrc, dst, wgt, stl, u_per_row); break;
          case 6: dot = cw_push_row<6>(src, ysrc, dst, wgt, stl, u_per_row); break;
          case 4: dot = cw_push_row<4>(src, ysrc, dst, wgt, stl, u_per_row); break;
          default: dot = cw_push_row<8>(src, ysrc, dst, wgt, stl, u_per_row); break;
        }
#pragma unroll
        for (int o = 32; o >= 1; o >>= 1) {
          dot += __shfl_xor(dot, o, 64);
        }
        if (stl == 0) {
          dtopkw[slot] = dot;
        }
      }
    }
    int rend = rhi;
    __syncthreads();
    const int blo = (b.dstbase + rlo) / hk_moe::BLOCK_ROW;
    const int bhi = (b.dstbase + rend - 1) / hk_moe::BLOCK_ROW;
    if (tid <= bhi - blo) {
      const int blk = blo + tid;
      const int ra = blk * hk_moe::BLOCK_ROW - b.dstbase, rz = ra + hk_moe::BLOCK_ROW;
      const int n = ((rz < rend) ? rz : rend) - ((ra > rlo) ? ra : rlo);
      (void)__hip_atomic_fetch_add(dsig + blk, n, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}
// Credit consumption mirrors the pushers' adds; unlike storing 0, an early add of the next call is kept.
__device__ __forceinline__ void moe_bwd_dispatch_credit(int *sig, const int *cnt_by_src, const int *start_by_src,
                                                        const int *row_off_e, int E, int sigb) {
  constexpr int BR = hk_moe::BLOCK_ROW;
  for (int q = (int)threadIdx.x; q < MOE_EP * E; q += C_PUSH_THREADS) {
    const int s = q / E, el = q % E;
    const int c = cnt_by_src[sig_index(s, el)];
    if (c <= 0) {
      continue;
    }
    const int r0 = row_off_e[el] + start_by_src[sig_index(s, el)], r1 = r0 + c;
    for (int blk = r0 / BR; blk * BR < r1; blk++) {
      const int lo = (r0 > blk * BR) ? r0 : blk * BR, hi = (r1 < (blk + 1) * BR) ? r1 : (blk + 1) * BR;
      (void)__hip_atomic_fetch_add(sig + dsig_index(s, blk, sigb), lo - hi, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

struct CCombBlk {
  int dst;
  int rb;
  int rb_hi;
  int off;
  int cnt;
};

struct CCombine {
  const unsigned short *Xp;
  const CCombBlk *blks;
  const int *src, *slot;
  const int *owed;
  int *tile_done;
  hk_moe::LegBPeers peers;
  int nblks, my_rank, ncomm, H;
};

constexpr int CCOMB_MASK_W = 8;

__device__ __forceinline__ int cw_units_of(int w, int nblks, int ncomm) {
  return (w < nblks) ? (nblks - w + ncomm - 1) / ncomm : 0;
}

__device__ __forceinline__ void moe_bwd_combine_init(const CCombine *__restrict__ pp, int w, unsigned *s_done) {
  const CW_G CCombine *p = (const CW_G CCombine *)pp;
  const CW_G int *blks = (const CW_G int *)p->blks;
  static_assert(sizeof(CCombBlk) == 5 * sizeof(int), "CCombBlk is read as 5 ints");
  const int ncomm = p->ncomm, nmine = cw_units_of(w, p->nblks, ncomm);
  if (threadIdx.x < 64) {
    const int lane = (int)threadIdx.x;
    for (int t0 = 0; t0 < 32 * CCOMB_MASK_W; t0 += 64) {
      const int t = t0 + lane;
      const unsigned long long m = __ballot(t < nmine && blks[5 * (w + t * ncomm) + 4] == 0);
      if (lane < 2) {
        s_done[t0 / 32 + lane] = (unsigned)(m >> (32 * lane));
      }
    }
  }
}

__device__ __forceinline__ int moe_bwd_combine_probe(const CCombine *__restrict__ pp, int w, int tN, unsigned *s_done,
                                                     int *s_sel) {
  const CW_G CCombine *p = (const CW_G CCombine *)pp;
  __syncthreads();  // every wave has read the previous probe's *s_sel
  if (threadIdx.x < 64) {
    const CW_G int *blks = (const CW_G int *)p->blks;
    CW_G int *td = (CW_G int *)p->tile_done;
    const int lane = (int)threadIdx.x, ncomm = p->ncomm, nmine = cw_units_of(w, p->nblks, ncomm);
    int sel = -2;
    for (int t0 = 0; t0 < nmine; t0 += 64) {
      const int t = t0 + lane;
      bool left = false, ripe = false;
      if (t < nmine && !((s_done[t >> 5] >> (t & 31)) & 1u)) {
        const int j = w + t * ncomm, hi = blks[5 * j + 2];
        left = ripe = true;
        for (int r = blks[5 * j + 1]; r <= hi; r++) {
          if (__hip_atomic_load(td + r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tN) {
            ripe = false;
            break;
          }
        }
      }
      const unsigned long long mr = __ballot(ripe), ml = __ballot(left);
      if (mr) {
        sel = t0 + __builtin_ctzll(mr);
        break;
      }
      if (ml) {
        sel = -1;
      }
    }
    if (lane == 0) {
      if (sel >= 0) {
        s_done[sel >> 5] |= 1u << (sel & 31);
        const int j = w + sel * ncomm, hi = blks[5 * j + 2];
        for (int r = blks[5 * j + 1]; r <= hi; r++) {
          (void)__hip_atomic_load(td + r, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      *s_sel = sel;
    }
  }
  __syncthreads();
  return __builtin_amdgcn_readfirstlane(*s_sel);
}

template <int PB>
__device__ __forceinline__ void cw_copy_row(const CW_G cw_v4i *src, cw_v4i *dst, int stl, int u_per_row) {
  constexpr int TSG = 64;
  int u = stl;
  for (; u + (PB - 1) * TSG < u_per_row; u += PB * TSG) {
    cw_v4i x[PB];
#pragma unroll
    for (int z = 0; z < PB; z++) {
      x[z] = src[u + z * TSG];
    }
#pragma unroll
    for (int z = 0; z < PB; z++) {
      cw_pay_store(dst, u + z * TSG, x[z]);
    }
  }
  for (; u < u_per_row; u += TSG) {
    cw_pay_store(dst, u, src[u]);
  }
}

// Unit j's dXp rows pushed back to the token owners' combine slots.
__device__ __forceinline__ void moe_bwd_combine_push_unit(const CCombine *__restrict__ pp, int j) {
  const CW_G CCombine *p = (const CW_G CCombine *)pp;
  constexpr int RS = C_PUSH_THREADS / 64, TSG = 64;
  const int tid = (int)threadIdx.x, sg = tid / TSG, stl = tid % TSG;
  const int H = p->H, u_per_row = H / 8, pb = moe_prod_burst(H);
  const CW_G int *blks = (const CW_G int *)p->blks;
  const CW_G int *src  = (const CW_G int *)p->src;
  const CW_G int *slot = (const CW_G int *)p->slot;
  const CW_G unsigned short *Xp = (const CW_G unsigned short *)p->Xp;
  const int dst = __builtin_amdgcn_readfirstlane(blks[5 * j + 0]);
  const int off = __builtin_amdgcn_readfirstlane(blks[5 * j + 3]);
  const int cnt = __builtin_amdgcn_readfirstlane(blks[5 * j + 4]);
  bf16 *dbase = fs_uniform_ptr(p->peers.comb[dst]);
  for (int i0 = 0; i0 < cnt; i0 += RS) {
    const int i = i0 + sg;
    if (i < cnt) {
      const int srow = __builtin_amdgcn_readfirstlane(src[off + i]);
      const int sl   = __builtin_amdgcn_readfirstlane(slot[off + i]);
      const CW_G cw_v4i *s = (const CW_G cw_v4i *)(Xp + (size_t)srow * H);
      cw_v4i *d = fs_uniform_ptr((cw_v4i *)(dbase + (size_t)sl * H));
      switch (pb) {
        case 7: cw_copy_row<7>(s, d, stl, u_per_row); break;
        case 6: cw_copy_row<6>(s, d, stl, u_per_row); break;
        case 4: cw_copy_row<4>(s, d, stl, u_per_row); break;
        default: cw_copy_row<8>(s, d, stl, u_per_row); break;
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    CW_G int *acc = (CW_G int *)fs_uniform_ptr(p->peers.acc[dst]);
    (void)__hip_atomic_fetch_add(acc + p->my_rank, cnt, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __forceinline__ void moe_bwd_combine_wait(const CCombine *__restrict__ pp) {
  const CW_G CCombine *p = (const CW_G CCombine *)pp;
  if (threadIdx.x < MOE_EP) {
    const int s = (int)threadIdx.x;
    CW_G int *acc = (CW_G int *)fs_uniform_ptr(p->peers.acc[p->my_rank]) + s;
    const int owed = ((const CW_G int *)p->owed)[s];
    int n = 0;
    while (__hip_atomic_load(acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < owed) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(acc, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    (void)__hip_atomic_fetch_add(acc, -owed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
}
// pk: the forward plan's packed routing (rank << 16 | expert, -1: dropped), dtk: dtopkw (written for routed slots).
struct CSide {
  const float *dtk;
  const int *pk;
  int T, k, E;
};
constexpr int CSIDE_FOLD_TOK = 16;
__host__ __device__ __forceinline__ int cside_fold_items(int T) {
  return (T + CSIDE_FOLD_TOK - 1) / CSIDE_FOLD_TOK;
}

// Item i's dprobs rows [t][E]: each routed slot's dtopkw at its expert, zero elsewhere (the fold is the rows' only
// writer); a dropped slot's expert field 0xFFFF matches no column.
template <int K>
__device__ __forceinline__ void cw_dprobs_rows(const CW_G CSide *s, int t0, int tid_f, float *dprobs) {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef int v2i __attribute__((ext_vector_type(2)));
  typedef float v2f __attribute__((ext_vector_type(2)));
  const int nq = s->E >> 2, m = min(CSIDE_FOLD_TOK, s->T - t0) * nq;
  const CW_G v2i *ib = (const CW_G v2i *)fs_uniform_ptr(s->pk + (size_t)t0 * K);
  const CW_G v2f *wb = (const CW_G v2f *)fs_uniform_ptr(s->dtk + (size_t)t0 * K);
  CW_G v4f *pb = (CW_G v4f *)fs_uniform_ptr(dprobs + (size_t)t0 * s->E);
  for (int x = tid_f; x < m; x += C_PUSH_THREADS) {
    const int tt = x / nq, c0 = (x - tt * nq) * 4;
    v4f o = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (K == 1) {
      const int e = (((const CW_G int *)ib)[tt] & 0xFFFF) - c0;
      const float w = ((const CW_G float *)wb)[tt];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        o[q] = e == q ? w : o[q];
      }
    } else {
      v2i id[K / 2];
      v2f wv[K / 2];
#pragma unroll
      for (int j = 0; j < K / 2; j++) {
        id[j] = ib[tt * (K / 2) + j];
        wv[j] = wb[tt * (K / 2) + j];
      }
#pragma unroll
      for (int j = 0; j < K; j++) {
        const int e = (id[j / 2][j % 2] & 0xFFFF) - c0;
        const float w = wv[j / 2][j % 2];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          o[q] = e == q ? w : o[q];
        }
      }
    }
    __builtin_nontemporal_store(o, pb + x);
  }
}

// dX of item i's tokens: bf16 of the sum of their k dXp rows (of the routed slots: a dropped slot's row is never
// written this call), and their dprobs rows.
template <int K>
__device__ __forceinline__ void cw_fold_item_k(const CW_G CSide *s, const bf16 *comb, int H, int i, bf16 *dx,
                                               float *dprobs) {
  constexpr int NQ = 2;
  const int T = s->T, k = K, nv = H >> 3, t0 = i * CSIDE_FOLD_TOK;
  const int n = min(CSIDE_FOLD_TOK, T - t0) * nv;
  const CW_G v8s *cb = (const CW_G v8s *)fs_uniform_ptr(comb + (size_t)t0 * k * H);
  CW_G v8s *ob = (CW_G v8s *)fs_uniform_ptr(dx + (size_t)t0 * H);
  int tid_f = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_f));  // opaque: else its lane offset is hoisted out of the caller's persistent loop
  for (int x0 = tid_f; x0 < n; x0 += NQ * C_PUSH_THREADS) {
    float acc[NQ][8];
#pragma unroll
    for (int q = 0; q < NQ; q++) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        acc[q][e] = 0.0f;
      }
    }
    v8s v[NQ][K];
    uint32_t m[NQ][K];
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const int x = min(x0 + q * C_PUSH_THREADS, n - 1);
      const int tt = x / nv, cv = x - tt * nv;
      const CW_G v8s *src = cb + (size_t)tt * k * nv + cv;
#pragma unroll
      for (int j = 0; j < K; j++) {
        m[q][j] = ~(uint32_t)(((const CW_G int *)fs_uniform_ptr(s->pk))[(size_t)(t0 + tt) * K + j] >> 31);
        v[q][j] = __builtin_nontemporal_load(src + (size_t)j * nv);
      }
    }
#pragma unroll
    for (int q = 0; q < NQ; q++) {
#pragma unroll
      for (int j = 0; j < K; j++) {
        float f[8];
        v[q][j] = hk_moe::moe_and8(v[q][j], m[q][j]);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          acc[q][e] += bits2f(v[q][j][e]);
        }
      }
    }
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const int x = x0 + q * C_PUSH_THREADS;
      if (x < n) {
        v8s o;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          o[e] = f2bf_bits(acc[q][e]);
        }
        __builtin_nontemporal_store(o, ob + x);
      }
    }
  }
  cw_dprobs_rows<K>(s, t0, tid_f, dprobs);
}

// KS: the small-k kernels (k in {1, 2, 4}); the others take k in {6, 8, 10}.
template <bool KS>
__device__ __forceinline__ void cw_fold_item_t(const CSide *__restrict__ sp, const bf16 *comb, int H, int i, bf16 *dx,
                                               float *dprobs) {
  const CW_G CSide *s = (const CW_G CSide *)sp;
  if constexpr (KS) {
    switch (__builtin_amdgcn_readfirstlane(s->k)) {
      case 1: cw_fold_item_k<1>(s, comb, H, i, dx, dprobs); break;
      case 2: cw_fold_item_k<2>(s, comb, H, i, dx, dprobs); break;
      default: cw_fold_item_k<4>(s, comb, H, i, dx, dprobs); break;
    }
  } else {
    switch (__builtin_amdgcn_readfirstlane(s->k)) {
      case 6: cw_fold_item_k<6>(s, comb, H, i, dx, dprobs); break;
      case 8: cw_fold_item_k<8>(s, comb, H, i, dx, dprobs); break;
      default: cw_fold_item_k<10>(s, comb, H, i, dx, dprobs); break;
    }
  }
}

template <bool KS>
__device__ __forceinline__ void cw_fold_item(const CSide *__restrict__ sp, const bf16 *comb, int H, int i, bf16 *dx,
                                             float *dprobs) {
  cw_fold_item_t<KS>(sp, comb, H, i, dx, dprobs);
}

__device__ __forceinline__ bool moe_bwd_combine_arrived(const CCombine *__restrict__ pp) {
  const CW_G CCombine *p = (const CW_G CCombine *)pp;
  const CW_G int *acc  = (const CW_G int *)fs_uniform_ptr(p->peers.acc[p->my_rank]);
  const CW_G int *owed = (const CW_G int *)p->owed;
  int s = (int)threadIdx.x & 63;
  asm volatile("" : "+v"(s));  // opaque: else the s < MOE_EP mask is kept live across the caller's loop
  const bool late = s < MOE_EP && __hip_atomic_load(acc + s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < owed[s];
  if (__ballot(late)) {
    return false;
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  return true;
}

#undef CW_G

}
