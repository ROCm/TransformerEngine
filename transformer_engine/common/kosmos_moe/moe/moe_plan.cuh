/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE forward's start phase: plans a routing in device memory and builds every routing-derived table in place.
#pragma once

namespace hk_moe {

struct MoePlanRec {
  int status;
  int Mpad, Mpad_max, nt1, nt2;
  int offrows, dself, crows, cself;
  int rounds, forced;
  int peak;
};

static_assert(sizeof(RbBlk) == 4 * sizeof(int), "RbBlk is written as 4 ints");
static_assert(PLAN_MAXU <= NUM_THREADS, "one expert and one unit per thread");
static_assert(PLAN_IDS_OFF >= NUM_WARPS * PLAN_MAXU * (int)sizeof(int) &&
                  PLAN_RUN_OFF >= PLAN_IDS_OFF + (PLAN_HEAD + PLAN_CH) * (int)sizeof(int) && PLAN_HEAD > 10 &&
                  PLAN_LAYOUT_OFF >= PLAN_RUN_OFF + PLAN_MAXU * (int)sizeof(int),
              "chunk counts, chunk ids, running counts, then the layout");
static_assert(MOE_EP * PLAN_MAXU * (int)sizeof(int) <= PLAN_IDS_OFF &&
                  PLAN_IDS_OFF + 2 * PLAN_MAXU * (int)sizeof(int) <= PLAN_LAYOUT_OFF,
              "a round's clipped counts, then MXFP8's send offsets and starts (plan_clip), below the layout");

// The thread id; O: opaque (bf16, whose start phase is in the round loop: lane masks of threadIdx would be hoisted
// out of it and held across the tile loop).
template <bool O>
__device__ __forceinline__ int plan_tid() {
  return O ? fs_tid() : (int)threadIdx.x;
}

// v from the lane o below (lane: this lane's id, opaque in MXFP8's rounds: __shfl_up's own lane id and the shuffle
// addresses derived from it would be hoisted out of the round loop).
template <bool OS>
__device__ __forceinline__ int plan_shfl_up(int v, int o, int lane) {
  if constexpr (OS) {
    return __builtin_amdgcn_ds_bpermute((lane >= o ? lane - o : lane) << 2, v);
  } else {
    return __shfl_up(v, o, kittens::WARP_THREADS);
  }
}

// Exclusive scan of one value per thread over the workgroup; *total gets the sum.
template <bool O = false, bool OS = false>
__device__ __forceinline__ int plan_scan(int x, int *tmp, int *total) {
  const int tid = plan_tid<O>(), lane = tid % kittens::WARP_THREADS, wv = tid / kittens::WARP_THREADS;
  int v = x;
#pragma unroll
  for (int o = 1; o < kittens::WARP_THREADS; o <<= 1) {
    const int y = plan_shfl_up<OS>(v, o, lane);
    v += lane >= o ? y : 0;
  }
  if (lane == kittens::WARP_THREADS - 1) {
    tmp[wv] = v;
  }
  __syncthreads();
  int base = 0, tot = 0;
#pragma unroll
  for (int q = 0; q < NUM_WARPS; q++) {
    const int t = tmp[q];
    base += q < wv ? t : 0;
    tot += t;
  }
  __syncthreads();
  *total = tot;
  return base + v - x;
}

// Two exclusive scans at once (as plan_scan); *ta and *tb get the sums.
template <bool O = false, bool OS = false>
__device__ __forceinline__ int2 plan_scan2(int a, int b, int *tmp, int *ta, int *tb) {
  const int tid = plan_tid<O>(), lane = tid % kittens::WARP_THREADS, wv = tid / kittens::WARP_THREADS;
  int va = a, vb = b;
#pragma unroll
  for (int o = 1; o < kittens::WARP_THREADS; o <<= 1) {
    const int ya = plan_shfl_up<OS>(va, o, lane), yb = plan_shfl_up<OS>(vb, o, lane);
    va += lane >= o ? ya : 0;
    vb += lane >= o ? yb : 0;
  }
  if (lane == kittens::WARP_THREADS - 1) {
    tmp[wv] = va;
    tmp[NUM_WARPS + wv] = vb;
  }
  __syncthreads();
  int basea = 0, baseb = 0, suma = 0, sumb = 0;
#pragma unroll
  for (int q = 0; q < NUM_WARPS; q++) {
    const int xa = tmp[q], xb = tmp[NUM_WARPS + q];
    basea += q < wv ? xa : 0;
    baseb += q < wv ? xb : 0;
    suma += xa;
    sumb += xb;
  }
  __syncthreads();
  *ta = suma;
  *tb = sumb;
  return int2{basea + va - a, baseb + vb - b};
}

// e's rank among the lanes below this one holding the same e >= 0 (lanes agreeing with it on every bit).
__device__ __forceinline__ int plan_wave_rank(int e) {
  static_assert(PLAN_MAXU <= 1 << 9, "expert ids fit 9 bits");
  const int lane = (int)threadIdx.x % kittens::WARP_THREADS;
  unsigned long long m = __ballot(e >= 0);
#pragma unroll
  for (int b = 0; b < 9; b++) {
    const unsigned long long v = __ballot((e >> b) & 1);
    m &= (e >> b) & 1 ? v : ~v;
  }
  return __popcll(m & ((1ull << lane) - 1ull));
}

__device__ __forceinline__ int plan_pad(int n) {
  return (n + EXPERT_PADDING - 1) / EXPERT_PADDING * EXPERT_PADDING;
}

template <bool O = false>
__device__ __forceinline__ void plan_zero_rows(bf16 *base, int lo, int hi, int H) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  __attribute__((address_space(1))) v4i *z = (__attribute__((address_space(1))) v4i *)fs_g(base + (size_t)lo * H);
  int c = 0;
  if constexpr (O) {
    asm volatile("" : "+v"(c));  // a constant zero vector would be hoisted out of the round loop
  }
  for (int u = plan_tid<O>(); u < (hi - lo) * (H / 8); u += NUM_THREADS) {
    z[u] = v4i{c, c, c, c};
  }
}

__device__ __forceinline__ void plan_wait(gint *c, int want, bool sys) {
  int n = 0;
  if (sys) {
    while ((int)((unsigned)__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - (unsigned)want) < 0) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    while (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// Chunk c of this rank's routing: sanitize (an out-of-range id gets weight 0), per-expert counts, and each entry's
// stable rank among the chunk's entries of its expert, packed as rank << 16 | expert into p.pk (-1: dropped).
__device__ __forceinline__ int plan_chunk(FsArgsC *a, const int *idx, const float *w, int c, char *scr,
                                          int (&pk)[PLAN_SUB], float (&wt)[PLAN_SUB]) {
  const int tid = (int)threadIdx.x, k = a->topk, TK = a->T * k, EG = MOE_EP * a->E_loc, x0 = c * PLAN_CH;
  const int wv = tid / kittens::WARP_THREADS;
  int *wc = (int *)scr, *se = (int *)(scr + PLAN_IDS_OFF) + PLAN_HEAD, *run = (int *)(scr + PLAN_RUN_OFF);
  const auto *gi = fs_g(idx);
  int eg[PLAN_SUB], bad = 0;
#pragma unroll
  for (int s = 0; s < PLAN_SUB; s++) {
    const int x = x0 + s * NUM_THREADS + tid;
    eg[s] = x < TK ? gi[x] : -1;
    wt[s] = x < TK ? fs_g(w)[x] : 0.0f;
  }
  const int hx = x0 - PLAN_HEAD + tid;
  const int hv = tid < PLAN_HEAD && hx >= 0 ? gi[hx] : -1;
#pragma unroll
  for (int s = 0; s < PLAN_SUB; s++) {
    se[s * NUM_THREADS + tid] = eg[s];
  }
  if (tid < PLAN_HEAD) {
    se[tid - PLAN_HEAD] = hv;
  }
  if (tid < EG) {
    run[tid] = 0;
  }
  __syncthreads();
#pragma unroll
  for (int s = 0; s < PLAN_SUB; s++) {
    const int x = x0 + s * NUM_THREADS + tid;
    if (x < TK) {
      if (eg[s] < 0 || eg[s] >= EG) {
        bad |= PLAN_BAD_ID;
        eg[s] = -1;
      } else {
        for (int q = x - x % k; q < x; q++) {
          bad |= se[q - x0] == eg[s] ? PLAN_DUP_ID : 0;
        }
      }
      fs_g(a->wts)[x] = eg[s] < 0 ? 0.0f : wt[s];
    }
  }
#pragma unroll
  for (int s = 0; s < PLAN_SUB; s++) {
    const int x = x0 + s * NUM_THREADS + tid, e = eg[s];
    for (int i = tid; i < NUM_WARPS * EG; i += NUM_THREADS) {
      wc[i] = 0;
    }
    __syncthreads();
    int r = plan_wave_rank(e);
    if (e >= 0) {
      atomicAdd(&wc[wv * EG + e], 1);
    }
    __syncthreads();
    pk[s] = -1;
    if (e >= 0) {
      r += run[e];
      for (int q = 0; q < wv; q++) {
        r += wc[q * EG + e];
      }
      pk[s] = r << 16 | e;
    }
    if (x < TK) {
      fs_g(a->p.pk)[x] = pk[s];
    }
    __syncthreads();
    if (tid < EG) {
      int h = 0;
      for (int q = 0; q < NUM_WARPS; q++) {
        h += wc[q * EG + tid];
      }
      run[tid] += h;
    }
    __syncthreads();
  }
  if (tid < EG) {
    fs_g(a->p.hist)[(size_t)c * EG + tid] = run[tid];
  }
  return bad;
}

// Expert tid's count summed over chunks [0, c1) (tot) and [0, c0) (the return value).
__device__ __forceinline__ int plan_hist_sum(FsArgsC *a, int c0, int c1, int *tot) {
  const int tid = (int)threadIdx.x, EG = MOE_EP * a->E_loc;
  int pre = 0, sum = 0;
  if (tid < EG) {
    const gint *h = fs_g(a->p.hist) + tid;
    int c = 0;
    for (; c + 8 <= c1; c += 8) {
      int v[8];
#pragma unroll
      for (int z = 0; z < 8; z++) {
        v[z] = h[(size_t)(c + z) * EG];
      }
#pragma unroll
      for (int z = 0; z < 8; z++) {
        pre += c + z < c0 ? v[z] : 0;
        sum += v[z];
      }
    }
    for (; c < c1; c++) {
      const int v = h[(size_t)c * EG];
      pre += c < c0 ? v : 0;
      sum += v;
    }
  }
  *tot = sum;
  return pre;
}

// This rank's per-expert totals into every rank's count slot (and tpe, if set), then a release add on each rank's
// count flag.
__device__ __forceinline__ void plan_publish(FsArgsC *a, int tot, unsigned epoch, int *tpe) {
  const int tid = (int)threadIdx.x, EG = MOE_EP * a->E_loc, me = a->my_rank;
  if (tid < EG) {
    const size_t off = ((size_t)(epoch & 1u) * MOE_EP + me) * EG + tid;
    for (int d = 0; d < MOE_EP; d++) {
      __builtin_nontemporal_store(tot, fs_g(a->p.cnts[d]) + off);
    }
    if (tpe) {
      fs_g(tpe)[tid] = tot;
    }
  }
  // The slots this rank released, with the counts: the host's entry for this call (its epoch tags it; a stale one
  // releases nothing).
  if (a->p.rel_in && tid < MOE_RM_WORDS) {
    const auto *in = (const __attribute__((address_space(1))) unsigned *)a->p.rel_in +
                     (size_t)(epoch % MOE_RM_RING) * (1 + MOE_RM_WORDS);
    const unsigned tag = __hip_atomic_load(in, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const unsigned v = tag == epoch ? __hip_atomic_load(in + 1 + tid, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) : 0u;
    const size_t off = ((size_t)(epoch & 1u) * MOE_EP + me) * MOE_RM_WORDS + tid;
    for (int d = 0; d < MOE_EP; d++) {
      __builtin_nontemporal_store((int)v, fs_g(a->p.rmask[d]) + off);
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid < MOE_EP) {
    __hip_atomic_fetch_add((gint *)a->p.cflag[tid] + me, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// The forward's last workgroup, threads w < MOE_RM_WORDS (its moe_plan_layout acquired every rank's counts): the slots
// every rank released (the AND of the masks published with the counts) lose their pool records before its pool update,
// and the agreed mask goes to the host's ring (read once the call completed). The calling context's slot is never
// released.
__device__ __forceinline__ void plan_release_agree(FsArgsC *a, unsigned epoch, int w) {
  const gint *m = fs_g(a->p.rmask[a->my_rank]) + (size_t)(epoch & 1u) * MOE_EP * MOE_RM_WORDS + w;
  unsigned v = ~0u;
  for (int r = 0; r < MOE_EP; r++) {
    v &= (unsigned)m[(size_t)r * MOE_RM_WORDS];
  }
  if (a->pool) {
    gint *pool = fs_g(a->pool);
    const int n = pool[1];
    for (unsigned b = v; b; b &= b - 1) {
      const int j = w * 32 + __builtin_ctz(b);
      if (j < n && j != a->pslot) {
        pool[2 + 3 * j] = 0, pool[3 + 3 * j] = 0, pool[4 + 3 * j] = 0;
      }
    }
  }
  auto *out = (__attribute__((address_space(1))) unsigned *)a->p.rel_out +
              (size_t)(epoch % MOE_RM_RING) * (1 + MOE_RM_WORDS);
  __hip_atomic_store(out + 1 + w, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (w == 0) {
    __hip_atomic_store(out, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Chunk c's entries into send_slot (and send_w) at unit offset (L->sof) + chunk prefix (L->col) + rank in chunk.
__device__ __forceinline__ void plan_scatter(FsArgsC *a, const MoeLayout *L, int c, const int (&pk)[PLAN_SUB],
                                             const float (&wt)[PLAN_SUB]) {
  const int tid = (int)threadIdx.x, E = a->E_loc;
#pragma unroll
  for (int s = 0; s < PLAN_SUB; s++) {
    if (pk[s] >= 0) {
      const int eg = pk[s] & 0xFFFF, i = L->sof[eg % E * MOE_EP + eg / E] + L->col[eg] + (pk[s] >> 16);
      fs_g(a->send_slot)[i] = c * PLAN_CH + s * NUM_THREADS + tid;
      if (a->p.send_w) {
        fs_g(a->p.send_w)[i] = wt[s];
      }
    }
  }
}

// Stable counting sort of the routing into send_slot: workgroups wi < nch count chunks wi, wi + nw, ... (PC_ARR),
// then place them by the whole count matrix (PC_SCAT); workgroup nch (0 in a narrow launch) publishes the totals.
__device__ __forceinline__ void moe_plan_local(FsArgsC *a, const int *idx, const float *w, int wi, int nw,
                                               unsigned epoch, char *scr, int *tpe) {
  MoeLayout *L = (MoeLayout *)(scr + PLAN_LAYOUT_OFF);
  const int tid = (int)threadIdx.x, nch = a->p.nch, E = a->E_loc, EG = MOE_EP * E;
  const bool pub = wi == (nw > nch ? nch : 0);
  gint *ctr = fs_g(a->p.ctr);
  const bool keep = wi + nw < nch;
  int bad = 0, mine = 0, pk[PLAN_SUB];
  float wt[PLAN_SUB];
  for (int c = wi; c < nch; c += nw) {
    bad |= plan_chunk(a, idx, w, c, scr, pk, wt);
    mine++;
  }
  if (!mine && !pub) {
    return;
  }
  if (bad) {
    __hip_atomic_fetch_or(ctr + PC_BAD, bad, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (mine) {
      __hip_atomic_fetch_add(ctr + PC_ARR, mine, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    plan_wait(ctr + PC_ARR, nch, false);
  }
  __syncthreads();
  int tot;
  int pre = plan_hist_sum(a, wi, nch, &tot);
  if (pub) {
    plan_publish(a, tot, epoch, tpe);
  }
  if (!mine) {
    return;
  }
  if (tid < EG) {
    L->col[tid] = tot;
  }
  __syncthreads();
  int total;
  const int so = plan_scan(tid < EG ? L->col[(tid % MOE_EP) * E + tid / MOE_EP] : 0, L->tmp, &total);
  if (tid < EG) {
    L->sof[tid] = so;
  }
  for (int c = wi; c < nch; c += nw) {
    if (keep) {
      int t;
      pre = plan_hist_sum(a, c, c, &t);
#pragma unroll
      for (int s = 0; s < PLAN_SUB; s++) {
        const int x = c * PLAN_CH + s * NUM_THREADS + tid;
        pk[s] = x < a->T * a->topk ? fs_g(a->p.pk)[x] : -1;
        wt[s] = pk[s] >= 0 ? fs_g(w)[x] : 0.0f;
      }
    }
    __syncthreads();
    if (tid < EG) {
      L->col[tid] = pre;
    }
    __syncthreads();
    plan_scatter(a, L, c, pk, wt);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    __hip_atomic_fetch_add(ctr + PC_SCAT, mine, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// The layout from the count matrix cs. Every rank derives the same Mpad_max, so an over-capacity routing gives every
// rank the empty layout; so does fz (the pool forces the call into rounds).
// MXFP8 training with I < 1024 (the forward writes the column-wise Xp) pads each (source, expert) segment to 32 rows,
// so no column-wise block spans two sources.
// O: the opaque thread id (in a round loop).
template <bool MXK, bool O = !MXK, typename CS>
__device__ __forceinline__ void plan_derive(FsArgsC *a, MoeLayout *L, CS cs, int fz = 0) {
  const int tid = plan_tid<O>(), E = a->E_loc, EG = MOE_EP * E, me = a->my_rank;
  const int e = tid / MOE_EP, x = tid % MOE_EP, g = x * E + e;
  int v[MOE_EP], snd = 0, rcv = 0;
  if (tid < EG) {
#pragma unroll
    for (int s = 0; s < MOE_EP; s++) {
      v[s] = cs[s * EG + tid];
    }
    snd = cs[me * EG + g];
    rcv = cs[x * EG + me * E + e];
  }
  int col = 0, part = 0;
  if (tid < EG) {
#pragma unroll
    for (int s = 0; s < MOE_EP; s++) {
      col += v[s];
      part += s < me ? v[s] : 0;
    }
  }
  const bool p32 = MXK && a->mxt && a->xc_fwd;
  int colp = col;
  if constexpr (MXK) {
    if (p32 && tid < EG) {
      int pp = 0;
      colp = 0;
#pragma unroll
      for (int s = 0; s < MOE_EP; s++) {
        colp += plan_pad32(v[s]);
        pp += s < me ? plan_pad32(v[s]) : 0;
      }
      L->colp[tid] = colp, L->ppart[tid] = pp;
    }
  }
  int tp, tc;
  const int2 pc = plan_scan2<O, MXK && O>(plan_pad(colp), col, L->tmp, &tp, &tc);
  if (tid < EG) {
    L->col[tid] = col, L->part[tid] = part, L->pof[tid] = pc.x, L->cpof[tid] = pc.y;
  }
  __syncthreads();
  int mx = 0, mine = 0;
  for (int d = 0; d < MOE_EP; d++) {
    const int n = (d + 1 < MOE_EP ? L->pof[(d + 1) * E] : tp) - L->pof[d * E];
    mx = max(mx, n);
    mine = d == me ? n : mine;
  }
  const bool ovf = mx > a->p.cap || fz;
  snd = ovf ? 0 : snd;
  rcv = ovf ? 0 : rcv;
  int td, tr;
  const int2 su = plan_scan2<O, MXK && O>(snd, rcv, L->tmp, &td, &tr);
  const int uof = su.y;
  if constexpr (MXK) {
    if (p32) {
      int tu;
      const int up = plan_scan<O, MXK && O>(plan_pad32(rcv), L->tmp, &tu);
      if (tid < EG) {
        L->uofp[tid] = up;
      }
    }
  }
  if (tid < EG) {
    L->snd[tid] = snd, L->sof[tid] = su.x, L->rcv[tid] = rcv, L->uof[tid] = uof;
    L->dbase[tid] = ovf ? 0 : L->pof[g] - L->pof[x * E] + (MXK && p32 ? L->ppart[g] : L->part[g]);
    L->sbase[tid] = ovf ? 0 : L->cpof[g] - L->cpof[x * E] + L->part[g];
    if (x == 0) {
      L->est[e] = uof;
    }
  }
  const int cn = tid < E && !ovf ? (MXK && p32 ? L->colp[me * E + tid] : L->col[me * E + tid]) : 0;
  if (tid < E) {
    L->ro[tid] = ovf ? 0 : L->pof[me * E + tid] - L->pof[me * E];
    L->cn[tid] = cn;
  }
  if (tid == 0) {
    L->est[E] = tr;
    L->mpad_max = mx, L->ovf = ovf, L->mpad = ovf ? 0 : mine;
    L->drows = td, L->crows = tr;
    if constexpr (!MXK) {
      L->cur = L->gr = L->ar = 0;
      if (a->pool) {
        const int cur = fs_g(a->pool)[0], R = (ovf ? 0 : mx) + BLOCK_ROW;
        L->cur = cur, L->gr = cur + R, L->ar = pool_a_row(cur, R, a->pgq);
      }
    } else {
      // MXFP8 training: the block's base and rows (gr).
      L->cur = a->pool ? fs_g(a->pool)[0] : 0, L->gr = (ovf ? 0 : mx) + BLOCK_ROW;
    }
  }
  __syncthreads();
}

// This call's count matrix again, its slot from L (as the derivation's cs, kept live it would spill).
__device__ __forceinline__ const gint *plan_counts(FsArgsC *a, const MoeLayout *L) {
  return fs_g(a->p.cnts[a->my_rank]) + (size_t)L->cpar * MOE_EP * (MOE_EP * a->E_loc);
}

// Over capacity: receiver d's rows cut into windows of <= cap rows at expert boundaries (inside a larger expert at
// 256-row multiples); thread d finds window r and d's count. Passes: the most rounds plus the empty overflow pass.
// MXK with xc_fwd (MXFP8 training, I < 1024; the host sets xc_fwd for no other call): an expert's rows with padded
// segments (colp), so every cut is on a 32-row group boundary.
template <bool MXK = false>
__device__ __forceinline__ void plan_windows(FsArgsC *a, MoeLayout *L, int r) {
  const int tid = plan_tid<true>(), E = a->E_loc, EG = MOE_EP * E, me = a->my_rank;
  const gint *cs = plan_counts(a, L);
  const int cap = a->p.cap / BLOCK_ROW * BLOCK_ROW;
  const bool p32 = MXK && a->xc_fwd;
  if (tid < MOE_EP) {
    const int d = tid, b0 = L->pof[d * E];
    int q = 0, A = 0, wa = -1, wb = -1, tot = 0;
    for (int e = 0; e < E; e++) {
      const int base = L->pof[d * E + e] - b0, end = base + plan_pad(p32 ? L->colp[d * E + e] : L->col[d * E + e]);
      while (end - A > cap) {
        const int nA = base > A ? base : A + cap;
        if (q == r) {
          wa = A, wb = nA;
        }
        q++, A = nA;
      }
      tot = end;
    }
    if (q == r) {
      wa = A, wb = tot;
    }
    L->wa[d] = wa < 0 ? tot : wa, L->wb[d] = wb < 0 ? tot : wb, L->wn[d] = q + 1;
    int o = 0;
    for (int e = 0; e < E; e++) {
      o += cs[me * EG + d * E + e];
    }
    L->wo[d] = o;
  }
  __syncthreads();
  if (tid == 0) {
    int n = 0;
    for (int d = 0; d < MOE_EP; d++) {
      n = max(n, L->wn[d]);
    }
    L->nr = n + 1, L->mpad_max = cap;
  }
}

// Round r's layout from the counts clipped to the windows (a row's index less its window start), sends from the full
// order. wm: each local expert's wgrads in round r, 0 none, 1 stored (first rows, or no rows in round 0), 2 added.
// MXK: segments padded as plan_windows'; the sends' offsets and starts in the scratch below the layout, as
// plan_derive<true> writes colp and ppart.
template <bool MXK = false>
__device__ __forceinline__ void plan_clip(FsArgsC *a, MoeLayout *L, int r) {
  typedef __attribute__((address_space(3))) int lint;
  const int tid = plan_tid<true>(), E = a->E_loc, EG = MOE_EP * E, me = a->my_rank;
  const gint *cs = plan_counts(a, L);
  const bool p32 = MXK && a->xc_fwd;
  if (tid < E) {
    const int g = me * E + tid, base = L->pof[g] - L->pof[me * E], A = L->wa[me], B = L->wb[me];
    const int c = p32 ? L->colp[g] : L->col[g];
    L->wm[tid] = c ? (base + c > A && base < B ? (base < A ? 2 : 1) : 0) : r == 0;
  }
  lint *cm = (lint *)((char *)L - PLAN_LAYOUT_OFF);
  int *sfv = MXK ? (int *)((char *)L - PLAN_LAYOUT_OFF + PLAN_IDS_OFF) : L->colp;
  int *ppv = MXK ? (int *)((char *)L - PLAN_LAYOUT_OFF + PLAN_IDS_OFF) + PLAN_MAXU : L->ppart;
  int t;
  const int sf = plan_scan<true, MXK>(tid < EG ? cs[me * EG + (tid % MOE_EP) * E + tid / MOE_EP] : 0, L->tmp, &t);
  if (tid < EG) {
    const int d = tid / E, base = L->pof[tid] - L->pof[d * E], A = L->wa[d], B = L->wb[d];
    int lo = base;
#pragma unroll
    for (int s = 0; s < MOE_EP; s++) {
      const int v = cs[s * EG + tid];
      cm[s * EG + tid] = max(0, min(lo + v, B) - max(lo, A));
      if (s == me) {
        ppv[tid] = min(max(A - lo, 0), v);
      }
      lo += p32 ? plan_pad32(v) : v;
    }
    sfv[tid] = sf;
  }
  __syncthreads();
  const int nr = L->nr, cap = L->mpad_max;
  plan_derive<MXK, true>(a, L, (const lint *)cm);
  if (tid < EG) {
    L->sof[tid] = sfv[tid] + ppv[(tid % MOE_EP) * E + tid / MOE_EP];
  }
  if (tid == 0) {
    L->nr = nr, L->mpad_max = cap;
    if constexpr (!MXK) {
      if (a->pool) {
        L->gr = L->cur + cap + BLOCK_ROW, L->ar = pool_a_row(L->cur, cap + BLOCK_ROW, a->pgq);
      }
    }
  }
  __syncthreads();
}

// Every workgroup: waits until all 8 ranks have published this call's counts (every bucket group), then derives the
// layout into L; RK (the fused forwards, the backwards' rounds) over capacity: r < 0 counts the rounds (the layout
// stays empty), r >= 0 is round r's.
template <bool MXK, bool PK, bool RK, bool O>
__device__ __forceinline__ void moe_plan_layout(FsArgsC *a, MoeLayout *L, unsigned epoch, int r, int fz) {
  const int tid = plan_tid<O>(), E = a->E_loc, EG = MOE_EP * E, me = a->my_rank;
  if (tid < MOE_EP) {
    plan_wait(fs_g(a->p.cflag[me]) + tid, (int)(epoch + 1u), true);
  }
  __syncthreads();
  const gint *cs = fs_g(a->p.cnts[me]) + (size_t)(epoch & 1u) * MOE_EP * EG;
  if constexpr (RK) {
    if (tid == 0) {
      L->cpar = epoch & 1u;
    }
  }
  // fz: a storing forward that cannot keep a full-capacity Xp (MXFP8: block) in the pool runs in rounds, keeping
  // nothing (its start reads that from the pool; its backward's rounds from the record).
  plan_derive<MXK, O>(a, L, cs, fz);
  if constexpr (PK) {
    // The call's full-routing peak, before any round window clips it: one running max per call for the host.
    if (blockIdx.x == 0 && tid == 0) {
      auto *rec = (__attribute__((address_space(1))) MoePlanRec *)fs_g(a->p.rec);
      (void)__hip_atomic_fetch_max(&rec->peak, L->mpad_max, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  if constexpr (RK) {
    if (!L->ovf) {
      if (tid == 0) {
        L->nr = 1;
      }
    } else {
      plan_windows<MXK>(fs_args((const MoeFwdArgs *)a), L, r < 0 ? 0 : r);
      if (r >= 0) {
        plan_clip<MXK>(fs_args((const MoeFwdArgs *)a), L, r);
      }
    }
    __syncthreads();
  }
}

// The tile queues' offsets into L: GEMM1 tiles in expert order, GEMM2 tiles by XCD bucket (experts b, b + 8, ...).
template <bool O, bool OS>
__device__ __forceinline__ void moe_plan_queues(FsArgsC *a, MoeLayout *L) {
  const int tid = plan_tid<O>(), E = a->E_loc;
  const int nb = tid < E ? plan_pad(L->cn[tid]) / BLOCK_ROW : 0;
  int t1;
  const int o1 = plan_scan<O, OS>(nb * (2 * a->I / BLOCK_COL), L->tmp, &t1);
  if (tid < E) {
    int p = tid / NUM_XCDS_AFF;
    for (int q = 0; q < tid % NUM_XCDS_AFF; q++) {
      p += (E - q + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF;
    }
    L->e2[p] = tid;
    L->o2[p] = nb * (a->H / BLOCK_COL);
    L->o1[tid] = o1;
  }
  if (tid == 0) {
    L->o1[E] = t1;
  }
  __syncthreads();
  int t2;
  const int o2 = plan_scan<O, OS>(tid < E ? L->o2[tid] : 0, L->tmp, &t2);
  if (tid < E) {
    L->o2[tid] = o2;
  }
  if (tid == 0) {
    L->o2[E] = t2;
  }
  __syncthreads();
}

static_assert(sizeof(SendBlk) == 5 * sizeof(int) && sizeof(hk_comm::CCombBlk) == 5 * sizeof(int),
              "the backward's unit tables are written as 5 ints");

__device__ __forceinline__ void plan_put5(int *tab, int j, int v0, int v1, int v2, int v3, int v4) {
  gint *q = fs_g(tab) + 5 * j;
  q[0] = v0, q[1] = v1, q[2] = v2, q[3] = v3, q[4] = v4;
}

// This call's scalars for the host (MoePlanRec) and the backward's row counts; the backward's crews and wgrad run
// lengths are the context's, set once from the shape (moe_impl.cuh, bwd_shape_plan).
template <bool MXK, bool RK>
__device__ __forceinline__ void plan_scalars(FsArgsC *a, const MoeLayout *L) {
  const int E = a->E_loc, me = a->my_rank, H = a->H, I = a->I, Mpad = L->mpad, nrb = Mpad / BLOCK_ROW;
  int st = __hip_atomic_load(fs_g(a->p.ctr) + PC_BAD, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  st |= L->ovf && MXK && !RK ? PLAN_OVERFLOW : 0;
  st |= RK && L->nr > 1 ? PLAN_ROUNDS : 0;
  int dself = 0, cself = 0;
  for (int e = 0; e < E; e++) {
    dself += L->snd[e * MOE_EP + me];
    cself += L->rcv[e * MOE_EP + me];
  }
  const int nt1 = nrb * (2 * I / BLOCK_COL), nt2 = nrb * (H / BLOCK_COL);
  auto *rec = (__attribute__((address_space(1))) MoePlanRec *)fs_g(a->p.rec);
  rec->status = st, rec->Mpad = Mpad, rec->Mpad_max = L->mpad_max, rec->nt1 = nt1, rec->nt2 = nt2;
  rec->offrows = L->drows - dself, rec->dself = dself, rec->crows = L->crows, rec->cself = cself;
  if constexpr (RK) {
    rec->rounds = L->nr;
  }
  if (a->p.ba) {
    auto *ba = (__attribute__((address_space(1))) SingleArgs *)fs_g((SingleArgs *)a->p.ba);
    ba->a.w.act_rows = Mpad + 1, ba->b.w.act_rows = Mpad + 1, ba->nrow = Mpad + 1;
    if constexpr (RK) {
      // Over capacity: the backward's rows are each round's (its own tables, rinfo); the operands' bounds are cap's.
      ba->nr = L->nr;
      if (L->nr > 1) {
        ba->a.w.act_rows = ba->b.w.act_rows = ba->nrow = L->mpad_max + 1;
      }
    }
  }
  if (st) {
    __hip_atomic_fetch_or(fs_g(a->p.status), st, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// This call's Xp rows in the pool, R: every rank's Mpad_max and the zero row's block (an over-capacity routing, whose
// outputs are undefined: that block only), from the plan's record.
__device__ __forceinline__ int plan_pool_rows(const int *p) {
  const auto *rec = (const __attribute__((address_space(1))) MoePlanRec *)fs_g(p);
  return (rec->status & PLAN_OVERFLOW ? 0 : rec->Mpad_max) + BLOCK_ROW;
}

// Whether this call was forced into rounds, for its backward's rounds (single_round_tables), which re-derive its layout.
__device__ __forceinline__ void plan_set_forced(FsArgsC *a, int fz) {
  ((__attribute__((address_space(1))) MoePlanRec *)fs_g(a->p.rec))->forced = fz;
}

// The backward of a forward that keeps nothing pushes its X again.
__device__ __forceinline__ void plan_set_x0(FsArgsC *a, const bf16 *x) {
  ((__attribute__((address_space(1))) SingleArgs *)fs_g(a->p.ba))->x0 = x;
}

// A stored forward's last workgroup keeps its block of the pool from its base: Xp, then G and A if they fit
// (pool_update; nothing after rounds, which reuse the block); for what it does not keep, the backward's operands are
// its scratch. Not inlined: inlined it raises the router kernels' SGPR peak.
__device__ __noinline__ void pool_commit(const MoeFwdArgs *args, int *scr, int R, bool rounds) {
  const auto *a = (const __attribute__((address_space(1))) MoeFwdArgs *)fs_uniform_ptr(args);
  gint *pool = fs_g(a->pool);
  const int cur = pool[0];
  const int keep = pool_update(pool, a->pslot, a->prows, a->phead, a->pres, a->pres, cur, cur + R,
                               pool_end(cur, R, a->pgq, a->paq), scr, rounds);
  if (fs_tid() == 0) {
    // The backward's operands, written here only (one writer): the kept block's Xp, G and A; for what is not kept,
    // A recomputed into ascr (G into its dG) and, with nothing kept, Xp from X pushed again into pxre.
    const size_t H = a->H;
    auto *ba = (__attribute__((address_space(1))) SingleArgs *)fs_g(a->p.ba);
    ba->a.w.act = keep == 2 ? a->act + (size_t)pool_a_row(cur, R, a->pgq) * H : a->pascr;
    ba->b.w.act = keep ? a->p.zf + (size_t)cur * H : a->pxre;
    ba->g = a->p.zf + (size_t)(cur + R) * H;
    if (keep < 2) {
      __hip_atomic_fetch_or(fs_g(a->p.status), keep ? PLAN_KEPT_XP : rounds ? PLAN_KEPT_NONE : PLAN_ONEPASS_NONE,
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// MXFP8 training's pool_commit: a stored forward's last workgroup keeps its whole block (G, the column-wise A and with
// I < 1024 the column-wise Xp) or, after rounds or with no room, nothing; it writes the backward's operands (one
// writer). Nothing kept: its backward runs in rounds, recomputing them at the pool's head.
__device__ __noinline__ void pool_commit_mx(const MoeFwdArgs *args, int *scr, int R, bool rounds) {
  const auto *a = (const __attribute__((address_space(1))) MoeFwdArgs *)fs_uniform_ptr(args);
  gint *pool = fs_g(a->pool);
  const int cur = pool[0], H = a->H, I = a->I, top = cur + pool_mx_rows(R, H, I, a->xc_fwd);
  const int keep = pool_update(pool, a->pslot, a->prows, a->phead, 0, a->phead, cur, top, top, scr, rounds);
  if (fs_tid() == 0) {
    uint8_t *b = a->pbase + (size_t)cur * 2 * H;
    auto *ba = (__attribute__((address_space(1))) SingleArgs *)fs_g(a->p.ba);
    ba->g = (const bf16 *)b, ba->actc = b + pool_mx_ac(R, I), ba->actcs = b + pool_mx_acs(R, I);
    if (a->xc_fwd) {
      ba->xpc = b + pool_mx_xc(R, I), ba->xpcs = b + pool_mx_xcs(R, H, I);
    }
    if (!keep) {
      __hip_atomic_fetch_or(fs_g(a->p.status), rounds ? PLAN_KEPT_NONE : PLAN_ONEPASS_NONE, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// The forward's receive-side tables, sliced over workgroups wi of nw; workgroup 0 also zeroes the receive arena's zero
// row. With padded segments the arrival tables count each source's padding rows, zeroed here.
template <bool MXK, bool RK, bool O>
__device__ __forceinline__ void moe_plan_tables(FsArgsC *a, const MoeLayout *L, int wi, int nw) {
  const int tid = plan_tid<O>(), E = a->E_loc, EG = MOE_EP * E;
  const int gi = wi * NUM_THREADS + tid, gs = nw * NUM_THREADS;
  const int nrb = L->mpad / BLOCK_ROW;
  const bool p32 = MXK && a->mxt && a->xc_fwd;
  for (int u = gi; u < EG; u += gs) {
    const int e = u / MOE_EP, x = u % MOE_EP, cs = L->snd[u], cr = L->rcv[u];
    const int roff = L->ro[e], st = p32 ? L->uofp[u] - L->uofp[e * MOE_EP] : L->uof[u] - L->est[e], lo = roff + st;
    const int rbp = max(0, min(roff / BLOCK_ROW, nrb - 1));
    gint *rb = fs_g((int *)a->rbs) + 4 * u;
    rb[0] = cr ? lo / BLOCK_ROW : rbp, rb[1] = x, rb[2] = L->uof[u], rb[3] = cr;
    fs_g(a->rb_hi)[u] = cr ? (lo + cr - 1) / BLOCK_ROW : rbp;
    fs_g(a->start_by_src)[sig_index(x, e)] = st;
    fs_g(a->cnt_by_src)[sig_index(x, e)] = p32 ? plan_pad32(cr) : cr;
  }
  if (gi < E) {
    fs_g(a->cnt)[gi] = L->cn[gi];
    fs_g(a->row_off)[gi] = L->ro[gi];
  }
  if (gi < MOE_EP) {
    int o = 0;
    for (int e = 0; e < E; e++) {
      o += L->snd[e * MOE_EP + gi];
    }
    if constexpr (RK) {
      o = L->nr > 1 ? L->wo[gi] : o;
    }
    fs_g(a->owed)[gi] = o;
  }
  for (int ci = gi; ci < L->crows; ci += gs) {
    const int e = fs_seg(L->est, E, ci);
    if (p32) {
      int u = e * MOE_EP;
      while (u + 1 < (e + 1) * MOE_EP && L->uof[u + 1] <= ci) {
        u++;
      }
      fs_g(a->lb_src)[ci] = L->ro[e] + L->uofp[u] - L->uofp[e * MOE_EP] + ci - L->uof[u];
    } else {
      fs_g(a->lb_src)[ci] = L->ro[e] + ci - L->est[e];
    }
  }
  if (wi == 0) {
    plan_zero_rows<O>(a->p.zf, (MXK ? 0 : L->cur) + L->mpad, (MXK ? 0 : L->cur) + L->mpad + 1, a->H);
  }
  if constexpr (MXK) {
    for (int u = wi; p32 && u < EG; u += nw) {
      const int e = u / MOE_EP, cr = L->rcv[u], lo = L->ro[e] + L->uofp[u] - L->uofp[e * MOE_EP] + cr;
      plan_zero_rows<O>(a->p.zf, lo, lo + plan_pad32(cr) - cr, a->H);
    }
  }
}

// After the forward's tables are released: the backward's tables and this call's scalars, then the padding rows the
// wgrads read directly (an expert's rows up to its count rounded to K_STEP_64, not gathered) are zeroed; the backward
// zeroes those of a column-wise Xp it writes (mx_xcol_zero).
template <bool MXK, bool RK, bool O>
__device__ __forceinline__ void moe_plan_late(FsArgsC *a, const MoeLayout *L, int wi, int nw, bool sc) {
  const int tid = plan_tid<O>(), E = a->E_loc, EG = MOE_EP * E;
  const bool p32 = MXK && a->mxt && a->xc_fwd;
  if (a->p.ba) {
    for (int u = wi * NUM_THREADS + tid; u < EG; u += nw * NUM_THREADS) {
      const int e = u / MOE_EP, x = u % MOE_EP, cs = L->snd[u], cr = L->rcv[u];
      const int st = p32 ? L->uofp[u] - L->uofp[e * MOE_EP] : L->uof[u] - L->est[e], lo = L->ro[e] + st;
      plan_put5(a->p.bblks, u, x, e, L->sof[u], cs, cs ? L->dbase[u] : 0);
      plan_put5(a->p.cblks, u, x, cr ? lo / BLOCK_ROW : 0, cr ? (lo + cr - 1) / BLOCK_ROW : -1, L->uof[u], cr);
      if constexpr (RK) {
        fs_g(a->sbase)[u] = L->sbase[u];
      }
      fs_g(a->p.cnt_by_src)[sig_index(x, e)] = p32 ? plan_pad32(cr) : cr;
      fs_g(a->p.start_by_src)[sig_index(x, e)] = cr ? st : 0;
    }
  }
  if (wi == 0 && tid == 0 && sc) {
    plan_scalars<MXK, RK>(a, L);
  }
  if (L->ovf || (RK && L->nr > 1)) {
    return;
  }
  for (int e = wi; e < E; e += nw) {
    const int c = L->cn[e], lo = (MXK ? 0 : L->cur) + L->ro[e] + c;
    plan_zero_rows<O>(a->p.zf, lo, lo + (K_STEP_64 - c % K_STEP_64) % K_STEP_64, a->H);
    if constexpr (MXK) {
      if (a->mxt && a->xc_fwd && c) {
        const int hi = L->ro[e] + K_STEP_128 * max(2, (c + K_STEP_128 - 1) / K_STEP_128), lp = L->ro[e] + plan_pad32(c);
        uint8_t *xb = a->pbase + (size_t)L->cur * 2 * a->H;
        plan_zero_rows<O>((bf16 *)(xb + pool_mx_xc(L->gr, a->I)), lp, hi, a->H / 2);
        plan_zero_rows<O>((bf16 *)(xb + pool_mx_xcs(L->gr, a->H, a->I)), lp / 32, hi / 32, a->H / 2);
      }
    }
  }
}


// The backward of a forward over capacity: one workgroup derives round r's tables with the forward's plan (its layout
// in the tile region lds), and round r's rows and wgrad modes into rinfo. MX: MXFP8 training, which reads no gather or
// tile queue (no index phase).
template <bool MX = false>
__device__ __forceinline__ void single_round_tables(const SingleArgs *args, unsigned char *lds, int r) {
  const auto *ka = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  FsArgsC *fa = fs_args(fs_uniform_ptr(ka->fa));
  MoeLayout *L = reinterpret_cast<MoeLayout *>(lds + PLAN_LAYOUT_OFF);
  const unsigned epoch = (unsigned)__builtin_amdgcn_readfirstlane(fs_g(fa->ep)[0]) - 1u;
  if constexpr (MX) {
    moe_plan_layout<true, false, true, true>(fa, L, epoch, r,
                                             ((const __attribute__((address_space(1))) MoePlanRec *)fs_g(fa->p.rec))->forced);
    moe_plan_tables<true, true, true>(fa, L, 0, 1);
    moe_plan_late<true, true, true>(fa, L, 0, 1, false);
  } else {
    moe_plan_layout<false>(fa, L, epoch, r,
                           ((const __attribute__((address_space(1))) MoePlanRec *)fs_g(fa->p.rec))->forced);
    moe_plan_queues<true>(fa, L);
    moe_plan_tables<false>(fa, L, 0, 1);
    fs_index_phase<false, true>(fa, L, 0, 1, false);
    moe_plan_late<false>(fa, L, 0, 1, false);
  }
  gint *ri = fs_g(ka->rinfo);
  const int t = fs_tid();
  if (t == 0) {
    ri[RINFO_NROW] = L->mpad + 1;
    if constexpr (!MX) {
      // The index phase counted itself into the forward's idx_done, which its next forward waits on: back to 0.
      __hip_atomic_store(fs_g(fa->idx_done), 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  if (t < fa->E_loc) {
    ri[RINFO_WM + t] = L->wm[t];
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
}

// The round's rows and wgrad modes from rinfo into LDS.
__device__ __forceinline__ void single_round_info(const SingleArgs *args, int *nrow, unsigned char *wm) {
  const auto *ka = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  const gint *ri = fs_g(ka->rinfo);
  if (fs_tid() == 0) {
    *nrow = ri[RINFO_NROW];
  }
  for (int x = fs_tid(); x < ka->E; x += NUM_THREADS) {
    wm[x] = (unsigned char)ri[RINFO_WM + x];
  }
  __syncthreads();
}

// Before round 0: the last workgroup to arrive derives its tables, then releases the others.
template <bool MX = false>
__device__ __forceinline__ void single_round_open(const SingleArgs *args, unsigned char *lds, int nwg) {
  const auto *ka = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  gint *ri = fs_g(ka->rinfo);
  if (single_last_wg(ka->rinfo + RINFO_ARR, nwg)) {
    single_round_tables<MX>(args, lds, 0);
    if (fs_tid() == 0) {
      __hip_atomic_store(ri + RINFO_ARR, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(ri + RINFO_GO, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else if (fs_tid() == 0) {
    plan_wait(ri + RINFO_GO, 1, false);
  }
  __syncthreads();
}

// After round r (not the last): the last workgroup resets its counters, takes back its dY and X arrivals, derives
// round r + 1's tables, stores RFREE in every peer (its next round may write here) and releases the others. MX: also
// the dY and X quantization items' counts.
template <bool MX = false>
__device__ __forceinline__ void single_round_end(const SingleArgs *args, int *tc, unsigned char *lds, int r, int nrb,
                                                 int nwg, unsigned bcall) {
  const auto *ka = (const __attribute__((address_space(1))) SingleArgs *)fs_uniform_ptr(args);
  gint *ri = fs_g(ka->rinfo);
  // Round r's weight gradients are added to in a later round, on any XCD: released before arriving, then acquired.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  if (single_last_wg(tc + NUM_XCDS_AFF, nwg)) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int t = fs_tid(), E = ka->E, me = ka->a.g.push.my_rank, sigb = ka->a.g.push.sigb;
    for (int i = t; i < SINGLE_NCTR; i += NUM_THREADS) {
      __hip_atomic_store(fs_g(tc) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    for (int i = t; i <= nrb; i += NUM_THREADS) {
      __hip_atomic_store(fs_g(ka->b.c.tile_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(fs_g(ka->dg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(fs_g(ka->rg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if constexpr (MX) {
      for (int i = t; i < nrb * (ka->H / MX_QCOLS); i += NUM_THREADS) {
        __hip_atomic_store(fs_g(ka->qdy_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(fs_g(ka->qx_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    hk_comm::moe_bwd_dispatch_credit(ka->a.g.push.peers.sig[me], ka->a.g.cnt_by_src, ka->a.g.start_by_src,
                                     ka->row_off, E, sigb);
    hk_comm::moe_bwd_dispatch_credit(ka->xsg[me], ka->a.g.cnt_by_src, ka->a.g.start_by_src, ka->row_off, E, sigb);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    single_round_tables<MX>(args, lds, r + 1);
    if (t < MOE_EP) {
      __hip_atomic_store((gint *)ka->a.g.push.peers.sig[t] + hk_comm::bfree_index(MOE_EP + me, sigb),
                         (int)(bcall << 16 | (unsigned)(r + 2)), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (t == 0) {
      __hip_atomic_store(ri + RINFO_GO, r + 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else if (fs_tid() == 0) {
    plan_wait(ri + RINFO_GO, r + 2, false);
  }
  __syncthreads();
}

}
