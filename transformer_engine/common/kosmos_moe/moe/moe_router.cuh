/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE router fused into the MoE kernels: logits GEMM, score, top-k and aux-loss inputs, and the score backward.
#pragma once

namespace hk_moe {

constexpr int RT_SOFTMAX = 0, RT_SIGMOID = 1, RT_SQRTSOFTPLUS = 2, RT_SOFTMAX_PRE = 3;
// The score function of a translation unit built once per score (moe_router.cu, ...); a compile-time policy.
#ifdef MOE_ROUTER_SCORE
constexpr int MoeScore = MOE_ROUTER_SCORE;
#else
constexpr int MoeScore = RT_SOFTMAX;
#endif
constexpr int RG_ROWS  = 64;
constexpr int RG_COLS  = 128;
constexpr int RG_KS    = 64;
constexpr int RG_PF    = 4;
constexpr int RG_EPL   = 8;
constexpr int RG_NB    = 4;
constexpr int RG_TOPM  = 4;
constexpr int RG_KMAX  = 10;
constexpr int RG_STAGE = (RG_ROWS + RG_COLS) * RG_KS * 2;
constexpr int RG_AUX   = NUM_WARPS * RG_EPL * 64 * 4;
constexpr int RC_DISP = 0, RC_PC = 1, RC_RP = 2, RC_RDG = 3, RC_N = 4;
constexpr float RG_EPS = 1e-20f;

typedef __bf16 rg_bf8 __attribute__((ext_vector_type(8)));
typedef float rg_f16 __attribute__((ext_vector_type(16)));
typedef int rg_v4i __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const rg_v4i rg_gcv4;
typedef __attribute__((address_space(3))) rg_v4i rg_lv4;
typedef __attribute__((address_space(3))) float rg_lf;
typedef const __attribute__((address_space(4))) RouterArgs RArgsC;

static_assert(2 * RG_STAGE <= 65536 && 3 * RG_AUX <= 65536, "the router's LDS fits in the start phase's As");
static_assert(RG_ROWS * RG_KS * 2 == 16 * NUM_THREADS && RG_COLS * RG_KS * 2 == 32 * NUM_THREADS,
              "one A chunk and two B chunks of 16 bytes per thread and stage");
static_assert(RG_ROWS % NUM_WARPS == 0 && BLOCK_ROW % RG_ROWS == 0, "routing row blocks");

__host__ __device__ __forceinline__ unsigned rg_hash(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Force-balanced routing: an N(0, 1) logit per (seed, token, expert), Box-Muller on two hashes (hardware log2 and
// cosine; v_cos_f32 takes revolutions).
__device__ __forceinline__ float rg_randn(unsigned seed, int t, int e) {
  const unsigned a = rg_hash(seed ^ rg_hash((unsigned)t * 0x9E3779B1u + (unsigned)e));
  const unsigned b = rg_hash(a ^ 0x85EBCA6Bu);
  const float u1 = (float)((a >> 8) + 1u) * (1.0f / 16777216.0f);
  const float u2 = (float)(b >> 8) * (1.0f / 16777216.0f);
  return sqrtf(-2.0f * 0.69314718056f * __builtin_amdgcn_logf(u1)) * __builtin_amdgcn_cosf(u2);
}

__device__ __forceinline__ float rg_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The scores of sigmoid and sqrtsoftplus (top k of score + expert bias, weights from the scores).
template <int SC>
constexpr bool RG_UNIT = SC == RT_SIGMOID || SC == RT_SQRTSOFTPLUS;

// Softplus as torch's (threshold 20), and sqrt, on the hardware exp, log and sqrt: the libm forms' constants spill
// SGPRs. log(1 + e) by Goldberg's correction, which keeps its relative accuracy for small e.
__device__ __forceinline__ float rg_softplus(float x) {
  const float e = __expf(x), u = 1.0f + e;
  return x > 20.0f ? x : u == 1.0f ? e : __logf(u) * e * __builtin_amdgcn_rcpf(u - 1.0f);
}

// The per-expert score of a logit (softmax and pre-softmax: the logit; pre-softmax normalizes in rg_scores).
template <int SC>
__device__ __forceinline__ float rg_score(float l) {
  if constexpr (SC == RT_SIGMOID) {
    return rg_sigmoid(l);
  } else if constexpr (SC == RT_SQRTSOFTPLUS) {
    return __builtin_amdgcn_sqrtf(rg_softplus(l));
  } else {
    return l;
  }
}

// d score / d logit at logit l, score s (sigmoid, sqrtsoftplus; 0 where sqrt(softplus) underflows to 0).
template <int SC>
__device__ __forceinline__ float rg_dscore(float l, float s) {
  if constexpr (SC == RT_SIGMOID) {
    return s * (1.0f - s);
  } else {
    return s > 0.0f ? (l > 20.0f ? 1.0f : __builtin_amdgcn_rcpf(1.0f + __expf(-l))) * __builtin_amdgcn_rcpf(2.0f * s)
                    : 0.0f;
  }
}

// Byte offset of 16-byte chunk c of row r in a stage (rows of RG_KS bf16): conflict-free for 16-lane phases.
__device__ __forceinline__ uint32_t rg_swz(int r, int c) {
  return (uint32_t)(r * (RG_KS * 2) + ((c ^ ((r >> 1) & 7)) << 4));
}

__device__ __forceinline__ rg_lv4 *rg_l4(uint32_t a) {
  return (rg_lv4 *)(uintptr_t)a;
}

__device__ __forceinline__ void rg_fetch(rg_v4i (&v)[3], rg_gcv4 *xa, rg_gcv4 *wa0, rg_gcv4 *wa1, bool b0, bool b1,
                                         int ks) {
  const int o = ks * (RG_KS / 8);
  v[0] = xa[o];
  v[1] = b0 ? wa0[o] : rg_v4i{0, 0, 0, 0};
  v[2] = b1 ? wa1[o] : rg_v4i{0, 0, 0, 0};
}

__device__ __forceinline__ void rg_put(uint32_t st, const rg_v4i (&v)[3], int r, int c) {
  const uint32_t b = st + RG_ROWS * RG_KS * 2;
  *rg_l4(st + rg_swz(r, c)) = v[0];
  *rg_l4(b + rg_swz(r, c)) = v[1];
  *rg_l4(b + rg_swz(r + RG_ROWS, c)) = v[2];
}

__device__ __forceinline__ void rg_mma(rg_f16 &acc, uint32_t st, int ar, int br, int ch) {
  const uint32_t b = st + RG_ROWS * RG_KS * 2;
#pragma unroll
  for (int kk = 0; kk < RG_KS / 16; kk++) {
    const rg_v4i x = *rg_l4(st + rg_swz(ar, 2 * kk + ch));
    const rg_v4i y = *rg_l4(b + rg_swz(br, 2 * kk + ch));
    acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(__builtin_bit_cast(rg_bf8, x), __builtin_bit_cast(rg_bf8, y), acc,
                                                  0, 0, 0);
  }
}

// logits[t0, t0 + 64) x [e0, e0 + 128) = X Wr^T (+ lb) over K = H, fp32: 8 waves of 32 x 32, two LDS stages of RG_KS
// and RG_PF stages in flight in registers. Rows of Wr past E read as zero; columns [E, El) store -inf.
__device__ __forceinline__ void rg_gemm_unit(const bf16 *X, const bf16 *Wr, const RouterCall *rcl, float *logits, int H,
                                             int E, int El, int t0, int e0, uint32_t lds) {
  const int tid = (int)threadIdx.x, wv = tid >> 6, lane = tid & 63, r = tid >> 3, c = tid & 7;
  const bool b0 = e0 + r < E, b1 = e0 + r + RG_ROWS < E;
  rg_gcv4 *xa  = (rg_gcv4 *)(fs_g(X) + (size_t)(t0 + r) * H + c * 8);
  rg_gcv4 *wa0 = (rg_gcv4 *)(fs_g(Wr) + (size_t)(b0 ? e0 + r : 0) * H + c * 8);
  rg_gcv4 *wa1 = (rg_gcv4 *)(fs_g(Wr) + (size_t)(b1 ? e0 + r + RG_ROWS : 0) * H + c * 8);
  const int nks = H / RG_KS;
  rg_v4i v[RG_PF][3];
#pragma unroll
  for (int s = 0; s < RG_PF; s++) {
    rg_fetch(v[s], xa, wa0, wa1, b0, b1, s);
  }
  rg_put(lds, v[0], r, c);
  if (RG_PF < nks) {
    rg_fetch(v[0], xa, wa0, wa1, b0, b1, RG_PF);
  }
  __syncthreads();
  rg_f16 acc = {};
  const int ar = (wv >> 2) * 32 + (lane & 31), br = (wv & 3) * 32 + (lane & 31), ch = lane >> 5;
  for (int ks = 0; ks < nks; ks += RG_PF) {
#pragma unroll
    for (int u = 0; u < RG_PF; u++) {
      const int k = ks + u;
      rg_mma(acc, lds + (k & 1) * RG_STAGE, ar, br, ch);
      if (k + 1 < nks) {
        rg_put(lds + ((k + 1) & 1) * RG_STAGE, v[(u + 1) % RG_PF], r, c);
        if (k + 1 + RG_PF < nks) {
          rg_fetch(v[(u + 1) % RG_PF], xa, wa0, wa1, b0, b1, k + 1 + RG_PF);
        }
      }
      __syncthreads();
    }
  }
  // C: lane holds column lane % 32 and rows 8 (i / 4) + 4 (lane / 32) + i % 4 of its wave's 32 x 32.
  const int col = e0 + (wv & 3) * 32 + (lane & 31);
  if (col < El) {
    const float *lb = fs_uniform_ptr(rcl->lbias);
    const float b = lb && col < E ? fs_g(lb)[col] : 0.0f;
    auto *lg = fs_g(logits) + (size_t)(t0 + (wv >> 2) * 32 + 4 * (lane >> 5)) * El + col;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      lg[(size_t)(8 * (i / 4) + i % 4) * El] = col < E ? acc[i] + b : -INFINITY;
    }
  }
}

// Every lane gets the wave's largest v, ties to the smallest e.
__device__ __forceinline__ void rg_argmax(float &v, int &e) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    const float ov = __shfl_xor(v, o, 64);
    const int oe = __shfl_xor(e, o, 64);
    if (ov > v || (ov == v && oe < e)) {
      v = ov;
      e = oe;
    }
  }
}

__device__ __forceinline__ float rg_wsum(float v) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    v += __shfl_xor(v, o, 64);
  }
  return v;
}

__device__ __forceinline__ float rg_wmax(float v) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    v = fmaxf(v, __shfl_xor(v, o, 64));
  }
  return v;
}

// Removes the top n of x (lane holds experts lane + 64 q) by value, ties to the lower expert: they become -inf.
// Lane j < n gets slot j's expert in sel.
__device__ __forceinline__ void rg_topk(float (&x)[RG_EPL], int n, int lane, int &sel) {
  for (int j = 0; j < n; j++) {
    float v = -INFINITY;
    int e = 1 << 30;
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      if (x[q] > v) {
        v = x[q];
        e = lane + 64 * q;
      }
    }
    rg_argmax(v, e);
    sel = lane == j ? e : sel;
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      x[q] = lane + 64 * q == e ? -INFINITY : x[q];
    }
  }
}

// s at this lane's expert e, from e's owner lane.
__device__ __forceinline__ float rg_at(const float (&s)[RG_EPL], int e) {
  float x = 0.0f;
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    const float y = __shfl(s[q], e & 63, 64);
    x = (e >> 6) == q ? y : x;
  }
  return x;
}

__device__ __forceinline__ void rg_add(float (&d)[RG_EPL], int e, float v, int lane) {
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    d[q] += lane + 64 * q == e ? v : 0.0f;
  }
}

// Token t's logits (lane holds experts lane + 64 q): the GEMM's, or N(0, 1) draws when forced. Rows have El
// entries, -inf past E: every per-expert mask below derives from the data (a lane-index mask is hoisted into SGPRs).
__device__ __forceinline__ void rg_logits(RArgsC *r, int t, unsigned seed, int lane, float (&l)[RG_EPL]) {
  const int nq = r->El >> 6;
  const auto *lg = fs_g(r->logits) + (size_t)t * r->El;
  const bool force = r->force;
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    l[q] = -INFINITY;
    if (q < nq) {
      const float v = lg[lane + 64 * q];
      l[q] = force && v != -INFINITY ? rg_randn(seed, t, lane + 64 * q) : v;
    }
  }
}

// Group-limited selection: keeps x only in the gtopk groups with the largest sum of their top k / gtopk entries
// (ties to the lower group). gs is the wave's LDS row; the whole workgroup calls this in step.
__device__ __forceinline__ void rg_groups(RArgsC *r, int k, int lane, rg_lf *gs, float (&x)[RG_EPL]) {
  const int E = r->E, G = r->groups, gsz = E / G, m = k / r->gtopk, nq = r->El >> 6;
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    if (q < nq) {
      gs[lane + 64 * q] = x[q];
    }
  }
  __syncthreads();
  float gv = -INFINITY;
  if (lane < G) {
    float top[RG_TOPM];
#pragma unroll
    for (int j = 0; j < RG_TOPM; j++) {
      top[j] = -INFINITY;
    }
    for (int z = 0; z < gsz; z++) {
      float y = gs[lane * gsz + z];
#pragma unroll
      for (int j = 0; j < RG_TOPM; j++) {
        if (j < m && y > top[j]) {
          const float o = top[j];
          top[j] = y;
          y = o;
        }
      }
    }
    gv = 0.0f;
#pragma unroll
    for (int j = 0; j < RG_TOPM; j++) {
      gv += j < m ? top[j] : 0.0f;
    }
  }
  __syncthreads();
  unsigned long long keep = 0ull;
  for (int j = 0; j < r->gtopk; j++) {
    float v = gv;
    int g = lane < G ? lane : 1 << 30;
    rg_argmax(v, g);
    keep |= 1ull << g;
    gv = lane == g ? -INFINITY : gv;
  }
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    x[q] = (keep >> (((lane + 64 * q) / gsz) & 63)) & 1ull ? x[q] : -INFINITY;
  }
}

// This call's draw seed: the caller's device word mixed with the rank; workgroup 0 keeps it for the backward.
__device__ __forceinline__ unsigned rg_seed(const RouterCall *rcl, const RouterArgs *ra) {
  const unsigned *sp = fs_uniform_ptr(rcl->seed);
  const unsigned s = sp ? __builtin_amdgcn_readfirstlane(fs_g(sp)[0]) : 0u;
  return rg_hash(s + 0x9E3779B9u * (unsigned)(fs_args(ra)->rank + 1));
}

// Token t's scores s from its logits l (pre-softmax: the softmax over all experts).
template <int SC>
__device__ __forceinline__ void rg_scores(const float (&l)[RG_EPL], float (&s)[RG_EPL]) {
  if constexpr (SC == RT_SOFTMAX_PRE) {
    float mx = -INFINITY, z = 0.0f;
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      mx = fmaxf(mx, l[q]);
    }
    mx = rg_wmax(mx);
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      s[q] = expf(l[q] - mx);
      z += s[q];
    }
    z = rg_wsum(z);
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      s[q] /= z;
    }
  } else {
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      s[q] = rg_score<SC>(l[q]);
    }
  }
}

// Lane j < k's dispatch weight numerator: s at its slot's expert, or (softmax) exp(logit - top logit).
template <int SC>
__device__ __forceinline__ float rg_num(const float (&s)[RG_EPL], int sel, int k, int lane) {
  const float sv = rg_at(s, sel), m0 = __shfl(sv, 0, 64);
  return lane < k ? (SC == RT_SOFTMAX ? expf(sv - m0) : sv) : 0.0f;
}

// The weights' normalizer of the numerators ev: softmax their sum, sigmoid and sqrtsoftplus their sum when k > 1,
// pre-softmax none.
template <int SC>
__device__ __forceinline__ float rg_norm(float ev, int k) {
  if constexpr (SC == RT_SOFTMAX) {
    return rg_wsum(ev);
  } else if constexpr (RG_UNIT<SC>) {
    return k > 1 ? rg_wsum(ev) + RG_EPS : 1.0f;
  } else {
    return 1.0f;
  }
}

// Aux-loss scores of one token: softmax over E (pre-softmax: s), or the unit scores over their sum; zero past E.
template <int SC>
__device__ __forceinline__ float rg_aux_scores(const float (&l)[RG_EPL], const float (&s)[RG_EPL],
                                               float (&p)[RG_EPL]) {
  if constexpr (SC == RT_SOFTMAX_PRE) {
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      p[q] = s[q];
    }
    return 1.0f;
  }
  float mx = -INFINITY, zs = 0.0f;
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    mx = fmaxf(mx, l[q]);
  }
  mx = RG_UNIT<SC> ? 0.0f : rg_wmax(mx);
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    p[q] = RG_UNIT<SC> ? s[q] : expf(l[q] - mx);
    zs += p[q];
  }
  return rg_wsum(zs) + (RG_UNIT<SC> ? RG_EPS : 0.0f);
}

// Routes token t (one wave): idx / w, and the token's aux-loss scores and aux selection into its wave's LDS rows.
template <int SC>
__device__ __forceinline__ void rg_route_token(const RouterArgs *ra, const RouterCall *rcl, int t, rg_lf *gs,
                                               rg_lf *prow, rg_lf *frow) {
  RArgsC *r = fs_args(ra);
  const int lane = fs_tid() & 63, E = r->E, nq = r->El >> 6, k = r->k;
  const unsigned seed = r->force ? rg_seed(rcl, ra) : 0u;
  float l[RG_EPL], s[RG_EPL], x[RG_EPL];
  rg_logits(r, t, seed, lane, l);
  rg_scores<SC>(l, s);
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    const float *bias = fs_uniform_ptr(rcl->bias);
    x[q] = l[q] == -INFINITY ? -INFINITY : RG_UNIT<SC> && bias ? s[q] + fs_g(bias)[min(lane + 64 * q, E - 1)] : s[q];
  }
  if (r->groups) {
    rg_groups(r, k, lane, gs, x);
  }
  int sel = 0;
  rg_topk(x, k, lane, sel);
  const float ev = rg_num<SC>(s, sel, k, lane);
  const float z = rg_norm<SC>(ev, k);
  if (lane < k) {
    fs_g(r->idx)[(size_t)t * k + lane] = sel;
    fs_g(r->w)[(size_t)t * k + lane] = ev / z * r->scale;
  }
  if (!r->nb) {
    return;
  }
  float p[RG_EPL];
  const float zs = rg_aux_scores<SC>(l, s, p);
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    p[q] = p[q] / zs;
    x[q] = l[q] == -INFINITY ? -INFINITY : p[q];
  }
  rg_topk(x, k, lane, sel);
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    if (q < nq) {
      prow[lane + 64 * q] = p[q];
      frow[lane + 64 * q] = x[q] == -INFINITY ? 1.0f : 0.0f;
    }
  }
}

// Routes row block rb, 8 rounds of one token per wave, summing the aux-loss scores and selections per (group, expert)
// in wave order. Per-round values are re-derived from the opaque thread id and args (hoisted, they pin SGPRs).
template <int SC>
__device__ __forceinline__ void rg_route_block(const RouterArgs *ra, const RouterCall *rcl, int rb, uint32_t lds) {
  constexpr int RPT = RG_NB * RG_EPL * 64 / NUM_THREADS;
  float as[RPT], ac[RPT];
#pragma unroll
  for (int u = 0; u < RPT; u++) {
    as[u] = 0.0f;
    ac[u] = 0.0f;
  }
#pragma unroll 1
  for (int i = 0; i < RG_ROWS / NUM_WARPS; i++) {
    const int wv = fs_tid() >> 6;
    rg_lf *pr = (rg_lf *)(uintptr_t)(lds + RG_AUX);
    rg_lf *fr = pr + NUM_WARPS * RG_EPL * 64;
    rg_route_token<SC>(ra, rcl, rb * RG_ROWS + wv * (RG_ROWS / NUM_WARPS) + i,
                       (rg_lf *)(uintptr_t)(lds + wv * RG_EPL * 64 * 4), pr + wv * RG_EPL * 64, fr + wv * RG_EPL * 64);
    RArgsC *r = fs_args(ra);
    const int E = r->E, nb = r->nb, n = nb * E, tid = fs_tid();
    if (nb) {
      __syncthreads();
#pragma unroll
      for (int u = 0; u < RPT; u++) {
        if (u * NUM_THREADS < n) {
          const int x = min(tid + u * NUM_THREADS, n - 1), b = x / E, e = x - b * E;
#pragma unroll 1
          for (int w = 0; w < NUM_WARPS; w++) {
            const float on = (rb * RG_ROWS + w * (RG_ROWS / NUM_WARPS) + i) % nb == b ? 1.0f : 0.0f;
            as[u] += on * pr[w * RG_EPL * 64 + e];
            ac[u] += on * fr[w * RG_EPL * 64 + e];
          }
        }
      }
      __syncthreads();
    }
  }
  RArgsC *r = fs_args(ra);
  const int n = r->nb * r->E, tid = fs_tid();
#pragma unroll
  for (int u = 0; u < RPT; u++) {
    const int x = tid + u * NUM_THREADS;
    if (x < n) {
      fs_g(r->psum)[(size_t)rb * n + x] = as[u];
      fs_g(r->pcnt)[(size_t)rb * n + x] = (int)ac[u];
    }
  }
}

// The last routed row block: the aux-map counts (the backward's input) and the aux loss, both in a fixed order.
__device__ __forceinline__ void rg_aux_reduce(const RouterArgs *ra, float *aux_loss, uint32_t lds) {
  RArgsC *r = fs_args(ra);
  const int tid = (int)threadIdx.x, n = r->nb * r->E, nrb = r->T / RG_ROWS;
  rg_lf *red = (rg_lf *)(uintptr_t)lds;
  float part = 0.0f;
  for (int x = tid; x < n; x += NUM_THREADS) {
    float ss = 0.0f;
    int cs = 0;
    for (int b = 0; b < nrb; b++) {
      ss += fs_g(r->psum)[(size_t)b * n + x];
      cs += fs_g(r->pcnt)[(size_t)b * n + x];
    }
    fs_g(r->cnt)[x] = cs;
    part += ss * (float)cs;
  }
  red[tid] = part;
  __syncthreads();
  if (tid == 0 && aux_loss) {
    float tot = 0.0f;
    for (int i = 0; i < NUM_THREADS; i++) {
      tot += red[i];
    }
    fs_g(aux_loss)[0] = r->aux_c * tot;
  }
  __syncthreads();
}

__device__ __forceinline__ bool rg_arrive(int *ctr, int last) {
  __shared__ int s_last;
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    s_last = __hip_atomic_fetch_add(fs_g(ctr), 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) == last;
    if (s_last) {
      (void)__hip_atomic_load(fs_g(ctr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  __syncthreads();
  return s_last;
}

// Forward start phase: GEMM units of 64 tokens x 128 experts; a row block's last unit routes it, the last routed block
// reduces the aux loss; wait: until all are routed. Args are re-read per unit (kept across the loop, they pin SGPRs).
template <int SC>
__device__ __forceinline__ void moe_router_fwd(const MoeFwdArgs *args, const RouterCall *rcl, const bf16 *X, int wg,
                                               int nwg, bool wait, char *scr) {
  const uint32_t lds = (uint32_t)(uintptr_t)scr;
  const RouterArgs *ra = &args->r;
  const unsigned sd = rg_seed(rcl, ra);
  if (wg == 0 && threadIdx.x == 0) {
    fs_g(fs_args(ra)->seedv)[0] = sd;
  }
  int nrb = fs_args(ra)->T / RG_ROWS;
  for (int u = wg;; u += nwg) {
    FsArgsC *a = fs_args(args);
    RArgsC *r = &a->r;
    const int ncb = r->Ec / RG_COLS;
    if (u >= nrb * ncb) {
      break;
    }
    const int rb = u / ncb;
    rg_gemm_unit(X, fs_uniform_ptr(rcl->wr), rcl, r->logits, a->H, r->E, r->El, rb * RG_ROWS,
                 (u % ncb) * RG_COLS, lds);
    if (rg_arrive(fs_args(ra)->rbdone + rb, ncb - 1)) {
      rg_route_block<SC>(ra, rcl, rb, lds);
      if (rg_arrive(fs_args(ra)->routed, nrb - 1) && fs_args(ra)->nb) {
        rg_aux_reduce(ra, fs_uniform_ptr(rcl->aux_loss), lds);
      }
    }
    nrb = fs_args(ra)->T / RG_ROWS;
  }
  if (wait) {
    if (threadIdx.x == 0) {
      int n = 0;
      gint *c = fs_g(fs_args(ra)->routed);
      while (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < nrb) {
        spin_backoff(n);
      }
      (void)__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
  }
}

// The forward's router counters (per layer), by its last workgroup.
template <bool C = false>
__device__ __forceinline__ void moe_router_fwd_reset(FsArgsC *a) {
  fs_reset_words<C>(fs_g(a->r.rbdone), a->r.T / RG_ROWS);
  fs_reset_words<C>(fs_g(a->r.routed), 1);
}

// The backward's router counters (in the workspace), by its last workgroup.
template <bool C = false>
__device__ __forceinline__ void moe_router_bwd_reset(const RouterArgs *ra) {
  RArgsC *r = fs_args(ra);
  fs_reset_words<C>(fs_g(r->bctr), RC_N);
  fs_reset_words<C>(fs_g(r->rpdone), r->T / BLOCK_ROW);
  fs_reset_words<C>(fs_g(r->rdgdone), r->T / BLOCK_ROW);
}

// Backward item i: dlogits of tokens [64 i, 64 i + 64) (8 per wave) from the dispatch weights' gradient dtk and the
// aux-loss gradient aux_g * cnt, through the score function; bf16 [T][Ed], zero past E. dlb: the logit bias's gradient
// is wanted, each wave's dlogits sums (fp32) go to row 8 i + wave of dlbp. DT: also into dlt transposed, [Ed][T].
template <int SC, bool DT = false>
__device__ __forceinline__ void moe_router_prep(const RouterArgs *ra, int i, int k, float aux_g, unsigned seed,
                                                bool dlb, bf16 *dlt = nullptr) {
  float db[RG_EPL];
#pragma unroll
  for (int q = 0; q < RG_EPL; q++) {
    db[q] = 0.0f;
  }
#pragma unroll 1
  for (int z = 0; z < RG_ROWS / NUM_WARPS; z++) {
    RArgsC *r = fs_args(ra);
    const int tid = fs_tid(), wv = tid >> 6, lane = tid & 63;
    const int t = i * RG_ROWS + wv * (RG_ROWS / NUM_WARPS) + z, E = r->E, nb = r->nb;
    float l[RG_EPL], s[RG_EPL], d[RG_EPL];
    rg_logits(r, t, seed, lane, l);
    rg_scores<SC>(l, s);
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      d[q] = 0.0f;
    }
    const int sel = lane < k ? fs_g(r->idx)[(size_t)t * k + lane] : 0;
    const float dq = lane < k ? fs_g(r->dtk)[(size_t)t * k + lane] : 0.0f;
    const float ev = rg_num<SC>(s, sel, k, lane);
    // dp: the routing loss's gradient at the slot's expert in the logit (softmax) or the score (the others).
    float dp = r->scale * dq;
    if (SC == RT_SOFTMAX || (RG_UNIT<SC> && k > 1)) {
      const float zk = rg_norm<SC>(ev, k);
      const float ad = rg_wsum(dq * (ev / zk));
      dp = SC == RT_SOFTMAX ? r->scale * (ev / zk) * (dq - ad) : r->scale / zk * (dq - ad);
    }
    for (int j = 0; j < k; j++) {
      rg_add(d, __shfl(sel, j, 64), __shfl(dp, j, 64), lane);
    }
    if (nb) {
      float p[RG_EPL], g[RG_EPL], pg = 0.0f;
      const float zs = rg_aux_scores<SC>(l, s, p);
      const auto *cn = fs_g(r->cnt) + (size_t)(t % nb) * E;
#pragma unroll
      for (int q = 0; q < RG_EPL; q++) {
        g[q] = l[q] == -INFINITY ? 0.0f : aux_g * (float)cn[min(lane + 64 * q, E - 1)];
        p[q] = RG_UNIT<SC> ? p[q] : p[q] / zs;
        pg += p[q] * g[q];
      }
      pg = rg_wsum(pg);
#pragma unroll
      for (int q = 0; q < RG_EPL; q++) {
        if constexpr (SC == RT_SOFTMAX_PRE) {
          d[q] += g[q];
        } else {
          d[q] += RG_UNIT<SC> ? g[q] / zs - pg / (zs * zs) : p[q] * (g[q] - pg);
        }
      }
    }
    if constexpr (SC == RT_SOFTMAX_PRE) {
      float pd = 0.0f;
#pragma unroll
      for (int q = 0; q < RG_EPL; q++) {
        pd += s[q] * d[q];
      }
      pd = rg_wsum(pd);
#pragma unroll
      for (int q = 0; q < RG_EPL; q++) {
        d[q] = s[q] * (d[q] - pd);
      }
    } else if constexpr (RG_UNIT<SC>) {
#pragma unroll
      for (int q = 0; q < RG_EPL; q++) {
        d[q] *= rg_dscore<SC>(l[q], s[q]);
      }
    }
    auto *dl = fs_g((short *)r->dl) + (size_t)t * r->Ed;
    const int nd = r->Ed >> 6;
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      d[q] = l[q] == -INFINITY ? 0.0f : d[q];
      db[q] += d[q];
      if (q < nd) {
        dl[lane + 64 * q] = f2bf_bits(d[q]);
        if constexpr (DT) {
          fs_g((short *)dlt)[(size_t)(lane + 64 * q) * r->T + t] = f2bf_bits(d[q]);
        }
      }
    }
  }
  if (dlb) {
    RArgsC *r = fs_args(ra);
    const int tid = fs_tid(), lane = tid & 63, E = r->E;
    auto *row = fs_g(r->dlbp) + (size_t)(i * NUM_WARPS + (tid >> 6)) * E;
#pragma unroll
    for (int q = 0; q < RG_EPL; q++) {
      if (lane + 64 * q < E) {
        row[lane + 64 * q] = db[q];
      }
    }
  }
}

// The logit bias's gradient: the dlbp rows summed in a fixed order into dlb (the last dlogits item's workgroup). acc 0:
// dlb fp32, overwritten; 1: fp32, added into; 2: bf16, added into (main_grad accumulation).
__device__ __forceinline__ void moe_router_dlb(const RouterArgs *ra, float *dlb, int acc) {
  RArgsC *r = fs_args(ra);
  const int E = r->E, nrow = r->T / RG_ROWS * NUM_WARPS;
  for (int e = (int)threadIdx.x; e < E; e += NUM_THREADS) {
    float v = 0.0f;
    for (int j = 0; j < nrow; j++) {
      v += fs_g(r->dlbp)[j * E + e];
    }
    if (acc == 2) {
      __attribute__((address_space(1))) unsigned short *d =
          (__attribute__((address_space(1))) unsigned short *)fs_g((unsigned short *)(void *)dlb) + e;
      *d = (unsigned short)(pk_bf16(__builtin_bit_cast(float, (unsigned)*d << 16) + v, 0.0f) & 0xffffu);
    } else if (acc == 1) {
      fs_g(dlb)[e] += v;
    } else {
      fs_g(dlb)[e] = v;
    }
  }
}

}
