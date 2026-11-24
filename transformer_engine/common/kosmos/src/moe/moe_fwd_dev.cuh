/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MoE forward device code: the one-launch kernel moe_fwd_single.
#pragma once
#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "../utils/gemm_common.cuh"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

namespace hk_moe {

using namespace kosmos;

constexpr int MOE_EP = 8;

constexpr int EXPERT_PADDING = BLOCK_ROW;

constexpr int MOE_WALK_GM = 4;
constexpr int MOE_WALK_U  = 32;

struct TileDesc {
  int tile_m;
  int tile_n;
  int expert;
};

// Gather offset in bytes, 32-bit: the host refuses shapes whose rows * row stride reaches 2^32.
#define MOE_SW_OFF(r, stride, col) ((uint32_t)(r) * (uint32_t)(stride) + (uint32_t)(col))

// True in the last workgroup to finish, which then zeroes the launch's counters for the next call. Wave 0's
// counter writes drain (s_waitcnt 0) before the relaxed arrive, so the reset follows every other access.
__device__ __forceinline__ bool moe_last_wg(int *arrive) {
  __shared__ int s_last;
  __syncthreads();
  if (threadIdx.x == 0) {
    __builtin_amdgcn_s_waitcnt(0);
    s_last = __hip_atomic_fetch_add(arrive, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == (int)gridDim.x - 1;
  }
  __syncthreads();
  return s_last;
}

__device__ __forceinline__ void moe_walk_caps(int *s_cap, int num_tiles, int t = (int)threadIdx.x) {
  if (t < NUM_XCDS_AFF) {
    const int b  = t;
    const int nu = (num_tiles + MOE_WALK_U - 1) / MOE_WALK_U;
    int c = b < nu ? ((nu - b + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF) * MOE_WALK_U : 0;
    if (b < nu && (nu - 1) % NUM_XCDS_AFF == b) {
      c -= nu * MOE_WALK_U - num_tiles;
    }
    s_cap[b] = c;
  }
}

__device__ __forceinline__ int moe_walk_claim(int *ctr, const int *s_cap, int home, int num_tiles) {
  const int ih = atomicAdd(&ctr[home], 1);
  if (ih < s_cap[home]) {
    return (home + NUM_XCDS_AFF * (ih / MOE_WALK_U)) * MOE_WALK_U + ih % MOE_WALK_U;
  }
#pragma unroll 1
  for (int s8 = 1; s8 < NUM_XCDS_AFF; s8++) {
    const int b   = (home + s8) & (NUM_XCDS_AFF - 1);
    const int cap = s_cap[b];
    if (__hip_atomic_load(&ctr[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= cap) {
      continue;
    }
    const int i = atomicAdd(&ctr[b], 1);
    if (i < cap) {
      return (b + NUM_XCDS_AFF * (i / MOE_WALK_U)) * MOE_WALK_U + i % MOE_WALK_U;
    }
  }
  return num_tiles;
}

using namespace kittens;

// A uniform value through an empty asm: derived values are computed at the use, not hoisted out of a loop.
template <typename T>
__device__ __forceinline__ T fs_opaque_s(T v) {
  asm volatile("" : "+s"(v));
  return v;
}

// threadIdx.x through an empty asm, taken at the use: what is derived from it is computed there instead of being
// hoisted out of the persistent tile loop and carried across the K-loops in VGPRs.
__device__ __forceinline__ int fs_tid() {
  int t = (int)threadIdx.x;
  asm volatile("" : "+v"(t));
  __builtin_assume(t >= 0 && t < NUM_THREADS);
  return t;
}

// As moe_last_wg, for nwg workgroups and the opaque thread id (in the bf16 round loop, where gridDim's address and
// the thread masks would be held across the tile loop).
__device__ __forceinline__ bool moe_last_wg_n(int *arrive, int nwg) {
  __shared__ int s_last;
  __syncthreads();
  if (fs_tid() == 0) {
    __builtin_amdgcn_s_waitcnt(0);
    s_last = __hip_atomic_fetch_add(arrive, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == nwg - 1;
  }
  __syncthreads();
  return s_last;
}

// p, a __shared__ object, with its constant LDS address formed here (an asm s_mov 0 plus the address): a v_mov of the
// address would be hoisted out of the persistent tile loop and carried across the K-loops.
template <typename T>
__device__ __forceinline__ T *fs_lds(T *p) {
  uint32_t z;
  asm volatile("s_mov_b32 %0, 0" : "=s"(z));
  return (T *)(__attribute__((address_space(3))) T *)(z + (uint32_t)reinterpret_cast<uintptr_t>(p));
}

// C in a VGPR, materialized here: a v_mov of a constant would be hoisted out of the tile loop and carried likewise.
template <int C>
__device__ __forceinline__ int fs_imm() {
  int v;
  asm volatile("v_mov_b32 %0, %1" : "=v"(v) : "i"(C));
  return v;
}

template <typename ST, bool KX = false>
__device__ __forceinline__ void gather_parts(int lane, int warp, int *rows, int *colb) {
  using T = typename ST::dtype;
  constexpr int bpt = 16;
  constexpr int bpm = bpt * NUM_THREADS;
  constexpr int mpt = (ST::rows * ST::cols * sizeof(T)) / bpm;
  constexpr int bpw = bpt * kittens::WARP_THREADS;
  constexpr int nw  = NUM_THREADS / kittens::WARP_THREADS;
  const int wid = warp % nw;
  static_assert(!KX || ST::underlying_subtile_bytes % bpw == 0, "the subtile parity is the wave's");
#pragma unroll
  for (int i = 0; i < mpt; i++) {
    const int lbo = (lane * bpt) + (wid * bpw) + (i * nw * bpw);
    const int sid = lbo / ST::underlying_subtile_bytes;
    const int sr  = sid / ST::underlying_subtiles_per_row;
    const int sc  = sid % ST::underlying_subtiles_per_row;
    const int slb = lbo % ST::underlying_subtile_bytes;
    const int r   = slb / ST::underlying_subtile_row_bytes;
    const int c   = (slb % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t kx = KX ? ((wid * bpw + i * nw * bpw) / ST::underlying_subtile_bytes & 1) * 64 : 0;
    const uint32_t sw = ST::swizzle({r, c}) ^ kx;
    rows[i] = (sw / ST::underlying_subtile_row_bytes) + sr * ST::underlying_subtile_rows;
    colb[i] =
        (int)(((sw % ST::underlying_subtile_row_bytes) / sizeof(T) + sc * ST::underlying_subtile_cols) * sizeof(T));
  }
}

// bf16 RNE (kittens' store() truncates) into P, a 16-bit pointer in any address space (global or LDS staging).
template <typename P, typename RT>
__device__ __forceinline__ void store_c_tile(P *base, const RT &src, int row_unit, int col_unit, int row_stride,
                                             int lane) {
  static_assert(sizeof(P) == 2, "store_c_tile: a 16-bit destination");
  constexpr int packing = 2;
  P *dst_ptr = base + (size_t)(row_unit * RT::rows) * row_stride + col_unit * RT::cols;
  const int row_offset = RT::base_tile_stride * (lane / RT::base_tile_cols);
  const int col_offset = lane % RT::base_tile_cols;

#pragma unroll
  for (int i = 0; i < RT::height; i++) {
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
      const int col = j * RT::base_tile_cols + col_offset;
#pragma unroll
      for (int k = 0; k < RT::base_tile_num_strides; k++) {
        const int row = i * RT::base_tile_rows + row_offset + k * RT::base_tile_elements_per_stride_group;
#pragma unroll
        for (int l = 0; l < RT::base_tile_stride / packing; l++) {
          const int idx = l + k * RT::base_tile_stride / packing;
          const bf16_2 q = base_types::convertor<bf16_2, float2>::convert(
              make_float2(src.tiles[i][j].data[idx].x, src.tiles[i][j].data[idx].y));
          uint32_t w;
          __builtin_memcpy(&w, &q, 4);
          dst_ptr[(row + l * 2) * row_stride + col] = (uint16_t)(w & 0xffffu);
          dst_ptr[(row + l * 2 + 1) * row_stride + col] = (uint16_t)(w >> 16);
        }
      }
    }
  }
}

template <bool WT>
__device__ __forceinline__ void wide_store_x4_so(const void *base, uint32_t voff, u32x4_t v) {
  if constexpr (WT) {
    asm volatile("global_store_dwordx4 %0, %1, %2 sc0 sc1\n\ts_waitcnt expcnt(0)\n\ts_nop 7" ::"v"(voff), "v"(v),
                 "s"(base) : "memory");
  } else {
    asm volatile("global_store_dwordx4 %0, %1, %2\n\ts_waitcnt expcnt(0)\n\ts_nop 7" ::"v"(voff), "v"(v), "s"(base)
                 : "memory");
  }
}

template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void fs_load_frag(RT &dst, const ST &src, int lane) {
  static_assert(RT::rows == ST::rows && RT::cols == ST::cols, "register tile and shared tile must match");
  static_assert(std::is_same_v<typename RT::dtype, bf16_2> && RT::base_tile_stride == 8 &&
                    RT::base_tile_num_strides == 1,
                "bf16, one 16-byte stride per lane");
  static_assert(ST::underlying_subtile_rows == RT::base_tile_rows && ST::underlying_subtile_cols == RT::base_tile_cols,
                "one ds_read_b128 per base tile");
  static_assert(OFF + ((ST::subtiles_per_col - 1) * ST::underlying_subtiles_per_row + ST::subtiles_per_row - 1) *
                        ST::underlying_subtile_bytes <= 65535,
                "ds_read offset field");
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&src.data[0])) +
                        src.swizzle({lane % RT::base_tile_rows, RT::base_tile_stride * (lane / RT::base_tile_rows)});
#pragma unroll
  for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
#pragma unroll
    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
      asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                   : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[ii][jj].data[0]))
                   : "v"(addr), "i"(OFF + (ii * ST::underlying_subtiles_per_row + jj) * ST::underlying_subtile_bytes)
                   : "memory");
    }
  }
}

template <bool WT, typename CT>
__device__ __forceinline__ void store_T16_bf16(bf16 *base, const CT &ct, int m0, int n0, int ld, int lane) {
  const bf16 *tb = base + (size_t)m0 * ld + n0;
  const uint32_t lo = (uint32_t)((lane & 15) * ld + 16 * ((lane >> 4) & 1) + 8 * (lane >> 5)) * sizeof(bf16);
#pragma unroll
  for (int mb = 0; mb < CT::width; mb++) {
    wide_store_x4_so<WT>(tb, lo + (uint32_t)(mb * 16 * ld) * sizeof(bf16), T16_row(ct, mb));
  }
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// GEMM1's GLU activation (dgrad2 recomputes it from G): SwiGLU, or TE's clamped_swiglu_v2, a = min(g, L),
// (clamp(u, -L, L) + O) a sigmoid(A a), whose backward zeroes dgate where g > L and dup where |u| > L.
struct SwiGLU {
  static constexpr bool CLAMP = false;
};
template <float L, float A, float O>
struct ClampedGLU {
  static constexpr bool  CLAMP  = true;
  static constexpr float LIMIT  = L;
  static constexpr float ALPHA  = A;
  static constexpr float OFFSET = O;
};
using GluClamp10 = ClampedGLU<10.0f, 1.0f, 0.0f>;
// The activation of a translation unit built once per activation (moe_train_api.cu, moe_train_api_ks.cu).
#if MOE_ACT_CLAMP10
using MoeAct = GluClamp10;
#else
using MoeAct = SwiGLU;
#endif

// The ClampedGLU bounds -L, L, set per epilogue call: as constants they would be hoisted into SGPRs across the tile
// loop, which then spill.
template <typename ACT>
__device__ __forceinline__ void glu_bounds(float &lo, float &hi) {
  lo = hi = 0.0f;
  if constexpr (ACT::CLAMP) {
    lo = -ACT::LIMIT, hi = ACT::LIMIT;
    asm volatile("" : "+v"(lo), "+v"(hi));
  }
}

// ClampedGLU's sigmoid takes v_rcp_f32 (1 ulp): the IEEE division's v_div_scale masks spill SGPRs.
template <typename ACT>
__device__ __forceinline__ float glu(float g, float u, float lo, float hi) {
  if constexpr (ACT::CLAMP) {
    const float a  = fminf(g, hi);
    const float uc = __builtin_amdgcn_fmed3f(u, lo, hi);
    return a * __builtin_amdgcn_rcpf(1.0f + __expf(-ACT::ALPHA * a)) * (ACT::OFFSET == 0.0f ? uc : uc + ACT::OFFSET);
  } else {
    return silu(g) * u;
  }
}

template <typename ACT, typename RT>
__device__ __forceinline__ void swiglu_inplace(RT &g, const RT &u) {
  constexpr int NPK = sizeof(g.tiles[0][0].data) / sizeof(g.tiles[0][0].data[0]);
  float lo, hi;
  glu_bounds<ACT>(lo, hi);
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
#pragma unroll
      for (int d = 0; d < NPK; d++) {
        g.tiles[i][j].data[d].x = glu<ACT>(g.tiles[i][j].data[d].x, u.tiles[i][j].data[d].x, lo, hi);
        g.tiles[i][j].data[d].y = glu<ACT>(g.tiles[i][j].data[d].y, u.tiles[i][j].data[d].y, lo, hi);
      }
    }
  }
}

struct DispPeers {
  bf16 *recv[MOE_EP];
  int *sig[MOE_EP];
};

struct SendBlk {
  int dst;
  int el;
  int off;
  int cnt;
  int dstbase;
};

constexpr int SIG_STRIDE = 64;

// XFREE[s] (after the arrival signals): backward calls rank s had issued when its latest forward began, i.e. rank s
// no longer reads the Xp of any earlier call. Stored by s, monotone, zeroed with the arenas.
constexpr int XFREE_OFF = MOE_EP * SIG_STRIDE;
// RFREE[s] (after XFREE): rank s finished a round of its call, epoch << 16 | (round + 1). Stored by s, monotone.
constexpr int RFREE_OFF = XFREE_OFF + MOE_EP;

__host__ __device__ __forceinline__ int sig_index(int src, int el) {
  return src * SIG_STRIDE + el;
}

__device__ __forceinline__ void spin_backoff(int &n) {
  if (n < 6) {
    __builtin_amdgcn_s_sleep(1);
    n++;
  } else {
    __builtin_amdgcn_s_sleep(8);
  }
}

constexpr int COMB_BLOCK = 256;

typedef short v8bf __attribute__((ext_vector_type(8)));
__device__ __forceinline__ float bits2f(short h) {
  unsigned int u = (unsigned int)(unsigned short)h << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
__device__ __forceinline__ short f2bf_bits(float f) {
  return __builtin_bit_cast(short, (__bf16)f);
}

struct RbBlk {
  int rb;
  int dst;
  int off;
  int cnt;
};

struct LegBPeers {
  bf16 *comb[MOE_EP];
  int *acc[MOE_EP];
};

constexpr int FOLD_TOK = 8;

// A combine row's words ANDed with m: 0 reads a dropped entry's slot, never written this call, as zeros.
__device__ __forceinline__ v8bf moe_and8(v8bf v, uint32_t m) {
  typedef uint32_t u4 __attribute__((ext_vector_type(4)));
  return __builtin_bit_cast(v8bf, __builtin_bit_cast(u4, v) & m);
}

// DR: an entry of weight 0 (dropped, its slot unwritten this call) adds nothing. YC: the combine rows as read also go
// to yc (null: not kept).
template <int TOPK, bool DR = true, bool YC = false>
__device__ __forceinline__ void moe_fold_item_k(const bf16 *comb, const float *wts, bf16 *out, int T, int H, int i,
                                                const bf16 *ys = nullptr, bf16 *yc = nullptr) {
  typedef __attribute__((address_space(1))) const v8bf gcv8;
  typedef __attribute__((address_space(1))) v8bf gv8;
  typedef __attribute__((address_space(1))) const float gcf;
  const int nv = H >> 3, t0 = i * FOLD_TOK;
  const int n = min(FOLD_TOK, T - t0) * nv;
  gcv8 *cb = (gcv8 *)(comb + (size_t)t0 * TOPK * H);
  gcf *wb = (gcf *)(wts + (size_t)t0 * TOPK);
  gv8 *ob = (gv8 *)(out + (size_t)t0 * H);
  for (int x = (int)threadIdx.x; x < n; x += NUM_THREADS) {
    const int tt = x / nv, cv = x - tt * nv;
    v8bf v[TOPK];
    float w[TOPK];
#pragma unroll
    for (int j = 0; j < TOPK; j++) {
      v[j] = __builtin_nontemporal_load(cb + ((size_t)tt * TOPK + j) * nv + cv);
    }
    if constexpr (YC) {
      if (yc) {
        gv8 *yb = (gv8 *)(yc + (size_t)t0 * TOPK * H);
#pragma unroll
        for (int j = 0; j < TOPK; j++) {
          __builtin_nontemporal_store(v[j], yb + ((size_t)tt * TOPK + j) * nv + cv);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < TOPK; j++) {
      w[j] = wb[tt * TOPK + j];
      if constexpr (DR) {
        const uint32_t m = w[j] != 0.0f ? ~0u : 0u;
        v[j] = moe_and8(v[j], m);
      }
    }
    v8bf o;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < TOPK; j++) {
        acc = fmaf(w[j], bits2f(v[j][e]), acc);
      }
      o[e] = f2bf_bits(acc);
    }
    __builtin_nontemporal_store(o, ob + x);
  }
}

// KS: the small-k kernels (k in {1, 2, 4}); the others take k in {6, 8, 10}.
template <bool KS, bool DR = true, bool YC = false>
__device__ __forceinline__ void moe_fold_item(const bf16 *comb, const float *wts, bf16 *out, int T, int H, int topk,
                                              int i, const bf16 *ys = nullptr, bf16 *yc = nullptr) {
  if constexpr (KS) {
    switch (topk) {
      case 1:
        moe_fold_item_k<1, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
      case 2:
        moe_fold_item_k<2, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
      default:
        moe_fold_item_k<4, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
    }
  } else {
    switch (topk) {
      case 6:
        moe_fold_item_k<6, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
      case 10:
        moe_fold_item_k<10, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
      default:
        moe_fold_item_k<8, DR, YC>(comb, wts, out, T, H, i, ys, yc);
        break;
    }
  }
}

// sy_done, nsy: the shared expert's fc2 tiles (not built: null); kept, as dropping the caller's nsy changes its code.
__device__ __forceinline__ bool moe_fold_arrived(const int *acc, const int *owed, const int *pushed, int ncomm,
                                                 const int *sy_done = nullptr, int nsy = 0) {
  typedef __attribute__((address_space(1))) const int gci;
  const int s = fs_tid() & 63;
  bool late = s < MOE_EP &&
              __hip_atomic_load((gci *)acc + s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < ((gci *)owed)[s];
  late = late || (s == MOE_EP && __hip_atomic_load((gci *)pushed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < ncomm);
  if (sy_done) {
    late = late || (s == MOE_EP + 1 && __hip_atomic_load((gci *)sy_done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < nsy);
  }
  if (__ballot(late)) {
    return false;
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  return true;
}

struct MoePlanTabs {
  int *ctr;
  int *hist, *pk;
  int *cnts[MOE_EP], *cflag[MOE_EP];
  float *send_w;
  int *bblks, *cblks, *cnt_by_src, *start_by_src;
  bf16 *zf;
  int *rec, *status;
  void *ba;
  int cap, nch;
};

// The fused router's per-layer state: not built (the training API takes the caller's routing); MoeFwdArgs and
// SingleArgs keep it, zero, so the kernels' argument layout stays as it was.
struct RouterArgs {
  float *logits, *w, *psum, *dtk;
  int *idx, *pcnt, *cnt, *rbdone, *routed, *bctr, *rpdone, *rdgdone, *rid;
  unsigned *seedv;
  bf16 *dl, *dxr;
  float *dlbp;
  int T, E, El, Ec, Ed, k, groups, gtopk, nb, force, rank;
  float scale, aux_c;
};

struct MoeFwdArgs {
  int *send_slot;
  DispPeers dpeers;
  int *my_sig;
  int *cnt_by_src, *start_by_src, *row_off, *cnt;
  TileDesc *q1, *q2;
  int *gather;
  int *gid;
  int *idx_done;
  bf16 *act, *c2;
  int *a_done;
  int *tile_done;
  int *g2_ctr, *fold_ctr;
  int *disp_done;
  RbBlk *rbs;
  int *lb_src, *rb_hi;
  int *slot[MOE_EP];
  LegBPeers lpeers;
  float *wts;
  int *owed;
  MoePlanTabs p;
  int nblks, nrbs, my_rank, dncomm, cncomm;
  int H, I, T, topk, E_loc;
  // r through sactcs, and pbase: the fused router's, the shared expert's and MXFP8's operands, not built; zero.
  RouterArgs r;
  bf16 *sact, *sy;
  int *sa_done, *sy_done;
  uint8_t *actc, *actcs, *xq;
  int *xq_done;
  int mxt, xc_fwd;
  uint8_t *xcp[MOE_EP], *xcsp[MOE_EP], *xshp[MOE_EP];
  uint8_t *sactc, *sactcs;
  // The layer's {epoch, xepoch} and the workspace's {backward count, latest call was a forward}, on the device so
  // graph replays and eager calls advance them alike.
  unsigned *ep, *wep;
  // bf16 training's pool (null: none), this context's slot in it, its rows, a call's rows (head), the Xp rows reserved
  // for each layer still to store (res), and G's and A's pool rows per 256 rows (pool_q).
  int *pool;
  int pslot, prows, phead, pres, pgq, paq;
  // The backward's A when a forward keeps Xp only or nothing (SingleArgs::ascr), its Xp when it keeps nothing (the
  // workspace's region its X is pushed again into).
  bf16 *pascr, *pxre;
  // bf16 over capacity: the round barrier's release word (local); each send unit's first row in its destination's
  // slot array (the backward's rounds push the slots again).
  int *rgo, *sbase;
  uint8_t *pbase;
};

// The training API's data flow (kosmos_moe_train_forward / _backward), by the tier the forward keeps (TRAIN_INT_KEPT:
// KOSMOS_MOE_SAVE_* 0 nothing, 1 Xp, 2 Xp, G and A; -1 inference), decided from the first pass's layout alike on every
// rank.
// GEMM1 writes G to the caller's g (tier 2, else nowhere) and A to the caller's a (tier 2, else the instance's act).
// A kept Xp (tiers 1 and 2) is written by GEMM1 and the drained tail (XP_GEMM1_*).
// The fold copies each token slot's Y row to the caller's y (tiers 0 to 2).
// The combine's tail copies the receive slots and the instance's tables into the route record (route_fwd_copy); the
// last workgroup writes its count matrix and header (route_fwd_finish).
// The backward copies the record's tables back into the instance (route_bwd_restore), then reads Xp, G and A (tier 2),
// reads Xp and recomputes G and A (tier 1), or pushes X again and recomputes G and A (tier 0).

// A table segment the route record keeps: (instance table, byte offset in the record, bytes).
struct RouteSeg {
  void *dev;
  unsigned off, bytes;
};

// The route record: a header of ROUTE_HDR_* ints at 0, the call's count matrix (8 x 8 E_loc ints) at
// ROUTE_COUNTS_OFF, the receive slots at off_slot, then the table segments.
constexpr int ROUTE_COUNTS_OFF = 256;
constexpr int ROUTE_HDR_KEPT = 0, ROUTE_HDR_NR = 1, ROUTE_HDR_NROW = 2, ROUTE_HDR_MPAD = 3, ROUTE_HDR_MPADX = 4,
              ROUTE_HDR_STATUS = 5, ROUTE_HDR_EPOCH = 6, ROUTE_HDR_CROWS = 7, ROUTE_HDR_MAGIC = 8;
constexpr int ROUTE_MAGIC = 0x4B4D5632;
// The instance's counters: copy claims and finished plan workgroups (forward), the start barrier (backward).
constexpr int INST_CTR_CLAIM = 0, INST_CTR_LATE = 1, INST_CTR_BWD = 2, INST_CTR_N = 4;
// Of every XP_GEMM1_PERIOD row blocks of a kept Xp, GEMM1 writes XP_GEMM1_BLOCKS from the A tiles it already holds in
// LDS (an extra write, no extra read) and the drained tail copies the rest (a read and a write).
// The 3-of-4 split was chosen by measurement (2026-11-22).
constexpr int XP_GEMM1_PERIOD = 4, XP_GEMM1_BLOCKS = 3;

// A training forward's operands: the caller's saved buffers (written by tier), the route record (null: inference), the
// tier cap and the rows the buffers hold; the instance's counters and the record's table segments.
struct TrainApiCall {
  bf16 *xp, *g, *a, *y;
  uint8_t *route;
  int *ctr;
  const RouteSeg *seg;
  int nseg, tier, rows, off_slot;
};

// The TrainApiCall in LDS: pointers (train_ptrs, TRAIN_PTR_*) and ints (train_ints, TRAIN_INT_*). From the first
// pass's layout: TRAIN_INT_KEPT to TRAIN_INT_CROWS, and where GEMM1 writes G (TRAIN_PTR_G_OUT, null: nowhere) and A
// (TRAIN_PTR_A_OUT) and the fold copies Y (TRAIN_PTR_Y_OUT, null: nowhere).
constexpr int TRAIN_PTR_XP = 0, TRAIN_PTR_G = 1, TRAIN_PTR_A = 2, TRAIN_PTR_Y = 3, TRAIN_PTR_ROUTE = 4,
              TRAIN_PTR_CTR = 5, TRAIN_PTR_SEG = 6, TRAIN_PTR_G_OUT = 7, TRAIN_PTR_A_OUT = 8, TRAIN_PTR_Y_OUT = 9,
              TRAIN_PTR_N = 10;
constexpr int TRAIN_INT_NSEG = 0, TRAIN_INT_TIER = 1, TRAIN_INT_ROWS = 2, TRAIN_INT_OFFSLOT = 3, TRAIN_INT_KEPT = 4,
              TRAIN_INT_MPAD = 5, TRAIN_INT_MPADX = 6, TRAIN_INT_NR = 7, TRAIN_INT_CROWS = 8, TRAIN_INT_ITEM = 9,
              TRAIN_INT_N = 10;
__device__ __forceinline__ unsigned long long (&train_ptrs())[TRAIN_PTR_N] {
  __shared__ unsigned long long s[TRAIN_PTR_N];
  return s;
}
__device__ __forceinline__ int (&train_ints())[TRAIN_INT_N] {
  __shared__ int s[TRAIN_INT_N];
  return s;
}


constexpr int PLAN_MAXU       = MOE_EP * SIG_STRIDE;
constexpr int PLAN_SUB        = 4;
constexpr int PLAN_CH         = PLAN_SUB * NUM_THREADS;
constexpr int PLAN_HEAD       = 16;
constexpr int PLAN_IDS_OFF    = 16384;
constexpr int PLAN_RUN_OFF    = 26624;
constexpr int PLAN_LAYOUT_OFF = 32768;

constexpr int PLAN_BAD_ID     = 1;
constexpr int PLAN_DUP_ID     = 2;
constexpr int PLAN_OVERFLOW   = 4;
// Notes, not errors (bf16 training): the call ran in rounds; its forward kept Xp only; it kept nothing.
constexpr int PLAN_ROUNDS     = 8;
constexpr int PLAN_KEPT_XP    = 16;
constexpr int PLAN_KEPT_NONE  = 32;
// An error: a single-pass call kept nothing (the plan sends a call that cannot keep its Xp into rounds).
constexpr int PLAN_ONEPASS_NONE = 64;
constexpr int PC_ARR = 0, PC_SCAT = 1, PC_BAD = 2, PC_N = 3;

// A call's routing layout in LDS: units u = e * MOE_EP + x, experts eg = d * E_loc + e.
struct MoeLayout {
  int col[PLAN_MAXU], part[PLAN_MAXU], pof[PLAN_MAXU], cpof[PLAN_MAXU];
  int snd[PLAN_MAXU], sof[PLAN_MAXU], rcv[PLAN_MAXU], uof[PLAN_MAXU], dbase[PLAN_MAXU], sbase[PLAN_MAXU];
  int ro[SIG_STRIDE + 1], cn[SIG_STRIDE], est[SIG_STRIDE + 1];
  int o1[SIG_STRIDE + 1], o2[SIG_STRIDE + 1], e2[SIG_STRIDE];
  int tmp[2 * NUM_WARPS];
  int mpad, mpad_max, ovf, drows, crows;
  int colp[PLAN_MAXU], ppart[PLAN_MAXU], uofp[PLAN_MAXU];
  int cur, gr, ar;
  int nr, wa[MOE_EP], wb[MOE_EP], wn[MOE_EP], wo[MOE_EP], wm[SIG_STRIDE], cpar;
};

typedef gl<bf16, 1, 1, -1, -1> FsGl;
typedef const __attribute__((address_space(4))) MoeFwdArgs FsArgsC;
typedef __attribute__((address_space(1))) int gint;

constexpr int FS_G1 = 0, FS_G2 = 1;

template <typename T>
__device__ __forceinline__ T *fs_uniform_ptr(T *p) {
  const unsigned long long a = (unsigned long long)p;
  const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)a);
  const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(a >> 32));
  return (T *)(((unsigned long long)hi << 32) | lo);
}

// make_srsrc with every dword forced scalar: its stride select otherwise lands in a VGPR under SGPR pressure and
// every LDS-DMA through it becomes a waterfall loop.
__device__ __forceinline__ i32x4 fs_srd(const void *p, uint32_t range, uint32_t stride) {
  const i32x4 r = make_srsrc(p, range, stride);
  return i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
               __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
}

// A plan's args block in the constant address space (s_load, not FLAT), opaque to the optimizer: fields are re-loaded
// where used, not hoisted out of the tile loop into long-lived registers.
template <typename A>
__device__ __forceinline__ const __attribute__((address_space(4))) A *fs_args(const A *p) {
  const __attribute__((address_space(4))) A *q = (const __attribute__((address_space(4))) A *)p;
  asm volatile("" : "+s"(q));
  return q;
}

// Uniform and address space 1: a pointer read from memory is generic, and every access through it would be FLAT
// (lgkmcnt, and the FLAT-store hazard).
template <typename T>
__device__ __forceinline__ __attribute__((address_space(1))) T *fs_g(T *p) {
  return (__attribute__((address_space(1))) T *)fs_uniform_ptr(p);
}

template <typename T>
__device__ __forceinline__ T *train_ptr(int i) {
  return fs_uniform_ptr((T *)*fs_lds(&train_ptrs()[i]));
}

__device__ __forceinline__ int train_int(int i) {
  return __builtin_amdgcn_readfirstlane(*fs_lds(&train_ints()[i]));
}

// 16-byte words [lo, hi) of src to dst, by the workgroup.
__device__ __forceinline__ void wg_copy16(void *dst, const void *src, size_t lo, size_t hi) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  gv4i *d = (gv4i *)fs_uniform_ptr(dst);
  const gv4i *s = (const gv4i *)fs_uniform_ptr(src);
  size_t i = lo + fs_tid();
  for (; i + 3 * NUM_THREADS < hi; i += 4 * NUM_THREADS) {
    v4i v[4];
#pragma unroll
    for (int z = 0; z < 4; z++) {
      v[z] = __builtin_nontemporal_load(s + i + z * NUM_THREADS);
    }
#pragma unroll
    for (int z = 0; z < 4; z++) {
      __builtin_nontemporal_store(v[z], d + i + z * NUM_THREADS);
    }
  }
  for (; i < hi; i += NUM_THREADS) {
    __builtin_nontemporal_store(__builtin_nontemporal_load(s + i), d + i);
  }
}

// C: stride NUM_THREADS (blockDim.x after a long loop keeps the kernarg pointer live).
template <bool C = false>
__device__ __forceinline__ void fs_reset_words(gint *w, int n) {
  if constexpr (C) {
    for (int i = fs_tid(); i < n; i += NUM_THREADS) {
      __hip_atomic_store(w + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    for (int i = fs_tid(); i < n; i += blockDim.x) {
      __hip_atomic_store(w + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

__device__ __forceinline__ FsGl fs_view(const FsGl &proto, const bf16 *p, int rows, int cols) {
  FsGl v(proto);
  v.raw_ptr = fs_uniform_ptr(const_cast<bf16 *>(p));
  v.rows_internal = __builtin_amdgcn_readfirstlane(rows);
  v.cols_internal = __builtin_amdgcn_readfirstlane(cols);
  return v;
}

// bf16 training's pool, in rows of 2H bytes: word 0 the next call's base row, word 1 the context slots in use, then
// {base, xtop, top} per slot (top 0: none, xtop -1: a forward that kept nothing; top == xtop: Xp kept, G and A not). A call's block from its base: Xp (R
// rows), G (R x 2I bf16), A (R x I bf16), each 256 rows of G and A taking gq and aq pool rows (pool_q); R is every
// rank's Mpad plus the zero row's 256-row block.
constexpr int POOL_MAXCTX = NUM_THREADS - 1;
// After the records: whether the next storing call keeps a full-capacity Xp, whatever its slot, then the same for the
// slot updated last (its own record replaced), then that slot (pool_update, pool_fit).
constexpr int POOL_FIT = 2 + 3 * POOL_MAXCTX;

__host__ __device__ __forceinline__ int pool_q(int cols, int H) {
  return (BLOCK_ROW * cols + H - 1) / H;
}

__host__ __device__ __forceinline__ int pool_a_row(int base, int R, int gq) {
  return base + R + R / BLOCK_ROW * gq;
}

__host__ __device__ __forceinline__ int pool_end(int base, int R, int gq, int aq) {
  return pool_a_row(base, R, gq) + R / BLOCK_ROW * aq;
}

// No record but slot's overlaps rows [c, c + w) (rec: the {base, xtop, top} records, in LDS or the pool).
template <typename P>
__host__ __device__ __forceinline__ bool pool_free(P rec, int n, int slot, int c, int w) {
  bool f = true;
  for (int j = 0; j < n; j++) {
    const int b = rec[3 * j], e = rec[3 * j + 2];
    f &= j == slot || e == 0 || b >= c + w || e <= c;
  }
  return f;
}

// Candidate t's base: 0, every other record's top, then this call's xtop and top.
template <typename P>
__host__ __device__ __forceinline__ int pool_cand(P rec, int n, int slot, int t, int xtop, int top) {
  return t < n ? (t == slot ? 0 : rec[3 * t + 2]) : t == n ? 0 : t == n + 1 ? xtop : top;
}

// A call from base keeps its Xp (up to xtop) with candidate c as the next base: the test pool_update and the plan
// (moe_plan.cuh, plan_pool_fits) share.
template <typename P>
__host__ __device__ __forceinline__ bool pool_keeps_xp(P rec, int n, int slot, int rows, int head, int base, int xtop, int t,
                                                       int c) {
  return (c > 0 || t == n) && pool_free(rec, n, slot, c, head) && c + head <= rows && (base >= c + head || xtop <= c);
}

// A call's last workgroup: the slot's record set to {base, xtop, top} (top 0: freed) and the next base found, the
// lowest of 0 and the records' tops with a call's head rows free above it. A forward keeps G and A only if res more
// rows stay free for each other slot without a record (the layers still to come), else Xp only, else nothing; returns
// 2, 1 or 0. One pass: each thread tests one candidate base for every outcome. scr: 3 POOL_MAXCTX + 6 ints of LDS.
__device__ __forceinline__ int pool_update(gint *pool, int slot, int rows, int head, int res, int xr, int base, int xtop,
                                           int top, int *scr, bool none = false) {
  const int t = fs_tid(), n = pool[1];
  int *rec = scr + 6;
  for (int i = t; i < 3 * n; i += NUM_THREADS) {
    rec[i] = pool[2 + i];
  }
  if (t < 5) {
    scr[t] = t == 4 ? 0 : 0x7FFFFFFF;
  }
  __syncthreads();
  if (t < n && t != slot && rec[3 * t + 2] == 0 && rec[3 * t + 1] == 0) {
    atomicAdd(&scr[4], 1);
  }
  __syncthreads();
  const int need = head + scr[4] * res;
  // Candidates: 0, every other record's top, and this call's own xtop and top.
  if (t <= n + 2) {
    const int c = pool_cand(rec, n, slot, t, xtop, top);
    const bool ok = c > 0 || t == n, fh = pool_free(rec, n, slot, c, head);
    const bool f = ok && pool_free(rec, n, slot, c, need) && c + need <= rows && (base >= c + need || top <= c);
    const bool fh2 = ok && fh && c + head <= rows && (base >= c + head || top <= c);
    const bool x = pool_keeps_xp(rec, n, slot, rows, head, base, xtop, t, c);
    const bool z = ok && fh && c + head <= rows;
    if (top && f) {
      atomicMin(&scr[0], c);
    }
    if (top && fh2) {
      atomicMin(&scr[1], c);
    }
    if (top && x) {
      atomicMin(&scr[2], c);
    }
    if (z) {
      atomicMin(&scr[3], c);
    }
  }
  __syncthreads();
  const int keep = none ? 0 : scr[0] != 0x7FFFFFFF ? 2 : scr[2] != 0x7FFFFFFF ? 1 : 0;
  if (t == 0) {
    rec[3 * slot] = top ? base : 0, rec[3 * slot + 1] = keep ? xtop : top ? -1 : 0;
    rec[3 * slot + 2] = keep == 2 ? top : keep ? xtop : 0, scr[5] = scr[keep == 2 ? 1 : keep ? 2 : 3];
    pool[2 + 3 * slot] = rec[3 * slot], pool[3 + 3 * slot] = rec[3 * slot + 1], pool[4 + 3 * slot] = rec[3 * slot + 2];
    pool[0] = scr[5];
  }
  if (t == 0) {
    scr[4] = 0;
  }
  __syncthreads();
  // For the next storing call's plan: a full-capacity Xp (xr rows) kept at the new base, by any slot, and by this one.
  const int nb = scr[5];
  if (t <= n + 1) {
    if (pool_keeps_xp(rec, n, -1, rows, head, nb, nb + xr, t, pool_cand(rec, n, -1, t, nb + xr, 0))) {
      atomicOr(&scr[4], 1);
    }
    if (pool_keeps_xp(rec, n, slot, rows, head, nb, nb + xr, t, pool_cand(rec, n, slot, t, nb + xr, 0))) {
      atomicOr(&scr[4], 2);
    }
  }
  __syncthreads();
  if (t == 0) {
    pool[POOL_FIT] = scr[4] & 1, pool[POOL_FIT + 1] = scr[4] >> 1, pool[POOL_FIT + 2] = slot;
  }
  return keep;
}

__device__ __forceinline__ int fs_seg(const int *off, int n, int x) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (off[mid] <= x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

__device__ __forceinline__ void moe_plan_local(FsArgsC *a, const int *idx, const float *w, int wi, int nw,
                                               unsigned epoch, char *scr, int *tpe);
template <bool PK = false, bool RK = true, bool O = true>
__device__ __forceinline__ void moe_plan_layout(FsArgsC *a, MoeLayout *L, unsigned epoch, int r = -1, int fz = 0);
__device__ __forceinline__ void plan_wait(gint *c, int want, bool sys);
__device__ __forceinline__ void plan_set_forced(FsArgsC *a, int fz);
template <bool O>
__device__ __forceinline__ void moe_plan_queues(FsArgsC *a, MoeLayout *L);
template <bool RK = true, bool O = true>
__device__ __forceinline__ void moe_plan_tables(FsArgsC *a, const MoeLayout *L, int wi, int nw);
template <bool RK = true, bool O = true>
__device__ __forceinline__ void moe_plan_late(FsArgsC *a, const MoeLayout *L, int wi, int nw, bool sc = true);
__device__ __forceinline__ int plan_pool_rows(const int *rec);
__device__ __forceinline__ void plan_set_x0(FsArgsC *a, const bf16 *x);

// A stored forward's last workgroup keeps its block of the pool (moe_plan.cuh).
__device__ __noinline__ void pool_commit(const MoeFwdArgs *args, int *scr, int R, bool rounds);
// The training API's forward: one copy item into the record and the caller's Xp, once the queue is drained (false: none
// left); the last workgroup's record header and count matrix (moe_plan.cuh).
__device__ __forceinline__ bool route_fwd_copy(const MoeFwdArgs *args, uint8_t *route, int *ctr, const RouteSeg *seg,
                                             bf16 *xp, int nseg, int off_slot, int kept, int mpad, int nr, int crows,
                                             unsigned epoch, int nw, int *s_item);
__device__ __forceinline__ void route_fwd_finish(const MoeFwdArgs *args, uint8_t *route, int *ctr, int kept, int tier,
                                               int mpad, int mpadx, int nr, int crows, unsigned epoch);

template <bool O = false>
__device__ __forceinline__ void fs_index_phase(FsArgsC *a, const MoeLayout *L, int w, int nw, bool shq = true) {
  const int E = a->E_loc, tN1 = 2 * a->I / BLOCK_COL, tN2 = a->H / BLOCK_COL, Mpad = L->mpad;
  const int *s_ro = L->ro, *s_cn = L->cn, *s_o1 = L->o1, *s_o2 = L->o2, *s_e2 = L->e2;
  const int nt1 = s_o1[E], nt2 = s_o2[E];
  const int g0 = w * NUM_THREADS + (O ? fs_tid() : (int)threadIdx.x), gs = nw * NUM_THREADS;
  auto *q1 = fs_g(a->q1);
  auto *q2 = fs_g(a->q2);
  for (int t = g0; t < nt1; t += gs) {
    const int e     = fs_seg(s_o1, E, t);
    const int mt    = s_o1[e + 1] - s_o1[e];
    const int li    = t - s_o1[e];
    const int mtr   = mt / tN1;
    const int band  = li / (MOE_WALK_GM * tN1);
    const int first = band * MOE_WALK_GM;
    const int gsz   = min(mtr - first, MOE_WALK_GM);
    const int q     = li - band * MOE_WALK_GM * tN1;
    q1[t].tile_m = s_ro[e] / BLOCK_ROW + first + q % gsz;
    q1[t].tile_n = q / gsz;
    q1[t].expert = e;
  }
  for (int t = g0; t < nt2; t += gs) {
    const int p  = fs_seg(s_o2, E, t);
    const int e  = s_e2[p];
    const int li = t - s_o2[p];
    q2[t].tile_m = s_ro[e] / BLOCK_ROW + li / tN2;
    q2[t].tile_n = li % tN2;
    q2[t].expert = e;
  }
  auto *gather = fs_g(a->gather);
  auto *gid    = fs_g(a->gid);
  for (int r = g0; r <= Mpad; r += gs) {
    gid[r] = r;
    if (r < Mpad) {
      const int e = fs_seg(s_ro, E, r);
      gather[r] = r < s_ro[e] + s_cn[e] ? r : Mpad;
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(fs_g(a->idx_done), 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ __forceinline__ void fs_xfree_store(FsArgsC *a, int w, int xepoch) {
  if (threadIdx.x == 0 && w < MOE_EP) {
    __hip_atomic_store((gint *)a->dpeers.sig[w] + XFREE_OFF + a->my_rank, xepoch, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// MT: the row's remainder in one masked pass (as fs_combine_push's BURST 0). OT: the thread id re-derived (opaque)
// rather than taken from the caller.
template <bool MT = false, bool OT = false>
__device__ __forceinline__ void fs_dispatch_role(FsArgsC *a, const MoeLayout *L, const bf16 *__restrict__ X, int w,
                                                 int H, int xepoch) {
  const int tid     = OT ? fs_tid() : (int)threadIdx.x;
  const int my_rank = __builtin_amdgcn_readfirstlane(a->my_rank);
  const int ncomm   = __builtin_amdgcn_readfirstlane(a->dncomm);
  const int k       = __builtin_amdgcn_readfirstlane(a->topk);
  if (tid == 0) {
    gint *xf = (gint *)a->my_sig + XFREE_OFF + w % MOE_EP;
    const unsigned want = (unsigned)xepoch;
    int n = 0;
    while ((int)((unsigned)__hip_atomic_load(xf, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - want) < 0) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(xf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    gint *sc = fs_g(a->p.ctr) + PC_SCAT;
    n = 0;
    while (__hip_atomic_load(sc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < a->p.nch) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(sc, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  typedef __attribute__((address_space(1))) const v4i gcv4i;
  constexpr int RS  = NUM_THREADS / kittens::WARP_THREADS;
  constexpr int TSG = kittens::WARP_THREADS;
  const int u_per_row = H / 8;
  const int sg = tid / TSG, stl = tid % TSG;
  const int NSP   = ncomm / MOE_EP;
  const int gslc  = w / MOE_EP;
  const int nblks = __builtin_amdgcn_readfirstlane(a->nblks);
  const auto *send_slot = fs_g(a->send_slot);
  // Row split: workgroups w, w + MOE_EP, ... push slices of the same unit; every slice, empty or not, release-adds 1.
  // Each row also carries its routing slot t * k + j into the destination's slot array, for the combine.
  for (int j = w % MOE_EP; j < nblks; j += MOE_EP) {
    SendBlk b;
    b.dst = j % MOE_EP, b.el = j / MOE_EP;
    b.off = __builtin_amdgcn_readfirstlane(L->sof[j]);
    b.cnt = __builtin_amdgcn_readfirstlane(L->snd[j]);
    b.dstbase = __builtin_amdgcn_readfirstlane(L->dbase[j] + (L->cur));
    const int sbase = __builtin_amdgcn_readfirstlane(L->sbase[j]);
    auto *dbase = fs_g(a->dpeers.recv[b.dst]);
    auto *dslot = fs_g(a->slot[b.dst]) + sbase;
    auto *dsig  = fs_g(a->dpeers.sig[b.dst]) + sig_index(my_rank, b.el);
    const int rlo = (int)((long long)b.cnt * gslc / NSP);
    const int rhi = (int)((long long)b.cnt * (gslc + 1) / NSP);
    if (b.cnt == 0) {
      if (tid == 0) {
        (void)__hip_atomic_fetch_add(dsig, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      __syncthreads();
      continue;
    }
    for (int i0 = rlo; i0 < rhi; i0 += RS) {
      const int i = i0 + sg;
      if (i < rhi) {
        const int slot = __builtin_amdgcn_readfirstlane(send_slot[b.off + i]);
        const int srow = slot / k;
        if (stl == 0) {
          __builtin_nontemporal_store(slot, dslot + i);
        }
        gcv4i *src = (gcv4i *)(fs_g(X) + (size_t)srow * H);
        gv4i *dst = (gv4i *)(dbase + (size_t)(b.dstbase + i) * H);

        int u = stl;
        for (; u + 7 * TSG < u_per_row; u += 8 * TSG) {
          const v4i x0 = src[u], x1 = src[u + TSG];
          const v4i x2 = src[u + 2 * TSG], x3 = src[u + 3 * TSG];
          const v4i x4 = src[u + 4 * TSG], x5 = src[u + 5 * TSG];
          const v4i x6 = src[u + 6 * TSG], x7 = src[u + 7 * TSG];
          __builtin_nontemporal_store(x0, &dst[u]);
          __builtin_nontemporal_store(x1, &dst[u + TSG]);
          __builtin_nontemporal_store(x2, &dst[u + 2 * TSG]);
          __builtin_nontemporal_store(x3, &dst[u + 3 * TSG]);
          __builtin_nontemporal_store(x4, &dst[u + 4 * TSG]);
          __builtin_nontemporal_store(x5, &dst[u + 5 * TSG]);
          __builtin_nontemporal_store(x6, &dst[u + 6 * TSG]);
          __builtin_nontemporal_store(x7, &dst[u + 7 * TSG]);
        }
        if (u < u_per_row) {
          v4i x[7];
#pragma unroll
          for (int z = 0; z < 7; z++) {
            if (u + z * TSG < u_per_row) {
              x[z] = src[u + z * TSG];
            }
          }
#pragma unroll
          for (int z = 0; z < 7; z++) {
            if (u + z * TSG < u_per_row) {
              __builtin_nontemporal_store(x[z], &dst[u + z * TSG]);
            }
          }
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      (void)__hip_atomic_fetch_add(dsig, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// Wave 0 claims a unit u = e * MOE_EP + w % MOE_EP (experts scanned from a per-workgroup offset): non-empty, row blocks
// [rb, rb_hi] done, won on its claim word, then acquired. Returns u, -1 (none ripe yet) or -3 (every unit taken).
__device__ __forceinline__ int fs_combine_claim(FsArgsC *a, int w, unsigned *s_dmask, int *s_dsel, int *s_dall) {
  __syncthreads();
  if (threadIdx.x == 0 && !*s_dall) {
    *s_dall = __hip_atomic_load(fs_g(a->disp_done), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= a->dncomm;
  }
  __syncthreads();
  if (!__builtin_amdgcn_readfirstlane(*s_dall)) {
    return -1;
  }
  if (threadIdx.x < 64) {
    const int E = a->E_loc, x = w % MOE_EP, r = (w / MOE_EP) * E / (a->cncomm / MOE_EP), tN = a->H / BLOCK_COL;
    const auto *rbs   = fs_g(a->rbs);
    const auto *rb_hi = fs_g(a->rb_hi);
    auto *td    = fs_g(a->tile_done);
    auto *claim = fs_g(a->fold_ctr) + 2;
    const int e = fs_tid() & 63;
    const unsigned long long m0 = ((unsigned long long)s_dmask[1] << 32) | s_dmask[0];
    bool done = e >= E || ((m0 >> e) & 1ull);
    bool ripe = false;
    if (!done) {
      const int u = e * MOE_EP + x;
      if (rbs[u].cnt == 0 || __hip_atomic_load(claim + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
        done = true;
      } else {
        ripe = true;
        for (int R = rbs[u].rb; R <= rb_hi[u]; R++) {
          if (__hip_atomic_load(td + R, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tN) {
            ripe = false;
            break;
          }
        }
      }
    }
    unsigned long long md = __ballot(done), mr = __ballot(ripe);
    int sel = -1;
    while (mr) {
      const unsigned long long rot = r ? (mr >> r) | (mr << (64 - r)) : mr;
      const int c = (__builtin_ctzll(rot) + r) & 63;
      mr &= ~(1ull << c);
      md |= 1ull << c;
      int won = 0;
      if (e == 0) {
        int zero = 0;
        won = __hip_atomic_compare_exchange_strong(claim + c * MOE_EP + x, &zero, 1, __ATOMIC_RELAXED,
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      if (__builtin_amdgcn_readfirstlane(won)) {
        sel = c * MOE_EP + x;
        break;
      }
    }
    if (sel < 0 && !~md) {
      sel = -3;
    }
    if (e == 0) {
      s_dmask[0] = (unsigned)md;
      s_dmask[1] = (unsigned)(md >> 32);
      if (sel >= 0) {
        for (int R = rbs[sel].rb; R <= rb_hi[sel]; R++) {
          (void)__hip_atomic_load(td + R, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      *s_dsel = sel;
    }
  }
  __syncthreads();
  return __builtin_amdgcn_readfirstlane(*s_dsel);
}

// The forward's pointer and epoch arguments, staged in LDS (s_kp) and read back at each use: held in SGPRs across
// the start phase and the tile loop, they spill.
constexpr int FK_X = 0, FK_W1 = 1, FK_W2 = 2, FK_OUT = 3, FK_G = 4, FK_IDX = 5, FK_W = 6, FK_EP = 7, FK_N = 8;

// The arrival word value of this call (its epoch + 1).
__device__ __forceinline__ int fk_want(const unsigned long long *kp) {
  return __builtin_amdgcn_readfirstlane((int)(unsigned)*fs_lds(&kp[FK_EP])) + 1;
}

// The forward arena's per-expert arrival words, after the per-source counts: word g = s * E_loc + e holds the epoch + 1
// of the latest call whose unit (expert e of source s) has landed here.
__device__ __forceinline__ gint *fs_unit_flags(FsArgsC *a, int r) {
  return fs_g(a->lpeers.acc[r]) + MOE_EP;
}

// FR: publish the unit's arrival word (fk_want(kp)) for the per-block fold readiness instead of the per-source count.
// BURST 0: depth 8 with a masked last pass (H without an exact depth, moe_prod_burst).
template <int BURST, bool FR>
__device__ __forceinline__ void fs_combine_push(FsArgsC *a, int j, int H, const unsigned long long *kp) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4i gv4i;
  typedef __attribute__((address_space(1))) const v4i gcv4i;
  constexpr int RS  = NUM_THREADS / kittens::WARP_THREADS;
  constexpr int TSG = kittens::WARP_THREADS;
  constexpr int PB  = BURST ? BURST : 8;
  const int tid = fs_tid(), sg = tid / TSG, stl = tid % TSG;
  const int u_per_row = H / 8;
  const auto *rbp = fs_g(a->rbs) + j;
  RbBlk rb;
  rb.rb = rbp->rb, rb.dst = rbp->dst, rb.off = rbp->off, rb.cnt = rbp->cnt;
  const auto *c2      = fs_g(a->c2);
  const auto *lb_src  = fs_g(a->lb_src);
  const auto *lb_slot = fs_g(a->slot[a->my_rank]);
  auto *dbase = fs_g(a->lpeers.comb[rb.dst]);
  for (int i0 = 0; i0 < rb.cnt; i0 += RS) {
    const int i = i0 + sg;
    if (i < rb.cnt) {
      const int srow = __builtin_amdgcn_readfirstlane(lb_src[rb.off + i]);
      const int slot = __builtin_amdgcn_readfirstlane(lb_slot[rb.off + i]);
      gcv4i *src = (gcv4i *)(c2 + (size_t)srow * H);
      gv4i *dst = (gv4i *)(dbase + (size_t)slot * H);
      int u = stl;
      for (; u + (PB - 1) * TSG < u_per_row; u += PB * TSG) {
        v4i x[PB];
#pragma unroll
        for (int z = 0; z < PB; z++) {
          x[z] = src[u + z * TSG];
        }
#pragma unroll
        for (int z = 0; z < PB; z++) {
          __builtin_nontemporal_store(x[z], &dst[u + z * TSG]);
        }
      }
      if constexpr (BURST == 0) {
        // Loads clamp to the row's last unit; a slot stores when the wave's first lane is in the row (a scalar
        // branch, no exec masks), its lanes past the row rewriting the last unit.
        const int ub = __builtin_amdgcn_readfirstlane(u - stl);
        v4i x[PB - 1];
#pragma unroll
        for (int z = 0; z < PB - 1; z++) {
          x[z] = src[min(u + z * TSG, u_per_row - 1)];
        }
#pragma unroll
        for (int z = 0; z < PB - 1; z++) {
          if (ub + z * TSG < u_per_row) {
            __builtin_nontemporal_store(x[z], &dst[min(u + z * TSG, u_per_row - 1)]);
          }
        }
      } else {
        for (; u < u_per_row; u += TSG) {
          __builtin_nontemporal_store(src[u], &dst[u]);
        }
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    if constexpr (FR) {
      __hip_atomic_store(fs_unit_flags(a, rb.dst) + a->my_rank * a->E_loc + j / MOE_EP, fk_want(kp), __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    } else {
      __hip_atomic_fetch_add(fs_g(a->lpeers.acc[rb.dst]) + a->my_rank, rb.cnt, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ gint *fs_my_acc(FsArgsC *a) {
  return fs_g(a->lpeers.acc[a->my_rank]);
}

// Wave 0: every row of fold item i (at most 128) has landed (each row's unit arrival word is want), then acquire.
// A dropped entry (expert id outside [0, 8 E_loc)) has no row.
__device__ __forceinline__ bool fs_fold_block_ready(FsArgsC *a, const unsigned long long *kp, int i) {
  typedef __attribute__((address_space(1))) const int gci;
  const int want = fk_want(kp);
  const int *idx = fs_uniform_ptr((const int *)*fs_lds(&kp[FK_IDX]));
  const gint *fl = fs_unit_flags(a, a->my_rank);
  const int k = a->topk, n = min(FOLD_TOK, a->T - i * FOLD_TOK) * k;
  const unsigned EG = MOE_EP * a->E_loc;
  gci *ib = (gci *)idx + (size_t)i * FOLD_TOK * k;
  const int r = fs_tid() & 63;
  const int e0 = r < n ? ib[r] : -1, e1 = r + 64 < n ? ib[r + 64] : -1;
  int f0 = want, f1 = want;
  if ((unsigned)e0 < EG) {
    f0 = __hip_atomic_load(fl + e0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if ((unsigned)e1 < EG) {
    f1 = __hip_atomic_load(fl + e1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (__ballot(f0 != want || f1 != want)) {
    return false;
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  return true;
}

template <bool R, typename T>
__device__ __forceinline__ T *fk_ptr(T *v, const unsigned long long *kp, int i) {
  if constexpr (R) {
    return fs_uniform_ptr((T *)kp[i]);
  } else {
    return v;
  }
}

template <bool R>
__device__ __forceinline__ int fk_epoch(int v, const unsigned long long *kp, int sh) {
  if constexpr (R) {
    return __builtin_amdgcn_readfirstlane((int)(unsigned)(kp[FK_EP] >> sh));
  } else {
    return v;
  }
}


// The call's pointers from LDS (s_kp) at each use.
#define FKP(T, V, I) fk_ptr<true>((T *)(V), s_kp, I)

}


namespace hk_moe {

// G_group::load, or (DMA) dma_load: the backward's loads, which keep m0 compiler-managed.
template <bool DMA, typename ST, typename GL>
__device__ __forceinline__ void g1_load(ST &dst, const GL &src, const kittens::coord<ST> &idx, const uint32_t *sw,
                                        i32x4 srd, const void *base, uint32_t lds) {
  if constexpr (DMA) {
    dma_load(dst, src, idx, sw, srd, base, lds);
  } else {
    G_group::load(dst, src, idx, sw, srd, base, lds);
  }
}

// The training forward's Xp from GEMM1's A stages: this lane's two 16-byte parts of the tile's rows at k-step 0, first
// 128 rows at a0 and a1, second at b0 and b1, for the k-steps [k0, k0 + kn) (both even); xl: the parts in LDS.
struct G1Xp {
  char *a0, *a1, *b0, *b1;
  uint32_t xl;
  int k0, kn;
};

// If w: this lane's parts of the A stage OFF bytes from As[0][0] (lane-linear, as its own DMA wrote them) to p0 + o and
// p1 + o. The branch is inside the asm, so the K loop stays one basic block for the register allocator.
template <int OFF>
__device__ __forceinline__ void g1_xp_store(uint32_t xl, const char *p0, const char *p1, int w, uint64_t o) {
  typedef int v4i __attribute__((ext_vector_type(4)));
  v4i d0, d1;
  uint64_t t0, t1;
  asm volatile("s_cmp_eq_u32 %6, 0\n\t"
               "s_cbranch_scc1 .Lg1xp%=\n\t"
               "ds_read_b128 %0, %4 offset:%9\n\t"
               "ds_read_b128 %1, %4 offset:%10\n\t"
               "v_lshl_add_u64 %2, %5, 0, %7\n\t"
               "v_lshl_add_u64 %3, %8, 0, %7\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               "global_store_dwordx4 %2, %0, off nt\n\t"
               "global_store_dwordx4 %3, %1, off nt\n\t"
               "s_waitcnt expcnt(0)\n\t"
               "s_nop 7\n"
               ".Lg1xp%=:"
               : "=&v"(d0), "=&v"(d1), "=&v"(t0), "=&v"(t1)
               : "v"(xl), "v"(p0), "s"(w), "s"(o), "v"(p1), "i"(OFF), "i"(OFF + 16 * NUM_THREADS)
               : "memory", "scc");
}

// GEMM1's K loop for one 256 x 256 tile: the C^T fragments c00..c11 of A's rows times B's 128-row halves bc0 and bc1,
// over k_tiles steps of 64 through the LDS stages As and Bs (DMA: g1_load; BOFF: Bs that many bytes after As). XP: also
// A's k-steps of xp to Xp.
template <bool DMA, uint32_t BOFF, bool XP = false, typename ST, typename RTC>
__device__ __forceinline__ void g1_kloop(ST (&As)[2][2], ST (&Bs)[2][2], const FsGl &A, const FsGl &B_local,
                                         i32x4 a_srd, i32x4 b_srd, const bf16 *a_base, const bf16 *b_base,
                                         const uint32_t *sw_A, const uint32_t *sw_A1, const uint32_t *sw_B,
                                         uint32_t a_lds_00, uint32_t a_lds_01, uint32_t a_lds_10, uint32_t a_lds_11,
                                         uint32_t b_lds_00, uint32_t b_lds_01, uint32_t b_lds_10, uint32_t b_lds_11,
                                         int bc0, int bc1, int k_tiles, int lane, int warp_m, int warp_n, RTC &c00,
                                         RTC &c01, RTC &c10, RTC &c11, const G1Xp &xp = {}) {
  using RT_A = rt_bf<REG_M, K_STEP_64, row_l, rt_16x32_s>;
  using RT_B = rt_bf<REG_N, K_STEP_64, row_l, rt_16x32_s>;
  constexpr int SZA = sizeof(ST);
  constexpr int SZB = sizeof(ST);
  if constexpr (BOFF != 0) {
    // B's stages at a fixed offset from A's: derived here, not held in four more registers.
    b_lds_00 = a_lds_00 + BOFF, b_lds_01 = a_lds_01 + BOFF, b_lds_10 = a_lds_10 + BOFF, b_lds_11 = a_lds_11 + BOFF;
  }
  const auto sa0 = subtile_inplace<REG_M, K_STEP_64>(As[0][0], {warp_m, 0});
  const auto sb0 = subtile_inplace<REG_N, K_STEP_64>(Bs[0][0], {warp_n, 0});
  RT_A a_tile;
  RT_B b_tile_0, b_tile_1;
  zero(c00);
  zero(c01);
  zero(c10);
  zero(c11);

  int tic = 0, toc = 1;

  dma_load(Bs[tic][0], B_local, {0, 0, bc0, 0}, sw_B, b_srd, b_base, b_lds_00);
  dma_load(As[tic][0], A, {0, 0, 0, 0}, sw_A, a_srd, a_base, a_lds_00);
  dma_load(Bs[tic][1], B_local, {0, 0, bc1, 0}, sw_B, b_srd, b_base, b_lds_01);
  dma_load(As[tic][1], A, {0, 0, 0, 0}, sw_A1, a_srd, a_base, a_lds_01);

  if (warp_m == 1) {
    __builtin_amdgcn_s_barrier();
  }
  asm volatile("s_waitcnt vmcnt(4)");
  __builtin_amdgcn_s_barrier();

  dma_load(Bs[toc][0], B_local, {0, 0, bc0, 1}, sw_B, b_srd, b_base, b_lds_10);
  dma_load(As[toc][0], A, {0, 0, 0, 1}, sw_A, a_srd, a_base, a_lds_10);
  dma_load(Bs[toc][1], B_local, {0, 0, bc1, 1}, sw_B, b_srd, b_base, b_lds_11);

  asm volatile("s_waitcnt vmcnt(4)");
  __builtin_amdgcn_s_barrier();
  fs_load_frag<0>(b_tile_0, sb0, lane);

#pragma unroll 1
  for (int tile = 0; tile < k_tiles - 2; tile += 2) {
    // XP: k-step tile's stages after A[1][1]'s DMA (P4 waits for 4 more), tile + 1's after B[1][0]'s (P7, P8: 4 more);
    // the waits that retire them come 2-3 phases later than an earlier issue would give.
    const int xw = XP ? __builtin_amdgcn_readfirstlane((unsigned)(tile - xp.k0) < (unsigned)xp.kn) : 0;
    fs_load_frag<0>(a_tile, sa0, lane);
    g1_load<DMA>(As[1][1], A, {0, 0, 0, tile + 1}, sw_A1, a_srd, a_base, a_lds_11);
    if constexpr (XP) {
      const uint64_t xo = (uint32_t)(tile * K_STEP_64 * 2);
      g1_xp_store<0>(xp.xl, xp.a0, xp.a1, xw, xo);
      g1_xp_store<SZA>(xp.xl, xp.b0, xp.b1, xw, xo);
    }
    asm volatile("s_waitcnt lgkmcnt(8)");
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c00, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    fs_load_frag<SZB>(b_tile_1, sb0, lane);
    g1_load<DMA>(Bs[0][0], B_local, {0, 0, bc0, tile + 2}, sw_B, b_srd, b_base, b_lds_00);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c01, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<SZA>(a_tile, sa0, lane);
    g1_load<DMA>(As[0][0], A, {0, 0, 0, tile + 2}, sw_A, a_srd, a_base, a_lds_00);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c10, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    fs_load_frag<2 * SZB>(b_tile_0, sb0, lane);
    g1_load<DMA>(Bs[0][1], B_local, {0, 0, bc1, tile + 2}, sw_B, b_srd, b_base, b_lds_01);
    if constexpr (XP) {
      asm volatile("s_cmp_eq_u32 %0, 0\n\t"
                   "s_cbranch_scc1 .Lg1v6%=\n\t"
                   "s_waitcnt vmcnt(10)\n\t"
                   "s_branch .Lg1v6e%=\n"
                   ".Lg1v6%=:\n\t"
                   "s_waitcnt vmcnt(6)\n"
                   ".Lg1v6e%=:" ::"s"(xw)
                   : "memory", "scc");
    } else {
      asm volatile("s_waitcnt vmcnt(6)");
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c11, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<2 * SZA>(a_tile, sa0, lane);
    g1_load<DMA>(As[0][1], A, {0, 0, 0, tile + 2}, sw_A1, a_srd, a_base, a_lds_01);
    asm volatile("s_waitcnt lgkmcnt(8)");
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c00, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    fs_load_frag<3 * SZB>(b_tile_1, sb0, lane);
    g1_load<DMA>(Bs[1][0], B_local, {0, 0, bc0, tile + 3}, sw_B, b_srd, b_base, b_lds_10);
    if constexpr (XP) {
      const uint64_t xo = (uint32_t)((tile + 1) * K_STEP_64 * 2);
      g1_xp_store<2 * SZA>(xp.xl, xp.a0, xp.a1, xw, xo);
      g1_xp_store<3 * SZA>(xp.xl, xp.b0, xp.b1, xw, xo);
    }
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c01, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<3 * SZA>(a_tile, sa0, lane);
    g1_load<DMA>(As[1][0], A, {0, 0, 0, tile + 3}, sw_A, a_srd, a_base, a_lds_10);
    if constexpr (XP) {
      asm volatile("s_cmp_eq_u32 %0, 0\n\t"
                   "s_cbranch_scc1 .Lg1v10%=\n\t"
                   "s_waitcnt vmcnt(14)\n\t"
                   "s_branch .Lg1ve%=\n"
                   ".Lg1v10%=:\n\t"
                   "s_waitcnt vmcnt(10)\n"
                   ".Lg1ve%=:" ::"s"(xw)
                   : "memory", "scc");
    } else {
      asm volatile("s_waitcnt vmcnt(10)" ::: "memory");  // P3's Bs[0][0] DMA, read in P15
    }
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c10, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    fs_load_frag<0>(b_tile_0, sb0, lane);
    g1_load<DMA>(Bs[1][1], B_local, {0, 0, bc1, tile + 3}, sw_B, b_srd, b_base, b_lds_11);
    if constexpr (XP) {
      asm volatile("s_cmp_eq_u32 %0, 0\n\t"
                   "s_cbranch_scc1 .Lg1v4%=\n\t"
                   "s_waitcnt vmcnt(8)\n\t"
                   "s_branch .Lg1v4e%=\n"
                   ".Lg1v4%=:\n\t"
                   "s_waitcnt vmcnt(4)\n"
                   ".Lg1v4e%=:" ::"s"(xw)
                   : "memory", "scc");
    } else {
      asm volatile("s_waitcnt vmcnt(4)");
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c11, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
  }

  {
    int tile = k_tiles - 2;
    fs_load_frag<0>(b_tile_0, sb0, lane);
    fs_load_frag<0>(a_tile, sa0, lane);
    g1_load<DMA>(As[toc][1], A, {0, 0, 0, tile + 1}, sw_A1, a_srd, a_base, a_lds_11);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c00, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<SZB>(b_tile_1, sb0, lane);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c01, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<SZA>(a_tile, sa0, lane);
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c10, a_tile, b_tile_0);
    mma_ABt_T(c11, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    tic ^= 1;
    toc ^= 1;
  }

  {
    fs_load_frag<2 * SZB>(b_tile_0, sb0, lane);
    fs_load_frag<2 * SZA>(a_tile, sa0, lane);
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c00, a_tile, b_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<3 * SZB>(b_tile_1, sb0, lane);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c01, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    fs_load_frag<3 * SZA>(a_tile, sa0, lane);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt_T(c10, a_tile, b_tile_0);
    mma_ABt_T(c11, a_tile, b_tile_1);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
  }
  if constexpr (XP) {
    const int tile = k_tiles - 2, xw = __builtin_amdgcn_readfirstlane((unsigned)(tile - xp.k0) < (unsigned)xp.kn);
    const uint64_t xo = (uint32_t)(tile * K_STEP_64 * 2);
    g1_xp_store<0>(xp.xl, xp.a0, xp.a1, xw, xo);
    g1_xp_store<SZA>(xp.xl, xp.b0, xp.b1, xw, xo);
    g1_xp_store<2 * SZA>(xp.xl, xp.a0, xp.a1, xw, xo + K_STEP_64 * 2);
    g1_xp_store<3 * SZA>(xp.xl, xp.b0, xp.b1, xw, xo + K_STEP_64 * 2);
  }
}

// bf16 over capacity, after round r: the last workgroup resets its counters, takes back its arrival signals (one per
// slice and round), stores RFREE in every peer (its next round may write here) and releases the others.
__device__ __forceinline__ void fs_round_end(const MoeFwdArgs *args, int *tile_counter, int r, int nrb,
                                             unsigned epoch, int nwg) {
  if (moe_last_wg_n(tile_counter + NUM_XCDS_AFF, nwg)) {
    FsArgsC *rp = fs_args(args);
    const int sl = rp->dncomm / MOE_EP, E = rp->E_loc;
    fs_reset_words<true>(fs_g(tile_counter), NUM_XCDS_AFF + 1);
    fs_reset_words<true>(fs_g(rp->g2_ctr), 1);
    fs_reset_words<true>(fs_g(rp->disp_done), 1);
    fs_reset_words<true>(fs_g(rp->idx_done), 1);
    fs_reset_words<true>(fs_g(rp->tile_done), nrb);
    fs_reset_words<true>(fs_g(rp->a_done), nrb);
    fs_reset_words<true>(fs_g(rp->fold_ctr) + 1, 1 + rp->nrbs);
    for (int i = fs_tid(); i < MOE_EP * E; i += NUM_THREADS) {
      gint *w = fs_g(rp->my_sig) + sig_index(i / E, i % E);
      int n = 0;
      while ((int)((unsigned)__hip_atomic_load(w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - (epoch + 1u) * sl) <
             0) {
        spin_backoff(n);
      }
      (void)__hip_atomic_fetch_add(w, -sl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (fs_tid() < MOE_EP) {
      __hip_atomic_store((gint *)rp->dpeers.sig[fs_tid()] + RFREE_OFF + rp->my_rank,
                         (int)(epoch << 16 | (unsigned)(r + 1)), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (fs_tid() == 0) {
      __hip_atomic_store(fs_g(rp->rgo), r + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else if (threadIdx.x == 0) {
    plan_wait(fs_g(fs_args(args)->rgo), r + 1, false);
  }
  __syncthreads();
}

// The training API's forward (moe_train_api.cu). KS: the small-k kernel (k in {1, 2, 4}, moe_fold_item). ACT: GEMM1's
// activation. The fold starts per item once the item's rows have landed, overlapping the combine's tail.
template <int BURST, bool KS = false, typename ACT = SwiGLU>
__global__ __launch_bounds__(NUM_THREADS, 2)
void moe_fwd_single(const FsGl A0, const bf16 *__restrict__ X, const bf16 *__restrict__ W1, const bf16 *__restrict__ W2,
                    bf16 *__restrict__ fold_out, bf16 *__restrict__ G, const int *__restrict__ topk_idx,
                    const float *__restrict__ topk_w, int *__restrict__ tile_counter,
                    const MoeFwdArgs *__restrict__ args, const TrainApiCall call) {
  FsArgsC *ka = (FsArgsC *)args;
  const int wg     = (int)blockIdx.x;
  const int tid    = (int)threadIdx.x;
  const int epoch  = __builtin_amdgcn_readfirstlane(fs_g(ka->ep)[0]);
  const int xepoch = __builtin_amdgcn_readfirstlane(fs_g(ka->ep)[1]);
  using ST_A = st_bf<HALF_ROW, K_STEP_64, st_16x32_s>;
  using ST_B = st_bf<HALF_COL, K_STEP_64, st_16x32_s>;
  using RT_A = rt_bf<REG_M, K_STEP_64, row_l, rt_16x32_s>;
  using RT_B = rt_bf<REG_N, K_STEP_64, row_l, rt_16x32_s>;
  __shared__ ST_A As[2][2];
  __shared__ ST_B Bs[2][2];
  __shared__ int s_nt[2];
  __shared__ int s_mpad;
  __shared__ int s_cur, s_gr, s_ar;
  __shared__ unsigned long long s_kp[FK_N];
  if (tid == 0) {
    s_kp[FK_X] = (unsigned long long)X, s_kp[FK_W1] = (unsigned long long)W1, s_kp[FK_W2] = (unsigned long long)W2;
    s_kp[FK_OUT] = (unsigned long long)fold_out, s_kp[FK_G] = (unsigned long long)G;
    s_kp[FK_IDX] = (unsigned long long)topk_idx, s_kp[FK_W] = (unsigned long long)topk_w;
    s_kp[FK_EP] = (unsigned long long)(unsigned)epoch | (unsigned long long)(unsigned)xepoch << 32;
  }
  __syncthreads();
  if (tid == 0) {
    train_ptrs()[TRAIN_PTR_XP] = (unsigned long long)call.xp, train_ptrs()[TRAIN_PTR_G] = (unsigned long long)call.g;
    train_ptrs()[TRAIN_PTR_A] = (unsigned long long)call.a, train_ptrs()[TRAIN_PTR_Y] = (unsigned long long)call.y;
    train_ptrs()[TRAIN_PTR_ROUTE] = (unsigned long long)call.route,
        train_ptrs()[TRAIN_PTR_CTR] = (unsigned long long)call.ctr;
    train_ptrs()[TRAIN_PTR_SEG] = (unsigned long long)call.seg;
    train_ints()[TRAIN_INT_NSEG] = call.nseg, train_ints()[TRAIN_INT_TIER] = call.tier,
        train_ints()[TRAIN_INT_ROWS] = call.rows;
    train_ints()[TRAIN_INT_OFFSLOT] = call.off_slot;
  }
  __syncthreads();
  static_assert(PLAN_LAYOUT_OFF + sizeof(MoeLayout) <= sizeof(As), "the start phase's scratch fits in As");
  // Over capacity the call runs in rounds (moe_plan_layout), each from its start phase to the drained tile loop, with
  // the fold in the last; s_rnd is the round, s_nr the call's rounds.
  __shared__ int s_rnd, s_nr, s_ngr, s_fz;
  if (tid == 0) {
    s_rnd = 0, s_ngr = (int)gridDim.x;
    s_kp[FK_X] = (unsigned long long)X;
  }
  __syncthreads();
  {
    // Start phase in As: plan (index workgroups), layout (all), tables and index phase (index workgroups), dispatch.
    char *scr = reinterpret_cast<char *>(&As[0][0]);
    MoeLayout *L = reinterpret_cast<MoeLayout *>(scr + PLAN_LAYOUT_OFF);
    const int dn = ka->dncomm, d0 = (int)gridDim.x > dn ? dn : 0, nw = (int)gridDim.x - d0;
    if (wg < dn) {
      fs_xfree_store(ka, wg, fk_epoch<true>(xepoch, s_kp, 32));
    }
    if (wg >= d0) {
      moe_plan_local(ka, FKP(const int, topk_idx, FK_IDX), FKP(const float, topk_w, FK_W), wg - d0, nw,
                     (unsigned)fk_epoch<true>(epoch, s_kp, 0), scr, nullptr);
    }
  }
  MoeLayout *const L0 = reinterpret_cast<MoeLayout *>(reinterpret_cast<char *>(&As[0][0]) + PLAN_LAYOUT_OFF);
  // A storing forward whose full-capacity Xp has no room in the pool runs in rounds (moe_plan_layout's fz); the
  // training API has no pool (fz 0).
  if (tid == 0) {
    const gint *pw = ka->pool ? fs_g(ka->pool) + POOL_FIT : nullptr;
    const int f0 = pw ? pw[0] : 1, f1 = pw ? pw[1] : 1, fs = pw ? pw[2] : 0;
    const bool fz = pw && FKP(bf16, G, FK_G) != nullptr && !(fs == ka->pslot ? f1 : f0);
    s_fz = fz;
    if (wg == 0 && ka->pool) {
      plan_set_forced(ka, fz);
    }
  }
  __syncthreads();
  moe_plan_layout<true, true, true>(ka, L0, (unsigned)fk_epoch<true>(epoch, s_kp, 0), -1, *fs_lds(&s_fz));
  if (__builtin_amdgcn_readfirstlane(L0->nr) == 1) {
    // The first pass's layout decides: a call in one pass runs the round-free code.
    constexpr bool RND = false;
#define FWD_RMID fwd_rmid_0
#include "moe_fwd_pass.inc"
#undef FWD_RMID
  } else {
    constexpr bool RND = true;
#define FWD_RMID fwd_rmid_1
#include "moe_fwd_pass.inc"
#undef FWD_RMID
  }
}

#undef FKP

// The push depth whose passes cover a bf16 row exactly; 0 if none: the forward's pushes then take depth 8 with a
// masked last pass, the backward's depth 8 with a lane-wise tail.
constexpr int moe_prod_burst(int H) {
  const int NB = (H / 8) / 64;
  for (int d : {8, 7, 6, 4}) {
    if (NB % d == 0) {
      return d;
    }
  }
  return 0;
}

static_assert(moe_prod_burst(7168) == 7, "dsv3/dsv4pro/kimi27 burst depth");
static_assert(moe_prod_burst(6144) == 6, "glm52 burst depth");
static_assert(moe_prod_burst(4096) == 8, "qwen235/qwen35 burst depth");
static_assert(moe_prod_burst(2048) == 4, "qwen30 burst depth");
static_assert(moe_prod_burst(5120) == 0, "llama4 burst depth: masked");

// GEMM1 work per dispatched row grows with I, so a wider I needs fewer pushers.
constexpr int moe_prod_ncomm(int I) {
  return I >= 3072 ? 16 : (I >= 1024 ? 24 : 32);
}

static_assert(moe_prod_ncomm(768) == 32, "qwen30 ncomm");
static_assert(moe_prod_ncomm(1024) == 24, "qwen35 ncomm");
static_assert(moe_prod_ncomm(2048) == 24, "dsv3 ncomm");
static_assert(moe_prod_ncomm(3072) == 16, "dsv4pro ncomm");
static_assert(moe_prod_ncomm(768) % NUM_XCDS_AFF == 0 && moe_prod_ncomm(2048) % NUM_XCDS_AFF == 0 &&
                  moe_prod_ncomm(3072) % NUM_XCDS_AFF == 0,
              "ncomm must be a multiple of the XCD count");

}
