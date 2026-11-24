/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MXFP8 training quantization: bf16 -> MXFP8 items, the column-wise epilogue of A, lane reductions and scale fetches.
#pragma once

namespace hk_moe {

template <int R>
using MxRtK = rt<fp8e4m3, R, K_STEP_128, col_l, rt_16x128_s>;
using MxStK = st_fp8e4m3<K_STEP_128, HALF_COL, st_16x128_s>;
template <bool COL>
using MxStT = std::conditional_t<COL, MxStK, MxSt>;
template <bool COL, int R>
using MxRtT = std::conditional_t<COL, MxRtK<R>, rt_fp8e4m3<R, K_STEP_128>>;
using MxRtAK = MxRtK<REG_M>;
using MxRtBK = MxRtK<REG_N>;
using MxRtC = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
using MxRtCT = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;

// kittens' fp8 row (or COL) layout read addresses, once per tile from an opaque lane. The column layout's second read
// XORs the absolute address, so t0 is 2048-aligned.
template <bool COL, typename RT>
struct MxLanes {
  uint32_t v[COL ? 2 * RT::height : 2];
};

template <bool COL, typename ST, typename RT>
__device__ __forceinline__ MxLanes<COL, RT> mx_lanes(uint32_t t0, int off) {
  MxLanes<COL, RT> l;
  int lane = fs_tid() % kittens::WARP_THREADS;
  asm volatile("" : "+v"(lane));
  if constexpr (COL) {
    const int block = lane / 16, lw = lane % 16, kg = lw / 2, mh = lw & 1;
    // Odd subtiles (gather_parts<MxStK, true>) XOR column bit 6: a transposed read's lanes 0-15 and 16-31 read adjacent
    // subtiles, which then fall in disjoint LDS banks.
    const int kx = (lane & 16) << 2;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
      const uint32_t o = (uint32_t)(block * ST::underlying_subtile_bytes) +
                         ST::swizzle({kg, (off + i * RT::base_tile_rows + mh * 8) ^ kx});
      l.v[i] = t0 + o;
      l.v[RT::height + i] = t0 + (o ^ 1088u);
    }
  } else {
    const uint32_t base = t0 + (uint32_t)(off / ST::underlying_subtile_rows * ST::underlying_subtile_bytes);
#pragma unroll
    for (int k = 0; k < 2; k++) {
      l.v[k] = base + ST::swizzle({lane % 16, 16 * (lane / 16) + 64 * k});
    }
  }
  return l;
}

template <int OFF, typename ST, bool COL, typename RT>
__device__ __forceinline__ void mx_load(RT &dst, const MxLanes<COL, RT> &l) {
  constexpr int sb = ST::underlying_subtile_bytes;
  if constexpr (COL) {
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
      for (int x = 0; x < 2; x++) {
        asm volatile("ds_read_b64_tr_b8 %0, %2 offset:%3\n"
                     "ds_read_b64_tr_b8 %1, %2 offset:%4\n"
                     : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[2 * x])),
                       "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[2 * x + 4]))
                     : "v"(l.v[x * RT::height + i]), "i"(OFF), "i"(OFF + 4 * sb)
                     : "memory");
      }
    }
  } else {
    static_assert(RT::width == 1 && RT::base_tile_stride == 16 && RT::base_tile_num_strides == 2, "fp8 16x128 row_l");
#pragma unroll
    for (int k = 0; k < 2; k++) {
#pragma unroll
      for (int i = 0; i < RT::height; i++) {
        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                     : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[i][0].data[4 * k]))
                     : "v"(l.v[k]), "i"(OFF + i * sb)
                     : "memory");
      }
    }
  }
}

// Addresses formed at the use: nothing carried out of the K-loop into its tail.
template <bool COL, int OFF, typename ST, typename RT>
__device__ __forceinline__ void mx_load_now(RT &dst, uint32_t t0, int off) {
  mx_load<OFF, ST>(dst, mx_lanes<COL, ST, RT>(t0, off));
}

struct MxWarp {
  int ft, wm, wn, wv, lane;
};

__device__ __forceinline__ MxWarp mx_warp() {
  const int ft = fs_tid();
  return {ft, __builtin_amdgcn_readfirstlane(ft / (kittens::WARP_THREADS * WARPS_COL)),
          __builtin_amdgcn_readfirstlane(ft / kittens::WARP_THREADS % WARPS_COL),
          __builtin_amdgcn_readfirstlane(ft / kittens::WARP_THREADS), ft % kittens::WARP_THREADS};
}

__device__ __forceinline__ uint32_t mx_dma_base(const void *tile, int wv) {
  return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile))) +
         (uint32_t)wv * 16 * kittens::WARP_THREADS;
}


// As mx_scale_step with both descriptors formed here: only the bases are carried across the K-loop.
__device__ __forceinline__ void mx_scale_step(const uint8_t *a_sc, uint32_t a_range, uint32_t a_soff,
                                              const uint8_t *b_sc, uint32_t b_range, uint32_t b_soff, uint32_t voff,
                                              int buf) {
  const int w = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS);
  const uint32_t lds = mx_scale_lds_at(buf, w);
  const bool a = w < NUM_WARPS / 2;
  const uint8_t *p = fs_uniform_ptr(a ? a_sc : b_sc);
  asm volatile("" : "+s"(p));
  mx_scale_dma(fs_srd(p, a ? a_range : b_range, 0), voff, a ? a_soff : b_soff, lds);
}

typedef unsigned int mx_u4 __attribute__((ext_vector_type(4)));

// Rows past the descriptor's range read as zero.
__device__ __forceinline__ void mx_qload(mx_u4 (&w)[8], i32x4 s, uint32_t vo, int sb, int ld) {
  asm volatile("" : "+s"(ld));
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint32_t o = vo + (uint32_t)((32 * sb + i) * ld * 2);
    w[i] = __builtin_bit_cast(mx_u4, llvm_amdgcn_raw_buffer_load_b128(s, o, 0, 0));
  }
  // Keeps the scheduler from sinking the prefetch below the previous block's waits.
  __builtin_amdgcn_sched_barrier(0);
}

typedef short mx_s2 __attribute__((ext_vector_type(2)));
typedef __bf16 mx_bf2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t mx_hmax(uint32_t a, uint32_t b) {
  return __builtin_bit_cast(uint32_t, mx_vmax(__builtin_bit_cast(mx_h2, a), __builtin_bit_cast(mx_h2, b)));
}

// mx_exp + 127 of two packed |bf16| maxima: (u >> 23) - 8, plus one for a mantissa above 0x60, at least 0.
__device__ __forceinline__ uint32_t mx_sc2(uint32_t h) {
  const mx_h2 v = (__builtin_bit_cast(mx_h2, h) + (mx_h2)0x1f) >> (mx_h2)7;
  return __builtin_bit_cast(uint32_t, __builtin_elementwise_sub_sat(v, (mx_h2)8));
}

// Scale byte 0 is 2^-127, as the conversion reads it.
__device__ __forceinline__ float mx_scf(uint32_t s) {
  return __builtin_bit_cast(float, s << 23);
}

// One rounding of x / 2^e, bit-identical to mx_pk4v of the f32 products.
__device__ __forceinline__ uint32_t mx_cvt4(uint32_t x0, uint32_t x1, float f0, float f1) {
  mx_s2 r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16((mx_s2){0, 0}, __builtin_bit_cast(mx_bf2, x0), f0, false);
  r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(r, __builtin_bit_cast(mx_bf2, x1), f1, true);
  return __builtin_bit_cast(uint32_t, r);
}

// Rows pair up (2 p, 2 p + 1) lo / hi in one register; maxima compare |bf16| bit patterns as u16.
template <bool ROW, bool COL>
__device__ __forceinline__ void mx_qput(const mx_u4 (&w)[8], int sb, int K, int ldq, i32x4 rs, i32x4 cs, i32x4 ss,
                                        int c, int rg) {
  asm volatile("" : "+s"(K), "+s"(ldq));
  // Byte selectors held in VGPRs: as v_perm operands they would otherwise take SGPRs.
  uint32_t pl = 0x05040100u, ph = 0x07060302u, pe = 0x06040200u, po = 0x07050301u;
  asm volatile("" : "+v"(pl), "+v"(ph), "+v"(pe), "+v"(po));
  uint32_t a[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      a[i][q] = w[i][q] & 0x7fff7fffu;
    }
  }
  if constexpr (COL) {
    uint32_t sc[4];
    float f[8];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint32_t m = a[0][q];
#pragma unroll
      for (int i = 1; i < 8; i++) {
        m = mx_hmax(m, a[i][q]);
      }
      sc[q] = mx_sc2(__builtin_bit_cast(uint32_t, mx_max_x16_x32(__builtin_bit_cast(mx_h2, m))));
      f[2 * q] = mx_scf(sc[q] & 0xffu);
      f[2 * q + 1] = mx_scf(sc[q] >> 16);
    }
#pragma unroll
    for (int p = 0; p < 4; p++) {
      uint32_t d[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const uint32_t lo = __builtin_amdgcn_perm(w[2 * p + 1][q], w[2 * p][q], pl);
        const uint32_t hi = __builtin_amdgcn_perm(w[2 * p + 1][q], w[2 * p][q], ph);
        d[q] = mx_cvt4(lo, hi, f[2 * q], f[2 * q + 1]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint32_t sel = h ? po : pe;
        const uint64_t o = (uint64_t)__builtin_amdgcn_perm(d[1], d[0], sel) |
                           (uint64_t)__builtin_amdgcn_perm(d[3], d[2], sel) << 32;
        llvm_amdgcn_raw_buffer_store_b64(o, cs, (uint32_t)(rg * K + c), (32 * sb + 2 * p + h) * K, 0);
      }
    }
    const uint64_t sw = (uint64_t)__builtin_amdgcn_perm(sc[1], sc[0], pe) |
                        (uint64_t)__builtin_amdgcn_perm(sc[3], sc[2], pe) << 32;
    llvm_amdgcn_raw_buffer_store_b64(sw, ss, (uint32_t)c, sb * K, 0);
  }
  if constexpr (ROW) {
#pragma unroll
    for (int p = 0; p < 4; p++) {
      uint32_t m[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        m[h] = mx_hmax(mx_hmax(a[2 * p + h][0], a[2 * p + h][1]), mx_hmax(a[2 * p + h][2], a[2 * p + h][3]));
      }
      const mx_h2 x = __builtin_bit_cast(mx_h2, mx_hmax(__builtin_amdgcn_perm(m[1], m[0], pl),
                                                   __builtin_amdgcn_perm(m[1], m[0], ph)));
      const uint32_t s = mx_sc2(__builtin_bit_cast(uint32_t, mx_dpp_max<0x4E>(mx_dpp_max<0xB1>(x))));
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int i = 2 * p + h;
        const uint32_t sr = h ? s >> 16 : s & 0xffu;
        const float f = mx_scf(sr);
        const uint64_t o = (uint64_t)mx_cvt4(w[i][0], w[i][1], f, f) | (uint64_t)mx_cvt4(w[i][2], w[i][3], f, f) << 32;
        const int so = (32 * sb + i) * ldq;
        llvm_amdgcn_raw_buffer_store_b64(o, rs, (uint32_t)(rg * ldq + c), so, 0);
        llvm_amdgcn_raw_buffer_store_b8((uint8_t)sr, rs, (uint32_t)(rg * ldq + K + c / 32), so, 0);
      }
    }
  }
}

// bf16(w x) of the lane's 8 rows (probs on input), row i's weight wt[i].
__device__ __forceinline__ void mx_pin_rows(mx_u4 (&w)[8], const float (&wt)[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    w[i] = __builtin_bit_cast(mx_u4, moe_pin_scale(__builtin_bit_cast(v8bf, w[i]), wt[i]));
  }
}

// Rows r0.. (zero past nv) by MX_QCOLS columns: row-wise into rq (K bytes, then K / 32 scales per row), with COL
// column-wise into cq, cs. Lane l of wave w: 8 columns (16 w + l % 16) * 8 by rows 8 (l / 16).. of each 32 rows.
// PIN: each row r is quantized as bf16(pw[r] x) (probs on input; the weights of rows r0 + [0, 256)). !ROW: column-wise
// only.
template <bool COL, bool PIN = false, bool ROW = true>
__device__ __forceinline__ void mx_quant_block(const bf16 *src, int ld, int r0, int nv, int c0, int K, uint8_t *rq,
                                               int ldq, uint8_t *cq, uint8_t *cs, const float *pw = nullptr) {
  const int t = fs_tid(), l = t % kittens::WARP_THREADS;
  const int c = c0 + (t / kittens::WARP_THREADS * 16 + l % 16) * 8, rg = 8 * (l / 16);
  const i32x4 ls = fs_srd(src + (size_t)r0 * ld, (uint32_t)(nv * ld * 2), 0);
  const i32x4 rs = fs_srd(rq + (size_t)r0 * ldq, 0xffffffffu, 0);
  const i32x4 qs = fs_srd(cq + (size_t)r0 * K, 0xffffffffu, 0);
  const i32x4 ss = fs_srd(cs + (size_t)r0 / 32 * K, 0xffffffffu, 0);
  const uint32_t vo = (uint32_t)((rg * ld + c) * 2);
  mx_u4 a[8], b[8];
  float wt[8];
  // The next 32-row block loads during this one's stores; the last prefetch is past the range and reads nothing. A
  // block's weights load before the next block's prefetch, so waiting for them does not wait for it.
  mx_qload(a, ls, vo, 0, ld);
#pragma unroll 1
  for (int sb = 0; sb < BLOCK_ROW / 32; sb += 2) {
    if constexpr (PIN) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        wt[i] = fs_g(pw)[r0 + 32 * sb + rg + i];
      }
    }
    mx_qload(b, ls, vo, sb + 1, ld);
    if constexpr (PIN) {
      mx_pin_rows(a, wt);
    }
    mx_qput<ROW, COL>(a, sb, K, ldq, rs, qs, ss, c, rg);
    if constexpr (PIN) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        wt[i] = fs_g(pw)[r0 + 32 * sb + 32 + rg + i];
      }
    }
    mx_qload(a, ls, vo, sb + 2, ld);
    if constexpr (PIN) {
      mx_pin_rows(b, wt);
    }
    mx_qput<ROW, COL>(b, sb + 1, K, ldq, rs, qs, ss, c, rg);
  }
}

// Column-wise MXFP8 of the 32 bf16 rows src + xr[t] * H (xr[t] < 0: a zero row) into q, s at arena row r0. PIN: row t
// is quantized as bf16(w x), w the float bits xr[32 + t].
template <bool PIN = false>
__device__ __forceinline__ void mx_xcol_rows(const bf16 *src, const int *xr, int H, uint8_t *q, uint8_t *s, int r0) {
  const int t = fs_tid(), l = t % kittens::WARP_THREADS;
  const int cw = (t / kittens::WARP_THREADS * 16 + l % 16) * 8, rg = 8 * (l / 16);
  const i32x4 xs = fs_srd(src, 0x7fffffffu, 0);
  const i32x4 qs = fs_srd(q + (size_t)r0 * H, 0xffffffffu, 0);
  const i32x4 ss = fs_srd(s + (size_t)r0 / 32 * H, 0xffffffffu, 0);
  float wt[8];
  if constexpr (PIN) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      wt[i] = __builtin_bit_cast(float, xr[32 + rg + i]);
    }
  }
#pragma unroll 1
  for (int c = cw; c < H; c += MX_QCOLS) {
    mx_u4 w[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      // A zero row's offset is past the descriptor's range, so it reads as zero.
      const int sr = xr[rg + i];
      const uint32_t o = sr >= 0 ? (uint32_t)(sr * H + c) * 2 : 0x80000000u;
      w[i] = __builtin_bit_cast(mx_u4, llvm_amdgcn_raw_buffer_load_b128(xs, o, 0, 0));
    }
    if constexpr (PIN) {
      mx_pin_rows(w, wt);
    }
    mx_qput<false, true>(w, 0, H, 0, xs, qs, ss, c, rg);
  }
}

// A segment [a, a + c) of an expert's rows splits into whole 32-row groups [ha, fb), owned by its source, and boundary
// rows: head [a, he) (the group of a) and tail [tb, a + c) (the group of a + c - 1), quantized by the receiver.
struct MxSeg {
  int a, b, ha, fb, he, tb;
  __device__ __forceinline__ MxSeg(int a0, int c) : a(a0), b(a0 + c), ha(plan_pad32(a0)), fb((a0 + c) & ~31) {
    he = min(ha, b);
    tb = max(fb, he);
  }
  __device__ __forceinline__ int owned() const {
    return max(fb - ha, 0) / 32;
  }
  __device__ __forceinline__ int nbnd() const {
    return he - a + b - tb;
  }
};

// Shadow row of segment row r (r % 32 within the source's head or tail slot) of source x, expert e.
__device__ __forceinline__ int mx_xsh_row(const MxSeg &g, int e, int x, int r) {
  return ((e * MOE_EP + x) * 2 + (r < g.he ? 0 : 1)) * 32 + r % 32;
}

// The words a receiver's boundary-row arrivals add to (one per expert), then the shadow rows.
__device__ __forceinline__ int *mx_xsh_words(uint8_t *xsh) {
  return (int *)xsh;
}
__device__ __forceinline__ bf16 *mx_xsh_rows(uint8_t *xsh) {
  return (bf16 *)(xsh + 4096);
}

// Unit rows [r0, r1) (arena rows; bf16 X tokens send_slot[off + r - a] / k) into the shadow sh of source x, expert e;
// PIN: as bf16(w x), w the slot's weight pw[slot].
template <bool PIN = false>
__device__ __forceinline__ void mx_xcol_shadow(const bf16 *X, const int *send_slot, int k, int H, int off,
                                               const MxSeg &g, int r0, int r1, int e, int x, bf16 *sh,
                                               const float *pw = nullptr) {
  typedef __attribute__((address_space(1))) const mx_u4 gcv4i;
  typedef __attribute__((address_space(1))) mx_u4 gv4i;
#pragma unroll 1
  for (int r = r0; r < r1; r++) {
    const int tok = fs_g(send_slot)[off + r - g.a] / k;
    const gcv4i *src = (const gcv4i *)(fs_g(X) + (size_t)tok * H);
    gv4i *dst = (gv4i *)(fs_g(sh) + (size_t)mx_xsh_row(g, e, x, r) * H);
    if constexpr (PIN) {
      const float wt = fs_g(pw)[fs_g(send_slot)[off + r - g.a]];
      for (int c = fs_tid(); c < H / 8; c += NUM_THREADS) {
        dst[c] = __builtin_bit_cast(mx_u4, moe_pin_scale(__builtin_bit_cast(v8bf, src[c]), wt));
      }
      continue;
    }
    for (int c = fs_tid(); c < H / 8; c += NUM_THREADS) {
      dst[c] = src[c];
    }
  }
}

// Unit j's send offset, count and destination arena row: from the layout (forward) or the unit table (backward).
struct MxUnitsL {
  const MoeLayout *L;
  __device__ __forceinline__ int3 operator()(int j) const {
    return {L->sof[j], L->snd[j], L->dbase[j]};
  }
};

struct MxUnitsT {
  const __attribute__((address_space(1))) int *blks;
  __device__ __forceinline__ int3 operator()(int j) const {
    return {blks[5 * j + 2], blks[5 * j + 3], blks[5 * j + 4]};
  }
};

// Dispatch workgroup w (of ncomm) writes its slice of each unit's column-wise Xp groups to destination w % 8, sliced as
// its rows; returns the number of groups written. PAD: the unit's segment is padded to 32 rows (zero rows), so it owns
// all its groups; else it owns its whole groups, sliced as their concatenation over the units, counted per unit on
// the destination's word xw[MOE_EP * expert], and the unit whose groups start in its slice pushes its boundary rows
// (bf16) into the destination's shadow, counted on the expert's word. PIN: the rows are bf16(w x), w the slot's weight
// pw[slot].
template <bool PAD, bool PIN = false, typename U>
__device__ __forceinline__ int mx_xcol_units(const U &unit, int nblks, int ncomm, int w, const bf16 *X,
                                             const int *send_slot, int k, int H, int me, uint8_t *q, uint8_t *s,
                                             uint8_t *xsh, int *xw = nullptr, const float *pw = nullptr) {
  __shared__ int s_xr[PIN ? 64 : 32];
  const int nsp = ncomm / MOE_EP, gslc = w / MOE_EP, t = fs_tid();
  int n = 0, tot = 0, base = 0;
  if constexpr (!PAD) {
    for (int j = w % MOE_EP; j < nblks; j += MOE_EP) {
      const int3 u = unit(j);
      tot += MxSeg(u.z, u.y).owned();
    }
  }
  const int g0 = tot * gslc / nsp, g1 = tot * (gslc + 1) / nsp;
  for (int j = w % MOE_EP; j < nblks; j += MOE_EP) {
    const int3 u = unit(j);
    const MxSeg g(u.z, u.y);
    const int ng = PAD ? plan_pad32(u.y) / 32 : g.owned(), i0 = PAD ? 0 : g.ha - g.a;
    const int lo = PAD ? ng * gslc / nsp : min(max(g0 - base, 0), ng);
    const int hi = PAD ? ng * (gslc + 1) / nsp : max(min(g1 - base, ng), lo);
    for (int i = lo; i < hi; i++, n++) {
      __syncthreads();
      if (t < 32) {
        s_xr[t] = i0 + 32 * i + t < u.y ? fs_g(send_slot)[u.x + i0 + 32 * i + t] / k : -1;
        if constexpr (PIN) {
          const int r = i0 + 32 * i + t;
          s_xr[32 + t] = __builtin_bit_cast(int, r < u.y ? fs_g(pw)[fs_g(send_slot)[u.x + r]] : 0.0f);
        }
      }
      __syncthreads();
      mx_xcol_rows<PIN>(X, s_xr, H, q, s, u.z + i0 + 32 * i);
    }
    if constexpr (!PAD) {
      const int e = j / MOE_EP;
      const bool bnd = u.y && g.nbnd() && base >= g0 && (base < g1 || gslc == nsp - 1);
      if (bnd) {
        mx_xcol_shadow<PIN>(X, send_slot, k, H, u.x, g, g.a, g.he, e, me, mx_xsh_rows(xsh), pw);
        mx_xcol_shadow<PIN>(X, send_slot, k, H, u.x, g, g.tb, g.b, e, me, mx_xsh_rows(xsh), pw);
      }
      if (bnd || hi > lo) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        if (t == 0 && hi > lo) {
          (void)__hip_atomic_fetch_add(fs_g(xw) + MOE_EP * e, hi - lo, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        if (t == 0 && bnd) {
          (void)__hip_atomic_fetch_add(fs_g(mx_xsh_words(xsh)) + e, g.nbnd(), __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      base += ng;
    }
  }
  return n;
}

// Calls f(g) for each column-wise group g of expert e that no source owns (each once, in order); seg(e, x) gives source
// x's segment start and count within the expert.
template <typename S, typename F>
__device__ __forceinline__ void mx_xcol_bgroups(const S &seg, int e, const F &f) {
  int last = -1;
#pragma unroll 1
  for (int x = 0; x < MOE_EP; x++) {
    const int2 sc = seg(e, x);
    if (sc.y == 0) {
      continue;
    }
    const MxSeg g(sc.x, sc.y);
    if (g.he > g.a && g.a / 32 != last) {
      last = g.a / 32;
      f(last);
    }
    if (g.b > g.tb && (g.b - 1) / 32 != last) {
      last = (g.b - 1) / 32;
      f(last);
    }
  }
}

// Source x's segment (start within the expert, count) of expert e: from the layout (forward) or the tables (backward).
struct MxSegL {
  const MoeLayout *L;
  __device__ __forceinline__ int2 operator()(int e, int x) const {
    return {L->uof[e * MOE_EP + x] - L->est[e], L->rcv[e * MOE_EP + x]};
  }
};

struct MxSegT {
  const __attribute__((address_space(1))) int *st, *cnt;
  __device__ __forceinline__ int2 operator()(int e, int x) const {
    return {st[sig_index(x, e)], cnt[sig_index(x, e)]};
  }
};

struct MxSegS {
  const int *st, *cnt;
  __device__ __forceinline__ int2 operator()(int, int x) const {
    return {st[x], cnt[x]};
  }
};

struct MxList {
  int *l;
  __device__ __forceinline__ void operator()(int g) const {
    l[++l[0]] = g;
  }
};

// The column-wise groups of expert e no source owns, quantized from the shadow rows once all have landed (expert rows
// from ro; past the count, zero).
template <typename S>
__device__ __forceinline__ void mx_xcol_recv(const S &seg, int e, int ro, int H, uint8_t *xsh, uint8_t *q,
                                             uint8_t *s) {
  __shared__ int s_st[MOE_EP], s_cnt[MOE_EP], s_gl[2 * MOE_EP + 1], s_xr[32];
  const int t = fs_tid();
  const MxSegS ls{s_st, s_cnt};
  __syncthreads();
  if (t < MOE_EP) {
    const int2 sc = seg(e, t);
    s_st[t] = sc.x, s_cnt[t] = sc.y;
  }
  __syncthreads();
  if (t == 0) {
    int need = 0;
#pragma unroll 1
    for (int x = 0; x < MOE_EP; x++) {
      need += s_cnt[x] ? MxSeg(s_st[x], s_cnt[x]).nbnd() : 0;
    }
    s_gl[0] = 0;
    mx_xcol_bgroups(ls, e, MxList{s_gl});
    if (need) {
      gint *wd = fs_g(mx_xsh_words(xsh)) + e;
      int n = 0;
      while (__hip_atomic_load(wd, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
        spin_backoff(n);
      }
      (void)__hip_atomic_load(wd, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
      // Subtracting this call's rows, not storing 0, keeps an early add of the next call.
      (void)__hip_atomic_fetch_add(wd, -need, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  __syncthreads();
  const int ng = s_gl[0];
#pragma unroll 1
  for (int i = 1; i <= ng; i++) {
    const int gi = s_gl[i];
    if (t < 32) {
      const int r = 32 * gi + t;
      int v = -1;
#pragma unroll 1
      for (int x = 0; x < MOE_EP; x++) {
        if (s_cnt[x] && r >= s_st[x] && r < s_st[x] + s_cnt[x]) {
          v = mx_xsh_row(MxSeg(s_st[x], s_cnt[x]), e, x, r);
        }
      }
      s_xr[t] = v;
    }
    __syncthreads();
    mx_xcol_rows(mx_xsh_rows(xsh), s_xr, H, q, s, ro + 32 * gi);
    __syncthreads();
  }
}

// Column-wise MXFP8 of 32-row block bb of a C^T tile (lane: rows 16 mb + l % 16, columns c + 16 h + 4 (l / 16)..).
template <int AS, typename CT>
__device__ __forceinline__ void mx_ct_cols(__attribute__((address_space(AS))) uint8_t *q, int ldq,
                                           __attribute__((address_space(AS))) uint8_t *s, const CT &ct, int bb, int c,
                                           int lane) {
  static_assert(CT::height == 2 && CT::width == 4 && CT::base_tile_rows == 16, "one 32-column block, 64 rows");
  typedef __attribute__((address_space(AS))) uint32_t w32;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float f[4];
    uint32_t sw = 0;
#pragma unroll
    for (int x = 0; x < 4; x++) {
      float m = 0.0f;
#pragma unroll
      for (int mb = 2 * bb; mb < 2 * bb + 2; mb++) {
        const auto &d = ct.tiles[h][mb].data[x >> 1];
        m = fmaxf(m, fabsf((x & 1) ? d.y : d.x));
      }
      const int e = mx_exp(mx_max16(m));
      f[x] = mx_pow2(-e);
      sw |= (uint32_t)(e + 127) << (8 * x);
    }
#pragma unroll
    for (int mb = 2 * bb; mb < 2 * bb + 2; mb++) {
      const auto &d = ct.tiles[h][mb].data;
      *(w32 *)(q + (16 * mb + (lane & 15)) * ldq + c + 16 * h) =
          mx_pk4v(d[0].x * f[0], d[0].y * f[1], d[1].x * f[2], d[1].y * f[3]);
    }
    if ((lane & 15) == 0) {
      *(w32 *)(s + c + 16 * h) = sw;
    }
  }
}

template <typename CT>
__device__ __forceinline__ void store_T16_mx_col(uint8_t *cq, uint8_t *cs, const CT &ct, int m0, int n0, int ld,
                                                 int lane) {
#pragma unroll
  for (int bb = 0; bb < 2; bb++) {
    mx_ct_cols<1>(fs_g(cq) + (size_t)m0 * ld, ld, fs_g(cs) + (size_t)((m0 + 32 * bb) / 32) * ld, ct, bb,
                  n0 + 4 * (lane >> 4), lane);
  }
}

}
