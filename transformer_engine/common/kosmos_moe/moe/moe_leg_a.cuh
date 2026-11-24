/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MoE backward tile helpers: dgrad2's GLU-bwd epilogue and the wgrad tile.
#pragma once

namespace hk_moe {

struct LegAWgrad2 {
  const bf16 *act;
  int act_rows, act_cols;
  int run;
};

struct LegAArgs {
  NnArrivalGate g;
  LegAWgrad2 w;
};

__device__ uint16_t moe_buf_load_u16(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i16");
constexpr uint32_t MOE_AUX_WT = 17;

// lo, hi: glu_bounds; the masks are TE's: dgate where g <= L, dup where |u| <= L.
template <int CTRL>
__device__ __forceinline__ float glu_dpp(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}

// act: the activation itself (for the shared expert gate's gradient).
template <typename ACT>
__device__ __forceinline__ void swiglu_bwd_elem(float dav, float gv, float uv, float lo, float hi, float &dgate,
                                                float &dup, float &act) {
  if constexpr (ACT::CLAMP) {
    const float a  = fminf(gv, hi);
    const float s  = __builtin_amdgcn_rcpf(1.0f + __expf(-ACT::ALPHA * a));
    const float uc = __builtin_amdgcn_fmed3f(uv, lo, hi);
    const float uo = ACT::OFFSET == 0.0f ? uc : uc + ACT::OFFSET;
    dgate = a == gv ? dav * uo * s * (1.0f + ACT::ALPHA * a * (1.0f - s)) : 0.0f;
    dup = uc == uv ? dav * a * s : 0.0f;
    act = a * s * uo;
  } else {
    const float s    = 1.0f / (1.0f + __expf(-gv));
    const float sil  = gv * s;
    const float dsil = s * (1.0f + gv * (1.0f - s));
    dgate = dav * uv * dsil;
    dup = dav * sil;
    act = sil * uv;
  }
}

// MEG: G and dG in Megatron's [gate; up] column order, up columns up_soff bytes after their gate column (else the
// interleaved 128-column order, up 256 bytes after gate). GATE (MEG): the shared expert's sigmoid gate, sgl[0, 256) the
// row block's per-token gates in LDS: dA is scaled by them, and each row's sum of dA act over the tile's 32 columns
// goes to sgl[256 (1 + c) + row], c the column unit within the 256-column tile (the gate's gradient).
template <typename ACT, bool MEG = false, bool GATE = false, typename RT>
__device__ __forceinline__ void swiglu_bwd_c_tile(i32x4 g_srd, i32x4 dg_srd, const RT &src, int row_unit,
                                                  int col_unit, int row_stride, uint32_t lane_voff,
                                                  uint32_t up_soff = 0,
                                                  __attribute__((address_space(3))) float *sgl = nullptr,
                                                  int lane = 0) {
  static_assert(!GATE || MEG, "the gate is the shared expert's");
  static_assert(128 % RT::cols == 0, "an accumulator must not straddle a 128-column gate|up chunk");
  static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "col_l 16x16 accumulator");
  constexpr int NE = RT::height * RT::width * 4;
  const int i0 = col_unit * RT::cols;
  const uint32_t sbase   = MEG ? (uint32_t)(((row_unit * RT::rows) * row_stride + i0) * 2)
                               : (uint32_t)(((row_unit * RT::rows) * row_stride + (i0 / 128) * 256 + i0 % 128) * 2);
  const uint32_t rstride = (uint32_t)row_stride * 2;
  uint32_t gu[NE];
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const uint32_t so = sbase + (uint32_t)(i * RT::base_tile_rows + r) * rstride;
        const uint32_t vo = lane_voff + j * RT::base_tile_cols * 2;
        const uint32_t g  = moe_buf_load_u16(g_srd, vo, so, 0);
        const uint32_t u  = MEG ? moe_buf_load_u16(g_srd, vo, so + up_soff, 0) : moe_buf_load_u16(g_srd, vo + 256, so, 0);
        gu[(i * RT::width + j) * 4 + r] = (g & 0xffffu) | (u << 16);
      }
    }
  }
  // Keep every load ahead of the math; otherwise the scheduler interleaves them and waits vmcnt(0) per element.
  __builtin_amdgcn_sched_barrier(0);
  float lo, hi;
  glu_bounds<ACT>(lo, hi);
  if constexpr (GATE) {
    typedef float f4 __attribute__((ext_vector_type(4)));
    const int r0 = row_unit * RT::rows + 4 * (lane >> 4);
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
      const f4 sgv = *(const __attribute__((address_space(3))) f4 *)(sgl + r0 + i * RT::base_tile_rows);
      f4 ds = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const uint32_t so = sbase + (uint32_t)(i * RT::base_tile_rows + r) * rstride;
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
          const uint32_t vo = lane_voff + j * RT::base_tile_cols * 2;
          const uint32_t w  = gu[(i * RT::width + j) * 4 + r];
          const float dav = (r & 1) ? src.tiles[i][j].data[r >> 1].y : src.tiles[i][j].data[r >> 1].x;
          float dgate, dup, act;
          swiglu_bwd_elem<ACT>(dav * sgv[r], __builtin_bit_cast(float, w << 16),
                               __builtin_bit_cast(float, w & 0xffff0000u), lo, hi, dgate, dup, act);
          ds[r] = fmaf(dav, act, ds[r]);
          llvm_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, (__bf16)dgate), dg_srd, vo, so, 0);
          llvm_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, (__bf16)dup), dg_srd, vo, so + up_soff, 0);
        }
        // The row's 16 lanes: quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror.
        ds[r] += glu_dpp<0xB1>(ds[r]);
        ds[r] += glu_dpp<0x4E>(ds[r]);
        ds[r] += glu_dpp<0x141>(ds[r]);
        ds[r] += glu_dpp<0x140>(ds[r]);
      }
      // Every lane of the 16 holds its rows' sums; all write them (one LDS address per row group).
      *(__attribute__((address_space(3))) f4 *)(sgl + BLOCK_ROW * (1 + col_unit % (BLOCK_COL / RT::cols)) + r0 +
                                                 i * RT::base_tile_rows) = ds;
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const uint32_t so = sbase + (uint32_t)(i * RT::base_tile_rows + r) * rstride;
        const uint32_t vo = lane_voff + j * RT::base_tile_cols * 2;
        const uint32_t w  = gu[(i * RT::width + j) * 4 + r];
        const float dav = (r & 1) ? src.tiles[i][j].data[r >> 1].y : src.tiles[i][j].data[r >> 1].x;
        float dgate, dup, act;
        swiglu_bwd_elem<ACT>(dav, __builtin_bit_cast(float, w << 16), __builtin_bit_cast(float, w & 0xffff0000u), lo,
                             hi, dgate, dup, act);
        llvm_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, (__bf16)dgate), dg_srd, vo, so, 0);
        if constexpr (MEG) {
          llvm_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, (__bf16)dup), dg_srd, vo, so + up_soff, 0);
        } else {
          llvm_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, (__bf16)dup), dg_srd, vo + 256, so, 0);
        }
      }
    }
  }
}

template <typename ST>
__device__ __forceinline__ void lega_prefill_sw(int laneid, int row_stride, uint32_t *swizzled_offsets) {
  using T = typename ST::dtype;
  constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
  constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
  constexpr int memcpy_per_tile  = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * NUM_THREADS);
  static_assert(memcpy_per_tile * (bytes_per_thread * NUM_THREADS) == ST::rows * ST::cols * sizeof(T), "no leftovers");
  constexpr int num_warps = NUM_THREADS / kittens::WARP_THREADS;
  const int warpid = kittens::warpid() % num_warps;
#pragma unroll
  for (int i = 0; i < memcpy_per_tile; i++) {
    const int lane_byte_offset         = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
    const int subtile_id               = lane_byte_offset / ST::underlying_subtile_bytes;
    const int subtile_row              = subtile_id / ST::underlying_subtiles_per_row;
    const int subtile_col              = subtile_id % ST::underlying_subtiles_per_row;
    const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;
    int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
    int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t swizzled_shared_byte_offset = ST::swizzle({row, col});
    const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
    const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
    swizzled_offsets[i] = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);
  }
}

template <typename ST, int N>
struct LegaLoader {
  ST (&tiles)[2][2];
  const FsGl &gl;
  const int &kt0, &k_real, &blk;
  uint32_t (&sw)[N];
  i32x4 &srd;
  const bf16 *&base;
  uint32_t (&lds)[2][2];

  __device__ __forceinline__ void load(int p, int j, int kt) {
    dma_load(tiles[p][j], gl, {0, 0, kt0 + min(kt, k_real - 1), blk * 2 + j}, sw, srd, base, lds[p][j]);
  }

  __device__ __forceinline__ void load_h(int p, int j, int kt, int h) {
    const coord<ST> c = {0, 0, kt0 + min(kt, k_real - 1), blk * 2 + j};
    if (h) {
      dma_load_h<1>(tiles[p][j], gl, c, sw, srd, base, lds[p][j]);
    } else {
      dma_load_h<0>(tiles[p][j], gl, c, sw, srd, base, lds[p][j]);
    }
  }
};

constexpr int LEGA_LDS_TILES = 131072;
typedef __attribute__((address_space(3))) uint16_t lega_l16;

// Eight bf16 pairs of a and b added in fp32, rounded to nearest even.
__device__ __forceinline__ __attribute__((ext_vector_type(4))) int lega_add_bf16x8(
    __attribute__((ext_vector_type(4))) int a, __attribute__((ext_vector_type(4))) int b) {
  __attribute__((ext_vector_type(4))) int o;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float lo = __builtin_bit_cast(float, a[i] << 16) + __builtin_bit_cast(float, b[i] << 16);
    const float hi =
        __builtin_bit_cast(float, a[i] & (int)0xffff0000) + __builtin_bit_cast(float, b[i] & (int)0xffff0000);
    const unsigned ul = __builtin_bit_cast(unsigned, lo), uh = __builtin_bit_cast(unsigned, hi);
    o[i] = (int)(((ul + 0x7fffu + ((ul >> 16) & 1u)) >> 16) | ((uh + 0x7fffu + ((uh >> 16) & 1u)) & 0xffff0000u));
  }
  return o;
}

// CLIP: rows of the tile at or past rlim are not stored. ACC: rlim > BLOCK_ROW adds the tile to the bf16 values in
// place (rounded once more to nearest even).
template <uint32_t AUX, bool CLIP = false, bool ACC = false, typename RT>
__device__ __forceinline__ void store_c_staged(unsigned char *lds, const RT &c00, const RT &c01, const RT &c10,
                                               const RT &c11, int rf0, int rf1, int cf0, int cf1, int lane, bf16 *tb,
                                               int ld, int rlim = BLOCK_ROW) {
  const bool acc = ACC && rlim > BLOCK_ROW;
  static_assert(BLOCK_ROW * BLOCK_COL * sizeof(bf16) == LEGA_LDS_TILES, "the C tile must fill the tile region");
  typedef int dv4 __attribute__((ext_vector_type(4)));
  lega_l16 *st = (lega_l16 *)(__attribute__((address_space(3))) unsigned char *)lds;
  __syncthreads();
  store_c_tile(st, c00, rf0, cf0, BLOCK_COL, lane);
  store_c_tile(st, c01, rf0, cf1, BLOCK_COL, lane);
  store_c_tile(st, c10, rf1, cf0, BLOCK_COL, lane);
  store_c_tile(st, c11, rf1, cf1, BLOCK_COL, lane);
  __syncthreads();
  const __attribute__((address_space(3))) dv4 *sv = (const __attribute__((address_space(3))) dv4 *)st;
  __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void *)tb, (short)0, (int)0xffffffffu, 0x00020000);
  int tid_c = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_c));
#pragma unroll
  for (int k = 0; k < BLOCK_ROW * BLOCK_COL / 8 / NUM_THREADS; k++) {
    const int q = tid_c + k * NUM_THREADS, r = q >> 5, ch = q & 31;
    dv4 v = sv[q];
    if (!CLIP || r < rlim) {
      if (acc) {
        v = lega_add_bf16x8(v, __builtin_bit_cast(dv4, __builtin_amdgcn_raw_buffer_load_b128(
                                                           rs, (unsigned)((r * ld + ch * 8) * 2), 0, 0)));
      }
      __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(__attribute__((ext_vector_type(4))) unsigned int, v),
                                             rs, (unsigned)((r * ld + ch * 8) * 2), 0, AUX);
    }
  }
}

// The accumulator RT at (row_unit, col_unit) as fp32 into an LDS tile of row_stride floats.
template <typename RT>
__device__ __forceinline__ void sh_lds_f32(__attribute__((address_space(3))) float *base, const RT &src, int row_unit,
                                           int col_unit, int row_stride, int lane) {
  static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "col_l 16x16 accumulator");
  __attribute__((address_space(3))) float *p = base + (row_unit * RT::rows) * row_stride + col_unit * RT::cols;
  const int row_offset = RT::base_tile_stride * (lane / RT::base_tile_cols);
  const int col_offset = lane % RT::base_tile_cols;
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
      const int row = i * RT::base_tile_rows + row_offset, col = j * RT::base_tile_cols + col_offset;
#pragma unroll
      for (int l = 0; l < 2; l++) {
        p[(row + l * 2) * row_stride + col] = src.tiles[i][j].data[l].x;
        p[(row + l * 2 + 1) * row_stride + col] = src.tiles[i][j].data[l].y;
      }
    }
  }
}

// One 128-row half of a 256x256 fp32 C tile, staged in LDS (lds), stored to dst (row stride ld floats).
__device__ __forceinline__ void sh_half_park(const unsigned char *lds, float *dst, int ld) {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef __attribute__((address_space(1))) v4f gv4f;
  const __attribute__((address_space(3))) v4f *sv =
      (const __attribute__((address_space(3))) v4f *)(const __attribute__((address_space(3))) unsigned char *)lds;
  int tid_c = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_c));
#pragma unroll 4
  for (int k = 0; k < HALF_ROW * BLOCK_COL / 4 / NUM_THREADS; k++) {
    const int q = tid_c + k * NUM_THREADS;
    *(gv4f *)(fs_g(dst) + (size_t)(q >> 6) * ld + (q & 63) * 4) = sv[q];
  }
}

// The fp32 C tile (rows rf0 / rf0 + 2 in units of RT::rows) to dst through LDS, in two 128-row halves.
template <typename RT>
__device__ __forceinline__ void sh_tile_park(unsigned char *lds, const RT &c00, const RT &c01, const RT &c10,
                                             const RT &c11, int rf0, int cf0, int cf1, int lane, float *dst, int ld) {
  static_assert(HALF_ROW * BLOCK_COL * sizeof(float) == LEGA_LDS_TILES, "a C half must fill the tile region");
  static_assert(2 * RT::rows == HALF_ROW, "rf0 covers the first half, rf0 + 2 the second");
  __attribute__((address_space(3))) float *st =
      (__attribute__((address_space(3))) float *)(__attribute__((address_space(3))) unsigned char *)lds;
  __syncthreads();
  sh_lds_f32(st, c00, rf0, cf0, BLOCK_COL, lane);
  sh_lds_f32(st, c01, rf0, cf1, BLOCK_COL, lane);
  __syncthreads();
  sh_half_park(lds, dst, ld);
  __syncthreads();
  sh_lds_f32(st, c10, rf0, cf0, BLOCK_COL, lane);
  sh_lds_f32(st, c11, rf0, cf1, BLOCK_COL, lane);
  __syncthreads();
  sh_half_park(lds, dst + (size_t)HALF_ROW * ld, ld);
}

// The fp32 C tile added into dw at element off (row stride ld) through LDS in two 128-row halves, rows at or past rlim
// skipped: acc 1 dw is fp32 (main_grad), 2 bf16 (each element bf16(old + sum), rounded once).
template <typename RT>
__device__ __forceinline__ void wg_tile_acc(unsigned char *lds, const RT &c00, const RT &c01, const RT &c10,
                                            const RT &c11, int rf0, int cf0, int cf1, int lane, void *dw, size_t off,
                                            int ld, int acc, int rlim) {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef unsigned v2u __attribute__((ext_vector_type(2)));
  static_assert(HALF_ROW * BLOCK_COL * sizeof(float) == LEGA_LDS_TILES, "a C half must fill the tile region");
  static_assert(2 * RT::rows == HALF_ROW, "rf0 covers the first half, rf0 + 2 the second");
  __attribute__((address_space(3))) float *st =
      (__attribute__((address_space(3))) float *)(__attribute__((address_space(3))) unsigned char *)lds;
  const __attribute__((address_space(3))) v4f *sv = (const __attribute__((address_space(3))) v4f *)st;
  int tid_c = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_c));
  for (int h = 0; h < 2; h++) {
    __syncthreads();
    sh_lds_f32(st, h ? c10 : c00, rf0, cf0, BLOCK_COL, lane);
    sh_lds_f32(st, h ? c11 : c01, rf0, cf1, BLOCK_COL, lane);
    __syncthreads();
    for (int q = tid_c; q < HALF_ROW * BLOCK_COL / 4; q += NUM_THREADS) {
      const int r = h * HALF_ROW + (q >> 6);
      if (r >= rlim) {
        continue;
      }
      v4f v = sv[q];
      const size_t g = off + (size_t)r * ld + (q & 63) * 4;
      if (acc == 1) {
        __attribute__((address_space(1))) v4f *d = (__attribute__((address_space(1))) v4f *)(fs_g((float *)dw) + g);
        *d = *d + v;
      } else {
        __attribute__((address_space(1))) v2u *d = (__attribute__((address_space(1))) v2u *)(fs_g((bf16 *)dw) + g);
        const v2u m = *d;
        v[0] += __builtin_bit_cast(float, m[0] << 16), v[1] += __builtin_bit_cast(float, m[0] & 0xffff0000u);
        v[2] += __builtin_bit_cast(float, m[1] << 16), v[3] += __builtin_bit_cast(float, m[1] & 0xffff0000u);
        *d = v2u{pk_bf16(v[0], v[1]), pk_bf16(v[2], v[3])};
      }
    }
  }
  __syncthreads();
}

// R: the B operand is (act, act_rows, act_cols) instead of wa's, and rows of dw at or past rlim are not stored. SHE:
// rlim = -1 - p parks the tile in fp32 instead, in partial region p of the shared expert's buffer at dw (p 0: fc2 [H][I],
// 1 and 2: fc1 slices [2I][H]), then counts it on the region's tile counter (stored after the 5 H I floats). ACC: rlim
// > BLOCK_ROW adds the tile to dw (an expert's rows split over rounds). wacc 1 or 2 (main_grad accumulation): the tile
// is added into dw instead (wg_tile_acc; every round alike, rows at or past min(rlim, BLOCK_ROW) skipped).
template <bool R = false, bool SHE = false, bool ACC = false>
__device__ __forceinline__ void lega_wgrad2_tile(const FsGl &a_gl, const LegAWgrad2 *__restrict__ wa, unsigned char *lds,
                                                 int e, int nblk, int mblk, int kt0, int k_real, bf16 *dw,
                                                 const bf16 *act = nullptr, int act_rows = 0, int act_cols = 0,
                                                 int rlim = BLOCK_ROW, int wacc = 0) {
  // The pipeline retires k-steps in pairs; for an odd k_real the extra step reloads k-step k_real - 1 (in bounds,
  // never read) and tail 2 skips its reads and MMAs but keeps its barriers.
  const int k_tiles = k_real + (k_real & 1);
  using ST_AB = st_bf<K_STEP_64, HALF_COL, st_16x32_s>;
  using RT_A = rt<bf16, K_STEP_64, NT_REG_M / 2, col_l, rt_32x16_s>;
  using RT_B = rt<bf16, K_STEP_64, NT_REG_N / 2, col_l, rt_32x16_s>;
  using RT_C = rt_fl<NT_REG_M / 2, NT_REG_N / 2, col_l, rt_16x16_s>;
  static_assert(4 * sizeof(ST_AB) * 2 == LEGA_LDS_TILES, "NT tiles must fill the shared region exactly");
  ST_AB(&As)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(lds);
  ST_AB(&Bs)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(lds + 4 * sizeof(ST_AB));
  // Operand fields made provably wave-uniform (readfirstlane): a generic SRD base lands in VGPRs and puts a waterfall
  // loop on each LDS-DMA load.
  const __attribute__((address_space(1))) LegAWgrad2 *wg = (const __attribute__((address_space(1))) LegAWgrad2 *)wa;
  FsGl b_gl(a_gl);
  if constexpr (R) {
    b_gl.raw_ptr = fs_uniform_ptr(const_cast<bf16 *>(act));
    b_gl.rows_internal = __builtin_amdgcn_readfirstlane(act_rows);
    b_gl.cols_internal = __builtin_amdgcn_readfirstlane(act_cols);
  } else {
    b_gl.raw_ptr = fs_uniform_ptr(const_cast<bf16 *>(wg->act));
    b_gl.rows_internal = __builtin_amdgcn_readfirstlane(wg->act_rows);
    b_gl.cols_internal = __builtin_amdgcn_readfirstlane(wg->act_cols);
  }
  const int A_LEAD = a_gl.cols(), B_LEAD = b_gl.cols();

  // Warp coordinates from an opaque thread id: derived from warpid() at kernel scope, their per-lane products are
  // hoisted and stay live across the NN body.
  int tid_o = (int)threadIdx.x;
  asm volatile("" : "+v"(tid_o));
  const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
  const int warp_m = wv_o / WARPS_COL;
  const int warp_n = wv_o % WARPS_COL;
  const int wid    = wv_o % NUM_WARPS;
  constexpr int bpt           = ST_AB::underlying_subtile_bytes_per_thread;
  constexpr int cps           = K_STEP_64 * BLOCK_COL * sizeof(bf16) / (bpt * NUM_THREADS);
  constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
  const uintptr_t lds_off = wid * elem_per_warp * sizeof(bf16);
  uint32_t a_lds[2][2], b_lds[2][2];
#pragma unroll
  for (int p_ = 0; p_ < 2; p_++) {
#pragma unroll
    for (int j_ = 0; j_ < 2; j_++) {
      a_lds[p_][j_] = __builtin_amdgcn_readfirstlane(
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[p_][j_].data[0]) + lds_off));
      b_lds[p_][j_] = __builtin_amdgcn_readfirstlane(
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[p_][j_].data[0]) + lds_off));
    }
  }
  const bf16 *a_base = (const bf16 *)&a_gl[{0, 0, 0, 0}];
  const bf16 *b_base = (const bf16 *)&b_gl[{0, 0, 0, 0}];
  i32x4 a_srd = make_srsrc(a_base, (uint32_t)((size_t)a_gl.rows() * A_LEAD * sizeof(bf16)));
  i32x4 b_srd = make_srsrc(b_base, (uint32_t)((size_t)b_gl.rows() * B_LEAD * sizeof(bf16)));
  uint32_t sw_A[cps / 2], sw_B[cps / 2];
  int lane_t = (int)threadIdx.x & 63;
  asm volatile("" : "+v"(lane_t));
  lega_prefill_sw<ST_AB>(lane_t, A_LEAD, sw_A);
  lega_prefill_sw<ST_AB>(lane_t, B_LEAD, sw_B);

  LegaLoader<ST_AB, cps / 2> ldA{As, a_gl, kt0, k_real, nblk, sw_A, a_srd, a_base, a_lds};
  LegaLoader<ST_AB, cps / 2> ldB{Bs, b_gl, kt0, k_real, mblk, sw_B, b_srd, b_base, b_lds};
#define NT_READ_A(T, P, J) \
  nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(T, As[0][0], warp_m * (NT_REG_M / 2), lane_t)
#define NT_READ_B(T, P, J) \
  nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(T, Bs[0][0], warp_n * (NT_REG_N / 2), lane_t)

  RT_C c00, c01, c10, c11;
  RT_A A_tile;
  RT_B B_tile_0, B_tile_1;
  const int lane0 = lane_t;

  if (k_tiles > 0) {
    ldB.load(0, 0, 0);
    ldA.load(0, 0, 0);
    ldB.load(0, 1, 0);
    ldA.load(0, 1, 0);
  }
  if (warp_m == 1) {
    __builtin_amdgcn_s_barrier();
  }
#include "moe_nt_kloop.inc"
#undef NT_READ_A
#undef NT_READ_B
  const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
  const int cf0 = __builtin_amdgcn_readfirstlane(warp_n);
  int lane_epi = lane0;
  asm volatile("" : "+v"(lane_epi));
  if constexpr (SHE) {
    if (rlim < 0) {
      // fc2: A_LEAD = H, B_LEAD = I; fc1: A_LEAD = 2I, B_LEAD = H.
      const int p = -1 - rlim, hi = p ? A_LEAD / 2 * B_LEAD : A_LEAD * B_LEAD;
      float *base = (float *)(void *)fs_uniform_ptr(dw);
      sh_tile_park(lds, c00, c01, c10, c11, rf0, cf0, cf0 + WARPS_COL, lane_epi,
                   base + (size_t)hi * (p ? 2 * p - 1 : 0) + (size_t)nblk * BLOCK_ROW * B_LEAD + mblk * BLOCK_COL,
                   B_LEAD);
      // Every wave's stores drained, then one release add on the tile's counter.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (fs_tid() == 0) {
        const int n2 = hi / (BLOCK_ROW * BLOCK_COL);
        gint *wd = fs_g((int *)(base + (size_t)5 * hi));
        (void)__hip_atomic_fetch_add(wd + (p ? n2 : 0) + nblk * (B_LEAD / BLOCK_COL) + mblk, 1, __ATOMIC_RELEASE,
                                     __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
      return;
    }
  }
  if (wacc) {
    wg_tile_acc(lds, c00, c01, c10, c11, rf0, cf0, cf0 + WARPS_COL, lane_epi, fs_uniform_ptr((void *)dw),
                ((size_t)e * A_LEAD + nblk * BLOCK_ROW) * B_LEAD + mblk * BLOCK_COL, B_LEAD, wacc,
                min(rlim, BLOCK_ROW));
    return;
  }
  bf16 *tb = fs_uniform_ptr(dw) + ((size_t)e * A_LEAD + nblk * BLOCK_ROW) * B_LEAD + mblk * BLOCK_COL;
  store_c_staged<0, R, ACC>(lds, c00, c01, c10, c11, rf0, rf0 + WARPS_ROW, cf0, cf0 + WARPS_COL, lane_epi, tb, B_LEAD,
                            rlim);
  __syncthreads();
}

}
