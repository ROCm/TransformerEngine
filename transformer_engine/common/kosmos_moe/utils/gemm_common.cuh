/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// Device pieces shared by the MoE and TP+SP GEMMs: tile geometry, LDS-DMA, swapped-operand MFMA, NT reads.
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"

#include <cstdint>

namespace kosmos {

constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int K_STEP_64  = 64;
constexpr int K_STEP_128 = 128;

constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 4;
constexpr int NUM_WARPS = WARPS_ROW * WARPS_COL;

constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

using G_group = kittens::group<NUM_WARPS>;

constexpr int REG_M = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N = BLOCK_COL / WARPS_COL / 2;

constexpr int HALF_ROW = BLOCK_ROW / 2;
constexpr int HALF_COL = BLOCK_COL / 2;

constexpr int NUM_XCDS_AFF = 8;
constexpr int GRID_CAP     = 256;

typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
typedef float f32x2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

// LDS_OFF is added after the opaque copy, so a loop keeps one base per tile instead of one per buffer.
template <int LDS_OFF = 0, typename ST, typename GL, typename COORD = kittens::coord<ST>>
__device__ __forceinline__ void dma_load(ST &, const GL &src, const COORD &idx, const uint32_t *__restrict__ sw,
                                         kittens::i32x4 srd, const void *base_ptr, uint32_t lds_base) {
  using T = typename ST::dtype;
  constexpr int bytes_per_memcpy = 16 * NUM_THREADS;
  constexpr int n                = (ST::rows * ST::cols * (int)sizeof(T)) / bytes_per_memcpy;
  kittens::coord<> uc = idx.template unit_coord<2, 3>();
  const T *gptr = (const T *)&src[uc];
  const uint32_t soff = kittens::to_sgpr_u32((uint32_t)((const char *)gptr - (const char *)base_ptr));
  uint32_t lds_cur = lds_base;
  asm volatile("" : "+s"(lds_cur));
  lds_cur += LDS_OFF;
#pragma unroll
  for (int i = 0; i < n; i++) {
    int32_t lds_byte = lds_cur;
    asm volatile("" : "+s"(lds_byte));
    kittens::llvm_amdgcn_raw_buffer_load_lds(srd, (kittens::as3_uint32_ptr)(uintptr_t)(uint32_t)lds_byte, 16, sw[i],
                                             soff, 0, static_cast<int>(kittens::coherency::cache_all));
    lds_cur += bytes_per_memcpy;
  }
}

template <int H, typename ST, typename GL, typename COORD = kittens::coord<ST>>
__device__ __forceinline__ void dma_load_h(ST &, const GL &src, const COORD &idx, const uint32_t *__restrict__ sw,
                                           kittens::i32x4 srd, const void *base_ptr, uint32_t lds_base) {
  using T = typename ST::dtype;
  constexpr int bytes_per_memcpy = 16 * NUM_THREADS;
  static_assert((ST::rows * ST::cols * (int)sizeof(T)) / bytes_per_memcpy == 2 && (H == 0 || H == 1));
  kittens::coord<> uc = idx.template unit_coord<2, 3>();
  const T *gptr = (const T *)&src[uc];
  const uint32_t soff = kittens::to_sgpr_u32((uint32_t)((const char *)gptr - (const char *)base_ptr));
#if __clang_major__ >= 24
  // readfirstlane before the asm (HipKittens #87): clang 24 can hold lds_base in a VGPR, which "+s" alone cannot take.
  // Not under clang 23, where it would change the backward kernels' ISA.
  const int32_t lds_byte = kittens::to_sgpr_u32(lds_base + H * bytes_per_memcpy);
#else
  int32_t lds_byte = lds_base + H * bytes_per_memcpy;
  asm volatile("" : "+s"(lds_byte));
#endif
  kittens::llvm_amdgcn_raw_buffer_load_lds(srd, (kittens::as3_uint32_ptr)(uintptr_t)(uint32_t)lds_byte, 16, sw[H],
                                           soff, 0, static_cast<int>(kittens::coherency::cache_all));
}

template <typename DT, typename A, typename B>
__device__ __forceinline__ void mma_ABt_T(DT &dT, const A &a, const B &b) {
  static_assert(DT::rows == B::rows && DT::cols == A::rows, "dT must be (B rows) x (A rows)");
  static_assert(A::cols == B::cols, "reduction dims differ");
#pragma unroll
  for (int n = 0; n < A::height; n++) {
#pragma unroll
    for (int m = 0; m < B::height; m++) {
#pragma unroll
      for (int k = 0; k < A::width; k++) {
        kittens::mma_ABt_base(dT.tiles[m][n], b.tiles[m][k], a.tiles[n][k], dT.tiles[m][n]);
      }
    }
  }
}

__device__ __forceinline__ uint32_t pk_bf16(float x, float y) {
  f32x2_t f = {x, y};
  return __builtin_bit_cast(uint32_t, __builtin_convertvector(f, bf16x2_t));
}

template <typename CT>
__device__ __forceinline__ u32x4_t T16_row(const CT &ct, int mb) {
  static_assert(CT::height == 2 && CT::base_tile_rows == 16 && CT::base_tile_cols == 16, "16x16 CT expected");
  const uint32_t x0 = pk_bf16(ct.tiles[0][mb].data[0].x, ct.tiles[0][mb].data[0].y);
  const uint32_t x1 = pk_bf16(ct.tiles[0][mb].data[1].x, ct.tiles[0][mb].data[1].y);
  const uint32_t y0 = pk_bf16(ct.tiles[1][mb].data[0].x, ct.tiles[1][mb].data[0].y);
  const uint32_t y1 = pk_bf16(ct.tiles[1][mb].data[1].x, ct.tiles[1][mb].data[1].y);
  const auto r0 = __builtin_amdgcn_permlane16_swap(x0, y0, false, false);
  const auto r1 = __builtin_amdgcn_permlane16_swap(x1, y1, false, false);
  return u32x4_t{(uint32_t)r0[0], (uint32_t)r1[0], (uint32_t)r0[1], (uint32_t)r1[1]};
}

template <int OFF = 0, bool OWN_LANE = false, typename RT, typename ST>
__device__ __forceinline__ void nt_load_frag(RT &dst, const ST &src, int col_base, int lane) {
  static_assert(RT::base_tile_rows == 32 && RT::base_tile_cols == 16, "nt_load_frag: rt_32x16 base only");
  static_assert(ST::underlying_subtile_rows == 16 && ST::underlying_subtile_cols == 32, "st_16x32 only");
  static_assert(OFF + (RT::height - 1) * 2 * ST::underlying_subtiles_per_row * ST::underlying_subtile_bytes +
                        ((RT::width - 1) / 2) * ST::underlying_subtile_bytes + 256 <= 65535,
                "ds_read offset field");
  const int L = OWN_LANE ? kittens::laneid() : lane;
  const int g = L / 16, r_in = (L % 16) / 4, c_in = 4 * (L % 4);
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&src.data[0])) +
                        (g / 2) * ST::underlying_subtiles_per_row * ST::underlying_subtile_bytes +
                        (col_base / 32) * ST::underlying_subtile_bytes;
  const int r = 8 * (g % 2) + r_in;
  const uint32_t b0 = base + ST::swizzle({r, c_in});
  const uint32_t b1 = base + ST::swizzle({r, 16 + c_in});
  constexpr int KF = 2 * ST::underlying_subtiles_per_row * ST::underlying_subtile_bytes;
  constexpr int SB = ST::underlying_subtile_bytes;
#pragma unroll
  for (int kf = 0; kf < RT::height; kf++) {
#pragma unroll
    for (int mb = 0; mb < RT::width; mb++) {
      // The first output is early-clobber: aliasing it with the address register gave silent zero rows.
      asm volatile("ds_read_b64_tr_b16 %0, %2 offset:%3\n"
                   "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
                   : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[kf][mb].data[0])),
                     "=v"(*reinterpret_cast<float2 *>(&dst.tiles[kf][mb].data[2]))
                   : "v"((mb % 2) ? b1 : b0), "i"(OFF + kf * KF + (mb / 2) * SB), "i"(OFF + kf * KF + (mb / 2) * SB + 256)
                   : "memory");
    }
  }
}

template <typename RT, typename ST>
__device__ __forceinline__ void nt_load_frag(RT &dst, const ST &src, int col_base) {
  nt_load_frag<0, true>(dst, src, col_base, 0);
}

// Lane address of row fragment k of tile row_tile in buffer 0 of an ST[2][2] array; load_rows adds buffer BUF.
// The lane is opaque so a persistent kernel recomputes the address per tile instead of holding it across tiles.
template <typename RT, typename ST>
__device__ __forceinline__ uint32_t row_frag_addr(const ST &st, int row_tile, int k) {
  int lane = kittens::laneid();
  asm volatile("" : "+v"(lane));
  const int row  = lane % RT::base_tile_rows;
  const int col  = RT::base_tile_stride * (lane / RT::base_tile_rows) + k * RT::base_tile_elements_per_stride_group;
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0])) +
         row_tile * (RT::rows / ST::underlying_subtile_rows) * ST::underlying_subtile_bytes + ST::swizzle({row, col});
}

template <typename ST, int BUF, typename RT>
__device__ __forceinline__ void load_rows(RT &dst, uint32_t addr0, uint32_t addr1) {
  static_assert(BUF * sizeof(ST) + (RT::height - 1) * ST::underlying_subtile_bytes <= 65535, "ds_read offset field");
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
    asm volatile("ds_read_b128 %0, %1 offset:%2"
                 : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[i][0].data[0]))
                 : "v"(addr0), "i"(BUF * sizeof(ST) + i * ST::underlying_subtile_bytes)
                 : "memory");
    asm volatile("ds_read_b128 %0, %1 offset:%2"
                 : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[i][0].data[4]))
                 : "v"(addr1), "i"(BUF * sizeof(ST) + i * ST::underlying_subtile_bytes)
                 : "memory");
  }
}

template <typename RT, typename ST>
__device__ __forceinline__ void col_frag_addrs(uint32_t (&addr)[RT::height], const ST &st, int col_offset) {
  int lane = kittens::laneid();
  asm volatile("" : "+v"(lane));
  const int l_within = lane % 16;
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
    const int m_col = col_offset + i * RT::base_tile_rows + (l_within & 1) * 8;
    addr[i] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0])) +
              (lane / 16) * ST::underlying_subtile_bytes + ST::swizzle({l_within / 2, m_col});
  }
}

// The XOR selects the upper K half; buffer offsets are multiples of 16 KiB, so it commutes with the immediate.
template <typename ST, int BUF, typename RT>
__device__ __forceinline__ void load_cols(RT &dst, const uint32_t (&addr)[RT::height]) {
  constexpr int OFF = BUF * sizeof(ST);
  constexpr int SUB = ST::underlying_subtile_bytes;
  static_assert(sizeof(ST) % 16384 == 0 && OFF + 4 * SUB <= 65535, "ds_read offset field");
#pragma unroll
  for (int i = 0; i < RT::height; i++) {
    asm volatile("ds_read_b64_tr_b8 %0, %2 offset:%3\n"
                 "ds_read_b64_tr_b8 %1, %2 offset:%4\n"
                 : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[0])),
                   "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[4]))
                 : "v"(addr[i]), "i"(OFF), "i"(OFF + 4 * SUB)
                 : "memory");
    asm volatile("ds_read_b64_tr_b8 %0, %2 offset:%3\n"
                 "ds_read_b64_tr_b8 %1, %2 offset:%4\n"
                 : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[2])),
                   "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[6]))
                 : "v"(addr[i] ^ 1088u), "i"(OFF), "i"(OFF + 4 * SUB)
                 : "memory");
  }
}

template <bool COLWISE, int STEP, int NG>
__global__ void pack_scales_kernel(const uint8_t *__restrict__ scales, uint32_t *__restrict__ ln, int dim, int scale_K,
                                   int tiles_per_col) {
  constexpr int PAD_WORDS = (NG - 1) * STEP + 64;
  __shared__ uint32_t tile[PAD_WORDS];

  const int kb_base = (static_cast<int>(blockIdx.x) / tiles_per_col) * 4;
  const int row0    = (static_cast<int>(blockIdx.x) % tiles_per_col) * 256;
  for (int i = threadIdx.x; i < PAD_WORDS; i += blockDim.x) {
    uint32_t p = 0;
    if (i < 256) {
      if constexpr (COLWISE) {
        const size_t base = static_cast<size_t>(kb_base) * dim + row0 + i;
        p = (uint32_t)scales[base] | ((uint32_t)scales[base + dim] << 8) | ((uint32_t)scales[base + 2 * dim] << 16) |
            ((uint32_t)scales[base + 3 * dim] << 24);
      } else {
        __builtin_memcpy(&p, &scales[static_cast<size_t>(row0 + i) * scale_K + kb_base], 4);
      }
    }
    tile[i] = p;
  }
  __syncthreads();

  const int grp = threadIdx.x / 64;
  ln[(static_cast<size_t>(blockIdx.x) * NG + grp) * 64 + threadIdx.x % 64] =
      kittens::pack_scales(reinterpret_cast<const kittens::fp8e8m0 *>(tile), grp * STEP);
}

template <bool COLWISE, int STEP, int NG>
void launch_pack_scales(const uint8_t *scales, uint32_t *ln, int dim, int scale_K, int k_iters, hipStream_t stream) {
  const int tiles_per_col = dim / 256;
  pack_scales_kernel<COLWISE, STEP, NG><<<k_iters * tiles_per_col, NG * 64, 0, stream>>>(scales, ln, dim, scale_K,
                                                                                          tiles_per_col);
}

}
