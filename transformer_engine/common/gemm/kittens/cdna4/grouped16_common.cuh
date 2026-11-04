/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/

// Device pieces of the gfx950 BF16/FP16 grouped GEMM: tile geometry, expert table, LDS-DMA, MFMA, epilogues.

#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "../grouped_sched.cuh"
#include "grouped16.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

using namespace kittens;

constexpr int BLOCK_ROW   = 256;
constexpr int BLOCK_COL   = 256;
constexpr int HALF_ROW    = BLOCK_ROW / 2;
constexpr int HALF_COL    = BLOCK_COL / 2;
constexpr int K_STEP      = 64;
constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int NUM_WARPS   = WARPS_ROW * WARPS_COL;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int REG_M       = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N       = BLOCK_COL / WARPS_COL / 2;
constexpr int LDS_TILES   = 131072;
constexpr int XCDS        = 8;
constexpr int MAX_EXPERTS = 64;

enum Layout { L_TN, L_NN, L_NT };

// A buffer offset past every descriptor's range (extents stay below 2 GiB): loads return 0, stores drop.
constexpr uint32_t OOB = 0x80000000u;

typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2_t __attribute__((ext_vector_type(2)));
typedef float f32x2_t __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) unsigned char lds_u8;

// Experts of one launch, in launch order; unit_off[e]..unit_off[e+1] are expert e's tiles.
struct Table {
    const void *a[MAX_EXPERTS];
    const void *b[MAX_EXPERTS];
    void *d[MAX_EXPERTS];
    const void *bias[MAX_EXPERTS];
    int rows[MAX_EXPERTS];
    int unit_off[MAX_EXPERTS + 1];
};
static_assert(sizeof(Table) + 64 <= 4096, "kernel arguments are limited to 4 KiB");

// threadIdx.x through an empty asm: what derives from it is computed at the use, not carried across the K-loops.
__device__ __forceinline__ int fs_tid() {
    int t = (int)threadIdx.x;
    asm volatile("" : "+v"(t));
    __builtin_assume(t >= 0 && t < NUM_THREADS);
    return t;
}

template <typename T>
__device__ __forceinline__ T *fs_uniform_ptr(T *p) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)a);
    const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(a >> 32));
    return (T *)(((unsigned long long)hi << 32) | lo);
}

// A buffer descriptor with the stride field 0 (soffset stays inside the range check), every dword scalar.
__device__ __forceinline__ i32x4 fs_srd(const void *p, uint32_t range) {
    const i32x4 r = make_srsrc(p, range, 0);
    return i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
                 __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
}

// Two LDS-DMA copies of 16 B per lane (one 128 x 64 16-bit tile) at byte offset soff, into LDS at lds_base.
__device__ __forceinline__ void dma2(const uint32_t *__restrict__ sw, i32x4 srd, uint32_t soff, uint32_t lds_base) {
    const uint32_t so = __builtin_amdgcn_readfirstlane(soff);
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur));
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int32_t lds_byte = lds_cur;
        asm volatile("" : "+s"(lds_byte));
        llvm_amdgcn_raw_buffer_load_lds(srd, (as3_uint32_ptr)(uintptr_t)(uint32_t)lds_byte, 16, sw[i], so, 0,
                                        static_cast<int>(coherency::cache_all));
        lds_cur += 16 * NUM_THREADS;
    }
}

template <int H>
__device__ __forceinline__ void dma_h(const uint32_t *__restrict__ sw, i32x4 srd, uint32_t soff, uint32_t lds_base) {
    static_assert(H == 0 || H == 1);
    const uint32_t so = __builtin_amdgcn_readfirstlane(soff);
    int32_t lds_byte = lds_base + H * 16 * NUM_THREADS;
    asm volatile("" : "+s"(lds_byte));
    llvm_amdgcn_raw_buffer_load_lds(srd, (as3_uint32_ptr)(uintptr_t)(uint32_t)lds_byte, 16, sw[H], so, 0,
                                    static_cast<int>(coherency::cache_all));
}

// Byte offsets (swizzled for ST) of this lane's 16 B LDS-DMA pieces in a tile of rows with row_bytes per row.
template <typename ST>
__device__ __forceinline__ void tile_lane_offsets(int lane, int warp, int row_bytes, uint32_t *off) {
    using T = typename ST::dtype;
    constexpr int bpt = 16;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int mpt = (ST::rows * ST::cols * (int)sizeof(T)) / bpm;
    constexpr int bpw = bpt * kittens::WARP_THREADS;
    const int wid = warp % NUM_WARPS;
#pragma unroll
    for (int i = 0; i < mpt; i++) {
        const int lbo = (lane * bpt) + (wid * bpw) + (i * NUM_WARPS * bpw);
        const int sid = lbo / ST::underlying_subtile_bytes;
        const int sr  = sid / ST::underlying_subtiles_per_row;
        const int sc  = sid % ST::underlying_subtiles_per_row;
        const int slb = lbo % ST::underlying_subtile_bytes;
        const int r   = slb / ST::underlying_subtile_row_bytes;
        const int c   = (slb % ST::underlying_subtile_row_bytes) / (int)sizeof(T);
        const uint32_t sw = ST::swizzle({r, c});
        const int row = (int)(sw / ST::underlying_subtile_row_bytes) + sr * ST::underlying_subtile_rows;
        const int col = (int)((sw % ST::underlying_subtile_row_bytes) / sizeof(T)) + sc * ST::underlying_subtile_cols;
        off[i] = (uint32_t)row * (uint32_t)row_bytes + (uint32_t)(col * (int)sizeof(T));
    }
}

// D (16x16 fp32, col layout) += A (16x32) B^T (16x32): the bf16 or fp16 MFMA, by operand type.
template <typename DB, typename AB, typename BB>
__device__ __forceinline__ void mfma16(DB &d, const AB &a, const BB &b) {
    kittens::mfma161632(d.data, a.data, b.data, d.data);
}

// dT (B rows x A rows) += (A B^T)^T: the swapped-operand MFMA of the TN body.
template <typename DT, typename A, typename B>
__device__ __forceinline__ void mma_ABt_T(DT &dT, const A &a, const B &b) {
    static_assert(DT::rows == B::rows && DT::cols == A::rows && A::cols == B::cols);
#pragma unroll
    for (int n = 0; n < A::height; n++) {
#pragma unroll
        for (int m = 0; m < B::height; m++) {
#pragma unroll
            for (int k = 0; k < A::width; k++) {
                mfma16(dT.tiles[m][n], b.tiles[m][k], a.tiles[n][k]);
            }
        }
    }
}

template <typename D, typename A, typename B>
__device__ __forceinline__ void mma_ABt(D &d, const A &a, const B &b) {
    static_assert(D::rows == A::rows && D::cols == B::rows && A::cols == B::cols);
#pragma unroll
    for (int n = 0; n < D::height; n++) {
#pragma unroll
        for (int m = 0; m < D::width; m++) {
#pragma unroll
            for (int k = 0; k < A::width; k++) {
                mfma16(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k]);
            }
        }
    }
}

template <typename D, typename A, typename B>
__device__ __forceinline__ void mma_AtB(D &d, const A &a, const B &b) {
    static_assert(D::rows == A::cols && D::cols == B::cols && A::rows == B::rows);
#pragma unroll
    for (int n = 0; n < D::height; n++) {
#pragma unroll
        for (int m = 0; m < D::width; m++) {
#pragma unroll
            for (int k = 0; k < A::height; k++) {
                mfma16(d.tiles[n][m], a.tiles[k][n], b.tiles[k][m]);
            }
        }
    }
}

// Row fragments of a 16x32-subtile shared tile, one ds_read_b128 per base tile (the TN operand reads).
template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void load_frag_rows(RT &dst, const ST &src, int lane) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols, "register tile and shared tile must match");
    static_assert(RT::base_tile_stride == 8 && RT::base_tile_num_strides == 1, "16-bit, one 16-byte stride per lane");
    static_assert(ST::underlying_subtile_rows == RT::base_tile_rows &&
                      ST::underlying_subtile_cols == RT::base_tile_cols,
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
                         : "v"(addr),
                           "i"(OFF + (ii * ST::underlying_subtiles_per_row + jj) * ST::underlying_subtile_bytes)
                         : "memory");
        }
    }
}

// The NN A operand (row fragments) at immediate OFF.
template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void nn_read_A(RT &dst, const ST &src) {
    static_assert(RT::base_tile_stride == 8, "16-bit, stride 8");
    static_assert(ST::underlying_subtile_rows >= RT::base_tile_rows &&
                      ST::underlying_subtile_cols >= RT::base_tile_cols,
                  "shared subtile >= register subtile");
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    constexpr int rps     = ST::underlying_subtile_cols / RT::base_tile_cols;
    constexpr int rpc     = ST::underlying_subtile_rows / RT::base_tile_rows;
    const int lid        = kittens::laneid();
    const int row_offset = lid % RT::base_tile_rows;
    const int col_offset = RT::base_tile_stride * (lid / RT::base_tile_rows);
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
#pragma unroll
    for (int k = 0; k < RT::base_tile_num_strides; k++) {
#pragma unroll
        for (int i = 0; i < rpc; i++) {
#pragma unroll
            for (int j = 0; j < rps; j++) {
                const int row = i * RT::base_tile_rows + row_offset;
                const int col = j * RT::base_tile_cols + col_offset + k * RT::base_tile_elements_per_stride_group;
                const uint32_t addr = src_ptr + src.swizzle({row, col});
                const int idx = k * RT::base_tile_stride / packing;
#pragma unroll
                for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
#pragma unroll
                    for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                        const int offset = (ii * ST::underlying_subtiles_per_row + jj) * ST::underlying_subtile_bytes;
                        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                                     : "=v"(*reinterpret_cast<float4 *>(
                                           &dst.tiles[ii * rpc + i][jj * rps + j].data[idx]))
                                     : "v"(addr), "i"(offset + OFF)
                                     : "memory");
                    }
                }
            }
        }
    }
}

// Transposed (K-major) fragments for the NT body: ds_read_b64_tr_b16 pairs.
template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void nt_load_frag(RT &dst, const ST &src, int col_base, int lane) {
    static_assert(RT::base_tile_rows == 32 && RT::base_tile_cols == 16, "nt_load_frag: rt_32x16 base only");
    static_assert(ST::underlying_subtile_rows == 16 && ST::underlying_subtile_cols == 32, "st_16x32 only");
    static_assert(OFF + (RT::height - 1) * 2 * ST::underlying_subtiles_per_row * ST::underlying_subtile_bytes +
                          ((RT::width - 1) / 2) * ST::underlying_subtile_bytes + 256 <= 65535,
                  "ds_read offset field");
    const int g = lane / 16, r_in = (lane % 16) / 4, c_in = 4 * (lane % 4);
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
            // The first output is early-clobber: aliasing it with the address register gives silent zero rows.
            asm volatile("ds_read_b64_tr_b16 %0, %2 offset:%3\n"
                         "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
                         : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[kf][mb].data[0])),
                           "=v"(*reinterpret_cast<float2 *>(&dst.tiles[kf][mb].data[2]))
                         : "v"((mb % 2) ? b1 : b0), "i"(OFF + kf * KF + (mb / 2) * SB),
                           "i"(OFF + kf * KF + (mb / 2) * SB + 256)
                         : "memory");
        }
    }
}

// Buffer loads and stores with the intrinsics' native vector types.
__device__ u32x4_t buf_load_v4(i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ u32x2_t buf_load_v2(i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ void buf_store_v4(u32x4_t v, i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v4i32");
__device__ void buf_store_v2(u32x2_t v, i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v2i32");

// A 32-bit word as float. __builtin_bit_cast of a vector element lvalue (bit_cast(float, a[i])) reads element 0 for
// every i with this compiler: copy the element to a scalar first.
__device__ __forceinline__ float as_f32(uint32_t w) {
    return __builtin_bit_cast(float, w);
}

// Output element types: 16-bit results round to nearest even.
template <typename O>
__device__ __forceinline__ uint32_t pack2(float x, float y) {
    if constexpr (std::is_same_v<O, __bf16>) {
        return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2_t{x, y}, bf16x2_t));
    } else {
        static_assert(std::is_same_v<O, _Float16>);
        return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2_t{x, y}, f16x2_t));
    }
}

template <typename O>
__device__ __forceinline__ float lo16(uint32_t w) {
    if constexpr (std::is_same_v<O, __bf16>) {
        return __builtin_bit_cast(float, w << 16);
    } else {
        return (float)__builtin_bit_cast(_Float16, (uint16_t)(w & 0xffffu));
    }
}

template <typename O>
__device__ __forceinline__ float hi16(uint32_t w) {
    if constexpr (std::is_same_v<O, __bf16>) {
        return __builtin_bit_cast(float, w & 0xffff0000u);
    } else {
        return (float)__builtin_bit_cast(_Float16, (uint16_t)(w >> 16));
    }
}

// Eight consecutive output elements at voff through srd (rows past the descriptor's range drop).
template <typename O>
__device__ __forceinline__ void store8(i32x4 srd, uint32_t voff, const float (&v)[8]) {
    if constexpr (std::is_same_v<O, float>) {
        const u32x4_t a = {__builtin_bit_cast(uint32_t, v[0]), __builtin_bit_cast(uint32_t, v[1]),
                           __builtin_bit_cast(uint32_t, v[2]), __builtin_bit_cast(uint32_t, v[3])};
        const u32x4_t b = {__builtin_bit_cast(uint32_t, v[4]), __builtin_bit_cast(uint32_t, v[5]),
                           __builtin_bit_cast(uint32_t, v[6]), __builtin_bit_cast(uint32_t, v[7])};
        buf_store_v4(a, srd, voff, 0, 0);
        buf_store_v4(b, srd, voff + 16, 0, 0);
    } else {
        const u32x4_t a = {pack2<O>(v[0], v[1]), pack2<O>(v[2], v[3]), pack2<O>(v[4], v[5]), pack2<O>(v[6], v[7])};
        buf_store_v4(a, srd, voff, 0, 0);
    }
}

template <typename O>
__device__ __forceinline__ void load8(i32x4 srd, uint32_t voff, float (&v)[8]) {
    if constexpr (std::is_same_v<O, float>) {
        const u32x4_t a = buf_load_v4(srd, voff, 0, 0);
        const u32x4_t b = buf_load_v4(srd, voff + 16, 0, 0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i]     = as_f32(a[i]);
            v[4 + i] = as_f32(b[i]);
        }
    } else {
        const u32x4_t a = buf_load_v4(srd, voff, 0, 0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[2 * i]     = lo16<O>(a[i]);
            v[2 * i + 1] = hi16<O>(a[i]);
        }
    }
}

template <typename O>
__device__ __forceinline__ void store4(i32x4 srd, uint32_t voff, const float (&v)[4]) {
    if constexpr (std::is_same_v<O, float>) {
        const u32x4_t a = {__builtin_bit_cast(uint32_t, v[0]), __builtin_bit_cast(uint32_t, v[1]),
                           __builtin_bit_cast(uint32_t, v[2]), __builtin_bit_cast(uint32_t, v[3])};
        buf_store_v4(a, srd, voff, 0, 0);
    } else {
        const u32x2_t a = {pack2<O>(v[0], v[1]), pack2<O>(v[2], v[3])};
        buf_store_v2(a, srd, voff, 0, 0);
    }
}

template <typename O>
__device__ __forceinline__ void load4(i32x4 srd, uint32_t voff, float (&v)[4]) {
    if constexpr (std::is_same_v<O, float>) {
        const u32x4_t a = buf_load_v4(srd, voff, 0, 0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = as_f32(a[i]);
        }
    } else {
        const u32x2_t a = buf_load_v2(srd, voff, 0, 0);
        v[0] = lo16<O>(a[0]);
        v[1] = hi16<O>(a[0]);
        v[2] = lo16<O>(a[1]);
        v[3] = hi16<O>(a[1]);
    }
}

/* TN epilogue of one transposed accumulator (REG_N x REG_M, rows = output columns): each lane stores 8
 * consecutive columns of one output row per 16-row block, through srd (based at the tile's first row,
 * ranged to the expert's rows, so rows past them drop); columns past ld drop. bias_srd is ranged to
 * the ld bias values. D = alpha acc (+ beta D) (+ bias). */
template <typename O, bool BIAS, bool ACC, typename BT, typename CT>
__device__ __forceinline__ void epi_t16(i32x4 srd, const CT &ct, int m0, int n0, int ld, int lane, float alpha,
                                        float beta, i32x4 bias_srd) {
    static_assert(CT::height == 2 && CT::base_tile_rows == 16 && CT::base_tile_cols == 16, "16x16 CT expected");
    const int c8 = 16 * ((lane >> 4) & 1) + 8 * (lane >> 5);
    float bv[8];
    if constexpr (BIAS) {
        const u32x4_t w = buf_load_v4(bias_srd, (uint32_t)(n0 + c8) * (uint32_t)sizeof(BT), 0, 0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            bv[2 * i]     = lo16<BT>(w[i]);
            bv[2 * i + 1] = hi16<BT>(w[i]);
        }
    }
    const uint32_t lo = n0 + c8 < ld ? (uint32_t)((m0 + (lane & 15)) * ld + n0 + c8) * (uint32_t)sizeof(O) : OOB;
#pragma unroll
    for (int mb = 0; mb < CT::width; mb++) {
        const float a[4] = {ct.tiles[0][mb].data[0].x, ct.tiles[0][mb].data[0].y, ct.tiles[0][mb].data[1].x,
                            ct.tiles[0][mb].data[1].y};
        const float b[4] = {ct.tiles[1][mb].data[0].x, ct.tiles[1][mb].data[0].y, ct.tiles[1][mb].data[1].x,
                            ct.tiles[1][mb].data[1].y};
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const auto r = __builtin_amdgcn_permlane16_swap(__builtin_bit_cast(uint32_t, a[i]),
                                                            __builtin_bit_cast(uint32_t, b[i]), false, false);
            v[i]     = __builtin_bit_cast(float, (uint32_t)r[0]) * alpha;
            v[4 + i] = __builtin_bit_cast(float, (uint32_t)r[1]) * alpha;
        }
        const uint32_t off = lo + (uint32_t)(mb * 16 * ld) * (uint32_t)sizeof(O);
        if constexpr (ACC) {
            float c[8];
            load8<O>(srd, off, c);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                v[i] = fmaf(beta, c[i], v[i]);
            }
        }
        if constexpr (BIAS) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                v[i] += bv[i];
            }
        }
        store8<O>(srd, off, v);
    }
}

// One col-layout accumulator (row_unit, col_unit) to a 16-bit LDS tile of row_stride elements, scaled by alpha.
template <typename O, typename RT>
__device__ __forceinline__ void lds_put16(__attribute__((address_space(3))) uint16_t *base, const RT &src,
                                          int row_unit, int col_unit, int row_stride, int lane, float alpha) {
    static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "col_l 16x16 accumulator");
    __attribute__((address_space(3))) uint16_t *p = base + (row_unit * RT::rows) * row_stride + col_unit * RT::cols;
    const int row_offset = 4 * (lane / RT::base_tile_cols);
    const int col_offset = lane % RT::base_tile_cols;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const int row = i * RT::base_tile_rows + row_offset, col = j * RT::base_tile_cols + col_offset;
#pragma unroll
            for (int l = 0; l < 2; l++) {
                const uint32_t w = pack2<O>(src.tiles[i][j].data[l].x * alpha, src.tiles[i][j].data[l].y * alpha);
                p[(row + l * 2) * row_stride + col]     = (uint16_t)(w & 0xffffu);
                p[(row + l * 2 + 1) * row_stride + col] = (uint16_t)(w >> 16);
            }
        }
    }
}

template <typename RT>
__device__ __forceinline__ void lds_put32(__attribute__((address_space(3))) float *base, const RT &src, int row_unit,
                                          int col_unit, int row_stride, int lane) {
    static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "col_l 16x16 accumulator");
    __attribute__((address_space(3))) float *p = base + (row_unit * RT::rows) * row_stride + col_unit * RT::cols;
    const int row_offset = 4 * (lane / RT::base_tile_cols);
    const int col_offset = lane % RT::base_tile_cols;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const int row = i * RT::base_tile_rows + row_offset, col = j * RT::base_tile_cols + col_offset;
#pragma unroll
            for (int l = 0; l < 2; l++) {
                p[(row + l * 2) * row_stride + col]     = src.tiles[i][j].data[l].x;
                p[(row + l * 2 + 1) * row_stride + col] = src.tiles[i][j].data[l].y;
            }
        }
    }
}

/* NN/NT epilogue of a 256x256 tile held as four col-layout accumulators at (rf0, cf0), (rf0, cf0 + 4),
 * (rf0 + 2, cf0), (rf0 + 2, cf0 + 4) in units of (RT::rows, RT::cols), staged through the 128 KiB tile
 * LDS. srd is based at the tile's first element and ranged to the valid rows (rows past them drop);
 * columns from cols on drop. 16-bit without accumulate: one 16-bit pass. Otherwise fp32 in two 128-row
 * halves, D = alpha acc + beta D. */
template <typename O, bool ACC, typename RT>
__device__ __forceinline__ void epi_staged(unsigned char *lds, const RT &c00, const RT &c01, const RT &c10,
                                           const RT &c11, int rf0, int cf0, int lane, i32x4 srd, int ld, int cols,
                                           float alpha, float beta) {
    static_assert(2 * RT::rows == HALF_ROW && 8 * RT::cols == BLOCK_COL, "tile geometry");
    __syncthreads();
    int tid_c = (int)threadIdx.x;
    asm volatile("" : "+v"(tid_c));
    if constexpr (!std::is_same_v<O, float> && !ACC) {
        auto *st = (__attribute__((address_space(3))) uint16_t *)(lds_u8 *)lds;
        lds_put16<O>(st, c00, rf0, cf0, BLOCK_COL, lane, alpha);
        lds_put16<O>(st, c01, rf0, cf0 + WARPS_COL, BLOCK_COL, lane, alpha);
        lds_put16<O>(st, c10, rf0 + WARPS_ROW, cf0, BLOCK_COL, lane, alpha);
        lds_put16<O>(st, c11, rf0 + WARPS_ROW, cf0 + WARPS_COL, BLOCK_COL, lane, alpha);
        __syncthreads();
        const __attribute__((address_space(3))) u32x4_t *sv = (const __attribute__((address_space(3))) u32x4_t *)st;
        // Thread tid_c stores 8 columns at 8 (tid_c % 32) of rows tid_c / 32 + 16 k.
        const int ch = tid_c & 31;
        const uint32_t off0 = ch * 8 < cols ? (uint32_t)(((tid_c >> 5) * ld + ch * 8) * 2) : OOB;
#pragma unroll
        for (int k = 0; k < BLOCK_ROW * BLOCK_COL / 8 / NUM_THREADS; k++) {
            buf_store_v4(sv[tid_c + k * NUM_THREADS], srd, off0 + (uint32_t)(k * (NUM_THREADS / 32) * ld * 2), 0, 0);
        }
    } else {
        auto *st = (__attribute__((address_space(3))) float *)(lds_u8 *)lds;
        const __attribute__((address_space(3))) f32x2_t *sv = (const __attribute__((address_space(3))) f32x2_t *)st;
        // Thread tid_c stores 4 columns at 4 (tid_c % 64) of rows tid_c / 64 + 8 k of each half.
        const int ch = tid_c & 63;
        const uint32_t off0 = ch * 4 < cols ? (uint32_t)(((tid_c >> 6) * ld + ch * 4) * (int)sizeof(O)) : OOB;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (h) {
                __syncthreads();
            }
            lds_put32(st, h ? c10 : c00, rf0, cf0, BLOCK_COL, lane);
            lds_put32(st, h ? c11 : c01, rf0, cf0 + WARPS_COL, BLOCK_COL, lane);
            __syncthreads();
#pragma unroll
            for (int k = 0; k < HALF_ROW * BLOCK_COL / 4 / NUM_THREADS; k++) {
                const int q = tid_c + k * NUM_THREADS;
                const f32x2_t x = sv[2 * q], y = sv[2 * q + 1];
                float v[4] = {x[0] * alpha, x[1] * alpha, y[0] * alpha, y[1] * alpha};
                const uint32_t off = off0 + (uint32_t)((h * HALF_ROW + k * (NUM_THREADS / 64)) * ld * (int)sizeof(O));
                if constexpr (ACC) {
                    float c[4];
                    load4<O>(srd, off, c);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        v[i] = fmaf(beta, c[i], v[i]);
                    }
                }
                store4<O>(srd, off, v);
            }
        }
    }
}

// The persistent loop's claim: thread 0 claims, the block reads it back.
__device__ __forceinline__ int claim(int *ctr, int units, int *s_unit) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *s_unit = grouped_sched::walk_claim<XCDS>(ctr, (int)blockIdx.x % XCDS, units);
    }
    __syncthreads();
    return __builtin_amdgcn_readfirstlane(*s_unit);
}

template <typename T>
struct Elem;
template <>
struct Elem<bf16> {
    using raw = __bf16;
};
template <>
struct Elem<half> {
    using raw = _Float16;
};

inline void hk_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

// Host side: one launch family per layout (grouped16_{tn,nn,nt}.cpp).
struct Launch {
    const Table *tables;
    const int *experts_per_table;
    int num_tables;
    int N, K;
    float alpha, beta;
    int grid_cap;
    int *ctr;
    hipStream_t stream;
};

bool launch_tn(const Launch &l, int in_dtype, int out_dtype, bool bias, bool acc);
bool launch_nn(const Launch &l, int in_dtype, int out_dtype, bool acc);
bool launch_nt(const Launch &l, int in_dtype, int out_dtype, bool bgrad, bool acc);

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
