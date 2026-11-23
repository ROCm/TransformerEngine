/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "kittens.cuh"
#include "../kittens_common.h"
#include "../kittens_kernel_common.cuh"
#include "grouped_f16_sched.cuh"
#include <cstdint>
#include <type_traits>

namespace te_kittens::cdna4::grouped_f16 {

constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int NUM_WARPS   = WARPS_ROW * WARPS_COL;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int BLOCK_ROW   = 256;
constexpr int BLOCK_COL   = 256;
constexpr int BLOCK_K     = 64;
constexpr int HALF_ROW    = BLOCK_ROW / 2;
constexpr int HALF_COL    = BLOCK_COL / 2;
constexpr int REG_M       = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N       = BLOCK_COL / WARPS_COL / 2;
constexpr int LDS_TILES   = 131072;

typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2_t __attribute__((ext_vector_type(2)));
typedef float f32x2_t __attribute__((ext_vector_type(2)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) unsigned char lds_u8;

template <typename T>
using raw16_t = std::conditional_t<std::is_same_v<T, kittens::bf16>, __bf16, _Float16>;

struct GroupedTab {
    const void *a[MAX_TAB_EXPERTS];
    const void *b[MAX_TAB_EXPERTS];
    void *d[MAX_TAB_EXPERTS];
    const void *bias[MAX_TAB_EXPERTS];
    int rows[MAX_TAB_EXPERTS];
    int tile_off[MAX_TAB_EXPERTS + 1];
};

__device__ __forceinline__ int opaque_tid() {
    int t = (int)threadIdx.x;
    asm volatile("" : "+v"(t));
    __builtin_assume(t >= 0 && t < NUM_THREADS);
    return t;
}

template <typename T>
__device__ __forceinline__ T *uniform_ptr(T *p) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)a);
    const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(a >> 32));
    return (T *)(((unsigned long long)hi << 32) | lo);
}

__device__ __forceinline__ kittens::i32x4 uniform_srd(const void *p, uint32_t range) {
    const kittens::i32x4 r = kittens::make_srsrc(p, range, 0);
    return kittens::i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
                          __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
}

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

template <int OFF, typename RT, typename ST>
__device__ __forceinline__ void load_frag_rows(RT &dst, const ST &src, int lane) {
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols, "register tile and shared tile must match");
    static_assert(RT::base_tile_stride == 8 && RT::base_tile_num_strides == 1, "16-bit, one 16-byte stride per lane");
    static_assert(ST::underlying_subtile_rows == RT::base_tile_rows &&
                  ST::underlying_subtile_cols == RT::base_tile_cols, "one ds_read_b128 per base tile");
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
            // The first output is early-clobber: aliasing it with the address register gives silent zero rows
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

__device__ u32x4_t buf_load_v4(kittens::i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ u32x2_t buf_load_v2(kittens::i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ void buf_store_v4(u32x4_t v, kittens::i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v4i32");
__device__ void buf_store_v2(u32x2_t v, kittens::i32x4 srd, uint32_t voff, uint32_t soff, uint32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v2i32");

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

template <typename O, bool BIAS, bool ACCUMULATE, typename BT, typename CT>
__device__ __forceinline__ void epi_t16(kittens::i32x4 srd, const CT &ct, int m0, int n0, int ld, int lane,
                                        const BT *bias) {
    static_assert(CT::height == 2 && CT::base_tile_rows == 16 && CT::base_tile_cols == 16, "16x16 CT expected");
    const int c8 = 16 * ((lane >> 4) & 1) + 8 * (lane >> 5);
    float bv[8];
    if constexpr (BIAS) {
        typedef __attribute__((address_space(1))) const u32x4_t gu4;
        const u32x4_t w = *(gu4 *)((const __attribute__((address_space(1))) BT *)bias + n0 + c8);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            bv[2 * i]     = lo16<BT>(w[i]);
            bv[2 * i + 1] = hi16<BT>(w[i]);
        }
    }
    const uint32_t lo = (uint32_t)((m0 + (lane & 15)) * ld + n0 + c8) * (uint32_t)sizeof(O);
#pragma unroll
    for (int mb = 0; mb < CT::width; mb++) {
        const float a[4] = {ct.tiles[0][mb].data[0].x, ct.tiles[0][mb].data[0].y, ct.tiles[0][mb].data[1].x,
                            ct.tiles[0][mb].data[1].y};
        const float b[4] = {ct.tiles[1][mb].data[0].x, ct.tiles[1][mb].data[0].y, ct.tiles[1][mb].data[1].x,
                            ct.tiles[1][mb].data[1].y};
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const auto r = __builtin_amdgcn_permlane16_swap(__builtin_bit_cast(uint32_t, a[i]), __builtin_bit_cast(uint32_t, b[i]), false, false);
            v[i]     = __builtin_bit_cast(float, (uint32_t)r[0]);
            v[4 + i] = __builtin_bit_cast(float, (uint32_t)r[1]);
        }
        const uint32_t off = lo + (uint32_t)(mb * 16 * ld) * (uint32_t)sizeof(O);
        if constexpr (ACCUMULATE) {
            if constexpr (std::is_same_v<O, float>) {
                const f32x4_t d0 = __builtin_bit_cast(f32x4_t, buf_load_v4(srd, off, 0, 0));
                const f32x4_t d1 = __builtin_bit_cast(f32x4_t, buf_load_v4(srd, off + 16, 0, 0));
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    v[i] += d0[i];
                    v[4 + i] += d1[i];
                }
            } else {
                const u32x4_t d0 = buf_load_v4(srd, off, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    v[2 * i] += lo16<O>(d0[i]);
                    v[2 * i + 1] += hi16<O>(d0[i]);
                }
            }
        }
        if constexpr (BIAS) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                v[i] += bv[i];
            }
        }
        if constexpr (std::is_same_v<O, float>) {
            const u32x4_t w0 = {__builtin_bit_cast(uint32_t, v[0]), __builtin_bit_cast(uint32_t, v[1]),
                                __builtin_bit_cast(uint32_t, v[2]), __builtin_bit_cast(uint32_t, v[3])};
            const u32x4_t w1 = {__builtin_bit_cast(uint32_t, v[4]), __builtin_bit_cast(uint32_t, v[5]),
                                __builtin_bit_cast(uint32_t, v[6]), __builtin_bit_cast(uint32_t, v[7])};
            buf_store_v4(w0, srd, off, 0, 0);
            buf_store_v4(w1, srd, off + 16, 0, 0);
        } else {
            const u32x4_t w0 = {pack2<O>(v[0], v[1]), pack2<O>(v[2], v[3]), pack2<O>(v[4], v[5]), pack2<O>(v[6], v[7])};
            buf_store_v4(w0, srd, off, 0, 0);
        }
    }
}

template <typename O, typename RT>
__device__ __forceinline__ void lds_put16(__attribute__((address_space(3))) uint16_t *base, const RT &src,
                                          int row_unit, int col_unit, int row_stride, int lane) {
    static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "kittens::col_l 16x16 accumulator");
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
                const uint32_t w = pack2<O>(src.tiles[i][j].data[l].x, src.tiles[i][j].data[l].y);
                p[(row + l * 2) * row_stride + col]     = (uint16_t)(w & 0xffffu);
                p[(row + l * 2 + 1) * row_stride + col] = (uint16_t)(w >> 16);
            }
        }
    }
}

template <typename RT>
__device__ __forceinline__ void lds_put32(__attribute__((address_space(3))) float *base, const RT &src, int row_unit,
                                          int col_unit, int row_stride, int lane) {
    static_assert(RT::base_tile_stride == 4 && RT::base_tile_num_strides == 1, "kittens::col_l 16x16 accumulator");
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

template <typename O, bool ACCUMULATE, typename RT>
__device__ __forceinline__ void epi_staged(unsigned char *lds, const RT &c00, const RT &c01, const RT &c10,
                                           const RT &c11, int rf0, int cf0, int lane, kittens::i32x4 srd, int ld) {
    static_assert(2 * RT::rows == HALF_ROW && 8 * RT::cols == BLOCK_COL, "tile geometry");
    __syncthreads();
    int tid_c = (int)threadIdx.x;
    asm volatile("" : "+v"(tid_c));
    if constexpr (!std::is_same_v<O, float> && !ACCUMULATE) {
        auto *st = (__attribute__((address_space(3))) uint16_t *)(lds_u8 *)lds;
        lds_put16<O>(st, c00, rf0, cf0, BLOCK_COL, lane);
        lds_put16<O>(st, c01, rf0, cf0 + WARPS_COL, BLOCK_COL, lane);
        lds_put16<O>(st, c10, rf0 + WARPS_ROW, cf0, BLOCK_COL, lane);
        lds_put16<O>(st, c11, rf0 + WARPS_ROW, cf0 + WARPS_COL, BLOCK_COL, lane);
        __syncthreads();
        const __attribute__((address_space(3))) u32x4_t *sv = (const __attribute__((address_space(3))) u32x4_t *)st;
#pragma unroll
        for (int k = 0; k < BLOCK_ROW * BLOCK_COL / 8 / NUM_THREADS; k++) {
            const int q = tid_c + k * NUM_THREADS, r = q >> 5, ch = q & 31;
            buf_store_v4(sv[q], srd, (uint32_t)((r * ld + ch * 8) * 2), 0, 0);
        }
    } else {
        auto *st = (__attribute__((address_space(3))) float *)(lds_u8 *)lds;
        const __attribute__((address_space(3))) f32x2_t *sv = (const __attribute__((address_space(3))) f32x2_t *)st;
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
                const int q = tid_c + k * NUM_THREADS, r = q >> 6, ch = q & 63;
                const f32x2_t x = sv[2 * q], y = sv[2 * q + 1];
                float v[4] = {x[0], x[1], y[0], y[1]};
                const uint32_t off = (uint32_t)(((h * HALF_ROW + r) * ld + ch * 4) * (int)sizeof(O));
                if constexpr (ACCUMULATE) {
                    if constexpr (std::is_same_v<O, float>) {
                        const f32x4_t d = __builtin_bit_cast(f32x4_t, buf_load_v4(srd, off, 0, 0));
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            v[i] += d[i];
                        }
                    } else {
                        const u32x2_t d = buf_load_v2(srd, off, 0, 0);
                        v[0] += lo16<O>(d[0]);
                        v[1] += hi16<O>(d[0]);
                        v[2] += lo16<O>(d[1]);
                        v[3] += hi16<O>(d[1]);
                    }
                }
                if constexpr (std::is_same_v<O, float>) {
                    const u32x4_t w = {__builtin_bit_cast(uint32_t, v[0]), __builtin_bit_cast(uint32_t, v[1]),
                                       __builtin_bit_cast(uint32_t, v[2]), __builtin_bit_cast(uint32_t, v[3])};
                    buf_store_v4(w, srd, off, 0, 0);
                } else {
                    const u32x2_t w = {pack2<O>(v[0], v[1]), pack2<O>(v[2], v[3])};
                    buf_store_v2(w, srd, off, 0, 0);
                }
            }
        }
    }
}

void launch_grouped_f16_tn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool bias, bool accumulate, hipStream_t stream);
void launch_grouped_f16_nn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool accumulate, hipStream_t stream);
void launch_grouped_f16_nt(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool bgrad, bool accumulate, hipStream_t stream);

}  // namespace te_kittens::cdna4::grouped_f16
