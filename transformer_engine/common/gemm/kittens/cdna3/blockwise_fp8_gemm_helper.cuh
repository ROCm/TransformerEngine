/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <type_traits>
#include "kittens.cuh"
#include "../../../util/math.h"
using namespace te_kittens::blockwise;  // NOLINT(build/namespaces)

typedef int int32x4_lds_t __attribute__((ext_vector_type(4)));
struct __attribute__((packed)) buf_res { const void *ptr; uint32_t range; uint32_t config; };
__device__ inline int32x4_lds_t make_buf_res(const void *ptr, uint32_t size) {
    buf_res r{ptr, size, 0x00020000u};
    return __builtin_bit_cast(int32x4_lds_t, r);
}
extern "C" __device__ __uint128_t
llvm_amdgcn_raw_buffer_load_b128(int32x4_lds_t rsrc, int voffset, int soffset,
                                 int aux) __asm("llvm.amdgcn.raw.buffer.load.v4f32");
extern "C" __device__ float
llvm_amdgcn_s_buffer_load_f32(int32x4_lds_t rsrc, int offset,
                              int cachepolicy) __asm("llvm.amdgcn.s.buffer.load.f32");
extern "C" __device__ float
llvm_amdgcn_raw_buffer_load_f32(int32x4_lds_t rsrc, int voffset, int soffset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");

template <bool A_E4M3, bool B_E4M3>
__device__ __forceinline__ void mfma_fmt(float2 (&D)[2], const kittens::fp8e4m3_4 (&A)[2],
                                         const kittens::fp8e4m3_4 (&B)[2], const float2 (&C)[2]) {
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;
    if constexpr (!A_E4M3 && B_E4M3) {
        *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(*(long*)A, *(long*)B, *(float4_t*)C, 0, 0, 0);
    } else if constexpr (A_E4M3 && !B_E4M3) {
        *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(*(long*)A, *(long*)B, *(float4_t*)C, 0, 0, 0);
    } else {
        *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(*(long*)A, *(long*)B, *(float4_t*)C, 0, 0, 0);
    }
}

template <bool A_E4M3 = true, bool B_E4M3 = true, typename RT_C, typename RT_A, typename RT_B>
__device__ __forceinline__ void mma_ABt(RT_C &acc, const RT_A &a, const RT_B &b) {
    if constexpr (A_E4M3 && B_E4M3) {
        kittens::mma_ABt(acc, a, b, acc);
    } else {
        #pragma unroll
        for (int n = 0; n < acc.height; n++)
            #pragma unroll
            for (int m = 0; m < acc.width; m++)
                mfma_fmt<A_E4M3, B_E4M3>(acc.tiles[n][m].data, a.tiles[n][0].data,
                                         b.tiles[m][0].data, acc.tiles[n][m].data);
    }
}


template <int HEIGHT>
__device__ inline void load_scale_global_reg(float (&sa_reg)[HEIGHT * 4], const float *sa_base,
                                             int local_m_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    int32x4_lds_t srsrc = make_buf_res((const void*)sa_base, range_bytes);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, m0 * 4, 0, 0);
        *reinterpret_cast<float4*>(&sa_reg[i * 4]) = *reinterpret_cast<float4*>(&raw);
    }
}

template <int WIDTH>
__device__ inline void load_scaleB_global_reg(float (&sb_reg)[WIDTH], const float *sb_base,
                                              int local_n_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int col_l = lane % 16;
    int32x4_lds_t srsrc = make_buf_res((const void*)sb_base, range_bytes);
    #pragma unroll
    for (int j = 0; j < WIDTH; j++) {
        const int n0 = local_n_base + j * 16 + col_l;
        sb_reg[j] = llvm_amdgcn_raw_buffer_load_f32(srsrc, n0 * 4, 0, 0);
    }
}

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

template <typename OType>
__device__ inline float rtne_cast_roundtrip(float v) {
    if constexpr (std::is_same_v<OType, float>) {
        return v;
    } else {
        return static_cast<float>(kittens::base_types::convertor<OType, float>::convert(rtne_bias(v)));
    }
}

template <typename OType>
__device__ inline OType convert_out(float v) {
    if constexpr (std::is_same_v<OType, kittens::bf16>) {
        return kittens::base_types::convertor<OType, float>::convert(rtne_bias(v));
    } else {
        return kittens::base_types::convertor<OType, float>::convert(v);
    }
}

template <typename OType, typename AccType>
__device__ inline void store_output(OType *c_ptr, const AccType &Cacc,
                                    int Rtile, int Ctile, int M, int N) {
    const int lane = kittens::laneid();
    const int m_base = Rtile * AccType::rows + 4 * (lane / 16);
    const int n_base = Ctile * AccType::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_base + j * 16;
            if (col >= N) continue;
            if (m0 + 0 < M) c_ptr[(m0 + 0) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[0].x);
            if (m0 + 1 < M) c_ptr[(m0 + 1) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[0].y);
            if (m0 + 2 < M) c_ptr[(m0 + 2) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[1].x);
            if (m0 + 3 < M) c_ptr[(m0 + 3) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[1].y);
        }
    }
}

template <typename OType, GemmEpilogue EPILOGUE, typename AccType>
__device__ inline void apply_epilogue(
    AccType &Cacc, int Rtile, int Ctile, int M, int N,
        const void *bias, int bias_dtype,
        const void *gelu_aux, int gelu_aux_dtype,
        const OType *c_in, float beta) {
    constexpr bool HAS_BIAS = epilogue_has_bias(EPILOGUE);
    constexpr bool HAS_GELU = epilogue_has_gelu(EPILOGUE);
    constexpr bool HAS_BETA = epilogue_has_beta(EPILOGUE);
    const int lane = kittens::laneid();
    const int m_base = Rtile * AccType::rows + 4 * (lane / 16);
    const int n_base = Ctile * AccType::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_base + j * 16;
            if (col >= N) continue;
            float v[4] = {
                Cacc.tiles[i][j].data[0].x, Cacc.tiles[i][j].data[0].y,
                Cacc.tiles[i][j].data[1].x, Cacc.tiles[i][j].data[1].y,
            };
            float bias_v = 0.f;
            if constexpr (HAS_BIAS) bias_v = read_elem(bias, bias_dtype, col);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int m_g = m0 + r;
                if (m_g >= M) continue;
                float x = v[r];
                if constexpr (HAS_BIAS) x += bias_v;
                if constexpr (HAS_BETA) {
                    x = rtne_cast_roundtrip<OType>(x);
                    x += beta * static_cast<float>(c_in[m_g * N + col]);
                }
                if constexpr (HAS_GELU) {
                    x *= transformer_engine::dgelu<float, float>(
                        read_elem(gelu_aux, gelu_aux_dtype, m_g * N + col), {});
                }
                v[r] = x;
            }
            Cacc.tiles[i][j].data[0].x = v[0];
            Cacc.tiles[i][j].data[0].y = v[1];
            Cacc.tiles[i][j].data[1].x = v[2];
            Cacc.tiles[i][j].data[1].y = v[3];
        }
    }
}

template <int NUM_THREADS, typename ST, typename GL>
__device__ inline void load_tile_masked(ST &dst, const GL &src, int row_blk,
                                        int k_blk, int row_dim, int K) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total = (ST::rows * ST::cols) / elem_per_memcpy;
    const int row_stride = src.template stride<2>();
    const int row_base = row_blk * ST::rows;
    const int k_base = k_blk * ST::cols;
    kittens::coord<> uc = kittens::coord<ST>(0, 0, row_blk, k_blk).template unit_coord<2, 3>();
    T *src_ptr = (T *)&src[uc];
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int tid = threadIdx.x;
    #pragma unroll
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        const int row = idx / memcpy_per_row;
        const int col = (idx % memcpy_per_row) * elem_per_memcpy;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (row_base + row < row_dim && k_base + col < K)
            v = kittens::load_global_vec4((float4 *)(src_ptr + (row * row_stride + col)));
        kittens::store_shared_vec(dst.idx(dst_ptr, {row, col}), {v.x, v.y});
        kittens::store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {v.z, v.w});
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4], float sb) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float s0 = sa_reg[i * 4 + 0] * sb;
        const float s1 = sa_reg[i * 4 + 1] * sb;
        const float s2 = sa_reg[i * 4 + 2] * sb;
        const float s3 = sa_reg[i * 4 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d1d(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4],
    const float (&sb_reg)[AccType::width]) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float a0 = sa_reg[i * 4 + 0];
        const float a1 = sa_reg[i * 4 + 1];
        const float a2 = sa_reg[i * 4 + 2];
        const float a3 = sa_reg[i * 4 + 3];
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const float sb = sb_reg[j];
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * (a0 * sb);
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * (a1 * sb);
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * (a2 * sb);
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * (a3 * sb);
        }
    }
}
