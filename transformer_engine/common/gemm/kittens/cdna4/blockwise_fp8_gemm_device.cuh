/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <type_traits>
#include "kittens.cuh"
#include "../../../util/math.h"

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

template <typename AccType>
__device__ inline void apply_rtne_bias(AccType &acc) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++)
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            acc.tiles[i][j].data[0].x = rtne_bias(acc.tiles[i][j].data[0].x);
            acc.tiles[i][j].data[0].y = rtne_bias(acc.tiles[i][j].data[0].y);
            acc.tiles[i][j].data[1].x = rtne_bias(acc.tiles[i][j].data[1].x);
            acc.tiles[i][j].data[1].y = rtne_bias(acc.tiles[i][j].data[1].y);
        }
}

template <typename OType, typename AccType>
__device__ inline void store_masked(OType *c_ptr, const AccType &acc,
                                     int m_off, int n_off, int M, int N) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    const int col_g = lane % 16;
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_off + i * 16 + row_g;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_off + j * 16 + col_g;
            if (col >= N) continue;
            if (m0 + 0 < M) c_ptr[(m0 + 0) * N + col] = kittens::base_types::convertor<OType, float>::convert(acc.tiles[i][j].data[0].x);
            if (m0 + 1 < M) c_ptr[(m0 + 1) * N + col] = kittens::base_types::convertor<OType, float>::convert(acc.tiles[i][j].data[0].y);
            if (m0 + 2 < M) c_ptr[(m0 + 2) * N + col] = kittens::base_types::convertor<OType, float>::convert(acc.tiles[i][j].data[1].x);
            if (m0 + 3 < M) c_ptr[(m0 + 3) * N + col] = kittens::base_types::convertor<OType, float>::convert(acc.tiles[i][j].data[1].y);
        }
    }
}

__device__ inline float read_elem(const void *p, int dtype, int idx) {
    if (dtype == 6) return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(p)[idx]);
    if (dtype == 5) return __half2float(reinterpret_cast<const __half *>(p)[idx]);
    return reinterpret_cast<const float *>(p)[idx];
}

template <typename OType>
__device__ inline float rtne_cast_roundtrip(float v) {
    if constexpr (std::is_same_v<OType, float>) {
        return v;
    } else {
        return static_cast<float>(kittens::base_types::convertor<OType, float>::convert(rtne_bias(v)));
    }
}

template <typename OType, bool HAS_BIAS, bool HAS_GELU, bool HAS_BETA, typename AccType>
__device__ inline void apply_epilogue(
    AccType &acc, int m_off, int n_off, int M, int N,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const OType *c_in, float beta) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    const int col_g = lane % 16;
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_off + i * 16 + row_g;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_off + j * 16 + col_g;
            if (col >= N) continue;
            float v[4] = {
                acc.tiles[i][j].data[0].x, acc.tiles[i][j].data[0].y,
                acc.tiles[i][j].data[1].x, acc.tiles[i][j].data[1].y,
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
            acc.tiles[i][j].data[0].x = v[0];
            acc.tiles[i][j].data[0].y = v[1];
            acc.tiles[i][j].data[1].x = v[2];
            acc.tiles[i][j].data[1].y = v[3];
        }
    }
}

template <int HEIGHT>
struct RowScale {
    float v[HEIGHT][4];
};

template <typename AccType>
__device__ inline RowScale<AccType::height> load_row_scale_lds(
    const float *sa_lds, int local_m_base) {
    RowScale<AccType::height> rs;
    const int row_g = 4 * (kittens::laneid() / 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        rs.v[i][0] = sa_lds[m0 + 0];
        rs.v[i][1] = sa_lds[m0 + 1];
        rs.v[i][2] = sa_lds[m0 + 2];
        rs.v[i][3] = sa_lds[m0 + 3];
    }
    return rs;
}

template <typename AccType>
__device__ inline RowScale<AccType::height> load_row_scale(
    const float *sa_row_k, int local_m_base, int m_valid) {
    RowScale<AccType::height> rs;
    const int row_g = 4 * (kittens::laneid() / 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        const int c0 = m0 + 0 < m_valid ? m0 + 0 : m_valid - 1;
        const int c1 = m0 + 1 < m_valid ? m0 + 1 : m_valid - 1;
        const int c2 = m0 + 2 < m_valid ? m0 + 2 : m_valid - 1;
        const int c3 = m0 + 3 < m_valid ? m0 + 3 : m_valid - 1;
        rs.v[i][0] = sa_row_k[c0];
        rs.v[i][1] = sa_row_k[c1];
        rs.v[i][2] = sa_row_k[c2];
        rs.v[i][3] = sa_row_k[c3];
    }
    return rs;
}

template <typename AccType>
__device__ inline void scale_accumulate(
    AccType &acc, const AccType &partial, const RowScale<AccType::height> &rs, float sb_tile) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float s0 = rs.v[i][0] * sb_tile;
        const float s1 = rs.v[i][1] * sb_tile;
        const float s2 = rs.v[i][2] * sb_tile;
        const float s3 = rs.v[i][3] * sb_tile;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            acc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            acc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            acc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            acc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

template <int WIDTH>
struct ColScale { float v[WIDTH]; };

template <typename AccType>
__device__ inline ColScale<AccType::width> load_col_scale(
    const float *sb_col_k, int local_n_base, int n_valid) {
    ColScale<AccType::width> cs;
    const int col_g = kittens::laneid() % 16;
    #pragma unroll
    for (int j = 0; j < AccType::width; j++) {
        const int n0 = local_n_base + j * 16 + col_g;
        cs.v[j] = n0 < n_valid ? sb_col_k[n0] : 0.f;
    }
    return cs;
}

template <typename AccType>
__device__ inline void scale_accumulate_1d1d(
    AccType &acc, const AccType &partial, const RowScale<AccType::height> &rs,
    const ColScale<AccType::width> &cs) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const float sc = cs.v[j];
            acc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * (rs.v[i][0] * sc);
            acc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * (rs.v[i][1] * sc);
            acc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * (rs.v[i][2] * sc);
            acc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * (rs.v[i][3] * sc);
        }
    }
}

template <int NUM_THREADS, typename ST, typename T>
__device__ inline void load_tile_masked(ST &dst, const T *src_base, int row_stride_elems,
                                        int row_blk, int k_blk, int row_dim, int K) {
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int elem_per_half   = sizeof(float2) / sizeof(T);
    constexpr int memcpy_per_row  = ST::cols / elem_per_memcpy;
    constexpr int total           = (ST::rows * ST::cols) / elem_per_memcpy;
    const int row_base = row_blk * ST::rows;
    const int k_base   = k_blk * ST::cols;
    const T *src_ptr = src_base + (size_t)row_base * row_stride_elems + k_base;
    const uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    constexpr int sub_rows = ST::underlying_subtile_rows;
    constexpr int sub_cols = ST::underlying_subtile_cols;
    constexpr int sub_bytes = ST::underlying_subtile_bytes;
    constexpr int subs_per_row = ST::underlying_subtiles_per_row;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        const int row = idx / memcpy_per_row;
        const int col = (idx % memcpy_per_row) * elem_per_memcpy;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (row_base + row < row_dim && k_base + col < K)
            v = kittens::load_global_vec4(
                reinterpret_cast<const float4 *>(src_ptr + (size_t)row * row_stride_elems + col));
        const int sub_id = (row / sub_rows) * subs_per_row + (col / sub_cols);
        const int sub_off = sub_id * sub_bytes;
        const int r = row % sub_rows, c = col % sub_cols;
        kittens::store_shared_vec(dst_ptr + sub_off + dst.swizzle({r, c}), {v.x, v.y});
        kittens::store_shared_vec(dst_ptr + sub_off + dst.swizzle({r, c + elem_per_half}), {v.z, v.w});
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
}
