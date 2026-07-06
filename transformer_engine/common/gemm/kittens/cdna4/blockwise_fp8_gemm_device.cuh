/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "kittens.cuh"

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
