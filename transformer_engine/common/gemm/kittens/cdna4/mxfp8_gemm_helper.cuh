/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "kittens.cuh"

namespace te_kittens::cdna4::mxfp8 {

// Packs raw scales into the lane-native layout the GEMM kernels read: one block per 256-row scale
// tile packs its words into shared, then NG*64 lane threads emit one fp8e8m0_4 per (group, lane).
// A: STEP=64,NG=4 (256 words/tile); B: STEP=32,NG=8 (512 words, hi/lo tile pair).
// COLWISE=false: raw uint8 [dim, K/32] row-major; COLWISE=true: [K/32, dim] col-major.
// FUSED=1: one launch for all experts (shared k_iters); output at ln + expert * expert_stride.
// FUSED=2: per-expert k_iters; output at ln + output_offsets[expert]. Grid.x is sized for the
//          largest expert, so blocks past a given expert's tile count exit early.
template<bool COLWISE, int STEP, int NG, int FUSED = 0>
__global__ void pack_scales_kernel([[maybe_unused]] const uint8_t *__restrict__ scales,
    uint32_t *__restrict__ ln, int dim, int scale_K, int k_iters, int tiles_per_col,
    [[maybe_unused]] const uint8_t *const *__restrict__ scale_ptrs = nullptr,
    [[maybe_unused]] int expert_stride = 0,
    [[maybe_unused]] const int *__restrict__ k_iters_arr = nullptr,
    [[maybe_unused]] const int *__restrict__ output_offsets = nullptr) {

    constexpr int TILE_WORDS = 256;
    constexpr int PAD_WORDS  = (NG - 1) * STEP + 64; // covers OOB pack_scales read
    __shared__ uint32_t tile[PAD_WORDS];

    const uint8_t *my_scales;
    uint32_t *my_ln;
    int my_k_iters, my_scale_K;

    if constexpr (FUSED == 1) {
        int expert_id = blockIdx.y;
        my_scales  = scale_ptrs[expert_id];
        my_ln      = ln + (size_t)expert_id * expert_stride;
        my_k_iters = k_iters;
        my_scale_K = scale_K;
    } else if constexpr (FUSED == 2) {
        int expert_id = blockIdx.y;
        my_scales  = scale_ptrs[expert_id];
        my_ln      = ln + output_offsets[expert_id];
        my_k_iters = k_iters_arr[expert_id];
        my_scale_K = my_k_iters * 4;
    } else {
        my_scales  = scales;
        my_ln      = ln;
        my_k_iters = k_iters;
        my_scale_K = scale_K;
    }

    int tile_id = blockIdx.x;
    if (tile_id >= my_k_iters * tiles_per_col) return;

    int k_iter  = tile_id / tiles_per_col;
    int cblk    = tile_id % tiles_per_col;
    int kb_base = k_iter * 4;
    int row0    = cblk * TILE_WORDS;

    for (int i = threadIdx.x; i < PAD_WORDS; i += blockDim.x) {
        uint32_t p = 0;
        if (i < TILE_WORDS) {
            int row = row0 + i;
            if constexpr (COLWISE) {
                int base = kb_base * dim + row;
                p  =  (uint32_t)my_scales[base]              | ((uint32_t)my_scales[base +     dim] << 8)
                   | ((uint32_t)my_scales[base + 2 * dim] << 16) | ((uint32_t)my_scales[base + 3 * dim] << 24);
            } else {
                __builtin_memcpy(&p, &my_scales[(size_t)row * my_scale_K + kb_base], 4);
            }
        }
        tile[i] = p; // OOB tail (i>=256) zero-filled
    }
    __syncthreads();

    int tid = threadIdx.x, lane = tid % 64, grp = tid / 64;
    kittens::fp8e8m0_4 out = kittens::pack_scales((const kittens::fp8e8m0 *)tile, grp * STEP);
    my_ln[((size_t)tile_id * NG + grp) * 64 + lane] = out;
}

template<bool COLWISE, int STEP, int NG>
void launch_pack_scales(const uint8_t *scales, uint32_t *ln, int dim, int scale_K, int k_iters, hipStream_t stream) {
    int tiles_per_col = dim / 256;
    pack_scales_kernel<COLWISE, STEP, NG><<<k_iters * tiles_per_col, NG * 64, 0, stream>>>(
        scales, ln, dim, scale_K, k_iters, tiles_per_col);
}

}  // namespace te_kittens::cdna4::mxfp8
