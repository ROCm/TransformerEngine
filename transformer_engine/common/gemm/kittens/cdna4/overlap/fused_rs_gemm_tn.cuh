/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include "overlap_common.cuh"
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hk_rs_tn {

using namespace hk_overlap;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

using namespace kittens;

#ifndef COMM_WG
#define COMM_WG 8
#endif

// RS_MAX_TP, the RS_SENT_* poison and RsPeers come from hk_overlap; this is the device-side copy of
// the poison that comm_gemm.cpp's sentinel_pattern_agrees() reads back to confirm host/device agree.
__device__ __constant__ unsigned int rs_sent_dw_device = RS_SENT_DW;

struct CdWalk {
    int q, tn, dq, dr;
};

__host__ __device__ __forceinline__
CdWalk cd_walk_init(int pos, int stride, int tiles_N) {
    CdWalk w;
    w.q  = pos / tiles_N;
    w.tn = pos - w.q * tiles_N;
    w.dq = stride / tiles_N;
    w.dr = stride - w.dq * tiles_N;
    return w;
}

template <int TP>
__host__ __device__ __forceinline__
TileDesc cd_walk_desc(CdWalk w, int my_pe, int bands) {
    const int j  = w.q % TP;
    const int tm = w.q / TP;
    TileDesc td;
    td.chunk_id = (j == 0) ? my_pe : ((j - 1 < my_pe) ? j - 1 : j);
    td.tile_m   = td.chunk_id * bands + tm;
    td.tile_n   = w.tn;
    return td;
}

__host__ __device__ __forceinline__
CdWalk cd_walk_next(CdWalk w, int tiles_N) {
    w.q  += w.dq;
    w.tn += w.dr;
    if (w.tn >= tiles_N) {
        w.tn -= tiles_N;
        w.q  += 1;
    }
    return w;
}

// Grid layout: [0, ncomm) communication workgroups, the rest draining the tile queue.
template <int TP>
__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_rs_bf16_gemm(const gl<bf16, 1, 1, -1, -1> A, const gl<bf16, 1, 1, -1, -1> B,
                             bf16 *__restrict__ local_stage, bf16 *__restrict__ out,
                             const TileDesc *__restrict__ work_queue, int num_tiles,
                             int *__restrict__ tile_counter, const RsPeers peers,
                             int my_pe, int ncomm,
                             int bands, int tiles_N,
                             int wb_group, uint64_t warn_ticks) {
    const int M       = A.rows();
    const int K       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_tiles = K / K_STEP;
    (void)M;

    const size_t band_elems = (size_t)BLOCK_ROW * N_TOTAL;

#include "tn_prologue.inc"

    if ((int)blockIdx.x < ncomm) {
        // The comm workgroups are the reduce-scatter: they read all eight sources and write `out` once.
        pull_reduce_all_sent<TP, true>(my_pe, ncomm, bands, local_stage, peers, out, band_elems,
                                       warn_ticks);
    }

    const int hy_ncw = (int)gridDim.x - ncomm;
    const int hy_vid = (int)blockIdx.x - ncomm;
    const int hy_R   = (hy_ncw > 0) ? (num_tiles / hy_ncw) : 0;
    const int hy_S   = hy_R * hy_ncw;
    int hy_left      = (hy_vid >= 0) ? hy_R : 0;
    int hy_tick      = hy_vid;
    int wb_tick      = 0;
    CdWalk hy_cd     = cd_walk_init((hy_vid >= 0) ? hy_vid : 0, (hy_ncw > 0) ? hy_ncw : 1, tiles_N);
    while (true) {
        __shared__ int s_tile_idx;
        const bool hy_static = (hy_left > 0);
        if (threadIdx.x == 0) {
            if (hy_static) {
                s_tile_idx = hy_tick;
            } else {
                s_tile_idx = hy_S + atomicAdd(tile_counter, 1);
            }
        }
        __syncthreads();
        int tile_idx = s_tile_idx;
        if (tile_idx >= num_tiles) break;

        TileDesc desc;
        if (hy_static) {
            desc    = cd_walk_desc<TP>(hy_cd, my_pe, bands);
            hy_cd   = cd_walk_next(hy_cd, tiles_N);
            hy_tick += hy_ncw;
            hy_left -= 1;
        } else {
            desc = work_queue[tile_idx];
        }

        const int owner = desc.chunk_id;
        const int lrow  = desc.tile_m - owner * bands;

        int block_row = desc.tile_m;
        int block_col = desc.tile_n;

        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        zero(c00); zero(c01); zero(c10); zero(c11);

        int tic = 0, toc = 1;

        G_group::load(Bs[tic][0], B, {0, 0, block_col * 2,     0}, sw_B, b_srd, b_base, b_lds_00);
        G_group::load(As[tic][0], A, {0, 0, block_row * 2,     0}, sw_A, a_srd, a_base, a_lds_00);
        G_group::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds_01);
        G_group::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds_01);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        G_group::load(Bs[toc][0], B, {0, 0, block_col * 2,     1}, sw_B, b_srd, b_base, b_lds_10);
        G_group::load(As[toc][0], A, {0, 0, block_row * 2,     1}, sw_A, a_srd, a_base, a_lds_10);
        G_group::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds_11);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

#pragma unroll 1
#include "tn_mainloop.inc"

        bf16 *sbase = local_stage + ((size_t)owner * bands + lrow) * band_elems;

        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + warp_n);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + warp_n);
        int lane_epi  = kittens::laneid();
        asm volatile("" : "+v"(lane_epi));

        store_c_tile<bf16>(sbase, c00, rf0, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c01, rf0, cf1, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c10, rf1, cf0, N_TOTAL, lane_epi);
        store_c_tile<bf16>(sbase, c11, rf1, cf1, N_TOTAL, lane_epi);

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        wb_tick += 1;
        if (wb_tick >= wb_group) {
            wb_tick = 0;
            if (threadIdx.x == 0) {
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            }
        }
        __syncthreads();

    }   // end persistent loop

    if (wb_tick != 0 && threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    }
}

static std::vector<TileDesc> build_rs_work_queue(int M, int N_total, int K, int tp_size, int my_pe) {
    (void)K;
    const int tiles_N = N_total / BLOCK_COL;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    std::vector<TileDesc> queue;
    queue.reserve((size_t)tiles_N * bands * tp_size);

    auto emit = [&](int chunk, int tm) {
        for (int tn = 0; tn < tiles_N; tn++) {
            TileDesc td{};
            td.chunk_id = chunk;
            td.tile_m   = chunk * bands + tm;
            td.tile_n   = tn;
            queue.push_back(td);
        }
    };

    for (int tm = 0; tm < bands; tm++) {
        emit(my_pe, tm);
        for (int c = 0; c < tp_size; c++) {
            if (c != my_pe) {
                emit(c, tm);
            }
        }
    }
    return queue;
}

struct RsLaunchCfg {
    int comm_wg  = COMM_WG;
    int wb_group = 1;
    uint64_t warn_ticks = 0;   // 0 disables the fold's stall report
};

static void launch_persistent_rs(int M, int N_TOTAL, int K, bf16 *d_a, bf16 *d_b,
                                 bf16 *d_local_stage, bf16 *d_out, TileDesc *d_queue, int num_tiles,
                                 int *d_tile_counter, RsPeers peers,
                                 int my_pe, int tp_size, const RsLaunchCfg &cfg,
                                 hipStream_t stream) {
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;
    const int bands   = (M / tp_size) / BLOCK_ROW;

    gl<bf16, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<bf16, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);

    const int ncomm = (tp_size - 1) * cfg.comm_wg;

    int grid = tiles_M * tiles_N + ncomm;
    if (grid > GRID_CAP) grid = GRID_CAP;
    if (grid < ncomm + 1) grid = ncomm + 1;
#define RS_LAUNCH(TPV)                                                                         \
    persistent_rs_bf16_gemm<TPV><<<grid, NUM_THREADS, 0, stream>>>(                            \
        A_gl, B_gl, d_local_stage, d_out, d_queue, num_tiles, d_tile_counter, peers,           \
        my_pe, ncomm, bands, tiles_N, cfg.wb_group, cfg.warn_ticks)

    switch (tp_size) {
        case 8: RS_LAUNCH(8); break;
        case 4: RS_LAUNCH(4); break;
        default: break;
    }
#undef RS_LAUNCH
}

}  // namespace hk_rs_tn
