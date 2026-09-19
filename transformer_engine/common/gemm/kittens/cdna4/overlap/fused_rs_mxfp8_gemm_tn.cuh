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

namespace hk_mxfp8_rs_tn {

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

// RS_MAX_TP, the RS_SENT_* poison, RsPeers and pull_reduce_all_sent come from hk_overlap; this is
// the device-side copy of the poison, mirroring the bf16 RS path.
__device__ __constant__ unsigned int rs_sent_dw_device = RS_SENT_DW;

using G = kittens::group<NUM_WARPS>;

// The MXFP8 K step, 128 rather than hk_overlap::K_STEP's bf16 64.
constexpr int BLOCK_K = 128;

// Scale tile shared by mxfp8 kernels; one fp8e8m0_4 per (group, lane)
using ST_Scale = kittens::st<kittens::fp8e8m0, 16, 64, kittens::st_16x64_s>;

// Reads the pre-packed lane-native scale for group lg on this lane
__device__ __forceinline__ kittens::fp8e8m0_4 lane_rd(const ST_Scale &s, int lg) {
    return reinterpret_cast<const uint32_t *>(s.data)[lg * 64 + kittens::laneid()];
}

// C is [N_TOTAL, M], the orientation mxfp8_gemm.cpp writes, so the col_l accumulators are
// transposed before the store. Identical to the AG TN epilogue; the RS path just aims it at the
// staging buffer instead of the caller's output.
template<typename RT_C, typename RT_C_T, typename OutGL>
__device__ __forceinline__ void gemm_epilogue(
    RT_C &cA, RT_C &cB, RT_C &cC, RT_C &cD,
    const OutGL &C, int block_row, int block_col, int warp_m, int warp_n) {

    auto oc = [&](int n_off, int m_off) {
        return kittens::coord<RT_C_T>{0, 0, block_col * WARPS_COL * 2 + n_off,
                                            block_row * WARPS_ROW * 2 + m_off};
    };

    RT_C_T oA, oB, oC, oD;
    kittens::transpose(oA, cA); kittens::transpose(oB, cB);
    kittens::transpose(oC, cC); kittens::transpose(oD, cD);

    kittens::store(C, oA, oc(warp_n, warp_m));
    kittens::store(C, oB, oc(WARPS_COL + warp_n, warp_m));
    kittens::store(C, oC, oc(warp_n, WARPS_ROW + warp_m));
    kittens::store(C, oD, oc(WARPS_COL + warp_n, WARPS_ROW + warp_m));
}

struct CdWalk {
    int q, tf, dq, dr;   // tf indexes the free axis: M, now that the bands run along N
};

__host__ __device__ __forceinline__
CdWalk cd_walk_init(int pos, int stride, int tiles_free) {
    CdWalk w;
    w.q  = pos / tiles_free;
    w.tf = pos - w.q * tiles_free;
    w.dq = stride / tiles_free;
    w.dr = stride - w.dq * tiles_free;
    return w;
}

template <int TP>
__host__ __device__ __forceinline__
TileDesc cd_walk_desc(CdWalk w, int my_pe, int bands) {
    const int j   = w.q % TP;
    const int tnb = w.q / TP;
    TileDesc td;
    td.chunk_id = (j == 0) ? my_pe : ((j - 1 < my_pe) ? j - 1 : j);
    td.tile_n   = td.chunk_id * bands + tnb;
    td.tile_m   = w.tf;
    return td;
}

__host__ __device__ __forceinline__
CdWalk cd_walk_next(CdWalk w, int tiles_free) {
    w.q  += w.dq;
    w.tf += w.dr;
    if (w.tf >= tiles_free) {
        w.tf -= tiles_free;
        w.q  += 1;
    }
    return w;
}

// Grid layout: [0, ncomm) communication workgroups, the rest draining the tile queue.
template <int TP, int CBSZ, int BLGP>
__global__ __launch_bounds__(NUM_THREADS, 2)
void persistent_rs_mxfp8_gemm(const gl<fp8e4m3, 1, 1, -1, -1> A, const gl<fp8e4m3, 1, 1, -1, -1> B,
                             bf16 *__restrict__ local_stage, bf16 *__restrict__ out,
                             const gl<bf16, 1, 1, -1, -1> C_stage,
                             const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
                             const gl<fp8e8m0, -1, 1, 16, 64> scale_B_gl,
                             const TileDesc *__restrict__ work_queue, int num_tiles,
                             int *__restrict__ tile_counter, const RsPeers peers,
                             int my_pe, int ncomm,
                             int bands, int wb_group, 
                             uint64_t warn_ticks) {
    const int M       = A.rows();
    const int K       = A.cols();
    const int N_TOTAL = B.rows();
    const int k_iters = K / BLOCK_K;

    const size_t band_elems = (size_t)BLOCK_COL * M;

    #include "mxfp8_tn_prologue.inc"

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
    CdWalk hy_cd     = cd_walk_init((hy_vid >= 0) ? hy_vid : 0, (hy_ncw > 0) ? hy_ncw : 1, tiles_M);
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
            hy_cd   = cd_walk_next(hy_cd, tiles_M);
            hy_tick += hy_ncw;
            hy_left -= 1;
        } else {
            desc = work_queue[tile_idx];
        }

        int block_row = desc.tile_m;
        int block_col = desc.tile_n;

        int a_half0 = block_row * 2;
        int a_half1 = a_half0 + 1;
        int b_half0 = block_col * 2;
        int b_half1 = b_half0 + 1;

        int sa_stride = tiles_M;
        int sb_stride = tiles_N;
        int sa_batch = block_row;
        int sb_batch = block_col;

        RT_A a;
        RT_B b0, b1;
        RT_C cA, cB, cC, cD;
        kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

        int tic = 0, toc = 1;
        int tic_scales = 0, toc_scales = 1;

        G::load(Bs[tic][0], B_local, {0, 0, b_half0, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
        G::load(As[tic][0], A_local, {0, 0, a_half0, 0}, sw_A);
        G::load(Bs[tic][1], B_local, {0, 0, b_half1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
        G::load(As[tic][1], A_local, {0, 0, a_half1, 0}, sw_A);

        if (warp_m == 1) __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(4)"); // wait for tic[0] halves; tic[1] halves still in flight
        __builtin_amdgcn_s_barrier();

        G::load(As[toc][0], A_local, {0, 0, a_half0, 1}, sw_A);
        G::load(Bs[toc][0], B_local, {0, 0, b_half0, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
        G::load(Bs[toc][1], B_local, {0, 0, b_half1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for tic[1] halves; 3 toc loads + scales in flight
        __builtin_amdgcn_s_barrier();

        G::load(scale_A_smem[0], scale_A_gl, {block_row, 0, 0, 0});
        G::load(scale_B_lo[0],   scale_B_gl, {2 * block_col,     0, 0, 0});
        G::load(scale_B_hi[0],   scale_B_gl, {2 * block_col + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM before first MMA
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        #define MXFP8_AG_SKEW_CLOSE 1
        #include "../mxfp8_tn_mainloop.inc"
        #undef MXFP8_AG_SKEW_CLOSE

        // The stage is [dest_rank][band][BLOCK_COL x M]; destination `chunk_id` owns a contiguous
        // run of N tiles, so chunk_id*bands + (tile_n - chunk_id*bands) == tile_n == block_col and
        // the whole stage indexes as one [N_TOTAL, M] array -- the AG path's C, verbatim.
        gemm_epilogue<RT_C, RT_C_T>(cA, cB, cC, cD, C_stage, block_row, block_col, warp_m, warp_n);

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
    const int tiles_M = M / BLOCK_ROW;
    const int bands   = (N_total / tp_size) / BLOCK_COL;

    std::vector<TileDesc> queue;
    queue.reserve((size_t)tiles_M * bands * tp_size);

    auto emit = [&](int chunk, int tn) {
        for (int tm = 0; tm < tiles_M; tm++) {
            TileDesc td{};
            td.chunk_id = chunk;
            td.tile_m   = tm;
            td.tile_n   = chunk * bands + tn;
            queue.push_back(td);
        }
    };

    for (int tn = 0; tn < bands; tn++) {
        emit(my_pe, tn);
        for (int c = 0; c < tp_size; c++) {
            if (c != my_pe) {
                emit(c, tn);
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

template <int CBSZ, int BLGP>
static void launch_persistent_rs(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b,
                                 bf16 *d_local_stage, bf16 *d_out, uint32_t* packed_sa,
                                 uint32_t* packed_sb, TileDesc *d_queue, int num_tiles,
                                 int *d_tile_counter, RsPeers peers,
                                 int my_pe, int tp_size, const RsLaunchCfg &cfg,
                                 hipStream_t stream) {
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;
    const int bands   = (N_TOTAL / tp_size) / BLOCK_COL;
    const int k_iters = K / BLOCK_K;

    gl<fp8e4m3, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<fp8e4m3, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);

    // The staging buffer in full is exactly the [N_TOTAL, M] output the AG TN path writes.
    gl<bf16, 1, 1, -1, -1> C_stage(d_local_stage, nullptr, nullptr, (size_t)N_TOTAL, (size_t)M);

    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                                     k_iters * tiles_M, nullptr, nullptr, nullptr);

    // B scale buffer is 2 tiles for the hi/lo group split
    gl<fp8e8m0, -1, 1, 16, 64> SB_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                                     2 * k_iters * tiles_N, nullptr, nullptr, nullptr);

    const int ncomm = (tp_size - 1) * cfg.comm_wg;

    int grid = tiles_M * tiles_N + ncomm;
    if (grid > GRID_CAP) grid = GRID_CAP;
    if (grid < ncomm + 1) grid = ncomm + 1;
#define RS_LAUNCH(TPV)                                                                                     \
    persistent_rs_mxfp8_gemm<TPV, CBSZ, BLGP><<<grid, NUM_THREADS, 0, stream>>>(                            \
        A_gl, B_gl, d_local_stage, d_out, C_stage, SA_gl, SB_gl, d_queue, num_tiles, d_tile_counter,        \
        peers, my_pe, ncomm, bands, cfg.wb_group, cfg.warn_ticks)

    switch (tp_size) {
        case 8: RS_LAUNCH(8); break;
        case 4: RS_LAUNCH(4); break;
        default: break;
    }
#undef RS_LAUNCH
}

using persistent_rs_fn_t = void (*)(int, int, int, fp8e4m3 *, fp8e4m3 *, bf16 *, bf16 *, uint32_t *,
                                    uint32_t *, TileDesc *, int, int *, RsPeers, int, int,
                                    const RsLaunchCfg &, hipStream_t);

// MFMA cbsz/blgp operand format codes: 0 = e4m3, 1 = e5m2, per operand. HYBRID recipes quantize
// backward tensors e5m2 while forward ones stay e4m3, so the pair genuinely mixes. Unlike the AG
// side there is no layout dimension to dispatch on: fused RS is TN only.
static persistent_rs_fn_t get_persistent_rs_fn(KittensDType a_dt, KittensDType b_dt) {
    const int a_fp8_code = fp8_code(a_dt), b_fp8_code = fp8_code(b_dt);
    if (a_fp8_code == 0 && b_fp8_code == 0) return launch_persistent_rs<0, 0>;
    if (a_fp8_code == 0 && b_fp8_code == 1) return launch_persistent_rs<0, 1>;
    if (a_fp8_code == 1 && b_fp8_code == 0) return launch_persistent_rs<1, 0>;
    if (a_fp8_code == 1 && b_fp8_code == 1) return launch_persistent_rs<1, 1>;
    return nullptr;
}

}  // namespace hk_mxfp8_rs_tn