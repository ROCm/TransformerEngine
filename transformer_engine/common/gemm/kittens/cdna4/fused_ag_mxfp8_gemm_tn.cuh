/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"

namespace hk_mxfp8_ag_tn {

using namespace kittens;

constexpr int SCHED_ROUNDS = 2;
constexpr int NUM_WARPS    = 8;
constexpr int WARPS_ROW    = 2;
constexpr int WARPS_COL    = 4;
constexpr int BLOCK_ROW    = 256;
constexpr int BLOCK_COL    = 256;
constexpr int K_STEP       = 128;
constexpr int HALF_ROW     = BLOCK_ROW / 2;
constexpr int HALF_COL     = BLOCK_COL / 2;
constexpr int REG_M        = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N        = BLOCK_COL / WARPS_COL / 2;
constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

using G_group = kittens::group<NUM_WARPS>;

struct TileDesc {
    int chunk_id;
    int tile_m;
    int tile_n;
};

// Per-PE pointer to each peer's [M,K] A buffer.
struct PeerPtrs {
    fp8e4m3 *base[8];
};

static void launch_persistent(int M, int N_TOTAL, int K, fp8e4m3 *d_a, fp8e4m3 *d_b, bf16 *d_c, uint32_t* packed_sa, uint32_t* packed_sb,
                              TileDesc *d_queue, int num_tiles, int *d_tile_counter, PeerPtrs peers, unsigned int *d_arrive,
                              int my_pe, int tp_size, int gath_wg, int m_local, size_t chunk_bytes, int xcd_bucket,
                              XcdBuckets buckets, int *d_bucket_ctr, hipStream_t stream) {
    const int tiles_M         = M / BLOCK_ROW;
    const int tiles_N         = N_TOTAL / BLOCK_COL;
    const int tiles_per_chunk = m_local / BLOCK_ROW;
    const int k_iters = K / K_STEP;

    gl<fp8e4m3, 1, 1, -1, -1> A_gl(d_a, nullptr, nullptr, (size_t)M,       (size_t)K);
    gl<fp8e4m3, 1, 1, -1, -1> B_gl(d_b, nullptr, nullptr, (size_t)N_TOTAL, (size_t)K);
    gl<bf16, 1, 1, -1, -1> C_gl(d_c, nullptr, nullptr, (size_t)M,       (size_t)N_TOTAL);

    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sa)),
                                     k_iters * tiles_M, nullptr, nullptr, nullptr);

    gl<fp8e8m0, -1, 1, 16, 64> SB_gl(reinterpret_cast<kittens::fp8e8m0 *>(const_cast<uint32_t *>(packed_sb)),
                                     2 * k_iters * tiles_N, nullptr, nullptr, nullptr);

    // Gatherers are dedicated (blockIdx < NGATH) and only join the compute queue once their peer's chunk has landed.
    const int NGATH = (tp_size - 1) * gath_wg;
    static const int grid_cap = getenv("HK_GRID_CAP") ? atoi(getenv("HK_GRID_CAP")) : 256;
    int grid = tiles_M * tiles_N + NGATH;
    if (grid_cap > 0 && grid > grid_cap) grid = grid_cap;
    if (grid < NGATH) grid = NGATH;

    persistent_ag_mxfp8_gemm<<<grid, NUM_THREADS, 0, stream>>>(
        A_gl, B_gl, C_gl, SA_gl, SB_gl, d_queue, num_tiles, d_tile_counter, peers,
        d_arrive, my_pe, tp_size, gath_wg, tiles_per_chunk, chunk_bytes,
        xcd_bucket, buckets, d_bucket_ctr);
}

using persistent_fn_t = void (*)(int, int, int, fp8e4m3 *, fp8e4m3 *, bf16 *, uint32_t *, uint32_t *, TileDesc *, int, int *, PeerPtrs,
                                 unsigned int *, int, int, int, int, size_t, int,
                                 XcdBuckets, int *, hipStream_t);

static persistent_fn_t get_persistent_fn(int M, int N, int K) {
    (void)M; (void)N; (void)K;
    return launch_persistent;
}

} // namespace hk_mxfp8_ag_tn