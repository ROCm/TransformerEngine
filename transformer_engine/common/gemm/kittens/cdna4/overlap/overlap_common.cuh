/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include "kittens.cuh"
#include <vector>

namespace hk_overlap {

constexpr int BLOCK_ROW = 256;
constexpr int BLOCK_COL = 256;
constexpr int K_STEP    = 64;

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

constexpr int GRID_CAP = 256;

struct XcdBuckets {
    int off[NUM_XCDS_AFF];
    int cnt[NUM_XCDS_AFF];
};

__host__ __device__ __forceinline__
int tile_xcd(int chunk_id) { return chunk_id & (NUM_XCDS_AFF - 1); }

// Steal order: own bucket, then the local chunk, then the other XCDs.
__device__ __forceinline__
int steal_bucket_tile(const XcdBuckets &buckets, int *__restrict__ bucket_ctr, int my_pe,
                      int num_tiles) {
    const int b0 = (int)blockIdx.x % NUM_XCDS_AFF;
    for (int s = 0; s <= NUM_XCDS_AFF; s++) {
        int bb;
        if (s == 0) {
            bb = b0;
        } else if (s == 1) {
            bb = my_pe;
        } else {
            bb = (b0 + s - 1) & (NUM_XCDS_AFF - 1);
        }
        if (buckets.cnt[bb] == 0) continue;
        if (__hip_atomic_load(&bucket_ctr[bb], __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) >= buckets.cnt[bb]) continue;
        const int idx = atomicAdd(&bucket_ctr[bb], 1);
        if (idx < buckets.cnt[bb]) return buckets.off[bb] + idx;
    }
    return num_tiles;
}

template <typename TD>
static std::vector<TD> bucketize_by_xcd(const std::vector<TD> &q, XcdBuckets &bk) {
    std::vector<TD> out;
    out.reserve(q.size());
    for (int b = 0; b < NUM_XCDS_AFF; b++) {
        bk.off[b] = (int)out.size();
        for (size_t i = 0; i < q.size(); i++) {
            if (tile_xcd(q[i].chunk_id) == b) {
                out.push_back(q[i]);
            }
        }
        bk.cnt[b] = (int)out.size() - bk.off[b];
    }
    return out;
}

__device__ __forceinline__
__hip_bfloat16 bf16_add(__hip_bfloat16 x, __hip_bfloat16 y) {
    return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
}

#ifndef GATH_WG
#define GATH_WG 8
#endif
#define AG_PUBLISH(p) __hip_atomic_fetch_add((p), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT)
#define AG_SPIN(p)    __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define AG_ACQUIRE(p) ((void)__hip_atomic_load((p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT))

}  // namespace hk_overlap
