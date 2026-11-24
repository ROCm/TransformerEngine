/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"

namespace te_kittens::cdna4 {

constexpr int SCHED_ROUNDS = 2;
constexpr int NUM_XCDS_AFF = 8;

struct XcdBuckets {
    int off[NUM_XCDS_AFF];
    int cnt[NUM_XCDS_AFF];
};

// Next tile of this XCD's share (t = xcd + NUM_XCDS_AFF * i), else one stolen from another XCD.
__device__ __forceinline__ int claim_xcd_tile_strided(int *bucket_ctr, int total) {
    const int x0 = (int)blockIdx.x;
#pragma unroll 1
    for (int s = 0; s < NUM_XCDS_AFF; s++) {
        const int x   = (x0 + s) & (NUM_XCDS_AFF - 1);
        const int cnt = (total - x + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF;
        if (__hip_atomic_load(&bucket_ctr[x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= cnt) {
            continue;
        }
        const int i = atomicAdd(&bucket_ctr[x], 1);
        if (i < cnt) {
            return x + NUM_XCDS_AFF * i;
        }
    }
    return total;
}

}  // namespace te_kittens::cdna4
