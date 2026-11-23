/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include "xcd_sched_common.cuh"

namespace te_kittens::cdna4::grouped_f16 {

constexpr int WALK_U  = 32;
constexpr int WALK_GM = 4;

__device__ __forceinline__ int walk_cap(int b, int units) {
    const int nu = (units + WALK_U - 1) / WALK_U;
    int cap = b < nu ? (nu - b + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF * WALK_U : 0;
    if (b < nu && (nu - 1) % NUM_XCDS_AFF == b) {
        cap -= nu * WALK_U - units;
    }
    return cap;
}

__device__ __forceinline__ int walk_unit(int b, int i) {
    return (b + NUM_XCDS_AFF * (i / WALK_U)) * WALK_U + i % WALK_U;
}

__device__ __forceinline__ int claim(int *ctr, int units, int *s_unit) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int home = (int)blockIdx.x % NUM_XCDS_AFF;
        const int ih   = atomicAdd(&ctr[home], 1);
        int unit = units;
        if (ih < walk_cap(home, units)) {
            unit = walk_unit(home, ih);
        }
#pragma unroll 1
        for (int s = 1; s < NUM_XCDS_AFF && unit == units; s++) {
            const int b   = (home + s) % NUM_XCDS_AFF;
            const int cap = walk_cap(b, units);
            if (__hip_atomic_load(&ctr[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= cap) {
                continue;
            }
            const int i = atomicAdd(&ctr[b], 1);
            if (i < cap) {
                unit = walk_unit(b, i);
            }
        }
        *s_unit = unit;
    }
    __syncthreads();
    return __builtin_amdgcn_readfirstlane(*s_unit);
}

__device__ __forceinline__ void unit_tile(int li, int w_tiles, int d_tiles, int &tw, int &td) {
    const int band  = li / (WALK_GM * w_tiles);
    const int first = band * WALK_GM;
    const int gsz   = d_tiles - first < WALK_GM ? d_tiles - first : WALK_GM;
    const int q     = li - band * WALK_GM * w_tiles;
    td = first + q % gsz;
    tw = q / gsz;
}

}  // namespace te_kittens::cdna4::grouped_f16
