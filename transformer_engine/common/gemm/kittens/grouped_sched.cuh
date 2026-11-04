/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/* Arch-neutral tile scheduler of the persistent grouped GEMMs (cdna4 bf16/fp16).
 * Expert e owns units unit_off[e]..unit_off[e+1]; units are dealt in blocks of WALK_U to XCDS
 * claim buckets (block i in bucket i % XCDS), a workgroup claims from its home bucket and then
 * steals from the others. Each unit is claimed exactly once and runs its whole K loop, so results
 * do not depend on the grid or the claim order. The XCDS counters are zeroed before every launch
 * (a memset node: TE's workspace is shared scratch, so no zero state survives between calls). */

#pragma once

#include "hip/hip_runtime.h"

namespace te_kittens {
namespace grouped_sched {

constexpr int WALK_U  = 32;
constexpr int WALK_GM = 4;

// Units of claim bucket b.
template <int XCDS>
__host__ __device__ __forceinline__ int walk_cap(int b, int units) {
    const int nu = (units + WALK_U - 1) / WALK_U;
    int cap = b < nu ? (nu - b + XCDS - 1) / XCDS * WALK_U : 0;
    if (b < nu && (nu - 1) % XCDS == b) {
        cap -= nu * WALK_U - units;
    }
    return cap;
}

// The unit behind slot i of bucket b.
template <int XCDS>
__host__ __device__ __forceinline__ int walk_unit(int b, int i) {
    return (b + XCDS * (i / WALK_U)) * WALK_U + i % WALK_U;
}

// Claims from the home bucket first, then steals from the others; units when all are empty.
template <int XCDS>
__device__ __forceinline__ int walk_claim(int *ctr, int home, int units) {
    const int ih = atomicAdd(&ctr[home], 1);
    if (ih < walk_cap<XCDS>(home, units)) {
        return walk_unit<XCDS>(home, ih);
    }
#pragma unroll 1
    for (int s = 1; s < XCDS; s++) {
        const int b   = (home + s) % XCDS;
        const int cap = walk_cap<XCDS>(b, units);
        if (__hip_atomic_load(&ctr[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= cap) {
            continue;
        }
        const int i = atomicAdd(&ctr[b], 1);
        if (i < cap) {
            return walk_unit<XCDS>(b, i);
        }
    }
    return units;
}

// The last e < n with off[e] <= u; MAXE is a power of two and n <= MAXE.
template <int MAXE, typename OFF>
__host__ __device__ __forceinline__ int unit_expert(const OFF &off, int n, int u) {
    int e = 0;
    for (int step = MAXE / 2; step > 0; step /= 2) {
        if (e + step < n && off[e + step] <= u) {
            e += step;
        }
    }
    return e;
}

/* Local unit li of an expert with w_tiles weight-side and d_tiles data-side tiles: bands of WALK_GM
 * data-side tiles, the data-side tile walked fastest inside a band. */
__host__ __device__ __forceinline__ void unit_tile(int li, int w_tiles, int d_tiles, int &tw, int &td) {
    const int band  = li / (WALK_GM * w_tiles);
    const int first = band * WALK_GM;
    const int gsz   = d_tiles - first < WALK_GM ? d_tiles - first : WALK_GM;
    const int q     = li - band * WALK_GM * w_tiles;
    td = first + q % gsz;
    tw = q / gsz;
}

}  // namespace grouped_sched
}  // namespace te_kittens
