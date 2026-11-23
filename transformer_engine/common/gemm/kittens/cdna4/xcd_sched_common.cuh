/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

#include "hip/hip_runtime.h"

namespace te_kittens::cdna4 {

constexpr int SCHED_ROUNDS = 2;
constexpr int NUM_XCDS_AFF = 8;
constexpr int MAX_TAB_EXPERTS = 128;

struct XcdBuckets {
    int off[NUM_XCDS_AFF];
    int cnt[NUM_XCDS_AFF];
};

template <typename T>
__device__ __forceinline__ int hk_upper_bound(const T *arr, int n, T val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (arr[mid] <= val) { lo = mid; } else { hi = mid - 1; }
    }
    return lo;
}

}  // namespace te_kittens::cdna4
