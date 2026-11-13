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

}  // namespace te_kittens::cdna4
