/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

struct KittensFusedAgGemmArgs {
    const void *A;
    void *ub;
    void *D;
    const void *scale_A;
    const void *scale_B;
    const void *peer_ub;
    int peer_first;
    int peer_count;
    void *arrive_local;
    const void *arrive_peers;
    size_t arrive_offset;
    size_t arrive_stride;
    uint64_t arrive_value;
    int m, n, k;
    bool transa;
    int rank, nranks;
    size_t chunk_bytes;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

bool kittens_fused_ag_gemm_supported(int sm_arch);

// Drops the cached work-queue plans and peer base pointers
void kittens_fused_ag_gemm_reset();

bool kittens_fused_ag_gemm_bf16(const KittensFusedAgGemmArgs &args);

bool kittens_fused_ag_gemm_mxfp8(const KittensFusedAgGemmArgs &args);
