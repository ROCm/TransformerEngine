/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

struct KittensAgGemmArgs {
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
    size_t scale_base_offset;
    size_t scale_chunk_bytes;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
    void *gather_dst;     // Bulk all-gather only
};

bool kittens_fused_ag_gemm_supported(int sm_arch);

// Drops the cached work-queue plans and peer base pointers
void kittens_comm_gemm_reset();

bool kittens_fused_ag_gemm_bf16(const KittensAgGemmArgs &args);

bool kittens_fused_ag_gemm_mxfp8(const KittensAgGemmArgs &args);

bool kittens_bulk_ag_gemm_bf16(const KittensAgGemmArgs &args);
