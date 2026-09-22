/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "kittens_common.h"

// MXFP8 row-scaled and column-scaled tensors are not interchangeable, so the NN kernel's wgrad
// needs its own column-scaled copy of the tensor the GEMM gathers row-scaled. Two ways to get it,
// priced differently and needing different host-side work:
enum class KittensAuxColwiseMode {
    Transpose = 0,   // re-encode the row-wise bytes in place (arXiv:2511.02302); no communication
    Gather    = 1,   // a second all-gather of an already column-scaled shard the host staged
};

// Runtime mode selection, read once. The Python side reads the same variable in
// transformer_engine/pytorch/module/base.py (ag_wgrad_copy_mode()) and THE TWO MUST AGREE: only
// Gather wants this rank's column-scaled shard staged into the region before the dgrad GEMM
// launches, and staging for the mode the kernel is not running leaves wgrad reading stale bytes.
//   NVTE_AG_WGRAD_COPY=transpose   (default) derive the copy in the dgrad kernel
//   NVTE_AG_WGRAD_COPY=gather      carry a second all-gather inside the dgrad kernel
inline KittensAuxColwiseMode kittens_aux_colwise_mode() {
    static const KittensAuxColwiseMode mode = [] {
        const char *v = std::getenv("NVTE_AG_WGRAD_COPY");
        return (v != nullptr && std::strcmp(v, "gather") == 0) ? KittensAuxColwiseMode::Gather
                                                               : KittensAuxColwiseMode::Transpose;
    }();
    return mode;
}

// The region holding that column-scaled copy. Nothing in this GEMM reads it -- no compute block
// waits on it -- so in Gather mode the gathered tensor's dtype and scales are the caller's
// business, the same contract the BULK path uses. It must shard exactly as the gathered tensor
// does either way.
struct KittensAuxColwiseRegion {
    KittensAuxColwiseMode mode;
    void *dst;                  // this rank's base for the column-scaled tensor
    size_t chunk_bytes;
    size_t scale_base_offset;
    size_t scale_chunk_bytes;
    // Gather mode only: its own registered region, so its own peer table, and its own signal
    // space, since the communicator that owns it never runs an overlap method of its own. Left
    // zero in Transpose mode, where nothing is communicated through the region.
    const void *peer_ub;        // peer base table for this region
    void *arrive_local;
    size_t arrive_offset;
    size_t arrive_stride;
    uint64_t arrive_value;
};

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
    KittensDType a_dtype;
    KittensDType b_dtype;
    const KittensAuxColwiseRegion *aux_colwise;   // Column-scaled wgrad copy; nullptr when absent
};

struct KittensRsGemmArgs {
    const void *A;
    const void *B;
    void *D;
    void *ub;
    const void *peer_ub;
    int peer_first;
    int peer_count;
    void *arrive_local;
    const void *arrive_peers;
    size_t arrive_offset;
    size_t arrive_stride;
    uint64_t arrive_value;
    int m, n, k;
    int rank, nranks;
    size_t shard_bytes;
    void *workspace;
    size_t workspace_size;
    hipStream_t stream;
};

bool kittens_fused_ag_gemm_supported(int sm_arch);

// Drops the cached work-queue plans and peer base pointers
void kittens_comm_gemm_reset();

bool kittens_fused_ag_gemm_bf16(const KittensAgGemmArgs &args);

bool kittens_fused_ag_gemm_mxfp8(const KittensAgGemmArgs &args);

bool kittens_bulk_ag_gemm_bf16(const KittensAgGemmArgs &args);

bool kittens_bulk_ag_gemm_mxfp8(const KittensAgGemmArgs &args);

bool kittens_bulk_rs_gemm_supported(int sm_arch);

bool kittens_bulk_rs_gemm_bf16(const KittensRsGemmArgs &args);

bool kittens_fused_rs_gemm_supported(int sm_arch);

bool kittens_fused_rs_gemm_shape_ok(int tokens, int hidden, int k, int tp_size);

bool kittens_fused_rs_gemm_bf16(const KittensRsGemmArgs &args);

size_t kittens_fused_rs_region_bytes(size_t chunk_bytes, int tp_size);
