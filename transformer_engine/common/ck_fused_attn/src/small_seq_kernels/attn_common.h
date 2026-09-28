/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <map>
#include <string>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                                    \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while(0)

// All named small-seq kernel entities live in this namespace to avoid clashing with
// similarly-named symbols in other fused-attn backends (e.g. FmhaKernelConfig).
namespace small_seq_kernels {

// ---------------------------------------------------------------------------
// Causal mask type
// ---------------------------------------------------------------------------

enum class CausalMaskType
{
    DISABLE      = 0,
    TOP_LEFT     = 1,
    BOTTOM_RIGHT = 2
};

// inline to avoid ODR violation across multiple translation units (C++17)
inline std::map<CausalMaskType, std::string> CausalMaskTypeName = {
    {CausalMaskType::DISABLE, "DISABLE"},
    {CausalMaskType::TOP_LEFT, "TOP_LEFT"},
    {CausalMaskType::BOTTOM_RIGHT, "BOTTOM_RIGHT"}};

// ---------------------------------------------------------------------------
// Kernel configuration struct
//
// Template parameters encode the static layout dimensions used by all kernels.
// Runtime variability (actual batch size and actual Q/KV lengths per batch) is
// handled at runtime: batch is passed as a kernel-launch argument (mapped to the
// grid z-dimension) and per-batch sequence lengths come from cu_seqlens.
// ---------------------------------------------------------------------------

template <int HEAD_NUM,
          int MAX_SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          CausalMaskType MAKS_TYPE = CausalMaskType::DISABLE,
          int MAX_SEQ_Q            = 1>
struct FmhaKernelConfig
{
    static constexpr int head_num                  = HEAD_NUM;
    static constexpr int max_seq_q                 = MAX_SEQ_Q;
    // Backward compat alias for scalar fwd/bwd kernels (hardcoded seq_q=1)
    static constexpr int seq_q                     = 1;
    static constexpr int max_seq_kv                = MAX_SEQ_KV;
    static constexpr int head_dim                  = HEAD_DIM;
    static constexpr int step2_block_size          = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask      = ENABLE_DROPOUT_MASK;
    static constexpr enum CausalMaskType mask_type = MAKS_TYPE;
};

// ---------------------------------------------------------------------------
// Device helpers: BSHD uniform layout vs THD varlen (cu_seqlens)
//
// When uniform_seq_len > 0 the kernel runs in dense BSHD mode: every batch
// has the same sequence length and offsets are batch_idx * uniform_seq_len.
// When uniform_seq_len == 0, lengths/offsets come from cu_seqlens_*.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int get_seq_len(int batch_idx,
                                           int uniform_seq_len,
                                           const int* cu_seqlens) {
    if(uniform_seq_len > 0) return uniform_seq_len;
    return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx];
}

__device__ __forceinline__ int get_token_offset(int batch_idx,
                                               int uniform_seq_len,
                                               const int* cu_seqlens_padded) {
    if(uniform_seq_len > 0) return batch_idx * uniform_seq_len;
    return cu_seqlens_padded[batch_idx];
}


}  // namespace small_seq_kernels
