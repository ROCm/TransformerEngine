/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file fused_attn_smallseq.hpp
 *  \brief Unfused small-seq (varlen) attention for ROCm: seq_q=1, max_seqlen_kv<=16, THD only.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {

/** Workspace size in bytes for small-seq forward path (launcher uses output_S; this is for any
 *  caller scratch, e.g. get_runtime_max_seqlen). Minimum 8 for atomic. */
size_t fused_attn_smallseq_fwd_workspace_size(size_t b,
                                             size_t h_q,
                                             size_t max_seqlen_kv,
                                             DType dtype);

/** Workspace size in bytes for small-seq backward path (grad_attn then grad_scores). */
size_t fused_attn_smallseq_bwd_workspace_size(size_t b,
                                              size_t h_q,
                                              size_t max_seqlen_kv,
                                              DType dtype);

/** Forward: Q,K,V -> O; attention weights written to attn_weights_buffer (same as output_S).
 *  attn_weights_buffer is also used as internal workspace (scores then overwritten by attn
 *  weights). No separate workspace required for the launcher; caller may use workspace for
 *  get_runtime_max_seqlen (8 bytes). */
void fused_attn_smallseq_fwd(size_t b,
                             size_t h_q,
                             size_t h_kv,
                             size_t max_seqlen_kv,
                             size_t d_qk,
                             size_t d_v,
                             bool is_training,
                             float attn_scale,
                             float dropout,
                             const void* devPtrQ,
                             const void* devPtrK,
                             const void* devPtrV,
                             void* devPtrO,
                             void* attn_weights_buffer,
                             const void* devPtrCuSeqlensKV,
                             const void* devPtrSeqOffsetsKV,
                             const void* rng_seed,
                             const void* rng_offset,
                             DType qkv_dtype,
                             void* workspace,
                             size_t* workspace_size,
                             cudaStream_t stream);

/** Backward: dO, O, attn_weights -> dQ, dK, dV. attn_weights is the buffer from forward
 *  (output_S). workspace must be at least fused_attn_smallseq_bwd_workspace_size. */
void fused_attn_smallseq_bwd(size_t b,
                             size_t h_q,
                             size_t h_kv,
                             size_t max_seqlen_kv,
                             size_t d_qk,
                             size_t d_v,
                             float attn_scale,
                             float dropout,
                             const void* devPtrQ,
                             const void* devPtrK,
                             const void* devPtrV,
                             const void* devPtrO,
                             const void* devPtrdO,
                             const void* attn_weights,
                             void* devPtrdQ,
                             void* devPtrdK,
                             void* devPtrdV,
                             const void* devPtrCuSeqlensKV,
                             const void* devPtrSeqOffsetsKV,
                             DType qkv_dtype,
                             void* workspace,
                             size_t* workspace_size,
                             cudaStream_t stream);

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_
