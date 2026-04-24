/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file fused_attn_small_seq.h
 *  \brief Small-seq (varlen) attention for ROCm: seq_q=1, max_seqlen_kv<=16, THD only.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALL_SEQ_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALL_SEQ_H_

#include <transformer_engine/transformer_engine.h>
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {

bool is_small_seq_attn_supported(
    NVTEDType q_dtype,
    NVTEDType kv_dtype,
    NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type,
    float dropout,
    size_t num_attn_heads,
    size_t num_gqa_groups,
    size_t max_seqlen_q,
    size_t max_seqlen_kv,
    size_t head_dim_qk,
    size_t head_dim_v,
    int64_t window_size_left,
    int64_t window_size_right);

/** Workspace size in bytes for small-seq backward path */
size_t fused_attn_small_seq_bwd_workspace_size(size_t b,
                                              size_t h_q,
                                              size_t max_seqlen_kv,
                                              DType dtype);

/** Forward: Q,K,V -> O; attention weights written to attn_weights_buffer (same as output_S).
 *  attn_weights_buffer is also used as internal workspace (scores then overwritten by attn
 *  weights). */
void fused_attn_small_seq_fwd(size_t b,
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
 *  (output_S). workspace must be at least fused_attn_small_seq_bwd_workspace_size.
 *  max_seqlen_kv is the runtime max KV length when invoked from nvte_fused_attn_small_seq_bwd. */
void fused_attn_small_seq_bwd(size_t b,
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

#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALL_SEQ_H_
