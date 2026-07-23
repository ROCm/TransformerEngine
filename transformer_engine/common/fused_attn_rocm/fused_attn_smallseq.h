/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_

// size_t (used throughout the declarations below) comes from transformer_engine/fused_attn.h,
// matching the sibling headers in this module (fused_attn_ck.h, fused_attn_aotriton.h, utils.h).
#include <transformer_engine/fused_attn.h>

namespace transformer_engine {
namespace fused_attn_rocm {

constexpr size_t kSmallSeqMaxSeqlen = 17;

/** Static config only (no packed max seqlen as proof of per-segment lengths). */
bool small_seq_static_config_ok(NVTEDType q_dtype,
                                NVTEDType kv_dtype,
                                NVTE_Bias_Type bias_type,
                                float dropout,
                                size_t head_dim_qk,
                                size_t head_dim_v,
                                size_t num_attn_heads,
                                size_t num_gqa_groups,
                                NVTE_Mask_Type mask_type);

/** Same checks as small_seq_static_config_ok(), but returns the first failing condition as a
 *  human-readable string (or nullptr if the static config is eligible). Used for
 *  NVTE_LOG_CK_CONFIG diagnostics so a disabled small-seq path reports *why* it was skipped. */
const char* small_seq_static_config_reason(NVTEDType q_dtype,
                                           NVTEDType kv_dtype,
                                           NVTE_Bias_Type bias_type,
                                           float dropout,
                                           size_t head_dim_qk,
                                           size_t head_dim_v,
                                           size_t num_attn_heads,
                                           size_t num_gqa_groups,
                                           NVTE_Mask_Type mask_type);

bool is_runtime_small_seq_eligible(size_t runtime_max_seqlen_q, size_t runtime_max_seqlen_kv);

size_t small_seq_fwd_extra_workspace_bytes(size_t max_tokens_q);
size_t small_seq_bwd_extra_workspace_bytes();

bool is_nvte_ck_small_seq_enabled();

/** nullptr if the arch + NVTE_FUSED_ATTN_CK_SMALLSEQ env gate passes, otherwise the reason it
 *  fails (arch not gfx942/gfx950, or the env var not set to "1"). Mirrors
 *  is_nvte_ck_small_seq_enabled() for NVTE_LOG_CK_CONFIG diagnostics. */
const char* small_seq_enable_reason();

bool fused_attn_smallseq_fwd(size_t batch_size,
                             size_t num_heads,
                             size_t head_dim_qk,
                             size_t max_tokens_q,
                             size_t max_tokens_kv,
                             float attn_scale,
                             const void* dev_ptr_q,
                             const void* dev_ptr_k,
                             const void* dev_ptr_v,
                             void* dev_ptr_o,
                             void* dev_ptr_softmax_lse,
                             const void* dev_ptr_cu_seqlens_q,
                             const void* dev_ptr_cu_seqlens_q_padded,
                             const void* dev_ptr_cu_seqlens_kv,
                             const void* dev_ptr_cu_seqlens_kv_padded,
                             const void* dev_ptr_padded_q_to_batch,
                             NVTEDType dtype,
                             cudaStream_t stream);

bool fused_attn_smallseq_bwd(size_t batch_size,
                             size_t num_heads,
                             size_t head_dim_qk,
                             size_t max_tokens_q,
                             size_t max_tokens_kv,
                             float attn_scale,
                             const void* dev_ptr_q,
                             const void* dev_ptr_k,
                             const void* dev_ptr_v,
                             const void* dev_ptr_do,
                             const void* dev_ptr_softmax_lse,
                             void* dev_ptr_dq,
                             void* dev_ptr_dk,
                             void* dev_ptr_dv,
                             const void* dev_ptr_cu_seqlens_q,
                             const void* dev_ptr_cu_seqlens_q_padded,
                             const void* dev_ptr_cu_seqlens_kv,
                             const void* dev_ptr_cu_seqlens_kv_padded,
                             NVTEDType dtype,
                             cudaStream_t stream);

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif
