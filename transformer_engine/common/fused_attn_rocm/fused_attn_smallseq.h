/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_SMALLSEQ_H_

#include <cstddef>

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

bool is_runtime_small_seq_eligible(size_t runtime_max_seqlen_q, size_t runtime_max_seqlen_kv);

bool supports_hip_small_seq(size_t num_attn_heads,
                            size_t num_gqa_groups,
                            size_t head_dim_qk,
                            size_t head_dim_v);

size_t small_seq_extra_workspace_bytes(size_t max_tokens_q);

bool is_nvte_ck_small_seq_enabled();

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
