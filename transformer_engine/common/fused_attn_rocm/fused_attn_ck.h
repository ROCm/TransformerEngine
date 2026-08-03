/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file fused_attn_ck.h
 *  \brief Enums and functions for fused attention ck backend.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_CK_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_CK_H_
#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {
// check the fused attn config to see whether it's ck backend supported
bool is_ck_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  NVTE_Softmax_Type softmax_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk,
  size_t head_dim_v,
  int64_t window_size_left,
  int64_t window_size_right);

constexpr size_t kSmallSeqMaxSeqlen = 17;

// Small-seq eligibility is split into static and runtime checks:
// - Static: config known at call time (dtype, head dims, heads, bias, dropout, mask).
//   Used during workspace sizing to reserve THD probe buffers and to gate the small-seq path
//   before any device data is read.
// - Runtime: actual per-batch max seqlen for THD/ragged inputs (from cu_seqlens on device).
//   Even when static config matches, individual batches may exceed kSmallSeqMaxSeqlen; probe
//   at execute time and fall back to regular CK when out of range.
bool is_small_seq_supported_static(DType dtype,
                                   NVTE_Bias_Type bias_type,
                                   NVTE_Mask_Type mask_type,
                                   float dropout,
                                   size_t head_dim_qk,
                                   size_t head_dim_v,
                                   size_t num_attn_heads,
                                   size_t num_gqa_groups);

bool is_small_seq_supported_runtime(size_t runtime_max_seqlen_q,
                                    size_t runtime_max_seqlen_kv);
}  // namespace fused_attn_rocm

void fused_attn_ck_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_Bias, 
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream);

void fused_attn_ck_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  Tensor* output_dBias,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream);
}  // namespace transformer_engine
#endif //#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_CK_H_
