/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file fused_attn_unfused_smallseq.h
 *  \brief Enums and functions for unfused attention optimized for small sequences.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_UNFUSED_SMALLSEQ_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_UNFUSED_SMALLSEQ_H_
#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {

// Check the fused attn config to see whether it's unfused_smallseq backend supported
// This backend is optimized for: seq_q=1, seq_kv<=16, THD layout, BF16/FP16
bool is_unfused_smallseq_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk, 
  size_t head_dim_v, 
  int64_t window_size_left, 
  int64_t window_size_right);

// Get runtime maximum sequence lengths for Q and KV from device pointers
// Returns a pair: (max_seqlen_q, max_seqlen_kv)
// Requires host-device synchronization
std::pair<uint64_t, uint64_t> get_runtime_max_seqlen_q_kv(
  uint64_t batch_size,
  const void* cu_seqlens_q_ptr,
  const void* cu_seqlens_kv_ptr,
  const void* cu_seqlens_q_padded_ptr,
  const void* cu_seqlens_kv_padded_ptr,
  void* workspace,
  cudaStream_t stream);

// Calculate workspace size for forward pass
// If workspace is nullptr, sets workspace_size and returns
// Otherwise, uses provided workspace
void fused_attn_unfused_smallseq_fwd_workspace_size(
  size_t b, size_t h_q, size_t max_seqlen_q, size_t max_seqlen_kv,
  DType dtype,
  void* workspace,
  size_t* workspace_size);

// Calculate workspace size for backward pass
// If workspace is nullptr, sets workspace_size and returns
// Otherwise, uses provided workspace
void fused_attn_unfused_smallseq_bwd_workspace_size(
  size_t b, size_t h_q, size_t max_seqlen_q, size_t max_seqlen_kv,
  DType dtype,
  void* workspace,
  size_t* workspace_size);

// Forward pass for unfused small sequence attention
void fused_attn_unfused_smallseq_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, 
  size_t d_qk, size_t d_v,
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

// Backward pass for unfused small sequence attention
void fused_attn_unfused_smallseq_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, 
  size_t d_qk, size_t d_v,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, 
  const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
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

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif //#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_UNFUSED_SMALLSEQ_H_
