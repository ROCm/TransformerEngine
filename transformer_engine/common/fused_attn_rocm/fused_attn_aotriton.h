/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/


/*! \file fused_attn_aotriton.h
 *  \brief Enums and functions for fused attention aotriton backend.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_AOTRITON_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_AOTRITON_H_
#include "transformer_engine/fused_attn.h"
#include "../common.h"

namespace transformer_engine {
namespace fused_attn_rocm {
// check the fused attn config to see whether it's aotriton backend supported
bool is_aotriton_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim, 
  int64_t window_size_left, 
  int64_t window_size_right);
}  // namespace fused_attn_rocm

void fused_attn_aotriton_fwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_QKV,
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream);

void fused_attn_aotriton_bwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_QKV, const Tensor* input_O, const Tensor* input_dO,
  const Tensor* output_S,
  Tensor* output_dQKV,
  const Tensor* input_cu_seqlens,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream);

void fused_attn_aotriton_fwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_KV,
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream);

void fused_attn_aotriton_bwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_KV, const Tensor* input_O, const Tensor* input_dO,
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dKV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream);

void fused_attn_aotriton_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream);

void fused_attn_aotriton_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO, 
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream);
}  // namespace transformer_engine
#endif //#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_AOTRITON_H_
