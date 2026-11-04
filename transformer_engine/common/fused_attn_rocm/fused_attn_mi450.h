/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/*! \file fused_attn_mi450.h
 *  \brief Fused attention backend for the gfx1250 (MI450) kernels in 3rdparty/te-mi450-attn-kernels.
 *
 *  The kernels are reached through the dispatcher of that repo's attention library (the C ABI of kernels/attn_abi.h),
 *  which runs each config on the first of its backends that supports it.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_MI450_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_MI450_H_
#include <string>
#include "transformer_engine/fused_attn.h"
#include "../common.h"

namespace transformer_engine {
namespace fused_attn_rocm {
// check the fused attn config to see whether the mi450 backend supports it; on false, `reason` (if given)
// receives why
bool is_mi450_backend_supported(
  bool is_training,
  bool deterministic,
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
  int64_t window_size_right,
  std::string* reason = nullptr);
}  // namespace fused_attn_rocm

void fused_attn_mi450_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  NVTE_QKV_Layout qkv_layout,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
  Tensor* output_O, NVTETensorPack* Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream);

void fused_attn_mi450_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  bool deterministic,
  NVTE_QKV_Layout qkv_layout,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
  const Tensor* input_O, const Tensor* input_dO, const Tensor* input_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  Tensor* workspace,
  cudaStream_t stream);
}  // namespace transformer_engine
#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_ROCM_FUSED_ATTN_MI450_H_
