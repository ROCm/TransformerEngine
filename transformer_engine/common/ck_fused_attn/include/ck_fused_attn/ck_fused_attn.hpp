/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_H
#define CK_FUSED_ATTN_H

#include<iostream>
#include<string>
#include<cstdint>
#include<hip/hip_runtime.h>

namespace ck_fused_attn{

// input qkv dtypes
enum class DType {
  kFloat16    = 0,  /*!< 16-bit float (E5M10) */
  kBFloat16   = 1,  /*!< 16-bit bfloat (E8M7) */
  kNumTypes         /*!< Number of supported types */
};

// keep sync with mask_enum in mask.hpp
enum class MaskType {
  no_mask = 0,
  mask_top_left = 1,
  mask_bottom_right = 2,
  window_generic = 3,
};

// keep sync with bias_enum in bias.hpp
enum class BiasType{
  no_bias          = 0,
  elementwise_bias = 1,
  alibi            = 2,
};


// Fields shared by forward and backward attention requests. Layout is inferred
// from cu_seqlen_q_ptr: non-null means group mode (THD/varlen, ignores batch
// strides and bias fields); null means batch mode (BSHD/SBHD, ignores
// cu_seqlen and max_tokens fields).
struct CKAttnCommonArgs {
  DType dtype = DType::kNumTypes;

  // Shapes
  uint64_t b = 0, h = 0, hg = 0, s_q = 0, s_kv = 0, d_qk = 0, d_v = 0;
  uint64_t bias_b = 0, bias_h = 0;        // batch mode
  uint64_t max_tokens_q = 0;              // group mode

  // Q / K / V (batch-stride is 0 in group mode)
  const void* q_ptr = nullptr;
  uint64_t stride_b_q = 0, stride_h_q = 0, stride_s_q = 0;
  const void* k_ptr = nullptr;
  uint64_t stride_b_k = 0, stride_h_k = 0, stride_s_k = 0;
  const void* v_ptr = nullptr;
  uint64_t stride_b_v = 0, stride_h_v = 0, stride_s_v = 0;

  // Bias (batch mode only)
  const void* bias_ptr = nullptr;
  const void* alibi_slope_ptr = nullptr;
  BiasType attn_bias_type = BiasType::no_bias;

  // Group-mode cu_seqlens
  const void* cu_seqlen_q_ptr = nullptr;
  const void* cu_seqlen_kv_ptr = nullptr;
  const void* cu_seqlen_q_padded_ptr = nullptr;
  const void* cu_seqlen_kv_padded_ptr = nullptr;

  // Mask + window
  MaskType attn_mask_type = MaskType::no_mask;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;

  // Dropout / RNG
  float scaling_factor = 1.f;
  float dropout_probability = 0.f;
  void* philox_seed_ptr = nullptr;
  void* philox_offset_ptr = nullptr;

  // Softmax sink (learnable / off-by-one)
  const void* sink_ptr = nullptr;
  bool has_sink = false;
  int64_t sink_size = 0;

  // O layout (o_ptr lives in derived because fwd writes it / bwd reads it)
  uint64_t stride_b_o = 0, stride_h_o = 0, stride_s_o = 0;

  // V3 ASM kernel selection (shared knob)
  int how_v3_bf16_cvt = 0;

  bool is_group_mode() const { return cu_seqlen_q_ptr != nullptr; }
};

// Forward attention request.
struct CKAttnFwdArgs : CKAttnCommonArgs {
  bool is_training = false;

  // Output (writable)
  void* o_ptr = nullptr;
  void* lse_ptr = nullptr;

  // V3 ASM kernel selection
  bool uses_fwd_v3 = false;
};

// Backward attention request.
struct CkAttnBwdArgs : CKAttnCommonArgs {
  uint64_t max_tokens_kv = 0;             // group mode

  // O / LSE / dO (forward outputs feeding the backward; read-only)
  const void* o_ptr = nullptr;
  const void* lse_ptr = nullptr;
  const void* do_ptr = nullptr;
  uint64_t stride_b_do = 0, stride_h_do = 0, stride_s_do = 0;

  // dQ
  void* dq_ptr = nullptr;
  uint64_t stride_b_dq = 0, stride_h_dq = 0, stride_s_dq = 0;
  void* dq_acc_ptr = nullptr;

  // dK / dV expanded (MQA/GQA reduction inputs; null when h==hg)
  void* dk_expanded_ptr = nullptr;
  void* dv_expanded_ptr = nullptr;
  uint64_t stride_b_dk_expanded = 0, stride_h_dk_expanded = 0, stride_s_dk_expanded = 0;
  uint64_t stride_b_dv_expanded = 0, stride_h_dv_expanded = 0, stride_s_dv_expanded = 0;

  // dK / dV (final outputs)
  void* dk_ptr = nullptr;
  uint64_t stride_b_dk = 0, stride_h_dk = 0, stride_s_dk = 0;
  void* dv_ptr = nullptr;
  uint64_t stride_b_dv = 0, stride_h_dv = 0, stride_s_dv = 0;

  // dBias (batch mode only)
  void* dbias_expanded_ptr = nullptr;
  void* dbias_ptr = nullptr;

  // Softmax sink gradient
  void* d_sink_ptr = nullptr;

  // Workspace shared with forward LSE
  void* lse_workspace_ptr = nullptr;

  // V3 ASM kernel selection
  bool deterministic = false;
  bool uses_bwd_v3 = false;
  bool is_v3_atomic_fp32 = false;
};

hipError_t ck_attn_fwd(const CKAttnFwdArgs& args, hipStream_t stream);
hipError_t ck_attn_bwd(const CkAttnBwdArgs& args, hipStream_t stream);

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_H

