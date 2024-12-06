/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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
enum DType {
  kFloat16    = 0,  /*!< 16-bit float (E5M10) */
  kBFloat16   = 1,  /*!< 16-bit bfloat (E8M7) */
  kNumTypes         /*!< Number of supported types */
};

// keep sync with mask_enum in mask.hpp
enum MaskType {
  no_mask = 0,
  mask_top_left = 1,
  mask_bottom_right = 2,
  window_generic = 3,
};

// keep sync with bias_enum in bias.hpp
enum BiasType{
  no_bias          = 0,
  elementwise_bias = 1,
  alibi            = 2,
};


hipError_t ck_attn_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* bias_ptr,
  const void* alibi_slope_ptr,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  BiasType attn_bias_type,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  void* lse_ptr, 
  hipStream_t stream);

  
hipError_t ck_attn_bwd(  
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* bias_ptr,
  const void* alibi_slope_ptr,
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr, 
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  float scaling_factor,
  float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  BiasType attn_bias_type,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* dq_ptr, 
  uint64_t stride_b_dq, uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dq_acc_ptr,
  void* dk_expanded_ptr,
  void* dv_expanded_ptr,
  uint64_t stride_b_dkv_expanded, uint64_t stride_h_dkv_expanded, uint64_t stride_s_dkv_expanded,
  void* dk_ptr, 
  uint64_t stride_b_dk, uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_b_dv, uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* dbias_expanded_ptr,
  void* dbias_ptr,
  void* workspace_ptr,
  bool deterministic,
  bool uses_bwd_v3,
  bool is_v3_atomic_fp32,
  bool is_v3_spec,
  int how_v3_bf16_cvt,
  hipStream_t stream);

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_H
