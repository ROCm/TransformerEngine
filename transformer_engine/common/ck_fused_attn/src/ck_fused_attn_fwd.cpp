/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "ck_tile/host.hpp"
#include "mha_fwd.h"
#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{

// print the fmha traits and args when calling ck apis
void log_fwd_config(const char* func_name,
                    const std::string data_type_str,
                    const bool is_group_mode,
                    const bool has_logits_soft_cap,
                    const mask_enum mask_type,
                    const bias_enum bias_type,
                    const bool has_lse,
                    const bool has_dropout,
                    const bool is_v_rowmajor,
                    const bool do_fp8_static_quant,
                    const bool uses_fwd_v3,
                    const fmha_fwd_args& fmha_args){
  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_fused_attn_log_config = true;
  }
  if (ck_fused_attn_log_config) {
    std::cout<<std::endl<<func_name<<std::endl;

    // debug fmha_traits
    std::cout<<"fmha_traits: "<<std::endl;
    std::cout<<"hdim_q: "<<fmha_args.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_args.hdim_v<<std::endl;
    std::cout<<"data_type: "<<data_type_str<<std::endl;
    std::cout<<"is_group_mode: "<<is_group_mode<<std::endl;
    std::cout<<"is_v_rowmajor: "<<is_v_rowmajor<<std::endl;
    std::cout<<"mask_type: "<<static_cast<std::underlying_type<mask_enum>::type>(mask_type)<<std::endl;
    std::cout<<"bias_type: "<<static_cast<std::underlying_type<bias_enum>::type>(bias_type)<<std::endl;
    std::cout<<"has_logits_soft_cap: "<<has_logits_soft_cap<<std::endl;
    std::cout<<"has_lse: "<<has_lse<<std::endl;
    std::cout<<"has_dropout: "<<has_dropout<<std::endl;
    std::cout<<"do_fp8_static_quant: "<<do_fp8_static_quant<<std::endl;
    std::cout<<"uses_fwd_v3: "<<uses_fwd_v3<<std::endl;

    // debug fmha_args
    std::cout<<"fmha_args: "<<std::endl;
    std::cout<<"q_ptr: "<<fmha_args.q_ptr<<std::endl;
    std::cout<<"k_ptr: "<<fmha_args.k_ptr<<std::endl;
    std::cout<<"v_ptr: "<<fmha_args.v_ptr<<std::endl;
    std::cout<<"bias_ptr: "<<fmha_args.bias_ptr<<std::endl;
    std::cout<<"rand_val_ptr: "<<fmha_args.rand_val_ptr<<std::endl;
    std::cout<<"lse_ptr: "<<fmha_args.lse_ptr<<std::endl;
    std::cout<<"o_ptr: "<<fmha_args.o_ptr<<std::endl;
    std::cout<<"seqstart_q_ptr: "<<fmha_args.seqstart_q_ptr<<std::endl;
    std::cout<<"seqstart_k_ptr: "<<fmha_args.seqstart_k_ptr<<std::endl;
    std::cout<<"seqlen_k_ptr: "<<fmha_args.seqlen_k_ptr<<std::endl;

    std::cout<<"seqlen_q: "<<fmha_args.seqlen_q<<std::endl;
    std::cout<<"seqlen_k: "<<fmha_args.seqlen_k<<std::endl;
    std::cout<<"batch: "<<fmha_args.batch<<std::endl;
    std::cout<<"max_seqlen_q: "<<fmha_args.max_seqlen_q<<std::endl;
    std::cout<<"hdim_q: "<<fmha_args.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_args.hdim_v<<std::endl;
    std::cout<<"nhead_q: "<<fmha_args.nhead_q<<std::endl;
    std::cout<<"nhead_k: "<<fmha_args.nhead_k<<std::endl;
    std::cout<<"scale_s: "<<fmha_args.scale_s<<std::endl;
    std::cout<<"scale_p: "<<fmha_args.scale_p<<std::endl;
    std::cout<<"scale_o: "<<fmha_args.scale_o<<std::endl;
    std::cout<<"logits_soft_cap: "<<fmha_args.logits_soft_cap<<std::endl;
    std::cout<<"stride_q: "<<fmha_args.stride_q<<std::endl;
    std::cout<<"stride_k: "<<fmha_args.stride_k<<std::endl;
    std::cout<<"stride_v: "<<fmha_args.stride_v<<std::endl;
    std::cout<<"stride_bias: "<<fmha_args.stride_bias<<std::endl;
    std::cout<<"stride_randval: "<<fmha_args.stride_randval<<std::endl;
    std::cout<<"stride_o: "<<fmha_args.stride_o<<std::endl;
    std::cout<<"nhead_stride_q: "<<fmha_args.nhead_stride_q<<std::endl;
    std::cout<<"nhead_stride_k: "<<fmha_args.nhead_stride_k<<std::endl;
    std::cout<<"nhead_stride_v: "<<fmha_args.nhead_stride_v<<std::endl;
    std::cout<<"nhead_stride_bias: "<<fmha_args.nhead_stride_bias<<std::endl;
    std::cout<<"nhead_stride_randval: "<<fmha_args.nhead_stride_randval<<std::endl;
    std::cout<<"nhead_stride_lse: "<<fmha_args.nhead_stride_lse<<std::endl;
    std::cout<<"nhead_stride_o: "<<fmha_args.nhead_stride_o<<std::endl;
    std::cout<<"batch_stride_q: "<<fmha_args.batch_stride_q<<std::endl;
    std::cout<<"batch_stride_k: "<<fmha_args.batch_stride_k<<std::endl;
    std::cout<<"batch_stride_v: "<<fmha_args.batch_stride_v<<std::endl;
    std::cout<<"batch_stride_bias: "<<fmha_args.batch_stride_bias<<std::endl;
    std::cout<<"batch_stride_randval: "<<fmha_args.batch_stride_randval<<std::endl;
    std::cout<<"batch_stride_lse: "<<fmha_args.batch_stride_lse<<std::endl;
    std::cout<<"batch_stride_o: "<<fmha_args.batch_stride_o<<std::endl;
    std::cout<<"window_size_left: "<<fmha_args.window_size_left<<std::endl;
    std::cout<<"window_size_right: "<<fmha_args.window_size_right<<std::endl;
    std::cout<<"mask_type: "<<fmha_args.mask_type<<std::endl;
    std::cout<<"min_seqlen_q: "<<fmha_args.min_seqlen_q<<std::endl;
    std::cout<<"p_drop: "<<fmha_args.p_drop<<std::endl;
    std::cout<<"s_randval: "<<fmha_args.s_randval<<std::endl;
    std::cout<<"dropout_seed_ptr: "<<std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
    std::cout<<"dropout_offset_ptr: "<<std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
  }
}

hipError_t ck_attn_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v, uint64_t bias_b, uint64_t bias_h,
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
  bool uses_fwd_v3,
  hipStream_t stream){

  bool has_dropout = (is_training && dropout_probability > 0.f);
  bool has_lse = (lse_ptr != nullptr);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d_qk;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d_v;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float scale_p = 1.f;
  float scale_o = 1.f;
  float logits_soft_cap = 0.f;
  float p_drop = dropout_probability;
  bool is_group_mode = false;
  bool is_v_rowmajor = true;
  bool has_logits_soft_cap = 0.f < logits_soft_cap;
  bool do_fp8_static_quant = false;

  bias_enum bias_type;
  BiasShape bias_shape; 
  std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(attn_bias_type, b, h, bias_b, bias_h);
 
  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);
  
  ck_tile::stream_config stream_config{stream};

  std::string data_type_str = get_data_type_str(dtype);

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // bias is of shape [b, h , s_q, s_kv]
    const ck_tile::index_t stride_bias = max_seqlen_k;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_o = stride_s_o;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    const ck_tile::index_t nhead_stride_bias = (bias_shape==BiasShape::k1HSS || bias_shape==BiasShape::kBHSS) ? max_seqlen_q * max_seqlen_k: 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t nhead_stride_randval = 0;
    // softmax_lse is of shape [b, h, s_q]
    const ck_tile::index_t nhead_stride_lse = max_seqlen_q;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = stride_b_q;
    const ck_tile::index_t batch_stride_k = stride_b_k;
    const ck_tile::index_t batch_stride_v = stride_b_v;
    const ck_tile::index_t batch_stride_bias = (bias_shape==BiasShape::k11SS || bias_shape==BiasShape::k1HSS) ? 0: (bias_shape==BiasShape::kBHSS? bias_h* max_seqlen_q * max_seqlen_k: max_seqlen_q*max_seqlen_k);
    //TODO: randval never used, can we remove it
    const ck_tile::index_t batch_stride_randval = 0;
    // softmax_lse is of shape [b, h, s_q]
    const ck_tile::index_t batch_stride_lse = nhead * max_seqlen_q;
    const ck_tile::index_t batch_stride_o = stride_b_o;

    return fmha_fwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         bias_type==bias_enum::alibi? alibi_slope_ptr :bias_ptr,
                         nullptr,//rand_val_ptr
                         lse_ptr,
                         o_ptr,
                         nullptr,//cu_seqlen_q
                         nullptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         max_seqlen_q,
                         max_seqlen_k,
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         scale_p,
                         scale_o,
                         logits_soft_cap,
                         stride_q,
                         stride_k,
                         stride_v,
                         bias_type==bias_enum::alibi? 0: stride_bias, // upstream TE only requires standard (vanilla) alibi slopes
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         0, // min_seqlen_q
                         p_drop,
                         false,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();
  
  // print ck traits and args when needed
  log_fwd_config(__FUNCTION__, data_type_str, is_group_mode, has_logits_soft_cap, mask_type, bias_type, has_lse, has_dropout, is_v_rowmajor, do_fp8_static_quant, uses_fwd_v3, fmha_args);

  float average_runtime = aiter::mha_fwd(fmha_args,
                                         stream_config,
                                         data_type_str,
                                         is_group_mode,
                                         mask_type,
                                         bias_type,
                                         has_lse,
                                         uses_fwd_v3);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  return hipSuccess;
}

hipError_t ck_attn_varlen_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v,
  const void* q_ptr, 
  uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_h_v, uint64_t stride_s_v,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_kv_ptr,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* o_ptr, 
  uint64_t stride_h_o, uint64_t stride_s_o,
  void* lse_thd_ptr,
  bool uses_fwd_v3,
  hipStream_t stream){

  bool has_dropout = (is_training && dropout_probability > 0.f);
  bool has_lse = (lse_thd_ptr != nullptr);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d_qk;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d_v;
  ck_tile::index_t max_seqlen_q = s_q;

  float scale_s = scaling_factor;
  float scale_p = 1.f;
  float scale_o = 1.f;
  float logits_soft_cap = 0.f;
  float p_drop = dropout_probability;
  bool is_group_mode = true;
  bool is_v_rowmajor = true;
  bool has_logits_soft_cap = 0.f < logits_soft_cap;
  bool do_fp8_static_quant = false;

  // THD does not work with bias
 
  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);
  
  bias_enum bias_type = bias_enum::no_bias;
  ck_tile::stream_config stream_config{stream};

  std::string data_type_str = get_data_type_str(dtype);

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t stride_bias = 0;
    // randval not used
    const ck_tile::index_t stride_randval = 0;
    const ck_tile::index_t stride_o = stride_s_o;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t nhead_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t nhead_stride_randval = 0;
    // In CK group mode, softmax_lse can be of shape [h, b*s_q] with effective data in first total_seqlen_q places
    const ck_tile::index_t nhead_stride_lse = batch*max_seqlen_q;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = 0;
    const ck_tile::index_t batch_stride_k = 0;
    const ck_tile::index_t batch_stride_v = 0;
    // bias not used in THD qkv layout
    const ck_tile::index_t batch_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t batch_stride_randval = 0;
    const ck_tile::index_t batch_stride_lse = 0;
    const ck_tile::index_t batch_stride_o = 0;

    return fmha_fwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,//bias_ptr
                         nullptr,//rand_val_ptr
                         lse_thd_ptr,
                         o_ptr,
                         cu_seqlen_q_ptr,//cu_seqlen_q
                         cu_seqlen_kv_ptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         0, //seqlen_q, unused in group mode
                         0, //seqlen_kv, unused in group mode
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         scale_p,
                         scale_o,
                         logits_soft_cap,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         0, // min_seqlen_q
                         p_drop,
                         false,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();

  // print ck traits and args when needed
  log_fwd_config(__FUNCTION__, data_type_str, is_group_mode, has_logits_soft_cap, mask_type, bias_type, has_lse, has_dropout, is_v_rowmajor, do_fp8_static_quant, uses_fwd_v3, fmha_args);

  float average_runtime = aiter::mha_fwd(fmha_args,
                                         stream_config,
                                         data_type_str,
                                         is_group_mode,
                                         mask_type,
                                         bias_type,
                                         has_lse,
                                         uses_fwd_v3);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  return hipSuccess;
}

}//namespace ck_fused_attn

