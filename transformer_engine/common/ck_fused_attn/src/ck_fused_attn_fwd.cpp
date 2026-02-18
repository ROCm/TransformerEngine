/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "ck_tile/host.hpp"
#include "mha_fwd.h"
#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{

// print the fmha traits and fmha_args when calling ck apis
void log_fwd_config(const char* func_name, bool has_dropout, const aiter::mha_fwd_args& fmha_args, bool ck_log_config){
  if (!ck_log_config) {
    return;
  }

  std::cout << "\n" << func_name << "\n";

  // debug fmha_traits
  std::cout<<"\nfmha_traits: \n";
  log_value("hdim_q", fmha_args.hdim_q);
  log_value("hdim_v", fmha_args.hdim_v);
  log_value("data_type", fmha_args.data_type);
  log_value("is_group_mode", fmha_args.is_group_mode);
  log_value("has_lse", fmha_args.has_lse);
  log_value("has_dropout", has_dropout);
  log_value("skip_min_seqlen_q", (fmha_args.min_seqlen_q != 0));
  log_value("use_asm_v3", fmha_args.use_asm_v3);
  log_value("how_v3_bf16_cvt", fmha_args.how_v3_bf16_cvt);
  // debug fmha_args
  std::cout<<"\nfmha_args: \n";

  log_value("q_ptr", fmha_args.q_ptr);
  log_value("k_ptr", fmha_args.k_ptr);
  log_value("v_ptr", fmha_args.v_ptr);
  log_value("bias_ptr", fmha_args.bias_ptr);
  log_value("rand_val_ptr", fmha_args.rand_val_ptr);
  log_value("lse_ptr", fmha_args.lse_ptr);
  log_value("o_ptr", fmha_args.o_ptr);

  log_value("seqstart_q_ptr", fmha_args.seqstart_q_ptr);
  log_value("seqstart_k_ptr", fmha_args.seqstart_k_ptr);
  log_value("seqlen_q_ptr", fmha_args.seqlen_q_ptr);
  log_value("seqlen_k_ptr", fmha_args.seqlen_k_ptr);
  log_value("cu_seqlen_q_ptr", fmha_args.cu_seqlen_q_ptr);
  log_value("cu_seqlen_k_ptr", fmha_args.cu_seqlen_k_ptr);

  log_value("seqlen_q", fmha_args.seqlen_q);
  log_value("seqlen_k", fmha_args.seqlen_k);
  log_value("batch", fmha_args.batch);
  log_value("max_seqlen_q", fmha_args.max_seqlen_q);
  log_value("hdim_q", fmha_args.hdim_q);
  log_value("hdim_v", fmha_args.hdim_v);
  log_value("nhead_q", fmha_args.nhead_q);
  log_value("nhead_k", fmha_args.nhead_k);

  log_value("scale_s", fmha_args.scale_s);
  log_value("logits_soft_cap", fmha_args.logits_soft_cap);

  log_value("stride_q", fmha_args.stride_q);
  log_value("stride_k", fmha_args.stride_k);
  log_value("stride_v", fmha_args.stride_v);
  log_value("stride_bias", fmha_args.stride_bias);
  log_value("stride_randval", fmha_args.stride_randval);
  log_value("stride_o", fmha_args.stride_o);
  log_value("nhead_stride_q", fmha_args.nhead_stride_q);
  log_value("nhead_stride_k", fmha_args.nhead_stride_k);
  log_value("nhead_stride_v", fmha_args.nhead_stride_v);
  log_value("nhead_stride_bias", fmha_args.nhead_stride_bias);
  log_value("nhead_stride_randval", fmha_args.nhead_stride_randval);
  log_value("nhead_stride_lse", fmha_args.nhead_stride_lse);
  log_value("nhead_stride_o", fmha_args.nhead_stride_o);
  log_value("batch_stride_q", fmha_args.batch_stride_q);
  log_value("batch_stride_k", fmha_args.batch_stride_k);
  log_value("batch_stride_v", fmha_args.batch_stride_v);
  log_value("batch_stride_bias", fmha_args.batch_stride_bias);
  log_value("batch_stride_randval", fmha_args.batch_stride_randval);
  log_value("batch_stride_lse", fmha_args.batch_stride_lse);
  log_value("batch_stride_o", fmha_args.batch_stride_o);

  log_value("window_size_left", fmha_args.window_size_left);
  log_value("window_size_right", fmha_args.window_size_right);
  log_value("mask_type", fmha_args.mask_type);
  log_value("bias_type", fmha_args.bias_type);
  log_value("min_seqlen_q", fmha_args.min_seqlen_q);

  log_value("p_drop", fmha_args.p_drop);
  log_value("s_randval", fmha_args.s_randval);

  log_value("dropout_seed_ptr", std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
  log_value("dropout_offset_ptr", std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
}

void dump_fwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-fwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
}

hipError_t _ck_attn_fwd_impl(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v, uint64_t bias_b, uint64_t bias_h,
  uint64_t max_tokens_q,
  const void* q_ptr,
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr,
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr,
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* bias_ptr,
  const void* alibi_slope_ptr,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_kv_ptr,
  const void* cu_seqlen_q_padded_ptr, const void* cu_seqlen_kv_padded_ptr,
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
  int how_v3_bf16_cvt,
  bool is_group_mode,
  const char* func_name,
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
  float logits_soft_cap = 0.f;
  float p_drop = dropout_probability;

  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);

  bool ck_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_log_config = true;
  }
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, ck_log_config};

  bias_enum bias_type = bias_enum::no_bias;
  BiasShape bias_shape = BiasShape::k11SS;

  aiter::mha_fwd_args fmha_args{};
  fmha_args.q_ptr = q_ptr;
  fmha_args.k_ptr = k_ptr;
  fmha_args.v_ptr = v_ptr;

  fmha_args.batch    = batch;
  fmha_args.seqlen_q = max_seqlen_q; // unused in group mode
  fmha_args.hdim_q   = hdim_q;
  fmha_args.hdim_v   = hdim_v;
  fmha_args.nhead_q  = nhead;
  fmha_args.nhead_k  = nhead_k;

  fmha_args.stride_q       = stride_s_q;
  fmha_args.stride_k       = stride_s_k;
  fmha_args.stride_v       = stride_s_v;
  fmha_args.nhead_stride_q = stride_h_q;
  fmha_args.nhead_stride_k = stride_h_k;
  fmha_args.nhead_stride_v = stride_h_v;
  fmha_args.batch_stride_q = stride_b_q;
  fmha_args.batch_stride_k = stride_b_k;
  fmha_args.batch_stride_v = stride_b_v;

  if (is_group_mode) {
    fmha_args.seqstart_q_ptr = cu_seqlen_q_padded_ptr==nullptr? cu_seqlen_q_ptr: cu_seqlen_q_padded_ptr;
    fmha_args.seqstart_k_ptr = cu_seqlen_kv_padded_ptr==nullptr? cu_seqlen_kv_ptr: cu_seqlen_kv_padded_ptr;
    fmha_args.seqlen_q_ptr = nullptr;
    fmha_args.seqlen_k_ptr = nullptr;
    fmha_args.cu_seqlen_q_ptr = cu_seqlen_q_ptr;
    fmha_args.cu_seqlen_k_ptr = cu_seqlen_kv_ptr;
  } else {
    fmha_args.seqstart_q_ptr = nullptr;
    fmha_args.seqstart_k_ptr = nullptr;
    fmha_args.seqlen_q_ptr = nullptr;
    fmha_args.seqlen_k_ptr = nullptr;
    fmha_args.cu_seqlen_q_ptr = nullptr;
    fmha_args.cu_seqlen_k_ptr = nullptr;
    std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(attn_bias_type, b, h, bias_b, bias_h);
  }

  fmha_args.bias_ptr = bias_type==bias_enum::alibi? alibi_slope_ptr :bias_ptr;
  fmha_args.lse_ptr  = lse_ptr;
  fmha_args.o_ptr    = o_ptr;

  fmha_args.block_scale_seqstart_q_ptr = nullptr;
  fmha_args.block_scale_seqstart_k_ptr = nullptr;
  fmha_args.sink_ptr = nullptr;
  fmha_args.seqlen_k     = max_seqlen_k; // unused in group mode (or kvcache enabled)
  fmha_args.max_seqlen_q = max_seqlen_q;

  fmha_args.scale_s = scale_s;

  fmha_args.logits_soft_cap = logits_soft_cap;

  // bias is of shape [b, h , s_q, s_kv]
  fmha_args.stride_bias = is_group_mode? 0 : (bias_type==bias_enum::alibi? 0: max_seqlen_k);
  fmha_args.stride_o          = stride_s_o;
  fmha_args.nhead_stride_bias = get_nhead_stride_bias(bias_shape, max_seqlen_q, max_seqlen_k, is_group_mode);
  fmha_args.batch_stride_bias = get_batch_stride_bias(bias_h, bias_shape, max_seqlen_q, max_seqlen_k, is_group_mode, true);
  // softmax_lse is of shape [b, h, s_q]
  fmha_args.nhead_stride_lse  = is_group_mode? max_tokens_q : max_seqlen_q;
  fmha_args.batch_stride_lse  = is_group_mode? 0 : nhead * max_seqlen_q;
  fmha_args.nhead_stride_o    = stride_h_o;
  fmha_args.batch_stride_o    = stride_b_o;

  fmha_args.window_size_left  = left;
  fmha_args.window_size_right = right;
  fmha_args.mask_type         = static_cast<ck_tile::index_t>(mask_type);

  fmha_args.rand_val_ptr = nullptr;

  fmha_args.stride_randval       = max_seqlen_k;
  // Unused
  fmha_args.nhead_stride_randval = 0; 
  fmha_args.batch_stride_randval = 0;
  fmha_args.nhead_stride_q_descale = 0;
  fmha_args.nhead_stride_k_descale = 0;
  fmha_args.nhead_stride_v_descale = 0;
  fmha_args.batch_stride_q_descale = 0;
  fmha_args.batch_stride_k_descale = 0;
  fmha_args.batch_stride_v_descale = 0;

  fmha_args.p_drop    = p_drop;
  fmha_args.s_randval = 0;
  fmha_args.drop_seed_offset = std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr};
  fmha_args.use_asm_v3      = uses_fwd_v3;
  fmha_args.how_v3_bf16_cvt = how_v3_bf16_cvt;
  fmha_args.v3_api_check    = false;
  fmha_args.data_type       = get_data_type_str(dtype);
  fmha_args.is_group_mode   = is_group_mode;
  fmha_args.bias_type       = static_cast<int>(bias_type);
  fmha_args.has_lse         = lse_ptr!=nullptr;
  fmha_args.qscale_type     = static_cast<int>(quant_scale_enum::no_scale);
  fmha_args.has_sink        = false;
  fmha_args.q_descale_ptr    = nullptr;
  fmha_args.k_descale_ptr    = nullptr;
  fmha_args.v_descale_ptr    = nullptr;
  fmha_args.sink_size        = 0;
  fmha_args.min_seqlen_q     = 0;
  fmha_args.block_scale_size_q  = 0;
  fmha_args.block_scale_size_kv = 0;

  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")){
    if(is_group_mode && std::string(env_p) == "1"){
      if(ck_log_config){
        std::cout << "attn_fwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(b, cu_seqlen_q_ptr, cu_seqlen_q_padded_ptr, lse_ptr, stream);
    }
  }

  // print ck traits and fmha_args when needed
  log_fwd_config(func_name, has_dropout, fmha_args, ck_log_config);
  float average_runtime = aiter::mha_fwd(fmha_args, stream_config);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  return hipSuccess;
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
  int how_v3_bf16_cvt,
  hipStream_t stream){

  return _ck_attn_fwd_impl(
    dtype,
    b, h, hg, s_q, s_kv, d_qk, d_v, bias_b, bias_h,
    0,
    q_ptr, stride_b_q, stride_h_q, stride_s_q,
    k_ptr, stride_b_k, stride_h_k, stride_s_k,
    v_ptr, stride_b_v, stride_h_v, stride_s_v,
    bias_ptr,
    alibi_slope_ptr,
    nullptr, nullptr, // cu_seqlen_q_ptr, cu_seqlen_kv_ptr,
    nullptr, nullptr, // cu_seqlen_q_padded_ptr, cu_seqlen_kv_padded_ptr
    is_training,
    scaling_factor,
    dropout_probability,
    philox_seed_ptr, philox_offset_ptr,
    attn_bias_type,
    attn_mask_type,
    window_size_left, window_size_right,
    o_ptr,
    stride_b_o, stride_h_o, stride_s_o,
    lse_ptr,
    uses_fwd_v3,
    how_v3_bf16_cvt,
    false,
    __FUNCTION__, // func_name
    stream
  );
}

hipError_t ck_attn_varlen_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v,
  uint64_t max_tokens_q,
  const void* q_ptr,
  uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr,
  uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr,
  uint64_t stride_h_v, uint64_t stride_s_v,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_kv_ptr,
  const void* cu_seqlen_q_padded_ptr, const void* cu_seqlen_kv_padded_ptr,
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
  int how_v3_bf16_cvt,
  hipStream_t stream){

  return _ck_attn_fwd_impl(
    dtype,
    b, h, hg, s_q, s_kv, d_qk, d_v, 0, 0,
    max_tokens_q,
    q_ptr, 0, stride_h_q, stride_s_q,
    k_ptr, 0, stride_h_k, stride_s_k,
    v_ptr, 0, stride_h_v, stride_s_v,
    nullptr, // bias_ptr,
    nullptr, // alibi_slope_ptr
    cu_seqlen_q_ptr, cu_seqlen_kv_ptr,
    cu_seqlen_q_padded_ptr, cu_seqlen_kv_padded_ptr,
    is_training,
    scaling_factor,
    dropout_probability,
    philox_seed_ptr, philox_offset_ptr,
    BiasType::no_bias,
    attn_mask_type,
    window_size_left, window_size_right,
    o_ptr,
    0, stride_h_o, stride_s_o,
    lse_thd_ptr,
    uses_fwd_v3,
    how_v3_bf16_cvt,
    true,
    __FUNCTION__, // func_name
    stream
  );
}

}//namespace ck_fused_attn

