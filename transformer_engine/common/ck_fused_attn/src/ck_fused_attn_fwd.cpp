/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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

namespace {
std::ostream* get_fwd_log_stream() {
  thread_local std::ofstream log_file;
  thread_local bool attempted = false;
  thread_local bool opened = false;
  thread_local bool requested = false;
  thread_local std::string log_dir_str;
  if (!attempted) {
    attempted = true;
    if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG")) {
      log_dir_str = std::string(env_p);
      requested = !log_dir_str.empty() && log_dir_str != "0";
    }
    if (requested) {
      opened = open_ck_fused_attn_log_file(log_file, "ck_fused_attn_fwd", log_dir_str);
    }
  }
  if (!requested) {
    return nullptr;
  }
  if (!opened) {
    return &std::cout;
  }
  return &log_file;
}
}  // namespace

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
                    const bool how_v3_bf16_cvt,
                    const fmha_fwd_args& fmha_args){
  if (auto* log_file = get_fwd_log_stream()) {
    *log_file << "\n" << func_name << "\n";

    // debug fmha_traits
    *log_file << "\n" << "fmha_traits: " << "\n";
    *log_file << "hdim_q: " << fmha_args.hdim_q << "\n";
    *log_file << "hdim_v: " << fmha_args.hdim_v << "\n";
    *log_file << "data_type: " << data_type_str << "\n";
    *log_file << "is_group_mode: " << is_group_mode << "\n";
    *log_file << "is_v_rowmajor: " << is_v_rowmajor << "\n";
    *log_file << "has_logits_soft_cap: " << has_logits_soft_cap << "\n";
    *log_file << "mask_type: " << static_cast<std::underlying_type<mask_enum>::type>(mask_type) << "\n";
    *log_file << "bias_type: " << static_cast<std::underlying_type<bias_enum>::type>(bias_type) << "\n";
    *log_file << "has_lse: " << has_lse << "\n";
    *log_file << "has_dropout: " << has_dropout << "\n";
    *log_file << "do_fp8_static_quant: " << do_fp8_static_quant << "\n";
    *log_file << "skip_min_seqlen_q: " << (fmha_args.min_seqlen_q != 0) << "\n";
    *log_file << "uses_fwd_v3: " << uses_fwd_v3 << "\n";
    *log_file << "how_v3_bf16_cvt: " << how_v3_bf16_cvt << "\n";

    // debug fmha_args
    *log_file << "\n" << "fmha_args: " << "\n";

    *log_file << "q_ptr: " << fmha_args.q_ptr << "\n";
    *log_file << "k_ptr: " << fmha_args.k_ptr << "\n";
    *log_file << "v_ptr: " << fmha_args.v_ptr << "\n";
    *log_file << "bias_ptr: " << fmha_args.bias_ptr << "\n";
    *log_file << "rand_val_ptr: " << fmha_args.rand_val_ptr << "\n";
    *log_file << "lse_ptr: " << fmha_args.lse_ptr << "\n";
    *log_file << "o_ptr: " << fmha_args.o_ptr << "\n";

    *log_file << "seqstart_q_ptr: " << fmha_args.seqstart_q_ptr << "\n";
    *log_file << "seqstart_k_ptr: " << fmha_args.seqstart_k_ptr << "\n";
    *log_file << "seqlen_q_ptr: " << fmha_args.seqlen_q_ptr << "\n";
    *log_file << "seqlen_k_ptr: " << fmha_args.seqlen_k_ptr << "\n";
    *log_file << "cu_seqlen_q_ptr: " << fmha_args.cu_seqlen_q_ptr << "\n";
    *log_file << "cu_seqlen_k_ptr: " << fmha_args.cu_seqlen_k_ptr << "\n";

    *log_file << "seqlen_q: " << fmha_args.seqlen_q << "\n";
    *log_file << "seqlen_k: " << fmha_args.seqlen_k << "\n";
    *log_file << "batch: " << fmha_args.batch << "\n";
    *log_file << "max_seqlen_q: " << fmha_args.max_seqlen_q << "\n";
    *log_file << "hdim_q: " << fmha_args.hdim_q << "\n";
    *log_file << "hdim_v: " << fmha_args.hdim_v << "\n";
    *log_file << "nhead_q: " << fmha_args.nhead_q << "\n";
    *log_file << "nhead_k: " << fmha_args.nhead_k << "\n";

    *log_file << "scale_s: " << fmha_args.scale_s << "\n";

    *log_file << "logits_soft_cap: " << fmha_args.logits_soft_cap << "\n";

    *log_file << "stride_q: " << fmha_args.stride_q << "\n";
    *log_file << "stride_k: " << fmha_args.stride_k << "\n";
    *log_file << "stride_v: " << fmha_args.stride_v << "\n";
    *log_file << "stride_bias: " << fmha_args.stride_bias << "\n";
    *log_file << "stride_randval: " << fmha_args.stride_randval << "\n";
    *log_file << "stride_o: " << fmha_args.stride_o << "\n";
    *log_file << "nhead_stride_q: " << fmha_args.nhead_stride_q << "\n";
    *log_file << "nhead_stride_k: " << fmha_args.nhead_stride_k << "\n";
    *log_file << "nhead_stride_v: " << fmha_args.nhead_stride_v << "\n";
    *log_file << "nhead_stride_bias: " << fmha_args.nhead_stride_bias << "\n";
    *log_file << "nhead_stride_randval: " << fmha_args.nhead_stride_randval << "\n";
    *log_file << "nhead_stride_lse: " << fmha_args.nhead_stride_lse << "\n";
    *log_file << "nhead_stride_o: " << fmha_args.nhead_stride_o << "\n";
    *log_file << "batch_stride_q: " << fmha_args.batch_stride_q << "\n";
    *log_file << "batch_stride_k: " << fmha_args.batch_stride_k << "\n";
    *log_file << "batch_stride_v: " << fmha_args.batch_stride_v << "\n";
    *log_file << "batch_stride_bias: " << fmha_args.batch_stride_bias << "\n";
    *log_file << "batch_stride_randval: " << fmha_args.batch_stride_randval << "\n";
    *log_file << "batch_stride_lse: " << fmha_args.batch_stride_lse << "\n";
    *log_file << "batch_stride_o: " << fmha_args.batch_stride_o << "\n";

    *log_file << "window_size_left: " << fmha_args.window_size_left << "\n";
    *log_file << "window_size_right: " << fmha_args.window_size_right << "\n";
    *log_file << "mask_type: " << fmha_args.mask_type << "\n";
    *log_file << "min_seqlen_q: " << fmha_args.min_seqlen_q << "\n";

    *log_file << "p_drop: " << fmha_args.p_drop << "\n";
    *log_file << "s_randval: " << fmha_args.s_randval << "\n";

    *log_file << "dropout_seed_ptr: " << std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)) << "\n";
    *log_file << "dropout_offset_ptr: " << std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)) << "\n";
  }
}

void dump_fwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-fwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
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
  
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, get_fwd_log_stream() != nullptr};

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
                         nullptr, //q_descale_ptr
                         nullptr, //k_descale_ptr
                         nullptr, //v_descale_ptr
                         nullptr,//rand_val_ptr
                         lse_ptr,
                         o_ptr,
                         nullptr, //seqstart_q_ptr
                         nullptr, //seqstart_k_ptr
                         nullptr, //seqlen_q_ptr
                         nullptr, //seqlen_k_ptr
                         nullptr, //cu_padded_q_ptr
                         nullptr, //cu_padded_k_ptr
                         max_seqlen_q,
                         max_seqlen_k,
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
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
                         0, // sink_size
                         static_cast<ck_tile::index_t>(mask_type),
                         0, // min_seqlen_q
                         p_drop,
                         false,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();
  
  // print ck traits and args when needed
  log_fwd_config(__FUNCTION__, data_type_str, is_group_mode, has_logits_soft_cap, mask_type, bias_type, has_lse, has_dropout, is_v_rowmajor, do_fp8_static_quant, uses_fwd_v3, how_v3_bf16_cvt, fmha_args);

  float average_runtime = aiter::mha_fwd(fmha_args,
                                         stream_config,
                                         data_type_str,
                                         is_group_mode,
                                         mask_type,
                                         bias_type,
                                         has_lse,
                                         quant_scale_enum::no_scale,
                                         uses_fwd_v3, 
                                         false,//has_sink
                                         how_v3_bf16_cvt);
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  return hipSuccess;
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

  bool has_dropout = (is_training && dropout_probability > 0.f);
  bool has_lse = (lse_thd_ptr != nullptr);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d_qk;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d_v;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_kv = s_kv;

  float scale_s = scaling_factor;
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
  
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, get_fwd_log_stream() != nullptr};


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
    // use packed lse of shape [h, max_tokens_q]
    const ck_tile::index_t nhead_stride_lse = max_tokens_q;
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
                         nullptr, //q_descale_ptr
                         nullptr, //k_descale_ptr
                         nullptr, //v_descale_ptr
                         nullptr,//rand_val_ptr
                         lse_thd_ptr,
                         o_ptr,
                         cu_seqlen_q_padded_ptr==nullptr? cu_seqlen_q_ptr: cu_seqlen_q_padded_ptr, //seqstart_q_ptr
                         cu_seqlen_kv_padded_ptr==nullptr? cu_seqlen_kv_ptr: cu_seqlen_kv_padded_ptr, //seqstart_k_ptr
                         nullptr, //seqlen_q_ptr
                         nullptr, //seqlen_k_ptr
                         cu_seqlen_q_ptr, //cu_seqlen_q_ptr
                         cu_seqlen_kv_ptr, //cu_seqlen_k_ptr
                         max_seqlen_q, //seqlen_q, unused in group mode
                         max_seqlen_kv, //seqlen_kv, unused in group mode
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
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
                         0, // sink_size
                         static_cast<ck_tile::index_t>(mask_type),
                         0, // min_seqlen_q
                         p_drop,
                         false,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();
  // modify the max_seqlen_q for better performance in 0-length cases
  // lse_thd_ptr used as buffer
  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")){
    if(std::string(env_p) == "1"){
      if (auto* log_file = get_fwd_log_stream()) {
        *log_file
            << "attn_fwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.\n";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(b, cu_seqlen_q_ptr, cu_seqlen_q_padded_ptr, lse_thd_ptr, stream);
    }
  }
  // print ck traits and args when needed
  log_fwd_config(__FUNCTION__, data_type_str, is_group_mode, has_logits_soft_cap, mask_type, bias_type, has_lse, has_dropout, is_v_rowmajor, do_fp8_static_quant, uses_fwd_v3, how_v3_bf16_cvt, fmha_args);

  float average_runtime = aiter::mha_fwd(
    fmha_args,
    stream_config,
    data_type_str,
    is_group_mode,
    mask_type,
    bias_type,
    has_lse,
    quant_scale_enum::no_scale,
    uses_fwd_v3, 
    false,//has_sink
    how_v3_bf16_cvt);
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  return hipSuccess;
}

}//namespace ck_fused_attn

