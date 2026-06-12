/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "qola_mha_fwd.h"
#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{

// print the fmha traits and fmha_args when calling ck apis
void log_fwd_config(const char* func_name, bool has_dropout, const aiter::mha_fwd_args& fmha_args, std::ostream* log_file){

  (*log_file) << "\n" << func_name << "\n";

  // debug fmha_traits
  (*log_file) << "\nfmha_traits: \n";
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "data_type", fmha_args.data_type);
  log_value(log_file, "is_group_mode", fmha_args.is_group_mode);
  log_value(log_file, "has_lse", fmha_args.has_lse);
  log_value(log_file, "has_dropout", has_dropout);
  log_value(log_file, "skip_min_seqlen_q", (fmha_args.min_seqlen_q != 0));
  log_value(log_file, "use_asm_v3", fmha_args.use_asm_v3);
  log_value(log_file, "how_v3_bf16_cvt", fmha_args.how_v3_bf16_cvt);
  // debug fmha_args
  (*log_file) << "\nfmha_args: \n";

  log_value(log_file, "q_ptr", fmha_args.q_ptr);
  log_value(log_file, "k_ptr", fmha_args.k_ptr);
  log_value(log_file, "v_ptr", fmha_args.v_ptr);
  log_value(log_file, "bias_ptr", fmha_args.bias_ptr);
  log_value(log_file, "rand_val_ptr", fmha_args.rand_val_ptr);
  log_value(log_file, "lse_ptr", fmha_args.lse_ptr);
  log_value(log_file, "o_ptr", fmha_args.o_ptr);

  log_value(log_file, "seqstart_q_ptr", fmha_args.seqstart_q_ptr);
  log_value(log_file, "seqstart_k_ptr", fmha_args.seqstart_k_ptr);
  log_value(log_file, "seqlen_q_ptr", fmha_args.seqlen_q_ptr);
  log_value(log_file, "seqlen_k_ptr", fmha_args.seqlen_k_ptr);
  log_value(log_file, "cu_seqlen_q_ptr", fmha_args.cu_seqlen_q_ptr);
  log_value(log_file, "cu_seqlen_k_ptr", fmha_args.cu_seqlen_k_ptr);

  log_value(log_file, "seqlen_q", fmha_args.seqlen_q);
  log_value(log_file, "seqlen_k", fmha_args.seqlen_k);
  log_value(log_file, "batch", fmha_args.batch);
  log_value(log_file, "max_seqlen_q", fmha_args.max_seqlen_q);
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "nhead_q", fmha_args.nhead_q);
  log_value(log_file, "nhead_k", fmha_args.nhead_k);
  log_value(log_file, "scale_s", fmha_args.scale_s);
  log_value(log_file, "logits_soft_cap", fmha_args.logits_soft_cap);

  log_value(log_file, "stride_q", fmha_args.stride_q);
  log_value(log_file, "stride_k", fmha_args.stride_k);
  log_value(log_file, "stride_v", fmha_args.stride_v);
  log_value(log_file, "stride_bias", fmha_args.stride_bias);
  log_value(log_file, "stride_randval", fmha_args.stride_randval);
  log_value(log_file, "stride_o", fmha_args.stride_o);
  log_value(log_file, "nhead_stride_q", fmha_args.nhead_stride_q);
  log_value(log_file, "nhead_stride_k", fmha_args.nhead_stride_k);
  log_value(log_file, "nhead_stride_v", fmha_args.nhead_stride_v);
  log_value(log_file, "nhead_stride_bias", fmha_args.nhead_stride_bias);
  log_value(log_file, "nhead_stride_randval", fmha_args.nhead_stride_randval);
  log_value(log_file, "nhead_stride_lse", fmha_args.nhead_stride_lse);
  log_value(log_file, "nhead_stride_o", fmha_args.nhead_stride_o);
  log_value(log_file, "batch_stride_q", fmha_args.batch_stride_q);
  log_value(log_file, "batch_stride_k", fmha_args.batch_stride_k);
  log_value(log_file, "batch_stride_v", fmha_args.batch_stride_v);
  log_value(log_file, "batch_stride_bias", fmha_args.batch_stride_bias);
  log_value(log_file, "batch_stride_randval", fmha_args.batch_stride_randval);
  log_value(log_file, "batch_stride_lse", fmha_args.batch_stride_lse);
  log_value(log_file, "batch_stride_o", fmha_args.batch_stride_o);

  log_value(log_file, "window_size_left", fmha_args.window_size_left);
  log_value(log_file, "window_size_right", fmha_args.window_size_right);
  log_value(log_file, "mask_type", fmha_args.mask_type);
  log_value(log_file, "bias_type", fmha_args.bias_type);
  log_value(log_file, "min_seqlen_q", fmha_args.min_seqlen_q);
  log_value(log_file, "p_drop", fmha_args.p_drop);
  log_value(log_file, "s_randval", fmha_args.s_randval);

  log_value(log_file, "dropout_seed_ptr", std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
  log_value(log_file, "dropout_offset_ptr", std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
}

void dump_fwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-fwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
}

hipError_t ck_attn_fwd(const CKAttnFwdArgs& args, hipStream_t stream){

  bool has_dropout = (args.is_training && args.dropout_probability > 0.f);

  auto* log_file = get_ck_log_stream();
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, get_ck_log_stream() != nullptr};

  bias_enum bias_type = bias_enum::no_bias;
  BiasShape bias_shape = BiasShape::k11SS;
  if (!args.is_group_mode()) {
    std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(&args);
  }

  aiter::mha_fwd_args fmha_args{};
  fmha_args.q_ptr = args.q_ptr;
  fmha_args.k_ptr = args.k_ptr;
  fmha_args.v_ptr = args.v_ptr;

  fmha_args.batch    = args.b;
  fmha_args.seqlen_q = args.s_q; // unused in group mode
  fmha_args.hdim_q   = args.d_qk;
  fmha_args.hdim_v   = args.d_v;
  fmha_args.nhead_q  = args.h;
  fmha_args.nhead_k  = args.hg;

  fmha_args.stride_q       = args.stride_s_q;
  fmha_args.stride_k       = args.stride_s_k;
  fmha_args.stride_v       = args.stride_s_v;
  fmha_args.nhead_stride_q = args.stride_h_q;
  fmha_args.nhead_stride_k = args.stride_h_k;
  fmha_args.nhead_stride_v = args.stride_h_v;
  fmha_args.batch_stride_q = args.stride_b_q;
  fmha_args.batch_stride_k = args.stride_b_k;
  fmha_args.batch_stride_v = args.stride_b_v;

  if (args.is_group_mode()) {
    fmha_args.seqstart_q_ptr = args.cu_seqlen_q_padded_ptr==nullptr? args.cu_seqlen_q_ptr : args.cu_seqlen_q_padded_ptr;
    fmha_args.seqstart_k_ptr = args.cu_seqlen_kv_padded_ptr==nullptr? args.cu_seqlen_kv_ptr : args.cu_seqlen_kv_padded_ptr;
    fmha_args.cu_seqlen_q_ptr = args.cu_seqlen_q_ptr;
    fmha_args.cu_seqlen_k_ptr = args.cu_seqlen_kv_ptr;
  } else {
    fmha_args.seqstart_q_ptr = nullptr;
    fmha_args.seqstart_k_ptr = nullptr;
    fmha_args.cu_seqlen_q_ptr = nullptr;
    fmha_args.cu_seqlen_k_ptr = nullptr;
  }
  fmha_args.seqlen_q_ptr = nullptr;
  fmha_args.seqlen_k_ptr = nullptr;

  fmha_args.bias_ptr = bias_type==bias_enum::alibi? args.alibi_slope_ptr : args.bias_ptr;
  fmha_args.lse_ptr  = args.lse_ptr;
  fmha_args.o_ptr    = args.o_ptr;

  fmha_args.block_scale_seqstart_q_ptr = nullptr;
  fmha_args.block_scale_seqstart_k_ptr = nullptr;
  fmha_args.sink_ptr = nullptr;
  fmha_args.seqlen_k     = args.s_kv; // unused in group mode (or kvcache enabled)
  fmha_args.max_seqlen_q = args.s_q;

  fmha_args.scale_s = args.scaling_factor;

  fmha_args.logits_soft_cap = 0.f;

  // bias is of shape [b, h , s_q, s_kv]
  fmha_args.stride_bias = args.is_group_mode()? 0 : (bias_type==bias_enum::alibi? 0: args.s_kv);
  fmha_args.stride_o          = args.stride_s_o;
  fmha_args.nhead_stride_bias = get_nhead_stride_bias(bias_shape, args.s_q, args.s_kv, args.is_group_mode());
  fmha_args.batch_stride_bias = get_batch_stride_bias(args.bias_h, bias_shape, args.s_q, args.s_kv, args.is_group_mode(), true);
  // softmax_lse is of shape [b, h, s_q]
  fmha_args.nhead_stride_lse  = args.is_group_mode()? args.max_tokens_q : args.s_q;
  fmha_args.batch_stride_lse  = args.is_group_mode()? 0 : args.h * args.s_q;
  fmha_args.nhead_stride_o    = args.stride_h_o;
  fmha_args.batch_stride_o    = args.stride_b_o;

  fmha_args.window_size_left  = args.window_size_left;
  fmha_args.window_size_right = args.window_size_right;
  fmha_args.mask_type         = static_cast<ck_tile::index_t>(static_cast<mask_enum>(args.attn_mask_type));

  fmha_args.rand_val_ptr = nullptr;

  fmha_args.stride_randval       = args.s_kv;
  // Unused
  fmha_args.nhead_stride_randval = 0;
  fmha_args.batch_stride_randval = 0;
  fmha_args.nhead_stride_q_descale = 0;
  fmha_args.nhead_stride_k_descale = 0;
  fmha_args.nhead_stride_v_descale = 0;
  fmha_args.batch_stride_q_descale = 0;
  fmha_args.batch_stride_k_descale = 0;
  fmha_args.batch_stride_v_descale = 0;

  fmha_args.p_drop    = args.dropout_probability;
  fmha_args.s_randval = 0;
  fmha_args.drop_seed_offset = std::pair<const void*, const void*>{args.philox_seed_ptr, args.philox_offset_ptr};
  fmha_args.use_asm_v3      = args.uses_fwd_v3;
  fmha_args.how_v3_bf16_cvt = args.how_v3_bf16_cvt;
  fmha_args.v3_api_check    = false;
  fmha_args.data_type       = get_data_type_str(args.dtype);
  fmha_args.is_group_mode   = args.is_group_mode();
  fmha_args.bias_type       = static_cast<int>(bias_type);
  fmha_args.has_lse         = args.lse_ptr!=nullptr;
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
    if(args.is_group_mode() && std::string(env_p) == "1"){
      if(log_file){
        *log_file << "attn_fwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(args.b, args.cu_seqlen_q_ptr, args.cu_seqlen_q_padded_ptr, args.lse_ptr, stream);
    }
  }

  // print ck traits and fmha_args when needed
  if(log_file){
     log_fwd_config(__FUNCTION__, has_dropout, fmha_args, log_file);
  }
  float average_runtime = QOLA_NS(mha_fwd)(fmha_args, stream_config);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  return hipSuccess;
}

// print the splitkv fmha_args when calling ck apis
void log_fwd_splitkv_config(const char* func_name, const aiter::mha_fwd_splitkv_args& fmha_args, std::ostream* log_file){
  (*log_file) << "\n" << func_name << "\n";
  (*log_file) << "\nfmha_splitkv_args: \n";
  log_value(log_file, "q_ptr", fmha_args.q_ptr);
  log_value(log_file, "k_ptr", fmha_args.k_ptr);
  log_value(log_file, "v_ptr", fmha_args.v_ptr);
  log_value(log_file, "lse_acc_ptr", fmha_args.lse_acc_ptr);
  log_value(log_file, "o_acc_ptr", fmha_args.o_acc_ptr);
  log_value(log_file, "lse_ptr", fmha_args.lse_ptr);
  log_value(log_file, "o_ptr", fmha_args.o_ptr);
  log_value(log_file, "seqstart_q_ptr", fmha_args.seqstart_q_ptr);
  log_value(log_file, "seqstart_k_ptr", fmha_args.seqstart_k_ptr);
  log_value(log_file, "batch", fmha_args.batch);
  log_value(log_file, "max_seqlen_q", fmha_args.max_seqlen_q);
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "nhead_q", fmha_args.nhead_q);
  log_value(log_file, "nhead_k", fmha_args.nhead_k);
  log_value(log_file, "num_splits", fmha_args.num_splits);
  log_value(log_file, "scale_s", fmha_args.scale_s);
  log_value(log_file, "nhead_stride_lse_acc", fmha_args.nhead_stride_lse_acc);
  log_value(log_file, "split_stride_lse_acc", fmha_args.split_stride_lse_acc);
  log_value(log_file, "nhead_stride_o_acc", fmha_args.nhead_stride_o_acc);
  log_value(log_file, "split_stride_o_acc", fmha_args.split_stride_o_acc);
  log_value(log_file, "stride_o_acc", fmha_args.stride_o_acc);
  log_value(log_file, "window_size_left", fmha_args.window_size_left);
  log_value(log_file, "window_size_right", fmha_args.window_size_right);
  log_value(log_file, "mask_type", fmha_args.mask_type);
}

// Split-KV (Flash-Decoding) forward. Group mode only: the accumulator layout and
// stride math below mirror AITER's varlen reference
// (csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu::get_ck_fmha_varlen_fwd_splitkv_args),
// where lse_acc is [nhead, num_splits, max_tokens_q] and o_acc is
// [nhead, num_splits, max_tokens_q, d_v], both fp32 and batch-stride 0.
hipError_t ck_attn_fwd_splitkv(const CKAttnFwdArgs& args, hipStream_t stream){

  auto* log_file = get_ck_log_stream();
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, get_ck_log_stream() != nullptr};

  const uint64_t max_tokens_q = args.max_tokens_q;
  const uint64_t d_v          = args.d_v;
  const uint64_t num_splits   = static_cast<uint64_t>(args.num_splits);

  aiter::mha_fwd_splitkv_args fmha_splitkv_args{};
  fmha_splitkv_args.q_ptr = args.q_ptr;
  fmha_splitkv_args.k_ptr = args.k_ptr;
  fmha_splitkv_args.v_ptr = args.v_ptr;
  fmha_splitkv_args.bias_ptr = nullptr;
  fmha_splitkv_args.lse_acc_ptr = args.lse_acc_ptr;
  fmha_splitkv_args.o_acc_ptr   = args.o_acc_ptr;
  fmha_splitkv_args.lse_ptr     = args.lse_ptr;
  fmha_splitkv_args.o_ptr       = args.o_ptr;

  // No paged-KV cache: block table / paging fields stay disabled.
  fmha_splitkv_args.block_table_ptr = nullptr;
  fmha_splitkv_args.batch_stride_block_table = 0;
  fmha_splitkv_args.page_block_size = 0;
  fmha_splitkv_args.is_gappy = false;
  fmha_splitkv_args.cache_batch_idx = nullptr;

  // Group mode: seqstart drives the per-sequence lengths (prefer padded offsets).
  fmha_splitkv_args.seqstart_q_ptr = args.cu_seqlen_q_padded_ptr==nullptr? args.cu_seqlen_q_ptr : args.cu_seqlen_q_padded_ptr;
  fmha_splitkv_args.seqstart_k_ptr = args.cu_seqlen_kv_padded_ptr==nullptr? args.cu_seqlen_kv_ptr : args.cu_seqlen_kv_padded_ptr;
  fmha_splitkv_args.seqlen_k_ptr = nullptr;
  fmha_splitkv_args.sink_ptr = nullptr;

  fmha_splitkv_args.seqlen_q     = args.s_q; // unused in group mode
  fmha_splitkv_args.seqlen_k     = args.s_kv; // unused in group mode
  fmha_splitkv_args.batch        = args.b;
  fmha_splitkv_args.max_seqlen_q = args.s_q;
  fmha_splitkv_args.hdim_q       = args.d_qk;
  fmha_splitkv_args.hdim_v       = args.d_v;
  fmha_splitkv_args.nhead_q      = args.h;
  fmha_splitkv_args.nhead_k      = args.hg;
  fmha_splitkv_args.num_splits   = args.num_splits;

  fmha_splitkv_args.scale_s = args.scaling_factor;
  fmha_splitkv_args.scale_p = 1.f;
  fmha_splitkv_args.scale_o = 1.f;
  fmha_splitkv_args.logits_soft_cap = 0.f;

  fmha_splitkv_args.stride_q       = args.stride_s_q;
  fmha_splitkv_args.stride_k       = args.stride_s_k;
  fmha_splitkv_args.stride_v       = args.stride_s_v;
  fmha_splitkv_args.stride_bias    = 0;
  fmha_splitkv_args.stride_o_acc   = d_v;
  fmha_splitkv_args.stride_o       = args.stride_s_o;
  fmha_splitkv_args.nhead_stride_q = args.stride_h_q;
  fmha_splitkv_args.nhead_stride_k = args.stride_h_k;
  fmha_splitkv_args.nhead_stride_v = args.stride_h_v;
  fmha_splitkv_args.nhead_stride_bias = 0;
  // softmax_lse is [nhead, max_tokens_q] in group mode (matches non-split path).
  fmha_splitkv_args.nhead_stride_lse     = max_tokens_q;
  fmha_splitkv_args.nhead_stride_lse_acc = num_splits * max_tokens_q;
  fmha_splitkv_args.nhead_stride_o_acc   = num_splits * max_tokens_q * d_v;
  fmha_splitkv_args.nhead_stride_o       = args.stride_h_o;
  fmha_splitkv_args.batch_stride_q   = 0;
  fmha_splitkv_args.batch_stride_k   = 0;
  fmha_splitkv_args.batch_stride_v   = 0;
  fmha_splitkv_args.batch_stride_bias = 0;
  fmha_splitkv_args.batch_stride_lse     = 0;
  fmha_splitkv_args.batch_stride_lse_acc = 0;
  fmha_splitkv_args.batch_stride_o_acc   = 0;
  fmha_splitkv_args.batch_stride_o   = 0;
  fmha_splitkv_args.split_stride_lse_acc = max_tokens_q;
  fmha_splitkv_args.split_stride_o_acc   = max_tokens_q * d_v;

  fmha_splitkv_args.window_size_left  = args.window_size_left;
  fmha_splitkv_args.window_size_right = args.window_size_right;
  fmha_splitkv_args.sink_size = 0;
  mask_enum mask_type = static_cast<mask_enum>(args.attn_mask_type);
  fmha_splitkv_args.mask_type = static_cast<ck_tile::index_t>(mask_type);

  bias_enum bias_type = bias_enum::no_bias;
  bool has_lse = args.lse_ptr != nullptr;

  if(log_file){
    log_fwd_splitkv_config(__FUNCTION__, fmha_splitkv_args, log_file);
  }
  float average_runtime = QOLA_NS(mha_fwd_splitkv)(
    fmha_splitkv_args, stream_config, get_data_type_str(args.dtype),
    args.is_group_mode(), mask_type, bias_type, has_lse, /*has_sink=*/false);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn splitkv fwd pass.");
  }
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  return hipSuccess;
}

}//namespace ck_fused_attn

