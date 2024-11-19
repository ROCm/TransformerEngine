/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "ck_tile/host.hpp"
#include "bias.hpp"
#include "mask.hpp"
#include "fmha_bwd.hpp"

namespace ck_fused_attn{

// define dk_dv_reduce function only for fp16 and bf16 types
template<typename DataType>
__global__ void dk_dv_reduce(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_kv, uint64_t d,
  const DataType *dk_expanded,
  const DataType *dv_expanded,
  uint64_t stride_b_dkv_expanded, uint64_t stride_h_dkv_expanded, uint64_t stride_s_dkv_expanded,
  DataType *dk,
  DataType *dv,
  //k,v, dk, dv guaranteed to have the same stride
  uint64_t stride_b_dkv, uint64_t stride_h_dkv, uint64_t stride_s_dkv){
  
  uint64_t batch_idx = blockIdx.x;
  uint64_t seqlen_idx = blockIdx.y;
  uint64_t head_k_idx = blockIdx.z;
  uint64_t hdim_idx = threadIdx.x;
  
  // h guaranteed to be multiples of hg
  uint64_t head_idx_offset = h / hg;

  float sum_dk = 0.0f;
  float sum_dv = 0.0f;

  assert(hdim_dix<d);
  uint64_t read_idx = batch_idx*stride_b_dkv_expanded + head_k_idx*head_idx_offset*stride_h_dkv_expanded + seqlen_idx*stride_s_dkv_expanded + hdim_idx;
  uint64_t write_idx = batch_idx*stride_b_dkv + head_k_idx*stride_h_dkv + seqlen_idx* stride_s_dkv + hdim_idx;
  
  for(uint64_t ii = 0; ii < head_idx_offset; ii++){
    // bf16 requires special casting in CK
    if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
      sum_dk += ck_tile::bf16_to_float(dk_expanded[read_idx]);
      sum_dv += ck_tile::bf16_to_float(dv_expanded[read_idx]);
    }else{
      sum_dk += dk_expanded[read_idx];
      sum_dv += dv_expanded[read_idx];
    }
    read_idx += stride_h_dkv_expanded;
  }

  // bf16 requires special casting in CK
  if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
    dk[write_idx] = ck_tile::float_to_bf16(sum_dk);
    dv[write_idx] = ck_tile::float_to_bf16(sum_dv);
  }else{
    dk[write_idx] = sum_dk;
    dv[write_idx] = sum_dv;
  }
}

#define CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, type, ...)   \
switch (dtype) {                                            \
  case DType::kFloat16: {                                   \
    using type = ck_tile::half_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  case DType::kBFloat16: {                                  \
    using type = ck_tile::bf16_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  default:                                                  \
    throw std::runtime_error("Invalid type for 16 bit..");  \
}

hipError_t ck_attn_bwd(  
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr, 
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  float scaling_factor, float dropout_probability,
  uint64_t philox_seed, uint64_t philox_offset,
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
  void* workspace_ptr,
  hipStream_t stream){

  bool has_dropout = (dropout_probability > 0.f);
  bool has_dbias = false;
  bool is_mqa_gqa = (h > hg);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t seqlen_q = s_q;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d;
  ck_tile::index_t seqlen_k = s_kv;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float p_drop = dropout_probability;
  float p_undrop = 1.0 - p_drop;
  bool is_group_mode = false;
  bool s_randval = false;
  bool is_deterministic = false;

  bias_enum bias_type = bias_enum::no_bias;
  mask_enum mask_type;
  int32_t left, right;
  ck_tile::stream_config stream_config{stream};

  if (attn_mask_type == MaskType::no_mask){
    mask_type = mask_enum::no_mask;
  }else if(attn_mask_type == MaskType::mask_top_left){
    mask_type = mask_enum::mask_top_left;
  }else if(attn_mask_type == MaskType::mask_bottom_right){
    mask_type = mask_enum::mask_bottom_right;
  }else{
    mask_type = mask_enum::window_generic;
  }
  left = window_size_left;
  right = window_size_right;

  ck_tile::index_t shape_seqlen_q = seqlen_q;
  ck_tile::index_t shape_seqlen_k = seqlen_k;

  std::string data_type_str;
  if(dtype==DType::kFloat16){
    data_type_str = "fp16";
  }else if(dtype==DType::kBFloat16){
    data_type_str = "bf16";
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid dtype in ck_fused_attn.");
  }
  auto fmha_traits =
    fmha_bwd_traits{hdim_q,    hdim_v,    data_type_str, is_group_mode,
                    mask_type, bias_type, has_dbias,     has_dropout, 
                    s_randval, is_deterministic};

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // TODO: support bias later
    const ck_tile::index_t stride_bias = 0;
    const ck_tile::index_t stride_o = stride_s_o;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_do = stride_s_do;
    const ck_tile::index_t stride_dq = stride_s_dq;
    const ck_tile::index_t stride_dk = stride_s_dk;
    const ck_tile::index_t stride_dv = stride_s_dv;
    const ck_tile::index_t stride_dkv_expanded = stride_s_dkv_expanded;
    const ck_tile::index_t stride_dq_acc = h*d; //dq_acc of shape (B, S, H, D)
    // TODO: support bias later
    const ck_tile::index_t stride_dbias = 0;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // TODO: support bias later
    const ck_tile::index_t nhead_stride_bias = 0;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    const ck_tile::index_t nhead_stride_randval =
        shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t nhead_stride_do = stride_h_do;
    // TODO: buffer?
    const ck_tile::index_t nhead_stride_lsed = max_seqlen_q;
    const ck_tile::index_t nhead_stride_dq = stride_h_dq;
    const ck_tile::index_t nhead_stride_dk = stride_h_dk;
    const ck_tile::index_t nhead_stride_dv = stride_h_dv;
    const ck_tile::index_t nhead_stride_dkv_expanded = stride_h_dkv_expanded;
    const ck_tile::index_t nhead_stride_dbias = max_seqlen_k;
    const ck_tile::index_t nhead_stride_dq_acc = d; //dq_acc of shape (B, S, H, D)
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = stride_b_q;
    const ck_tile::index_t batch_stride_k = stride_b_k;
    const ck_tile::index_t batch_stride_v = stride_b_v;
    const ck_tile::index_t batch_stride_bias = 0;
    const ck_tile::index_t batch_stride_o = stride_b_o;
    const ck_tile::index_t batch_stride_randval =
        nhead * shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_do = stride_b_do;
    const ck_tile::index_t batch_stride_lsed = nhead * max_seqlen_q;
    const ck_tile::index_t batch_stride_dq = stride_b_dq;
    const ck_tile::index_t batch_stride_dk = stride_b_dk;
    const ck_tile::index_t batch_stride_dv = stride_b_dv;
    const ck_tile::index_t batch_stride_dkv_expanded = stride_b_dkv_expanded;
    const ck_tile::index_t batch_stride_dbias =
        nhead * shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_dq_acc = h*s_q*d; //dq_acc of shape (B, S, H, D)
    const ck_tile::index_t split_stride_dq_acc = b * h * s_q * d;

    return fmha_bwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,
                         o_ptr,
                         lse_ptr,
                         do_ptr,
                         workspace_ptr,
                         nullptr,
                         dq_ptr,
                         is_mqa_gqa? dk_expanded_ptr:dk_ptr,
                         is_mqa_gqa? dv_expanded_ptr:dv_ptr,
                         nullptr, // dbias_ptr
                         dq_acc_ptr, //dq_acc_buf
                         nullptr,//cu_seqlen_q
                         nullptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         shape_seqlen_q,
                         shape_seqlen_k,
                         batch,
                         max_seqlen_q,
                         max_seqlen_k,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_o,
                         stride_randval,
                         stride_do,
                         stride_dq_acc,//stride_dq_acc
                         stride_dq,//stride_dq
                         is_mqa_gqa? stride_dkv_expanded:stride_dk,
                         is_mqa_gqa? stride_dkv_expanded:stride_dv,
                         stride_dbias,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_o,
                         nhead_stride_randval,
                         nhead_stride_do,
                         nhead_stride_lsed,
                         nhead_stride_dq_acc, //nhead_stride_dq_acc
                         nhead_stride_dq,
                         is_mqa_gqa? nhead_stride_dkv_expanded:nhead_stride_dk,
                         is_mqa_gqa? nhead_stride_dkv_expanded:nhead_stride_dv,
                         nhead_stride_dbias,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_o,
                         batch_stride_randval,
                         batch_stride_do,
                         batch_stride_lsed,
                         batch_stride_dq_acc, //batch_stride_dq_acc
                         batch_stride_dq,
                         is_mqa_gqa? batch_stride_dkv_expanded:batch_stride_dk,
                         is_mqa_gqa? batch_stride_dkv_expanded:batch_stride_dv,
                         batch_stride_dbias,
                         split_stride_dq_acc,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         p_undrop,
                         {philox_seed, philox_offset}};
  }();

  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_fused_attn_log_config = true;
  }
  if (ck_fused_attn_log_config) {
    std::cout<<std::endl<<"run ck fmha_bwd: "<<std::endl;
    // fmha_traits debug
    std::cout<<"fmha_traits: "<<std::endl;
    std::cout<<"hdim_q: "<<fmha_traits.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_traits.hdim_v<<std::endl;
    std::cout<<"data_type: "<<fmha_traits.data_type<<std::endl;
    std::cout<<"is_group_mode: "<<fmha_traits.is_group_mode<<std::endl;
    std::cout<<"mask_type: "<<static_cast<std::underlying_type<mask_enum>::type>(fmha_traits.mask_type)<<std::endl;
    std::cout<<"bias_type: "<<static_cast<std::underlying_type<bias_enum>::type>(fmha_traits.bias_type)<<std::endl;
    std::cout<<"has_dbias: "<<fmha_traits.has_dbias<<std::endl;
    std::cout<<"has_dropout: "<<fmha_traits.has_dropout<<std::endl;
    std::cout<<"is_store_randval: "<<fmha_traits.is_store_randval<<std::endl;
    std::cout<<"is_deterministic: "<<fmha_traits.is_deterministic<<std::endl;

    // fmha_args debug
    std::cout<<"fmha_args: "<<std::endl;
    std::cout<<"q_ptr: "<<fmha_args.q_ptr<<std::endl;
    std::cout<<"k_ptr: "<<fmha_args.k_ptr<<std::endl;
    std::cout<<"v_ptr: "<<fmha_args.v_ptr<<std::endl;
    std::cout<<"bias_ptr: "<<fmha_args.bias_ptr<<std::endl;
    std::cout<<"o_ptr: "<<fmha_args.o_ptr<<std::endl;
    std::cout<<"lse_ptr: "<<fmha_args.lse_ptr<<std::endl;
    std::cout<<"do_ptr: "<<fmha_args.do_ptr<<std::endl;
    std::cout<<"d_ptr: "<<fmha_args.d_ptr<<std::endl;
    std::cout<<"rand_val_ptr: "<<fmha_args.rand_val_ptr<<std::endl;
    std::cout<<"dq_ptr: "<<fmha_args.dq_ptr<<std::endl;
    std::cout<<"dk_ptr: "<<fmha_args.dk_ptr<<std::endl;
    std::cout<<"dv_ptr: "<<fmha_args.dv_ptr<<std::endl;
    std::cout<<"dbias_ptr: "<<fmha_args.dbias_ptr<<std::endl;
    std::cout<<"seqstart_q_ptr: "<<fmha_args.seqstart_q_ptr<<std::endl;
    std::cout<<"seqstart_k_ptr: "<<fmha_args.seqstart_k_ptr<<std::endl;
    std::cout<<"seqlen_k_ptr: "<<fmha_args.seqlen_k_ptr<<std::endl;
    std::cout<<"seqlen_q: "<<fmha_args.seqlen_q<<std::endl;
    std::cout<<"seqlen_k: "<<fmha_args.seqlen_k<<std::endl;
    std::cout<<"batch: "<<fmha_args.batch<<std::endl;
    std::cout<<"max_seqlen_q: "<<fmha_args.max_seqlen_q<<std::endl;
    std::cout<<"max_seqlen_k: "<<fmha_args.max_seqlen_k<<std::endl;
    std::cout<<"hdim_q: "<<fmha_args.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_args.hdim_v<<std::endl;
    std::cout<<"nhead_q: "<<fmha_args.nhead_q<<std::endl;
    std::cout<<"nhead_k: "<<fmha_args.nhead_k<<std::endl;
    std::cout<<"scale: "<<fmha_args.scale<<std::endl;
    std::cout<<"stride_q: "<<fmha_args.stride_q<<std::endl;
    std::cout<<"stride_k: "<<fmha_args.stride_k<<std::endl;
    std::cout<<"stride_v: "<<fmha_args.stride_v<<std::endl;
    std::cout<<"stride_bias: "<<fmha_args.stride_bias<<std::endl;
    std::cout<<"stride_o: "<<fmha_args.stride_o<<std::endl;
    std::cout<<"stride_randval: "<<fmha_args.stride_randval<<std::endl;
    std::cout<<"stride_do: "<<fmha_args.stride_do<<std::endl;
    std::cout<<"stride_dq_acc: "<<fmha_args.stride_dq_acc<<std::endl;
    std::cout<<"stride_dq: "<<fmha_args.stride_dq<<std::endl;
    std::cout<<"stride_dk: "<<fmha_args.stride_dk<<std::endl;
    std::cout<<"stride_dv: "<<fmha_args.stride_dv<<std::endl;
    std::cout<<"stride_dbias: "<<fmha_args.stride_dbias<<std::endl;
    std::cout<<"nhead_stride_q: "<<fmha_args.nhead_stride_q<<std::endl;
    std::cout<<"nhead_stride_k: "<<fmha_args.nhead_stride_k<<std::endl;
    std::cout<<"nhead_stride_v: "<<fmha_args.nhead_stride_v<<std::endl;
    std::cout<<"nhead_stride_bias: "<<fmha_args.nhead_stride_bias<<std::endl;
    std::cout<<"nhead_stride_o: "<<fmha_args.nhead_stride_o<<std::endl;
    std::cout<<"nhead_stride_randval: "<<fmha_args.nhead_stride_randval<<std::endl;
    std::cout<<"nhead_stride_do: "<<fmha_args.nhead_stride_do<<std::endl;
    std::cout<<"nhead_stride_lsed: "<<fmha_args.nhead_stride_lsed<<std::endl;
    std::cout<<"nhead_stride_dq_acc: "<<fmha_args.nhead_stride_dq_acc<<std::endl;
    std::cout<<"nhead_stride_dq: "<<fmha_args.nhead_stride_dq<<std::endl;
    std::cout<<"nhead_stride_dk: "<<fmha_args.nhead_stride_dk<<std::endl;
    std::cout<<"nhead_stride_dv: "<<fmha_args.nhead_stride_dv<<std::endl;
    std::cout<<"nhead_stride_dbias: "<<fmha_args.nhead_stride_dbias<<std::endl;
    std::cout<<"batch_stride_q: "<<fmha_args.batch_stride_q<<std::endl;
    std::cout<<"batch_stride_k: "<<fmha_args.batch_stride_k<<std::endl;
    std::cout<<"batch_stride_v: "<<fmha_args.batch_stride_v<<std::endl;
    std::cout<<"batch_stride_bias: "<<fmha_args.batch_stride_bias<<std::endl;
    std::cout<<"batch_stride_o: "<<fmha_args.batch_stride_o<<std::endl;
    std::cout<<"batch_stride_randval: "<<fmha_args.batch_stride_randval<<std::endl;
    std::cout<<"batch_stride_do: "<<fmha_args.batch_stride_do<<std::endl;
    std::cout<<"batch_stride_lsed: "<<fmha_args.batch_stride_lsed<<std::endl;
    std::cout<<"batch_stride_dq_acc: "<<fmha_args.batch_stride_dq_acc<<std::endl;
    std::cout<<"batch_stride_dq: "<<fmha_args.batch_stride_dq<<std::endl;
    std::cout<<"batch_stride_dk: "<<fmha_args.batch_stride_dk<<std::endl;
    std::cout<<"batch_stride_dv: "<<fmha_args.batch_stride_dv<<std::endl;
    std::cout<<"batch_stride_dbias: "<<fmha_args.batch_stride_dbias<<std::endl;
    std::cout<<"window_size_left: "<<fmha_args.window_size_left<<std::endl;
    std::cout<<"window_size_right: "<<fmha_args.window_size_right<<std::endl;
    std::cout<<"mask_type: "<<fmha_args.mask_type<<std::endl;
    std::cout<<"p_drop: "<<fmha_args.p_drop<<std::endl;
    std::cout<<"p_undrop: "<<fmha_args.p_undrop<<std::endl;
    std::cout<<"dropout_seed: "<<std::get<0>(fmha_args.drop_seed_offset)<<std::endl;
    std::cout<<"dropout_offset: "<<std::get<1>(fmha_args.drop_seed_offset)<<std::endl;
  }
  float average_runtime = fmha_bwd(fmha_traits, fmha_args, stream_config);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn bwd pass.");
  }
  if(is_mqa_gqa){
    dim3 grid(b, s_kv, hg);
    dim3 block(d);
    if (ck_fused_attn_log_config){
      std::cout<<std::endl<<"run dk_dv_reduce: "<<std::endl;
      std::cout<<"dk_expanded_ptr: "<<dk_expanded_ptr<<std::endl;
      std::cout<<"dv_expanded_ptr: "<<dv_expanded_ptr<<std::endl;
      std::cout<<"stride_b_dkv_expanded: "<<stride_b_dkv_expanded<<std::endl;
      std::cout<<"stride_h_dkv_expanded: "<<stride_h_dkv_expanded<<std::endl;
      std::cout<<"stride_s_dkv_expanded: "<<stride_s_dkv_expanded<<std::endl;
      std::cout<<"dk_ptr: "<<dk_ptr<<std::endl;
      std::cout<<"dv_ptr: "<<dv_ptr<<std::endl;
      std::cout<<"stride_b_dk: "<<stride_b_dk<<std::endl;
      std::cout<<"stride_h_dk: "<<stride_h_dk<<std::endl;
      std::cout<<"stride_s_dk: "<<stride_s_dk<<std::endl;
    }
    CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
      hipLaunchKernelGGL(
        dk_dv_reduce<CK_TILE_TYPE>, grid, block, 0, stream,
        b, h, hg, s_kv, d,
        static_cast<CK_TILE_TYPE*>(dk_expanded_ptr),
        static_cast<CK_TILE_TYPE*>(dv_expanded_ptr),
        stride_b_dkv_expanded, stride_h_dkv_expanded, stride_s_dkv_expanded,
        static_cast<CK_TILE_TYPE*>(dk_ptr),
        static_cast<CK_TILE_TYPE*>(dv_ptr),
        stride_b_dk, stride_h_dk, stride_s_dk););
  }
  return hipSuccess;
}

}//namespace ck_fused_attn
