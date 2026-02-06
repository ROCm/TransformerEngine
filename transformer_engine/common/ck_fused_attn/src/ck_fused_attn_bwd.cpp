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
#include "mha_bwd.h"
#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{

// TODO: unify with binary search in TE/common/fused_attn(rocm)/util
// no device std::upper_bound
// in an increasing array with given size len, search for the index that:
// array[index] <= target < array[index+1]
// guaranteed that target >=0 and target <= cu_seqlen[end-1]
__forceinline__ __device__ int binary_search(int32_t target, const int32_t *array, uint64_t len) {
  int left = 1, right = len - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

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

  assert(hdim_idx<d);
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

// When d_qk != d_v, we need to reduce dk and dv separately
template<typename DataType>
__global__ void dk_or_dv_reduce(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_kv, uint64_t d,
  const DataType *dk_or_dv_expanded,
  uint64_t stride_b_dk_or_dv_expanded, uint64_t stride_h_dk_or_dv_expanded, uint64_t stride_s_dk_or_dv_expanded,
  DataType *dk_or_dv,
  //k,v, dk, dv guaranteed to have the same stride
  uint64_t stride_b_dk_or_dv, uint64_t stride_h_dk_or_dv, uint64_t stride_s_dk_or_dv){
  
  uint64_t batch_idx = blockIdx.x;
  uint64_t seqlen_idx = blockIdx.y;
  uint64_t head_k_or_v_idx = blockIdx.z;
  uint64_t hdim_idx = threadIdx.x;
  
  // h guaranteed to be multiples of hg
  uint64_t head_idx_offset = h / hg;

  float sum_dk_or_dv = 0.0f;

  assert(hdim_idx<d);
  uint64_t read_idx = batch_idx*stride_b_dk_or_dv_expanded + head_k_or_v_idx*head_idx_offset*stride_h_dk_or_dv_expanded + seqlen_idx*stride_s_dk_or_dv_expanded + hdim_idx;
  uint64_t write_idx = batch_idx*stride_b_dk_or_dv + head_k_or_v_idx*stride_h_dk_or_dv + seqlen_idx* stride_s_dk_or_dv + hdim_idx;
  
  for(uint64_t ii = 0; ii < head_idx_offset; ii++){
    // bf16 requires special casting in CK
    if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
      sum_dk_or_dv += ck_tile::bf16_to_float(dk_or_dv_expanded[read_idx]);
    }else{
      sum_dk_or_dv += dk_or_dv_expanded[read_idx];
    }
    read_idx += stride_h_dk_or_dv_expanded;
  }

  // bf16 requires special casting in CK
  if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
    dk_or_dv[write_idx] = ck_tile::float_to_bf16(sum_dk_or_dv);
  }else{
    dk_or_dv[write_idx] = sum_dk_or_dv;
  }
}

// define dk_dv_reduce function in THD layout only for fp16 and bf16 types
template<typename DataType>
__global__ void dk_dv_reduce_thd(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t d,
  const int32_t* cu_seqlen_kv_ptr,
  const int32_t* cu_seqlen_kv_padded_ptr,
  const DataType *dk_expanded,
  const DataType *dv_expanded,
  uint64_t stride_h_dkv_expanded, uint64_t stride_s_dkv_expanded,
  DataType *dk,
  DataType *dv,
  //k,v, dk, dv guaranteed to have the same stride
  uint64_t stride_h_dkv, uint64_t stride_s_dkv){

  uint64_t seqlen_idx = blockIdx.x;
  uint64_t head_k_idx = blockIdx.y;
  uint64_t hdim_idx = threadIdx.x;
  
  assert(hdim_idx<d);
  
  if(seqlen_idx >= *((cu_seqlen_kv_padded_ptr? cu_seqlen_kv_padded_ptr: cu_seqlen_kv_ptr)+b)){
    return;
  }
  if(cu_seqlen_kv_padded_ptr){
    uint64_t seq_idx = binary_search(seqlen_idx, cu_seqlen_kv_padded_ptr, b+1);
    uint64_t unpadded_size = cu_seqlen_kv_ptr[seq_idx+1] - cu_seqlen_kv_ptr[seq_idx];
    if(seqlen_idx >= cu_seqlen_kv_padded_ptr[seq_idx] + unpadded_size){
      return;
    }
  }
  // h guaranteed to be multiples of hg
  uint64_t head_idx_offset = h / hg;

  float sum_dk = 0.0f;
  float sum_dv = 0.0f;


  uint64_t read_idx = head_k_idx*head_idx_offset*stride_h_dkv_expanded + seqlen_idx*stride_s_dkv_expanded + hdim_idx;
  uint64_t write_idx = head_k_idx*stride_h_dkv + seqlen_idx* stride_s_dkv + hdim_idx;
  
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

// When d_qk != d_v, we need to reduce dk and dv separately
template<typename DataType>
__global__ void dk_or_dv_reduce_thd(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t d,
  const int32_t* cu_seqlen_kv_ptr,
  const int32_t* cu_seqlen_kv_padded_ptr,
  const DataType *dk_or_dv_expanded,
  uint64_t stride_h_dk_or_dv_expanded, uint64_t stride_s_dk_or_dv_expanded,
  DataType *dk_or_dv,
  //k,v, dk, dv guaranteed to have the same stride
  uint64_t stride_h_dk_or_dv, uint64_t stride_s_dk_or_dv){

  uint64_t seqlen_idx = blockIdx.x;
  uint64_t head_k_or_v_idx = blockIdx.y;
  uint64_t hdim_idx = threadIdx.x;
  
  assert(hdim_idx<d);

  if(seqlen_idx >= *((cu_seqlen_kv_padded_ptr? cu_seqlen_kv_padded_ptr: cu_seqlen_kv_ptr)+b)){
    return;
  }
  if(cu_seqlen_kv_padded_ptr){
    uint64_t seq_idx = binary_search(seqlen_idx, cu_seqlen_kv_padded_ptr, b+1);
    uint64_t unpadded_size = cu_seqlen_kv_ptr[seq_idx+1] - cu_seqlen_kv_ptr[seq_idx];
    if(seqlen_idx >= cu_seqlen_kv_padded_ptr[seq_idx] + unpadded_size){
      return;
    }
  }
  // h guaranteed to be multiples of hg
  uint64_t head_idx_offset = h / hg;

  float sum_dk_or_dv = 0.0f;

  uint64_t read_idx = head_k_or_v_idx*head_idx_offset*stride_h_dk_or_dv_expanded + seqlen_idx*stride_s_dk_or_dv_expanded + hdim_idx;
  uint64_t write_idx = head_k_or_v_idx*stride_h_dk_or_dv + seqlen_idx* stride_s_dk_or_dv + hdim_idx;
  
  for(uint64_t ii = 0; ii < head_idx_offset; ii++){
    // bf16 requires special casting in CK
    if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
      sum_dk_or_dv += ck_tile::bf16_to_float(dk_or_dv_expanded[read_idx]);
    }else{
      sum_dk_or_dv += dk_or_dv_expanded[read_idx];
    }
    read_idx += stride_h_dk_or_dv_expanded;
  }

  // bf16 requires special casting in CK
  if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
    dk_or_dv[write_idx] = ck_tile::float_to_bf16(sum_dk_or_dv);
  }else{
    dk_or_dv[write_idx] = sum_dk_or_dv;
  }
}


// define dbias_reduce functions only for fp16 and bf16 types
template<typename DataType>
__global__ void dbias_reduce_11ss(
  uint64_t b, uint64_t h, uint64_t s_q, uint64_t s_kv,
  const DataType *dbias_expanded,
  DataType *dbias){
  
  const uint64_t stride_h = s_q*s_kv;
  const uint64_t stride_b = h*s_q*s_kv;
  for(uint64_t ss_idx = blockIdx.x*blockDim.x + threadIdx.x; ss_idx < s_q*s_kv; ss_idx += blockDim.x * gridDim.x){
    //sum over b, h dims both
    float sum_dbias = 0.0f;
    for(uint64_t b_idx = 0; b_idx< b; b_idx++){
      for(uint64_t h_idx = 0; h_idx < h; h_idx++){
        if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
          // bf16 requires special casting in CK
          sum_dbias += ck_tile::bf16_to_float(dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx]);
        }else{
          sum_dbias += dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx];
        }
      }
    }
    if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
      dbias[ss_idx] = ck_tile::float_to_bf16(sum_dbias);
    }else{
      dbias[ss_idx] = sum_dbias;
    }
  }
}

// define dbias_reduce functions only for fp16 and bf16 types
template<typename DataType>
__global__ void dbias_reduce_1hss(
  uint64_t b, uint64_t h, uint64_t s_q, uint64_t s_kv,
  const DataType *dbias_expanded,
  DataType *dbias){
  
  const uint64_t stride_h = s_q*s_kv;
  const uint64_t stride_b = h*s_q*s_kv;
  for(uint64_t ss_idx = blockIdx.x*blockDim.x + threadIdx.x; ss_idx < s_q*s_kv; ss_idx += blockDim.x * gridDim.x){
    for(uint64_t h_idx = 0; h_idx < h; h_idx++){
      //sum over b dims only
      float sum_dbias = 0.0f;
      for(uint64_t b_idx = 0; b_idx< b; b_idx++){
        if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
          // bf16 requires special casting in CK
          sum_dbias += ck_tile::bf16_to_float(dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx]);
        }else{
          sum_dbias += dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx];
        }
      }
      if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
        dbias[ss_idx + h_idx*stride_h] = ck_tile::float_to_bf16(sum_dbias);
      }else{
        dbias[ss_idx + h_idx*stride_h] = sum_dbias;
      }
    }
  }
}

// define dbias_reduce functions only for fp16 and bf16 types
template<typename DataType>
__global__ void dbias_reduce_b1ss(
  uint64_t b, uint64_t h, uint64_t s_q, uint64_t s_kv,
  const DataType *dbias_expanded,
  DataType *dbias){
  
  const uint64_t stride_h = s_q*s_kv;
  const uint64_t stride_b = h*s_q*s_kv;
  for(uint64_t ss_idx = blockIdx.x*blockDim.x + threadIdx.x; ss_idx < s_q*s_kv; ss_idx += blockDim.x * gridDim.x){
    for(uint64_t b_idx = 0; b_idx< b; b_idx++){
      //sum over h dims only
      float sum_dbias = 0.0f;
      for(uint64_t h_idx = 0; h_idx < h; h_idx++){
        if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
          // bf16 requires special casting in CK
          sum_dbias += ck_tile::bf16_to_float(dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx]);
        }else{
          sum_dbias += dbias_expanded[b_idx*stride_b + h_idx*stride_h+ss_idx];
        }
      }
      if constexpr (std::is_same_v<DataType, ck_tile::bf16_t>){
        dbias[ss_idx + b_idx*stride_h] = ck_tile::float_to_bf16(sum_dbias);
      }else{
        dbias[ss_idx + b_idx*stride_h] = sum_dbias;
      }
    }
  }
}

// print the fmha_traits and args passed into ck apis
void log_bwd_config(const char* func_name, const aiter::mha_bwd_args& fmha_args, bool ck_log_config){
  if (!ck_log_config) {
    return;
  }

  auto log_value = [](const char* label, const auto& value) {
    std::cout << label << ": " << value << "\n";
  };

  std::cout << "\n" << func_name << "\n";

  // fmha_traits debug
  std::cout << "\nfmha_traits: \n";
  log_value("hdim_q", fmha_args.hdim_q);
  log_value("hdim_v", fmha_args.hdim_v);
  log_value("data_type", fmha_args.data_type);
  log_value("is_group_mode", fmha_args.is_group_mode);
  log_value("has_dbias", fmha_args.has_dbias);
  log_value("has_dropout", fmha_args.has_dropout);
  log_value("is_store_randval", fmha_args.is_store_randval);
  log_value("is_deterministic", fmha_args.is_deterministic);
  log_value("use_asm_v3", fmha_args.use_asm_v3);
  log_value("v3_atomic_fp32", fmha_args.v3_atomic_fp32);
  log_value("v3_bf16_cvt", fmha_args.v3_bf16_cvt);

  // fmha_args debug
  std::cout << "\nfmha_args: \n";
  log_value("q_ptr", fmha_args.q_ptr);
  log_value("k_ptr", fmha_args.k_ptr);
  log_value("v_ptr", fmha_args.v_ptr);
  log_value("bias_ptr", fmha_args.bias_ptr);
  log_value("o_ptr", fmha_args.o_ptr);
  log_value("lse_ptr", fmha_args.lse_ptr);
  log_value("do_ptr", fmha_args.do_ptr);
  log_value("d_ptr", fmha_args.d_ptr);
  log_value("rand_val_ptr", fmha_args.rand_val_ptr);
  log_value("dq_ptr", fmha_args.dq_ptr);
  log_value("dk_ptr", fmha_args.dk_ptr);
  log_value("dv_ptr", fmha_args.dv_ptr);
  log_value("dbias_ptr", fmha_args.dbias_ptr);
  log_value("dq_acc_ptr", fmha_args.dq_acc_ptr);

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
  log_value("max_seqlen_k", fmha_args.max_seqlen_k);
  log_value("hdim_q", fmha_args.hdim_q);
  log_value("hdim_v", fmha_args.hdim_v);
  log_value("nhead_q", fmha_args.nhead_q);
  log_value("nhead_k", fmha_args.nhead_k);
  log_value("scale", fmha_args.scale);
  log_value("stride_q", fmha_args.stride_q);
  log_value("stride_k", fmha_args.stride_k);
  log_value("stride_v", fmha_args.stride_v);
  log_value("stride_bias", fmha_args.stride_bias);
  log_value("stride_o", fmha_args.stride_o);
  log_value("stride_randval", fmha_args.stride_randval);
  log_value("stride_do", fmha_args.stride_do);
  log_value("stride_dq_acc", fmha_args.stride_dq_acc);
  log_value("stride_dq", fmha_args.stride_dq);
  log_value("stride_dk", fmha_args.stride_dk);
  log_value("stride_dv", fmha_args.stride_dv);
  log_value("stride_dbias", fmha_args.stride_dbias);
  log_value("nhead_stride_q", fmha_args.nhead_stride_q);
  log_value("nhead_stride_k", fmha_args.nhead_stride_k);
  log_value("nhead_stride_v", fmha_args.nhead_stride_v);
  log_value("nhead_stride_bias", fmha_args.nhead_stride_bias);
  log_value("nhead_stride_o", fmha_args.nhead_stride_o);
  log_value("nhead_stride_randval", fmha_args.nhead_stride_randval);
  log_value("nhead_stride_do", fmha_args.nhead_stride_do);
  log_value("nhead_stride_lsed", fmha_args.nhead_stride_lsed);
  log_value("nhead_stride_dq_acc", fmha_args.nhead_stride_dq_acc);
  log_value("nhead_stride_dq", fmha_args.nhead_stride_dq);
  log_value("nhead_stride_dk", fmha_args.nhead_stride_dk);
  log_value("nhead_stride_dv", fmha_args.nhead_stride_dv);
  log_value("nhead_stride_dbias", fmha_args.nhead_stride_dbias);
  log_value("batch_stride_q", fmha_args.batch_stride_q);
  log_value("batch_stride_k", fmha_args.batch_stride_k);
  log_value("batch_stride_v", fmha_args.batch_stride_v);
  log_value("batch_stride_bias", fmha_args.batch_stride_bias);
  log_value("batch_stride_o", fmha_args.batch_stride_o);
  log_value("batch_stride_randval", fmha_args.batch_stride_randval);
  log_value("batch_stride_do", fmha_args.batch_stride_do);
  log_value("batch_stride_lsed", fmha_args.batch_stride_lsed);
  log_value("batch_stride_dq_acc", fmha_args.batch_stride_dq_acc);
  log_value("batch_stride_dq", fmha_args.batch_stride_dq);
  log_value("batch_stride_dk", fmha_args.batch_stride_dk);
  log_value("batch_stride_dv", fmha_args.batch_stride_dv);
  log_value("batch_stride_dbias", fmha_args.batch_stride_dbias);
  log_value("window_size_left", fmha_args.window_size_left);
  log_value("window_size_right", fmha_args.window_size_right);
  log_value("mask_type", fmha_args.mask_type);
  log_value("bias_type", fmha_args.bias_type);
  log_value("p_drop", fmha_args.p_drop);
  log_value("p_undrop", fmha_args.p_undrop);
  log_value(
    "dropout_seed_ptr",
    std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))
  );
  log_value(
    "dropout_offset_ptr",
    std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))
  );
}

void dump_bwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-bwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
}

hipError_t _ck_attn_bwd_impl(  
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv,
  uint64_t d_qk, uint64_t d_v,
  uint64_t bias_b, uint64_t bias_h,
  uint64_t max_tokens_q, uint64_t max_tokens_kv,
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
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr,
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  float scaling_factor, float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  BiasType attn_bias_type,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* dq_ptr, 
  uint64_t stride_b_dq, uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dq_acc_ptr,
  void* dk_expanded_ptr,
  void* dv_expanded_ptr,
  uint64_t stride_b_dk_expanded, uint64_t stride_h_dk_expanded, uint64_t stride_s_dk_expanded,
  uint64_t stride_b_dv_expanded, uint64_t stride_h_dv_expanded, uint64_t stride_s_dv_expanded,
  void* dk_ptr, 
  uint64_t stride_b_dk, uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_b_dv, uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* dbias_expanded_ptr,
  void* dbias_ptr,
  void* lse_workspace_ptr,
  bool deterministic,
  bool uses_bwd_v3,
  bool is_v3_atomic_fp32,
  int how_v3_bf16_cvt,
  bool is_group_mode,
  const char* func_name,
  bool ck_log_config,
  hipStream_t stream){

  bool has_dropout = (dropout_probability > 0.f);
  bool has_dbias = dbias_ptr != nullptr;
  bool is_mqa_gqa = (h > hg);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t seqlen_q = s_q;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d_qk;
  ck_tile::index_t seqlen_k = s_kv;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d_v;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float p_drop = dropout_probability;
  float p_undrop = 1.0 - p_drop;
  bool s_randval = false;

  bias_enum bias_type = bias_enum::no_bias;
  BiasShape bias_shape = BiasShape::k11SS;
  if (!is_group_mode) {
    std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(attn_bias_type, b, h, bias_b, bias_h);
  }

  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);

  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");

  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, ck_log_config};

  std::string data_type_str = get_data_type_str(dtype);

  aiter::mha_bwd_args fmha_args{};
  fmha_args.mask_type = static_cast<int>(mask_type);
  fmha_args.use_asm_v3 = uses_bwd_v3;
  fmha_args.v3_atomic_fp32 = is_v3_atomic_fp32;
  fmha_args.v3_bf16_cvt = how_v3_bf16_cvt;
  fmha_args.v3_api_check = false;

  fmha_args.hdim_q = hdim_q;
  fmha_args.hdim_v = hdim_v;
  fmha_args.data_type = data_type_str;
  fmha_args.is_group_mode = is_group_mode;
  fmha_args.ck_mask_type = static_cast<int>(mask_type);
  fmha_args.bias_type = static_cast<int>(bias_type);
  fmha_args.has_dbias = (!is_group_mode) && has_dbias;
  fmha_args.has_dropout = has_dropout;
  fmha_args.is_store_randval = s_randval;
  fmha_args.is_deterministic = deterministic;

  fmha_args.q_ptr = q_ptr;
  fmha_args.k_ptr = k_ptr;
  fmha_args.v_ptr = v_ptr;
  fmha_args.bias_ptr = (bias_type==bias_enum::no_bias || is_group_mode) ? nullptr
                         : (bias_type==bias_enum::alibi? alibi_slope_ptr : bias_ptr);
  fmha_args.o_ptr = o_ptr;
  fmha_args.lse_ptr = lse_ptr;
  fmha_args.do_ptr = do_ptr;
  fmha_args.d_ptr = lse_workspace_ptr;
  fmha_args.rand_val_ptr = nullptr;
  fmha_args.dq_ptr = dq_ptr;
  fmha_args.dk_ptr = is_mqa_gqa? dk_expanded_ptr:dk_ptr;
  fmha_args.dv_ptr = is_mqa_gqa? dv_expanded_ptr:dv_ptr;
  fmha_args.dbias_ptr = ((!is_group_mode) && has_dbias)
                          ? (bias_shape==BiasShape::kBHSS ? dbias_ptr: dbias_expanded_ptr)
                          : nullptr;
  fmha_args.dq_acc_ptr = dq_acc_ptr;

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
  }

  fmha_args.seqlen_q = is_group_mode ? max_seqlen_q : seqlen_q;
  fmha_args.seqlen_k = is_group_mode ? max_seqlen_k : seqlen_k;
  fmha_args.batch = batch;
  fmha_args.max_seqlen_q = max_seqlen_q;
  fmha_args.max_seqlen_k = max_seqlen_k;
  fmha_args.nhead_q = nhead;
  fmha_args.nhead_k = nhead_k;
  fmha_args.scale = scale_s;

  // setup stride_* arguments
  fmha_args.stride_q = stride_s_q;
  fmha_args.stride_k = stride_s_k;
  fmha_args.stride_v = stride_s_v;
  fmha_args.stride_bias = (!is_group_mode && bias_type!=bias_enum::alibi) ? max_seqlen_k : 0;
  fmha_args.stride_o = stride_s_o;
  fmha_args.stride_randval = max_seqlen_k;
  fmha_args.stride_do = stride_s_do;
  fmha_args.stride_dq_acc = d_qk;
  fmha_args.stride_dq = stride_s_dq;
  fmha_args.stride_dk = is_mqa_gqa? stride_s_dk_expanded:stride_s_dk;
  fmha_args.stride_dv = is_mqa_gqa? stride_s_dv_expanded:stride_s_dv;
  fmha_args.stride_dbias = (!is_group_mode && bias_type!=bias_enum::alibi) ? max_seqlen_k : 0;

  // setup nhead_stride_* arguments
  fmha_args.nhead_stride_q = stride_h_q;
  fmha_args.nhead_stride_k = stride_h_k;
  fmha_args.nhead_stride_v = stride_h_v;
  fmha_args.nhead_stride_bias = (!is_group_mode && (bias_shape==BiasShape::k1HSS || bias_shape==BiasShape::kBHSS))
                                  ? max_seqlen_q * max_seqlen_k
                                  : 0;
  fmha_args.nhead_stride_o = stride_h_o;
  fmha_args.nhead_stride_randval = is_group_mode ? 0 : seqlen_q * max_seqlen_k;
  fmha_args.nhead_stride_do = stride_h_do;
  fmha_args.nhead_stride_lsed = is_group_mode ? max_tokens_q : max_seqlen_q;
  fmha_args.nhead_stride_dq_acc = static_cast<int64_t>((is_group_mode ? max_tokens_q : s_q) * d_qk);
  fmha_args.nhead_stride_dq = stride_h_dq;
  fmha_args.nhead_stride_dk = is_mqa_gqa? stride_h_dk_expanded:stride_h_dk;
  fmha_args.nhead_stride_dv = is_mqa_gqa? stride_h_dv_expanded:stride_h_dv;
  fmha_args.nhead_stride_dbias = (!is_group_mode) ? max_seqlen_q * max_seqlen_k : 0;

  // setup batch_stride_* arguments
  fmha_args.batch_stride_q = is_group_mode ? 0 : stride_b_q;
  fmha_args.batch_stride_k = is_group_mode ? 0 : stride_b_k;
  fmha_args.batch_stride_v = is_group_mode ? 0 : stride_b_v;
  fmha_args.batch_stride_bias = (!is_group_mode && (bias_shape==BiasShape::k11SS || bias_shape==BiasShape::k1HSS))
                                  ? 0
                                  : (is_group_mode ? 0 : bias_h * max_seqlen_q * max_seqlen_k);
  fmha_args.batch_stride_o = is_group_mode ? 0 : stride_b_o;
  fmha_args.batch_stride_randval = is_group_mode ? 0 : nhead * seqlen_q * max_seqlen_k;
  fmha_args.batch_stride_do = is_group_mode ? 0 : stride_b_do;
  fmha_args.batch_stride_lsed = is_group_mode ? 0 : nhead * max_seqlen_q;
  fmha_args.batch_stride_dq_acc = is_group_mode ? 0 : static_cast<int64_t>(h * s_q * d_qk);
  fmha_args.batch_stride_dq = is_group_mode ? 0 : stride_b_dq;
  fmha_args.batch_stride_dk = is_group_mode ? 0 : (is_mqa_gqa? stride_b_dk_expanded:stride_b_dk);
  fmha_args.batch_stride_dv = is_group_mode ? 0 : (is_mqa_gqa? stride_b_dv_expanded:stride_b_dv);
  fmha_args.batch_stride_dbias = is_group_mode ? 0 : h * max_seqlen_q * max_seqlen_k;
  fmha_args.split_stride_dq_acc = static_cast<int>(is_group_mode ? (max_tokens_q * h * d_qk) : (b * h * s_q * d_qk));

  fmha_args.window_size_left = left;
  fmha_args.window_size_right = right;
  fmha_args.p_drop = p_drop;
  fmha_args.p_undrop = p_undrop;
  fmha_args.drop_seed_offset = std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr};

  // modify the max_seqlen_q for better performance in 0-length cases
  // lse_thd_ptr used as buffer
  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")) {
    if(std::string(env_p) == "1"){
      if(ck_log_config){
        std::cout << "attn_bwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(b, cu_seqlen_q_ptr, nullptr, lse_workspace_ptr, stream);
      fmha_args.max_seqlen_k = get_runtime_max_seqlen(b, cu_seqlen_kv_ptr, nullptr, lse_workspace_ptr, stream);
    }
  }

  // print ck traits and args when needed
  log_bwd_config(func_name, fmha_args, ck_log_config);

  float average_runtime = aiter::mha_bwd(fmha_args, stream_config);
  if(dump_path){
    dump_bwd_timings(dump_path, average_runtime);
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn bwd pass.");
  }
  return hipSuccess;
}
hipError_t ck_attn_bwd(  
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
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr, 
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  float scaling_factor, float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  BiasType attn_bias_type,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* dq_ptr, 
  uint64_t stride_b_dq, uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dq_acc_ptr,
  void* dk_expanded_ptr,
  void* dv_expanded_ptr,
  uint64_t stride_b_dk_expanded, uint64_t stride_h_dk_expanded, uint64_t stride_s_dk_expanded,
  uint64_t stride_b_dv_expanded, uint64_t stride_h_dv_expanded, uint64_t stride_s_dv_expanded,
  void* dk_ptr, 
  uint64_t stride_b_dk, uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_b_dv, uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* dbias_expanded_ptr,
  void* dbias_ptr,
  void* lse_workspace_ptr,
  bool deterministic,
  bool uses_bwd_v3,
  bool is_v3_atomic_fp32,
  int how_v3_bf16_cvt,
  hipStream_t stream){

  bool has_dropout = (dropout_probability > 0.f);
  bool has_dbias = dbias_ptr!=nullptr;
  bool is_mqa_gqa = (h > hg);
  bias_enum bias_type;
  BiasShape bias_shape; 
  std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(attn_bias_type, b, h, bias_b, bias_h);

  bool ck_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_log_config = true;
  }

  hipError_t impl_status = _ck_attn_bwd_impl(
    dtype,
    b, h, hg, s_q, s_kv, d_qk, d_v,
    bias_b, bias_h,
    s_q, s_kv,
    q_ptr,
    stride_b_q, stride_h_q, stride_s_q,
    k_ptr,
    stride_b_k, stride_h_k, stride_s_k,
    v_ptr,
    stride_b_v, stride_h_v, stride_s_v,
    bias_ptr,
    alibi_slope_ptr,
    nullptr, nullptr,
    nullptr, nullptr,
    o_ptr,
    stride_b_o, stride_h_o, stride_s_o,
    lse_ptr,
    do_ptr,
    stride_b_do, stride_h_do, stride_s_do,
    scaling_factor, dropout_probability,
    philox_seed_ptr, philox_offset_ptr,
    attn_bias_type,
    attn_mask_type,
    window_size_left, window_size_right,
    dq_ptr,
    stride_b_dq, stride_h_dq, stride_s_dq,
    dq_acc_ptr,
    dk_expanded_ptr,
    dv_expanded_ptr,
    stride_b_dk_expanded, stride_h_dk_expanded, stride_s_dk_expanded,
    stride_b_dv_expanded, stride_h_dv_expanded, stride_s_dv_expanded,
    dk_ptr,
    stride_b_dk, stride_h_dk, stride_s_dk,
    dv_ptr,
    stride_b_dv, stride_h_dv, stride_s_dv,
    dbias_expanded_ptr,
    dbias_ptr,
    lse_workspace_ptr,
    deterministic,
    uses_bwd_v3,
    is_v3_atomic_fp32,
    how_v3_bf16_cvt,
    false,
    __FUNCTION__,
    ck_log_config,
    stream);
  if (impl_status != hipSuccess) {
    return impl_status;
  }
  if(is_mqa_gqa){
    dim3 grid(b, s_kv, hg);
    if (d_qk == d_v) {
      dim3 block(d_qk);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_dv_reduce: "<<std::endl;
        std::cout<<"dk_expanded_ptr: "<<dk_expanded_ptr<<std::endl;
        std::cout<<"dv_expanded_ptr: "<<dv_expanded_ptr<<std::endl;
        std::cout<<"stride_b_dkv_expanded: "<<stride_b_dk_expanded<<std::endl;
        std::cout<<"stride_h_dkv_expanded: "<<stride_h_dk_expanded<<std::endl;
        std::cout<<"stride_s_dkv_expanded: "<<stride_s_dk_expanded<<std::endl;
        std::cout<<"dk_ptr: "<<dk_ptr<<std::endl;
        std::cout<<"dv_ptr: "<<dv_ptr<<std::endl;
        std::cout<<"stride_b_dk: "<<stride_b_dk<<std::endl;
        std::cout<<"stride_h_dk: "<<stride_h_dk<<std::endl;
        std::cout<<"stride_s_dk: "<<stride_s_dk<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_dv_reduce<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, hg, s_kv, d_qk,
          static_cast<CK_TILE_TYPE*>(dk_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dv_expanded_ptr),
          stride_b_dk_expanded, stride_h_dk_expanded, stride_s_dk_expanded,
          static_cast<CK_TILE_TYPE*>(dk_ptr),
          static_cast<CK_TILE_TYPE*>(dv_ptr),
          stride_b_dk, stride_h_dk, stride_s_dk););
    } else {
      dim3 block_dk(d_qk);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_or_dv_reduce on dk: "<<std::endl;
        std::cout<<"dk_expanded_ptr: "<<dk_expanded_ptr<<std::endl;
        std::cout<<"stride_b_dk_expanded: "<<stride_b_dk_expanded<<std::endl;
        std::cout<<"stride_h_dk_expanded: "<<stride_h_dk_expanded<<std::endl;
        std::cout<<"stride_s_dk_expanded: "<<stride_s_dk_expanded<<std::endl;
        std::cout<<"dk_ptr: "<<dk_ptr<<std::endl;
        std::cout<<"stride_b_dk: "<<stride_b_dk<<std::endl;
        std::cout<<"stride_h_dk: "<<stride_h_dk<<std::endl;
        std::cout<<"stride_s_dk: "<<stride_s_dk<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_or_dv_reduce<CK_TILE_TYPE>, grid, block_dk, 0, stream,
          b, h, hg, s_kv, d_qk,
          static_cast<CK_TILE_TYPE*>(dk_expanded_ptr),
          stride_b_dk_expanded, stride_h_dk_expanded, stride_s_dk_expanded,
          static_cast<CK_TILE_TYPE*>(dk_ptr),
          stride_b_dk, stride_h_dk, stride_s_dk););

      dim3 block_dv(d_v);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_or_dv_reduce on dv: "<<std::endl;
        std::cout<<"dv_expanded_ptr: "<<dv_expanded_ptr<<std::endl;
        std::cout<<"stride_b_dv_expanded: "<<stride_b_dv_expanded<<std::endl;
        std::cout<<"stride_h_dv_expanded: "<<stride_h_dv_expanded<<std::endl;
        std::cout<<"stride_s_dv_expanded: "<<stride_s_dv_expanded<<std::endl;
        std::cout<<"dv_ptr: "<<dv_ptr<<std::endl;
        std::cout<<"stride_b_dv: "<<stride_b_dv<<std::endl;
        std::cout<<"stride_h_dv: "<<stride_h_dv<<std::endl;
        std::cout<<"stride_s_dv: "<<stride_s_dv<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_or_dv_reduce<CK_TILE_TYPE>, grid, block_dv, 0, stream,
          b, h, hg, s_kv, d_v,
          static_cast<CK_TILE_TYPE*>(dv_expanded_ptr),
          stride_b_dv_expanded, stride_h_dv_expanded, stride_s_dv_expanded,
          static_cast<CK_TILE_TYPE*>(dv_ptr),
          stride_b_dv, stride_h_dv, stride_s_dv););
    }
  }
  if(has_dbias && bias_shape!=BiasShape::kBHSS){
    // reduction kernels required for 11SS, 1HSS, and B1SS
    assert(dbias_ptr!=dbias_expanded_ptr);
    constexpr int THREADS_PER_BLOCK = 1024;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(1.0 * s_q * s_kv/THREADS_PER_BLOCK));
    if(bias_shape==BiasShape::k11SS){
      if (ck_log_config){
        std::cout<<std::endl<<"run dbias_reduce_11SS: "<<std::endl;
        std::cout<<"dbias_ptr: "<<dbias_ptr<<std::endl;
        std::cout<<"dbias_expanded_ptr: "<<dbias_expanded_ptr<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_11ss<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, s_q, s_kv,
          static_cast<CK_TILE_TYPE*>(dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dbias_ptr));); 
    }else if(bias_shape==BiasShape::k1HSS){
      if (ck_log_config){
        std::cout<<std::endl<<"run dbias_reduce_1HSS: "<<std::endl;
        std::cout<<"dbias_ptr: "<<dbias_ptr<<std::endl;
        std::cout<<"dbias_expanded_ptr: "<<dbias_expanded_ptr<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_1hss<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, s_q, s_kv,
          static_cast<CK_TILE_TYPE*>(dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dbias_ptr));); 
    }else if(bias_shape==BiasShape::kB1SS){
      if (ck_log_config){
        std::cout<<std::endl<<"run dbias_reduce_B1SS: "<<std::endl;
        std::cout<<"dbias_ptr: "<<dbias_ptr<<std::endl;
        std::cout<<"dbias_expanded_ptr: "<<dbias_expanded_ptr<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_b1ss<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, s_q, s_kv,
          static_cast<CK_TILE_TYPE*>(dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dbias_ptr));); 
    }
  }
  return hipSuccess;
}

hipError_t ck_attn_varlen_bwd(  
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v,
  uint64_t max_tokens_q, uint64_t max_tokens_kv,
  const void* q_ptr, 
  uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_h_v, uint64_t stride_s_v,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_kv_ptr,
  const void* cu_seqlen_q_padded_ptr, const void* cu_seqlen_kv_padded_ptr,
  const void* o_ptr, 
  uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_thd_ptr, 
  const void* do_ptr, 
  uint64_t stride_h_do, uint64_t stride_s_do,
  float scaling_factor, float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* dq_ptr, 
  uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dq_acc_ptr,
  void* dk_expanded_ptr,
  void* dv_expanded_ptr,
  uint64_t stride_h_dk_expanded, uint64_t stride_s_dk_expanded,
  uint64_t stride_h_dv_expanded, uint64_t stride_s_dv_expanded,
  void* dk_ptr, 
  uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* lse_workspace_ptr,
  bool deterministic,
  bool uses_bwd_v3,
  bool is_v3_atomic_fp32,
  int how_v3_bf16_cvt,
  hipStream_t stream){
  bool is_mqa_gqa = (h > hg);

  bool ck_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_log_config = true;
  }

  hipError_t impl_status = _ck_attn_bwd_impl(
    dtype,
    b, h, hg, s_q, s_kv, d_qk, d_v,
    0, 0,
    max_tokens_q, max_tokens_kv,
    q_ptr,
    0, stride_h_q, stride_s_q,
    k_ptr,
    0, stride_h_k, stride_s_k,
    v_ptr,
    0, stride_h_v, stride_s_v,
    nullptr,
    nullptr,
    cu_seqlen_q_ptr, cu_seqlen_kv_ptr,
    cu_seqlen_q_padded_ptr, cu_seqlen_kv_padded_ptr,
    o_ptr,
    0, stride_h_o, stride_s_o,
    lse_thd_ptr,
    do_ptr,
    0, stride_h_do, stride_s_do,
    scaling_factor, dropout_probability,
    philox_seed_ptr, philox_offset_ptr,
    BiasType::no_bias,
    attn_mask_type,
    window_size_left, window_size_right,
    dq_ptr,
    0, stride_h_dq, stride_s_dq,
    dq_acc_ptr,
    dk_expanded_ptr,
    dv_expanded_ptr,
    0, stride_h_dk_expanded, stride_s_dk_expanded,
    0, stride_h_dv_expanded, stride_s_dv_expanded,
    dk_ptr,
    0, stride_h_dk, stride_s_dk,
    dv_ptr,
    0, stride_h_dv, stride_s_dv,
    nullptr,
    nullptr,
    lse_workspace_ptr,
    deterministic,
    uses_bwd_v3,
    is_v3_atomic_fp32,
    how_v3_bf16_cvt,
    true,
    __FUNCTION__,
    ck_log_config,
    stream);
  if (impl_status != hipSuccess) {
    return impl_status;
  }
  if(is_mqa_gqa){
    dim3 grid(max_tokens_kv, hg);
    if (d_qk == d_v) {
      dim3 block(d_qk);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_dv_reduce_thd: "<<std::endl;
        std::cout<<"cu_seqlen_kv_ptr: "<<cu_seqlen_kv_ptr<<std::endl;
        std::cout<<"cu_seqlen_kv_padded_ptr: "<<cu_seqlen_kv_padded_ptr<<std::endl;
        std::cout<<"dk_expanded_ptr: "<<dk_expanded_ptr<<std::endl;
        std::cout<<"dv_expanded_ptr: "<<dv_expanded_ptr<<std::endl;
        std::cout<<"stride_h_dkv_expanded: "<<stride_h_dk_expanded<<std::endl;
        std::cout<<"stride_s_dkv_expanded: "<<stride_s_dk_expanded<<std::endl;
        std::cout<<"dk_ptr: "<<dk_ptr<<std::endl;
        std::cout<<"dv_ptr: "<<dv_ptr<<std::endl;
        std::cout<<"stride_h_dk: "<<stride_h_dk<<std::endl;
        std::cout<<"stride_s_dk: "<<stride_s_dk<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_dv_reduce_thd<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, hg, d_qk,
          static_cast<const int32_t*>(cu_seqlen_kv_ptr),
          static_cast<const int32_t*>(cu_seqlen_kv_padded_ptr),
          static_cast<CK_TILE_TYPE*>(dk_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dv_expanded_ptr),
          stride_h_dk_expanded, stride_s_dk_expanded,
          static_cast<CK_TILE_TYPE*>(dk_ptr),
          static_cast<CK_TILE_TYPE*>(dv_ptr),
          stride_h_dk, stride_s_dk););
    } else {
      dim3 block_dk(d_qk);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_or_dv_reduce_thd on dk: "<<std::endl;
        std::cout<<"cu_seqlen_kv_ptr: "<<cu_seqlen_kv_ptr<<std::endl;
        std::cout<<"cu_seqlen_kv_padded_ptr: "<<cu_seqlen_kv_padded_ptr<<std::endl;
        std::cout<<"dk_expanded_ptr: "<<dk_expanded_ptr<<std::endl;
        std::cout<<"stride_h_dk_expanded: "<<stride_h_dk_expanded<<std::endl;
        std::cout<<"stride_s_dk_expanded: "<<stride_s_dk_expanded<<std::endl;
        std::cout<<"dk_ptr: "<<dk_ptr<<std::endl;
        std::cout<<"stride_h_dk: "<<stride_h_dk<<std::endl;
        std::cout<<"stride_s_dk: "<<stride_s_dk<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_or_dv_reduce_thd<CK_TILE_TYPE>, grid, block_dk, 0, stream,
          b, h, hg, d_qk,
          static_cast<const int32_t*>(cu_seqlen_kv_ptr),
          static_cast<const int32_t*>(cu_seqlen_kv_padded_ptr),
          static_cast<CK_TILE_TYPE*>(dk_expanded_ptr),
          stride_h_dk_expanded, stride_s_dk_expanded,
          static_cast<CK_TILE_TYPE*>(dk_ptr),
          stride_h_dk, stride_s_dk););

      dim3 block_dv(d_v);
      if (ck_log_config){
        std::cout<<std::endl<<"run dk_or_dv_reduce_thd on dv: "<<std::endl;
        std::cout<<"cu_seqlen_kv_ptr: "<<cu_seqlen_kv_ptr<<std::endl;
        std::cout<<"cu_seqlen_kv_padded_ptr: "<<cu_seqlen_kv_padded_ptr<<std::endl;
        std::cout<<"dv_expanded_ptr: "<<dv_expanded_ptr<<std::endl;
        std::cout<<"stride_h_dv_expanded: "<<stride_h_dv_expanded<<std::endl;
        std::cout<<"stride_s_dv_expanded: "<<stride_s_dv_expanded<<std::endl;
        std::cout<<"dv_ptr: "<<dv_ptr<<std::endl;
        std::cout<<"stride_h_dv: "<<stride_h_dv<<std::endl;
        std::cout<<"stride_s_dv: "<<stride_s_dv<<std::endl;
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dk_or_dv_reduce_thd<CK_TILE_TYPE>, grid, block_dv, 0, stream,
          b, h, hg, d_v,
          static_cast<const int32_t*>(cu_seqlen_kv_ptr),
          static_cast<const int32_t*>(cu_seqlen_kv_padded_ptr),
          static_cast<CK_TILE_TYPE*>(dv_expanded_ptr),
          stride_h_dv_expanded, stride_s_dv_expanded,
          static_cast<CK_TILE_TYPE*>(dv_ptr),
          stride_h_dv, stride_s_dv););
    }
  }
  return hipSuccess;
}

}//namespace ck_fused_attn

