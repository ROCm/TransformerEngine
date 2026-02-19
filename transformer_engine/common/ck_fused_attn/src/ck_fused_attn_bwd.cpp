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

// We want to cache and reuse the log stream so we use thread_local here.
namespace {
std::ofstream* get_bwd_log_stream() {
  thread_local std::ofstream log_file;
  thread_local bool attempted = false;
  if (!attempted) {
    attempted = true;
    open_ck_fused_attn_log_file(log_file, "ck_fused_attn_bwd");
  }
  if (!log_file.is_open()) {
    return nullptr;
  }
  return &log_file;
}
}  // namespace

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
void log_bwd_config(const char* func_name,
                    const std::string data_type_str,
                    const bool is_group_mode,
                    const mask_enum mask_type,
                    const bias_enum bias_type,
                    const bool has_dbias,
                    const bool has_dropout,
                    const bool is_store_randval,
                    const bool is_deterministic,
                    const bool uses_bwd_v3,
                    const bool is_v3_atomic_fp32,
                    const int how_v3_bf16_cvt,
                    const fmha_bwd_args& fmha_args){
  if (auto* log_file = get_bwd_log_stream()) {
    *log_file << "\n" << func_name << "\n";

    // fmha_traits debug
    *log_file << "\n" << "fmha_traits: " << "\n";
    *log_file << "hdim_q: " << fmha_args.hdim_q << "\n";
    *log_file << "hdim_v: " << fmha_args.hdim_v << "\n";
    *log_file << "data_type: " << data_type_str << "\n";
    *log_file << "is_group_mode: " << is_group_mode << "\n";
    *log_file << "mask_type: " << static_cast<std::underlying_type<mask_enum>::type>(mask_type) << "\n";
    *log_file << "bias_type: " << static_cast<std::underlying_type<bias_enum>::type>(bias_type) << "\n";
    *log_file << "has_dbias: " << has_dbias << "\n";
    *log_file << "has_dropout: " << has_dropout << "\n";
    *log_file << "is_store_randval: " << is_store_randval << "\n";
    *log_file << "is_deterministic: " << is_deterministic << "\n";
    *log_file << "uses_bwd_v3: " << uses_bwd_v3 << "\n";
    *log_file << "is_v3_atomic_fp32: " << is_v3_atomic_fp32 << "\n";
    *log_file << "how_v3_bf16_cvt: " << how_v3_bf16_cvt << "\n";

    // fmha_args debug
    *log_file << "\n" << "fmha_args: " << "\n";
    *log_file << "q_ptr: " << fmha_args.q_ptr << "\n";
    *log_file << "k_ptr: " << fmha_args.k_ptr << "\n";
    *log_file << "v_ptr: " << fmha_args.v_ptr << "\n";
    *log_file << "bias_ptr: " << fmha_args.bias_ptr << "\n";
    *log_file << "o_ptr: " << fmha_args.o_ptr << "\n";
    *log_file << "lse_ptr: " << fmha_args.lse_ptr << "\n";
    *log_file << "do_ptr: " << fmha_args.do_ptr << "\n";
    *log_file << "d_ptr: " << fmha_args.d_ptr << "\n";
    *log_file << "rand_val_ptr: " << fmha_args.rand_val_ptr << "\n";
    *log_file << "dq_ptr: " << fmha_args.dq_ptr << "\n";
    *log_file << "dk_ptr: " << fmha_args.dk_ptr << "\n";
    *log_file << "dv_ptr: " << fmha_args.dv_ptr << "\n";
    *log_file << "dbias_ptr: " << fmha_args.dbias_ptr << "\n";
    *log_file << "dq_acc_ptr: " << fmha_args.dq_acc_ptr << "\n";

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
    *log_file << "max_seqlen_k: " << fmha_args.max_seqlen_k << "\n";
    *log_file << "hdim_q: " << fmha_args.hdim_q << "\n";
    *log_file << "hdim_v: " << fmha_args.hdim_v << "\n";
    *log_file << "nhead_q: " << fmha_args.nhead_q << "\n";
    *log_file << "nhead_k: " << fmha_args.nhead_k << "\n";
    *log_file << "scale: " << fmha_args.scale << "\n";
    *log_file << "stride_q: " << fmha_args.stride_q << "\n";
    *log_file << "stride_k: " << fmha_args.stride_k << "\n";
    *log_file << "stride_v: " << fmha_args.stride_v << "\n";
    *log_file << "stride_bias: " << fmha_args.stride_bias << "\n";
    *log_file << "stride_o: " << fmha_args.stride_o << "\n";
    *log_file << "stride_randval: " << fmha_args.stride_randval << "\n";
    *log_file << "stride_do: " << fmha_args.stride_do << "\n";
    *log_file << "stride_dq_acc: " << fmha_args.stride_dq_acc << "\n";
    *log_file << "stride_dq: " << fmha_args.stride_dq << "\n";
    *log_file << "stride_dk: " << fmha_args.stride_dk << "\n";
    *log_file << "stride_dv: " << fmha_args.stride_dv << "\n";
    *log_file << "stride_dbias: " << fmha_args.stride_dbias << "\n";
    *log_file << "nhead_stride_q: " << fmha_args.nhead_stride_q << "\n";
    *log_file << "nhead_stride_k: " << fmha_args.nhead_stride_k << "\n";
    *log_file << "nhead_stride_v: " << fmha_args.nhead_stride_v << "\n";
    *log_file << "nhead_stride_bias: " << fmha_args.nhead_stride_bias << "\n";
    *log_file << "nhead_stride_o: " << fmha_args.nhead_stride_o << "\n";
    *log_file << "nhead_stride_randval: " << fmha_args.nhead_stride_randval << "\n";
    *log_file << "nhead_stride_do: " << fmha_args.nhead_stride_do << "\n";
    *log_file << "nhead_stride_lsed: " << fmha_args.nhead_stride_lsed << "\n";
    *log_file << "nhead_stride_dq_acc: " << fmha_args.nhead_stride_dq_acc << "\n";
    *log_file << "nhead_stride_dq: " << fmha_args.nhead_stride_dq << "\n";
    *log_file << "nhead_stride_dk: " << fmha_args.nhead_stride_dk << "\n";
    *log_file << "nhead_stride_dv: " << fmha_args.nhead_stride_dv << "\n";
    *log_file << "nhead_stride_dbias: " << fmha_args.nhead_stride_dbias << "\n";
    *log_file << "batch_stride_q: " << fmha_args.batch_stride_q << "\n";
    *log_file << "batch_stride_k: " << fmha_args.batch_stride_k << "\n";
    *log_file << "batch_stride_v: " << fmha_args.batch_stride_v << "\n";
    *log_file << "batch_stride_bias: " << fmha_args.batch_stride_bias << "\n";
    *log_file << "batch_stride_o: " << fmha_args.batch_stride_o << "\n";
    *log_file << "batch_stride_randval: " << fmha_args.batch_stride_randval << "\n";
    *log_file << "batch_stride_do: " << fmha_args.batch_stride_do << "\n";
    *log_file << "batch_stride_lsed: " << fmha_args.batch_stride_lsed << "\n";
    *log_file << "batch_stride_dq_acc: " << fmha_args.batch_stride_dq_acc << "\n";
    *log_file << "batch_stride_dq: " << fmha_args.batch_stride_dq << "\n";
    *log_file << "batch_stride_dk: " << fmha_args.batch_stride_dk << "\n";
    *log_file << "batch_stride_dv: " << fmha_args.batch_stride_dv << "\n";
    *log_file << "batch_stride_dbias: " << fmha_args.batch_stride_dbias << "\n";
    *log_file << "window_size_left: " << fmha_args.window_size_left << "\n";
    *log_file << "window_size_right: " << fmha_args.window_size_right << "\n";
    *log_file << "mask_type: " << fmha_args.mask_type << "\n";
    *log_file << "p_drop: " << fmha_args.p_drop << "\n";
    *log_file << "p_undrop: " << fmha_args.p_undrop << "\n";
    *log_file << "dropout_seed_ptr: " << std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)) << "\n";
    *log_file << "dropout_offset_ptr: " << std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)) << "\n";
  }

}

void dump_bwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-bwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
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
  bool is_group_mode = false;
  bool s_randval = false;

  bias_enum bias_type;
  BiasShape bias_shape; 
  std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(attn_bias_type, b, h, bias_b, bias_h);
  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
 
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);
  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) != "")
      ck_fused_attn_log_config = true;
  }
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");

  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, ck_fused_attn_log_config};

  ck_tile::index_t shape_seqlen_q = seqlen_q;
  ck_tile::index_t shape_seqlen_k = seqlen_k;

  std::string data_type_str = get_data_type_str(dtype);

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // bias of shape (bias_b, bias_h, s_q, s_kv)
    const ck_tile::index_t stride_bias = max_seqlen_k;
    const ck_tile::index_t stride_o = stride_s_o;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_do = stride_s_do;
    const ck_tile::index_t stride_dq = stride_s_dq;
    const ck_tile::index_t stride_dk = stride_s_dk;
    const ck_tile::index_t stride_dv = stride_s_dv;
    const ck_tile::index_t stride_dk_expanded = stride_s_dk_expanded;
    const ck_tile::index_t stride_dv_expanded = stride_s_dv_expanded;
    const ck_tile::index_t stride_dq_acc = d_qk; //dq_acc of shape (nsplits, B, H, S, D)
    // dbias is of the same shape as bias
    // but ck only take dbias with BHSS
    const ck_tile::index_t stride_dbias = max_seqlen_k;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // bias input can be of different shapes (11SS, 1HSS, B1SS, and BHSS), but dbias must be of BHSS
    const ck_tile::index_t nhead_stride_bias = (bias_shape==BiasShape::k1HSS || bias_shape==BiasShape::kBHSS) ? max_seqlen_q * max_seqlen_k: 0;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    const ck_tile::index_t nhead_stride_randval =
        shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t nhead_stride_do = stride_h_do;
    const ck_tile::index_t nhead_stride_lsed = max_seqlen_q;
    const ck_tile::index_t nhead_stride_dq = stride_h_dq;
    const ck_tile::index_t nhead_stride_dk = stride_h_dk;
    const ck_tile::index_t nhead_stride_dv = stride_h_dv;
    const ck_tile::index_t nhead_stride_dk_expanded = stride_h_dk_expanded;
    const ck_tile::index_t nhead_stride_dv_expanded = stride_h_dv_expanded;
    // dbias can only be of BHSS
    const ck_tile::index_t nhead_stride_dbias = max_seqlen_q * max_seqlen_k;
    const ck_tile::index_t nhead_stride_dq_acc = s_q*d_qk; //dq_acc of shape (nsplits, B, H, S, D)
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = stride_b_q;
    const ck_tile::index_t batch_stride_k = stride_b_k;
    const ck_tile::index_t batch_stride_v = stride_b_v;
    // bias input can be of different shapes (11SS, 1HSS, B1SS, and BHSS), but dbias must be of BHSS
    // for B1SS and BHSS, batch stride for bias are both bias_h x s_q x s_kv (bias_h==1 for B1SS and bias_h == h for BHSS)
    const ck_tile::index_t batch_stride_bias = (bias_shape==BiasShape::k11SS || bias_shape==BiasShape::k1HSS) ? 0: bias_h* max_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_o = stride_b_o;
    const ck_tile::index_t batch_stride_randval =
        nhead * shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_do = stride_b_do;
    const ck_tile::index_t batch_stride_lsed = nhead * max_seqlen_q;
    const ck_tile::index_t batch_stride_dq = stride_b_dq;
    const ck_tile::index_t batch_stride_dk = stride_b_dk;
    const ck_tile::index_t batch_stride_dv = stride_b_dv;
    const ck_tile::index_t batch_stride_dk_expanded = stride_b_dk_expanded;
    const ck_tile::index_t batch_stride_dv_expanded = stride_b_dv_expanded;
    // for dbias, use h since h can be different from bias_h
    const ck_tile::index_t batch_stride_dbias = h* max_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_dq_acc = h*s_q*d_qk; //dq_acc of shape (nsplits, B, H, S, D)
    const ck_tile::index_t split_stride_dq_acc = b * h * s_q * d_qk;

    return fmha_bwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         bias_type==bias_enum::no_bias? nullptr : (bias_type==bias_enum::alibi? alibi_slope_ptr :bias_ptr),
                         o_ptr,
                         lse_ptr,
                         do_ptr,
                         lse_workspace_ptr,
                         nullptr,
                         dq_ptr,
                         is_mqa_gqa? dk_expanded_ptr:dk_ptr,
                         is_mqa_gqa? dv_expanded_ptr:dv_ptr,
                         has_dbias? (bias_shape==BiasShape::kBHSS ? dbias_ptr: dbias_expanded_ptr): nullptr,
                         dq_acc_ptr, //dq_acc_buf
                         nullptr,//seqstart_q_ptr
                         nullptr,//seqstart_k_ptr
                         nullptr, /* seqlen_q_ptr */
                         nullptr, /* seqlen_k_ptr */
                         nullptr, //cu_seqlen_q_ptr
                         nullptr, //cu_seqlen_k_ptr
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
                         bias_type==bias_enum::alibi? 0: stride_bias,
                         stride_o,
                         stride_randval,
                         stride_do,
                         stride_dq_acc,//stride_dq_acc
                         stride_dq,//stride_dq
                         is_mqa_gqa? stride_dk_expanded:stride_dk,
                         is_mqa_gqa? stride_dv_expanded:stride_dv,
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
                         is_mqa_gqa? nhead_stride_dk_expanded:nhead_stride_dk,
                         is_mqa_gqa? nhead_stride_dv_expanded:nhead_stride_dv,
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
                         is_mqa_gqa? batch_stride_dk_expanded:batch_stride_dk,
                         is_mqa_gqa? batch_stride_dv_expanded:batch_stride_dv,
                         batch_stride_dbias,
                         split_stride_dq_acc,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         p_undrop,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();

  // print ck traits and args when needed
  log_bwd_config(__FUNCTION__, data_type_str, is_group_mode, mask_type, bias_type, has_dbias, has_dropout, s_randval, deterministic, uses_bwd_v3, is_v3_atomic_fp32, how_v3_bf16_cvt, fmha_args);
  
  float average_runtime = aiter::mha_bwd(fmha_args,
                                         stream_config,
                                         data_type_str,
                                         is_group_mode,
                                         mask_type,
                                         bias_type,
                                         has_dbias,
                                         s_randval,
                                         deterministic,
                                         uses_bwd_v3,
                                         is_v3_atomic_fp32,
                                         how_v3_bf16_cvt);
  if(dump_path){
    dump_bwd_timings(dump_path, average_runtime);
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn bwd pass.");
  }
  if(is_mqa_gqa){
    dim3 grid(b, s_kv, hg);
    if (d_qk == d_v) {
      dim3 block(d_qk);
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_dv_reduce: " << "\n";
          *log_file << "dk_expanded_ptr: " << dk_expanded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << dv_expanded_ptr << "\n";
          *log_file << "stride_b_dkv_expanded: " << stride_b_dk_expanded << "\n";
          *log_file << "stride_h_dkv_expanded: " << stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dkv_expanded: " << stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << dk_ptr << "\n";
          *log_file << "dv_ptr: " << dv_ptr << "\n";
          *log_file << "stride_b_dk: " << stride_b_dk << "\n";
          *log_file << "stride_h_dk: " << stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << stride_s_dk << "\n";
        }
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
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_or_dv_reduce on dk: " << "\n";
          *log_file << "dk_expanded_ptr: " << dk_expanded_ptr << "\n";
          *log_file << "stride_b_dk_expanded: " << stride_b_dk_expanded << "\n";
          *log_file << "stride_h_dk_expanded: " << stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dk_expanded: " << stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << dk_ptr << "\n";
          *log_file << "stride_b_dk: " << stride_b_dk << "\n";
          *log_file << "stride_h_dk: " << stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << stride_s_dk << "\n";
        }
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
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_or_dv_reduce on dv: " << "\n";
          *log_file << "dv_expanded_ptr: " << dv_expanded_ptr << "\n";
          *log_file << "stride_b_dv_expanded: " << stride_b_dv_expanded << "\n";
          *log_file << "stride_h_dv_expanded: " << stride_h_dv_expanded << "\n";
          *log_file << "stride_s_dv_expanded: " << stride_s_dv_expanded << "\n";
          *log_file << "dv_ptr: " << dv_ptr << "\n";
          *log_file << "stride_b_dv: " << stride_b_dv << "\n";
          *log_file << "stride_h_dv: " << stride_h_dv << "\n";
          *log_file << "stride_s_dv: " << stride_s_dv << "\n";
        }
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
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dbias_reduce_11SS: " << "\n";
          *log_file << "dbias_ptr: " << dbias_ptr << "\n";
          *log_file << "dbias_expanded_ptr: " << dbias_expanded_ptr << "\n";
        }
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_11ss<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, s_q, s_kv,
          static_cast<CK_TILE_TYPE*>(dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dbias_ptr));); 
    }else if(bias_shape==BiasShape::k1HSS){
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dbias_reduce_1HSS: " << "\n";
          *log_file << "dbias_ptr: " << dbias_ptr << "\n";
          *log_file << "dbias_expanded_ptr: " << dbias_expanded_ptr << "\n";
        }
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_1hss<CK_TILE_TYPE>, grid, block, 0, stream,
          b, h, s_q, s_kv,
          static_cast<CK_TILE_TYPE*>(dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(dbias_ptr));); 
    }else if(bias_shape==BiasShape::kB1SS){
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dbias_reduce_B1SS: " << "\n";
          *log_file << "dbias_ptr: " << dbias_ptr << "\n";
          *log_file << "dbias_expanded_ptr: " << dbias_expanded_ptr << "\n";
        }
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

  bool has_dropout = (dropout_probability > 0.f);
  bool has_dbias = false;
  bool is_mqa_gqa = (h > hg);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d_qk;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d_v;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float p_drop = dropout_probability;
  float p_undrop = 1.0 - p_drop;
  bool is_group_mode = true;
  bool s_randval = false;

  // THD does not work with bias
  
  ck_tile::index_t left, right;
  left = window_size_left;
  right = window_size_right;
  mask_enum mask_type = static_cast<mask_enum>(attn_mask_type);
 
  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) != "")
      ck_fused_attn_log_config = true;
  } 
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, ck_fused_attn_log_config};

  std::string data_type_str = get_data_type_str(dtype);

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t stride_bias = 0;
    const ck_tile::index_t stride_o = stride_s_o;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_do = stride_s_do;
    const ck_tile::index_t stride_dq = stride_s_dq;
    const ck_tile::index_t stride_dk = stride_s_dk;
    const ck_tile::index_t stride_dv = stride_s_dv;
    const ck_tile::index_t stride_dk_expanded = stride_s_dk_expanded;
    const ck_tile::index_t stride_dv_expanded = stride_s_dv_expanded;
    const ck_tile::index_t stride_dq_acc = d_qk; //dq_acc of shape (nsplits, H, max_tokens_q, D_qk)
    // bias not used in THD qkv layout
    const ck_tile::index_t stride_dbias = 0;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t nhead_stride_bias = 0;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    const ck_tile::index_t nhead_stride_randval = 0;
    const ck_tile::index_t nhead_stride_do = stride_h_do;
    // use packed lse
    const ck_tile::index_t nhead_stride_lsed = max_tokens_q;
    const ck_tile::index_t nhead_stride_dq = stride_h_dq;
    const ck_tile::index_t nhead_stride_dk = stride_h_dk;
    const ck_tile::index_t nhead_stride_dv = stride_h_dv;
    const ck_tile::index_t nhead_stride_dk_expanded = stride_h_dk_expanded;
    const ck_tile::index_t nhead_stride_dv_expanded = stride_h_dv_expanded;
    // bias not used in THD qkv layout
    const ck_tile::index_t nhead_stride_dbias = 0;
    const ck_tile::index_t nhead_stride_dq_acc = max_tokens_q*d_qk; //dq_acc of shape (nsplits, H, max_tokens_q, D_qk)
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = 0;
    const ck_tile::index_t batch_stride_k = 0;
    const ck_tile::index_t batch_stride_v = 0;
    // bias not used in THD qkv layout
    const ck_tile::index_t batch_stride_bias = 0;
    const ck_tile::index_t batch_stride_o = 0;
    const ck_tile::index_t batch_stride_randval = 0;
    const ck_tile::index_t batch_stride_do = 0;
    const ck_tile::index_t batch_stride_lsed = 0;
    const ck_tile::index_t batch_stride_dq = 0;
    const ck_tile::index_t batch_stride_dk = 0;
    const ck_tile::index_t batch_stride_dv = 0;
    const ck_tile::index_t batch_stride_dk_expanded = 0;
    const ck_tile::index_t batch_stride_dv_expanded = 0;
    // bias not used in THD qkv layout
    const ck_tile::index_t batch_stride_dbias = 0;
    const ck_tile::index_t batch_stride_dq_acc = 0; //dq_acc of shape (nsplits, T, H, D)
    const ck_tile::index_t split_stride_dq_acc = max_tokens_q*h*d_qk;

    return fmha_bwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,
                         o_ptr,
                         lse_thd_ptr,
                         do_ptr,
                         lse_workspace_ptr,
                         nullptr,
                         dq_ptr,
                         is_mqa_gqa? dk_expanded_ptr:dk_ptr,
                         is_mqa_gqa? dv_expanded_ptr:dv_ptr,
                         nullptr, //dbias_ptr
                         dq_acc_ptr, //dq_acc_buf
                         cu_seqlen_q_padded_ptr==nullptr? cu_seqlen_q_ptr: cu_seqlen_q_padded_ptr, //seqstart_q_ptr
                         cu_seqlen_kv_padded_ptr==nullptr? cu_seqlen_kv_ptr: cu_seqlen_kv_padded_ptr, //seqstart_k_ptr
                         nullptr, /* seqlen_q_ptr */
                         nullptr, /* seqlen_k_ptr */
                         cu_seqlen_q_ptr, //cu_seqlen_q_ptr
                         cu_seqlen_kv_ptr, //cu_seqlen_k_ptr
                         max_seqlen_q, //seqlen_q, unused in group mode
                         max_seqlen_k, //seqlen_kv, unused in group mode
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
                         is_mqa_gqa? stride_dk_expanded:stride_dk,
                         is_mqa_gqa? stride_dv_expanded:stride_dv,
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
                         is_mqa_gqa? nhead_stride_dk_expanded:nhead_stride_dk,
                         is_mqa_gqa? nhead_stride_dv_expanded:nhead_stride_dv,
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
                         is_mqa_gqa? batch_stride_dk_expanded:batch_stride_dk,
                         is_mqa_gqa? batch_stride_dv_expanded:batch_stride_dv,
                         batch_stride_dbias,
                         split_stride_dq_acc,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         p_undrop,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();

  // modify the max_seqlen_q for better performance in 0-length cases
  // lse_thd_ptr used as buffer
  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")) {
    if(std::string(env_p) == "1"){
      if(ck_fused_attn_log_config){
        std::cout << "attn_bwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(b, cu_seqlen_q_ptr, nullptr, lse_workspace_ptr, stream);
      fmha_args.max_seqlen_k = get_runtime_max_seqlen(b, cu_seqlen_kv_ptr, nullptr, lse_workspace_ptr, stream);
    }
  }

  // print ck traits and args when needed
  log_bwd_config(__FUNCTION__, data_type_str, is_group_mode, mask_type, bias_enum::no_bias, has_dbias, has_dropout, s_randval, deterministic, uses_bwd_v3, is_v3_atomic_fp32, how_v3_bf16_cvt, fmha_args);

  float average_runtime = aiter::mha_bwd(fmha_args,
    stream_config,
    data_type_str,
    is_group_mode,
    mask_type,
    bias_enum::no_bias,
    has_dbias,
    s_randval,
    deterministic,
    uses_bwd_v3,
    is_v3_atomic_fp32,
    how_v3_bf16_cvt);
  if(dump_path){
    dump_bwd_timings(dump_path, average_runtime);
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn bwd pass.");
  }
  if(is_mqa_gqa){
    dim3 grid(max_tokens_kv, hg);
    if (d_qk == d_v) {
      dim3 block(d_qk);
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_dv_reduce_thd: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dk_expanded_ptr: " << dk_expanded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << dv_expanded_ptr << "\n";
          *log_file << "stride_h_dkv_expanded: " << stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dkv_expanded: " << stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << dk_ptr << "\n";
          *log_file << "dv_ptr: " << dv_ptr << "\n";
          *log_file << "stride_h_dk: " << stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << stride_s_dk << "\n";
        }
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
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_or_dv_reduce_thd on dk: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dk_expanded_ptr: " << dk_expanded_ptr << "\n";
          *log_file << "stride_h_dk_expanded: " << stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dk_expanded: " << stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << dk_ptr << "\n";
          *log_file << "stride_h_dk: " << stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << stride_s_dk << "\n";
        }
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
      if (ck_fused_attn_log_config){
        if (auto* log_file = get_bwd_log_stream()) {
          *log_file << "\n" << "run dk_or_dv_reduce_thd on dv: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << dv_expanded_ptr << "\n";
          *log_file << "stride_h_dv_expanded: " << stride_h_dv_expanded << "\n";
          *log_file << "stride_s_dv_expanded: " << stride_s_dv_expanded << "\n";
          *log_file << "dv_ptr: " << dv_ptr << "\n";
          *log_file << "stride_h_dv: " << stride_h_dv << "\n";
          *log_file << "stride_s_dv: " << stride_s_dv << "\n";
        }
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

