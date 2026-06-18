/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <hip/hip_runtime.h>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "ck_tile/host/pinned_host_releaser.hpp"
#include "qola_mha_bwd.h"
#include "ck_fused_attn_utils.hpp"

// Staged gfx1250 backward dispatch. When this build includes the CK-free V3
// backward library (te_v3_libmha_bwd.so, built for gfx1250), declare its
// namespaced entry point so ck_attn_bwd can route to it on gfx1250 devices at
// runtime. The CK-full path (QOLA_NS(mha_bwd) == qola::te::mha_bwd) is used on
// all other archs.
#if defined(NVTE_AITER_V3_BWD_GFX1250)
namespace qola { namespace te_v3 {
float mha_bwd(const aiter::mha_bwd_args& args, const ck_tile::stream_config& stream_config);
}}  // namespace qola::te_v3
#endif

namespace ck_fused_attn{

#if defined(NVTE_AITER_V3_BWD_GFX1250)
namespace {
// True when the active device is gfx1250 (gcnArchName may carry feature
// suffixes, e.g. "gfx1250:sramecc+", so match on prefix).
bool is_gfx1250_device(){
  int dev = 0;
  if(hipGetDevice(&dev) != hipSuccess){ return false; }
  hipDeviceProp_t prop{};
  if(hipGetDeviceProperties(&prop, dev) != hipSuccess){ return false; }
  return std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
}
}  // namespace
#endif

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
void log_bwd_config(const char* func_name, const aiter::mha_bwd_args& fmha_args, std::ostream* log_file){

  (*log_file) << "\n" << func_name << "\n";

  // fmha_traits debug
  (*log_file) << "\nfmha_traits: \n";
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "data_type", fmha_args.data_type);
  log_value(log_file, "is_group_mode", fmha_args.is_group_mode);
  log_value(log_file, "has_dbias", fmha_args.has_dbias);
  log_value(log_file, "has_dropout", fmha_args.has_dropout);
  log_value(log_file, "is_store_randval", fmha_args.is_store_randval);
  log_value(log_file, "is_deterministic", fmha_args.is_deterministic);
  log_value(log_file, "use_asm_v3", fmha_args.use_asm_v3);
  log_value(log_file, "v3_atomic_fp32", fmha_args.v3_atomic_fp32);
  log_value(log_file, "v3_bf16_cvt", fmha_args.v3_bf16_cvt);

  // fmha_args debug
  (*log_file) << "\nfmha_args: \n";
  log_value(log_file, "q_ptr", fmha_args.q_ptr);
  log_value(log_file, "k_ptr", fmha_args.k_ptr);
  log_value(log_file, "v_ptr", fmha_args.v_ptr);
  log_value(log_file, "bias_ptr", fmha_args.bias_ptr);
  log_value(log_file, "o_ptr", fmha_args.o_ptr);
  log_value(log_file, "lse_ptr", fmha_args.lse_ptr);
  log_value(log_file, "do_ptr", fmha_args.do_ptr);
  log_value(log_file, "d_ptr", fmha_args.d_ptr);
  log_value(log_file, "rand_val_ptr", fmha_args.rand_val_ptr);
  log_value(log_file, "dq_ptr", fmha_args.dq_ptr);
  log_value(log_file, "dk_ptr", fmha_args.dk_ptr);
  log_value(log_file, "dv_ptr", fmha_args.dv_ptr);
  log_value(log_file, "dbias_ptr", fmha_args.dbias_ptr);

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
  log_value(log_file, "max_seqlen_k", fmha_args.max_seqlen_k);
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "nhead_q", fmha_args.nhead_q);
  log_value(log_file, "nhead_k", fmha_args.nhead_k);
  log_value(log_file, "scale", fmha_args.scale);
  log_value(log_file, "stride_q", fmha_args.stride_q);
  log_value(log_file, "stride_k", fmha_args.stride_k);
  log_value(log_file, "stride_v", fmha_args.stride_v);
  log_value(log_file, "stride_bias", fmha_args.stride_bias);
  log_value(log_file, "stride_o", fmha_args.stride_o);
  log_value(log_file, "stride_randval", fmha_args.stride_randval);
  log_value(log_file, "stride_do", fmha_args.stride_do);
  log_value(log_file, "stride_dq", fmha_args.stride_dq);
  log_value(log_file, "stride_dk", fmha_args.stride_dk);
  log_value(log_file, "stride_dv", fmha_args.stride_dv);
  log_value(log_file, "stride_dbias", fmha_args.stride_dbias);
  log_value(log_file, "nhead_stride_q", fmha_args.nhead_stride_q);
  log_value(log_file, "nhead_stride_k", fmha_args.nhead_stride_k);
  log_value(log_file, "nhead_stride_v", fmha_args.nhead_stride_v);
  log_value(log_file, "nhead_stride_bias", fmha_args.nhead_stride_bias);
  log_value(log_file, "nhead_stride_o", fmha_args.nhead_stride_o);
  log_value(log_file, "nhead_stride_randval", fmha_args.nhead_stride_randval);
  log_value(log_file, "nhead_stride_do", fmha_args.nhead_stride_do);
  log_value(log_file, "nhead_stride_lsed", fmha_args.nhead_stride_lsed);
  log_value(log_file, "nhead_stride_dq", fmha_args.nhead_stride_dq);
  log_value(log_file, "nhead_stride_dk", fmha_args.nhead_stride_dk);
  log_value(log_file, "nhead_stride_dv", fmha_args.nhead_stride_dv);
  log_value(log_file, "nhead_stride_dbias", fmha_args.nhead_stride_dbias);
  log_value(log_file, "batch_stride_q", fmha_args.batch_stride_q);
  log_value(log_file, "batch_stride_k", fmha_args.batch_stride_k);
  log_value(log_file, "batch_stride_v", fmha_args.batch_stride_v);
  log_value(log_file, "batch_stride_bias", fmha_args.batch_stride_bias);
  log_value(log_file, "batch_stride_o", fmha_args.batch_stride_o);
  log_value(log_file, "batch_stride_randval", fmha_args.batch_stride_randval);
  log_value(log_file, "batch_stride_do", fmha_args.batch_stride_do);
  log_value(log_file, "batch_stride_lsed", fmha_args.batch_stride_lsed);
  log_value(log_file, "batch_stride_dq", fmha_args.batch_stride_dq);
  log_value(log_file, "batch_stride_dk", fmha_args.batch_stride_dk);
  log_value(log_file, "batch_stride_dv", fmha_args.batch_stride_dv);
  log_value(log_file, "batch_stride_dbias", fmha_args.batch_stride_dbias);
  log_value(log_file, "window_size_left", fmha_args.window_size_left);
  log_value(log_file, "window_size_right", fmha_args.window_size_right);
  log_value(log_file, "mask_type", fmha_args.mask_type);
  log_value(log_file, "bias_type", fmha_args.bias_type);
  log_value(log_file, "p_drop", fmha_args.p_drop);
  log_value(log_file, "p_undrop", fmha_args.p_undrop);
  log_value(log_file, "dropout_seed_ptr",
    std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))
  );
  log_value(log_file, "dropout_offset_ptr",
     std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))
  );
}

void dump_bwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-bwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
}

namespace {

#if ENABLE_CK
// Trait subset that determines AITER's internal bwd workspace footprint. Mirrors
// the fields ck_attn_bwd sets on mha_bwd_args so the size query and the dispatch
// stay in lockstep. fmha_bwd_traits lives in the CK example headers, absent in the
// CK-free build (gfx1250 v3-only tier), which has no v2 launcher to size.
::fmha_bwd_traits make_bwd_traits(const CkAttnBwdArgs& args){
  bool has_dropout = (args.dropout_probability > 0.f);
  bool has_dbias = args.dbias_ptr != nullptr;
  bias_enum bias_type = bias_enum::no_bias;
  if(!args.is_group_mode()){
    bias_type = get_ck_bias_type_shape(&args).first;
  }
  return ::fmha_bwd_traits{
    /* seqlen_q         */ static_cast<int>(args.is_group_mode() ? args.max_tokens_q : args.s_q),
    /* seqlen_k         */ static_cast<int>(args.is_group_mode() ? args.max_tokens_kv : args.s_kv),
    /* batch            */ static_cast<int>(args.b),
    /* max_seqlen_q     */ static_cast<int>(args.s_q),
    /* max_seqlen_k     */ static_cast<int>(args.s_kv),
    /* hdim_q           */ static_cast<int>(args.d_qk),
    /* hdim_v           */ static_cast<int>(args.d_v),
    /* nhead_q          */ static_cast<int>(args.h),
    /* nhead_k          */ static_cast<int>(args.hg),
    /* data_type        */ get_data_type_str(args.dtype),
    /* is_group_mode    */ args.is_group_mode(),
    /* mask_type        */ static_cast<mask_enum>(args.attn_mask_type),
    /* bias_type        */ bias_type,
    /* has_dbias        */ (!args.is_group_mode()) && has_dbias,
    /* has_dropout      */ has_dropout,
    /* is_store_randval */ false,
    /* is_deterministic */ args.deterministic,
  };
}
#endif

// dq_acc bytes the v3 asm path allocates via workspace_alloc. Mirrors aiter's
// fmha_v3_bwd sizing (csrc/cpp_itfs/mha_bwd.cu); returns 0 when v3 can't run so
// the CK launcher size dominates. Gating mirrors ck_attn_bwd's use_asm_v3.
size_t v3_dq_acc_bytes(const CkAttnBwdArgs& args){
  const bool use_asm_v3 = (args.s_q < 16) ? false : args.uses_bwd_v3;
  if(!use_asm_v3){
    return 0;
  }
  const size_t seqlen_q = args.is_group_mode() ? args.max_tokens_q : args.s_q;
  const size_t elem = args.is_v3_atomic_fp32 ? 4 : 2;
  const size_t a16_seq = (args.s_q + 15) / 16 * 16;
  const size_t a16_hdim = (args.d_qk == 192) ? 192 : 128;
  const size_t dq_acc_seq = args.is_v3_atomic_fp32 ? seqlen_q : a16_seq;
  const size_t dq_acc_hdim = args.is_v3_atomic_fp32 ? args.d_qk : a16_hdim;
  const size_t eff_batch = (args.is_group_mode() && args.is_v3_atomic_fp32) ? 1 : args.b;
  return eff_batch * args.h * dq_acc_seq * dq_acc_hdim * elem;
}

}  // namespace

size_t ck_attn_bwd_workspace_size(const CkAttnBwdArgs& args){
#if ENABLE_CK
  // v2 (CK launcher) reports its full device workspace (host metadata + dq_acc)
  // host-side; v3 (asm) allocates only dq_acc. v3 is tried first but may fall
  // back to v2, so reserve the larger of the two. The launcher symbol is forced
  // local by QoLA's export script, so the v2 size is queried through QoLA.
  const size_t v2_bytes = QOLA_NS(mha_bwd_workspace_size)(make_bwd_traits(args));
  const size_t v3_bytes = v3_dq_acc_bytes(args);
  return v2_bytes > v3_bytes ? v2_bytes : v3_bytes;
#else
  // CK-free build (gfx1250 v3-only tier): there is no v2 launcher to query, so
  // only the v3 asm dq_acc accumulator is reserved.
  return v3_dq_acc_bytes(args);
#endif
}

hipError_t ck_attn_bwd(const CkAttnBwdArgs& args, hipStream_t stream){

  bool has_dropout = (args.dropout_probability > 0.f);
  bool has_dbias = args.dbias_ptr != nullptr;
  bool is_mqa_gqa = (args.h > args.hg);

  auto* log_file = get_ck_log_stream();
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, log_file != nullptr};

  bias_enum bias_type = bias_enum::no_bias;
  BiasShape bias_shape = BiasShape::k11SS;
  if (!args.is_group_mode()) {
    std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(&args);
  }

  aiter::mha_bwd_args fmha_args{};
  fmha_args.sink_ptr = nullptr;
  fmha_args.d_sink_ptr = nullptr;
  fmha_args.mask_type = static_cast<int>(static_cast<mask_enum>(args.attn_mask_type));
  // Mirrors AITER's small-seqlen guard at aiter/ops/mha.py:1689.
  fmha_args.use_asm_v3 = (args.s_q < 16) ? false : args.uses_bwd_v3;
  fmha_args.v3_atomic_fp32 = args.is_v3_atomic_fp32;
  fmha_args.v3_bf16_cvt = args.how_v3_bf16_cvt;
  fmha_args.v3_api_check = false;

  fmha_args.hdim_q = args.d_qk;
  fmha_args.hdim_v = args.d_v;
  fmha_args.data_type = get_data_type_str(args.dtype);
  fmha_args.is_group_mode = args.is_group_mode();
  fmha_args.bias_type = static_cast<int>(bias_type);
  fmha_args.has_dbias = (!args.is_group_mode()) && has_dbias;
  fmha_args.has_dropout = has_dropout;
  fmha_args.is_store_randval = false;
  fmha_args.is_deterministic = args.deterministic;

  fmha_args.q_ptr = args.q_ptr;
  fmha_args.k_ptr = args.k_ptr;
  fmha_args.v_ptr = args.v_ptr;
  fmha_args.bias_ptr = (bias_type==bias_enum::no_bias || args.is_group_mode()) ? nullptr
                         : (bias_type==bias_enum::alibi? args.alibi_slope_ptr : args.bias_ptr);
  fmha_args.o_ptr = args.o_ptr;
  fmha_args.lse_ptr = args.lse_ptr;
  fmha_args.do_ptr = args.do_ptr;
  fmha_args.d_ptr = args.lse_workspace_ptr;
  fmha_args.rand_val_ptr = nullptr;
  fmha_args.dq_ptr = args.dq_ptr;
  fmha_args.dk_ptr = is_mqa_gqa? args.dk_expanded_ptr : args.dk_ptr;
  fmha_args.dv_ptr = is_mqa_gqa? args.dv_expanded_ptr : args.dv_ptr;
  fmha_args.dbias_ptr = ((!args.is_group_mode()) && has_dbias)
                          ? (bias_shape==BiasShape::kBHSS ? args.dbias_ptr : args.dbias_expanded_ptr)
                          : nullptr;

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

  // Group mode contract (matches aiter asm_mha_varlen_bwd.cu): seqlen_q/k
  // carry the total token counts, max_seqlen_q/k the per-sequence maximum.
  // aiter sizes dq_acc and related workspaces from seqlen_q; passing the
  // per-sequence length in group mode under-sizes them and the kernel writes
  // past the end.
  fmha_args.seqlen_q = args.is_group_mode() ? args.max_tokens_q : args.s_q;
  fmha_args.seqlen_k = args.is_group_mode() ? args.max_tokens_kv : args.s_kv;
  fmha_args.batch = args.b;
  fmha_args.max_seqlen_q = args.s_q;
  fmha_args.max_seqlen_k = args.s_kv;
  fmha_args.nhead_q = args.h;
  fmha_args.nhead_k = args.hg;
  fmha_args.scale = args.scaling_factor;

  // setup stride_* arguments
  fmha_args.stride_q = args.stride_s_q;
  fmha_args.stride_k = args.stride_s_k;
  fmha_args.stride_v = args.stride_s_v;
  // bias of shape (bias_b, bias_h, s_q, s_kv)
  fmha_args.stride_bias = (!args.is_group_mode() && bias_type!=bias_enum::alibi) ? args.s_kv : 0;
  fmha_args.stride_o = args.stride_s_o;
  fmha_args.stride_randval = args.s_kv;
  fmha_args.stride_do = args.stride_s_do;
  fmha_args.stride_dq = args.stride_s_dq;
  fmha_args.stride_dk = is_mqa_gqa? args.stride_s_dk_expanded : args.stride_s_dk;
  fmha_args.stride_dv = is_mqa_gqa? args.stride_s_dv_expanded : args.stride_s_dv;
  // dbias is of the same shape as bias
  // but ck only take dbias with BHSS
  fmha_args.stride_dbias = (!args.is_group_mode() && bias_type!=bias_enum::alibi) ? args.s_kv : 0;

  // setup nhead_stride_* arguments
  fmha_args.nhead_stride_q = args.stride_h_q;
  fmha_args.nhead_stride_k = args.stride_h_k;
  fmha_args.nhead_stride_v = args.stride_h_v;
  // bias input can be of different shapes (11SS, 1HSS, B1SS, and BHSS), but dbias must be of BHSS
  fmha_args.nhead_stride_bias = get_nhead_stride_bias(bias_shape, args.s_q, args.s_kv, args.is_group_mode());
  fmha_args.nhead_stride_o = args.stride_h_o;
  fmha_args.nhead_stride_randval = args.is_group_mode() ? 0 : args.s_q * args.s_kv;
  fmha_args.nhead_stride_do = args.stride_h_do;
  fmha_args.nhead_stride_lsed = args.is_group_mode() ? args.max_tokens_q : args.s_q;
  fmha_args.nhead_stride_dq = args.stride_h_dq;
  fmha_args.nhead_stride_dk = is_mqa_gqa? args.stride_h_dk_expanded : args.stride_h_dk;
  fmha_args.nhead_stride_dv = is_mqa_gqa? args.stride_h_dv_expanded : args.stride_h_dv;
  // dbias can only be of BHSS
  fmha_args.nhead_stride_dbias = args.is_group_mode()? 0 : args.s_q * args.s_kv;

  // setup batch_stride_* arguments
  fmha_args.batch_stride_q = args.is_group_mode() ? 0 : args.stride_b_q;
  fmha_args.batch_stride_k = args.is_group_mode() ? 0 : args.stride_b_k;
  fmha_args.batch_stride_v = args.is_group_mode() ? 0 : args.stride_b_v;
  fmha_args.batch_stride_bias = get_batch_stride_bias(args.bias_h, bias_shape, args.s_q, args.s_kv, args.is_group_mode(), false);
  fmha_args.batch_stride_o = args.is_group_mode() ? 0 : args.stride_b_o;
  fmha_args.batch_stride_randval = args.is_group_mode() ? 0 : args.h * args.s_q * args.s_kv;
  fmha_args.batch_stride_do = args.is_group_mode() ? 0 : args.stride_b_do;
  fmha_args.batch_stride_lsed = args.is_group_mode() ? 0 : args.h * args.s_q;
  fmha_args.batch_stride_dq = args.is_group_mode() ? 0 : args.stride_b_dq;
  fmha_args.batch_stride_dk = args.is_group_mode() ? 0 : (is_mqa_gqa? args.stride_b_dk_expanded : args.stride_b_dk);
  fmha_args.batch_stride_dv = args.is_group_mode() ? 0 : (is_mqa_gqa? args.stride_b_dv_expanded : args.stride_b_dv);
  // for dbias, use h since h can be different from bias_h
  fmha_args.batch_stride_dbias = args.is_group_mode() ? 0 : args.h * args.s_q * args.s_kv;

  fmha_args.window_size_left = args.window_size_left;
  fmha_args.window_size_right = args.window_size_right;
  fmha_args.p_drop = args.dropout_probability;
  fmha_args.p_undrop = 1.0 - args.dropout_probability;
  fmha_args.drop_seed_offset = std::pair<const void*, const void*>{args.philox_seed_ptr, args.philox_offset_ptr};

  // modify the max_seqlen_q for better performance in 0-length cases
  // lse_workspace_ptr used as buffer
  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")) {
    if(args.is_group_mode() && std::string(env_p) == "1"){
      if(log_file){
        *log_file << "attn_bwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(args.b, args.cu_seqlen_q_ptr, nullptr, args.lse_workspace_ptr, stream);
      fmha_args.max_seqlen_k = get_runtime_max_seqlen(args.b, args.cu_seqlen_kv_ptr, nullptr, args.lse_workspace_ptr, stream);
    }
  }

  // Device-side workspace for mha_bwd's internal allocations (launcher metadata
  // and the dq_acc accumulator) is reserved ahead of time by the caller (see
  // ck_attn_bwd_workspace_size) and carved here, matching the AOTriton bwd path.
  // workspace_alloc bump-allocates from that buffer instead of allocating per
  // call; only one allocation happens per dispatch, but the bump allocator stays
  // correct if aiter splits the request.
  void* ws_base = args.aiter_workspace_ptr;
  const size_t ws_capacity = args.aiter_workspace_bytes;
  size_t ws_offset = 0;
  fmha_args.workspace_alloc = [ws_base, ws_capacity, &ws_offset, stream](size_t bytes, bool zero_init) -> void* {
    if(bytes == 0){
      return nullptr;
    }
    constexpr size_t kAlign = 256;
    const size_t base = (ws_offset + kAlign - 1) & ~(kAlign - 1);
    if(ws_base == nullptr || base + bytes > ws_capacity){
      throw std::runtime_error("ck_fused_attn bwd: AITER workspace request exceeds reserved AOT buffer.");
    }
    void* ptr = static_cast<int8_t*>(ws_base) + base;
    ws_offset = base + bytes;
    if(zero_init){
      if(hipMemsetAsync(ptr, 0, bytes, stream) != hipSuccess){
        throw std::runtime_error("ck_fused_attn bwd: hipMemsetAsync failed for AITER workspace.");
      }
    }
    return ptr;
  };
  // Group mode needs a pinned host buffer for the async D2H seqstart pipeline.
  // aiter keeps the shared_ptr alive past kernel completion via a stream-tail
  // release; that release (and thus the deleter) fires from a HIP callback thread
  // holding runtime locks, so calling any HIP API from it (including hipHostFree)
  // would deadlock against concurrent main-thread HIP calls. Defer the free to
  // ck_tile::pinned_host_releaser's worker thread, which frees each buffer once
  // it is no longer in flight — small and group-mode-v2 only, but never leaked.
  fmha_args.pinned_host_alloc = [](size_t bytes) -> std::shared_ptr<void> {
    if(bytes == 0){
      return {};
    }
    void* ptr = nullptr;
    if(hipHostMalloc(&ptr, bytes, hipHostMallocDefault) != hipSuccess){
      throw std::runtime_error("ck_fused_attn bwd: hipHostMalloc failed for AITER pinned host buffer.");
    }
    return std::shared_ptr<void>(ptr, [](void* p){
      ck_tile::pinned_host_releaser::instance().enqueue(p);
    });
  };

  // print ck traits and args when needed
  if(log_file){
    log_bwd_config(__FUNCTION__, fmha_args, log_file);
  }

  // Graph-capture safety net. The CK v2 launcher (fmha_bwd / prepare_workspace_async)
  // schedules self-deleting hipLaunchHostFunc nodes that re-run and double-free on
  // every graph replay, so it must never be captured. Only the v3 asm path is
  // graph-replay-safe. Backend selection already steers graph-captured training off
  // these configs, but context-parallel and direct callers bypass that path, so we
  // refuse a v2-bound dispatch under active capture rather than corrupt memory on
  // replay. Conditions mirror AITER's fmha_v3_bwd gate (csrc/cpp_itfs/mha_bwd.cu).
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  if(hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
     capture_status != hipStreamCaptureStatusNone){
    int dev = 0;
    hipDeviceProp_t prop{};
    bool is_v3_arch = false;
    if(hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess){
      std::string arch_name(prop.gcnArchName);
      is_v3_arch = arch_name.find("gfx942") != std::string::npos ||
                   arch_name.find("gfx950") != std::string::npos;
    }
    bool resolves_to_v3 = fmha_args.use_asm_v3 && !fmha_args.is_deterministic &&
                          !fmha_args.has_dbias && fmha_args.bias_type == 0 &&
                          !fmha_args.has_dropout && is_v3_arch;
    if(!resolves_to_v3){
      throw std::runtime_error(
        "ck_fused_attn bwd: this configuration dispatches to the CK v2 launcher, which "
        "is not HIP-graph-replay-safe (self-deleting host nodes in prepare_workspace_async). "
        "Disable determinism/dropout/bias and run on gfx942/gfx950 with NVTE_CK_USES_BWD_V3=1 "
        "to use the v3 asm path, or set NVTE_FUSED_ATTN_CK=0 under CUDA graphs.");
    }
  }

  float average_runtime;
#if defined(NVTE_AITER_V3_BWD_GFX1250)
  if(is_gfx1250_device()){
    average_runtime = qola::te_v3::mha_bwd(fmha_args, stream_config);
  } else
#endif
  {
#if defined(NVTE_AITER_CK_FULL)
    average_runtime = QOLA_NS(mha_bwd)(fmha_args, stream_config);
#else
    throw std::runtime_error(
      "ck_fused_attn bwd: this build has no CK-full AITER backward library "
      "(no CDNA archs built); only the staged gfx1250 V3 path is present.");
#endif
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn bwd pass.");
  }
  if(dump_path){
    dump_bwd_timings(dump_path, average_runtime);
  }

  // Post-dispatch reductions for MQA/GQA: reduce dk_expanded/dv_expanded into dk/dv.
  // Batch and group modes use different kernels (batch carves by stride_b; group carves by cu_seqlen).
  if(is_mqa_gqa){
    if(args.is_group_mode()){
      dim3 grid(args.max_tokens_kv, args.hg);
      if(args.d_qk == args.d_v){
        dim3 block(args.d_qk);
        if (log_file) {
          *log_file << "\n" << "run dk_dv_reduce_thd: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << args.cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << args.cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dk_expanded_ptr: " << args.dk_expanded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << args.dv_expanded_ptr << "\n";
          *log_file << "stride_h_dkv_expanded: " << args.stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dkv_expanded: " << args.stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << args.dk_ptr << "\n";
          *log_file << "dv_ptr: " << args.dv_ptr << "\n";
          *log_file << "stride_h_dk: " << args.stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << args.stride_s_dk << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_dv_reduce_thd<CK_TILE_TYPE>, grid, block, 0, stream,
            args.b, args.h, args.hg, args.d_qk,
            static_cast<const int32_t*>(args.cu_seqlen_kv_ptr),
            static_cast<const int32_t*>(args.cu_seqlen_kv_padded_ptr),
            static_cast<CK_TILE_TYPE*>(args.dk_expanded_ptr),
            static_cast<CK_TILE_TYPE*>(args.dv_expanded_ptr),
            args.stride_h_dk_expanded, args.stride_s_dk_expanded,
            static_cast<CK_TILE_TYPE*>(args.dk_ptr),
            static_cast<CK_TILE_TYPE*>(args.dv_ptr),
            args.stride_h_dk, args.stride_s_dk););
      } else {
        dim3 block_dk(args.d_qk);
        if (log_file) {
          *log_file << "\n" << "run dk_or_dv_reduce_thd on dk: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << args.cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << args.cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dk_expanded_ptr: " << args.dk_expanded_ptr << "\n";
          *log_file << "stride_h_dk_expanded: " << args.stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dk_expanded: " << args.stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << args.dk_ptr << "\n";
          *log_file << "stride_h_dk: " << args.stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << args.stride_s_dk << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_or_dv_reduce_thd<CK_TILE_TYPE>, grid, block_dk, 0, stream,
            args.b, args.h, args.hg, args.d_qk,
            static_cast<const int32_t*>(args.cu_seqlen_kv_ptr),
            static_cast<const int32_t*>(args.cu_seqlen_kv_padded_ptr),
            static_cast<CK_TILE_TYPE*>(args.dk_expanded_ptr),
            args.stride_h_dk_expanded, args.stride_s_dk_expanded,
            static_cast<CK_TILE_TYPE*>(args.dk_ptr),
            args.stride_h_dk, args.stride_s_dk););

        dim3 block_dv(args.d_v);
        if (log_file) {
          *log_file << "\n" << "run dk_or_dv_reduce_thd on dv: " << "\n";
          *log_file << "cu_seqlen_kv_ptr: " << args.cu_seqlen_kv_ptr << "\n";
          *log_file << "cu_seqlen_kv_padded_ptr: " << args.cu_seqlen_kv_padded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << args.dv_expanded_ptr << "\n";
          *log_file << "stride_h_dv_expanded: " << args.stride_h_dv_expanded << "\n";
          *log_file << "stride_s_dv_expanded: " << args.stride_s_dv_expanded << "\n";
          *log_file << "dv_ptr: " << args.dv_ptr << "\n";
          *log_file << "stride_h_dv: " << args.stride_h_dv << "\n";
          *log_file << "stride_s_dv: " << args.stride_s_dv << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_or_dv_reduce_thd<CK_TILE_TYPE>, grid, block_dv, 0, stream,
            args.b, args.h, args.hg, args.d_v,
            static_cast<const int32_t*>(args.cu_seqlen_kv_ptr),
            static_cast<const int32_t*>(args.cu_seqlen_kv_padded_ptr),
            static_cast<CK_TILE_TYPE*>(args.dv_expanded_ptr),
            args.stride_h_dv_expanded, args.stride_s_dv_expanded,
            static_cast<CK_TILE_TYPE*>(args.dv_ptr),
            args.stride_h_dv, args.stride_s_dv););
      }
    } else {
      dim3 grid(args.b, args.s_kv, args.hg);
      if(args.d_qk == args.d_v){
        dim3 block(args.d_qk);
        if (log_file) {
          *log_file << "\n" << "run dk_dv_reduce: " << "\n";
          *log_file << "dk_expanded_ptr: " << args.dk_expanded_ptr << "\n";
          *log_file << "dv_expanded_ptr: " << args.dv_expanded_ptr << "\n";
          *log_file << "stride_b_dkv_expanded: " << args.stride_b_dk_expanded << "\n";
          *log_file << "stride_h_dkv_expanded: " << args.stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dkv_expanded: " << args.stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << args.dk_ptr << "\n";
          *log_file << "dv_ptr: " << args.dv_ptr << "\n";
          *log_file << "stride_b_dk: " << args.stride_b_dk << "\n";
          *log_file << "stride_h_dk: " << args.stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << args.stride_s_dk << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_dv_reduce<CK_TILE_TYPE>, grid, block, 0, stream,
            args.b, args.h, args.hg, args.s_kv, args.d_qk,
            static_cast<CK_TILE_TYPE*>(args.dk_expanded_ptr),
            static_cast<CK_TILE_TYPE*>(args.dv_expanded_ptr),
            args.stride_b_dk_expanded, args.stride_h_dk_expanded, args.stride_s_dk_expanded,
            static_cast<CK_TILE_TYPE*>(args.dk_ptr),
            static_cast<CK_TILE_TYPE*>(args.dv_ptr),
            args.stride_b_dk, args.stride_h_dk, args.stride_s_dk););
      } else {
        dim3 block_dk(args.d_qk);
        if (log_file) {
          *log_file << "\n" << "run dk_or_dv_reduce on dk: " << "\n";
          *log_file << "dk_expanded_ptr: " << args.dk_expanded_ptr << "\n";
          *log_file << "stride_b_dk_expanded: " << args.stride_b_dk_expanded << "\n";
          *log_file << "stride_h_dk_expanded: " << args.stride_h_dk_expanded << "\n";
          *log_file << "stride_s_dk_expanded: " << args.stride_s_dk_expanded << "\n";
          *log_file << "dk_ptr: " << args.dk_ptr << "\n";
          *log_file << "stride_b_dk: " << args.stride_b_dk << "\n";
          *log_file << "stride_h_dk: " << args.stride_h_dk << "\n";
          *log_file << "stride_s_dk: " << args.stride_s_dk << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_or_dv_reduce<CK_TILE_TYPE>, grid, block_dk, 0, stream,
            args.b, args.h, args.hg, args.s_kv, args.d_qk,
            static_cast<CK_TILE_TYPE*>(args.dk_expanded_ptr),
            args.stride_b_dk_expanded, args.stride_h_dk_expanded, args.stride_s_dk_expanded,
            static_cast<CK_TILE_TYPE*>(args.dk_ptr),
            args.stride_b_dk, args.stride_h_dk, args.stride_s_dk););

        dim3 block_dv(args.d_v);
        if (log_file) {
          *log_file << "\n" << "run dk_or_dv_reduce on dv: " << "\n";
          *log_file << "dv_expanded_ptr: " << args.dv_expanded_ptr << "\n";
          *log_file << "stride_b_dv_expanded: " << args.stride_b_dv_expanded << "\n";
          *log_file << "stride_h_dv_expanded: " << args.stride_h_dv_expanded << "\n";
          *log_file << "stride_s_dv_expanded: " << args.stride_s_dv_expanded << "\n";
          *log_file << "dv_ptr: " << args.dv_ptr << "\n";
          *log_file << "stride_b_dv: " << args.stride_b_dv << "\n";
          *log_file << "stride_h_dv: " << args.stride_h_dv << "\n";
          *log_file << "stride_s_dv: " << args.stride_s_dv << "\n";
        }
        CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
          hipLaunchKernelGGL(
            dk_or_dv_reduce<CK_TILE_TYPE>, grid, block_dv, 0, stream,
            args.b, args.h, args.hg, args.s_kv, args.d_v,
            static_cast<CK_TILE_TYPE*>(args.dv_expanded_ptr),
            args.stride_b_dv_expanded, args.stride_h_dv_expanded, args.stride_s_dv_expanded,
            static_cast<CK_TILE_TYPE*>(args.dv_ptr),
            args.stride_b_dv, args.stride_h_dv, args.stride_s_dv););
      }
    }
  }

  // dbias reduction (batch mode only) when bias shape isn't already BHSS
  if(!args.is_group_mode() && has_dbias && bias_shape!=BiasShape::kBHSS){
    assert(args.dbias_ptr != args.dbias_expanded_ptr);
    constexpr int THREADS_PER_BLOCK = 1024;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(1.0 * args.s_q * args.s_kv / THREADS_PER_BLOCK));
    if(bias_shape==BiasShape::k11SS){
      if (log_file) {
        *log_file << "\n" << "run dbias_reduce_11SS: " << "\n";
        *log_file << "dbias_ptr: " << args.dbias_ptr << "\n";
        *log_file << "dbias_expanded_ptr: " << args.dbias_expanded_ptr << "\n";
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_11ss<CK_TILE_TYPE>, grid, block, 0, stream,
          args.b, args.h, args.s_q, args.s_kv,
          static_cast<CK_TILE_TYPE*>(args.dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(args.dbias_ptr)););
    }else if(bias_shape==BiasShape::k1HSS){
      if (log_file) {
        *log_file << "\n" << "run dbias_reduce_1HSS: " << "\n";
        *log_file << "dbias_ptr: " << args.dbias_ptr << "\n";
        *log_file << "dbias_expanded_ptr: " << args.dbias_expanded_ptr << "\n";
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_1hss<CK_TILE_TYPE>, grid, block, 0, stream,
          args.b, args.h, args.s_q, args.s_kv,
          static_cast<CK_TILE_TYPE*>(args.dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(args.dbias_ptr)););
    }else if(bias_shape==BiasShape::kB1SS){
      if (log_file) {
        *log_file << "\n" << "run dbias_reduce_B1SS: " << "\n";
        *log_file << "dbias_ptr: " << args.dbias_ptr << "\n";
        *log_file << "dbias_expanded_ptr: " << args.dbias_expanded_ptr << "\n";
      }
      CK_FUSED_ATTN_TYPE_SWITCH_16BIT(args.dtype, CK_TILE_TYPE,
        hipLaunchKernelGGL(
          dbias_reduce_b1ss<CK_TILE_TYPE>, grid, block, 0, stream,
          args.b, args.h, args.s_q, args.s_kv,
          static_cast<CK_TILE_TYPE*>(args.dbias_expanded_ptr),
          static_cast<CK_TILE_TYPE*>(args.dbias_ptr)););
    }
  }
  return hipSuccess;
}


}//namespace ck_fused_attn

