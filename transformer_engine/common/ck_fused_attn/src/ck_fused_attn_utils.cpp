/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>
#include <hip/hip_runtime_api.h>
#include "ck_fused_attn_utils.hpp"
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "mask.hpp"
#include "bias.hpp"

namespace ck_fused_attn{

std::string get_data_type_str(DType dtype){
  std::string data_type_str;
  if(dtype==DType::kFloat16){
    data_type_str = "fp16";
  }else if(dtype==DType::kBFloat16){
    data_type_str = "bf16";
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid dtype in ck_fused_attn.");
  }
  return data_type_str;
}

BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h){
  //identify BHSS with high priority to include scenaiors when b=1 and h=1
  //reduce the chance of dbias_expand_ptr usage
  if(bias_b==b && bias_h==h){
    // treat as 1 if b or h is 1
    return BiasShape::kBHSS;
  }else if(bias_b==1 && bias_h==h){
    return BiasShape::k1HSS;
  }else if(bias_b==b && bias_h==1){
    return BiasShape::kB1SS;
  }else if(bias_b==1 && bias_h==1){
    return BiasShape::k11SS;
  }else{
    //should not happen
    throw std::runtime_error("Invalid bias_shape in ck_fused_attn.");
  }
  return BiasShape::kNumBiasShapes;
}

//get ck_tile bias_type and CK_FUSED_ATTN bias_shape
std::pair<bias_enum, BiasShape> get_ck_bias_type_shape(BiasType attn_bias_type, uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h){
  bias_enum bias_type;
  BiasShape bias_shape; 
  if (attn_bias_type==BiasType::no_bias){
    bias_type = bias_enum::no_bias;
  }else if (attn_bias_type==BiasType::elementwise_bias){
    bias_type = bias_enum::elementwise_bias;
    bias_shape = get_bias_shape(b, h, bias_b, bias_h);
  }else if (attn_bias_type==BiasType::alibi){
    bias_type = bias_enum::alibi;
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid bias_type in ck_fused_attn.");
  }
  return std::make_pair(bias_type, bias_shape); 
}

//CK_FUSED_ATTN MaskType to ck_tile mask enum
mask_enum get_ck_mask_type(MaskType attn_mask_type){
  mask_enum mask_type;
  if (attn_mask_type == MaskType::no_mask){
    mask_type = mask_enum::no_mask;
  }else if(attn_mask_type == MaskType::mask_top_left){
    mask_type = mask_enum::mask_top_left;
  }else if(attn_mask_type == MaskType::mask_bottom_right){
    mask_type = mask_enum::mask_bottom_right;
  }else{
    mask_type = mask_enum::window_generic;
  }

  return mask_type;
}

// no device std::upper_bound
// in an increasing array with given size len, search for the index that:
// array[index] <= target < array[target+1]
__forceinline__ __device__ int binary_search(int32_t target, const int32_t *array, int len) {
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

constexpr int THREADS_PER_WARP = 32;

// cuda kernel to convert softmax lse from [h, b*s_q] (with effective data in first total_q places) to [b, h, s_q]
// one warp in charge of one token index (h*sizeof(float) bytes)
// one lane (thread) in one warp is charge of one element in 32 segment trunk of h
__global__ void softmax_lse_from_thd_kernel(
  uint64_t b, uint64_t h, uint64_t s_q,
  const int32_t* cu_seqlen_q_ptr,
  const float* lse_thd_ptr,
  float* lse_ptr){

  int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
  int lane_idx = threadIdx.x % THREADS_PER_WARP;
  int num_warps = (blockDim.x * gridDim.x) / THREADS_PER_WARP;
  int num_total_tokens = cu_seqlen_q_ptr[b];

  for(int token_id = warp_idx; token_id < num_total_tokens; token_id += num_warps){
    int b_idx = binary_search(token_id, cu_seqlen_q_ptr, b+1);
    int s_idx = token_id - cu_seqlen_q_ptr[b_idx];

    for(int h_idx = lane_idx; h_idx < h; h_idx+=THREADS_PER_WARP){
      int bh_idx = b_idx*h + h_idx;
      lse_ptr[bh_idx * s_q + s_idx] = lse_thd_ptr[h_idx * b*s_q + token_id];
    }
  }
}

// kernel launcher for converting softmax in thd mode to [b, h, s_q] mode
void softmax_lse_from_thd(
  uint64_t b, uint64_t h, uint64_t s_q,
  const void* cu_seqlen_q_ptr,
  const void* lse_thd_ptr,
  void* lse_ptr, 
  hipStream_t stream){
  
  constexpr int THREADS_PER_BLOCK = 256;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * b * s_q * THREADS_PER_WARP/THREADS_PER_BLOCK));
  hipLaunchKernelGGL(
    softmax_lse_from_thd_kernel, grid, block, 0, stream,
    b, h, s_q, static_cast<const int32_t*>(cu_seqlen_q_ptr),
    static_cast<const float*>(lse_thd_ptr),
    static_cast<float*>(lse_ptr));

}

// convert the softmax lse from [b, h, s_q] into shape [h, b*s_q] (with effective data in first total_q places)
__global__ void softmax_lse_to_thd_kernel(
  uint64_t b, uint64_t h, uint64_t s_q,
  const int32_t* cu_seqlen_q_ptr,
  const float* lse_ptr,
  float* lse_thd_ptr){

  int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
  int lane_idx = threadIdx.x % THREADS_PER_WARP;
  int num_warps = (blockDim.x * gridDim.x) / THREADS_PER_WARP;
  int num_total_tokens = cu_seqlen_q_ptr[b];

  for(int token_id = warp_idx; token_id < num_total_tokens; token_id += num_warps){
    int b_idx = binary_search(token_id, cu_seqlen_q_ptr, b+1);
    int s_idx = token_id - cu_seqlen_q_ptr[b_idx];

    for(int h_idx = lane_idx; h_idx < h; h_idx+=THREADS_PER_WARP){
      int bh_idx = b_idx*h + h_idx;
      lse_thd_ptr[h_idx * b*s_q + token_id] = lse_ptr[bh_idx * s_q + s_idx];
    }
  }
}

// kernel launcher for converting softmax in [b, h, s_q] mode to thd mode ([h, b*sq] with total_seqlen_q values first)
void softmax_lse_to_thd(
  uint64_t b, uint64_t h, uint64_t s_q,
  const void* cu_seqlen_q_ptr,
  const void* lse_ptr,
  void* lse_thd_ptr, 
  hipStream_t stream){

  constexpr int THREADS_PER_BLOCK = 256;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * b * s_q * THREADS_PER_WARP/THREADS_PER_BLOCK));
  hipLaunchKernelGGL(
    softmax_lse_to_thd_kernel, grid, block, 0, stream,
    b, h, s_q, static_cast<const int32_t*>(cu_seqlen_q_ptr),
    static_cast<const float*>(lse_ptr),
    static_cast<float*>(lse_thd_ptr));
}

}//namespace ck_fused_attn
