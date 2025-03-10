/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include<utility>
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

// kernel to remove padding for q, k, v, o (dq, dk, dv, do)
template<typename DataType>
__global__ void remove_padding_kernel(
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged, // sometimes both cu_seqlen and cu_seqlen_padded given in bshd cases
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const DataType* data_ptr,
  const int32_t* cu_seqlen_ptr, const int32_t* cu_seqlen_padded_ptr,
  DataType* data_without_padding_ptr){

  // TE always has (B, S) first, then (H, D), so stride_total_seqlen will be minimal of stride_b and stride_s 
  uint64_t stride_total_seqlen = std::min(stride_b, stride_s);
  for(uint64_t hd_idx = blockIdx.x*blockDim.x + threadIdx.x; hd_idx < h*d; hd_idx += blockDim.x * gridDim.x){
    uint64_t h_idx = hd_idx/d;
    uint64_t d_idx = hd_idx%d;
    //loop over all batch
    for(uint64_t b_idx = 0; b_idx < b; b_idx++){
      for(uint64_t s_idx = 0; s_idx < cu_seqlen_ptr[b_idx+1] - cu_seqlen_ptr[b_idx]; s_idx++){
        if(is_ragged){
          // thd with padding
          // thd implies stride B > stride S
          *(data_without_padding_ptr + (cu_seqlen_ptr[b_idx] + s_idx)*stride_s + h_idx*stride_h+d_idx) = *(data_ptr + (cu_seqlen_padded_ptr[b_idx]+s_idx)*stride_s + h_idx *stride_h+d_idx);
        }else{
          // bshd or sbhd with padding
          *(data_without_padding_ptr + (cu_seqlen_ptr[b_idx] + s_idx)*stride_total_seqlen + h_idx*stride_h + d_idx) = *(data_ptr + b_idx*stride_b + s_idx*stride_s + h_idx*stride_h + d_idx);
        }
      }
    }
  }
}

// kernel launcher for remove padding in q, k, v, o (dq, dk, dv, do)
void remove_padding(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const void* data_ptr,
  const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr,
  void* data_without_padding_ptr,
  hipStream_t stream){
  
  // cu_seqlen_padded_ptr can be nullptr
  assert(cu_seqlen_ptr!=nullptr);
  constexpr int THREADS_PER_BLOCK = 1024;
  // parallel over h*d dimension
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * h * d/THREADS_PER_BLOCK));

  CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
    hipLaunchKernelGGL(
      remove_padding_kernel<CK_TILE_TYPE>, grid, block, 0, stream,
      b, h, s, d,
      is_ragged,
      stride_b, stride_h, stride_s,
      static_cast<const CK_TILE_TYPE*>(data_ptr),
      static_cast<const int32_t*>(cu_seqlen_ptr),
      static_cast<const int32_t*>(cu_seqlen_padded_ptr),
      static_cast<CK_TILE_TYPE*>(data_without_padding_ptr)););

}

// kernel to add padding for q, k, v, o (dq, dk, dv, do)
template<typename DataType>
__global__ void add_padding_kernel(
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const DataType* data_without_padding_ptr,
  const int32_t* cu_seqlen_ptr, const int32_t* cu_seqlen_padded_ptr,
  DataType* data_ptr){
  
  // TE always has (B, S) first, then (H, D), so stride_total_seqlen will be minimal of stride_b and stride_s 
  uint64_t stride_total_seqlen = std::min(stride_b, stride_s);
  for(uint64_t hd_idx = blockIdx.x*blockDim.x + threadIdx.x; hd_idx < h*d; hd_idx += blockDim.x * gridDim.x){
    uint64_t h_idx = hd_idx/d;
    uint64_t d_idx = hd_idx%d;
    //loop over all batch
    for(uint64_t b_idx = 0; b_idx < b; b_idx++){
      for(uint64_t s_idx = 0; s_idx < cu_seqlen_ptr[b_idx+1] - cu_seqlen_ptr[b_idx]; s_idx++){
        if(is_ragged){
          // thd with padding
          // thd implies stride B > stride S
          *(data_ptr + (cu_seqlen_padded_ptr[b_idx]+s_idx)*stride_s + h_idx *stride_h+d_idx) = *(data_without_padding_ptr + (cu_seqlen_ptr[b_idx] + s_idx)*stride_s + h_idx*stride_h+d_idx);
        }else{
          // bshd/sbhd with padding
          *(data_ptr + b_idx*stride_b + s_idx*stride_s + h_idx*stride_h + d_idx) = *(data_without_padding_ptr + (cu_seqlen_ptr[b_idx] + s_idx)*stride_total_seqlen + h_idx*stride_h+d_idx);
        }
      }
    }
  }
}

// kernel launcher for adding padding in q, k, v, o (dq, dk, dv, do)
void add_padding(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const void* data_without_padding_ptr,
  const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr,
  void* data_ptr,
  hipStream_t stream){
  
  // cu_seqlen_padded_ptr can be nullptr
  assert(cu_seqlen_ptr!=nullptr);
  constexpr int THREADS_PER_BLOCK = 1024;
  // parallel over h*d dimension
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * h * d/THREADS_PER_BLOCK));

  CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, CK_TILE_TYPE,
    hipLaunchKernelGGL(
      add_padding_kernel<CK_TILE_TYPE>, grid, block, 0, stream,
      b, h, s, d,
      is_ragged,
      stride_b, stride_h, stride_s,
      static_cast<const CK_TILE_TYPE*>(data_without_padding_ptr),
      static_cast<const int32_t*>(cu_seqlen_ptr),
      static_cast<const int32_t*>(cu_seqlen_padded_ptr),
      static_cast<CK_TILE_TYPE*>(data_ptr)););

}
}//namespace ck_fused_attn
