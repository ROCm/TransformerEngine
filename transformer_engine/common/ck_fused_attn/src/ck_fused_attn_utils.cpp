/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>
#include "ck_fused_attn_utils.hpp"
#include "aiter_fused_attn/aiter_fused_attn.hpp"
#include "mask.hpp"
#include "bias.hpp"

namespace aiter_fused_attn{

std::string get_data_type_str(DType dtype){
  std::string data_type_str;
  if(dtype==DType::kFloat16){
    data_type_str = "fp16";
  }else if(dtype==DType::kBFloat16){
    data_type_str = "bf16";
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid dtype in aiter_fused_attn.");
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
    throw std::runtime_error("Invalid bias_shape in aiter_fused_attn.");
  }
  return BiasShape::kNumBiasShapes;
}

//get ck_tile bias_type and AITER_FUSED_ATTN bias_shape
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
    throw std::runtime_error("Invalid bias_type in aiter_fused_attn.");
  }
  return std::make_pair(bias_type, bias_shape); 
}

}//namespace aiter_fused_attn
