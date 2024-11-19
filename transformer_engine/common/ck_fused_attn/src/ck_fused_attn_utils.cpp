/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{
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

}//namespace ck_fused_attn
