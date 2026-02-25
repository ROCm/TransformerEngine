/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
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

__global__ void get_runtime_max_seqlen_kernel(
  uint64_t b,
  const int32_t* cu_seqlen_ptr, 
  const int32_t* cu_seqlen_padded_ptr, 
  uint64_t *out) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid >= b){
    return;
  }
  if(cu_seqlen_padded_ptr){
    atomicMax(out, cu_seqlen_padded_ptr[tid+1] - cu_seqlen_padded_ptr[tid]);
  }else{
    atomicMax(out, cu_seqlen_ptr[tid+1] - cu_seqlen_ptr[tid]);
  }
}

uint64_t get_runtime_max_seqlen(uint64_t b, const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr, void* workspace, hipStream_t stream){
  uint64_t* runtime_max_seqlen_ptr = static_cast<uint64_t*>(workspace);
  uint64_t runtime_max_seqlen;
  //reset the result buffer to 0
  hipMemsetAsync(runtime_max_seqlen_ptr, 0, sizeof(uint64_t), stream);
  constexpr int threads = 128;
  // in case b ==0
  const int blocks = (static_cast<int64_t>(b) - 1) / threads + 1; // ceil
  get_runtime_max_seqlen_kernel<<<blocks, threads, 0, stream>>>(
    b, 
    static_cast<const int32_t*>(cu_seqlen_ptr),
    static_cast<const int32_t*>(cu_seqlen_padded_ptr),
    runtime_max_seqlen_ptr);
  hipMemcpyAsync(&runtime_max_seqlen, runtime_max_seqlen_ptr, sizeof(uint64_t), hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);

  const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG");
  if (env_p && std::string(env_p) == "1" && cu_seqlen_ptr != nullptr && b > 0) {
    std::vector<int32_t> host_cu(static_cast<size_t>(b) + 1);
    hipMemcpy(host_cu.data(), cu_seqlen_ptr, (static_cast<size_t>(b) + 1) * sizeof(int32_t), hipMemcpyDeviceToHost);
    uint64_t host_max = 0;
    for (uint64_t i = 0; i < b; i++) {
      int32_t len = host_cu[i + 1] - host_cu[i];
      uint64_t u = static_cast<uint64_t>(len);
      if (len < 0) {
        std::cout << "[get_runtime_max_seqlen] b=" << b << " NEGATIVE len at i=" << i
                  << " cu[" << i << "]=" << host_cu[i] << " cu[" << (i+1) << "]=" << host_cu[i+1]
                  << " (kernel would produce garbage uint64)" << std::endl;
      }
      if (u > host_max) host_max = u;
    }
    const size_t n = static_cast<size_t>(b) + 1;
    std::cout << "[get_runtime_max_seqlen] b=" << b << " shape=(" << n << ",) cu_seqlen[0..4]=";
    for (size_t i = 0; i < std::min(n, size_t(5)); i++) std::cout << host_cu[i] << " ";
    std::cout << " ... cu_seqlen[" << (n-5) << ".." << (n-1) << "]=";
    for (size_t i = n - std::min(n, size_t(5)); i < n; i++) std::cout << host_cu[i] << " ";
    std::cout << " host_max_seqlen=" << host_max << " device_returned=" << runtime_max_seqlen << std::endl;
  }

  return runtime_max_seqlen;
}

}//namespace ck_fused_attn
