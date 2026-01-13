/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>
#include <dlfcn.h>
#include <filesystem>
#include <mutex> //once_flag
#include "ck_fused_attn_utils.hpp"
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "mask.hpp"
#include "bias.hpp"


namespace ck_fused_attn{

void set_aiter_asm_dir() {
  static std::once_flag aiter_asm_dir_once;
  std::call_once(aiter_asm_dir_once, []() {
    hipDeviceProp_t prop;
    hipError_t res= hipGetDeviceProperties(&prop, 0);
    if (res != hipSuccess) {
      throw std::runtime_error(std::string(
        "hipGetDeviceProperties failed with error: ") + hipGetErrorString(res));
    }
    const char *arh_str = nullptr;
    switch (prop.major*10 + prop.minor) {
      case 94: // Gfx942
        arh_str = "gfx942/"; // trailing slash is mandatory
        break;
      case 95: // Gfx950
        arh_str = "gfx950/"; // trailing slash is mandatory
        break;
      default:
        // Unsupported V3 architecture
        return;
    }
    Dl_info info;
    dladdr((void*)set_aiter_asm_dir, &info);
    setenv("AITER_ASM_DIR",
           (std::filesystem::path(info.dli_fname).parent_path() / "aiter" / arh_str).c_str(), 1);
    if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
      if (std::string(env_p) == "1"){
        // Print the set environment variable for debugging purposes
        std::cout << "AITER_ASM_DIR set to: " << getenv("AITER_ASM_DIR") << std::endl;
      }
    }
  });
}

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
  return runtime_max_seqlen;
}

}//namespace ck_fused_attn
