/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>
#include <dlfcn.h>
#include <filesystem>
#include <mutex> //once_flag

#include <hip/hip_runtime_api.h>

#include "ck_fused_attn_utils.hpp"
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "mask.hpp"
#include "bias.hpp"


namespace ck_fused_attn{

ck_tile::index_t get_batch_stride_bias(
  ck_tile::index_t bias_h,
  BiasShape bias_shape,
  ck_tile::index_t max_seqlen_q,
  ck_tile::index_t max_seqlen_k,
  bool is_group_mode,
  bool is_fwd
){
  if(is_group_mode){
    return 0;
  }
  switch (bias_shape) {
    case BiasShape::k11SS:
    case BiasShape::k1HSS:
      return 0;
    case BiasShape::kB1SS:
      // dbias must be BHSS
      if(is_fwd){
        return max_seqlen_q * max_seqlen_k;
      }
    case BiasShape::kBHSS:
      return bias_h * max_seqlen_q * max_seqlen_k;
    default:
      throw std::runtime_error("Invalid bias shape");
  }
}
// for B1SS and BHSS, batch stride for bias are both
// bias_h x s_q x s_kv (bias_h==1 for B1SS and bias_h == h for BHSS)
ck_tile::index_t get_nhead_stride_bias(
  BiasShape bias_shape,
  ck_tile::index_t max_seqlen_q,
  ck_tile::index_t max_seqlen_k,
  bool is_group_mode
){
  if(is_group_mode){
    return 0;
  }
  switch (bias_shape) {
    case BiasShape::k1HSS:
    case BiasShape::kBHSS:
      return max_seqlen_q * max_seqlen_k;
    case BiasShape::k11SS:
    case BiasShape::kB1SS:
      return 0;
    default:
      throw std::runtime_error("Invalid bias shape");
  }
}

void set_aiter_asm_dir() {
  static std::once_flag aiter_asm_dir_once;
  std::call_once(aiter_asm_dir_once, []() {
    Dl_info info;
    dladdr((void*)set_aiter_asm_dir, &info);
    auto install_lib_path = std::filesystem::path(info.dli_fname).parent_path() / "aiter";
    const char* log_ck_config = std::getenv("NVTE_LOG_CK_CONFIG");
    auto editable_install_path = std::filesystem::path(info.dli_fname).parent_path().parent_path().parent_path() / "3rdparty" / "aiter" / "hsa";
    for(const auto& path : {install_lib_path, editable_install_path}) {
      if(std::filesystem::exists(path)) {
        setenv("AITER_ASM_DIR", path.c_str(), 1);
        if (log_ck_config && log_ck_config == std::string("1")) {
          std::cout << "AITER_ASM_DIR set to: " << getenv("AITER_ASM_DIR") << std::endl;
        }
        return;
      }
      if(log_ck_config && log_ck_config == std::string("1")) {
        std::cout << "Checked AITER_ASM_DIR path: " << path << " does not exist." << std::endl;
      }
    }
  });
}


const bool aiterAsmDirInitialized = (set_aiter_asm_dir(), true);

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
