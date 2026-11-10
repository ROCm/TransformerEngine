/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <utility>

#include <hip/hip_runtime_api.h>

#include <filesystem>
#include <sstream>
#include <thread>
#include <unistd.h>
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


bool open_ck_fused_attn_log_file(std::ofstream& log_file, const char* file_prefix, const std::string& log_dir_str) {
  // Explicitly use std::cout as a fallback
  std::filesystem::path log_dir(log_dir_str);
  std::ostringstream filename;
  filename << file_prefix << "_" << getpid() << "_" << std::this_thread::get_id() << ".log";
  log_file.open(log_dir / filename.str(), std::ios_base::app);
  if (!log_file.is_open()) {
    std::cerr << "Failed to open log file: " << (log_dir / filename.str()) << "\n";
    return false;
  }
  return true;
}

std::ostream* get_ck_log_stream() {
  thread_local std::ofstream log_file;
  thread_local std::ostream* log_stream = nullptr;
  thread_local bool initialized = false;
  if (!initialized) {
    initialized = true;
    if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG")) {
      std::string log_dir_str(env_p);
      if (!log_dir_str.empty() && log_dir_str != "0") {
        if (log_dir_str == "1") {
          log_stream = &std::cout;
        }
        else if (open_ck_fused_attn_log_file(log_file, "ck_fused_attn", log_dir_str)) {
          log_stream = &log_file;
        }
      }
    }
  }

  return log_stream;
}

// Return AITER's data_type kernel-selection tag. The tag may encode a
// combination of input and output types rather than the input dtype alone.
std::string get_data_type_str(DType dtype){
  std::string data_type_str;
  if(dtype==DType::kFloat16){
    data_type_str = "fp16";
  }else if(dtype==DType::kBFloat16){
    data_type_str = "bf16";
  }else if(dtype==DType::kFloat8){
    data_type_str = "fp8bf16";
    // "fp8bf16" selects FP8 input with BF16 output; "fp8" selects FP8 output.
  }else{
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

//get ck_tile bias_type and CK_FUSED_ATTN bias_shape from a fwd/bwd args struct
std::pair<bias_enum, BiasShape> get_ck_bias_type_shape(const CKAttnCommonArgs* args){
  bias_enum bias_type;
  BiasShape bias_shape;
  if (args->attn_bias_type==BiasType::no_bias){
    bias_type = bias_enum::no_bias;
  }else if (args->attn_bias_type==BiasType::elementwise_bias){
    bias_type = bias_enum::elementwise_bias;
    bias_shape = get_bias_shape(args->b, args->h, args->bias_b, args->bias_h);
  }else if (args->attn_bias_type==BiasType::alibi){
    bias_type = bias_enum::alibi;
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid bias_type in ck_fused_attn.");
  }
  return std::make_pair(bias_type, bias_shape);
}

namespace {

constexpr int kMaxSeqlenThreads = 256;

// One block per cu_seqlens array: block x reduces max(cu[i+1] - cu[i]) over i in [0, b) and writes out[x].
// No atomics, so the output needs no pre-zeroing. Negative lengths (malformed input) clamp to 0.
__global__ void __launch_bounds__(kMaxSeqlenThreads) runtime_max_seqlen_kernel(
  uint64_t b,
  const int32_t* cu_seqlen_0,
  const int32_t* cu_seqlen_1,
  uint64_t* out) {
  __shared__ int32_t smem[kMaxSeqlenThreads];
  const int32_t* cu_seqlen = blockIdx.x == 0 ? cu_seqlen_0 : cu_seqlen_1;
  const int tid = threadIdx.x;

  int32_t local_max = 0;
  for(uint64_t i = tid; i < b; i += kMaxSeqlenThreads){
    local_max = max(local_max, cu_seqlen[i + 1] - cu_seqlen[i]);
  }
  smem[tid] = local_max;
  __syncthreads();
  for(int stride = kMaxSeqlenThreads / 2; stride > 0; stride >>= 1){
    if(tid < stride){
      smem[tid] = max(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  if(tid == 0){
    out[blockIdx.x] = static_cast<uint64_t>(smem[0]);
  }
}

void check_hip(hipError_t err, const char* what){
  if(err != hipSuccess){
    throw std::runtime_error(std::string("ck_fused_attn: ") + what + " failed: " + hipGetErrorString(err));
  }
}

// Reduce num_arrays (1 or 2) cu_seqlens arrays in a single launch with a single blocking readback.
// workspace must hold num_arrays uint64_t values.
void runtime_max_seqlens(uint64_t b, int num_arrays, const void* cu_seqlen_0, const void* cu_seqlen_1,
                         void* workspace, uint64_t* host_out, hipStream_t stream){
  uint64_t* dev_out = static_cast<uint64_t*>(workspace);
  runtime_max_seqlen_kernel<<<num_arrays, kMaxSeqlenThreads, 0, stream>>>(
    b,
    static_cast<const int32_t*>(cu_seqlen_0),
    static_cast<const int32_t*>(cu_seqlen_1),
    dev_out);
  check_hip(hipGetLastError(), "runtime_max_seqlen_kernel launch");
  check_hip(hipMemcpyAsync(host_out, dev_out, num_arrays * sizeof(uint64_t), hipMemcpyDeviceToHost, stream),
            "runtime max_seqlen readback");
  check_hip(hipStreamSynchronize(stream), "runtime max_seqlen sync");
}

}  // namespace

uint64_t get_runtime_max_seqlen(uint64_t b, const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr, void* workspace, hipStream_t stream){
  uint64_t result = 0;
  runtime_max_seqlens(b, 1, cu_seqlen_padded_ptr ? cu_seqlen_padded_ptr : cu_seqlen_ptr, nullptr,
                      workspace, &result, stream);
  return result;
}

std::pair<uint64_t, uint64_t> get_runtime_max_seqlens(
  uint64_t b,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_q_padded_ptr,
  const void* cu_seqlen_kv_ptr, const void* cu_seqlen_kv_padded_ptr,
  void* workspace, hipStream_t stream){
  uint64_t result[2] = {0, 0};
  runtime_max_seqlens(b, 2,
                      cu_seqlen_q_padded_ptr ? cu_seqlen_q_padded_ptr : cu_seqlen_q_ptr,
                      cu_seqlen_kv_padded_ptr ? cu_seqlen_kv_padded_ptr : cu_seqlen_kv_ptr,
                      workspace, result, stream);
  return {result[0], result[1]};
}

}//namespace ck_fused_attn
