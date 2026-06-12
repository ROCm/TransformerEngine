/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <string>
#include <numeric> // Required for std::accumulate
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef USE_FUSED_ATTN_CK
#include <ck_fused_attn/ck_fused_attn.hpp>
#endif // USE_FUSED_ATTN_CK
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_ck.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

#ifdef USE_FUSED_ATTN_CK
// Returns false when a CK backward for this config would dispatch to the CK v2
// launcher (fmha_bwd / prepare_workspace_async), which schedules self-deleting
// hipLaunchHostFunc nodes that double-free on graph replay. Only the v3 asm bwd
// path is HIP-graph-replay-safe. Mirrors AITER's fmha_v3_bwd gate (mha_bwd.cu)
// for the conditions visible at backend-selection time; determinism is applied
// separately on the framework side.
static bool is_ck_bwd_graph_capture_safe(
  NVTE_Bias_Type bias_type,
  float dropout,
  size_t max_seqlen_q) {
  // The CK v2 launcher is reached whenever the v3 asm bwd path is not taken.
  // v3 requires gfx942/gfx950, no dropout, no bias, and (per TE's use_asm_v3 rule
  // in ck_fused_attn_bwd.cpp) max_seqlen_q >= 16. NVTE_CK_USES_BWD_V3 can force the
  // v2 path off entirely.
  bool uses_bwd_v3 = getenv<int>("NVTE_CK_USES_BWD_V3", 1);
  const std::string& arch = cuda::sm_arch_name();
  bool is_v3_arch = arch.find("gfx942") != std::string::npos ||
                    arch.find("gfx950") != std::string::npos;
  return uses_bwd_v3 &&
         is_v3_arch &&
         dropout == 0.f &&
         bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
         max_seqlen_q >= 16;
}
#endif // USE_FUSED_ATTN_CK

// check the fused attn config to see whether it's ck backend supported
// single filtering followed by joint filtering
bool is_ck_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  NVTE_Softmax_Type softmax_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk,
  size_t head_dim_v,
  int64_t window_size_left,
  int64_t window_size_right,
  bool is_training, bool cuda_graph) {

#ifdef USE_FUSED_ATTN_CK

  // debug info setting
  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }
  
  // single filters
  
  // filter based on num_heads and num_gqa_groups
  if(num_gqa_groups == 0 || num_attn_heads%num_gqa_groups != 0){
    if(nvte_log_ck_config){
      std::cout<<"Num of attention heads must be divisible by num of gqa groups"<<std::endl;
    }
    return false;
  }

  // filter based on data type
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    if(nvte_log_ck_config){
      std::cout<<"q, k, v data type has to be fp16 or bf16"<<std::endl;
    }
    return false;
  }

  // filter based on bias type
  // CK does not support pre_scale bias
  if(!(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS || bias_type == NVTE_Bias_Type::NVTE_ALIBI || bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)){
    if(nvte_log_ck_config){
      std::cout<<"CK fused attn does not support pre_scale bias"<<std::endl;
    }
    return false;
  }

  // filter based on head_dim
  // AITER/ck does not support hdim>=512
  if(head_dim_qk >= 512 || head_dim_v >= 512){
    if(nvte_log_ck_config){
      std::cout<<"AITER/CK fused attn does not support head dim >=512 yet"<<std::endl;
    }
    return false;
  }

  // filter based on softmax type
  if(softmax_type!=NVTE_VANILLA_SOFTMAX){
    if(nvte_log_ck_config){
      std::cout<<"AITER/CK fused attn does not support learnable sink yet"<<std::endl;
    }
    return false;
  }

  // joint filter based on sliding window and attn_mask
  bool is_causal = (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
                    attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK||
                    attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK||
                    attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  if(is_causal){
    // causal mask window must be with causal top left or causal bottom right mask type
    if (!((window_size_left ==-1 || window_size_left >=0) && window_size_right ==0 )){
      if(nvte_log_ck_config){
        std::cout<<"When mask contains causal, window size should be (-1, 0) or (>=0, 0)"<<std::endl;
      }
      return false;
    }
  }else if(attn_mask_type==NVTE_Mask_Type::NVTE_NO_MASK || attn_mask_type==NVTE_Mask_Type::NVTE_PADDING_MASK){
    // no mask must be with either (-1, -1) or (>=0, >=0)
    if (!((window_size_left == -1 && window_size_right == -1)||(window_size_left >= 0 && window_size_right >= 0))){
      if(nvte_log_ck_config){
        std::cout<<"When no mask, window size should be (-1, -1) or (>=0, >=0)"<<std::endl;
      }
      return false;
    }
  }

  // joint filter that MQA/GQA does not work with qkvpacked layout
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  bool is_qkvpacked = layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D;
  bool is_mqa_gqa = num_attn_heads > num_gqa_groups;
  if(is_mqa_gqa && is_qkvpacked){
    if(nvte_log_ck_config){
      std::cout<<"MQA/GQA cannot work with qkvpacked layout"<<std::endl;
    }
    return false;
  }
  
  // joint filter that qkvpacked layout requires seq length to be the same
  if(is_qkvpacked && max_seqlen_q!=max_seqlen_kv){
    if(nvte_log_ck_config){
      std::cout<<"qkv packed layout requires seqlen_q==seqlen_kv"<<std::endl;
    }
    return false;
  }

  // joint filter that THD layout should imply padding mask  
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  bool is_ragged = qkv_format==NVTE_QKV_Format::NVTE_THD;
  // in NVTE, padding can happen in both THD format or BSHD/SBHD format
  // For THD format, padding is natural
  // For BSHD/SBHD, padding can be inferred by a cu_seqlen which shows the actual seqlen for each batch, while the dim(S) is the max_seqlen
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  if(is_ragged && !is_padding){
    if(nvte_log_ck_config){
      std::cout<<"Ragged QKV input requires padding mask"<<std::endl;
    }
    return false;
  }
  
  // joint filter that THD/padding does not work with ALIBI bias or post_scale_bias
  if(is_padding && (bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS || bias_type==NVTE_Bias_Type::NVTE_ALIBI)){
    if(nvte_log_ck_config){
      std::cout<<"padding mask cannot work with post_scale_bias or alibi"<<std::endl;
    }
    return false;
  }

  // Under HIP-graph capture, CK backward must take the graph-replay-safe v3 asm
  // path; a config that would fall back to the CK v2 launcher is not graph-safe.
  // Reject such graph-captured training configs so selection falls through to a
  // graph-safe backend (the v2 host-pack hazard is backward-only, so inference is
  // unaffected). Determinism also forces v2 but is invisible here, so it is handled
  // on the framework side.
  if(is_training && cuda_graph &&
     !is_ck_bwd_graph_capture_safe(bias_type, dropout, max_seqlen_q)){
    if(nvte_log_ck_config){
      std::cout<<"CK backward would use the v2 launcher, which is not HIP-graph-replay-safe"<<std::endl;
    }
    return false;
  }
  return true;
#else
  return false;
#endif // USE_FUSED_ATTN_CK
}


#ifdef USE_FUSED_ATTN_CK
ck_fused_attn::DType nvte_to_ck_dtype(DType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == DType::aname) return ck_fused_attn::DType::dtname
  CAST_TYPE(kFloat16, kFloat16);
  CAST_TYPE(kBFloat16, kBFloat16);
  return ck_fused_attn::DType::kNumTypes;
#undef CAST_TYPE
}

ck_fused_attn::BiasType nvte_to_ck_bias_type(NVTE_Bias_Type t_bias_type){
#define CAST_TYPE(aname, dtname) if (t_bias_type == NVTE_Bias_Type::aname) return ck_fused_attn::BiasType::dtname
  CAST_TYPE(NVTE_NO_BIAS, no_bias);
  CAST_TYPE(NVTE_POST_SCALE_BIAS, elementwise_bias);
  CAST_TYPE(NVTE_ALIBI, alibi);
  return ck_fused_attn::BiasType::no_bias;
#undef CAST_TYPE
}

//         attn_mask_type                              |   window_size
//    -------------------------------------------------------------------------
//    no_mask, padding, arbitrary                      | (-1, -1) or (>=0, >=0)
//    causal, padding_causal                           | (-1,  0) or (>=0, 0)
//    causal_bottom_right, padding_causal_bottom_right | (-1,  0) or (>=0, 0)

// set the ck mask type based on nvte mask type and window size table above
ck_fused_attn::MaskType set_ck_mask(NVTE_Mask_Type nvte_mask_type, int64_t nvte_window_size_left, int64_t nvte_window_size_right){
  if (nvte_mask_type==NVTE_Mask_Type::NVTE_NO_MASK || nvte_mask_type==NVTE_Mask_Type::NVTE_PADDING_MASK){
    // window size in NVTE_NO_Mask can be (-1, -1) and (>=0, >=0)
    if(nvte_window_size_left==-1 && nvte_window_size_right==-1){
      // (-1, -1)
      return ck_fused_attn::MaskType::no_mask;
    }else{
      // (>=0, >=0)
      return ck_fused_attn::MaskType::mask_bottom_right;
    }
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK || nvte_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK){
    // nvte causal mask can map to (-1, 0) or (>=0, 0)
    return ck_fused_attn::MaskType::mask_top_left;
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK || nvte_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK){
    return ck_fused_attn::MaskType::mask_bottom_right;
  }
  return ck_fused_attn::MaskType::window_generic;
}

// Single-pass workspace allocator. In sizing mode (constructed with base==nullptr)
// allocate() returns nullptr but still accumulates total(); in execution mode it
// hands out chunks of the user-provided buffer. Same call sequence runs in both
// modes, so the size pass and the use pass cannot drift.
class WorkspacePlanner {
 public:
  explicit WorkspacePlanner(void* base)
    : cursor_(static_cast<int8_t*>(base)), is_sizing_(base == nullptr) {}

  void* allocate(size_t bytes) {
    void* ptr = is_sizing_ ? nullptr : static_cast<void*>(cursor_);
    if (!is_sizing_) cursor_ += bytes;
    total_ += bytes;
    return ptr;
  }

  size_t total() const { return total_; }
  bool is_sizing() const { return is_sizing_; }

 private:
  int8_t* cursor_;
  size_t total_ = 0;
  bool is_sizing_;
};

// nullptr-safe byte arithmetic for carving sub-buffers out of a planner allocation.
inline void* byte_offset(void* ptr, ptrdiff_t n) {
  return ptr ? static_cast<void*>(static_cast<int8_t*>(ptr) + n) : nullptr;
}
inline ptrdiff_t byte_diff(const void* a, const void* b) {
  return static_cast<const int8_t*>(a) - static_cast<const int8_t*>(b);
}

__global__
void generate_cu_seqlen_padded_kernel(
  uint32_t s_q, uint32_t s_kv, uint32_t b,
  int32_t* cu_seqlen_q_padded_ptr,
  int32_t* cu_seqlen_kv_padded_ptr
){
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < b+1; i += blockDim.x * gridDim.x){
    cu_seqlen_q_padded_ptr[i] = s_q * i;
    cu_seqlen_kv_padded_ptr[i] = s_kv * i;
  }
}

void generate_cu_seqlen_padded(
  uint32_t s_q, uint32_t s_kv, uint32_t b,
  void* cu_seqlen_q_padded_ptr,
  void* cu_seqlen_kv_padded_ptr,
  hipStream_t stream
){
  constexpr int THREADS_PER_BLOCK = 256;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * (b+1)/THREADS_PER_BLOCK));
  generate_cu_seqlen_padded_kernel<<<grid, block, 0, stream>>>(
    s_q, s_kv, b,
    static_cast<int32_t*>(cu_seqlen_q_padded_ptr),
    static_cast<int32_t*>(cu_seqlen_kv_padded_ptr)
  );
}

__global__ 
void generate_alibi_slope(uint64_t h, float* alibi_slope_ptr){
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < h; id += blockDim.x * gridDim.x){
    int n = exp2(floor(log2(h)));
    double m_0 = exp2(-8.0/n);
    if(id < n){
      //first n elements are pow(m_0, [1, 2, 3, ... n])
      alibi_slope_ptr[id] = pow(m_0, id + 1);
    }else{
      double m_hat_0 = exp2(-4.0/n);
      //(n+1, ... h) elements are pow(m_hat_0, [1, 3, 5, ...])
      alibi_slope_ptr[id] = pow(m_hat_0, 1 + (id - n)*2);
    }
  }
}

// no device std::upper_bound
// in an increasing array with given size len, search for the index that:
// array[index] <= target < array[index+1]
// guaranteed that target >=0 and target <= cu_seqlen[end-1]
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

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1250__)
constexpr int THREADS_PER_WAVEFRONT = 32;
#else
constexpr int THREADS_PER_WAVEFRONT = 64;
#endif
// Direction for pad_remap / pad_remap_lse:
//   Remove: padded -> unpadded
//   Add   : unpadded -> padded
enum class PadDirection { Remove, Add };

// kernel to copy q, k, v, o (dq, dk, dv, do) between padded and unpadded layouts.
// each wavefront is in charge of one token index (h*d*sizeof(DataType) bytes copy)
// one workitem (thread) in one wavefront is charge of one element in 32 segment trunk of h*d
template<typename DataType, bool is_ragged, PadDirection dir>
__global__ void pad_remap_kernel(
  // b guaranteed to be correct, extracted from input_cu_seqlens->data.shape[0] - 1
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  DataType* padded_ptr,
  const int32_t* cu_seqlen_ptr, const int32_t* cu_seqlen_padded_ptr,
  DataType* unpadded_ptr){

  int wavefront_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WAVEFRONT;
  int workitem_idx = threadIdx.x % THREADS_PER_WAVEFRONT;
  int num_wavefronts = (blockDim.x * gridDim.x) / THREADS_PER_WAVEFRONT;
  int num_total_tokens = cu_seqlen_ptr[b];
  //BSHD SBHD --> THD (BSHD)
  uint64_t stride_total_seqlen = std::min(stride_b, stride_s);
  for(int token_id = wavefront_idx; token_id<num_total_tokens; token_id += num_wavefronts){
    int b_idx = binary_search(token_id, cu_seqlen_ptr, b+1);
    int s_idx = token_id - cu_seqlen_ptr[b_idx];
    DataType* unpadded_cur = nullptr;
    DataType* padded_cur = nullptr;
    if constexpr(is_ragged){
      unpadded_cur = unpadded_ptr + stride_s* token_id;
      padded_cur = padded_ptr + (cu_seqlen_padded_ptr[b_idx] + s_idx)*stride_s;
    }else{
      unpadded_cur = unpadded_ptr + stride_total_seqlen* token_id;
      padded_cur = padded_ptr + (b_idx * stride_b + s_idx*stride_s);
    }
    for(int hd_idx = workitem_idx; hd_idx < h*d; hd_idx+=THREADS_PER_WAVEFRONT){
      int h_idx = hd_idx/d;
      int d_idx = hd_idx%d;
      if constexpr(dir == PadDirection::Add){
        padded_cur[h_idx*stride_h + d_idx] = unpadded_cur[h_idx*stride_h + d_idx];
      }else{
        unpadded_cur[h_idx*stride_h + d_idx] = padded_cur[h_idx*stride_h + d_idx];
      }
    }
  }
}

// kernel launcher for q, k, v, o (dq, dk, dv, do) padded<->unpadded copy
template<PadDirection dir>
void pad_remap(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  uint64_t max_tokens,
  bool is_ragged,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  void* padded_ptr,
  const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr,
  void* unpadded_ptr,
  hipStream_t stream){

  // cu_seqlen_padded_ptr can be nullptr
  assert(cu_seqlen_ptr!=nullptr);
  constexpr int THREADS_PER_BLOCK = 256;
  // parallel over h*d dimension
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * max_tokens * cuda::warp_size()/THREADS_PER_BLOCK));

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, DataType,
    if(is_ragged){
      pad_remap_kernel<DataType, true, dir><<<grid, block, 0, stream>>>(
        b, h, s, d,
        stride_b, stride_h, stride_s,
        static_cast<DataType*>(padded_ptr),
        static_cast<const int32_t*>(cu_seqlen_ptr),
        static_cast<const int32_t*>(cu_seqlen_padded_ptr),
        static_cast<DataType*>(unpadded_ptr));
    }else{
      pad_remap_kernel<DataType, false, dir><<<grid, block, 0, stream>>>(
        b, h, s, d,
        stride_b, stride_h, stride_s,
        static_cast<DataType*>(padded_ptr),
        static_cast<const int32_t*>(cu_seqlen_ptr),
        static_cast<const int32_t*>(cu_seqlen_padded_ptr),
        static_cast<DataType*>(unpadded_ptr));
    });
}

// Cuda kernel to copy softmax lse between padded and unpadded layouts.
// Also performs a format conversion between CK[h, total_tokens_q] and
// NVTE[total_tokens_q, h] formats. 
// padded layout (selected by is_ragged):
//   is_ragged=false: [b, h, s_q]
//   is_ragged=true : [max_tokens_q with padding, h]
// unpadded layout: [h, max_tokens_q]
// One wavefront in charge of one token index (h*sizeof(float) bytes)
// One workitem (thread) in one wavefront is charge of one element in THREADS_PER_WAVE_FRONT(64) segment trunk of h
template<bool is_ragged, PadDirection dir>
__global__ void pad_remap_lse_kernel(
  uint64_t b, uint64_t h, uint64_t s_q,
  uint64_t max_tokens_q,
  float* padded_lse_ptr,
  const int32_t* cu_seqlen_q_ptr, const int32_t* cu_seqlen_q_padded_ptr,
  float* unpadded_lse_ptr){

  int wavefront_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WAVEFRONT;
  int workitem_idx = threadIdx.x % THREADS_PER_WAVEFRONT;
  int num_wavefronts = (blockDim.x * gridDim.x) / THREADS_PER_WAVEFRONT;
  int num_total_tokens = cu_seqlen_q_ptr[b];

  for(int token_id = wavefront_idx; token_id < num_total_tokens; token_id += num_wavefronts){
    int b_idx = binary_search(token_id, cu_seqlen_q_ptr, b+1);
    int s_idx = token_id - cu_seqlen_q_ptr[b_idx];

    for(int h_idx = workitem_idx; h_idx < h; h_idx += THREADS_PER_WAVEFRONT){
      uint64_t padded_idx;
      if constexpr(is_ragged){
        padded_idx = (cu_seqlen_q_padded_ptr[b_idx]+s_idx)*h + h_idx;
      }else{
        padded_idx = b_idx*h*s_q + h_idx*s_q + s_idx;
      }
      uint64_t unpadded_idx = h_idx * max_tokens_q + token_id;
      if constexpr(dir == PadDirection::Add){
        padded_lse_ptr[padded_idx] = unpadded_lse_ptr[unpadded_idx];
      }else{
        unpadded_lse_ptr[unpadded_idx] = padded_lse_ptr[padded_idx];
      }
    }
  }
}

// kernel launcher for softmax_lse padded<->unpadded copy
template<PadDirection dir>
void pad_remap_lse(
  uint64_t b, uint64_t h, uint64_t s_q,
  uint64_t max_tokens_q,
  bool is_ragged,
  void* padded_lse_ptr,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_q_padded_ptr,
  void* unpadded_lse_ptr,
  hipStream_t stream){

  constexpr int THREADS_PER_BLOCK = 256;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * max_tokens_q * cuda::warp_size()/THREADS_PER_BLOCK));
  if(is_ragged){
    pad_remap_lse_kernel<true, dir><<<grid, block, 0, stream>>>(
      b, h, s_q, max_tokens_q,
      static_cast<float*>(padded_lse_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_padded_ptr),
      static_cast<float*>(unpadded_lse_ptr));
  }else{
    pad_remap_lse_kernel<false, dir><<<grid, block, 0, stream>>>(
      b, h, s_q, max_tokens_q,
      static_cast<float*>(padded_lse_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_padded_ptr),
      static_cast<float*>(unpadded_lse_ptr));
  }
}

// Flash-Decoding split-count heuristic, ported from AITER's CK example
// (3rdparty/composable_kernel/example/ck_tile/01_fmha/fmha_fwd_runner.hpp::
// num_splits_heuristic). Ported rather than included because that header is a
// host example unit and mha_common.h's variant pulls in torch types. Note the
// mha_common.h variant derives num_n_blocks from hdim_v and degenerates to 1;
// this occupancy-based version is the one that actually yields splits > 1.
// Returns 1 when the launch already fills the GPU.
static int ck_num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int max_splits) {
  if (batch_nhead_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min(max_splits, num_SMs);
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
    float eff = n_waves / std::ceil(n_waves);
    if (eff > max_efficiency) {
      max_efficiency = eff;
    }
    efficiency.push_back(eff);
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      return num_splits;
    }
  }
  return 1;
}

// Auto-select the split-KV count for a forward call. Deterministic in
// (shape, device), so the workspace sizing pass and the execution pass agree.
// Returns 1 (no split) on any failure or when dropout is active.
static int ck_compute_num_splits(uint64_t b, uint64_t h, uint64_t max_seqlen_q,
                                 float dropout_probability) {
  if (dropout_probability != 0.f) {
    return 1;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    return 1;
  }
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, device) != hipSuccess) {
    return 1;
  }
  // kM0 tile must match AITER's splitkv kernel generation (generate.py).
  const int kM0 = 64;
  const int num_m_blocks = (static_cast<int>(max_seqlen_q) + kM0 - 1) / kM0;
  return ck_num_splits_heuristic(static_cast<int>(b * h) * num_m_blocks,
                                 props.multiProcessorCount * 2, /*max_splits=*/128);
}

// actual fwd implementation, calling ck api directly
void fused_attn_ck_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v, uint64_t bias_b, uint64_t bias_h,
  size_t max_tokens_q, size_t max_tokens_kv,
  bool is_training, float scaling_factor, float dropout_probability,
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void *devPtrQ, void *devPtrK, void *devPtrV, void* devPtrBias,
  void *devPtrSoftmaxAux, void *devPtrO,
  void* devPtrDropoutSeed, void* devPtrDropoutOffset,
  void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  void* devPtrSeqOffsetsQ, void* devPtrSeqOffsetsKV,
  DType dtype,
  void *workspace, 
  size_t *workspace_size,
  cudaStream_t stream){

  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }

  bool nvte_ck_uses_fwd_v3 = getenv<int>("NVTE_CK_USES_FWD_V3", 1);
  int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);
  bool nvte_ck_zero_out_pad = getenv<int>("NVTE_CK_ZERO_OUT_PAD", 1);
  bool nvte_ck_uses_splitkv = getenv<int>("NVTE_CK_USES_SPLITKV", 1);
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(layout);
  bool is_ragged = qkv_format==NVTE_QKV_Format::NVTE_THD;
  bool is_SBHD = qkv_format==NVTE_QKV_Format::NVTE_SBHD || qkv_format==NVTE_QKV_Format::NVTE_SBHD_2BSHD;
  bool is_BSHD = qkv_format==NVTE_QKV_Format::NVTE_BSHD;

  bool is_padding = (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
                     mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool bshd_to_thd = is_BSHD && is_padding;

  // Split-KV (Flash-Decoding) is wired into the group-mode paths only (those
  // setting cu_seqlens: SBHD+padding, BSHD->THD, ragged THD), because the
  // accumulator-buffer layout below follows AITER's varlen splitkv reference.
  // It is restricted to inference (is_training==false) so the standard
  // training fwd/bwd LSE contract is untouched, and only engages when the
  // occupancy heuristic asks for more than one split.
  bool is_group_path = (is_SBHD && is_padding) || bshd_to_thd || is_ragged;
  int num_splits = 1;
  if(nvte_ck_uses_splitkv && !is_training && is_group_path){
    num_splits = ck_compute_num_splits(b, h, s_q, dropout_probability);
  }
  bool use_splitkv = num_splits > 1;

  // extract the qkv and o storage bytes to allocate buffer for padding removing
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  size_t q_storage_bytes = max_tokens_q*h*d_qk*nvte_dtype_size(dtype);
  size_t k_storage_bytes = max_tokens_kv*hg*d_qk*nvte_dtype_size(dtype);
  size_t v_storage_bytes = max_tokens_kv*hg*d_v*nvte_dtype_size(dtype);
  size_t o_storage_bytes = max_tokens_q*h*d_v*nvte_dtype_size(dtype);

  // Reserve workspace chunks. Same allocation sequence runs in sizing mode
  // (planner returns nullptr, accumulates total) and execution mode.
  WorkspacePlanner planner(workspace);

  void* devPtrAlibiSlope = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    // ck requires an alibi slope array even if in standard (vanilla) mode
    devPtrAlibiSlope = planner.allocate(h*sizeof(float));
  }

  void* devPtrSoftmaxLSEWithoutPadding = nullptr;
  if((is_SBHD && is_padding) || bshd_to_thd || is_ragged){
    devPtrSoftmaxLSEWithoutPadding = planner.allocate(h*max_tokens_q*sizeof(float));
  }

  // Split-KV fp32 accumulators: lse_acc [h, num_splits, max_tokens_q] and
  // o_acc [h, num_splits, max_tokens_q, d_v]. Sized only when splitkv engages;
  // num_splits is deterministic so this matches in the sizing and execution passes.
  void* devPtrLseAcc = nullptr;
  void* devPtrOAcc = nullptr;
  if(use_splitkv){
    devPtrLseAcc = planner.allocate(static_cast<size_t>(h)*num_splits*max_tokens_q*sizeof(float));
    devPtrOAcc   = planner.allocate(static_cast<size_t>(h)*num_splits*max_tokens_q*d_v*sizeof(float));
  }

  void* devPtrQWithoutPadding = nullptr;
  void* devPtrKWithoutPadding = nullptr;
  void* devPtrVWithoutPadding = nullptr;
  void* devPtrOWithoutPadding = nullptr;
  void* devPtrCuSeqlenPaddedQ = devPtrSeqOffsetsQ;
  void* devPtrCuSeqlenPaddedKV = devPtrSeqOffsetsKV;

  if(is_SBHD && is_padding){
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      // qkv packed: one combined buffer; carve k/v at the same byte offsets as the input qkv tensor
      void* qkv_buf = planner.allocate(q_storage_bytes + k_storage_bytes + v_storage_bytes);
      devPtrQWithoutPadding = qkv_buf;
      devPtrKWithoutPadding = byte_offset(qkv_buf, byte_diff(devPtrK, devPtrQ));
      devPtrVWithoutPadding = byte_offset(qkv_buf, byte_diff(devPtrV, devPtrQ));
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD || layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      // kv packed: separate q buffer, combined kv buffer
      devPtrQWithoutPadding = planner.allocate(q_storage_bytes);
      void* kv_buf = planner.allocate(k_storage_bytes + v_storage_bytes);
      devPtrKWithoutPadding = kv_buf;
      devPtrVWithoutPadding = byte_offset(kv_buf, byte_diff(devPtrV, devPtrK));
    }else{
      // qkv separated
      devPtrQWithoutPadding = planner.allocate(q_storage_bytes);
      devPtrKWithoutPadding = planner.allocate(k_storage_bytes);
      devPtrVWithoutPadding = planner.allocate(v_storage_bytes);
    }
    devPtrOWithoutPadding = planner.allocate(o_storage_bytes);
  }else if(bshd_to_thd){
    devPtrCuSeqlenPaddedQ = planner.allocate((b+1)*sizeof(int32_t));
    devPtrCuSeqlenPaddedKV = planner.allocate((b+1)*sizeof(int32_t));
  }

  // Sizing-only path: report and exit before any kernel/memset.
  if(planner.is_sizing()){
    *workspace_size = planner.total();
    if(nvte_log_ck_config){
      std::cout<<std::endl<<"attn_fwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }

  std::array<uint64_t, 4> q_stride;
  std::array<uint64_t, 4> k_stride;
  std::array<uint64_t, 4> v_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  // Initialize workspace buffers.
  if(devPtrAlibiSlope){
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
  }
  if(bshd_to_thd){
    generate_cu_seqlen_padded(s_q, s_kv, b, devPtrCuSeqlenPaddedQ, devPtrCuSeqlenPaddedKV, stream);
    if(nvte_log_ck_config){
      std::cout << "\nattn_fwd(ck): generating cu_seqlen_padded in BSHD+padding to THD+padding conversion.\n";
    }
  }
  if(is_padding && nvte_ck_zero_out_pad){
    // reset the final results since padded places need to be 0
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrO, 0, o_storage_bytes, stream));
  }

  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_fwd(ck): ";
    std::cout<<"layout: "<<layout<<", ";
    std::cout<<"max_tokens_q: "<<max_tokens_q<<", ";
    std::cout<<"max_tokens_kv: "<<max_tokens_kv<<", ";
    std::cout<<"is_ragged: "<<is_ragged<<", ";
    std::cout<<"is_padding: "<<is_padding<<", ";
    std::cout<<"is_SBHD: "<<is_SBHD<<", ";
    std::cout<<"is_BSHD: "<<is_BSHD<<", ";
    std::cout<<"bshd_to_thd: "<<bshd_to_thd<<", ";
    std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d_qk<<"), ";
    std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
    std::cout<<"k_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d_qk<<"), ";
    std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
    std::cout<<"v_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d_v<<"), ";
    std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";
    std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d_v<<"), ";
    std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    if(is_ragged){
      std::cout<<"M_shape: ("<<h<<", "<<max_tokens_q<<"), ";
      std::cout<<"M_stride: ("<<max_tokens_q<<", "<<1<<"), ";
    }else{
      std::cout<<"M_shape: ("<<b<<", "<<h<<", "<<s_q<<"), ";
      std::cout<<"M_stride: ("<<h*s_q<<", "<<s_q<<", "<<1<<"), ";
    }
    std::cout<<"is_training: "<<is_training<<", ";
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<", ";
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<", ";
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<")"<<", ";
    std::cout<<"nvte_ck_uses_fwd_v3: "<<nvte_ck_uses_fwd_v3<<", ";
    std::cout<<"use_splitkv: "<<use_splitkv<<", num_splits: "<<num_splits<<std::endl;
  }
  // Common fields filled here; mode-specific fields are overwritten below.
  ck_fused_attn::CKAttnFwdArgs ck_args;
  ck_args.dtype = nvte_to_ck_dtype(dtype);
  ck_args.b = b; ck_args.h = h; ck_args.hg = hg;
  ck_args.s_q = s_q; ck_args.s_kv = s_kv; ck_args.d_qk = d_qk; ck_args.d_v = d_v;
  ck_args.is_training = is_training;
  ck_args.scaling_factor = scaling_factor;
  ck_args.dropout_probability = dropout_probability;
  ck_args.philox_seed_ptr = devPtrDropoutSeed;
  ck_args.philox_offset_ptr = devPtrDropoutOffset;
  ck_args.attn_mask_type = set_ck_mask(mask_type, window_size_left, window_size_right);
  ck_args.window_size_left = window_size_left;
  ck_args.window_size_right = window_size_right;
  ck_args.uses_fwd_v3 = nvte_ck_uses_fwd_v3;
  ck_args.how_v3_bf16_cvt = nvte_ck_how_v3_bf16_cvt;

  if(is_SBHD && is_padding){
    // remove padding for q, k, v
    pad_remap<PadDirection::Remove>(dtype, b, h, s_q, d_qk, max_tokens_q, false, q_stride[0], q_stride[1], q_stride[2], devPtrQ, devPtrCuSeqlensQ, devPtrCuSeqlenPaddedQ, devPtrQWithoutPadding, stream);
    pad_remap<PadDirection::Remove>(dtype, b, hg, s_kv, d_qk, max_tokens_kv, false, k_stride[0], k_stride[1], k_stride[2], devPtrK, devPtrCuSeqlensKV, devPtrCuSeqlenPaddedKV, devPtrKWithoutPadding, stream);
    pad_remap<PadDirection::Remove>(dtype, b, hg, s_kv, d_v, max_tokens_kv, false, v_stride[0], v_stride[1], v_stride[2], devPtrV, devPtrCuSeqlensKV, devPtrCuSeqlenPaddedKV, devPtrVWithoutPadding, stream);
    // call varlen api using without_padding ptrs
    // for BSHD/SBHD, after padding removal, THD require stride_s update
    ck_args.max_tokens_q = max_tokens_q;
    ck_args.q_ptr = devPtrQWithoutPadding;
    ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[0];
    ck_args.k_ptr = devPtrKWithoutPadding;
    ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = std::min(k_stride[0], k_stride[2]);
    ck_args.v_ptr = devPtrVWithoutPadding;
    ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = std::min(v_stride[0], v_stride[2]);
    ck_args.cu_seqlen_q_ptr = devPtrCuSeqlensQ;
    ck_args.cu_seqlen_kv_ptr = devPtrCuSeqlensKV;
    ck_args.o_ptr = devPtrOWithoutPadding;
    ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[0];
    ck_args.lse_ptr = devPtrSoftmaxLSEWithoutPadding;
    if(use_splitkv){
      ck_args.num_splits = num_splits;
      ck_args.lse_acc_ptr = devPtrLseAcc;
      ck_args.o_acc_ptr = devPtrOAcc;
      NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd_splitkv(ck_args, stream));
    }else{
      NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd(ck_args, stream));
    }
    // add padding for o and softmax_lse
    pad_remap<PadDirection::Add>(dtype, b, h, s_q, d_v, max_tokens_q, false, o_stride[0], o_stride[1], o_stride[2], devPtrO, devPtrCuSeqlensQ, devPtrCuSeqlenPaddedQ, devPtrOWithoutPadding, stream);
    pad_remap_lse<PadDirection::Add>(b, h, s_q, max_tokens_q, false, devPtrSoftmaxAux, devPtrCuSeqlensQ, devPtrCuSeqlenPaddedQ, devPtrSoftmaxLSEWithoutPadding, stream);
  }else if(bshd_to_thd || is_ragged){
    ck_args.max_tokens_q = max_tokens_q;
    ck_args.q_ptr = devPtrQ;
    ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[2];
    ck_args.k_ptr = devPtrK;
    ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = k_stride[2];
    ck_args.v_ptr = devPtrV;
    ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = v_stride[2];
    ck_args.cu_seqlen_q_ptr = devPtrCuSeqlensQ;
    ck_args.cu_seqlen_kv_ptr = devPtrCuSeqlensKV;
    ck_args.cu_seqlen_q_padded_ptr = devPtrCuSeqlenPaddedQ;
    ck_args.cu_seqlen_kv_padded_ptr = devPtrCuSeqlenPaddedKV;
    ck_args.o_ptr = devPtrO;
    ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[2];
    ck_args.lse_ptr = devPtrSoftmaxLSEWithoutPadding;
    if(use_splitkv){
      ck_args.num_splits = num_splits;
      ck_args.lse_acc_ptr = devPtrLseAcc;
      ck_args.o_acc_ptr = devPtrOAcc;
      NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd_splitkv(ck_args, stream));
    }else{
      NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd(ck_args, stream));
    }
    // aiter asm output softmax_lse with padding
    pad_remap_lse<PadDirection::Add>(b, h, s_q, max_tokens_q, is_ragged, devPtrSoftmaxAux, devPtrCuSeqlenPaddedQ, devPtrCuSeqlenPaddedQ, devPtrSoftmaxLSEWithoutPadding, stream);
  }else{
    ck_args.bias_b = bias_b; ck_args.bias_h = bias_h;
    ck_args.q_ptr = devPtrQ;
    ck_args.stride_b_q = q_stride[0]; ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[2];
    ck_args.k_ptr = devPtrK;
    ck_args.stride_b_k = k_stride[0]; ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = k_stride[2];
    ck_args.v_ptr = devPtrV;
    ck_args.stride_b_v = v_stride[0]; ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = v_stride[2];
    ck_args.bias_ptr = devPtrBias;
    ck_args.alibi_slope_ptr = devPtrAlibiSlope;
    ck_args.attn_bias_type = nvte_to_ck_bias_type(bias_type);
    ck_args.o_ptr = devPtrO;
    ck_args.stride_b_o = o_stride[0]; ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[2];
    ck_args.lse_ptr = devPtrSoftmaxAux;
    NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd(ck_args, stream));
  }
}

void fused_attn_ck_bwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d_qk, uint64_t d_v, uint64_t bias_b, uint64_t bias_h,
  size_t max_tokens_q, size_t max_tokens_kv,
  float scaling_factor, float dropout_probability, 
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  void* devPtrQ, void* devPtrK, void* devPtrV,
  void* devPtrO, void* devPtrSoftmaxAux, void* devPtrBias,
  void* devPtrdQ, void* devPtrdK, void* devPtrdV, 
  void* devPtrdO, 
  void* devPtrdBias,
  void* devPtrDropoutSeed, 
  void* devPtrDropoutOffset,
  void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  void* devPtrSeqOffsetsQ, void* devPtrSeqOffsetsKV,
  DType dtype,
  void *workspace,
  size_t *workspace_size,
  cudaStream_t stream) {
  
  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  } 
  // bwd v3 is optional by enabling the following envs
  // default values follows the ck example setting
  bool nvte_ck_uses_bwd_v3 = getenv<int>("NVTE_CK_USES_BWD_V3", 1);
  bool nvte_ck_is_v3_atomic_fp32 = getenv<int>("NVTE_CK_IS_V3_ATOMIC_FP32", 1);
  int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);

  bool is_mqa_gqa = (h > hg);

  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(layout);
  bool is_ragged = qkv_format==NVTE_QKV_Format::NVTE_THD;
  bool is_SBHD = qkv_format==NVTE_QKV_Format::NVTE_SBHD || qkv_format==NVTE_QKV_Format::NVTE_SBHD_2BSHD;
  bool is_BSHD = qkv_format==NVTE_QKV_Format::NVTE_BSHD;
  bool is_padding = (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
                     mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool bshd_to_thd = is_BSHD && is_padding;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);

  // extract the qkv and o storage bytes to allocate buffer for padding removing
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  size_t q_storage_bytes = max_tokens_q*h*d_qk*nvte_dtype_size(dtype);
  size_t k_storage_bytes = max_tokens_kv*hg*d_qk*nvte_dtype_size(dtype);
  size_t v_storage_bytes = max_tokens_kv*hg*d_v*nvte_dtype_size(dtype);
  size_t o_storage_bytes = max_tokens_q*h*d_v*nvte_dtype_size(dtype);

  // Reserve workspace chunks. Same allocation sequence runs in sizing mode
  // (planner returns nullptr, accumulates total) and execution mode.
  WorkspacePlanner planner(workspace);

  // First h*max_tokens_q*sizeof(float) is the lse-d buffer (passed as softmax_lsed)
  void* lse_workspace = planner.allocate(h*max_tokens_q*sizeof(float));

  void* dk_expanded_ptr = nullptr;
  void* dv_expanded_ptr = nullptr;
  std::array<uint64_t, 4> dk_expanded_stride;
  std::array<uint64_t, 4> dv_expanded_stride;
  if(is_mqa_gqa){
    // generate kv expanded stride as if h_kv = h_q
    generateMatrixStrides(b, h, s_q, s_kv, d_qk, dk_expanded_stride.data(),
                          layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
    generateMatrixStrides(b, h, s_q, s_kv, d_v, dv_expanded_stride.data(),
                          layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

    // dk_expanded and dv_expanded share one combined buffer; carve dv at the
    // same byte offset as the input v relative to k
    void* dkv_expanded_buf = planner.allocate(max_tokens_kv*h*(d_qk+d_v)*nvte_dtype_size(dtype));
    dk_expanded_ptr = dkv_expanded_buf;
    if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(dtype)*h*d_qk);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(dtype)*d_qk);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(dtype)*max_tokens_kv*h*d_qk);
    } else{
      NVTE_ERROR("NVTE_3HD NVTE_H3D should have h=hg.");
    }
  }

  void* devPtrAlibiSlope = nullptr;
  void* dbias_expanded_ptr = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    // ck requires an alibi slope array even if in standard (vanilla) mode
    devPtrAlibiSlope = planner.allocate(h*sizeof(float));
  }else if(bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS){
    if(bias_b!=b || bias_h!= h){
      // ck requires a buffer dbias_expanded of size BHSS if bias is not BHSS
      dbias_expanded_ptr = planner.allocate(b*h*s_q*s_kv*nvte_dtype_size(dtype));
    }
  }

  void* devPtrSoftmaxLSEWithoutPadding = nullptr;
  void* devPtrQWithoutPadding = nullptr;
  void* devPtrKWithoutPadding = nullptr;
  void* devPtrVWithoutPadding = nullptr;
  void* devPtrOWithoutPadding = nullptr;
  void* devPtrdOWithoutPadding = nullptr;
  void* devPtrdQWithoutPadding = nullptr;
  void* devPtrdKWithoutPadding = nullptr;
  void* devPtrdVWithoutPadding = nullptr;
  void* devPtrCuSeqlenPaddedQ = devPtrSeqOffsetsQ;
  void* devPtrCuSeqlenPaddedKV = devPtrSeqOffsetsKV;

  if((is_SBHD && is_padding) || bshd_to_thd || is_ragged){
    devPtrSoftmaxLSEWithoutPadding = planner.allocate(h*max_tokens_q*sizeof(float));
  }
  if(is_SBHD && is_padding){
    // qkv inputs without padding
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      void* qkv_buf = planner.allocate(q_storage_bytes + k_storage_bytes + v_storage_bytes);
      devPtrQWithoutPadding = qkv_buf;
      devPtrKWithoutPadding = byte_offset(qkv_buf, byte_diff(devPtrK, devPtrQ));
      devPtrVWithoutPadding = byte_offset(qkv_buf, byte_diff(devPtrV, devPtrQ));
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD || layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      devPtrQWithoutPadding = planner.allocate(q_storage_bytes);
      void* kv_buf = planner.allocate(k_storage_bytes + v_storage_bytes);
      devPtrKWithoutPadding = kv_buf;
      devPtrVWithoutPadding = byte_offset(kv_buf, byte_diff(devPtrV, devPtrK));
    }else{
      devPtrQWithoutPadding = planner.allocate(q_storage_bytes);
      devPtrKWithoutPadding = planner.allocate(k_storage_bytes);
      devPtrVWithoutPadding = planner.allocate(v_storage_bytes);
    }
    // o, do without padding
    devPtrOWithoutPadding = planner.allocate(o_storage_bytes);
    devPtrdOWithoutPadding = planner.allocate(o_storage_bytes);
    // dq, dk, dv outputs without padding (same layout-dependent carving as inputs)
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      void* dqkv_buf = planner.allocate(q_storage_bytes + k_storage_bytes + v_storage_bytes);
      devPtrdQWithoutPadding = dqkv_buf;
      devPtrdKWithoutPadding = byte_offset(dqkv_buf, byte_diff(devPtrK, devPtrQ));
      devPtrdVWithoutPadding = byte_offset(dqkv_buf, byte_diff(devPtrV, devPtrQ));
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD || layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      devPtrdQWithoutPadding = planner.allocate(q_storage_bytes);
      void* dkv_buf = planner.allocate(k_storage_bytes + v_storage_bytes);
      devPtrdKWithoutPadding = dkv_buf;
      devPtrdVWithoutPadding = byte_offset(dkv_buf, byte_diff(devPtrV, devPtrK));
    }else{
      devPtrdQWithoutPadding = planner.allocate(q_storage_bytes);
      devPtrdKWithoutPadding = planner.allocate(k_storage_bytes);
      devPtrdVWithoutPadding = planner.allocate(v_storage_bytes);
    }
  }else if(bshd_to_thd){
    devPtrCuSeqlenPaddedQ = planner.allocate((b+1)*sizeof(int32_t));
    devPtrCuSeqlenPaddedKV = planner.allocate((b+1)*sizeof(int32_t));
  }

  // Sizing-only path: report and exit before any kernel/memset.
  if(planner.is_sizing()){
    *workspace_size = planner.total();
    if(nvte_log_ck_config){
      if(is_SBHD && is_padding){
        std::cout<<std::endl<<"attn_bwd(ck) need padding/unpadding workaround"<<std::endl;
      }
      std::cout<<std::endl<<"attn_bwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }

  std::array<uint64_t, 4> q_stride;
  std::array<uint64_t, 4> k_stride;
  std::array<uint64_t, 4> v_stride;
  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
  generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  //q and o are having the same seqlen but o has the same head_dim with v
  //k and v are having the same seqlen but k has the same head_dim with q
  //x and dx are having the same shape and stride

  // Initialize user-output dq/dk/dv buffers (CK uses atomic ops on dq).
  //TODO: remove the memset afer ck fixes the atomic operations
  if((layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) || (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D)){
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQ, 0, q_storage_bytes + k_storage_bytes+ v_storage_bytes, stream));
  }else{
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQ, 0, q_storage_bytes, stream));
    if(is_padding){
      if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdK, 0, k_storage_bytes + v_storage_bytes, stream));
      }else{
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdK, 0, k_storage_bytes, stream));
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdV, 0, v_storage_bytes, stream));
      }
    }
  }

  // Initialize workspace buffers.
  if(devPtrAlibiSlope){
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
  }
  if(bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && devPtrdBias!=nullptr){
    if(dbias_expanded_ptr){
      // CK requires zeroing dbias_expanded
      NVTE_CHECK_CUDA(cudaMemsetAsync(dbias_expanded_ptr, 0, nvte_dtype_size(dtype)*b*h*s_q*s_kv, stream));
    }else{
      // dbias is BHSS; zero in-place
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdBias, 0, nvte_dtype_size(dtype)*bias_b*bias_h*s_q*s_kv, stream));
    }
  }
  if(is_SBHD && is_padding){
    // zero the workspace dqkv buffers (CK atomic ops requirement, mirroring user-output init above)
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQWithoutPadding, 0, q_storage_bytes + k_storage_bytes + v_storage_bytes, stream));
    }else{
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQWithoutPadding, 0, q_storage_bytes, stream));
    }
  }
  if(bshd_to_thd){
    generate_cu_seqlen_padded(s_q, s_kv, b, devPtrCuSeqlenPaddedQ, devPtrCuSeqlenPaddedKV, stream);
    if(nvte_log_ck_config){
      std::cout << "\nattn_bwd(ck): generating cu_seqlen_padded in BSHD+padding to THD+padding conversion.\n";
    }
  }

  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_bwd(ck): ";
    std::cout<<"layout: "<<layout<<", ";
    std::cout<<"max_tokens_q: "<<max_tokens_q<<", ";
    std::cout<<"max_tokens_kv: "<<max_tokens_kv<<", ";
    std::cout<<"is_ragged: "<<is_ragged<<", ";
    std::cout<<"is_padding: "<<is_padding<<", ";
    std::cout<<"bshd_to_thd: "<<bshd_to_thd<<", ";
    std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d_qk<<"), ";
    std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
    std::cout<<"k_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d_qk<<"), ";
    std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
    std::cout<<"v_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d_v<<"), ";
    std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";
    std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d_v<<"), ";
    std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    if(is_ragged){
      std::cout<<"M_shape: ("<<h<<", "<<max_tokens_q<<"), ";
      std::cout<<"M_stride: ("<<max_tokens_q<<", "<<1<<"), ";
    }else{
      std::cout<<"M_shape: ("<<b<<", "<<h<<", "<<s_q<<"), ";
      std::cout<<"M_stride: ("<<h*s_q<<", "<<s_q<<", "<<1<<"), ";
    }
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<", ";
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<", ";
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<"), ";
    std::cout<<"deterministic: "<<deterministic<<", ";
    std::cout<<"nvte_ck_uses_bwd_v3: "<<nvte_ck_uses_bwd_v3<<", ";
    std::cout<<"nvte_ck_is_v3_atomic_fp32: "<<nvte_ck_is_v3_atomic_fp32<<", ";
    std::cout<<"nvte_ck_how_v3_bf16_cvt: "<<nvte_ck_how_v3_bf16_cvt<<std::endl;
  }
  // Common fields filled here; mode-specific fields are overwritten below.
  ck_fused_attn::CkAttnBwdArgs ck_args;
  ck_args.dtype = nvte_to_ck_dtype(dtype);
  ck_args.b = b; ck_args.h = h; ck_args.hg = hg;
  ck_args.s_q = s_q; ck_args.s_kv = s_kv; ck_args.d_qk = d_qk; ck_args.d_v = d_v;
  ck_args.scaling_factor = scaling_factor;
  ck_args.dropout_probability = dropout_probability;
  ck_args.philox_seed_ptr = devPtrDropoutSeed;
  ck_args.philox_offset_ptr = devPtrDropoutOffset;
  ck_args.attn_mask_type = set_ck_mask(mask_type, window_size_left, window_size_right);
  ck_args.window_size_left = window_size_left;
  ck_args.window_size_right = window_size_right;
  ck_args.dk_expanded_ptr = dk_expanded_ptr;
  ck_args.dv_expanded_ptr = dv_expanded_ptr;
  ck_args.lse_workspace_ptr = lse_workspace;
  ck_args.deterministic = deterministic;
  ck_args.uses_bwd_v3 = nvte_ck_uses_bwd_v3;
  ck_args.is_v3_atomic_fp32 = nvte_ck_is_v3_atomic_fp32;
  ck_args.how_v3_bf16_cvt = nvte_ck_how_v3_bf16_cvt;

  if(is_SBHD && is_padding){
    // remove padding for q, k, v, o, do
    pad_remap<PadDirection::Remove>(dtype, b, h, s_q, d_qk, max_tokens_q, false, q_stride[0], q_stride[1], q_stride[2], devPtrQ, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrQWithoutPadding, stream);
    pad_remap<PadDirection::Remove>(dtype, b, hg, s_kv, d_qk, max_tokens_kv, false, k_stride[0], k_stride[1], k_stride[2], devPtrK, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrKWithoutPadding, stream);
    pad_remap<PadDirection::Remove>(dtype, b, hg, s_kv, d_v, max_tokens_kv, false, v_stride[0], v_stride[1], v_stride[2], devPtrV, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrVWithoutPadding, stream);
    // o and do should be of same shape as q
    pad_remap<PadDirection::Remove>(dtype, b, h, s_q, d_v, max_tokens_q, false, o_stride[0], o_stride[1], o_stride[2], devPtrO, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrOWithoutPadding, stream);
    pad_remap<PadDirection::Remove>(dtype, b, h, s_q, d_v, max_tokens_q, false, o_stride[0], o_stride[1], o_stride[2], devPtrdO, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrdOWithoutPadding, stream);
    // also remove the padding for softmax lse
    pad_remap_lse<PadDirection::Remove>(b, h, s_q, max_tokens_q, false, devPtrSoftmaxAux, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrSoftmaxLSEWithoutPadding, stream);
    ck_args.max_tokens_q = max_tokens_q; ck_args.max_tokens_kv = max_tokens_kv;
    ck_args.q_ptr = devPtrQWithoutPadding;
    ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[0];
    ck_args.k_ptr = devPtrKWithoutPadding;
    ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = std::min(k_stride[0], k_stride[2]);
    ck_args.v_ptr = devPtrVWithoutPadding;
    ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = std::min(v_stride[0], v_stride[2]);
    ck_args.cu_seqlen_q_ptr = devPtrCuSeqlensQ;
    ck_args.cu_seqlen_kv_ptr = devPtrCuSeqlensKV;
    ck_args.o_ptr = devPtrOWithoutPadding;
    ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[0];
    ck_args.lse_ptr = devPtrSoftmaxLSEWithoutPadding;
    // dO and O share the same stride in TE
    ck_args.do_ptr = devPtrdOWithoutPadding;
    ck_args.stride_h_do = o_stride[1]; ck_args.stride_s_do = o_stride[0];
    // dQ/dK/dV share strides with Q/K/V
    ck_args.dq_ptr = devPtrdQWithoutPadding;
    ck_args.stride_h_dq = q_stride[1]; ck_args.stride_s_dq = q_stride[0];
    ck_args.stride_h_dk_expanded = dk_expanded_stride[1];
    ck_args.stride_s_dk_expanded = std::min(dk_expanded_stride[0], dk_expanded_stride[2]);
    ck_args.stride_h_dv_expanded = dv_expanded_stride[1];
    ck_args.stride_s_dv_expanded = std::min(dv_expanded_stride[0], dv_expanded_stride[2]);
    ck_args.dk_ptr = devPtrdKWithoutPadding;
    ck_args.stride_h_dk = k_stride[1]; ck_args.stride_s_dk = std::min(k_stride[0], k_stride[2]);
    ck_args.dv_ptr = devPtrdVWithoutPadding;
    ck_args.stride_h_dv = v_stride[1]; ck_args.stride_s_dv = std::min(v_stride[0], v_stride[2]);
    NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_bwd(ck_args, stream));
    // add padding for dq, dk, dv
    // dq, dk, dv of same shape as q, k, v
    pad_remap<PadDirection::Add>(dtype, b, h, s_q, d_qk, max_tokens_q, is_ragged, q_stride[0], q_stride[1], q_stride[2], devPtrdQ, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrdQWithoutPadding, stream);
    pad_remap<PadDirection::Add>(dtype, b, hg, s_kv, d_qk, max_tokens_kv, is_ragged, k_stride[0], k_stride[1], k_stride[2], devPtrdK, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrdKWithoutPadding, stream);
    pad_remap<PadDirection::Add>(dtype, b, hg, s_kv, d_v, max_tokens_kv, is_ragged, v_stride[0], v_stride[1], v_stride[2], devPtrdV, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrdVWithoutPadding, stream);
  }else if(bshd_to_thd || is_ragged){
    pad_remap_lse<PadDirection::Remove>(b, h, s_q, max_tokens_q, is_ragged, devPtrSoftmaxAux, devPtrCuSeqlenPaddedQ, devPtrCuSeqlenPaddedQ, devPtrSoftmaxLSEWithoutPadding, stream);
    ck_args.max_tokens_q = max_tokens_q; ck_args.max_tokens_kv = max_tokens_kv;
    ck_args.q_ptr = devPtrQ;
    ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[2];
    ck_args.k_ptr = devPtrK;
    ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = k_stride[2];
    ck_args.v_ptr = devPtrV;
    ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = v_stride[2];
    ck_args.cu_seqlen_q_ptr = devPtrCuSeqlensQ;
    ck_args.cu_seqlen_kv_ptr = devPtrCuSeqlensKV;
    ck_args.cu_seqlen_q_padded_ptr = devPtrCuSeqlenPaddedQ;
    ck_args.cu_seqlen_kv_padded_ptr = devPtrCuSeqlenPaddedKV;
    ck_args.o_ptr = devPtrO;
    ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[2];
    ck_args.lse_ptr = devPtrSoftmaxLSEWithoutPadding;
    // dO and O share the same stride
    ck_args.do_ptr = devPtrdO;
    ck_args.stride_h_do = o_stride[1]; ck_args.stride_s_do = o_stride[2];
    ck_args.dq_ptr = devPtrdQ;
    ck_args.stride_h_dq = q_stride[1]; ck_args.stride_s_dq = q_stride[2];
    ck_args.stride_h_dk_expanded = dk_expanded_stride[1]; ck_args.stride_s_dk_expanded = dk_expanded_stride[2];
    ck_args.stride_h_dv_expanded = dv_expanded_stride[1]; ck_args.stride_s_dv_expanded = dv_expanded_stride[2];
    ck_args.dk_ptr = devPtrdK;
    ck_args.stride_h_dk = k_stride[1]; ck_args.stride_s_dk = k_stride[2];
    ck_args.dv_ptr = devPtrdV;
    ck_args.stride_h_dv = v_stride[1]; ck_args.stride_s_dv = v_stride[2];
    NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_bwd(ck_args, stream));
  }else{
    ck_args.bias_b = bias_b; ck_args.bias_h = bias_h;
    ck_args.q_ptr = devPtrQ;
    ck_args.stride_b_q = q_stride[0]; ck_args.stride_h_q = q_stride[1]; ck_args.stride_s_q = q_stride[2];
    ck_args.k_ptr = devPtrK;
    ck_args.stride_b_k = k_stride[0]; ck_args.stride_h_k = k_stride[1]; ck_args.stride_s_k = k_stride[2];
    ck_args.v_ptr = devPtrV;
    ck_args.stride_b_v = v_stride[0]; ck_args.stride_h_v = v_stride[1]; ck_args.stride_s_v = v_stride[2];
    ck_args.bias_ptr = devPtrBias;
    ck_args.alibi_slope_ptr = devPtrAlibiSlope;
    ck_args.attn_bias_type = nvte_to_ck_bias_type(bias_type);
    ck_args.o_ptr = devPtrO;
    ck_args.stride_b_o = o_stride[0]; ck_args.stride_h_o = o_stride[1]; ck_args.stride_s_o = o_stride[2];
    ck_args.lse_ptr = devPtrSoftmaxAux;
    // dO and O share the same stride
    ck_args.do_ptr = devPtrdO;
    ck_args.stride_b_do = o_stride[0]; ck_args.stride_h_do = o_stride[1]; ck_args.stride_s_do = o_stride[2];
    ck_args.dq_ptr = devPtrdQ;
    ck_args.stride_b_dq = q_stride[0]; ck_args.stride_h_dq = q_stride[1]; ck_args.stride_s_dq = q_stride[2];
    ck_args.stride_b_dk_expanded = dk_expanded_stride[0];
    ck_args.stride_h_dk_expanded = dk_expanded_stride[1];
    ck_args.stride_s_dk_expanded = dk_expanded_stride[2];
    ck_args.stride_b_dv_expanded = dv_expanded_stride[0];
    ck_args.stride_h_dv_expanded = dv_expanded_stride[1];
    ck_args.stride_s_dv_expanded = dv_expanded_stride[2];
    ck_args.dk_ptr = devPtrdK;
    ck_args.stride_b_dk = k_stride[0]; ck_args.stride_h_dk = k_stride[1]; ck_args.stride_s_dk = k_stride[2];
    ck_args.dv_ptr = devPtrdV;
    ck_args.stride_b_dv = v_stride[0]; ck_args.stride_h_dv = v_stride[1]; ck_args.stride_s_dv = v_stride[2];
    ck_args.dbias_expanded_ptr = dbias_expanded_ptr;
    ck_args.dbias_ptr = devPtrdBias;
    NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_bwd(ck_args, stream));
  }
}
#endif // USE_FUSED_ATTN_CK
}  // namespace fused_attn_rocm

using namespace transformer_engine::fused_attn_rocm;

void fused_attn_ck_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_Bias, 
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_CK
  const DType QKV_type = input_Q->data.dtype;

  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;
  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }

  void *devPtrCuSeqlensQ = input_cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = input_cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = input_cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = input_cu_seqlens_kv_padded->data.dptr;

  size_t max_tokens_q = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), static_cast<size_t>(1), std::multiplies<size_t>())/h_q/d_qk;
  size_t max_tokens_kv = std::accumulate((input_K->data).shape.begin(), (input_K->data).shape.end(), static_cast<size_t>(1), std::multiplies<size_t>())/h_kv/d_qk;

  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if(is_ragged){
        output_S->data.shape = {max_tokens_q, h_q, 1};
      }else{
        output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if(is_ragged){
        output_S->data.shape = {max_tokens_q, h_q, 1};
      }else{
        output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }
  size_t workspace_size = 0;

  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  
  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, bias_b, bias_h,
    max_tokens_q, max_tokens_kv,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    devPtrQ, devPtrK, devPtrV, devPtrBias, 
    devPtrS, devPtrO,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
    devPtrCuSeqlensQ, devPtrCuSeqlensKV,
    devPtrSeqOffsetsQ, devPtrSeqOffsetsKV,
    QKV_type,
    workspace->data.dptr,
    &workspace_size,
    stream);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
#endif // USE_FUSED_ATTN_CK
}

void fused_attn_ck_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  Tensor* output_dBias,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream){
#ifdef USE_FUSED_ATTN_CK
  const DType QKV_type = input_Q->data.dtype;

  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = input_O->data.dptr;
  void *devPtrdO = input_dO->data.dptr;
  void *devPtrBias = nullptr;
  void *devPtrdBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    devPtrdBias = output_dBias->data.dptr;
    bias_b = output_dBias->data.shape[0];
    bias_h = output_dBias->data.shape[1];
  }

  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdK = output_dK->data.dptr;
  void *devPtrdV = output_dV->data.dptr;
  void *devPtrSoftmaxStats = output_S->data.dptr;

  void *devPtrCuSeqlensQ = input_cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = input_cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = input_cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = input_cu_seqlens_kv_padded->data.dptr;

  size_t workspace_size = 0;

  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  
  // extract the max_tokens for padding/unpadding and softmax_lse buffer
  // b from cu_seqlen and max_seqlen are not the actual storage batch and seqlen for pad_between_seqs case
  size_t max_tokens_q = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), static_cast<size_t>(1), std::multiplies<size_t>())/h_q/d_qk;
  size_t max_tokens_kv = std::accumulate((input_K->data).shape.begin(), (input_K->data).shape.end(), static_cast<size_t>(1), std::multiplies<size_t>())/h_kv/d_qk;

  fused_attn_ck_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, bias_b, bias_h,
    max_tokens_q, max_tokens_kv,
    attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    deterministic,
    devPtrQ, devPtrK, devPtrV, 
    devPtrO, devPtrSoftmaxStats, devPtrBias,
    devPtrdQ, devPtrdK, devPtrdV, 
    devPtrdO, devPtrdBias,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
    devPtrCuSeqlensQ, devPtrCuSeqlensKV, 
    devPtrSeqOffsetsQ, devPtrSeqOffsetsKV,
    QKV_type,
    workspace->data.dptr,
    &workspace_size,
    stream);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
#endif // USE_FUSED_ATTN_CK
}

}  // namespace transformer_engine
