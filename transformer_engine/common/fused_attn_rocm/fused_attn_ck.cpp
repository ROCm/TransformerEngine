/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#ifdef USE_FUSED_ATTN_CK
#include <ck_fused_attn/ck_fused_attn.hpp>
#endif // USE_FUSED_ATTN_CK
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_ck.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

// check the fused attn config to see whether it's ck backend supported
// single filtering followed by joint filtering
bool is_ck_backend_supported(
  bool is_training,
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk,
  size_t head_dim_v,
  int64_t window_size_left,
  int64_t window_size_right) {

#ifdef USE_FUSED_ATTN_CK

  // debug info setting
  const bool nvte_log_ck_config = getenv<bool>("NVTE_LOG_CK_CONFIG");

  // single filters
  
  // filter based on num_heads and num_gqa_groups
  if(num_gqa_groups == 0 || num_attn_heads%num_gqa_groups != 0){
    if(nvte_log_ck_config){
      std::cout<<"Num of attention heads must be divisible by num of gqa groups"<<std::endl;
    }
    return false;
  }

  const bool is_fp8 = q_dtype == NVTEDType::kNVTEFloat8E4M3;
  const bool is_16bit =
      q_dtype == NVTEDType::kNVTEFloat16 || q_dtype == NVTEDType::kNVTEBFloat16;
  //  q and kv have to have same dtype: FP16/BF16  or FP8 E4M3 
  if(q_dtype != kv_dtype || (!is_16bit && !is_fp8)){
    if(nvte_log_ck_config){
      std::cout<<"q, k, v data type has to be fp16, bf16, or fp8 e4m3"<<std::endl;
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

  // joint filter based on sliding window and attn_mask
  bool is_causal = is_causal_mask(attn_mask_type);
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
  bool is_padding = is_padding_mask(attn_mask_type);
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
  if(is_fp8){
    // FP8 is wired to AITER v3 ASM only. CK tile FP8 is not enabled, so turning
    // ASM off must report unsupported and let the caller select NVTE_No_Backend.
    // A later PR that enables CK FP8 can fall back to that path here instead.
    if(getenv<int>("NVTE_CK_USES_FWD_V3", 1) == 0){
      if(nvte_log_ck_config){
        std::cout<<"AITER FP8 ASM fwd is disabled by NVTE_CK_USES_FWD_V3=0"<<std::endl;
      }
      return false;
    }
    const bool fp8_hdim_supported =
        (head_dim_qk == 128 && head_dim_v == 128) ||
        (head_dim_qk == 256 && head_dim_v == 256);
    const bool fp8_window_supported =
        is_causal ? (window_size_left == -1 && window_size_right == 0)
                  : (window_size_left == -1 && window_size_right == -1);
    if(cuda::sm_arch() != 95 || is_training || !fp8_hdim_supported ||
       !fp8_window_supported || dropout != 0.f ||
       bias_type != NVTE_Bias_Type::NVTE_NO_BIAS ||
       (is_padding && !is_ragged)){
      if(nvte_log_ck_config){
        std::cout<<"AITER FP8 ASM fwd requires gfx950 inference, hd128/hd256, "
                   "no bias/dropout/sliding-window, and THD for padding masks"<<std::endl;
      }
      return false;
    }
  }
  return true;
#else
  return false;
#endif // USE_FUSED_ATTN_CK
}

bool is_small_seq_supported_static(DType dtype,
                                   NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type,
                                   NVTE_Mask_Type mask_type,
                                   float dropout,
                                   size_t head_dim_qk,
                                   size_t head_dim_v,
                                   size_t num_attn_heads,
                                   size_t num_gqa_groups,
                                   NVTE_Softmax_Type softmax_type,
                                   int64_t window_size_left,
                                   int64_t window_size_right) {
  if(dropout != 0.0f) return false;
  if(bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) return false;
  if(softmax_type != NVTE_VANILLA_SOFTMAX) return false;
  if(dtype != DType::kBFloat16) return false;
  if(head_dim_qk != head_dim_v) return false;
  if(head_dim_qk != 128 && head_dim_qk != 256) return false;
  if(num_gqa_groups == 0 || num_attn_heads % num_gqa_groups != 0) return false;
  if(num_attn_heads != num_gqa_groups) return false;
  if(num_attn_heads != 16 && num_attn_heads != 32) return false;
  // the small-seq kernels take Q, K and V as separate tensors without strides
  if(nvte_get_qkv_layout_group(qkv_layout) != NVTE_QKV_Layout_Group::NVTE_HD_HD_HD) return false;
  // they apply no causal mask or sliding window
  if(mask_type != NVTE_Mask_Type::NVTE_NO_MASK && mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK) return false;
  if(window_size_left != -1 || window_size_right != -1) return false;
  // the BSHD ones take no sequence lengths
  if(nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_BSHD && is_padding_mask(mask_type)) return false;
  return true;
}

bool is_small_seq_supported_runtime(size_t runtime_max_seqlen_q,
                                    size_t runtime_max_seqlen_kv) {
  return runtime_max_seqlen_q > 0 && runtime_max_seqlen_q <= kSmallSeqMaxSeqlen &&
         runtime_max_seqlen_kv > 0 && runtime_max_seqlen_kv <= kSmallSeqMaxSeqlen;
}


#ifdef USE_FUSED_ATTN_CK
ck_fused_attn::DType nvte_to_ck_dtype(DType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == DType::aname) return ck_fused_attn::DType::dtname
  CAST_TYPE(kFloat16, kFloat16);
  CAST_TYPE(kBFloat16, kBFloat16);
  CAST_TYPE(kFloat8E4M3, kFloat8);
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
ck_fused_attn::MaskType set_ck_mask(NVTE_Mask_Type nvte_mask_type, int64_t nvte_window_size_left, int64_t nvte_window_size_right,
                                    bool bottom_right_diagonal){
  if (nvte_mask_type==NVTE_Mask_Type::NVTE_NO_MASK || nvte_mask_type==NVTE_Mask_Type::NVTE_PADDING_MASK){
    // window size in NVTE_NO_Mask can be (-1, -1) and (>=0, >=0)
    if(nvte_window_size_left==-1 && nvte_window_size_right==-1){
      // (-1, -1)
      return ck_fused_attn::MaskType::no_mask;
    }else{
      // (>=0, >=0): the sliding window is aligned to the top-left or bottom-right corner of each
      // q/kv segment pair, as requested; the two differ whenever seqlen_q != seqlen_kv.
      return bottom_right_diagonal ? ck_fused_attn::MaskType::mask_bottom_right
                                   : ck_fused_attn::MaskType::mask_top_left;
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

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX12__)
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
// padded layout (selected by is_ragged):
//   is_ragged=false: [b, h, s_q]
//   is_ragged=true : [max_tokens_q with padding, h]
// unpadded layout (selected by token_major):
//   token_major=false: [h, max_tokens_q], as the CK kernels write it
//   token_major=true : [max_tokens_q, h], as the small-seq kernels write it
// One wavefront in charge of one token index (h*sizeof(float) bytes)
// One workitem (thread) in one wavefront is charge of one element in THREADS_PER_WAVE_FRONT(64) segment trunk of h
template<bool is_ragged, PadDirection dir, bool token_major>
__global__ void pad_remap_lse_kernel(
  uint64_t b, uint64_t h, uint64_t s_q,
  uint64_t max_tokens_q,
  float* padded_lse_ptr,
  const int32_t* cu_seqlen_q_ptr, const int32_t* cu_seqlen_q_padded_ptr,
  float* unpadded_lse_ptr){

  int wavefront_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WAVEFRONT;
  int workitem_idx = threadIdx.x % THREADS_PER_WAVEFRONT;
  int num_wavefronts = (blockDim.x * gridDim.x) / THREADS_PER_WAVEFRONT;
  // the BSHD small-seq kernels take no sequence lengths and write all b*s_q rows
  constexpr bool dense = token_major && !is_ragged;
  int num_total_tokens = dense ? b*s_q : cu_seqlen_q_ptr[b];

  for(int token_id = wavefront_idx; token_id < num_total_tokens; token_id += num_wavefronts){
    int b_idx, s_idx;
    if constexpr(dense){
      b_idx = token_id / s_q;
      s_idx = token_id % s_q;
    }else{
      b_idx = binary_search(token_id, cu_seqlen_q_ptr, b+1);
      s_idx = token_id - cu_seqlen_q_ptr[b_idx];
    }

    for(int h_idx = workitem_idx; h_idx < h; h_idx += THREADS_PER_WAVEFRONT){
      uint64_t padded_idx;
      if constexpr(is_ragged){
        padded_idx = (cu_seqlen_q_padded_ptr[b_idx]+s_idx)*h + h_idx;
      }else{
        padded_idx = b_idx*h*s_q + h_idx*s_q + s_idx;
      }
      uint64_t unpadded_idx;
      if constexpr(token_major){
        unpadded_idx = token_id*h + h_idx;
      }else{
        unpadded_idx = h_idx * max_tokens_q + token_id;
      }
      if constexpr(dir == PadDirection::Add){
        padded_lse_ptr[padded_idx] = unpadded_lse_ptr[unpadded_idx];
      }else{
        unpadded_lse_ptr[unpadded_idx] = padded_lse_ptr[padded_idx];
      }
    }
  }
}

// kernel launcher for softmax_lse padded<->unpadded copy
template<PadDirection dir, bool token_major = false>
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
    pad_remap_lse_kernel<true, dir, token_major><<<grid, block, 0, stream>>>(
      b, h, s_q, max_tokens_q,
      static_cast<float*>(padded_lse_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_padded_ptr),
      static_cast<float*>(unpadded_lse_ptr));
  }else{
    pad_remap_lse_kernel<false, dir, token_major><<<grid, block, 0, stream>>>(
      b, h, s_q, max_tokens_q,
      static_cast<float*>(padded_lse_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_ptr),
      static_cast<const int32_t*>(cu_seqlen_q_padded_ptr),
      static_cast<float*>(unpadded_lse_ptr));
  }
}

// How the CK/AITER kernels consume the inputs.
//   Batch         : BSHD/SBHD without padding; batch-mode API on the original tensors.
//   Group         : THD, or BSHD+padding (cu_seqlen_padded generated from s_q/s_kv);
//                   group-mode API on the original tensors.
//   GroupUnpadded : SBHD+padding; inputs are copied into unpadded workspace buffers and the
//                   group-mode API runs on those, then outputs are copied back.
enum class CkMode { Batch, Group, GroupUnpadded };

// Problem description shared by fwd and bwd, filled once from the NVTE tensors.
struct CkAttnProblem {
  uint64_t b = 0, h = 0, hg = 0, s_q = 0, s_kv = 0, d_qk = 0, d_v = 0;
  uint64_t bias_b = 0, bias_h = 0;
  // b from cu_seqlen and max_seqlen are not the actual storage batch and seqlen for the
  // pad_between_seqs case, so the token counts are taken from the tensor storage instead.
  size_t max_tokens_q = 0, max_tokens_kv = 0;
  float scaling_factor = 1.f;
  float dropout_probability = 0.f;
  NVTE_QKV_Layout layout;
  NVTE_Bias_Type bias_type;
  NVTE_Mask_Type mask_type;
  NVTE_Softmax_Type softmax_type;
  int64_t window_size_left = -1, window_size_right = -1;
  bool bottom_right_diagonal = false;
  DType dtype;

  void* q = nullptr;
  void* k = nullptr;
  void* v = nullptr;
  void* bias = nullptr;
  void* softmax_offset = nullptr;
  void* dropout_seed = nullptr;
  void* dropout_offset = nullptr;
  void* cu_seqlens_q = nullptr;
  void* cu_seqlens_kv = nullptr;
  void* seq_offsets_q = nullptr;
  void* seq_offsets_kv = nullptr;

  // Derived from the fields above by init_ck_problem().
  NVTE_QKV_Layout_Group layout_group;
  bool is_ragged = false, is_SBHD = false, is_BSHD = false, is_padding = false;
  bool bshd_to_thd = false;
  bool has_sink = false;
  CkMode mode = CkMode::Batch;

  bool is_group_mode() const { return mode != CkMode::Batch; }
};

struct CkFwdProblem : CkAttnProblem {
  bool is_training = false;
  DType o_dtype;
  void* o = nullptr;
  void* softmax_aux = nullptr;
  void* descale_q = nullptr;
  void* descale_k = nullptr;
  void* descale_v = nullptr;
};

struct CkBwdProblem : CkAttnProblem {
  bool deterministic = false;
  void* o = nullptr;
  void* softmax_aux = nullptr;
  void* d_o = nullptr;
  void* dq = nullptr;
  void* dk = nullptr;
  void* dv = nullptr;
  void* dbias = nullptr;
  void* d_softmax_offset = nullptr;
};

// Fill the fields fwd and bwd take from the same arguments, then derive the layout flags.
void init_ck_problem(CkAttnProblem& p,
                     size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv,
                     size_t d_qk, size_t d_v, float attn_scale, float dropout,
                     NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                     NVTE_Softmax_Type softmax_type,
                     int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
                     const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
                     const Tensor* input_SoftmaxOffset,
                     const Tensor* input_cu_seqlens_q, const Tensor* input_cu_seqlens_kv,
                     const Tensor* input_cu_seqlens_q_padded, const Tensor* input_cu_seqlens_kv_padded,
                     const Tensor* rng_state){
  p.b = b; p.h = h_q; p.hg = h_kv;
  p.s_q = max_seqlen_q; p.s_kv = max_seqlen_kv;
  p.d_qk = d_qk; p.d_v = d_v;
  p.max_tokens_q  = input_Q->data.numel()/h_q/d_qk;
  p.max_tokens_kv = input_K->data.numel()/h_kv/d_qk;
  p.scaling_factor = attn_scale;
  p.dropout_probability = dropout;
  p.layout = qkv_layout;
  p.bias_type = bias_type;
  p.mask_type = attn_mask_type;
  p.softmax_type = softmax_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.dtype = input_Q->data.dtype;

  p.q = input_Q->data.dptr;
  p.k = input_K->data.dptr;
  p.v = input_V->data.dptr;
  p.cu_seqlens_q = input_cu_seqlens_q->data.dptr;
  p.cu_seqlens_kv = input_cu_seqlens_kv->data.dptr;
  p.seq_offsets_q = input_cu_seqlens_q_padded->data.dptr;
  p.seq_offsets_kv = input_cu_seqlens_kv_padded->data.dptr;
  p.dropout_seed = rng_state->data.dptr;
  p.dropout_offset = reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  p.has_sink = softmax_type != NVTE_VANILLA_SOFTMAX;
  if(p.has_sink){
    p.softmax_offset = input_SoftmaxOffset->data.dptr;
  }

  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  p.layout_group = nvte_get_qkv_layout_group(qkv_layout);
  p.is_ragged = qkv_format==NVTE_QKV_Format::NVTE_THD;
  p.is_SBHD = qkv_format==NVTE_QKV_Format::NVTE_SBHD || qkv_format==NVTE_QKV_Format::NVTE_SBHD_2BSHD;
  p.is_BSHD = qkv_format==NVTE_QKV_Format::NVTE_BSHD;
  p.is_padding = is_padding_mask(attn_mask_type);
  p.bshd_to_thd = p.is_BSHD && p.is_padding;
  if(p.is_SBHD && p.is_padding){
    p.mode = CkMode::GroupUnpadded;
  }else if(p.bshd_to_thd || p.is_ragged){
    p.mode = CkMode::Group;
  }else{
    p.mode = CkMode::Batch;
  }
}

// Whether the AITER v3 ASM kernels may run on this device (not on gfx1250-strict).
bool ck_v3_arch_supported(){
  return cuda::sm_arch() != 125 || !cuda::sm_arch_name().starts_with("gfx1250-strict");
}

// Strides of q, k, v and o as laid out by NVTE: [0]=b, [1]=h, [2]=s, [3]=d.
struct CkStrides {
  std::array<uint64_t, 4> q, k, v, o;
};

CkStrides make_ck_strides(const CkAttnProblem& p){
  CkStrides st;
  generateMatrixStrides(p.b, p.h, p.s_q, p.s_kv, p.d_qk, st.q.data(),
                        p.layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(p.b, p.hg, p.s_q, p.s_kv, p.d_qk, st.k.data(),
                        p.layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(p.b, p.hg, p.s_q, p.s_kv, p.d_v, st.v.data(),
                        p.layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
  generateMatrixStrides(p.b, p.h, p.s_q, p.s_kv, p.d_v, st.o.data(),
                        p.layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
  return st;
}

// Map a tensor's NVTE strides onto the (batch, head, seq) strides CK expects in the given mode.
// Group mode has no batch stride. Unpadded buffers are token-contiguous with the smaller of the
// batch/seq strides as the token stride, matching how pad_remap_kernel writes them.
void set_ck_strides(CkMode mode, const std::array<uint64_t, 4>& st,
                    uint64_t& stride_b, uint64_t& stride_h, uint64_t& stride_s){
  stride_b = mode == CkMode::Batch ? st[0] : 0;
  stride_h = st[1];
  stride_s = mode == CkMode::GroupUnpadded ? std::min(st[0], st[2]) : st[2];
}

// Carve workspace buffers shaped like q, k and v. Packed layouts get one combined buffer with
// k/v at the same byte offsets as in the input, so the input's packed strides stay valid.
std::array<void*, 3> carve_like(WorkspacePlanner& planner, NVTE_QKV_Layout_Group layout_group,
                                const void* q, const void* k, const void* v,
                                size_t q_bytes, size_t k_bytes, size_t v_bytes){
  std::array<void*, 3> out{};
  if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
    void* qkv_buf = planner.allocate(q_bytes + k_bytes + v_bytes);
    out[0] = qkv_buf;
    out[1] = byte_offset(qkv_buf, byte_diff(k, q));
    out[2] = byte_offset(qkv_buf, byte_diff(v, q));
  }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD || layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
    out[0] = planner.allocate(q_bytes);
    void* kv_buf = planner.allocate(k_bytes + v_bytes);
    out[1] = kv_buf;
    out[2] = byte_offset(kv_buf, byte_diff(v, k));
  }else{
    out[0] = planner.allocate(q_bytes);
    out[1] = planner.allocate(k_bytes);
    out[2] = planner.allocate(v_bytes);
  }
  return out;
}

// Static small-seq gate shared by fwd and bwd (NVTE_FUSED_ATTN_CK_SMALLSEQ=1, gfx942/gfx950).
// THD additionally reserves the workspace for the runtime max-seqlen probe.
bool plan_small_seq(const CkAttnProblem& p, WorkspacePlanner& planner, void** probe_workspace){
  *probe_workspace = nullptr;
  if(getenv<std::string>("NVTE_FUSED_ATTN_CK_SMALLSEQ") != "1"){
    return false;
  }
  if(cuda::sm_arch() != 94 && cuda::sm_arch() != 95){
    return false;
  }
  if(!is_small_seq_supported_static(p.dtype, p.layout, p.bias_type, p.mask_type, p.dropout_probability,
                                    p.d_qk, p.d_v, p.h, p.hg, p.softmax_type,
                                    p.window_size_left, p.window_size_right)){
    return false;
  }
  if(p.is_ragged){
    *probe_workspace = planner.allocate(ck_fused_attn::small_seq_thd_extra_workspace_bytes());
    return true;
  }
  return p.is_BSHD && p.s_q == p.s_kv && p.s_q >= 2 && p.s_q <= kSmallSeqMaxSeqlen;
}

// THD small-seq runtime probe: true when every sequence in the batch fits the small-seq kernels.
bool small_seq_thd_fits(const CkAttnProblem& p, void* probe_workspace,
                        void* cu_seqlen_padded_q, void* cu_seqlen_padded_kv,
                        const char* tag, bool log, cudaStream_t stream){
  const size_t runtime_max_seqlen_q = static_cast<size_t>(ck_fused_attn::get_runtime_max_seqlen(
      p.b, p.cu_seqlens_q, cu_seqlen_padded_q, probe_workspace, stream));
  const size_t runtime_max_seqlen_kv = static_cast<size_t>(ck_fused_attn::get_runtime_max_seqlen(
      p.b, p.cu_seqlens_kv, cu_seqlen_padded_kv, byte_offset(probe_workspace, sizeof(uint64_t)), stream));
  const bool fits = is_small_seq_supported_runtime(runtime_max_seqlen_q, runtime_max_seqlen_kv);
  if(log){
    std::cout << std::endl << tag << "(ck small-seq, THD): runtime_max_seqlen_q: "
              << runtime_max_seqlen_q << ", runtime_max_seqlen_kv: " << runtime_max_seqlen_kv
              << ", flow: " << (fits ? "ck-smallseq" : "regular ck/aiter") << std::endl;
  }
  return fits;
}

// Fill the CK fields shared by fwd and bwd that do not depend on workspace pointers.
void fill_common_ck_args(const CkAttnProblem& p, ck_fused_attn::CKAttnCommonArgs& a){
  a.dtype = nvte_to_ck_dtype(p.dtype);
  a.b = p.b; a.h = p.h; a.hg = p.hg;
  a.s_q = p.s_q; a.s_kv = p.s_kv; a.d_qk = p.d_qk; a.d_v = p.d_v;
  a.scaling_factor = p.scaling_factor;
  a.dropout_probability = p.dropout_probability;
  a.philox_seed_ptr = p.dropout_seed;
  a.philox_offset_ptr = p.dropout_offset;
  a.attn_mask_type = set_ck_mask(p.mask_type, p.window_size_left, p.window_size_right, p.bottom_right_diagonal);
  a.window_size_left = p.window_size_left;
  a.window_size_right = p.window_size_right;
  a.has_sink = p.has_sink;
  a.sink_ptr = p.has_sink ? p.softmax_offset : nullptr;
  if(p.is_group_mode()){
    // a non-null cu_seqlen_q_ptr is what selects CK group mode
    a.max_tokens_q = p.max_tokens_q;
    a.cu_seqlen_q_ptr = p.cu_seqlens_q;
    a.cu_seqlen_kv_ptr = p.cu_seqlens_kv;
  }else{
    a.bias_b = p.bias_b; a.bias_h = p.bias_h;
    a.attn_bias_type = nvte_to_ck_bias_type(p.bias_type);
  }
}

void fused_attn_ck_fwd_impl(const CkFwdProblem& p, void* workspace, size_t* workspace_size, cudaStream_t stream){

  const bool nvte_log_ck_config = getenv<bool>("NVTE_LOG_CK_CONFIG");

  // ASM v3 does not support sink; force CK tile path
  const bool nvte_ck_uses_fwd_v3 = !p.has_sink && getenv<int>("NVTE_CK_USES_FWD_V3", 1) && ck_v3_arch_supported();
  const int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);
  const bool nvte_ck_zero_out_pad = getenv<int>("NVTE_CK_ZERO_OUT_PAD", 1);

  // storage bytes of the q, k, v and o buffers, used to size the unpadded copies
  const size_t q_storage_bytes = p.max_tokens_q*p.h*p.d_qk*nvte_dtype_size(p.dtype);
  const size_t k_storage_bytes = p.max_tokens_kv*p.hg*p.d_qk*nvte_dtype_size(p.dtype);
  const size_t v_storage_bytes = p.max_tokens_kv*p.hg*p.d_v*nvte_dtype_size(p.dtype);
  const size_t o_storage_bytes = p.max_tokens_q*p.h*p.d_v*nvte_dtype_size(p.o_dtype);

  // Reserve workspace chunks. Same allocation sequence runs in sizing mode
  // (planner returns nullptr, accumulates total) and execution mode.
  WorkspacePlanner planner(workspace);

  void* small_seq_probe_workspace = nullptr;
  const bool small_seq = plan_small_seq(p, planner, &small_seq_probe_workspace);

  void* alibi_slope = nullptr;
  if(p.bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    // ck requires an alibi slope array even if in standard (vanilla) mode
    alibi_slope = planner.allocate(p.h*sizeof(float));
  }

  void* lse_unpadded = nullptr;
  if(p.is_group_mode() || small_seq){
    lse_unpadded = planner.allocate(p.h*p.max_tokens_q*sizeof(float));
  }

  std::array<void*, 3> qkv_unpadded{};
  void* o_unpadded = nullptr;
  void* cu_seqlen_padded_q = p.seq_offsets_q;
  void* cu_seqlen_padded_kv = p.seq_offsets_kv;
  if(p.mode == CkMode::GroupUnpadded){
    qkv_unpadded = carve_like(planner, p.layout_group, p.q, p.k, p.v,
                              q_storage_bytes, k_storage_bytes, v_storage_bytes);
    o_unpadded = planner.allocate(o_storage_bytes);
  }else if(p.bshd_to_thd){
    cu_seqlen_padded_q = planner.allocate((p.b+1)*sizeof(int32_t));
    cu_seqlen_padded_kv = planner.allocate((p.b+1)*sizeof(int32_t));
  }

  // Everything except the workspace-dependent pointers is set here, so the split-KV query
  // below sees the same configuration as the launch.
  ck_fused_attn::CKAttnFwdArgs ck_args;
  fill_common_ck_args(p, ck_args);
  ck_args.o_dtype = nvte_to_ck_dtype(p.o_dtype);
  ck_args.q_descale_ptr = p.descale_q;
  ck_args.k_descale_ptr = p.descale_k;
  ck_args.v_descale_ptr = p.descale_v;
  ck_args.is_training = p.is_training;
  ck_args.uses_fwd_v3 = nvte_ck_uses_fwd_v3;
  ck_args.how_v3_bf16_cvt = nvte_ck_how_v3_bf16_cvt;

  ck_args.num_splits =
      (ck_args.uses_fwd_v3 && !p.has_sink && p.dtype != DType::kFloat8E4M3)
          ? ck_fused_attn::ck_attn_fwd_num_splits(ck_args)
          : 0;
  if(ck_args.num_splits > 0){
    const size_t splitkv_workspace_bytes = ck_fused_attn::ck_attn_fwd_workspace_size(ck_args);
    if(splitkv_workspace_bytes > 0){
      ck_args.splitkv_workspace_ptr = planner.allocate(splitkv_workspace_bytes);
    }
  }

  // Sizing-only path: report and exit before any kernel/memset.
  if(planner.is_sizing()){
    *workspace_size = planner.total();
    if(nvte_log_ck_config){
      std::cout<<std::endl<<"attn_fwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }

  NVTE_CHECK(!p.has_sink || p.softmax_offset != nullptr,
           "softmax_offset is required for non-vanilla softmax");

  const CkStrides st = make_ck_strides(p);

  // Initialize workspace buffers.
  if(alibi_slope){
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(p.h/1024.);
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, p.h, static_cast<float*>(alibi_slope));
  }
  if(p.bshd_to_thd){
    generate_cu_seqlen_padded(p.s_q, p.s_kv, p.b, cu_seqlen_padded_q, cu_seqlen_padded_kv, stream);
    if(nvte_log_ck_config){
      std::cout << "\nattn_fwd(ck): generating cu_seqlen_padded in BSHD+padding to THD+padding conversion.\n";
    }
  }
  if(p.is_padding && nvte_ck_zero_out_pad){
    // reset the final results since padded places need to be 0
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.o, 0, o_storage_bytes, stream));
  }
  if(lse_unpadded){
    // ck only writes the lse of the tokens covered by cu_seqlens, but pad_remap_lse below
    // copies the whole padded range back into softmax_aux. Prefill the buffer with
    // 0xF0 (a large negative number, same sentinel the framework layers use) so that the
    // padded rows end up with a well defined -inf like lse instead of stale workspace data.
    // Consumers that combine lse across calls, e.g. the jax ring attention softmax
    // correction, produce NaNs when they read that garbage.
    NVTE_CHECK_CUDA(cudaMemsetAsync(lse_unpadded, 0xF0, p.h*p.max_tokens_q*sizeof(float), stream));
  }

  if(nvte_log_ck_config){
    std::cout<<std::endl<<"attn_fwd(ck): ";
    std::cout<<"layout: "<<p.layout<<", ";
    std::cout<<"max_tokens_q: "<<p.max_tokens_q<<", ";
    std::cout<<"max_tokens_kv: "<<p.max_tokens_kv<<", ";
    std::cout<<"is_ragged: "<<p.is_ragged<<", ";
    std::cout<<"is_padding: "<<p.is_padding<<", ";
    std::cout<<"is_SBHD: "<<p.is_SBHD<<", ";
    std::cout<<"is_BSHD: "<<p.is_BSHD<<", ";
    std::cout<<"bshd_to_thd: "<<p.bshd_to_thd<<", ";
    std::cout<<"q_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<", "<<p.d_qk<<"), ";
    std::cout<<"q_stride: ("<<st.q[0]<<", "<<st.q[1]<<", "<<st.q[2]<<", "<<st.q[3]<<"), ";
    std::cout<<"k_shape: ("<<p.b<<", "<<p.hg<<", "<<p.s_kv<<", "<<p.d_qk<<"), ";
    std::cout<<"k_stride: ("<<st.k[0]<<", "<<st.k[1]<<", "<<st.k[2]<<", "<<st.k[3]<<"), ";
    std::cout<<"v_shape: ("<<p.b<<", "<<p.hg<<", "<<p.s_kv<<", "<<p.d_v<<"), ";
    std::cout<<"v_stride: ("<<st.v[0]<<", "<<st.v[1]<<", "<<st.v[2]<<", "<<st.v[3]<<"), ";
    std::cout<<"o_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<", "<<p.d_v<<"), ";
    std::cout<<"o_stride: ("<<st.o[0]<<", "<<st.o[1]<<", "<<st.o[2]<<", "<<st.o[3]<<"), ";
    std::cout<<"scaling_factor: "<<p.scaling_factor<<", ";
    if(p.is_ragged){
      std::cout<<"M_shape: ("<<p.h<<", "<<p.max_tokens_q<<"), ";
      std::cout<<"M_stride: ("<<p.max_tokens_q<<", "<<1<<"), ";
    }else{
      std::cout<<"M_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<"), ";
      std::cout<<"M_stride: ("<<p.h*p.s_q<<", "<<p.s_q<<", "<<1<<"), ";
    }
    std::cout<<"is_training: "<<p.is_training<<", ";
    std::cout<<"dropout_p: "<<p.dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<p.dropout_seed<<", philox_offset_ptr: "<<p.dropout_offset<<", ";
    std::cout<<"bias_type: "<<p.bias_type<<", ";
    std::cout<<"(bias_b, bias_h): ("<<p.bias_b<<", "<<p.bias_h<<"), ";
    std::cout<<"mask_type: "<<p.mask_type<<", ";
    std::cout<<"window_size: ("<<p.window_size_left<<", "<<p.window_size_right<<")"<<", ";
    std::cout<<"nvte_ck_uses_fwd_v3: "<<nvte_ck_uses_fwd_v3<<", ";
    std::cout<<"num_splits: "<<ck_args.num_splits<<", ";
    std::cout<<"has_sink: "<<p.has_sink<<", ";
    std::cout<<"sink_ptr: "<<ck_args.sink_ptr<<std::endl;
  }

  // ---------------------------------------------------------------------------
  // CK small-seq forward (NVTE_FUSED_ATTN_CK_SMALLSEQ), see plan_small_seq.
  //
  // BSHD: uniform s_q in [2, 17], ck_attn_smallseq_fwd_bshd on Q/K/V/O;
  //   LSE [tokens,h] -> pad_remap_lse -> softmax_aux [b,h,s].
  //
  // THD: probe runtime max seqlen via cu_seqlens; if <= 17,
  //   ck_attn_smallseq_fwd_thd with cu_seqlens; LSE [tokens,h] -> pad_remap_lse -> softmax_aux [tokens,h].
  // ---------------------------------------------------------------------------
  if(small_seq){
    if(p.is_BSHD){
      if(nvte_log_ck_config){
        std::cout << std::endl << "attn_fwd(ck small-seq, BSHD self-attn)" << std::endl;
      }
      ck_fused_attn::ck_attn_smallseq_fwd_bshd(
          p.b, p.h, p.s_q, p.s_kv, p.d_qk, p.scaling_factor, p.q, p.k, p.v, p.o,
          lse_unpadded, nvte_to_ck_dtype(p.dtype), stream);
      pad_remap_lse<PadDirection::Add, true>(p.b, p.h, p.s_q, p.max_tokens_q, false, p.softmax_aux,
                                             p.cu_seqlens_q, p.seq_offsets_q,
                                             lse_unpadded, stream);
      return;
    }
    if(small_seq_thd_fits(p, small_seq_probe_workspace, cu_seqlen_padded_q, cu_seqlen_padded_kv,
                          "attn_fwd", nvte_log_ck_config, stream)){
      ck_fused_attn::ck_attn_smallseq_fwd_thd(
          p.b, p.h, p.d_qk, p.max_tokens_q, p.max_tokens_kv, p.scaling_factor, p.q, p.k, p.v,
          p.o, lse_unpadded, p.cu_seqlens_q, cu_seqlen_padded_q,
          p.cu_seqlens_kv, cu_seqlen_padded_kv, nvte_to_ck_dtype(p.dtype), stream);
      pad_remap_lse<PadDirection::Add, true>(p.b, p.h, p.s_q, p.max_tokens_q, true, p.softmax_aux,
                                             cu_seqlen_padded_q, cu_seqlen_padded_q,
                                             lse_unpadded, stream);
      return;
    }
  }

  const bool unpadded = p.mode == CkMode::GroupUnpadded;
  if(unpadded){
    // remove padding for q, k, v
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.h, p.s_q, p.d_qk, p.max_tokens_q, false, st.q[0], st.q[1], st.q[2], p.q, p.cu_seqlens_q, cu_seqlen_padded_q, qkv_unpadded[0], stream);
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.hg, p.s_kv, p.d_qk, p.max_tokens_kv, false, st.k[0], st.k[1], st.k[2], p.k, p.cu_seqlens_kv, cu_seqlen_padded_kv, qkv_unpadded[1], stream);
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.hg, p.s_kv, p.d_v, p.max_tokens_kv, false, st.v[0], st.v[1], st.v[2], p.v, p.cu_seqlens_kv, cu_seqlen_padded_kv, qkv_unpadded[2], stream);
  }

  ck_args.q_ptr = unpadded ? qkv_unpadded[0] : p.q;
  set_ck_strides(p.mode, st.q, ck_args.stride_b_q, ck_args.stride_h_q, ck_args.stride_s_q);
  ck_args.k_ptr = unpadded ? qkv_unpadded[1] : p.k;
  set_ck_strides(p.mode, st.k, ck_args.stride_b_k, ck_args.stride_h_k, ck_args.stride_s_k);
  ck_args.v_ptr = unpadded ? qkv_unpadded[2] : p.v;
  set_ck_strides(p.mode, st.v, ck_args.stride_b_v, ck_args.stride_h_v, ck_args.stride_s_v);
  ck_args.o_ptr = unpadded ? o_unpadded : p.o;
  set_ck_strides(p.mode, st.o, ck_args.stride_b_o, ck_args.stride_h_o, ck_args.stride_s_o);
  ck_args.lse_ptr = p.is_group_mode() ? lse_unpadded : p.softmax_aux;
  if(p.mode == CkMode::Group){
    ck_args.cu_seqlen_q_padded_ptr = cu_seqlen_padded_q;
    ck_args.cu_seqlen_kv_padded_ptr = cu_seqlen_padded_kv;
  }else if(p.mode == CkMode::Batch){
    ck_args.bias_ptr = p.bias;
    ck_args.alibi_slope_ptr = alibi_slope;
  }
  NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_fwd(ck_args, stream));

  if(unpadded){
    // add padding for o and softmax_lse
    pad_remap<PadDirection::Add>(p.dtype, p.b, p.h, p.s_q, p.d_v, p.max_tokens_q, false, st.o[0], st.o[1], st.o[2], p.o, p.cu_seqlens_q, cu_seqlen_padded_q, o_unpadded, stream);
    pad_remap_lse<PadDirection::Add>(p.b, p.h, p.s_q, p.max_tokens_q, false, p.softmax_aux, p.cu_seqlens_q, cu_seqlen_padded_q, lse_unpadded, stream);
  }else if(p.mode == CkMode::Group){
    // aiter asm output softmax_lse with padding
    pad_remap_lse<PadDirection::Add>(p.b, p.h, p.s_q, p.max_tokens_q, p.is_ragged, p.softmax_aux, cu_seqlen_padded_q, cu_seqlen_padded_q, lse_unpadded, stream);
  }
}

void fused_attn_ck_bwd_impl(const CkBwdProblem& p, void* workspace, size_t* workspace_size, cudaStream_t stream){

  const bool nvte_log_ck_config = getenv<bool>("NVTE_LOG_CK_CONFIG");
  // bwd v3 is optional by enabling the following envs
  // default values follows the ck example setting
  // ASM v3 bwd does not compute the sink gradient; force the CK tile path so
  // has_sink users still get a correct (if slower) d_sink, mirroring the fwd
  // ASM v3 sink guard.
  const bool nvte_ck_uses_bwd_v3 = !p.has_sink && getenv<int>("NVTE_CK_USES_BWD_V3", 1) && ck_v3_arch_supported();
  const bool nvte_ck_is_v3_atomic_fp32 = getenv<int>("NVTE_CK_IS_V3_ATOMIC_FP32", 1);
  const int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);

  const bool is_mqa_gqa = (p.h > p.hg);

  // storage bytes of the q, k, v and o buffers, used to size the unpadded copies
  const size_t q_storage_bytes = p.max_tokens_q*p.h*p.d_qk*nvte_dtype_size(p.dtype);
  const size_t k_storage_bytes = p.max_tokens_kv*p.hg*p.d_qk*nvte_dtype_size(p.dtype);
  const size_t v_storage_bytes = p.max_tokens_kv*p.hg*p.d_v*nvte_dtype_size(p.dtype);
  const size_t o_storage_bytes = p.max_tokens_q*p.h*p.d_v*nvte_dtype_size(p.dtype);

  // Reserve workspace chunks. Same allocation sequence runs in sizing mode
  // (planner returns nullptr, accumulates total) and execution mode.
  WorkspacePlanner planner(workspace);

  void* small_seq_probe_workspace = nullptr;
  const bool small_seq = plan_small_seq(p, planner, &small_seq_probe_workspace);

  // First h*max_tokens_q*sizeof(float) is the lse-d buffer (passed as softmax_lsed)
  void* lse_workspace = planner.allocate(p.h*p.max_tokens_q*sizeof(float));

  // Everything except the workspace-dependent pointers is set here. The same object sizes
  // AITER's internal bwd workspace (launcher metadata + dq_acc accumulator) and drives the
  // launch, so the reservation cannot drift from what the dispatch uses.
  ck_fused_attn::CkAttnBwdArgs ck_args;
  fill_common_ck_args(p, ck_args);
  if(p.is_group_mode()){
    ck_args.max_tokens_kv = p.max_tokens_kv;
  }
  ck_args.deterministic = p.deterministic;
  ck_args.uses_bwd_v3 = nvte_ck_uses_bwd_v3;
  ck_args.is_v3_atomic_fp32 = nvte_ck_is_v3_atomic_fp32;
  ck_args.how_v3_bf16_cvt = nvte_ck_how_v3_bf16_cvt;
  ck_args.d_sink_ptr = p.has_sink ? p.d_softmax_offset : nullptr;
  ck_args.dbias_ptr = p.dbias;

  // AOT scratch for AITER's internal bwd workspace, carved by ck_attn_bwd's
  // workspace_alloc callback instead of allocated per call.
  ck_args.aiter_workspace_bytes = ck_fused_attn::ck_attn_bwd_workspace_size(ck_args);
  ck_args.aiter_workspace_ptr = planner.allocate(ck_args.aiter_workspace_bytes);

  void* dk_expanded_ptr = nullptr;
  void* dv_expanded_ptr = nullptr;
  std::array<uint64_t, 4> dk_expanded_stride{};
  std::array<uint64_t, 4> dv_expanded_stride{};
  if(is_mqa_gqa){
    // generate kv expanded stride as if h_kv = h_q
    generateMatrixStrides(p.b, p.h, p.s_q, p.s_kv, p.d_qk, dk_expanded_stride.data(),
                          p.layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
    generateMatrixStrides(p.b, p.h, p.s_q, p.s_kv, p.d_v, dv_expanded_stride.data(),
                          p.layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

    // dk_expanded and dv_expanded share one combined buffer; carve dv at the
    // same byte offset as the input v relative to k
    void* dkv_expanded_buf = planner.allocate(p.max_tokens_kv*p.h*(p.d_qk+p.d_v)*nvte_dtype_size(p.dtype));
    dk_expanded_ptr = dkv_expanded_buf;
    if(p.layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(p.dtype)*p.h*p.d_qk);
    }else if(p.layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(p.dtype)*p.d_qk);
    }else if(p.layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD){
      dv_expanded_ptr = byte_offset(dkv_expanded_buf, nvte_dtype_size(p.dtype)*p.max_tokens_kv*p.h*p.d_qk);
    }else{
      NVTE_ERROR("NVTE_3HD NVTE_H3D should have h=hg.");
    }
  }

  void* alibi_slope = nullptr;
  void* dbias_expanded_ptr = nullptr;
  if(p.bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    // ck requires an alibi slope array even if in standard (vanilla) mode
    alibi_slope = planner.allocate(p.h*sizeof(float));
  }else if(p.bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS){
    if(p.bias_b!=p.b || p.bias_h!= p.h){
      // ck requires a buffer dbias_expanded of size BHSS if bias is not BHSS
      dbias_expanded_ptr = planner.allocate(p.b*p.h*p.s_q*p.s_kv*nvte_dtype_size(p.dtype));
    }
  }

  void* lse_unpadded = nullptr;
  if(p.is_group_mode() || small_seq){
    lse_unpadded = planner.allocate(p.h*p.max_tokens_q*sizeof(float));
  }

  std::array<void*, 3> qkv_unpadded{};
  std::array<void*, 3> dqkv_unpadded{};
  void* o_unpadded = nullptr;
  void* do_unpadded = nullptr;
  void* cu_seqlen_padded_q = p.seq_offsets_q;
  void* cu_seqlen_padded_kv = p.seq_offsets_kv;
  if(p.mode == CkMode::GroupUnpadded){
    qkv_unpadded = carve_like(planner, p.layout_group, p.q, p.k, p.v,
                              q_storage_bytes, k_storage_bytes, v_storage_bytes);
    o_unpadded = planner.allocate(o_storage_bytes);
    do_unpadded = planner.allocate(o_storage_bytes);
    // dq, dk, dv outputs without padding (same layout-dependent carving as inputs)
    dqkv_unpadded = carve_like(planner, p.layout_group, p.q, p.k, p.v,
                               q_storage_bytes, k_storage_bytes, v_storage_bytes);
  }else if(p.bshd_to_thd){
    cu_seqlen_padded_q = planner.allocate((p.b+1)*sizeof(int32_t));
    cu_seqlen_padded_kv = planner.allocate((p.b+1)*sizeof(int32_t));
  }

  // Sizing-only path: report and exit before any kernel/memset.
  if(planner.is_sizing()){
    *workspace_size = planner.total();
    if(nvte_log_ck_config){
      if(p.mode == CkMode::GroupUnpadded){
        std::cout<<std::endl<<"attn_bwd(ck) need padding/unpadding workaround"<<std::endl;
      }
      std::cout<<std::endl<<"attn_bwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }

  //q and o are having the same seqlen but o has the same head_dim with v
  //k and v are having the same seqlen but k has the same head_dim with q
  //x and dx are having the same shape and stride
  const CkStrides st = make_ck_strides(p);

  // Initialize user-output dq/dk/dv buffers (CK uses atomic ops on dq, and the small-seq backward
  // skips sequences without keys, leaving their dQ rows to this memset).
  //TODO: remove the memset after ck fixes the atomic operations, but keep zeroing dq for the small-seq path
  if((p.layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) || (p.layout_group == NVTE_QKV_Layout_Group::NVTE_H3D)){
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.dq, 0, q_storage_bytes + k_storage_bytes+ v_storage_bytes, stream));
  }else{
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.dq, 0, q_storage_bytes, stream));
    if(p.is_padding){
      if(p.layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||p.layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
        NVTE_CHECK_CUDA(cudaMemsetAsync(p.dk, 0, k_storage_bytes + v_storage_bytes, stream));
      }else{
        NVTE_CHECK_CUDA(cudaMemsetAsync(p.dk, 0, k_storage_bytes, stream));
        NVTE_CHECK_CUDA(cudaMemsetAsync(p.dv, 0, v_storage_bytes, stream));
      }
    }
  }

  // Initialize workspace buffers.
  if(p.has_sink){
    // CK accumulates the sink gradient via atomicAdd (one scalar per head); requires zeroing
    NVTE_CHECK_CUDA(cudaMemsetAsync(p.d_softmax_offset, 0, sizeof(float)*p.h, stream));
  }
  // dk_expanded/dv_expanded scratch is written only for valid KV rows by the GQA
  // main kernel, but the post-dispatch reduction reads every row; pre-zero so
  // masked/padded rows don't feed uninitialized workspace into dk/dv (-> NaN).
  if(dk_expanded_ptr){
    NVTE_CHECK_CUDA(cudaMemsetAsync(dk_expanded_ptr, 0,
        p.max_tokens_kv*p.h*(p.d_qk+p.d_v)*nvte_dtype_size(p.dtype), stream));
  }
  if(alibi_slope){
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(p.h/1024.);
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, p.h, static_cast<float*>(alibi_slope));
  }
  if(p.bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && p.dbias!=nullptr){
    if(dbias_expanded_ptr){
      // CK requires zeroing dbias_expanded
      NVTE_CHECK_CUDA(cudaMemsetAsync(dbias_expanded_ptr, 0, nvte_dtype_size(p.dtype)*p.b*p.h*p.s_q*p.s_kv, stream));
    }else{
      // dbias is BHSS; zero in-place
      NVTE_CHECK_CUDA(cudaMemsetAsync(p.dbias, 0, nvte_dtype_size(p.dtype)*p.bias_b*p.bias_h*p.s_q*p.s_kv, stream));
    }
  }
  if(p.mode == CkMode::GroupUnpadded){
    // zero the workspace dqkv buffers (CK atomic ops requirement, mirroring user-output init above)
    if(p.layout_group==NVTE_QKV_Layout_Group::NVTE_3HD || p.layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      NVTE_CHECK_CUDA(cudaMemsetAsync(dqkv_unpadded[0], 0, q_storage_bytes + k_storage_bytes + v_storage_bytes, stream));
    }else{
      NVTE_CHECK_CUDA(cudaMemsetAsync(dqkv_unpadded[0], 0, q_storage_bytes, stream));
    }
  }
  if(p.bshd_to_thd){
    generate_cu_seqlen_padded(p.s_q, p.s_kv, p.b, cu_seqlen_padded_q, cu_seqlen_padded_kv, stream);
    if(nvte_log_ck_config){
      std::cout << "\nattn_bwd(ck): generating cu_seqlen_padded in BSHD+padding to THD+padding conversion.\n";
    }
  }

  if(nvte_log_ck_config){
    std::cout<<std::endl<<"attn_bwd(ck): ";
    std::cout<<"layout: "<<p.layout<<", ";
    std::cout<<"max_tokens_q: "<<p.max_tokens_q<<", ";
    std::cout<<"max_tokens_kv: "<<p.max_tokens_kv<<", ";
    std::cout<<"is_ragged: "<<p.is_ragged<<", ";
    std::cout<<"is_padding: "<<p.is_padding<<", ";
    std::cout<<"bshd_to_thd: "<<p.bshd_to_thd<<", ";
    std::cout<<"q_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<", "<<p.d_qk<<"), ";
    std::cout<<"q_stride: ("<<st.q[0]<<", "<<st.q[1]<<", "<<st.q[2]<<", "<<st.q[3]<<"), ";
    std::cout<<"k_shape: ("<<p.b<<", "<<p.hg<<", "<<p.s_kv<<", "<<p.d_qk<<"), ";
    std::cout<<"k_stride: ("<<st.k[0]<<", "<<st.k[1]<<", "<<st.k[2]<<", "<<st.k[3]<<"), ";
    std::cout<<"v_shape: ("<<p.b<<", "<<p.hg<<", "<<p.s_kv<<", "<<p.d_v<<"), ";
    std::cout<<"v_stride: ("<<st.v[0]<<", "<<st.v[1]<<", "<<st.v[2]<<", "<<st.v[3]<<"), ";
    std::cout<<"o_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<", "<<p.d_v<<"), ";
    std::cout<<"o_stride: ("<<st.o[0]<<", "<<st.o[1]<<", "<<st.o[2]<<", "<<st.o[3]<<"), ";
    std::cout<<"scaling_factor: "<<p.scaling_factor<<", ";
    if(p.is_ragged){
      std::cout<<"M_shape: ("<<p.h<<", "<<p.max_tokens_q<<"), ";
      std::cout<<"M_stride: ("<<p.max_tokens_q<<", "<<1<<"), ";
    }else{
      std::cout<<"M_shape: ("<<p.b<<", "<<p.h<<", "<<p.s_q<<"), ";
      std::cout<<"M_stride: ("<<p.h*p.s_q<<", "<<p.s_q<<", "<<1<<"), ";
    }
    std::cout<<"dropout_p: "<<p.dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<p.dropout_seed<<", philox_offset_ptr: "<<p.dropout_offset<<", ";
    std::cout<<"bias_type: "<<p.bias_type<<", ";
    std::cout<<"(bias_b, bias_h): ("<<p.bias_b<<", "<<p.bias_h<<"), ";
    std::cout<<"mask_type: "<<p.mask_type<<", ";
    std::cout<<"window_size: ("<<p.window_size_left<<", "<<p.window_size_right<<"), ";
    std::cout<<"deterministic: "<<p.deterministic<<", ";
    std::cout<<"nvte_ck_uses_bwd_v3: "<<nvte_ck_uses_bwd_v3<<", ";
    std::cout<<"nvte_ck_is_v3_atomic_fp32: "<<nvte_ck_is_v3_atomic_fp32<<", ";
    std::cout<<"nvte_ck_how_v3_bf16_cvt: "<<nvte_ck_how_v3_bf16_cvt<<", ";
    std::cout<<"has_sink: "<<p.has_sink<<", ";
    std::cout<<"sink_ptr: "<<ck_args.sink_ptr<<", ";
    std::cout<<"d_sink_ptr: "<<ck_args.d_sink_ptr<<std::endl;
  }

  // ---------------------------------------------------------------------------
  // CK small-seq backward (mirrors forward).
  // ---------------------------------------------------------------------------
  if(small_seq){
    if(p.is_BSHD){
      if(nvte_log_ck_config){
        std::cout << std::endl << "attn_bwd(ck small-seq, BSHD self-attn)" << std::endl;
      }
      pad_remap_lse<PadDirection::Remove, true>(p.b, p.h, p.s_q, p.max_tokens_q, false, p.softmax_aux,
                                                p.cu_seqlens_q, p.seq_offsets_q,
                                                lse_unpadded, stream);
      ck_fused_attn::ck_attn_smallseq_bwd_bshd(
          p.b, p.h, p.s_q, p.s_kv, p.d_qk, p.scaling_factor, p.q, p.k, p.v, p.d_o,
          lse_unpadded, p.dq, p.dk, p.dv, nvte_to_ck_dtype(p.dtype), stream);
      return;
    }
    if(small_seq_thd_fits(p, small_seq_probe_workspace, cu_seqlen_padded_q, cu_seqlen_padded_kv,
                          "attn_bwd", nvte_log_ck_config, stream)){
      pad_remap_lse<PadDirection::Remove, true>(p.b, p.h, p.s_q, p.max_tokens_q, true, p.softmax_aux,
                                                cu_seqlen_padded_q, cu_seqlen_padded_q,
                                                lse_unpadded, stream);
      ck_fused_attn::ck_attn_smallseq_bwd_thd(
          p.b, p.h, p.d_qk, p.max_tokens_q, p.max_tokens_kv, p.scaling_factor, p.q, p.k, p.v,
          p.d_o, lse_unpadded, p.dq, p.dk, p.dv,
          p.cu_seqlens_q, cu_seqlen_padded_q, p.cu_seqlens_kv, cu_seqlen_padded_kv,
          nvte_to_ck_dtype(p.dtype), stream);
      return;
    }
  }

  const bool unpadded = p.mode == CkMode::GroupUnpadded;
  if(unpadded){
    // remove padding for q, k, v, o, do
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.h, p.s_q, p.d_qk, p.max_tokens_q, false, st.q[0], st.q[1], st.q[2], p.q, p.cu_seqlens_q, cu_seqlen_padded_q, qkv_unpadded[0], stream);
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.hg, p.s_kv, p.d_qk, p.max_tokens_kv, false, st.k[0], st.k[1], st.k[2], p.k, p.cu_seqlens_kv, cu_seqlen_padded_kv, qkv_unpadded[1], stream);
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.hg, p.s_kv, p.d_v, p.max_tokens_kv, false, st.v[0], st.v[1], st.v[2], p.v, p.cu_seqlens_kv, cu_seqlen_padded_kv, qkv_unpadded[2], stream);
    // o and do should be of same shape as q
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.h, p.s_q, p.d_v, p.max_tokens_q, false, st.o[0], st.o[1], st.o[2], p.o, p.cu_seqlens_q, cu_seqlen_padded_q, o_unpadded, stream);
    pad_remap<PadDirection::Remove>(p.dtype, p.b, p.h, p.s_q, p.d_v, p.max_tokens_q, false, st.o[0], st.o[1], st.o[2], p.d_o, p.cu_seqlens_q, cu_seqlen_padded_q, do_unpadded, stream);
    // also remove the padding for softmax lse
    pad_remap_lse<PadDirection::Remove>(p.b, p.h, p.s_q, p.max_tokens_q, false, p.softmax_aux, p.cu_seqlens_q, cu_seqlen_padded_q, lse_unpadded, stream);
  }else if(p.mode == CkMode::Group){
    pad_remap_lse<PadDirection::Remove>(p.b, p.h, p.s_q, p.max_tokens_q, p.is_ragged, p.softmax_aux, cu_seqlen_padded_q, cu_seqlen_padded_q, lse_unpadded, stream);
  }

  ck_args.q_ptr = unpadded ? qkv_unpadded[0] : p.q;
  set_ck_strides(p.mode, st.q, ck_args.stride_b_q, ck_args.stride_h_q, ck_args.stride_s_q);
  ck_args.k_ptr = unpadded ? qkv_unpadded[1] : p.k;
  set_ck_strides(p.mode, st.k, ck_args.stride_b_k, ck_args.stride_h_k, ck_args.stride_s_k);
  ck_args.v_ptr = unpadded ? qkv_unpadded[2] : p.v;
  set_ck_strides(p.mode, st.v, ck_args.stride_b_v, ck_args.stride_h_v, ck_args.stride_s_v);
  ck_args.o_ptr = unpadded ? o_unpadded : p.o;
  set_ck_strides(p.mode, st.o, ck_args.stride_b_o, ck_args.stride_h_o, ck_args.stride_s_o);
  // dO and O share the same stride in TE
  ck_args.do_ptr = unpadded ? do_unpadded : p.d_o;
  set_ck_strides(p.mode, st.o, ck_args.stride_b_do, ck_args.stride_h_do, ck_args.stride_s_do);
  // dQ/dK/dV share strides with Q/K/V
  ck_args.dq_ptr = unpadded ? dqkv_unpadded[0] : p.dq;
  set_ck_strides(p.mode, st.q, ck_args.stride_b_dq, ck_args.stride_h_dq, ck_args.stride_s_dq);
  ck_args.dk_ptr = unpadded ? dqkv_unpadded[1] : p.dk;
  set_ck_strides(p.mode, st.k, ck_args.stride_b_dk, ck_args.stride_h_dk, ck_args.stride_s_dk);
  ck_args.dv_ptr = unpadded ? dqkv_unpadded[2] : p.dv;
  set_ck_strides(p.mode, st.v, ck_args.stride_b_dv, ck_args.stride_h_dv, ck_args.stride_s_dv);
  ck_args.dk_expanded_ptr = dk_expanded_ptr;
  set_ck_strides(p.mode, dk_expanded_stride,
                 ck_args.stride_b_dk_expanded, ck_args.stride_h_dk_expanded, ck_args.stride_s_dk_expanded);
  ck_args.dv_expanded_ptr = dv_expanded_ptr;
  set_ck_strides(p.mode, dv_expanded_stride,
                 ck_args.stride_b_dv_expanded, ck_args.stride_h_dv_expanded, ck_args.stride_s_dv_expanded);
  ck_args.lse_ptr = p.is_group_mode() ? lse_unpadded : p.softmax_aux;
  ck_args.lse_workspace_ptr = lse_workspace;
  if(p.mode == CkMode::Group){
    ck_args.cu_seqlen_q_padded_ptr = cu_seqlen_padded_q;
    ck_args.cu_seqlen_kv_padded_ptr = cu_seqlen_padded_kv;
  }else if(p.mode == CkMode::Batch){
    ck_args.bias_ptr = p.bias;
    ck_args.alibi_slope_ptr = alibi_slope;
    ck_args.dbias_expanded_ptr = dbias_expanded_ptr;
  }
  NVTE_CHECK_CUDA(ck_fused_attn::ck_attn_bwd(ck_args, stream));

  if(unpadded){
    // add padding for dq, dk, dv; dq, dk, dv of same shape as q, k, v
    pad_remap<PadDirection::Add>(p.dtype, p.b, p.h, p.s_q, p.d_qk, p.max_tokens_q, false, st.q[0], st.q[1], st.q[2], p.dq, p.cu_seqlens_q, cu_seqlen_padded_q, dqkv_unpadded[0], stream);
    pad_remap<PadDirection::Add>(p.dtype, p.b, p.hg, p.s_kv, p.d_qk, p.max_tokens_kv, false, st.k[0], st.k[1], st.k[2], p.dk, p.cu_seqlens_kv, cu_seqlen_padded_kv, dqkv_unpadded[1], stream);
    pad_remap<PadDirection::Add>(p.dtype, p.b, p.hg, p.s_kv, p.d_v, p.max_tokens_kv, false, st.v[0], st.v[1], st.v[2], p.dv, p.cu_seqlens_kv, cu_seqlen_padded_kv, dqkv_unpadded[2], stream);
  }
}
#endif // USE_FUSED_ATTN_CK
}  // namespace fused_attn_rocm

using namespace transformer_engine::fused_attn_rocm;

void fused_attn_ck_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  bool is_training, float attn_scale, float dropout,
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  NVTE_Softmax_Type softmax_type,
  int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_Bias,
  const Tensor* input_SoftmaxOffset,
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
  const DType O_type = output_O->data.dtype;
  const bool is_fp8 = QKV_type == DType::kFloat8E4M3;

  NVTE_CHECK(input_K->data.dtype == QKV_type && input_V->data.dtype == QKV_type,
             "ROCm fused attention requires Q, K, and V to have the same dtype.");
  if(is_fp8){
    NVTE_CHECK(!is_training, "AITER FP8 ASM fused attention is forward-inference only.");
    // FP8 E4M3 input is accepted only with BF16 output for AITER fp8bf16 kernels.
    NVTE_CHECK(O_type == DType::kBFloat16,
               "AITER fp8bf16 fused attention requires BF16 output.");
    NVTE_CHECK(input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING &&
                   input_K->scaling_mode == NVTE_DELAYED_TENSOR_SCALING &&
                   input_V->scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
               "AITER FP8 ASM fused attention supports delayed tensor scaling only.");
    auto valid_descale = [](const Tensor* tensor) {
      return tensor->scale_inv.dptr != nullptr &&
             tensor->scale_inv.dtype == DType::kFloat32 &&
             tensor->scale_inv.numel() == 1;
    };
    const bool all_descales_present =
        valid_descale(input_Q) && valid_descale(input_K) && valid_descale(input_V);
    // Workspace sizing uses null tensors. A real launch must carry the scales.
    if(input_Q->data.dptr != nullptr){
      NVTE_CHECK(all_descales_present,
                 "AITER FP8 ASM fused attention requires one FP32 scale_inv for each of Q, K, and V.");
    }
  }

  CkFwdProblem p;
  init_ck_problem(p, b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, attn_scale, dropout,
                  qkv_layout, bias_type, attn_mask_type, softmax_type,
                  window_size_left, window_size_right, bottom_right_diagonal,
                  input_Q, input_K, input_V, input_SoftmaxOffset,
                  input_cu_seqlens_q, input_cu_seqlens_kv,
                  input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, rng_state);
  if((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)){
    p.bias = input_Bias->data.dptr;
    p.bias_b = input_Bias->data.shape[0];
    p.bias_h = input_Bias->data.shape[1];
  }
  p.is_training = is_training;
  p.o_dtype = O_type;
  p.o = output_O->data.dptr;
  if(is_fp8){
    p.descale_q = input_Q->scale_inv.dptr;
    p.descale_k = input_K->scale_inv.dptr;
    p.descale_v = input_V->scale_inv.dptr;
  }

  size_t i = 0;
  if(Aux_CTX_Tensors->size == 0){
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_S->data.dptr = nullptr;
    if(p.is_ragged){
      output_S->data.shape = {p.max_tokens_q, h_q, 1};
    }else{
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
    }
    output_S->data.dtype = DType::kFloat32;

    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;

    if((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)){
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {p.bias_b, p.bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    }

    if(p.has_sink){
      Tensor *output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = nullptr;
      output_softmax_offset->data.shape = {1, h_q, 1, 1};
      output_softmax_offset->data.dtype = DType::kFloat32;
    }

    Aux_CTX_Tensors->size = i;
  }else if(Aux_CTX_Tensors->size >= 2){
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    p.softmax_aux = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    if((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)){
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_bias->data.dptr = p.bias;
    }
    if(p.has_sink){
      Tensor *output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = p.softmax_offset;
    }
  }else{
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  size_t workspace_size = 0;
  fused_attn_ck_fwd_impl(p, workspace->data.dptr, &workspace_size, stream);
  set_workspace_size(workspace, workspace_size);
  return;
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
#endif // USE_FUSED_ATTN_CK
}

void fused_attn_ck_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d_qk, size_t d_v,
  float attn_scale, float dropout,
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  NVTE_Softmax_Type softmax_type,
  int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias,
  const Tensor* input_SoftmaxOffset,
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  Tensor* output_dBias,
  Tensor* output_dSoftmaxOffset,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream){
#ifdef USE_FUSED_ATTN_CK
  CkBwdProblem p;
  init_ck_problem(p, b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, attn_scale, dropout,
                  qkv_layout, bias_type, attn_mask_type, softmax_type,
                  window_size_left, window_size_right, bottom_right_diagonal,
                  input_Q, input_K, input_V, input_SoftmaxOffset,
                  input_cu_seqlens_q, input_cu_seqlens_kv,
                  input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, rng_state);
  if((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)){
    p.bias = input_Bias->data.dptr;
    p.dbias = output_dBias->data.dptr;
    p.bias_b = output_dBias->data.shape[0];
    p.bias_h = output_dBias->data.shape[1];
  }
  if(p.has_sink){
    p.d_softmax_offset = output_dSoftmaxOffset->data.dptr;
  }
  p.deterministic = deterministic;
  p.o = input_O->data.dptr;
  p.d_o = input_dO->data.dptr;
  p.softmax_aux = output_S->data.dptr;
  p.dq = output_dQ->data.dptr;
  p.dk = output_dK->data.dptr;
  p.dv = output_dV->data.dptr;

  size_t workspace_size = 0;
  fused_attn_ck_bwd_impl(p, workspace->data.dptr, &workspace_size, stream);
  set_workspace_size(workspace, workspace_size);
  return;
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
#endif // USE_FUSED_ATTN_CK
}

}  // namespace transformer_engine
