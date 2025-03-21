/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <string>
#include <numeric> // Required for std::accumulate
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
  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }
  
  // single filters

  // filter based on head_dim
  //TODO: release after CK support support Multi-latent attention
  if(head_dim_qk != head_dim_v){
    if(nvte_log_ck_config){
      std::cout<<"CK fused attn does not support multi-latent attention"<<std::endl;
    }
    return false;
  }
  
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

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only gfx942 supported
  if(!(sm_arch_name_.find("gfx942")!=std::string::npos)){
    if(nvte_log_ck_config){
      std::cout<<"only gfx942 is supported"<<std::endl;
    }
    return false;
  }

  // joint filters

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
  return true;
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
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
      return ck_fused_attn::MaskType::mask_top_left;
    }
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK || nvte_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK){
    // nvte causal mask can map to (-1, 0) or (>=0, 0)
    return ck_fused_attn::MaskType::mask_top_left;
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK || nvte_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK){
    return ck_fused_attn::MaskType::mask_bottom_right;
  }
  return ck_fused_attn::MaskType::window_generic;
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
// kernel to remove padding for q, k, v, o (dq, dk, dv, do)
// each warp is in charge of one token index (h*d*sizeof(DataType) bytes copy)
// one lane (thread) in one warp is charge of one element in 32 segment trunk of h*d
template<typename DataType>
__global__ void remove_padding_kernel(
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged, // sometimes both cu_seqlen and cu_seqlen_padded given in bshd cases
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const DataType* data_ptr,
  const int32_t* cu_seqlen_ptr, const int32_t* cu_seqlen_padded_ptr,
  DataType* data_without_padding_ptr){

  int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
  int lane_idx = threadIdx.x % THREADS_PER_WARP;
  int num_warps = (blockDim.x * gridDim.x) / THREADS_PER_WARP;
  int num_total_tokens = cu_seqlen_ptr[b];

  uint64_t stride_total_seqlen = std::min(stride_b, stride_s);
  for(int token_id = warp_idx; token_id<num_total_tokens; token_id += num_warps){
    int b_idx = binary_search(token_id, cu_seqlen_ptr, b+1);
    int s_idx = token_id - cu_seqlen_ptr[b_idx];
    DataType* cur_without_padding = nullptr;
    const DataType* cur = nullptr;
    if(is_ragged){
      cur_without_padding = data_without_padding_ptr + stride_s* token_id;
      cur = data_ptr + (cu_seqlen_padded_ptr[b_idx] + s_idx)*stride_s;
    }else{
      cur_without_padding = data_without_padding_ptr + stride_total_seqlen* token_id;
      cur = data_ptr + (b_idx * stride_b + s_idx*stride_s);
    }
    for(int hd_idx = lane_idx; hd_idx < h*d; hd_idx+=THREADS_PER_WARP){
      int h_idx = hd_idx/d;
      int d_idx = hd_idx%d;
      cur_without_padding[h_idx*stride_h + d_idx] = cur[h_idx*stride_h + d_idx];
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
  constexpr int THREADS_PER_BLOCK = 256;
  // parallel over h*d dimension
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * b * s * THREADS_PER_WARP/THREADS_PER_BLOCK));

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, DataType,
    hipLaunchKernelGGL(
      remove_padding_kernel<DataType>, grid, block, 0, stream,
      b, h, s, d,
      is_ragged,
      stride_b, stride_h, stride_s,
      static_cast<const DataType*>(data_ptr),
      static_cast<const int32_t*>(cu_seqlen_ptr),
      static_cast<const int32_t*>(cu_seqlen_padded_ptr),
      static_cast<DataType*>(data_without_padding_ptr)););

}

// kernel to add padding for q, k, v, o (dq, dk, dv, do)
// reverse of remove_padding
template<typename DataType>
__global__ void add_padding_kernel(
  uint64_t b, uint64_t h, uint64_t s, uint64_t d,
  bool is_ragged,
  uint64_t stride_b, uint64_t stride_h, uint64_t stride_s, //stride_d is 1
  const DataType* data_without_padding_ptr,
  const int32_t* cu_seqlen_ptr, const int32_t* cu_seqlen_padded_ptr,
  DataType* data_ptr){

  int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
  int lane_idx = threadIdx.x % THREADS_PER_WARP;
  int num_warps = (blockDim.x * gridDim.x) / THREADS_PER_WARP;
  int num_total_tokens = cu_seqlen_ptr[b];

  uint64_t stride_total_seqlen = std::min(stride_b, stride_s);
  for(int token_id = warp_idx; token_id<num_total_tokens; token_id += num_warps){
    int b_idx = binary_search(token_id, cu_seqlen_ptr, b+1);
    int s_idx = token_id - cu_seqlen_ptr[b_idx];
    const DataType* cur_without_padding = nullptr;
    DataType* cur = nullptr;
    if(is_ragged){
      cur_without_padding = data_without_padding_ptr + stride_s* token_id;
      cur = data_ptr + (cu_seqlen_padded_ptr[b_idx] + s_idx)*stride_s;
    }else{
      cur_without_padding = data_without_padding_ptr + stride_total_seqlen* token_id;
      cur = data_ptr + (b_idx * stride_b + s_idx*stride_s);
    }
    for(int hd_idx = lane_idx; hd_idx < h*d; hd_idx+=THREADS_PER_WARP){
      int h_idx = hd_idx/d;
      int d_idx = hd_idx%d;
      cur[h_idx*stride_h + d_idx] = cur_without_padding[h_idx*stride_h + d_idx];
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
  constexpr int THREADS_PER_BLOCK = 256;
  // parallel over h*d dimension
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * b*s * THREADS_PER_WARP/THREADS_PER_BLOCK));

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, DataType,
    hipLaunchKernelGGL(
      add_padding_kernel<DataType>, grid, block, 0, stream,
      b, h, s, d,
      is_ragged,
      stride_b, stride_h, stride_s,
      static_cast<const DataType*>(data_without_padding_ptr),
      static_cast<const int32_t*>(cu_seqlen_ptr),
      static_cast<const int32_t*>(cu_seqlen_padded_ptr),
      static_cast<DataType*>(data_ptr)););

}
// actual fwd implementation, calling ck api directly
void fused_attn_ck_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
  bool pad_between_seqs, size_t q_storage_bytes, size_t k_storage_bytes, size_t v_storage_bytes, size_t o_storage_bytes,
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

  bool is_ragged = nvte_get_qkv_format(layout)==NVTE_QKV_Format::NVTE_THD; 
 
  // Exit to request upper level API to allocate memory if needed
  if(workspace==nullptr){
    // ck requires an alibi slope array even if in standard (vanilla) mode
    if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
      (*workspace_size)+= h*sizeof(float);
    }
    // softmax_lse buffer needed for THD qkv_layout
    if(is_ragged or pad_between_seqs){
      (*workspace_size)+= b*h*s_q*sizeof(float);
    }
    // request q, k, v, o buffer without padding
    if(pad_between_seqs){
      (*workspace_size)+= q_storage_bytes + k_storage_bytes + v_storage_bytes + o_storage_bytes;
    }
    if (nvte_log_ck_config) {
      std::cout<<std::endl<<"attn_fwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }
  // denote the next available section of workspace from upstream
  void* workspace_next = workspace;

  std::array<uint64_t, 4> q_stride;
  std::array<uint64_t, 4> k_stride;
  std::array<uint64_t, 4> v_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  void* devPtrAlibiSlope = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    devPtrAlibiSlope = workspace_next;
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    //assign standard alibi slope
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
    //move workspace to next unused section
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + h*sizeof(float));
  }
  // First b*h*sq*sizeof(float) in workspace are for lse in THD layout
  void* devPtrSoftmaxLSETHD = nullptr;
  if(is_ragged or pad_between_seqs){
    devPtrSoftmaxLSETHD = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + b*h*s_q*sizeof(float));
  }
  void* devPtrQWithoutPadding = nullptr;
  void* devPtrKWithoutPadding = nullptr;
  void* devPtrVWithoutPadding = nullptr;
  void* devPtrOWithoutPadding = nullptr;
  if(pad_between_seqs){
    //determine q, k ,v buffer based on the workspace next ptr and layout group
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
    //Q ptr always comes at first
    devPtrQWithoutPadding = workspace_next;
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      //keep the start address difference the same among q, k, and v
      devPtrKWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrQWithoutPadding) + (static_cast<int8_t *>(devPtrK) - static_cast<int8_t *>(devPtrQ)));
      devPtrVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrQWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrQ)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes + k_storage_bytes + v_storage_bytes);
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      //keep the start address difference the same between k and v
      devPtrKWithoutPadding = workspace_next;
      devPtrVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrKWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrK)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes + v_storage_bytes);
    }else{
      //qkv separated
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      devPtrKWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes);
      devPtrVWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + v_storage_bytes);
    }
    //determine the o buffer based on workspace next section
    devPtrOWithoutPadding = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + o_storage_bytes);
    // reset the final results since padded places need to be 0
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrO, 0, o_storage_bytes, stream));
  }

  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_fwd(ck): ";
    std::cout<<"layout: "<<layout<<", ";
    if(is_ragged){
      // THD
      std::cout<<"q_shape: ("<<b*s_q<<", "<<h<<", "<<d<<"), ";
      std::cout<<"q_stride: ("<<q_stride[2]<<", "<<q_stride[1]<<", "<<q_stride[3]<<"), ";
      std::cout<<"kv_shape: ("<<b*s_kv<<", "<<hg<<", "<<d<<"), ";
      std::cout<<"k_stride: ("<<k_stride[2]<<", "<<k_stride[1]<<", "<<k_stride[3]<<"), ";
      std::cout<<"v_stride: ("<<v_stride[2]<<", "<<v_stride[1]<<", "<<v_stride[3]<<"), ";

      std::cout<<"o_shape: ("<<b*s_q<<", "<<h<<", "<<d<<"), ";
      std::cout<<"o_stride: ("<<o_stride[2]<<", "<<o_stride[1]<<", "<<o_stride[3]<<"), ";
    }else{
      // non-THD
      std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
      std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
      std::cout<<"kv_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d<<"), ";
      std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
      std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";
 
      std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
      std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    }
    std::cout<<"pad_between_seqs: "<<pad_between_seqs<<", ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    std::cout<<"M_shape: ("<<b*h<<", "<<s_q<<"), ";
    std::cout<<"M_stride: ("<<s_q<<", "<<1<<"), ";
    std::cout<<"is_training: "<<is_training<<", ";
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<std::endl;
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<std::endl;
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<")"<<std::endl;
  }
  if(pad_between_seqs){
    // remove padding for q, k, v
    remove_padding(dtype, b, h, s_q, d, is_ragged, q_stride[0], q_stride[1], q_stride[2], devPtrQ, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrQWithoutPadding, stream);
    remove_padding(dtype, b, hg, s_kv, d, is_ragged, k_stride[0], k_stride[1], k_stride[2], devPtrK, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrKWithoutPadding, stream);
    remove_padding(dtype, b, hg, s_kv, d, is_ragged, v_stride[0], v_stride[1], v_stride[2], devPtrV, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrVWithoutPadding, stream);
    // call varlen api using without_padding ptrs
    // for BSHD/SBHD, after padding removal, THD require stride_s update
    using ck_fused_attn::ck_attn_varlen_fwd;
    NVTE_CHECK_CUDA(
      ck_attn_varlen_fwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d,
        devPtrQWithoutPadding,
        q_stride[1], (is_ragged? q_stride[2] : std::min(q_stride[0], q_stride[2])),
        devPtrKWithoutPadding,
        k_stride[1], (is_ragged? k_stride[2] : std::min(k_stride[0], k_stride[2])),
        devPtrVWithoutPadding,
        v_stride[1], (is_ragged? v_stride[2] : std::min(v_stride[0], v_stride[2])),
        devPtrCuSeqlensQ, devPtrCuSeqlensKV, 
        is_training, scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrOWithoutPadding,
        o_stride[1], (is_ragged? o_stride[2] : std::min(o_stride[0], o_stride[2])),
        devPtrSoftmaxLSETHD,
        devPtrSoftmaxAux,
        stream));
    // add padding for o
    // o share the same shape as q
    add_padding(dtype, b, h, s_q, d, is_ragged, o_stride[0], o_stride[1], o_stride[2], devPtrOWithoutPadding, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrO, stream);
  }else if(is_ragged){
    using ck_fused_attn::ck_attn_varlen_fwd;
    NVTE_CHECK_CUDA(
      ck_attn_varlen_fwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d,
        devPtrQ, 
        q_stride[1], q_stride[2],
        devPtrK, 
        k_stride[1], k_stride[2],
        devPtrV, 
        v_stride[1], v_stride[2],
        devPtrCuSeqlensQ, devPtrCuSeqlensKV, 
        is_training, scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrO,
        o_stride[1], o_stride[2],
        devPtrSoftmaxLSETHD,
        devPtrSoftmaxAux,
        stream));
    // softmax_lse will be fixed from [h, b*s_q] to [b, h, s] in ck_fused_attn
  }else{
    using ck_fused_attn::ck_attn_fwd;
    NVTE_CHECK_CUDA(
      ck_attn_fwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d, bias_b, bias_h,
        devPtrQ, 
        q_stride[0], q_stride[1], q_stride[2],
        devPtrK, 
        k_stride[0], k_stride[1], k_stride[2],
        devPtrV, 
        v_stride[0], v_stride[1], v_stride[2],
        devPtrBias,
        devPtrAlibiSlope,
        is_training, scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        nvte_to_ck_bias_type(bias_type),
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrO,
        o_stride[0], o_stride[1], o_stride[2],
        devPtrSoftmaxAux,
        stream));
  }
}

void fused_attn_ck_bwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
  bool pad_between_seqs, size_t q_storage_bytes, size_t k_storage_bytes, size_t v_storage_bytes, size_t o_storage_bytes,
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

  bool is_mqa_gqa = (h > hg);

  size_t kN0 = (d <= 128)? 128:64;
  size_t nsplits = deterministic? ceil(1.0*s_kv/kN0):1; 

  bool is_ragged = nvte_get_qkv_format(layout)==NVTE_QKV_Format::NVTE_THD; 
  // Exit to request upper level API to allocate memory if needed
  if(workspace==nullptr){
    size_t workspace_size_lse = b*h*s_q*sizeof(float);
    // CK requires dq_acc ptr, dq_acc depends on is deterministic
    size_t workspace_size_dq_acc = nsplits*b*h*s_q*d*sizeof(float);
    *workspace_size = workspace_size_lse + workspace_size_dq_acc;
    if(is_mqa_gqa){
      // allocate dk, dv (or dkv) as if h=hg
      size_t dkv_expanded_size = 2*b*h*s_kv*d*nvte_dtype_size(dtype);
      *workspace_size += dkv_expanded_size;
    }
    // ck requires an alibi slope array even if in standard (vanilla) mode
    if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
      (*workspace_size)+= h*sizeof(float);
    }else if ((bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) && (bias_b!=b or bias_h!=h)){
      //ck requires a buffer dbias_expanded of size BHSS if bias is not BHSS
      (*workspace_size) += b*h*s_q*s_kv*nvte_dtype_size(dtype);
    }
    if(is_ragged or pad_between_seqs){
      // transform the input softmax_lse of shape [b, h, s] into [h, b*s_q] with total_seqlen_q effective values
      (*workspace_size)+= b*h*s_q*sizeof(float);
    }
    // allocate the q, k, v, o, do, dq, dk, dv,
    if(pad_between_seqs){
      (*workspace_size)+= 2*(q_storage_bytes + k_storage_bytes + v_storage_bytes + o_storage_bytes);
    }
    if (nvte_log_ck_config) {
      std::cout<<std::endl<<"attn_bwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }
  // denote the next available section of workspace from upstream
  void* workspace_next = workspace;

  std::array<uint64_t, 4> q_stride;
  std::array<uint64_t, 4> k_stride;
  std::array<uint64_t, 4> v_stride;
  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  //q and o are having the same shape
  //k and v are having the same shape
  //x and dx are having the same shape and stride

  //initialize (zeroing out) some buffers due to ck requirement
  //ck bwd requires initialize dq since ck uses atomic operations
  //TODO: remove the memset afer ck fixes the atomic operations
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
  if((layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) or (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D)){
    // just memset all dq, dk, dv
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQ, 0, q_storage_bytes + k_storage_bytes+ v_storage_bytes, stream));
  }else{
    // HD_2HD, HD_H2D, HD_HD_HD can just memset dq itself
    NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQ, 0, q_storage_bytes, stream));
    // for pad between seqs case, we need to reset all dq, dk, dv
    if(pad_between_seqs){
      if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
        //kvpacked
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdK, 0, k_storage_bytes + v_storage_bytes, stream));
      }else{
        //q, k, v separated
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdK, 0, k_storage_bytes, stream));
        NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdV, 0, v_storage_bytes, stream));
      }
    }
  }
 
  // assign different kind of temporary buffers and initialize them due to ck requirement

  // First b*h*sq*sizeof(float) in workspace are for lse-d
  void* lse_workspace = workspace;
  workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + b*h*s_q*sizeof(float));

  // The next section are for dq_acc_ptr
  void* dq_acc_ptr = workspace_next;
  workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + nsplits*b*h*s_q*d*sizeof(float));
  // like dq, dq_acc mem also requires zeroing out
  //dq_acc is of shape (nsplits, B, S, H, D)
  NVTE_CHECK_CUDA(cudaMemsetAsync(dq_acc_ptr, 0, sizeof(float)*nsplits*b*h*s_q*d, stream));
 
  void* dk_expanded_ptr = nullptr;
  void* dv_expanded_ptr = nullptr;
  std::array<uint64_t, 4> dkv_expanded_stride;
  //mqa gqa mode
  if(is_mqa_gqa){
    //generate kv expanded stride as if h_kv = h_q
    generateMatrixStrides(b, h, s_q, s_kv, d, dkv_expanded_stride.data(),
                          layout, NVTE_QKV_Matrix::NVTE_K_Matrix);

    // dk_expanded arranged at the end of dq_acc_ptr
    dk_expanded_ptr = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + 2*b*h*s_kv*d*nvte_dtype_size(dtype));

    //dv_expanded_ptr depends on the actual layout
    if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + nvte_dtype_size(dtype)*h*d);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + nvte_dtype_size(dtype)*d);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + nvte_dtype_size(dtype)*b*h*s_kv*d);
    } else{
      NVTE_ERROR("NVTE_3HD NVTE_H3D should have h=hg.");
    }
    // zeroing out dkv expanded in case CK requires that
    NVTE_CHECK_CUDA(cudaMemsetAsync(dk_expanded_ptr, 0, 2*nvte_dtype_size(dtype)*b*h*s_kv*d, stream));
  }

  void* devPtrAlibiSlope = nullptr;
  void* dbias_expanded_ptr = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    devPtrAlibiSlope = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + h*sizeof(float));

    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    //assign standard alibi slope
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
  }else if((bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) && (devPtrdBias!=nullptr)){
    if(bias_b!=b or bias_h!= h){
      dbias_expanded_ptr = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + b*h*s_q*s_kv*nvte_dtype_size(dtype));
      // zeroing out dbias_expanded_ptr as CK requires that
      NVTE_CHECK_CUDA(cudaMemsetAsync(dbias_expanded_ptr, 0, nvte_dtype_size(dtype)*b*h*s_q*s_kv, stream));
    }else{
      // dbias_expanded_ptr not needed for BHSS shape
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdBias, 0, nvte_dtype_size(dtype)*bias_b*bias_h*s_q*s_kv, stream));
    }
  }
  
  void* devPtrSoftmaxLSETHD = nullptr;
  if(is_ragged or pad_between_seqs){
    devPtrSoftmaxLSETHD = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + b*h*s_q*sizeof(float));
  }
  
  void* devPtrQWithoutPadding = nullptr;
  void* devPtrKWithoutPadding = nullptr;
  void* devPtrVWithoutPadding = nullptr;
  void* devPtrOWithoutPadding = nullptr;
  void* devPtrdOWithoutPadding = nullptr;
  void* devPtrdQWithoutPadding = nullptr;
  void* devPtrdKWithoutPadding = nullptr;
  void* devPtrdVWithoutPadding = nullptr;
  if(pad_between_seqs){
    //determine q, k, v buffer based on the workspace next ptr and layout group
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
    //Q ptr always comes at first
    devPtrQWithoutPadding = workspace_next;
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      //keep the start address difference the same among q, k, and v
      devPtrKWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrQWithoutPadding) + (static_cast<int8_t *>(devPtrK) - static_cast<int8_t *>(devPtrQ)));
      devPtrVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrQWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrQ)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes + k_storage_bytes + v_storage_bytes);
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      //keep the start address difference the same between k and v
      devPtrKWithoutPadding = workspace_next;
      devPtrVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrKWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrK)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes + v_storage_bytes);
    }else{
      //qkv separated
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      devPtrKWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes);
      devPtrVWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + v_storage_bytes);
    }
    //determine the o, do buffer based on workspace next section
    devPtrOWithoutPadding = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + o_storage_bytes);
    devPtrdOWithoutPadding = workspace_next;
    workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + o_storage_bytes);

    //determine dq, dk, dv buffer based on the workspace next ptr and layout group
    //dQ ptr always comes at first
    devPtrdQWithoutPadding = workspace_next;
    if(layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D){
      //keep the start address difference the same among q, k, and v
      devPtrdKWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrdQWithoutPadding) + (static_cast<int8_t *>(devPtrK) - static_cast<int8_t *>(devPtrQ)));
      devPtrdVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrdQWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrQ)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes + k_storage_bytes + v_storage_bytes);

      // zeroing out the entire dqkv since packed
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQWithoutPadding, 0, q_storage_bytes + k_storage_bytes+ v_storage_bytes, stream));
    }else if(layout_group==NVTE_QKV_Layout_Group::NVTE_HD_2HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      //keep the start address difference the same between k and v
      devPtrdKWithoutPadding = workspace_next;
      devPtrdVWithoutPadding = static_cast<void *>(static_cast<int8_t *>(devPtrdKWithoutPadding) + (static_cast<int8_t *>(devPtrV) - static_cast<int8_t *>(devPtrK)));
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes + v_storage_bytes);

      // zeroing out just the dq itself
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQWithoutPadding, 0, q_storage_bytes, stream));
    }else{
      //qkv separated
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + q_storage_bytes);
      devPtrdKWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + k_storage_bytes);
      devPtrdVWithoutPadding = workspace_next;
      workspace_next = static_cast<void *>(static_cast<int8_t *>(workspace_next) + v_storage_bytes);

      // zeroing out just the dq itself
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdQWithoutPadding, 0, q_storage_bytes, stream));
    }
  }

  // bwd v3 is optional by enabling the following envs
  // default values follows the ck example setting
  bool nvte_ck_uses_bwd_v3 = getenv<int>("NVTE_CK_USES_BWD_V3", 0);
  bool nvte_ck_is_v3_atomic_fp32 = getenv<int>("NVTE_CK_IS_V3_ATOMIC_FP32", 1);
  int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);

  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_bwd(ck): ";
    std::cout<<"layout: "<<layout<<", ";
    if(is_ragged){
      // THD
      std::cout<<"q_shape: ("<<b*s_q<<", "<<h<<", "<<d<<"), ";
      std::cout<<"q_stride: ("<<q_stride[2]<<", "<<q_stride[1]<<", "<<q_stride[3]<<"), ";
      std::cout<<"kv_shape: ("<<b*s_kv<<", "<<hg<<", "<<d<<"), ";
      std::cout<<"k_stride: ("<<k_stride[2]<<", "<<k_stride[1]<<", "<<k_stride[3]<<"), ";
      std::cout<<"v_stride: ("<<v_stride[2]<<", "<<v_stride[1]<<", "<<v_stride[3]<<"), ";

      std::cout<<"o_shape: ("<<b*s_q<<", "<<h<<", "<<d<<"), ";
      std::cout<<"o_stride: ("<<o_stride[2]<<", "<<o_stride[1]<<", "<<o_stride[3]<<"), ";
    }else{
      // non-THD
      std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
      std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
      std::cout<<"kv_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d<<"), ";
      std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
      std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";

      std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
      std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    }
    std::cout<<"pad_between_seqs: "<<pad_between_seqs<<", ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    std::cout<<"M_shape: ("<<b*h<<", "<<s_q<<"), ";
    std::cout<<"M_stride: ("<<s_q<<", "<<1<<"), ";
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<std::endl;
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<std::endl;
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<")"<<std::endl;
    std::cout<<"deterministic: "<<deterministic<<std::endl;
  }
  if(pad_between_seqs){
    // remove padding for q, k, v, o, do
    remove_padding(dtype, b, h, s_q, d, is_ragged, q_stride[0], q_stride[1], q_stride[2], devPtrQ, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrQWithoutPadding, stream);
    remove_padding(dtype, b, hg, s_kv, d, is_ragged, k_stride[0], k_stride[1], k_stride[2], devPtrK, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrKWithoutPadding, stream);
    remove_padding(dtype, b, hg, s_kv, d, is_ragged, v_stride[0], v_stride[1], v_stride[2], devPtrV, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrVWithoutPadding, stream);
    // o and do should be of same shape as q
    remove_padding(dtype, b, h, s_q, d, is_ragged, o_stride[0], o_stride[1], o_stride[2], devPtrO, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrOWithoutPadding, stream);
    remove_padding(dtype, b, h, s_q, d, is_ragged, o_stride[0], o_stride[1], o_stride[2], devPtrdO, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrdOWithoutPadding, stream);

    using ck_fused_attn::ck_attn_varlen_bwd;
    NVTE_CHECK_CUDA(
      ck_attn_varlen_bwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d,
        devPtrQWithoutPadding,
        q_stride[1], (is_ragged? q_stride[2] : std::min(q_stride[0], q_stride[2])),
        devPtrKWithoutPadding,
        k_stride[1], (is_ragged? k_stride[2] : std::min(k_stride[0], k_stride[2])),
        devPtrVWithoutPadding,
        v_stride[1], (is_ragged? v_stride[2] : std::min(v_stride[0], v_stride[2])),
        devPtrCuSeqlensQ, devPtrCuSeqlensKV, 
        devPtrOWithoutPadding,
        o_stride[1], (is_ragged? o_stride[2] : std::min(o_stride[0], o_stride[2])),
        devPtrSoftmaxAux,
        devPtrdOWithoutPadding,
        o_stride[1], (is_ragged? o_stride[2] : std::min(o_stride[0], o_stride[2])), //dO and O share the same stride in TE
        scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrdQWithoutPadding,
        q_stride[1], (is_ragged? q_stride[2] : std::min(q_stride[0], q_stride[2])), //dq and q share the same stride in TE
        dq_acc_ptr,
        dk_expanded_ptr,
        dv_expanded_ptr,
        dkv_expanded_stride[1], (is_ragged? dkv_expanded_stride[2] : std::min(dkv_expanded_stride[0], dkv_expanded_stride[2])), //dK and K share the same stride
        devPtrdKWithoutPadding,
        k_stride[1], (is_ragged? k_stride[2] : std::min(k_stride[0], k_stride[2])), //dK and K share the same stride
        devPtrdVWithoutPadding,
        v_stride[1], (is_ragged? v_stride[2] : std::min(v_stride[0], v_stride[2])), //dV and V share the same stride
        lse_workspace, // softmax_lsed
        devPtrSoftmaxLSETHD,
        deterministic,
        // bwd_v3 not supported for THD
        stream));
    // add padding for dq, dk, dv
    // dq, dk, dv of same shape as q, k, v
    add_padding(dtype, b, h, s_q, d, is_ragged, q_stride[0], q_stride[1], q_stride[2], devPtrdQWithoutPadding, devPtrCuSeqlensQ, devPtrSeqOffsetsQ, devPtrdQ, stream);
    add_padding(dtype, b, hg, s_kv, d, is_ragged, k_stride[0], k_stride[1], k_stride[2], devPtrdKWithoutPadding, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrdK, stream);
    add_padding(dtype, b, hg, s_kv, d, is_ragged, v_stride[0], v_stride[1], v_stride[2], devPtrdVWithoutPadding, devPtrCuSeqlensKV, devPtrSeqOffsetsKV, devPtrdV, stream);

  }else if(is_ragged){
    using ck_fused_attn::ck_attn_varlen_bwd;
    NVTE_CHECK_CUDA(
      ck_attn_varlen_bwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d,
        devPtrQ,
        q_stride[1], q_stride[2],
        devPtrK,
        k_stride[1], k_stride[2],
        devPtrV,
        v_stride[1], v_stride[2],
        devPtrCuSeqlensQ, devPtrCuSeqlensKV, 
        devPtrO,
        o_stride[1], o_stride[2],
        devPtrSoftmaxAux,
        devPtrdO,
        o_stride[1], o_stride[2], //dO and O share the same stride
        scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrdQ,
        q_stride[1], q_stride[2], //dQ and Q share the same stride
        dq_acc_ptr, 
        dk_expanded_ptr,
        dv_expanded_ptr,
        dkv_expanded_stride[1], dkv_expanded_stride[2], //dK and K share the same stride
        devPtrdK,
        k_stride[1], k_stride[2], //dK and K share the same stride
        devPtrdV,
        v_stride[1], v_stride[2], //dV and V share the same stride
        lse_workspace, // softmax_lsed
        devPtrSoftmaxLSETHD,
        deterministic,
        // bwd_v3 not supported for THD
        stream));
  }else{
    using ck_fused_attn::ck_attn_bwd;
    NVTE_CHECK_CUDA(
      ck_attn_bwd(
        nvte_to_ck_dtype(dtype),
        b, h, hg, s_q, s_kv, d, bias_b, bias_h,
        devPtrQ,
        q_stride[0], q_stride[1], q_stride[2],
        devPtrK,
        k_stride[0], k_stride[1], k_stride[2],
        devPtrV,
        v_stride[0], v_stride[1], v_stride[2],
        devPtrBias,
        devPtrAlibiSlope,
        devPtrO,
        o_stride[0], o_stride[1], o_stride[2],
        devPtrSoftmaxAux,
        devPtrdO,
        o_stride[0], o_stride[1], o_stride[2], //dO and O share the same stride
        scaling_factor, dropout_probability,
        devPtrDropoutSeed, devPtrDropoutOffset,
        nvte_to_ck_bias_type(bias_type),
        set_ck_mask(mask_type, window_size_left, window_size_right),
        window_size_left, window_size_right,
        devPtrdQ,
        q_stride[0], q_stride[1], q_stride[2], //dQ and Q share the same stride
        dq_acc_ptr, 
        dk_expanded_ptr,
        dv_expanded_ptr,
        dkv_expanded_stride[0], dkv_expanded_stride[1], dkv_expanded_stride[2], //dK and K share the same stride
        devPtrdK,
        k_stride[0], k_stride[1], k_stride[2], //dK and K share the same stride
        devPtrdV,
        v_stride[0], v_stride[1], v_stride[2], //dV and V share the same stride
        dbias_expanded_ptr,
        devPtrdBias,
        lse_workspace,
        deterministic,
        nvte_ck_uses_bwd_v3,
        nvte_ck_is_v3_atomic_fp32,
        nvte_ck_how_v3_bf16_cvt,
        stream));
  }
}
#endif // USE_FUSED_ATTN_CK
}  // namespace fused_attn_rocm

using namespace transformer_engine::fused_attn_rocm;
void fused_attn_ck_fwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  const Tensor* input_QKV, const Tensor* input_Bias, 
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens,
  const Tensor* input_cu_seqlens_padded,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_CK
  const DType QKV_type = input_QKV->data.dtype;
  void *devPtrQKV = input_QKV->data.dptr;
  // determine the stride based on qkv layout
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = nvte_dtype_size(QKV_type) * h * d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = nvte_dtype_size(QKV_type) * d;
  }
  void *devPtrQ = static_cast<void *>(devPtrQKV);
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }
  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;
  void *devPtrCuSeqlens = input_cu_seqlens->data.dptr;
  void *devPtrSeqOffsets = input_cu_seqlens_padded->data.dptr;

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h, max_seqlen, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen, max_seqlen};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h, max_seqlen, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  size_t workspace_size = 0;

  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens->data.shape.empty()));
  
  // extract the qkv and o storage bytes to allocate buffer for padding removing
  size_t qkv_storage_bytes = 0;
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  qkv_storage_bytes = std::accumulate((input_QKV->data).shape.begin(), (input_QKV->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // ensure q, k ,v are of the same storage size
  assert(qkv_storage_bytes%3==0);
  // in qkvpacked layouts, o is of the same shape as q shape

  fused_attn_ck_fwd_impl(
    b, h, h, max_seqlen, max_seqlen, d, bias_b, bias_h,
    pad_between_seqs, qkv_storage_bytes/3, qkv_storage_bytes/3, qkv_storage_bytes/3, qkv_storage_bytes/3,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    devPtrQ, 
    devPtrK, 
    devPtrV, 
    devPtrBias,
    devPtrS, 
    devPtrO,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
    devPtrCuSeqlens, devPtrCuSeqlens,
    devPtrSeqOffsets, devPtrSeqOffsets,
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

void fused_attn_ck_bwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_QKV, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  const Tensor* output_S,
  Tensor* output_dQKV,
  Tensor* output_dBias,
  const Tensor* input_cu_seqlens,
  const Tensor* input_cu_seqlens_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_CK
  const DType QKV_type = input_QKV->data.dtype;
  //input tensor
  void *devPtrQKV = input_QKV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = nvte_dtype_size(QKV_type) * h * d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = nvte_dtype_size(QKV_type) * d;
  }
  void *devPtrQ = static_cast<void *>(devPtrQKV);
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);
  void *devPtrSoftmaxStats = output_S->data.dptr;

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

  // output tensor
  void *devPtrdQKV = output_dQKV->data.dptr;
  void *devPtrdQ = static_cast<void *>(devPtrdQKV);
  void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);

  void *devPtrCuSeqlens = input_cu_seqlens->data.dptr; 
  void *devPtrSeqOffsets = input_cu_seqlens_padded->data.dptr;
  
  size_t workspace_size = 0;

  // extract the qkv and o storage bytes to clear dq buffer
  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens->data.shape.empty()));
  // extract the qkv and o storage bytes to allocate buffer for padding removing
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  size_t qkv_storage_bytes = std::accumulate((input_QKV->data).shape.begin(), (input_QKV->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // ensure q, k ,v are of the same storage size
  assert(qkv_storage_bytes%3==0);
  // in qkvpacked layouts, o is of the same shape as q shape
  // dqkv has the same shape as qkv
  // do has the same shape as o

  fused_attn_ck_bwd_impl(
    b, h, h, max_seqlen, max_seqlen, d, bias_b, bias_h,
    pad_between_seqs, qkv_storage_bytes/3, qkv_storage_bytes/3, qkv_storage_bytes/3, qkv_storage_bytes/3,
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
    devPtrCuSeqlens, devPtrCuSeqlens,
    devPtrSeqOffsets, devPtrSeqOffsets,
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

void fused_attn_ck_fwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  const Tensor* input_Q, const Tensor* input_KV, const Tensor* input_Bias, 
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
  //input tensor
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = nvte_dtype_size(QKV_type)*h_kv*d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = nvte_dtype_size(QKV_type) * d;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }
  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;
  void *devPtrCuSeqlensQ = input_cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = input_cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = input_cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = input_cu_seqlens_kv_padded->data.dptr;

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }
  
  size_t workspace_size = 0;

  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_q_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens_q->data.shape.empty()));
  
  // extract the qkv and o storage bytes to allocate buffer for padding removing
  size_t q_storage_bytes = 0;
  size_t kv_storage_bytes = 0;
  // b from cu_seqlen is not the actual storage batch for pad_between_seq case
  q_storage_bytes = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  kv_storage_bytes = std::accumulate((input_KV->data).shape.begin(), (input_KV->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // ensure k ,v are of the same storage size
  assert(kv_storage_bytes%2==0);
  
  // also need a o buffer without padding
  // in kvpacked layout, o will have the same shape as q

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    pad_between_seqs, q_storage_bytes, kv_storage_bytes/2, kv_storage_bytes/2, q_storage_bytes, 
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

void fused_attn_ck_bwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_KV, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dKV,
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
  //input tensor
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = nvte_dtype_size(QKV_type) * h_kv * d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = nvte_dtype_size(QKV_type) * d;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

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
  // output tensor
  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdKV = output_dKV->data.dptr;
  void *devPtrdK = devPtrdKV;
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdKV) + stride);

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
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_q_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens_q->data.shape.empty()));
  
  // extract the qkv and o storage bytes to clear qkv buffer and allocate buffer for padding removing
  // b from cu_seqlen is not the actual storage batch for pad_between_seq case
  size_t q_storage_bytes = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  size_t kv_storage_bytes = std::accumulate((input_KV->data).shape.begin(), (input_KV->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // ensure k ,v are of the same storage size
  assert(kv_storage_bytes%2==0);
  
  // also need a o buffer without padding
  // in kvpacked layout, o will have the same shape as q

  fused_attn_ck_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    pad_between_seqs, q_storage_bytes, kv_storage_bytes/2, kv_storage_bytes/2, q_storage_bytes, 
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

void fused_attn_ck_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
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

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }
  size_t workspace_size = 0;

  bool is_ragged = nvte_get_qkv_format(qkv_layout)==NVTE_QKV_Format::NVTE_THD; 
  bool is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK || 
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                     attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_q_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens_q->data.shape.empty()));
  
  // extract the qkv and o storage bytes to allocate buffer for padding removing
  size_t q_storage_bytes = 0;
  size_t k_storage_bytes = 0;
  size_t v_storage_bytes = 0;
  size_t o_storage_bytes = 0;
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  q_storage_bytes = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  k_storage_bytes = std::accumulate((input_K->data).shape.begin(), (input_K->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  v_storage_bytes = std::accumulate((input_V->data).shape.begin(), (input_V->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // in qkvpacked layouts, o is of the same shape as q shape
  o_storage_bytes = std::accumulate((output_O->data).shape.begin(), (output_O->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    pad_between_seqs, q_storage_bytes, k_storage_bytes, v_storage_bytes, o_storage_bytes,
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
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
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
  bool pad_between_seqs = (is_ragged && !(input_cu_seqlens_q_padded->data.shape.empty())) || (!is_ragged && is_padding && !(input_cu_seqlens_q->data.shape.empty()));
  
  // extract the qkv and o storage bytes to allocate buffer for clearing buffer and padding removing
  // b from cu_seqlen is not the actual storage batch for pad_between_seqs case
  size_t q_storage_bytes = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  size_t k_storage_bytes = std::accumulate((input_K->data).shape.begin(), (input_K->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  size_t v_storage_bytes = std::accumulate((input_V->data).shape.begin(), (input_V->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);
  // in qkvpacked layouts, o is of the same shape as q shape
  size_t o_storage_bytes = std::accumulate((input_O->data).shape.begin(), (input_O->data).shape.end(), 1, std::multiplies<size_t>())*nvte_dtype_size(QKV_type);

  fused_attn_ck_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    pad_between_seqs, q_storage_bytes, k_storage_bytes, v_storage_bytes, o_storage_bytes,
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
