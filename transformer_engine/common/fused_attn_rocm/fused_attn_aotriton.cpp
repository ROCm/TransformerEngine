/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/


#include <iostream>
#include <string>
#ifdef USE_FUSED_ATTN_AOTRITON
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#endif // USE_FUSED_ATTN_AOTRITON
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_aotriton.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

// check the fused attn config to see whether it's aotriton backend supported
bool is_aotriton_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim,
  int64_t window_size_left,
  int64_t window_size_right) {

#ifdef USE_FUSED_ATTN_AOTRITON
  //TODO: release after TE integrates swa into AOTriton
  bool is_no_mask_window_size= window_size_left == -1 && window_size_right == -1;
  bool is_causal_mask_window_size = window_size_left ==-1 && window_size_right ==0;
  if(!(is_no_mask_window_size || is_causal_mask_window_size)){
    return false;
  }

  //aotriton fused attn does not support gqa mode now
  if(num_attn_heads!=num_gqa_groups){
    return false;
  }

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  bool is_qkvpacked = layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D;
  // qkvpacked layout requires seq length to be the same
  if(is_qkvpacked && max_seqlen_q!=max_seqlen_kv){
    return false;
  }

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only MI250 or MI300X supported
  if(!((sm_arch_name_.find("gfx942")!=std::string::npos) || (sm_arch_name_.find("gfx90a")!=std::string::npos))){
    return false;
  }
  
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    return false;
  }
  
  //Only BSHD, SBHD style layouts supported
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if(!(qkv_format == NVTE_QKV_Format::NVTE_SBHD||
    qkv_format == NVTE_QKV_Format::NVTE_BSHD)){
    return false;
  }
  
  // AOTriton does not support bias now
  if(!(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)){
    return false;
  }

  // Only no mask and causal mask supported
  if(!(attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK||
    attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)){
    return false;
  } 
  
  // causal does not work with s_q != s_kv
  if(max_seqlen_q!=max_seqlen_kv && attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK){
    return false;
  }

  return true;
#else
  NVTE_ERROR("AOTriton backend not compiled.");
  return false;
#endif // USE_FUSED_ATTN_AOTRITON
}


#ifdef USE_FUSED_ATTN_AOTRITON
aotriton::DType nvte_to_aotriton_dtype(NVTEDType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == NVTEDType::aname) return aotriton::DType::dtname
  CAST_TYPE(kNVTEByte, kUInt8);
  CAST_TYPE(kNVTEFloat32, kFloat32);
  CAST_TYPE(kNVTEFloat16, kFloat16);
  CAST_TYPE(kNVTEBFloat16, kBFloat16);
  return aotriton::DType::kUnknown;
#undef CAST_TYPE
}

// actual fwd implementation, calling aotriton api directly
void fused_attn_aotriton_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  bool is_training, float scaling_factor, float dropout_probability,
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  void *devPtrQ, void *devPtrK, void *devPtrV, 
  void *devPtrSoftmaxAux, void *devPtrO,
  const uint64_t* devPtrDropoutSeed, const uint64_t* devPtrDropoutOffset,
  //void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  aotriton::DType dtype,
  void *workspace, 
  size_t *workspace_size,
  cudaStream_t stream){

  // Exit to request upper level API to allocate memory if needed
  // Currently aotriton fused attn does not need workspace in fwd pass
  if(workspace==nullptr){
    *workspace_size = 0;
    return;
  }

  std::array<uint64_t, 4> q_stride;
  std::array<uint64_t, 4> k_stride;
  std::array<uint64_t, 4> v_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
  generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  std::array<uint64_t, 4> q_shape{b, h, s_q, d};
  std::array<uint64_t, 4> kv_shape{b, hg, s_kv, d};

  auto q_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrQ), q_shape, q_stride, dtype);
  auto k_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrK), kv_shape, k_stride, dtype);
  auto v_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrV), kv_shape, v_stride, dtype);


  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  auto o_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrO), q_shape, o_stride, dtype);
  auto M_tensor = aotriton::TensorView<2>(
    reinterpret_cast<intptr_t>(devPtrSoftmaxAux), 
    std::array<uint64_t, 2>{b * h, s_q}, 
    std::array<uint64_t, 2>{s_q, 1}, 
    aotriton::DType::kFloat32);
  auto encoded_softmax_tensor = aotriton::TensorView<4>(
    reinterpret_cast<intptr_t>(nullptr), 
    std::array<uint64_t, 4>{0, 0, 0, 0}, 
    std::array<uint64_t, 4>{1, 1, 1, 1}, 
    dtype);
  
  //devPtrDropoutSeed and devPtrDropoutOffset are actually device ptrs
  uint64_t philox_seed, philox_offset;
  if(is_training && dropout_probability > 0.f){
    (void)cudaStreamSynchronize(stream);
    (void)cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  }

  bool nvte_log_aotriton_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_AOTRITON_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_aotriton_config = true;
  }
  if (nvte_log_aotriton_config) {
    std::cout<<std::endl<<"attn_fwd(aotriton): ";
    std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
    std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
    std::cout<<"kv_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d<<"), ";
    std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
    std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    std::cout<<"M_shape: ("<<b*h<<", "<<s_q<<"), ";
    std::cout<<"M_stride: ("<<s_q<<", "<<1<<"), ";
    std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
    std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    std::cout<<"is_training: "<<is_training<<", ";
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed: "<<philox_seed<<", philox_offset: "<<philox_offset<<", ";
    std::cout<<"causal mask: "<<(mask_type==NVTE_CAUSAL_MASK)<<std::endl;
  }
  aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, dtype);
  using aotriton::v2::flash::attn_fwd;
  NVTE_CHECK_CUDA(attn_fwd(q_tensor,
                           k_tensor,
                           v_tensor,
                           empty_bias,
                           scaling_factor,
                           M_tensor,
                           o_tensor,
                           is_training? dropout_probability:0,
                           philox_seed,
                           philox_offset,
                           encoded_softmax_tensor,
                           mask_type==NVTE_CAUSAL_MASK,
                           stream));
}

void fused_attn_aotriton_bwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  float scaling_factor, float dropout_probability, 
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  void* devPtrQ, void* devPtrK, void* devPtrV,
  void* devPtrO, void* devPtrSoftmaxAux, 
  void* devPtrdQ, void* devPtrdK, void* devPtrdV, 
  void* devPtrdO, 
  const uint64_t* devPtrDropoutSeed, 
  const uint64_t* devPtrDropoutOffset,
  aotriton::DType dtype,
  void *workspace,
  size_t *workspace_size,
  cudaStream_t stream) {

  // Exit to request upper level API to allocate memory if needed
  if(workspace==nullptr){
    // CK only requires workspace for lse softmax
    *workspace_size = b*h*s_q*sizeof(float);
    return;
  }
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
  std::array<uint64_t, 4> q_shape{b, h, s_q, d};
  std::array<uint64_t, 4> kv_shape{b, hg, s_kv, d};
  
  // m and workspace are of the same shape and stride
  std::array<uint64_t, 2> m_shape{b * h, s_q};
  std::array<uint64_t, 2> m_stride{s_q, 1};

  // input tensors
  auto q_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrQ), q_shape, q_stride, dtype);
  auto k_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrK), kv_shape, k_stride, dtype);
  auto v_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrV), kv_shape, v_stride, dtype);
  auto o_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrO), q_shape, o_stride, dtype);
  auto do_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrdO), q_shape, o_stride, dtype);
  
  // output tensors
  auto dq_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrdQ), q_shape, q_stride, dtype);
  auto dk_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrdK), kv_shape, k_stride, dtype);
  auto dv_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(devPtrdV), kv_shape, v_stride, dtype);
  
  // auxilary tensors
  auto M_tensor = aotriton::TensorView<2>(reinterpret_cast<intptr_t>(devPtrSoftmaxAux), m_shape, m_stride, aotriton::DType::kFloat32);
  auto wkspace_tensor = aotriton::TensorView<2>(reinterpret_cast<intptr_t>(workspace), m_shape, m_stride, aotriton::DType::kFloat32);

  uint64_t philox_seed, philox_offset;
  if(dropout_probability > 0.f){
    (void)cudaStreamSynchronize(stream);
    (void)cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  }

  bool nvte_log_aotriton_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_AOTRITON_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_aotriton_config = true;
  }
  if (nvte_log_aotriton_config) {
    std::cout<<std::endl<<"attn_bwd(aotriton): ";
    std::cout<<"q_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
    std::cout<<"q_stride: ("<<q_stride[0]<<", "<<q_stride[1]<<", "<<q_stride[2]<<", "<<q_stride[3]<<"), ";
    std::cout<<"kv_shape: ("<<b<<", "<<hg<<", "<<s_kv<<", "<<d<<"), ";
    std::cout<<"k_stride: ("<<k_stride[0]<<", "<<k_stride[1]<<", "<<k_stride[2]<<", "<<k_stride[3]<<"), ";
    std::cout<<"v_stride: ("<<v_stride[0]<<", "<<v_stride[1]<<", "<<v_stride[2]<<", "<<v_stride[3]<<"), ";
    std::cout<<"scaling_factor: "<<scaling_factor<<", ";
    std::cout<<"M_shape: ("<<b*h<<", "<<s_q<<"), ";
    std::cout<<"M_stride: ("<<s_q<<", "<<1<<"), ";
    std::cout<<"o_shape: ("<<b<<", "<<h<<", "<<s_q<<", "<<d<<"), ";
    std::cout<<"o_stride: ("<<o_stride[0]<<", "<<o_stride[1]<<", "<<o_stride[2]<<", "<<o_stride[3]<<"), ";
    std::cout<<"dropout_p: "<<dropout_probability<<", ";
    std::cout<<"philox_seed: "<<philox_seed<<", philox_offset: "<<philox_offset<<", ";
    std::cout<<"causal mask: "<<(mask_type==NVTE_CAUSAL_MASK)<<std::endl;
  }
  aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, dtype);
  using aotriton::v2::flash::attn_bwd;
  NVTE_CHECK_CUDA(attn_bwd(q_tensor,
                           k_tensor,
                           v_tensor,
                           empty_bias,
                           scaling_factor,
                           o_tensor,
                           do_tensor,
                           dq_tensor,
                           dk_tensor,
                           dv_tensor,
                           empty_bias,
                           M_tensor,
                           wkspace_tensor,
                           dropout_probability,
                           philox_seed,
                           philox_offset,
                           mask_type==NVTE_CAUSAL_MASK,
                           stream));
}
#endif // USE_FUSED_ATTN_AOTRITON
}  // namespace fused_attn_rocm

using namespace transformer_engine::fused_attn_rocm;
void fused_attn_aotriton_fwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_QKV, const Tensor* input_Bias, 
  Tensor* output_O, Tensor* output_M, Tensor* output_rng_state,
  const Tensor* input_cu_seqlens,
  const Tensor* input_rng_state,
  Tensor *workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);
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

  //save the input rng state to Aux_CTX_Tensors
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  size_t workspace_size = 0;

  fused_attn_aotriton_fwd_impl(
    b, h, h, max_seqlen, max_seqlen, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    devPtrQ, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(QKV_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

void fused_attn_aotriton_bwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_QKV, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  Tensor* output_dQKV,
  const Tensor* input_cu_seqlens,
  const Tensor* input_M,
  const Tensor* input_rng_state,
  Tensor* workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);
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

  // output tensor
  void *devPtrdQKV = output_dQKV->data.dptr;
  void *devPtrdQ = static_cast<void *>(devPtrdQKV);
  void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);
  
  size_t workspace_size = 0;

  fused_attn_aotriton_bwd_impl(
    b, h, h, max_seqlen, max_seqlen, d,
    attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    devPtrQ, devPtrK, devPtrV, 
    input_O->data.dptr, input_M->data.dptr,
    devPtrdQ, devPtrdK, devPtrdV, 
    input_dO->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(QKV_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

void fused_attn_aotriton_fwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_KV, const Tensor* input_Bias, 
  Tensor* output_O, Tensor* output_M, Tensor* output_rng_state,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_rng_state,
  Tensor *workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);
  //input tensor
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = nvte_dtype_size(Q_type)*h_kv*d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = nvte_dtype_size(Q_type) * d;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

  //save the input rng state to Aux_CTX_Tensors
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  size_t workspace_size = 0;

  fused_attn_aotriton_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

void fused_attn_aotriton_bwd_kvpacked(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_KV, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  Tensor* output_dQ, Tensor* output_dKV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_M,
  const Tensor* input_rng_state,
  Tensor* workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);
  //input tensor
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = nvte_dtype_size(Q_type) * h_kv * d;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = nvte_dtype_size(Q_type) * d;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

  // output tensor
  void *devPtrdKV = output_dKV->data.dptr;
  void *devPtrdK = devPtrdKV;
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdKV) + stride);

  size_t workspace_size = 0;

  fused_attn_aotriton_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, devPtrK, devPtrV, 
    input_O->data.dptr, input_M->data.dptr,
    output_dQ->data.dptr, devPtrdK, devPtrdV, 
    input_dO->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

void fused_attn_aotriton_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_Bias, 
  Tensor* output_O, Tensor* output_M, Tensor* output_rng_state,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_rng_state,
  Tensor *workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  //save the input rng state to Aux_CTX_Tensors
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  size_t workspace_size = 0;

  fused_attn_aotriton_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

void fused_attn_aotriton_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_M,
  const Tensor* input_rng_state,
  Tensor* workspace,
  cudaStream_t stream){

#ifdef USE_FUSED_ATTN_AOTRITON
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  size_t workspace_size = 0;

  fused_attn_aotriton_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, 
    input_O->data.dptr, input_M->data.dptr,
    output_dQ->data.dptr, output_dK->data.dptr, output_dV->data.dptr, 
    input_dO->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
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
  NVTE_ERROR("AOTriton backend not compiled.");
#endif // USE_FUSED_ATTN_AOTRITON
}

}  // namespace transformer_engine
