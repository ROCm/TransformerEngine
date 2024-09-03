/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <string>
#include <ck_fused_attn/ck_fused_attn.hpp>
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_ck.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

// check the fused attn config to see whether it's ck backend supported
bool is_ck_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim) {

  //ck fused attn does not support gqa mode now
  if(num_attn_heads!=num_gqa_groups){
    return false;
  }

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only MI300X supported
  if(!(sm_arch_name_.find("gfx942")!=std::string::npos)){
    return false;
  }
  
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    return false;
  }
  
  //Only BSHD, SBHD style layouts supported
  if(!((nvte_get_qkv_format(qkv_layout)!= NVTE_QKV_Format::NVTE_SBHD)||
    (nvte_get_qkv_format(qkv_layout)!= NVTE_QKV_Format::NVTE_BSHD))){
    return false;
  }
  
  // AOTriton does not support bias now
  if(!(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)){
    return false;
  }

  // Only no mask and causal mask supported
  if(!((attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)||
    (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK))){
    return false;
  } 
  
  // causal does not work with s_q != s_kv
  if((max_seqlen_q!=max_seqlen_kv)&&(attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)){
    return false;
  }
  return true;
}


ck_fused_attn::DType nvte_to_ck_dtype(NVTEDType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == NVTEDType::aname) return ck_fused_attn::DType::dtname
  CAST_TYPE(kNVTEFloat16, kFloat16);
  CAST_TYPE(kNVTEBFloat16, kBFloat16);
  return ck_fused_attn::DType::kNumTypes;
#undef CAST_TYPE
}

// actual fwd implementation, calling ck api directly
void fused_attn_ck_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  bool is_training, float scaling_factor, float dropout_probability,
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  void *devPtrQ, void *devPtrK, void *devPtrV, 
  void *devPtrSoftmaxAux, void *devPtrO,
  const uint64_t* devPtrDropoutSeed, const uint64_t* devPtrDropoutOffset,
  //void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  ck_fused_attn::DType dtype,
  void *workspace, 
  size_t *workspace_size,
  cudaStream_t stream){

  // Exit to request upper level API to allocate memory if needed
  // Currently ck fused attn does not need workspace in fwd pass
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

  std::array<uint64_t, 4> o_stride;
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                        layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  //devPtrDropoutSeed and devPtrDropoutOffset are actually device ptrs
  uint64_t philox_seed, philox_offset;
  //skip this synchronization if dropout is not needed
  if(is_training && dropout_probability > 0.f){
    cudaStreamSynchronize(stream);
    cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  }

  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }
  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_fwd(ck): ";
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
  using ck_fused_attn::ck_attn_fwd;
  NVTE_CHECK_CUDA(
    ck_attn_fwd(
      dtype,
      b, h, hg, s_q, s_kv, d,
      devPtrQ, 
      q_stride[0], q_stride[1], q_stride[2],
      devPtrK, 
      k_stride[0], k_stride[1], k_stride[2],
      devPtrV, 
      v_stride[0], v_stride[1], v_stride[2],
      is_training, scaling_factor, dropout_probability,
      philox_seed, philox_offset,
      mask_type==NVTE_CAUSAL_MASK, //is_causal
      devPtrO,
      o_stride[0], o_stride[1], o_stride[2],
      devPtrSoftmaxAux,
      stream));
}

size_t ck_dtype_size(ck_fused_attn::DType t_dtype){
  switch(t_dtype){
    case ck_fused_attn::DType::kFloat16: 
      return 2;
    case ck_fused_attn::DType::kBFloat16: 
      return 2;
    default:
      return 1;
  }
  return 1;
}

void fused_attn_ck_bwd_impl(
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
  ck_fused_attn::DType dtype,
  void *workspace,
  size_t *workspace_size,
  cudaStream_t stream) {
  
  // Exit to request upper level API to allocate memory if needed
  if(workspace==nullptr){
    size_t workspace_size_lse = b*h*s_q*sizeof(float);
    // CK requires dq_acc ptr
    size_t workspace_size_dq_acc = b*h*s_q*d*sizeof(float);
    *workspace_size = workspace_size_lse + workspace_size_dq_acc;
    return;
  }

  //ck bwd requires initialize dq since ck uses atomic operations
  //TODO: remove the memset afer ck fixes the atomic operations
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
  if((layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) or (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D)){
    // just memset all dq, dk, dv
    cudaMemsetAsync(devPtrdQ, 0, ck_dtype_size(dtype)*b*h*s_q*d*3, stream);
  }else{
    // HD_2HD, HD_H2D, HD_HD_HD can just memset dq itself
    cudaMemsetAsync(devPtrdQ, 0, ck_dtype_size(dtype)*b*h*s_q*d, stream);
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
  
  uint64_t philox_seed, philox_offset;
  if(dropout_probability > 0.f){
    cudaStreamSynchronize(stream);
    cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  }
  // First b*h*sq*sizeof(float) in workspace are for lse
  // The remaining are for dq_acc_ptr
  void* dq_acc_ptr = static_cast<void *>(static_cast<int8_t*>(workspace) + b*h*s_q*sizeof(float));
  // like dq, dq_acc mem also requires zeroing out
  //dq_acc is of shape (B, S, H, D)
  NVTE_CHECK_CUDA(cudaMemsetAsync(dq_acc_ptr, 0, sizeof(float)*b*h*s_q*d, stream));

  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }
  if (nvte_log_ck_config) {
    std::cout<<std::endl<<"attn_bwd(ck): ";
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
  using ck_fused_attn::ck_attn_bwd;
  NVTE_CHECK_CUDA(
    ck_attn_bwd(
      dtype,
      b, h, hg, s_q, s_kv, d,
      devPtrQ,
      q_stride[0], q_stride[1], q_stride[2],
      devPtrK,
      k_stride[0], k_stride[1], k_stride[2],
      devPtrV,
      v_stride[0], v_stride[1], v_stride[2],
      devPtrO,
      o_stride[0], o_stride[1], o_stride[2],
      devPtrSoftmaxAux,
      devPtrdO,
      o_stride[0], o_stride[1], o_stride[2], //dO and O share the same stride
      scaling_factor, dropout_probability,
      philox_seed, philox_offset,
      mask_type==NVTE_CAUSAL_MASK, // is causal
      devPtrdQ,
      q_stride[0], q_stride[1], q_stride[2], //dQ and Q share the same stride
      dq_acc_ptr, 
      devPtrdK,
      k_stride[0], k_stride[1], k_stride[2], //dK and K share the same stride
      devPtrdV,
      v_stride[0], v_stride[1], v_stride[2], //dV and V share the same stride
      workspace,
      stream));
}
}  // namespace fused_attn_rocm

using namespace transformer_engine::fused_attn_rocm;
void fused_attn_ck_fwd_qkvpacked(
  size_t b, size_t h, size_t max_seqlen, size_t d,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  const Tensor* input_QKV, const Tensor* input_Bias, 
  Tensor* output_O, Tensor* output_M, Tensor* output_rng_state,
  const Tensor* input_cu_seqlens,
  const Tensor* input_rng_state,
  Tensor *workspace,
  cudaStream_t stream){

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

  fused_attn_ck_fwd_impl(
    b, h, h, max_seqlen, max_seqlen, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    devPtrQ, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_ck_dtype(QKV_type),
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
}

void fused_attn_ck_bwd_qkvpacked(
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

  fused_attn_ck_bwd_impl(
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
    nvte_to_ck_dtype(QKV_type),
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
}

void fused_attn_ck_fwd_kvpacked(
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

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_ck_dtype(Q_type),
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
}

void fused_attn_ck_bwd_kvpacked(
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

  fused_attn_ck_bwd_impl(
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
    nvte_to_ck_dtype(Q_type),
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
}

void fused_attn_ck_fwd(
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

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  //save the input rng state to Aux_CTX_Tensors
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  size_t workspace_size = 0;

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_ck_dtype(Q_type),
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
}

void fused_attn_ck_bwd(
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
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  size_t workspace_size = 0;

  fused_attn_ck_bwd_impl(
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
    nvte_to_ck_dtype(Q_type),
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
}

}  // namespace transformer_engine
