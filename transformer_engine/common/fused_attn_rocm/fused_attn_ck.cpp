/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

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
bool is_ck_backend_supported(
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

#ifdef USE_FUSED_ATTN_CK
  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }

  if(num_attn_heads%num_gqa_groups != 0){
    if(nvte_log_ck_config){
      std::cout<<"Num of attention heads must be divided by num of gqa groups"<<std::endl;
    }
    return false;
  }

  //swa filter
  if(attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK || attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK){
    // causal mask window must be with causal top left or causal bottom right mask type
    if (!((window_size_left ==-1 || window_size_left >=0) && window_size_right ==0 )){
      if(nvte_log_ck_config){
        std::cout<<"When mask contains causal, window size should be (-1, 0) or (>=0, 0)"<<std::endl;
      }
      return false;
    }
  }else if(attn_mask_type==NVTE_Mask_Type::NVTE_NO_MASK){
    // no mask must be with either (-1, -1) or (>=0, >=0)
    if (!((window_size_left == -1 && window_size_right == -1)||(window_size_left >= 0 && window_size_right >= 0))){
      if(nvte_log_ck_config){
        std::cout<<"When no mask, window size should be (-1, -1) or (>=0, >=0)"<<std::endl;
      }
      return false;
    }
  }

  bool is_mqa_gqa = num_attn_heads > num_gqa_groups;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);

  bool is_qkvpacked = layout_group==NVTE_QKV_Layout_Group::NVTE_3HD ||layout_group==NVTE_QKV_Layout_Group::NVTE_H3D;

  // MQA/GQA does not work with qkvpacked layout
  if(is_mqa_gqa && is_qkvpacked){
    if(nvte_log_ck_config){
      std::cout<<"When no mask, window size should be (-1, -1) or (>=0, >=0)"<<std::endl;
    }
    return false;
  }
  
  // qkvpacked layout requires seq length to be the same
  if(is_qkvpacked && max_seqlen_q!=max_seqlen_kv){
    if(nvte_log_ck_config){
      std::cout<<"qkv packed layout requires seqlen_q==seqlen_kv"<<std::endl;
    }
    return false;
  }

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only MI300X supported
  if(!(sm_arch_name_.find("gfx942")!=std::string::npos)){
    if(nvte_log_ck_config){
      std::cout<<"only MI300X is supported"<<std::endl;
    }
    return false;
  }
  
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    if(nvte_log_ck_config){
      std::cout<<"q, k, v data type has to be fp16 or bf16"<<std::endl;
    }
    return false;
  }
  
  //Only BSHD, SBHD style layouts supported
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if(!(qkv_format == NVTE_QKV_Format::NVTE_SBHD||
    qkv_format == NVTE_QKV_Format::NVTE_BSHD)){
    if(nvte_log_ck_config){
      std::cout<<"qkv format can only be BSHD or SBHD"<<std::endl;
    }
    return false;
  }
  
  // CK does not support pre_scale bias
  if(!(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS || bias_type == NVTE_Bias_Type::NVTE_ALIBI || bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)){
    if(nvte_log_ck_config){
      std::cout<<"CK fused attn does not support pre_scale bias"<<std::endl;
    }
    return false;
  }

  // Only no mask and causal (top left) and causal bottom right mask supported
  // TODO: support padding mask in CK
  if(!(attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
    attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
    attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK)){
    if(nvte_log_ck_config){
      std::cout<<"CK fused attn only support no_mask, causal_mask, causal_bottom_right_mask"<<std::endl;
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
  if (nvte_mask_type==NVTE_Mask_Type::NVTE_NO_MASK){
    // window size in NVTE_NO_Mask can be (-1, -1) and (>=0, >=0)
    if(nvte_window_size_left==-1 && nvte_window_size_right==-1){
      // (-1, -1)
      return ck_fused_attn::MaskType::no_mask;
    }else{
      // (>=0, >=0)
      return ck_fused_attn::MaskType::mask_top_left;
    }
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK){
    // nvte causal mask can map to (-1, 0) or (>=0, 0)
    return ck_fused_attn::MaskType::mask_top_left;
  }else if (nvte_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK){
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

// actual fwd implementation, calling ck api directly
void fused_attn_ck_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
  bool is_training, float scaling_factor, float dropout_probability,
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void *devPtrQ, void *devPtrK, void *devPtrV, void* devPtrBias,
  void *devPtrSoftmaxAux, void *devPtrO,
  void* devPtrDropoutSeed, void* devPtrDropoutOffset,
  //void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  ck_fused_attn::DType dtype,
  void *workspace, 
  size_t *workspace_size,
  cudaStream_t stream){

  bool nvte_log_ck_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_CK_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_ck_config = true;
  }
  // Exit to request upper level API to allocate memory if needed
  // Currently ck fused attn does not need workspace in fwd pass
  if(workspace==nullptr){
    *workspace_size = 0;
    // ck requires an alibi slope array even if in standard (vanilla) mode
    if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
      (*workspace_size)+= h*sizeof(float);
    }

    if (nvte_log_ck_config) {
      std::cout<<std::endl<<"attn_fwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
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

  void* devPtrAlibiSlope = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    devPtrAlibiSlope = workspace;
    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    //assign standard alibi slope
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
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
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<std::endl;
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<std::endl;
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<")"<<std::endl;
  }
  using ck_fused_attn::ck_attn_fwd;
  NVTE_CHECK_CUDA(
    ck_attn_fwd(
      dtype,
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
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d, uint64_t bias_b, uint64_t bias_h,
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
  ck_fused_attn::DType dtype,
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
  // Exit to request upper level API to allocate memory if needed
  if(workspace==nullptr){
    size_t workspace_size_lse = b*h*s_q*sizeof(float);
    // CK requires dq_acc ptr, dq_acc depends on is deterministic
    size_t workspace_size_dq_acc = nsplits*b*h*s_q*d*sizeof(float);
    *workspace_size = workspace_size_lse + workspace_size_dq_acc;
    if(is_mqa_gqa){
      // allocate dk, dv (or dkv) as if h=hg
      size_t dkv_expanded_size = 2*b*h*s_kv*d*ck_dtype_size(dtype);
      *workspace_size += dkv_expanded_size;
    }
    // ck requires an alibi slope array even if in standard (vanilla) mode
    if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
      (*workspace_size)+= h*sizeof(float);
    }else if ((bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) && (bias_b!=b or bias_h!=h)){
      //ck requires a buffer dbias_expanded of size BHSS if bias is not BHSS
      (*workspace_size) += b*h*s_q*s_kv*ck_dtype_size(dtype);
    }
    if (nvte_log_ck_config) {
      std::cout<<std::endl<<"attn_bwd(ck) requested workspace of size "<<*workspace_size<<std::endl;
    }
    return;
  }

  //ck bwd requires initialize dq since ck uses atomic operations
  //TODO: remove the memset afer ck fixes the atomic operations
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
  if((layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) or (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D)){
    // just memset all dq, dk, dv
    (void)cudaMemsetAsync(devPtrdQ, 0, ck_dtype_size(dtype)*b*h*s_q*d*3, stream);
  }else{
    // HD_2HD, HD_H2D, HD_HD_HD can just memset dq itself
    (void)cudaMemsetAsync(devPtrdQ, 0, ck_dtype_size(dtype)*b*h*s_q*d, stream);
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
  
  // First b*h*sq*sizeof(float) in workspace are for lse
  // The next section are for dq_acc_ptr
  void* dq_acc_ptr = static_cast<void *>(static_cast<int8_t*>(workspace) + b*h*s_q*sizeof(float));
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
    dk_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dq_acc_ptr) + nsplits*b*h*s_q*d*sizeof(float));

    //dv_expanded_ptr depends on the actual layout
    if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + ck_dtype_size(dtype)*h*d);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + ck_dtype_size(dtype)*d);
    } else if(layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD){
      dv_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + ck_dtype_size(dtype)*b*h*s_kv*d);
    } else{
      NVTE_ERROR("NVTE_3HD NVTE_H3D should have h=hg.");
    }
    // zeroing out dkv expanded in case CK requires that
    NVTE_CHECK_CUDA(cudaMemsetAsync(dk_expanded_ptr, 0, 2*ck_dtype_size(dtype)*b*h*s_kv*d, stream));
  }

  void* devPtrAlibiSlope = nullptr;
  void* dbias_expanded_ptr = nullptr;
  if(bias_type == NVTE_Bias_Type::NVTE_ALIBI){
    // alibi slope is the last section in the workspace buffer
    if(is_mqa_gqa){
      devPtrAlibiSlope = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + 2*b*h*s_kv*d*ck_dtype_size(dtype));
    }else{
      // devPtrAlibiSlope at the end of dq_acc_ptr if no mqa/gqa temp buffer needed
      devPtrAlibiSlope = static_cast<void *>(static_cast<int8_t*>(dq_acc_ptr) + nsplits*b*h*s_q*d*sizeof(float));
    }

    dim3 block, grid;
    block.x = 1024;
    grid.x = ceil(h/1024.);
    //assign standard alibi slope
    hipLaunchKernelGGL(generate_alibi_slope, grid, block, 0, stream, h, static_cast<float*>(devPtrAlibiSlope));
  }else if((bias_type==NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) && (devPtrdBias!=nullptr)){
    if(bias_b!=b or bias_h!= h){
      // dbias_expanded_ptr is the last section in the workspace buffer
      if(is_mqa_gqa){
        dbias_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dk_expanded_ptr) + 2*b*h*s_kv*d*ck_dtype_size(dtype));
      }else{
        // dbias_expanded_ptr at the end of dq_acc_ptr if no mqa/gqa temp buffer needed
        dbias_expanded_ptr = static_cast<void *>(static_cast<int8_t*>(dq_acc_ptr) + nsplits*b*h*s_q*d*sizeof(float));
      }
      // zeroing out dbias_expanded_ptr as CK requires that
      NVTE_CHECK_CUDA(cudaMemsetAsync(dbias_expanded_ptr, 0, ck_dtype_size(dtype)*b*h*s_q*s_kv, stream));
    }else{
      // dbias_expanded_ptr not needed for BHSS shape
      NVTE_CHECK_CUDA(cudaMemsetAsync(devPtrdBias, 0, ck_dtype_size(dtype)*bias_b*bias_h*s_q*s_kv, stream));
    }
  }
  
  // bwd v3 is optional by enabling the following envs
  // default values follows the ck example setting
  bool nvte_ck_uses_bwd_v3 = getenv<int>("NVTE_CK_USES_BWD_V3", 0);
  bool nvte_ck_is_v3_atomic_fp32 = getenv<int>("NVTE_CK_IS_V3_ATOMIC_FP32", 1);
  bool nvte_ck_is_v3_spec = getenv<int>("NVTE_CK_IS_V3_SPEC", 0);
  int nvte_ck_how_v3_bf16_cvt = getenv<int>("NVTE_CK_HOW_V3_BF16_CVT", 1);

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
    std::cout<<"philox_seed_ptr: "<<devPtrDropoutSeed<<", philox_offset_ptr: "<<devPtrDropoutOffset<<", ";
    std::cout<<"bias_type: "<<bias_type<<std::endl;
    std::cout<<"(bias_b, bias_h): ("<<bias_b<<", "<<bias_h<<"), ";
    std::cout<<"mask_type: "<<mask_type<<std::endl;
    std::cout<<"window_size: ("<<window_size_left<<", "<<window_size_right<<")"<<std::endl;
    std::cout<<"deterministic: "<<deterministic<<std::endl;
  }
  using ck_fused_attn::ck_attn_bwd;
  NVTE_CHECK_CUDA(
    ck_attn_bwd(
      dtype,
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
      workspace,
      deterministic,
      nvte_ck_uses_bwd_v3,
      nvte_ck_is_v3_atomic_fp32,
      nvte_ck_is_v3_spec,
      nvte_ck_how_v3_bf16_cvt,
      stream));
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

  fused_attn_ck_fwd_impl(
    b, h, h, max_seqlen, max_seqlen, d, bias_b, bias_h,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    devPtrQ, devPtrK, devPtrV, devPtrBias,
    devPtrS, devPtrO,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
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
  
  size_t workspace_size = 0;

  fused_attn_ck_bwd_impl(
    b, h, h, max_seqlen, max_seqlen, d, bias_b, bias_h,
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

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    devPtrQ, devPtrK, devPtrV, devPtrBias,
    devPtrS, devPtrO,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
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

  size_t workspace_size = 0;

  fused_attn_ck_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
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

  fused_attn_ck_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    window_size_left, window_size_right,
    devPtrQ, devPtrK, devPtrV, devPtrBias, 
    devPtrS, devPtrO,
    rng_state->data.dptr, 
    reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1),
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

  size_t workspace_size = 0;

  fused_attn_ck_bwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, bias_b, bias_h,
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
#else
  NVTE_ERROR("CK fused attn backend not compiled.");
#endif // USE_FUSED_ATTN_CK
}

}  // namespace transformer_engine
