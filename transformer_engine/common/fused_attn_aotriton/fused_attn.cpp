/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <iostream>
#include <string>

aotriton::DType nvte_to_aotriton_dtype(NVTEDType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == NVTEDType::aname) return aotriton::DType::dtname
  CAST_TYPE(kNVTEByte, kUInt8);
  CAST_TYPE(kNVTEFloat32, kFloat32);
  CAST_TYPE(kNVTEFloat16, kFloat16);
  CAST_TYPE(kNVTEBFloat16, kBFloat16);
  return aotriton::DType::kUnknown;
#undef CAST_TYPE
}

size_t nvte_dtype_size(NVTEDType t_dtype){
  switch(t_dtype){
    case NVTEDType::kNVTEByte: 
      return 1;
    case NVTEDType::kNVTEInt32: 
      return 4;
    case NVTEDType::kNVTEInt64: 
      return 8;
    case NVTEDType::kNVTEFloat32: 
      return 4;
    case NVTEDType::kNVTEFloat16: 
      return 2;
    case NVTEDType::kNVTEBFloat16: 
      return 2;
    case NVTEDType::kNVTEFloat8E4M3: 
    case NVTEDType::kNVTEFloat8E5M2: 
      return 1;
    default:
      return 1;
  }
  return 1;
}

// map NVTE_QKV_Layout to NVTE_QKV_Layout_Group
NVTE_QKV_Layout_Group nvte_get_qkv_layout_group(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_T3HD:
      return NVTE_QKV_Layout_Group::NVTE_3HD;
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_TH3D:
      return NVTE_QKV_Layout_Group::NVTE_H3D;
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
      return NVTE_QKV_Layout_Group::NVTE_HD_2HD;
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
      return NVTE_QKV_Layout_Group::NVTE_HD_H2D;
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
      return NVTE_QKV_Layout_Group::NVTE_HD_HD_HD;
    default:
      NVTE_ERROR("qkv_layout not supported!");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format
NVTE_QKV_Format nvte_get_qkv_format(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Layout::NVTE_T3HD:
    case NVTE_QKV_Layout::NVTE_TH3D:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
      return NVTE_QKV_Format::NVTE_THD;
    default:
      NVTE_ERROR("qkv_layout not supported!");
  }
}

enum NVTE_QKV_Matrix {
  NVTE_Q_Matrix            = 0,  // queries
  NVTE_K_Matrix            = 1,  // keys
  NVTE_V_Matrix            = 2,  // values
  NVTE_O_Matrix            = 3,  // final output
};

// get matrix strides based on matrix type
void generateMatrixStrides(
            uint64_t b, uint64_t h,
            uint64_t s_q, uint64_t s_kv,
            uint64_t d, uint64_t* stride,
            NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
    // AOTriton internally takes BHSD for implementation
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    switch (layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = 3 * d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = 2 * d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_T3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_q * 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_TH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_q * 3 * h * d;
                     stride[head_dim_idx] = 3 * d;
                     stride[seqlen_dim_idx] = 3 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = 2 * d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_kv * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
    }
}
// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
        NVTEDType q_dtype,
        NVTEDType kv_dtype,
        NVTE_QKV_Layout qkv_layout,
        NVTE_Bias_Type bias_type,
        NVTE_Mask_Type attn_mask_type,
        float dropout,
        size_t num_attn_heads, size_t num_gqa_groups,
        size_t max_seqlen_q, size_t max_seqlen_kv,
        size_t head_dim) {
  using namespace transformer_engine;
  
  //aotriton fused attn does not support gqa mode now
  if(num_attn_heads!=num_gqa_groups){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only MI250 or MI300X supported
  if(!((sm_arch_name_.find("gfx942")!=std::string::npos) || (sm_arch_name_.find("gfx90a")!=std::string::npos))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  
  //Only BSHD, SBHD style layouts supported
  if(!((nvte_get_qkv_format(qkv_layout)!= NVTE_QKV_Format::NVTE_SBHD)||
    (nvte_get_qkv_format(qkv_layout)!= NVTE_QKV_Format::NVTE_BSHD))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  
  // AOTriton does not support bias now
  if(!(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  // Only no mask and causal mask supported
  if(!((attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)||
    (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  } 
  
  // causal does not work with s_q != s_kv
  if((max_seqlen_q!=max_seqlen_kv)&&(attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  return NVTE_Fused_Attn_Backend::NVTE_AOTriton;
}

// actual fwd implementation, calling aotriton api directly
void fused_attn_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  bool is_training, float scaling_factor, float dropout_probability,
  NVTE_QKV_Layout layout,
  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
  void *devPtrQ, void *devPtrK, void *devPtrV, 
  void *devPtrSoftmaxAux, void *devPtrO,
  const uint64_t* devPtrDropoutSeed, const uint64_t* devPtrDropoutOffset,
  //void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
  aotriton::DType dtype,
  //void *workspace, size_t *workspace_size,
  cudaStream_t stream){

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
  cudaStreamSynchronize(stream);
  cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  bool nvte_log_aotriton_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_AOTRITON_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_aotriton_config = true;
  }
  if (nvte_log_aotriton_config) {
    std::cout<<std::endl<<"attn_fwd: ";
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

void fused_attn_bwd_impl(
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
  cudaStream_t stream) {

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
  
  // m and wkspace are of the same shape and stride
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
  cudaStreamSynchronize(stream);
  cudaMemcpy(&philox_seed, devPtrDropoutSeed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&philox_offset, devPtrDropoutOffset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  bool nvte_log_aotriton_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_AOTRITON_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_aotriton_config = true;
  }
  if (nvte_log_aotriton_config) {
    std::cout<<std::endl<<"attn_bwd: ";
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
// NVTE fused attention FWD with packed QKV
void nvte_fused_attn_fwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_CTX_Tensors,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            size_t max_seqlen,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *output_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
  Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);

  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    h = input_QKV->data.shape[ndim - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    h = input_QKV->data.shape[ndim - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_qkvpacked only supports H3D and 3HD layouts!");
  }
  size_t d = input_QKV->data.shape[ndim - 1];

  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h, h, max_seqlen, max_seqlen, d);

  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }

  // write input into q, k, v
  void *devPtrQKV = input_QKV->data.dptr;
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

  fused_attn_fwd_impl(
    b, h, h, max_seqlen, max_seqlen, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    devPtrQ, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(QKV_type),
    stream); 
}

// NVTE fused attention BWD with packed QKV
void nvte_fused_attn_bwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens,
            size_t max_seqlen,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    h = input_QKV->data.shape[ndim - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    h = input_QKV->data.shape[ndim - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_qkvpacked only supports H3D and 3HD layouts!");
  }
  size_t d = input_QKV->data.shape[ndim - 1];

  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h, h, max_seqlen, max_seqlen, d);

  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
  //input tensor
  void *devPtrQKV = input_QKV->data.dptr;
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
  
  // auxiliary tensors
  const Tensor *input_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  //extract the saved rng state from aux_ctx_tensor
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);

  fused_attn_bwd_impl(
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
    wkspace->data.dptr,
    stream);

}

// NVTE fused attention FWD with packed KV
void nvte_fused_attn_fwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_CTX_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *output_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
  Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);

  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  auto ndim = input_Q->data.shape.size();
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];
  auto ndim_kv = input_KV->data.shape.size();
  size_t h_kv = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    h_kv = input_KV->data.shape[ndim_kv - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    h_kv = input_KV->data.shape[ndim_kv - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_kvpacked only supports HD_H2D and HD_2HD layouts!");
  }
  
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);
  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }

  //input tensor
  void *devPtrKV = input_KV->data.dptr;
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

  fused_attn_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, devPtrK, devPtrV, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
    stream); 
}
// NVTE fused attention BWD with packed KV
void nvte_fused_attn_bwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dKV = reinterpret_cast<Tensor*>(dKV);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  auto ndim = input_Q->data.shape.size();
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];
  auto ndim_kv = input_KV->data.shape.size();
  size_t h_kv = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    h_kv = input_KV->data.shape[ndim_kv - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    h_kv = input_KV->data.shape[ndim_kv - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_kvpacked only supports HD_H2D and HD_2HD layouts!");
  }

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }

  //input tensor
  void *devPtrKV = input_KV->data.dptr;
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

  // auxiliary tensors (to be propagated to the backward pass later)
  const Tensor *input_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);

  fused_attn_bwd_impl(
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
    wkspace->data.dptr,
    stream);
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(
            const NVTETensor Q,
            const NVTETensor K,
            const NVTETensor V,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_CTX_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_K = reinterpret_cast<const Tensor*>(K);
  const Tensor *input_V = reinterpret_cast<const Tensor*>(V);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *output_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
  Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  //auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }

  //save the input rng state to Aux_CTX_Tensors
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  fused_attn_fwd_impl(
    b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
    is_training, attn_scale, dropout, 
    qkv_layout,
    bias_type, attn_mask_type,
    input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, 
    output_M->data.dptr, output_O->data.dptr,
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr), 
    reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1,
    nvte_to_aotriton_dtype(Q_type),
    stream);
}

// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(
            const NVTETensor Q,
            const NVTETensor K,
            const NVTETensor V,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dK,
            NVTETensor dV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_K = reinterpret_cast<const Tensor*>(K);
  const Tensor *input_V = reinterpret_cast<const Tensor*>(V);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dK = reinterpret_cast<Tensor*>(dK);
  Tensor *output_dV = reinterpret_cast<Tensor*>(dV);
  const Tensor *input_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

  if (fused_attention_backend != NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }

  fused_attn_bwd_impl(
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
    wkspace->data.dptr,
    stream);
}
