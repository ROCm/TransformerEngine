/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <string>
#include <tuple>
#include "transformer_engine/fused_attn.h"
#include "fused_attn_aotriton.h"
#include "fused_attn_ck.h"
#include "../common.h"

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

// check if sliding window size is compliant with attention mask type
// Follow from upstream NVTE pytorch check_set_window_size (https://github.com/NVIDIA/TransformerEngine/blob/7b284fef07cd3093d2670142c67cdd548828634b/transformer_engine/pytorch/attention.py#L5129)
//         attn_mask_type                              |   window_size
//    -------------------------------------------------------------------------
//    no_mask, padding, arbitrary                      | (-1, -1) or (>=0, >=0)
//    causal, padding_causal                           | (-1,  0) or (>=0, 0)
//    causal_bottom_right, padding_causal_bottom_right | (-1,  0) or (>=0, 0)
std::pair<int64_t, int64_t> check_set_window_size(NVTE_Mask_Type attn_mask_type, std::pair<int64_t, int64_t> window_size){
  //mask_type contain causal
  if(attn_mask_type==NVTE_CAUSAL_MASK || attn_mask_type==NVTE_PADDING_CAUSAL_MASK || attn_mask_type==NVTE_CAUSAL_BOTTOM_RIGHT_MASK || attn_mask_type==NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK){
    if(window_size==std::make_pair<int64_t, int64_t>(-1, -1) || (window_size.first >=0 && window_size.second!=0)){
      //TODO: better INFO logging
      std::cout<<"window_size should be (-1, 0) or (>=0, 0) for attn_mask_type="<<attn_mask_type<<std::endl;
      window_size.second = 0;
      return window_size;
    }else if( window_size!=std::make_pair<int64_t, int64_t>(-1, 0) && (window_size.first < 0 || window_size.second != 0)){
      NVTE_ERROR("window_size should be (-1, 0) or (>=0, 0) for attn_mask_type=" + std::to_string(attn_mask_type));
    }
  }else if(attn_mask_type==NVTE_NO_MASK || attn_mask_type==NVTE_PADDING_MASK){
    //no_mask and padding mask
    if(window_size==std::make_pair<int64_t, int64_t>(-1, 0)){
      //TODO: better INFO logging
      std::cout<<"window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type="<<attn_mask_type<<std::endl;
      window_size.second=-1;
      return window_size;
    }else if(window_size!=std::make_pair<int64_t, int64_t>(-1, -1) && (window_size.first < 0 or window_size.second < 0)){
      NVTE_ERROR("window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type=" + std::to_string(attn_mask_type)); 
    }
  }else{
    NVTE_ERROR("Invalid attn_mask_type: " + std::to_string(attn_mask_type));
  }
  return window_size;
}

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, float dropout, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v,
    int64_t window_size_left, int64_t window_size_right) {
  using namespace transformer_engine;
  
  // by default, fused attn is enabled
  bool nvte_fused_attn = true;
  if (const char* env_p = std::getenv("NVTE_FUSED_ATTN") ) {
    if (env_p != nullptr && std::string(env_p) == "0")
      nvte_fused_attn = false;
  }

  // by default, both ck and aotriton backends are enabled by nvte_fused_attn
  bool nvte_fused_attn_ck = nvte_fused_attn;
  bool nvte_fused_attn_aotriton = nvte_fused_attn;

  if (const char* env_p = std::getenv("NVTE_FUSED_ATTN_CK") ) {
    if (env_p != nullptr && std::string(env_p) == "0")
      nvte_fused_attn_ck = false;
  }
  if (const char* env_p = std::getenv("NVTE_FUSED_ATTN_AOTRITON") ) {
    if (env_p != nullptr && std::string(env_p) == "0")
      nvte_fused_attn_aotriton = false;
  }
  
  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  // first check whether ck can be used, then check aotriton
  if(nvte_fused_attn_ck && fused_attn_rocm::is_ck_backend_supported(
        q_dtype,
        kv_dtype,
        qkv_layout,
        bias_type,
        attn_mask_type,
        dropout,
        num_attn_heads, num_gqa_groups,
        max_seqlen_q, max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        window_size_left,
        window_size_right)){
    return NVTE_Fused_Attn_Backend::NVTE_CK;
  }else if(nvte_fused_attn_aotriton && fused_attn_rocm::is_aotriton_backend_supported(
              q_dtype,
              kv_dtype,
              qkv_layout,
              bias_type,
              attn_mask_type,
              dropout,
              num_attn_heads, num_gqa_groups,
              max_seqlen_q, max_seqlen_kv,
              head_dim_qk, 
              head_dim_v, 
              window_size_left,
              window_size_right)){
    return NVTE_Fused_Attn_Backend::NVTE_AOTriton;
  }
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
}


// NVTE fused attention FWD with packed QKV
void nvte_fused_attn_fwd_qkvpacked(const NVTETensor QKV, const NVTETensor Bias, NVTETensor S,
                                   NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                                   const NVTETensor cu_seqlens, const NVTETensor cu_seqlens_padded,
                                   const NVTETensor rng_state, size_t max_seqlen, bool is_training,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   int64_t window_size_left, int64_t window_size_right,
                                   NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
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

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      QKV_type, QKV_type, qkv_layout, bias_type, attn_mask_type, dropout, h, h, max_seqlen,
      max_seqlen, d, d, window_size_left, window_size_right);
  
  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    fused_attn_ck_fwd_qkvpacked(
      b, h, max_seqlen, d,
      is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      input_QKV, input_Bias, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    fused_attn_aotriton_fwd_qkvpacked(
      b, h, max_seqlen, d,
      is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
      input_QKV, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}

// NVTE fused attention BWD with packed QKV
void nvte_fused_attn_bwd_qkvpacked(const NVTETensor QKV, const NVTETensor O, const NVTETensor dO,
                                   const NVTETensor S, NVTETensor dP,
                                   const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQKV,
                                   NVTETensor dBias, const NVTETensor cu_seqlens,
                                   const NVTETensor cu_seqlens_padded, size_t max_seqlen,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   int64_t window_size_left, int64_t window_size_right,
                                   bool deterministic, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  Tensor *output_dBias = reinterpret_cast<Tensor *>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // auxiliary tensors
  const Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  //extract the saved rng state from aux_ctx_tensor
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
  Tensor *input_Bias = nullptr;

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

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      QKV_type, QKV_type, qkv_layout, bias_type, attn_mask_type, dropout, h, h, max_seqlen,
      max_seqlen, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    if((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)){
      input_Bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    }
    fused_attn_ck_bwd_qkvpacked(
      b, h, max_seqlen, d,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      false, // TODO: enable deterministic after CK team show us how
      input_QKV, input_O, input_dO, input_Bias, output_S,
      output_dQKV, output_dBias,
      input_cu_seqlens,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    //currently aotriton is deterministic
    fused_attn_aotriton_bwd_qkvpacked(
      b, h, max_seqlen, d,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      input_QKV, input_O, input_dO, output_S,
      output_dQKV,
      input_cu_seqlens,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}

// NVTE fused attention FWD with packed KV
void nvte_fused_attn_fwd_kvpacked(const NVTETensor Q, const NVTETensor KV, const NVTETensor Bias,
                                  NVTETensor S, NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                                  const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                                  const NVTETensor cu_seqlens_q_padded,
                                  const NVTETensor cu_seqlens_kv_padded, const NVTETensor rng_state,
                                  size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                                  float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                  int64_t window_size_left, int64_t window_size_right,
                                  NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
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

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv, max_seqlen_q,
      max_seqlen_kv, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    fused_attn_ck_fwd_kvpacked(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
      is_training, attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      input_Q, input_KV, input_Bias, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    fused_attn_aotriton_fwd_kvpacked(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
      is_training, attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      input_Q, input_KV, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}

// NVTE fused attention BWD with packed KV
void nvte_fused_attn_bwd_kvpacked(
    const NVTETensor Q, const NVTETensor KV, const NVTETensor O, const NVTETensor dO,
    const NVTETensor S, NVTETensor dP, const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ,
    NVTETensor dKV, NVTETensor dBias, const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
    const NVTETensor cu_seqlens_q_padded, const NVTETensor cu_seqlens_kv_padded,
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    int64_t window_size_left, int64_t window_size_right, bool deterministic, NVTETensor workspace,
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
  Tensor *output_dBias = reinterpret_cast<Tensor *>(dBias);

  // auxiliary tensors (to be propagated to the backward pass later)
  const Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
  Tensor *input_Bias = nullptr;

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
    NVTE_ERROR("nvte_fused_attn_bwd_kvpacked only supports HD_H2D and HD_2HD layouts!");
  }

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv, max_seqlen_q,
      max_seqlen_kv, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_Bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    }
    fused_attn_ck_bwd_kvpacked(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      false, // TODO: enable deterministic after CK team show us how
      input_Q, input_KV, input_O, input_dO, input_Bias, 
      output_S,
      output_dQ, output_dKV, output_dBias,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    // currently aotriton is deterministic
    fused_attn_aotriton_bwd_kvpacked(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      input_Q, input_KV, input_O, input_dO, 
      output_S,
      output_dQ, output_dKV,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, NVTETensor S, NVTETensor O,
                         NVTETensorPack *Aux_CTX_Tensors, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         int64_t window_size_left, int64_t window_size_right, NVTETensor workspace,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_K = reinterpret_cast<const Tensor*>(K);
  const Tensor *input_V = reinterpret_cast<const Tensor*>(V);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim - 2];
  size_t d_qk = input_Q->data.shape[ndim - 1];
  size_t d_v = input_V->data.shape[ndim - 1];

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv, max_seqlen_q,
      max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    fused_attn_ck_fwd(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk,
      is_training, attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      input_Q, input_K, input_V, input_Bias, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    fused_attn_aotriton_fwd(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk,
      is_training, attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      input_Q, input_K, input_V, 
      output_O, Aux_CTX_Tensors,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}

// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, int64_t window_size_left,
                         int64_t window_size_right, bool deterministic, NVTETensor workspace,
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
  Tensor *output_dBias = reinterpret_cast<Tensor *>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  const Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
  Tensor *input_Bias = nullptr;

  auto ndim = input_Q->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim - 2];
  size_t d_qk = input_Q->data.shape[ndim - 1];
  size_t d_v = input_V->data.shape[ndim - 1];

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  // fix the incompatible window size from upstream frameworks pytorch/jax
  std::tie(window_size_left, window_size_right) = check_set_window_size(attn_mask_type, std::make_pair(window_size_left, window_size_right));

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv, max_seqlen_q,
      max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_CK) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_Bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    }
    fused_attn_ck_bwd(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      window_size_left, window_size_right,
      false, // TODO: enable deterministic after CK team show us how
      input_Q, input_K, input_V, input_O, input_dO, input_Bias, 
      output_S,
      output_dQ, output_dK, output_dV, output_dBias,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  } else if(fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_AOTriton){
    // currently aotriton bwd is deterministic
    fused_attn_aotriton_bwd(
      b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk,
      attn_scale, dropout, 
      qkv_layout, bias_type, attn_mask_type,
      input_Q, input_K, input_V, input_O, input_dO,
      output_S,
      output_dQ, output_dK, output_dV,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream);
  }else{
    NVTE_ERROR("Invalid combination of data type and sequence length for rocm fused attention. \n");
  }
}
