/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn_aotriton.h"
#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>

aotriton::DType nvte_to_aotriton_dtype(NVTEDType t_dtype){
#define CAST_TYPE(aname, dtname) if (t_dtype == NVTEDType::aname) return aotriton::DType::dtname
  CAST_TYPE(kNVTEByte, kUInt8);
  CAST_TYPE(kNVTEFloat32, kFloat32);
  CAST_TYPE(kNVTEFloat16, kFloat16);
  CAST_TYPE(kNVTEBFloat16, kBFloat16);
  return aotriton::DType::kUnknown;
#undef CAST_TYPE
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
  NVTE_Fused_Attn_Backend backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;

  const int device_id = cuda::current_device();
  const std::string sm_arch_name_ = cuda::sm_arch_name(device_id);
  //only MI250 or MI300X supported
  if(!((sm_arch_name_=="gfx942") || (sm_arch_name_=="gfx90a"))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  
  // Q and KV must have the same data type, in fp16 or bf16
  if((q_dtype!=kv_dtype) || !((q_dtype==NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  
  //Only BSHD layout supported
  if(!((qkv_layout==NVTE_QKV_Layout::NVTE_BS3HD)||
    (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD)||
    (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD))){
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

  return NVTE_Fused_Attn_Backend::NVTE_AOTriton;
}

/*
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

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h, h, max_seqlen, max_seqlen, d);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      fused_attn_max_512_fwd_qkvpacked(
          b, h, max_seqlen, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      fused_attn_arbitrary_seqlen_fwd_qkvpacked(
          b, h, max_seqlen, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.0 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_fp8_fwd_qkvpacked(
            b, h, max_seqlen, d,
            is_training, attn_scale, dropout, qkv_layout,
            input_QKV, input_output_S, output_O,
            Aux_CTX_Tensors,
            input_cu_seqlens,
            input_rng_state,
            wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
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
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
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

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          QKV_type, QKV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h, h, max_seqlen, max_seqlen, d);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_qkvpacked(
          b, h, max_seqlen, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_dO,
          output_S,
          output_dQKV, output_dBias,
          input_cu_seqlens,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      Tensor *input_Bias, *input_rng_state;
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
          input_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
          input_Bias = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      } else {
          input_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      }
      fused_attn_arbitrary_seqlen_bwd_qkvpacked(
          b, h, max_seqlen, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_QKV, input_O, input_dO, input_Bias,
          output_S,
          output_dQKV, output_dBias,
          input_cu_seqlens, input_rng_state,
          wkspace, stream, handle);
#else
    const char *err_msg =
    "cuDNN 8.9.0 is required for BF16/FP16 fused attention "
    "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    const Tensor *input_M = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[2]);
    fused_attn_fp8_bwd_qkvpacked(
                    b, h, max_seqlen, d,
                    attn_scale, dropout, qkv_layout,
                    input_QKV, input_O, input_dO,
                    input_M, input_ZInv,
                    input_S, input_output_dP,
                    output_dQKV,
                    input_cu_seqlens,
                    input_rng_state,
                    wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
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
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
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

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      fused_attn_max_512_fwd_kvpacked(
          b, h_q, max_seqlen_q, max_seqlen_kv, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8903)
      fused_attn_arbitrary_seqlen_fwd_kvpacked(
          b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
          is_training, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_Bias, output_O,
          Aux_CTX_Tensors,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state,
          wkspace, stream, handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.3 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
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
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dKV = reinterpret_cast<Tensor*>(dKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
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

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend =
              nvte_get_fused_attn_backend(
                          Q_type, KV_type,
                          qkv_layout, bias_type, attn_mask_type,
                          dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      fused_attn_max_512_bwd_kvpacked(
          b, h_q, max_seqlen_q, max_seqlen_kv, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_dO,
          output_S,
          output_dQ, output_dKV, output_dBias,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8903)
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      Tensor *input_Bias, *input_rng_state;
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
          input_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
          input_Bias = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      } else {
          input_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      }
      fused_attn_arbitrary_seqlen_bwd_kvpacked(
          b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d,
          attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
          input_Q, input_KV, input_O, input_dO, input_Bias,
          output_S,
          output_dQ, output_dKV, output_dBias,
          input_cu_seqlens_q, input_cu_seqlens_kv,
          input_rng_state, wkspace, stream, handle);
#else
    const char *err_msg =
    "cuDNN 8.9.3 is required for BF16/FP16 fused attention "
    "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
*/
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
  //const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  //Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  //Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

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

  // aotriton takes qkv layout BHSD
  std::array<uint64_t, 4> q_shape{b, h_q, max_seqlen_q, d};
  std::array<uint64_t, 4> q_stride{h_q*max_seqlen_q*d, d, h_q*d, 1};

  std::array<uint64_t, 4> kv_shape{b, h_q, max_seqlen_kv, d};
  std::array<uint64_t, 4> kv_stride{h_q*max_seqlen_kv*d, d, h_q*d, 1};

  aotriton::DType dtype = nvte_to_aotriton_dtype(Q_type);
  
  //input tensor
  auto q_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_Q->data.dptr), q_shape, q_stride, dtype);
  auto k_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_K->data.dptr), kv_shape, kv_stride, dtype);
  auto v_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_V->data.dptr), kv_shape, kv_stride, dtype);

  // output tensors
  auto o_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(output_O->data.dptr), q_shape, q_stride, dtype);

  // auxiliary tensors (to be propagated to the backward pass later)
  Tensor *output_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
  auto M_tensor = aotriton::TensorView<2>(
    reinterpret_cast<intptr_t>(output_M->data.dptr), 
    std::array<uint64_t, 2>{b * h_q, max_seqlen_q}, 
    std::array<uint64_t, 2>{max_seqlen_q, 1}, 
    aotriton::DType::kFloat32);
  auto encoded_softmax_tensor = aotriton::TensorView<4>(
    reinterpret_cast<intptr_t>(nullptr), 
    std::array<uint64_t, 4>{0, 0, 0, 0}, 
    std::array<uint64_t, 4>{1, 1, 1, 1}, 
    dtype);

  uint64_t philox_seed = *(reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr));
  uint64_t philox_offset = *(reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1);
  //save the input rng state to Aux_CTX_Tensors
  Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
  output_rng_state->data.dptr = input_rng_state->data.dptr;

  hipError_t err; // TODO: Error handling
  using aotriton::v2::flash::attn_fwd;
  err = attn_fwd(q_tensor,
                 k_tensor,
                 v_tensor,
                 attn_scale,
                 M_tensor,
                 o_tensor,
                 is_training? dropout:0,
                 philox_seed,
                 philox_offset,
                 encoded_softmax_tensor,
                 attn_mask_type==NVTE_CAUSAL_MASK,
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
  //const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  //Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dK = reinterpret_cast<Tensor*>(dK);
  Tensor *output_dV = reinterpret_cast<Tensor*>(dV);
  //Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
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

  // aotriton takes qkv layout BHSD
  std::array<uint64_t, 4> q_shape{b, h_q, max_seqlen_q, d};
  std::array<uint64_t, 4> q_stride{h_q*max_seqlen_q*d, d, h_q*d, 1};

  std::array<uint64_t, 4> kv_shape{b, h_q, max_seqlen_kv, d};
  std::array<uint64_t, 4> kv_stride{h_q*max_seqlen_kv*d, d, h_q*d, 1};

  aotriton::DType dtype = nvte_to_aotriton_dtype(Q_type);

  // input tensor
  auto q_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_Q->data.dptr), q_shape, q_stride, dtype);
  auto k_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_K->data.dptr), kv_shape, kv_stride, dtype);
  auto v_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_V->data.dptr), kv_shape, kv_stride, dtype);
  auto o_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_O->data.dptr), q_shape, q_stride, dtype);
  auto do_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(input_dO->data.dptr), q_shape, q_stride, dtype);

  // output tensor
  auto dq_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(output_dQ->data.dptr), q_shape, q_stride, dtype);
  auto dk_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(output_dK->data.dptr), kv_shape, kv_stride, dtype);
  auto dv_tensor = aotriton::TensorView<4>(reinterpret_cast<intptr_t>(output_dV->data.dptr), kv_shape, kv_stride, dtype);

  // auxiliary tensors (to be propagated to the backward pass later)
  // M tensor, also known as softmax lse
  std::array<uint64_t, 2> m_shape{b * h_q, max_seqlen_q};
  std::array<uint64_t, 2> m_stride{max_seqlen_q, 1};
  const Tensor *input_M = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]); //softmax lse
  auto M_tensor = aotriton::TensorView<2>(reinterpret_cast<intptr_t>(input_M->data.dptr), m_shape, m_stride, aotriton::DType::kFloat32);
  // aotriton requires wkspace tensor same size as softmax lse
  auto wkspace_tensor = aotriton::TensorView<2>(reinterpret_cast<intptr_t>(wkspace->data.dptr), m_shape, m_stride, aotriton::DType::kFloat32);

  //extract the saved rng state from aux_ctx_tensor
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
  //extract the philox seed and offset
  uint64_t philox_seed = *(reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr));
  uint64_t philox_offset = *(reinterpret_cast<const uint64_t *>(input_rng_state->data.dptr) + 1);

  using aotriton::v2::flash::attn_bwd;
  hipError_t err; // TODO: Error handling
  err = attn_bwd(q_tensor,
                 k_tensor,
                 v_tensor,
                 attn_scale,
                 o_tensor,
                 do_tensor,
                 dq_tensor,
                 dk_tensor,
                 dv_tensor,
                 M_tensor,
                 wkspace_tensor,
                 dropout,
                 philox_seed,
                 philox_offset,
                 attn_mask_type==NVTE_CAUSAL_MASK,
                 stream);
}