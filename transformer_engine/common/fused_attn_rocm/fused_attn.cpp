/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include <cstdlib>
#include <iostream>
#include <mutex>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "ck_fused_attn.h"
#include "utils.h"

using namespace transformer_engine;

std::string get_dtype_str(NVTEDType dtype) {
    switch (dtype) {
        case NVTEDType::kNVTEByte:
            return "kNVTEByte";
        case NVTEDType::kNVTEInt32:
            return "kNVTEInt32";
        case NVTEDType::kNVTEInt64:
            return "kNVTEInt64";
        case NVTEDType::kNVTEFloat32:
            return "kNVTEFloat32";
        case NVTEDType::kNVTEFloat16:
            return "kNVTEFloat16";
        case NVTEDType::kNVTEBFloat16:
            return "kNVTEBFloat16";
        case NVTEDType::kNVTEFloat8E4M3:
            return "kNVTEFloat8E4M3";
        case NVTEDType::kNVTEFloat8E5M2:
            return "kNVTEFloat8E5M2";
        case NVTEDType::kNVTENumTypes:
            return "kNVTENumTypes";
        default:
            NVTE_ERROR("dtype not supported!");
    }
}

std::string get_qkv_layout_str(NVTE_QKV_Layout qkv_layout) {
    switch (qkv_layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
            return "NVTE_SB3HD";
        case NVTE_QKV_Layout::NVTE_SBH3D:
            return "NVTE_SBH3D";
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
            return "NVTE_SBHD_SB2HD";
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
            return "NVTE_SBHD_SBH2D";
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            return "NVTE_SBHD_SBHD_SBHD";
        case NVTE_QKV_Layout::NVTE_BS3HD:
            return "NVTE_BS3HD";
        case NVTE_QKV_Layout::NVTE_BSH3D:
            return "NVTE_BSH3D";
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
            return "NVTE_BSHD_BS2HD";
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
            return "NVTE_BSHD_BSH2D";
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
            return "NVTE_BSHD_BSHD_BSHD";
        case NVTE_QKV_Layout::NVTE_T3HD:
            return "NVTE_T3HD";
        case NVTE_QKV_Layout::NVTE_TH3D:
            return "NVTE_TH3D";
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            return "NVTE_THD_T2HD";
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            return "NVTE_THD_TH2D";
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            return "NVTE_THD_THD_THD";
        default:
            NVTE_ERROR("qkv_layout not supported!");
    }
}

std::string get_bias_type_str(NVTE_Bias_Type bias_type) {
    switch (bias_type) {
        case NVTE_Bias_Type::NVTE_NO_BIAS:
            return "NVTE_NO_BIAS";
        case NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS:
            return "NVTE_PRE_SCALE_BIAS";
        case NVTE_Bias_Type::NVTE_POST_SCALE_BIAS:
            return "NVTE_POST_SCALE_BIAS";
        case NVTE_Bias_Type::NVTE_ALIBI:
            return "NVTE_ALIBI";
        default:
            NVTE_ERROR("bias_type not supported!");
    }
}

std::string get_mask_type_str(NVTE_Mask_Type mask_type) {
    switch (mask_type) {
        case NVTE_Mask_Type::NVTE_NO_MASK:
            return "NVTE_NO_MASK";
        case NVTE_Mask_Type::NVTE_PADDING_MASK:
            return "NVTE_PADDING_MASK";
        case NVTE_Mask_Type::NVTE_CAUSAL_MASK:
            return "NVTE_CAUSAL_MASK";
        case NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK:
            return "NVTE_PADDING_CAUSAL_MASK";
        default:
            NVTE_ERROR("mask_type not supported!");
    }
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, NVTETensor S, NVTETensor O,
                         NVTETensorPack *Aux_CTX_Tensors, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTETensor workspace, cudaStream_t stream) {
    const Tensor *input_cu_seqlens_q  = reinterpret_cast<const Tensor *>(cu_seqlens_q);
    const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor *>(cu_seqlens_kv);
    const Tensor *input_rng_state     = reinterpret_cast<const Tensor *>(rng_state);
    const Tensor *input_Q             = reinterpret_cast<const Tensor *>(Q);
    const Tensor *input_K             = reinterpret_cast<const Tensor *>(K);
    const Tensor *input_V             = reinterpret_cast<const Tensor *>(V);
    const Tensor *input_Bias          = reinterpret_cast<const Tensor *>(Bias);
    Tensor *input_output_S            = reinterpret_cast<Tensor *>(S);
    Tensor *output_O                  = reinterpret_cast<Tensor *>(O);
    Tensor *wkspace                   = reinterpret_cast<Tensor *>(workspace);

    auto ndim   = input_Q->data.shape.size();
    size_t b    = input_cu_seqlens_q->data.shape[0] - 1;
    size_t h_q  = input_Q->data.shape[ndim - 2];
    size_t h_kv = input_K->data.shape[ndim - 2];
    size_t d    = input_Q->data.shape[ndim - 1];

    const NVTEDType Q_type  = static_cast<NVTEDType>(input_Q->data.dtype);
    const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

    fused_attn_rocm::ck_fused_attn_fwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, is_training, attn_scale, dropout, qkv_layout,
        bias_type, attn_mask_type, input_Q, input_K, input_V, input_Bias, output_O, Aux_CTX_Tensors,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_rng_state, wkspace, stream);
}

// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, size_t max_seqlen_q, size_t max_seqlen_kv,
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTETensor workspace, cudaStream_t stream) {
    const Tensor *input_cu_seqlens_q  = reinterpret_cast<const Tensor *>(cu_seqlens_q);
    const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor *>(cu_seqlens_kv);
    const Tensor *input_Q             = reinterpret_cast<const Tensor *>(Q);
    const Tensor *input_K             = reinterpret_cast<const Tensor *>(K);
    const Tensor *input_V             = reinterpret_cast<const Tensor *>(V);
    const Tensor *input_O             = reinterpret_cast<const Tensor *>(O);
    const Tensor *input_dO            = reinterpret_cast<const Tensor *>(dO);
    const Tensor *input_S             = reinterpret_cast<const Tensor *>(S);
    Tensor *input_output_dP           = reinterpret_cast<Tensor *>(dP);
    Tensor *output_dQ                 = reinterpret_cast<Tensor *>(dQ);
    Tensor *output_dK                 = reinterpret_cast<Tensor *>(dK);
    Tensor *output_dV                 = reinterpret_cast<Tensor *>(dV);
    Tensor *output_dBias              = reinterpret_cast<Tensor *>(dBias);
    Tensor *wkspace                   = reinterpret_cast<Tensor *>(workspace);

    auto ndim   = input_Q->data.shape.size();
    size_t b    = input_cu_seqlens_q->data.shape[0] - 1;
    size_t h_q  = input_Q->data.shape[ndim - 2];
    size_t h_kv = input_K->data.shape[ndim - 2];
    size_t d    = input_Q->data.shape[ndim - 1];

    const NVTEDType Q_type  = static_cast<NVTEDType>(input_Q->data.dtype);
    const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

    NVTE_Fused_Attn_Backend fused_attention_backend =
        nvte_get_fused_attn_backend(Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout,
                                    h_q, h_kv, max_seqlen_q, max_seqlen_kv, d);

    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    Tensor *input_Bias, *input_rng_state;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
        input_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        input_Bias      = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    } else {
        input_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    }

    bool deterministic = getenv<int>("NVTE_ALLOW_NONDETERMINISTIC_ALGO") == 0;

    fused_attn_rocm::ck_fused_attn_bwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, attn_scale, dropout, qkv_layout, bias_type,
        attn_mask_type, input_Q, input_K, input_V, input_O, input_dO, input_Bias, output_S,
        output_dQ, output_dK, output_dV, output_dBias, input_cu_seqlens_q, input_cu_seqlens_kv,
        input_rng_state, wkspace, deterministic, stream);
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
    NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, float dropout, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim) {
    NVTE_Fused_Attn_Backend backend = NVTE_Fused_Attn_Backend::NVTE_CK_FMHA;
    return backend;
}
