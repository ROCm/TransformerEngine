/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_arbitrary_seqlen.h
 *  \brief Functions for fused attention with seqlen > 512
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_

#include "transformer_engine/fused_attn.h"
#ifdef __HIP_PLATFORM_AMD__
#include "ck_tile/host.hpp"
#include "transformer_engine/bias.hpp"
#include "transformer_engine/mask.hpp"
#include "transformer_engine/fmha_fwd.hpp"
#include "transformer_engine/fmha_bwd.hpp"
#else
#include <cudnn.h>
#endif
#include "common/common.h"

namespace transformer_engine {
#ifdef __HIP_PLATFORM_AMD__
void fused_attn_arbitrary_seqlen_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
    bool is_training, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor* input_Q, const Tensor* input_K,
    const Tensor* input_V, const Tensor* input_Bias, Tensor* output_O,
    NVTETensorPack* Aux_CTX_Tensors, const Tensor* cu_seqlens_q,
    const Tensor* cu_seqlens_kv, const Tensor* rng_state,
    hipStream_t stream);

void fused_attn_arbitrary_seqlen_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, const Tensor* input_Q,
    const Tensor* input_K, const Tensor* input_V, const Tensor* input_O,
    const Tensor* input_dO, const Tensor* input_Bias, Tensor* output_S,
    Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
    Tensor* output_dBias, const Tensor* cu_seqlens_q,
    const Tensor* cu_seqlens_kv, const Tensor* rng_state,
    hipStream_t stream);
#else
#if (CUDNN_VERSION >= 8900)
void fused_attn_arbitrary_seqlen_fwd_qkvpacked(
                size_t batch, size_t num_attn_heads, size_t max_seqlen,
                size_t head_size, bool is_training, float attn_scale,
                float p_dropout, NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                const Tensor *input_QKV, const Tensor *input_Bias,
                Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                const Tensor *cu_seqlens, const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd_qkvpacked(
                size_t batch, size_t num_attn_heads, size_t max_seqlen,
                size_t head_dim, float attn_scale, float p_dropout,
                NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                NVTE_Mask_Type mask_type, const Tensor *input_QKV,
                const Tensor *input_O, const Tensor *input_dO,
                const Tensor *input_Bias, Tensor *output_S,
                Tensor *output_dQKV, Tensor *output_dBias,
                const Tensor *cu_seqlens, const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_fwd_kvpacked(
                size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                bool is_training, float attn_scale, float p_dropout,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_Bias,
                Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd_kvpacked(
                size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_O,
                const Tensor *input_dO, const Tensor *input_Bias, Tensor *output_S,
                Tensor *output_dQ, Tensor *output_dKV, Tensor *output_dBias,
                const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_fwd(
                size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                bool is_training, float attn_scale, float p_dropout,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                const Tensor *input_Q, const Tensor *input_K,
                const Tensor *input_V, const Tensor *input_Bias,
                Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd(
                size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                const Tensor *input_Q, const Tensor *input_K,
                const Tensor *input_V, const Tensor *input_O,
                const Tensor *input_dO, const Tensor *input_Bias, Tensor *output_S,
                Tensor *output_dQ, Tensor *output_dK,
                Tensor *output_dV, Tensor *output_dBias,
                const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                const Tensor *rng_state,
                Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

#endif  // CUDNN_VERSION >= 8900
#endif  // __HIP_PLATFORM_AMD__
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
