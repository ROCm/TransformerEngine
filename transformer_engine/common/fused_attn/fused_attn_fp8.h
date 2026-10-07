/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_fp8.h
 *  \brief Functions for fused attention for FP8
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_FP8_H_

#include <cudnn.h>

#include <string>

#include "config_and_params.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
// fused attention FWD FP8 with separate Q, K, V
<<<<<<< 27ccad5ed300521e7026f904a7d9def68d520ce3
void fused_attn_fp8_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, size_t num_tokens_q,
    size_t num_tokens_kv, bool is_training, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_Softmax_Type softmax_type,
    size_t window_size_left, size_t window_size_right, bool bottom_right_diagonal,
    const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
    const Tensor *input_SoftmaxOffset, Tensor *input_output_S, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *cu_seqlens_q_padded, const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, size_t num_tokens_q,
    size_t num_tokens_kv, float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
    NVTE_QKV_Format o_format, NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_QKV_Format do_scale_inv_format,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_Softmax_Type softmax_type,
    size_t window_size_left, size_t window_size_right, bool bottom_right_diagonal,
    bool deterministic, const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
    const Tensor *input_O, const Tensor *input_dO, const Tensor *input_dO_f16,
    const Tensor *input_M, const Tensor *input_S, const Tensor *input_SoftmaxOffset,
    Tensor *input_output_dP, const Tensor *output_dQ, const Tensor *output_dK,
    const Tensor *output_dV, Tensor *output_dSoftmaxOffset, const Tensor *cu_seqlens_q,
    const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
    const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state, Tensor *workspace,
    cudaStream_t stream, cudnnHandle_t handle);
=======
void fused_attn_fp8_fwd(const fused_attn::FusedAttnConfig &cfg, const Tensor *input_Q,
                        const Tensor *input_K, const Tensor *input_V,
                        const Tensor *input_SoftmaxOffset, Tensor *input_output_S, Tensor *output_O,
                        NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q,
                        const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
                        const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
                        Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(const fused_attn::FusedAttnConfig &cfg, const Tensor *input_Q,
                        const Tensor *input_K, const Tensor *input_V, const Tensor *input_O,
                        const Tensor *input_dO, const Tensor *input_dO_f16, const Tensor *input_M,
                        const Tensor *input_S, const Tensor *input_SoftmaxOffset,
                        Tensor *input_output_dP, const Tensor *output_dQ, const Tensor *output_dK,
                        const Tensor *output_dV, Tensor *output_dSoftmaxOffset,
                        const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                        const Tensor *cu_seqlens_q_padded, const Tensor *cu_seqlens_kv_padded,
                        const Tensor *rng_state, Tensor *workspace, cudaStream_t stream,
                        cudnnHandle_t handle);

std::string support_verdict_fp8(const fused_attn::FusedAttnConfig &cfg, fused_attn::Pass pass,
                                cudnnHandle_t handle);

>>>>>>> 796346c0e0497b1f56a8d36ec76e02db5a6fed47
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_FP8_H_
