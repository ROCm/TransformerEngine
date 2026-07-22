/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include "attn_fwd_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

template <typename T>
void attn_forward(const T* Q,
                  const T* K,
                  const T* V,
                  const T* dropout_mask,
                  float dropout_p,
                  T* O,
                  T* attn_weights,
                  int batch,
                  int head_num,
                  int max_kv_seq,
                  int head_dim,
                  CausalMaskType mask_type,
                  const int* cu_seqlens_q,
                  const int* cu_seqlens_q_padded,
                  const int* cu_seqlens_kv,
                  const int* cu_seqlens_kv_padded,
                  bool bf16_weights)
{
    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Allocate temporary buffers in float (matching GPU kernel precision)
    std::vector<float> scores(max_kv_seq);
    std::vector<float> attn_probs(max_kv_seq);

    // Total padded Q storage size
    int total_padded_q = cu_seqlens_q_padded[batch];

    // Initialize output to zero
    std::memset(O, 0, total_padded_q * head_num * head_dim * sizeof(T));
    if(attn_weights != nullptr)
    {
        // attn_weights: [total_padded_q, head_num, max_kv_seq]
        std::memset(attn_weights, 0, total_padded_q * head_num * max_kv_seq * sizeof(T));
    }

    // Process each batch and head
    for(int b = 0; b < batch; b++)
    {
        // Skip batches where actual Q seq length is 0
        int actual_q_seq = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        if(actual_q_seq == 0)
            continue;

        // Get actual KV sequence length for this batch
        int kv_seq    = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];
        int kv_offset = cu_seqlens_kv_padded[b];
        // Q padded storage offset
        int q_offset  = cu_seqlens_q_padded[b];

        for(int h = 0; h < head_num; h++)
        {
            // For each query position (actual_q_seq is 0 or 1, and we already checked != 0)
            for(int q_idx = 0; q_idx < actual_q_seq; q_idx++)
            {
                // Q: [total_padded_seq_q, head_num, head_dim]
                int offset_Q = ((q_offset + q_idx) * head_num + h) * head_dim;
                // O: [total_padded_seq_q, head_num, head_dim]
                int offset_O = ((q_offset + q_idx) * head_num + h) * head_dim;
                // attn_weights: [total_padded_q, head_num, max_kv_seq]
                int offset_attn    = ((q_offset + q_idx) * head_num + h) * max_kv_seq;
                int offset_dropout = dropout_mask ? ((q_offset + q_idx) * head_num + h) * max_kv_seq : 0;

                const T* Q_ptr       = Q + offset_Q;
                const T* dropout_ptr = dropout_mask ? dropout_mask + offset_dropout : nullptr;

                T* O_ptr    = O + offset_O;
                T* attn_ptr = attn_weights ? attn_weights + offset_attn : nullptr;

                // Step 1: Compute scores = Q @ K^T / sqrt(d_k)
                // Q: [1, head_dim], K: [kv_seq, head_dim] -> scores: [1, kv_seq]
                for(int kv_idx = 0; kv_idx < kv_seq; kv_idx++)
                {
                    int k_offset   = ((kv_offset + kv_idx) * head_num + h) * head_dim;
                    const T* K_ptr = K + k_offset;
                    float sum      = 0.0f;
                    for(int d = 0; d < head_dim; d++)
                        sum += float(Q_ptr[d]) * float(K_ptr[d]);
                    scores[kv_idx] = sum * scale;
                }

                // Step 2: Apply causal mask
                if(mask_type == CausalMaskType::TOP_LEFT)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j > q_idx)
                        {
                            scores[j] = -1e9f;
                        }
                    }
                }
                else if(mask_type == CausalMaskType::BOTTOM_RIGHT)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j < q_idx)
                        {
                            scores[j] = -1e9f;
                        }
                    }
                }

                // Step 3: Softmax (numerically stable, all in float)
                float max_val = -1e9f;
                for(int j = 0; j < kv_seq; j++)
                {
                    max_val = std::max(max_val, scores[j]);
                }

                float sum = 0.0f;
                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[j] = std::exp(scores[j] - max_val);
                    sum += attn_probs[j];
                }

                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[j] /= sum;
                }

                // Step 4: Apply dropout
                if(dropout_p > 0.0f && dropout_ptr != nullptr)
                {
                    for(int i = 0; i < kv_seq; i++)
                    {
                        attn_probs[i] *= float(dropout_ptr[i]) * dropout_scale;
                    }
                }

                // Save attention weights if requested (truncate to T for storage)
                if(attn_ptr != nullptr)
                {
                    for(int j = 0; j < kv_seq; j++)
                        attn_ptr[j] = T(attn_probs[j]);
                }

                // Truncate weights to bf16 (matches MFMA kernel: float→bhalf_t via SM_lds)
                if(bf16_weights)
                {
                    for(int j = 0; j < kv_seq; j++)
                        attn_probs[j] = float(hip_bfloat16(attn_probs[j]));
                }

                // Step 5: Compute output = attn_probs @ V
                // attn_probs: [1, kv_seq], V: [kv_seq, head_dim] -> O: [1, head_dim]
                for(int d = 0; d < head_dim; d++)
                {
                    float sum = 0.0f;
                    for(int kv_idx = 0; kv_idx < kv_seq; kv_idx++)
                    {
                        int v_offset = ((kv_offset + kv_idx) * head_num + h) * head_dim;
                        sum += attn_probs[kv_idx] * float(V[v_offset + d]);
                    }
                    O_ptr[d] = T(sum);
                }
            }
        }
    }
}

// Explicit instantiations
template void attn_forward<float>(const float*, const float*, const float*, const float*,
                                  float, float*, float*, int, int, int, int, CausalMaskType,
                                  const int*, const int*, const int*, const int*, bool);
template void attn_forward<hip_bfloat16>(const hip_bfloat16*, const hip_bfloat16*,
                                         const hip_bfloat16*, const hip_bfloat16*, float,
                                         hip_bfloat16*, hip_bfloat16*, int, int, int, int,
                                         CausalMaskType, const int*, const int*, const int*,
                                         const int*, bool);
