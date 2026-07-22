/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include "attn_bwd_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helper function implementations
// ---------------------------------------------------------------------------

template <typename T>
void matmul(const T* A, const T* B, T* C, int rows_a, int cols_a, int cols_b)
{
    for(int i = 0; i < rows_a; i++)
    {
        for(int j = 0; j < cols_b; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < cols_a; k++)
            {
                sum += float(A[i * cols_a + k]) * float(B[k * cols_b + j]);
            }
            C[i * cols_b + j] = T(sum);
        }
    }
}

template <typename T>
void transpose(const T* A, T* A_T, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

template <typename T>
void sum_last_dim(const T* A, T* sums, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < cols; j++)
        {
            sum += float(A[i * cols + j]);
        }
        sums[i] = T(sum);
    }
}

// ---------------------------------------------------------------------------
// Backward pass implementation
// ---------------------------------------------------------------------------

template <typename T>
void attn_backward(const T* Q,
                   const T* K,
                   const T* V,
                   const T* grad_O,
                   const T* attn_weights,
                   const T* dropout_mask,
                   float dropout_p,
                   T* grad_Q,
                   T* grad_K,
                   T* grad_V,
                   int batch,
                   int head_num,
                   int max_kv_seq,
                   int head_dim,
                   CausalMaskType mask_type,
                   const int* cu_seqlens_q,
                   const int* cu_seqlens_q_padded,
                   const int* cu_seqlens_kv,
                   const int* cu_seqlens_kv_padded,
                   int total_padded_q,
                   int total_padded_kv_seq,
                   int max_seq_q,
                   bool bf16_weights)
{
    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Temporary buffers sized for multi-Q
    std::vector<T> K_cont_buf(max_kv_seq * head_dim);
    std::vector<T> V_cont_buf(max_kv_seq * head_dim);
    std::vector<T> grad_K_cont_buf(max_kv_seq * head_dim);
    std::vector<T> grad_V_cont_buf(max_kv_seq * head_dim);
    std::vector<float> grad_attn(max_seq_q * max_kv_seq);
    std::vector<float> grad_scores(max_seq_q * max_kv_seq);

    // Initialize gradients to zero
    std::memset(grad_Q, 0, total_padded_q * head_num * head_dim * sizeof(T));
    std::memset(grad_K, 0, total_padded_kv_seq * head_num * head_dim * sizeof(T));
    std::memset(grad_V, 0, total_padded_kv_seq * head_num * head_dim * sizeof(T));

    for(int b = 0; b < batch; b++)
    {
        // Skip batches where actual Q seq is 0
        int actual_q_seq = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        if(actual_q_seq == 0)
            continue;

        int kv_seq       = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];
        int q_off        = cu_seqlens_q_padded[b];  // padded Q storage offset
        int kv_off       = cu_seqlens_kv_padded[b]; // padded KV storage offset
        int kv_stride    = head_num * head_dim;

        for(int h = 0; h < head_num; h++)
        {
            // K/V: [total_padded_seq_kv, head_num, head_dim]
            int offset_kv_base = kv_off * head_num * head_dim + h * head_dim;
            const T* K_bh      = K + offset_kv_base;
            const T* V_bh      = V + offset_kv_base;
            T* grad_K_bh       = grad_K + offset_kv_base;
            T* grad_V_bh       = grad_V + offset_kv_base;

            // Flatten K/V into contiguous row-major buffers [kv_seq, head_dim]
            for(int i = 0; i < kv_seq; i++)
                for(int j = 0; j < head_dim; j++)
                {
                    K_cont_buf[i * head_dim + j] = K_bh[i * kv_stride + j];
                    V_cont_buf[i * head_dim + j] = V_bh[i * kv_stride + j];
                }

            // Zero grad_V accumulator
            std::fill(grad_V_cont_buf.begin(), grad_V_cont_buf.begin() + kv_seq * head_dim, T(0.0f));
            // Zero grad_K accumulator
            std::fill(grad_K_cont_buf.begin(), grad_K_cont_buf.begin() + kv_seq * head_dim, T(0.0f));

            // --- Process each Q row ---
            for(int q_idx = 0; q_idx < actual_q_seq; q_idx++)
            {
                // Q/grad_O/grad_Q: [total_padded_seq_q, head_num, head_dim]
                int offset_Q = ((q_off + q_idx) * head_num + h) * head_dim;
                const T* Q_bh_q       = Q + offset_Q;
                const T* grad_O_bh_q  = grad_O + offset_Q;
                T* grad_Q_bh_q        = grad_Q + offset_Q;

                // attn_weights/dropout_mask: [total_padded_q, head_num, max_kv_seq]
                int offset_attn = ((q_off + q_idx) * head_num + h) * max_kv_seq;
                const T* attn_bh_q    = attn_weights + offset_attn;
                int offset_drop = dropout_mask ? offset_attn : 0;
                const T* dropout_bh_q = dropout_mask ? dropout_mask + offset_drop : nullptr;

                // Step 1: grad_V[j,d] += attn[q,j] * grad_O[q,d]  (accumulate over Q rows)
                for(int j = 0; j < kv_seq; j++)
                {
                    float aw = float(attn_bh_q[j]);
                    if(bf16_weights) aw = float(hip_bfloat16(aw));
                    for(int d = 0; d < head_dim; d++)
                        grad_V_cont_buf[j * head_dim + d] =
                            T(float(grad_V_cont_buf[j * head_dim + d]) + aw * float(grad_O_bh_q[d]));
                }

                // Step 2: grad_attn[q,j] = dot(grad_O[q,:], V[j,:])
                for(int j = 0; j < kv_seq; j++)
                {
                    float s = 0.0f;
                    for(int d = 0; d < head_dim; d++)
                        s += float(grad_O_bh_q[d]) * float(V_cont_buf[j * head_dim + d]);
                    grad_attn[q_idx * max_kv_seq + j] = s;
                }

                // Step 3: Dropout backward
                if(dropout_p > 0.0f && dropout_bh_q != nullptr)
                    for(int j = 0; j < kv_seq; j++)
                        grad_attn[q_idx * max_kv_seq + j] *= float(dropout_bh_q[j]) * dropout_scale;

                // Step 4: Softmax backward — per Q row, independent
                // grad_score[q,j] = attn[q,j] * (grad_attn[q,j] - dot_sum)
                float dot_sum = 0.0f;
                for(int j = 0; j < kv_seq; j++)
                {
                    float aw = float(attn_bh_q[j]);
                    if(bf16_weights) aw = float(hip_bfloat16(aw));
                    dot_sum += grad_attn[q_idx * max_kv_seq + j] * aw;
                }
                for(int j = 0; j < kv_seq; j++)
                {
                    float aw = float(attn_bh_q[j]);
                    if(bf16_weights) aw = float(hip_bfloat16(aw));
                    grad_scores[q_idx * max_kv_seq + j] = aw * (grad_attn[q_idx * max_kv_seq + j] - dot_sum);
                }

                // Step 5: Mask backward
                if(mask_type == CausalMaskType::TOP_LEFT)
                {
                    for(int j = 0; j < kv_seq; j++)
                        if(j > q_idx) grad_scores[q_idx * max_kv_seq + j] = 0.0f;
                }

                // Step 6: grad_Q[q,d] = sum_j grad_scores[q,j] * K[j,d] * scale
                for(int d = 0; d < head_dim; d++)
                {
                    float s = 0.0f;
                    for(int j = 0; j < kv_seq; j++)
                        s += grad_scores[q_idx * max_kv_seq + j] * float(K_cont_buf[j * head_dim + d]);
                    grad_Q_bh_q[d] = T(s * scale);
                }

                // Step 7: grad_K[j,d] += grad_scores[q,j] * Q[q,d] * scale  (accumulate over Q rows)
                for(int j = 0; j < kv_seq; j++)
                    for(int d = 0; d < head_dim; d++)
                    {
                        float gs = grad_scores[q_idx * max_kv_seq + j];
                        grad_K_cont_buf[j * head_dim + d] =
                            T(float(grad_K_cont_buf[j * head_dim + d]) +
                              gs * float(Q_bh_q[d]) * scale);
                    }
            }

            // Copy grad_K and grad_V back to strided layout
            for(int i = 0; i < kv_seq; i++)
                for(int j = 0; j < head_dim; j++)
                {
                    grad_K_bh[i * kv_stride + j] = grad_K_cont_buf[i * head_dim + j];
                    grad_V_bh[i * kv_stride + j] = grad_V_cont_buf[i * head_dim + j];
                }
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void matmul<float>(const float*, const float*, float*, int, int, int);
template void matmul<hip_bfloat16>(const hip_bfloat16*, const hip_bfloat16*, hip_bfloat16*, int, int, int);

template void transpose<float>(const float*, float*, int, int);
template void transpose<hip_bfloat16>(const hip_bfloat16*, hip_bfloat16*, int, int);

template void sum_last_dim<float>(const float*, float*, int, int);
template void sum_last_dim<hip_bfloat16>(const hip_bfloat16*, hip_bfloat16*, int, int);

template void attn_backward<float>(const float*, const float*, const float*, const float*,
                                   const float*, const float*, float, float*, float*, float*,
                                   int, int, int, int, CausalMaskType,
                                   const int*, const int*, const int*, const int*, int, int,
                                   int, bool);
template void attn_backward<hip_bfloat16>(const hip_bfloat16*, const hip_bfloat16*,
                                          const hip_bfloat16*, const hip_bfloat16*,
                                          const hip_bfloat16*, const hip_bfloat16*, float,
                                          hip_bfloat16*, hip_bfloat16*, hip_bfloat16*,
                                          int, int, int, int, CausalMaskType,
                                          const int*, const int*, const int*, const int*, int, int,
                                          int, bool);
