/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once

#include "attn_common.h"
#include <type_traits>

namespace small_seq_kernels {

// ---------------------------------------------------------------------------
// Kernel 1: compute_scores_kernel
//
// Computes attention scores: Q @ K^T * scale
// Q layout:      [total_padded_q, head_num, head_dim]
// K layout:      [total_padded_kv_seq, head_num, head_dim]
// scores layout: [total_padded_q, head_num, max_seq_kv]
// ---------------------------------------------------------------------------

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_scores_kernel(const T* Q,
                                      const T* K,
                                      T* scores,
                                      float scale,
                                      const int* cu_seqlens_q,
                                      const int* cu_seqlens_q_padded,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch)
{
    // seq_q is 1 in static layout (storage), but actual Q length per batch may be 0 or 1.
    constexpr int seq_q = Config::seq_q; // == 1 (padded storage dimension)
    static_assert(seq_q == 1, "seq_q must be 1 for this kernel implementation.");
    constexpr int max_seq_kv        = Config::max_seq_kv;
    constexpr int head_dim          = Config::head_dim;
    constexpr int block_k           = 64;
    constexpr int thread_block_size = 64;
    constexpr int tasks_per_block   = TASKS_PER_BLOCK;

    int base_block_offset = blockIdx.x * thread_block_size * tasks_per_block;
    int thread_id         = threadIdx.x;

    for(int task = 0; task < tasks_per_block; task++)
    {
        int cur_batch_idx = base_block_offset + task * thread_block_size + thread_id;
        // Layout: [batch, seq_q(storage=1), head_num, head_dim]
        // cur_batch_idx represents the combined index for (batch * seq_q * head_num)
        int batch_idx    = cur_batch_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_batch_idx % (Config::seq_q * Config::head_num);
        int seq_idx      = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= batch)
            continue;

        // Skip batches where actual Q sequence length is 0.
        // Memory is still allocated (padded to seq_q=1), but no computation needed.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        // Get actual sequence length for this batch
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];

        // Q storage offset: cu_seqlens_q_padded[batch_idx] is the slot for this batch.
        // seq_idx is always 0 because seq_q == 1.
        int q_storage_offset = cu_seqlens_q_padded[batch_idx];

        float results[max_seq_kv];
        T fetch_Q[block_k];
        T fetch_K[block_k];
        // Q: [total_padded_seq_q, head_num, head_dim]
        T* Q_ptr = (T*)&Q[(q_storage_offset * Config::head_num + head_idx) * head_dim];
        // K: [total_padded_seq_kv, head_num, head_dim]
        T* K_ptr     = (T*)&K[(kv_offset * Config::head_num + head_idx) * head_dim];
        // scores workspace: [total_padded_q, head_num, max_seq_kv]
        // index by padded Q slot: cu_seqlens_q_padded[batch_idx]
        T* score_ptr =
            (T*)&scores[(cu_seqlens_q_padded[batch_idx] * Config::head_num + head_idx) *
                        max_seq_kv];
        uint4 ls_dwordx4_tmp_var;
        for(int i = 0; i < seq_kv; i++)
            results[i] = 0.0f;
        for(int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k)
        {
            if constexpr(std::is_same<T, hip_bfloat16>::value)
            {
                for(int k = 0; k < block_k / 8; k++)
                {
                    ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 8]);
                    fetch_Q[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                    fetch_Q[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                    fetch_Q[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                    fetch_Q[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                    fetch_Q[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                    fetch_Q[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                    fetch_Q[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                    fetch_Q[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                }
                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 8; k++)
                    {
                        // K layout: [batch, seq_kv, head_num, head_dim]
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * Config::head_num * head_dim +
                                                              dim_offset + k * 8]);
                        fetch_K[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                        fetch_K[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                        fetch_K[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                        fetch_K[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                        fetch_K[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                        fetch_K[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                        fetch_K[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                        fetch_K[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] +=
                            static_cast<float>(fetch_Q[k]) * static_cast<float>(fetch_K[k]);
                    }
                }
            }
            else
            {
                for(int k = 0; k < block_k / 4; k++)
                {
                    ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 4]);
                    fetch_Q[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                    fetch_Q[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                    fetch_Q[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                    fetch_Q[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                }
                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 4; k++)
                    {
                        // K layout: [batch, seq_kv, head_num, head_dim]
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * Config::head_num * head_dim +
                                                              dim_offset + k * 4]);
                        fetch_K[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                        fetch_K[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                        fetch_K[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                        fetch_K[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] += fetch_Q[k] * fetch_K[k];
                    }
                }
            }
        }
        for(int i = 0; i < seq_kv; i++)
        {
            score_ptr[i] = T(results[i] * scale);
        }
        // Zero out padding positions
        for(int i = seq_kv; i < max_seq_kv; i++)
        {
            score_ptr[i] = T(-1e9f);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: apply_mask_and_softmax_kernel
//
// Applies causal/KV-padding masks and computes numerically stable softmax.
// scores layout:      [total_padded_q, head_num, max_seq_kv]
// padded_q_to_batch:  host-precomputed reverse map [padded_q_slot] -> batch_idx
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__global__ void apply_mask_and_softmax_kernel(T* scores,
                                              const T* dropout_mask,
                                              float dropout_scale,
                                              const int* cu_seqlens_kv,
                                              const int* padded_q_to_batch,
                                              uint32_t total_elt)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_score_size     = max_seq_kv; // seq_q == 1
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_scores[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T row_max[row_num];
        __shared__ T row_sum[row_num];

        // scores layout: [total_padded_q, head_num, max_seq_kv]
        // global_row_idx encodes (padded_q_slot * head_num + head_idx)
        int global_row_idx  = cur_block_offset / max_seq_kv;
        int padded_q_slot   = global_row_idx / Config::head_num;
        int k_idx           = cur_block_offset % max_seq_kv;

        // Reverse-map padded Q slot to batch_idx via host-precomputed table.
        // All slots in the buffer are guaranteed to belong to a valid (active) batch
        // because empty-Q batches contribute no rows.
        int batch_idx = padded_q_to_batch[padded_q_slot];

        // Get actual KV sequence length for this batch
        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

        tmp_scores[thread_id] = scores[cur_block_offset];

        // Apply causal mask / KV-padding mask
        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            // q_idx == 0 (seq_q == 1); mask: k_idx > 0 || k_idx >= seq_kv
            if(k_idx > 0 || k_idx >= seq_kv)
                tmp_scores[thread_id] = T(-1e9f);
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            // q_idx == 0; mask: k_idx < 0 (never) || k_idx >= seq_kv
            if(k_idx >= seq_kv)
                tmp_scores[thread_id] = T(-1e9f);
        }
        else
        {
            if(k_idx >= seq_kv)
                tmp_scores[thread_id] = T(-1e9f);
        }
        __syncthreads();

        // Find max for each row (numerically stable softmax)
        if(thread_id < real_row_num)
        {
            T max_val = T(-1e9f);
#pragma unroll
            for(int i = 0; i < max_seq_kv; i++)
            {
                max_val = max(max_val, tmp_scores[thread_id * max_seq_kv + i]);
            }
            row_max[thread_id] = max_val;
        }
        __syncthreads();

        // Compute exp(score - max) and sum for each row
        T exp_val = T(exp(float(tmp_scores[thread_id] - row_max[thread_id / max_seq_kv])));
        tmp_scores[thread_id] = exp_val;
        __syncthreads();

        if(thread_id < real_row_num)
        {
            T sum = T(0.0f);
#pragma unroll
            for(int i = 0; i < max_seq_kv; i++)
            {
                sum += tmp_scores[thread_id * max_seq_kv + i];
            }
            row_sum[thread_id] = sum;
        }
        __syncthreads();

        // Normalize and apply dropout
        T attn_weight = tmp_scores[thread_id] / row_sum[thread_id / max_seq_kv];

        if constexpr(Config::enable_dropout_mask)
        {
            attn_weight = attn_weight * dropout_mask[cur_block_offset] * dropout_scale;
        }

        scores[cur_block_offset] = attn_weight;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: compute_output_kernel
//
// Computes attention output: attn_weights @ V
// attn_weights layout: [total_padded_q, head_num, max_seq_kv]
// V layout:            [total_padded_kv_seq, head_num, head_dim]
// O layout:            [total_padded_q, head_num, head_dim]
// ---------------------------------------------------------------------------

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 8>
__global__ void compute_output_kernel(const T* attn_weights,
                                      const T* V,
                                      T* O,
                                      const int* cu_seqlens_q,
                                      const int* cu_seqlens_q_padded,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch)
{
    constexpr int seq_q                 = Config::seq_q;
    constexpr int max_seq_kv            = Config::max_seq_kv;
    constexpr int head_dim              = Config::head_dim;
    constexpr int block_k               = BLOCK_K;
    constexpr int dwordx4_load_elt      = 16 / sizeof(T);
    constexpr int warp_size             = 64;
    constexpr int process_head_per_warp = warp_size / (head_dim / block_k);
    constexpr int tasks_per_block       = TASKS_PER_BLOCK;

    int base_block_offset   = blockIdx.x * process_head_per_warp * tasks_per_block;
    int thread_id           = threadIdx.x;
    int thread_batch_offset = thread_id / (head_dim / block_k);
    int thread_head_offset  = thread_id % (head_dim / block_k) * block_k;

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt],
        store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
    T result[block_k];
    T attn[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        // Layout: [batch, seq_q(storage=1), head_num, head_dim]
        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int seq_q_idx    = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= batch)
            continue;

        // Skip batches where actual Q seq length is 0 — no output to write.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        // Get actual sequence length for this batch
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];

        // Q output storage offset: one slot per batch, seq_q_idx is always 0.
        int q_storage_offset = cu_seqlens_q_padded[batch_idx];

#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            store_dwordx4_tmp_var[i].x = 0;
            store_dwordx4_tmp_var[i].y = 0;
            store_dwordx4_tmp_var[i].z = 0;
            store_dwordx4_tmp_var[i].w = 0;
        }
        // attn_weights layout: [total_padded_q, head_num, max_seq_kv]
        int attn_offset = (cu_seqlens_q_padded[batch_idx] * Config::head_num + head_idx) * max_seq_kv;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[attn_offset + i];
        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                // V layout: [total_padded_seq_kv, head_num, head_dim]
                load_dwordx4_tmp_var[i] =
                    *((uint4*)&V[((kv_offset + j) * Config::head_num + head_idx) * head_dim +
                                 thread_head_offset + i * dwordx4_load_elt]);
            }
#pragma unroll
            for(int b = 0; b < block_k; b++)
                ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
                    attn[j] *
                    ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
        }
#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            // O layout: [total_padded_seq_q, head_num, head_dim]
            *((uint4*)&O[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                         thread_head_offset + i * dwordx4_load_elt]) = store_dwordx4_tmp_var[i];
    }
}

// ---------------------------------------------------------------------------
// AttnForwardKernelLauncher
//
// Orchestrates the 3-kernel forward pipeline:
//   1. compute_scores_kernel   (Q @ K^T * scale)
//   2. apply_mask_and_softmax_kernel
//   3. compute_output_kernel   (attn_weights @ V)
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnForwardKernelLauncher
{
    // workspace layout: [total_padded_q, head_num, max_seq_kv]
    // total_padded_q = cu_seqlens_q_padded[bs] — known on host before kernel launch.
    static size_t calc_workspace_size(int total_padded_q)
    {
        constexpr int head_num   = Config::head_num;
        constexpr int max_seq_kv = Config::max_seq_kv;
        return (size_t)total_padded_q * head_num * max_seq_kv * sizeof(T);
    }

    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
                                    T* workspace,
                                    const int* cu_seqlens_q,
                                    const int* cu_seqlens_q_padded,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    const int* padded_q_to_batch,
                                    int total_padded_q,
                                    int batch)
    {
        const int bs             = batch;
        constexpr int head_num   = Config::head_num;
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;
        constexpr int head_dim   = Config::head_dim;
        constexpr int warp_size  = 64;

        const int merge_bs  = bs * head_num;
        float scale         = sqr_dk_scale;
        float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        // Step 1: QK^T scores — grid covers all (batch * head_num) tasks
        constexpr int kernel1_threads = 64;
        dim3 block(kernel1_threads);
        dim3 grid(merge_bs / kernel1_threads);
        compute_scores_kernel<T, Config, 1><<<grid, block>>>(
            Q, K, workspace, scale, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded, batch);

        // Step 2: Mask + softmax — grid covers [total_padded_q, head_num, max_seq_kv] elements
        constexpr int work_thread_num =
            Config::step2_block_size / max_seq_kv * max_seq_kv; // seq_q == 1
        uint32_t total_elt = (uint32_t)total_padded_q * head_num * max_seq_kv;
        dim3 grid2((total_elt + work_thread_num - 1) / work_thread_num);
        dim3 block2(Config::step2_block_size);
        apply_mask_and_softmax_kernel<T, Config>
            <<<grid2, block2>>>(workspace, dropout_mask, dropout_scale, cu_seqlens_kv,
                                padded_q_to_batch, total_elt);

        // Step 3: Weighted sum over V — grid covers all (batch * head_num) tasks
        constexpr int kernel3_block_k       = 8;
        constexpr int kernel3_threads       = 64;
        constexpr int process_head_per_warp = warp_size / (head_dim / kernel3_block_k);
        dim3 block3(kernel3_threads);
        dim3 grid3((merge_bs / process_head_per_warp + 2 - 1) / 2);
        compute_output_kernel<T, Config, 2, kernel3_block_k><<<grid3, block3>>>(
            workspace, V, O, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded, batch);
    }
};

}  // namespace small_seq_kernels
