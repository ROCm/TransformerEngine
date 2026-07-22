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
// Kernel 1: compute_grad_v_kernel
//
// Computes grad_V = attn_weights^T @ grad_O
// attn_weights layout: [total_padded_q, head_num, max_seq_kv]
// grad_O layout:       [total_padded_q, head_num, head_dim]
// grad_V layout:       [total_padded_kv_seq, head_num, head_dim]
// ---------------------------------------------------------------------------

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_v_kernel(const T* attn_weights,
                                      const T* grad_O,
                                      T* grad_V,
                                      const int* cu_seqlens_q,
                                      const int* cu_seqlens_q_padded,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch)
{
    constexpr int seq_q                 = Config::seq_q; // == 1
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

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt];
    T attn[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= batch)
            continue;

        // Skip batches where actual Q seq is 0 — no grad_O to read from.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_q_idx == 0

        // attn_weights layout: [total_padded_q, head_num, max_seq_kv]
        int attn_offset = (q_storage_offset * Config::head_num + head_idx) * max_seq_kv;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[attn_offset + i];

        // Compute grad_V = attn_weights^T @ grad_O
        for(int j = 0; j < seq_kv; j++)
        {
            uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                store_dwordx4_tmp_var[i].x = 0;
                store_dwordx4_tmp_var[i].y = 0;
                store_dwordx4_tmp_var[i].z = 0;
                store_dwordx4_tmp_var[i].w = 0;
            }

            // grad_O layout: [total_padded_seq_q, head_num, head_dim]
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                load_dwordx4_tmp_var[i] =
                    *((uint4*)&grad_O[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                                      thread_head_offset + i * dwordx4_load_elt]);
            }

#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
                    attn[j] *
                    ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
            }

#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                int grad_v_idx =
                    (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                    head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
                *((uint4*)&grad_V[grad_v_idx]) = store_dwordx4_tmp_var[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: compute_grad_attn_kernel
//
// Computes grad_attn = grad_O @ V^T  (same structure as compute_scores_kernel)
// grad_O layout:    [total_padded_q, head_num, head_dim]
// V layout:         [total_padded_kv_seq, head_num, head_dim]
// grad_attn layout: [total_padded_q, head_num, max_seq_kv]  (workspace reuse)
// ---------------------------------------------------------------------------

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_grad_attn_kernel(const T* grad_O,
                                         const T* V,
                                         T* grad_attn,
                                         const int* cu_seqlens_q,
                                         const int* cu_seqlens_q_padded,
                                         const int* cu_seqlens_kv,
                                         const int* cu_seqlens_kv_padded,
                                         int batch)
{
    constexpr int seq_q = Config::seq_q; // == 1
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
        int batch_idx     = cur_batch_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx  = cur_batch_idx % (Config::seq_q * Config::head_num);
        int head_idx      = seq_head_idx % Config::head_num;

        if(batch_idx >= batch)
            continue;

        // Skip batches where actual Q seq is 0 — no row exists in workspace for them.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_idx == 0

        float results[max_seq_kv];
        T fetch_grad_O[block_k];
        T fetch_V[block_k];

        // grad_O layout: [total_padded_seq_q, head_num, head_dim]
        T* grad_O_ptr =
            (T*)&grad_O[(q_storage_offset * Config::head_num + head_idx) * head_dim];

        const T* V_base =
            &V[cu_seqlens_kv_padded[batch_idx] * Config::head_num * head_dim + head_idx * head_dim];
        int V_stride = Config::head_num * head_dim;

        // workspace layout: [total_padded_q, head_num, max_seq_kv]
        T* grad_attn_ptr = (T*)&grad_attn[(q_storage_offset * Config::head_num + head_idx) * max_seq_kv];

        uint4 ls_dwordx4_tmp_var;

        for(int i = 0; i < seq_kv; i++)
            results[i] = 0.0f;

        for(int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k)
        {
            if constexpr(std::is_same<T, hip_bfloat16>::value)
            {
                for(int k = 0; k < block_k / 8; k++)
                {
                    ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 8]);
                    fetch_grad_O[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                    fetch_grad_O[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                    fetch_grad_O[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                    fetch_grad_O[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                    fetch_grad_O[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                    fetch_grad_O[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                    fetch_grad_O[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                    fetch_grad_O[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                }

                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 8; k++)
                    {
                        ls_dwordx4_tmp_var =
                            *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 8]);
                        fetch_V[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                        fetch_V[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                        fetch_V[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                        fetch_V[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                        fetch_V[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                        fetch_V[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                        fetch_V[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                        fetch_V[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] +=
                            static_cast<float>(fetch_grad_O[k]) * static_cast<float>(fetch_V[k]);
                    }
                }
            }
            else
            {
                for(int k = 0; k < block_k / 4; k++)
                {
                    ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 4]);
                    fetch_grad_O[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                    fetch_grad_O[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                    fetch_grad_O[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                    fetch_grad_O[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                }

                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 4; k++)
                    {
                        ls_dwordx4_tmp_var =
                            *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 4]);
                        fetch_V[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                        fetch_V[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                        fetch_V[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                        fetch_V[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] += fetch_grad_O[k] * fetch_V[k];
                    }
                }
            }
        }

        for(int i = 0; i < seq_kv; i++)
        {
            grad_attn_ptr[i] = T(results[i]);
        }
        // Zero out padding positions beyond seq_kv
        for(int i = seq_kv; i < max_seq_kv; i++)
        {
            grad_attn_ptr[i] = T(0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: softmax_backward_kernel
//
// Softmax backprop: attn * (grad_attn - sum(grad_attn * attn))
// Writes the result back into grad_attn (workspace reuse as grad_scores).
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__global__ void softmax_backward_kernel(const T* attn_weights,
                                        const T* dropout_mask,
                                        T* grad_attn,
                                        float dropout_scale,
                                        const int* cu_seqlens_kv,
                                        const int* padded_q_to_batch,
                                        uint32_t total_elt)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_grad_attn_size = max_seq_kv; // seq_q == 1
    constexpr int valid_thread_range = block_size / per_grad_attn_size * per_grad_attn_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_grad_score[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T reduce_grad_score[row_num];

        // [total_padded_q, head_num, max_seq_kv] flat layout
        int global_row_idx = cur_block_offset / max_seq_kv;
        int padded_q_slot  = global_row_idx / Config::head_num;
        int k_idx          = cur_block_offset % max_seq_kv;

        // All rows in the buffer belong to active batches (empty-Q batches have no row).
        int batch_idx = padded_q_to_batch[padded_q_slot];
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

        T grad_attn_value = grad_attn[cur_block_offset];
        if constexpr(Config::enable_dropout_mask)
        {
            grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
        }
        T attn_weight             = attn_weights[cur_block_offset];
        T grad_score              = grad_attn_value * attn_weight;
        tmp_grad_score[thread_id] = grad_score;
        __syncthreads();

        // Reduce within block
        if(thread_id < real_row_num)
        {
            T sum = T(0.0f);
#pragma unroll
            for(int i = 0; i < max_seq_kv; i++)
                sum += tmp_grad_score[thread_id * max_seq_kv + i];
            reduce_grad_score[thread_id] = sum;
        }
        __syncthreads();

        grad_score -= attn_weight * reduce_grad_score[thread_id / max_seq_kv];

        // Apply causal mask and KV-padding mask
        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            // q_idx == 0; mask: k_idx > 0 || k_idx >= seq_kv
            if(k_idx > 0 || k_idx >= seq_kv)
                grad_score = T(0.0f);
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            if(k_idx >= seq_kv)
                grad_score = T(0.0f);
        }
        else
        {
            if(k_idx >= seq_kv)
                grad_score = T(0.0f);
        }

        grad_attn[cur_block_offset] = grad_score;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: compute_grad_qk_kernel
//
// Computes grad_Q = grad_scores @ K * scale
//           grad_K = grad_scores^T @ Q * scale
// ---------------------------------------------------------------------------

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_qk_kernel(const T* grad_scores,
                                       const T* Q,
                                       const T* K,
                                       T* grad_Q,
                                       T* grad_K,
                                       float scale,
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

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt];
    T grad_score_vals[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int seq_q_idx    = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= batch)
            continue;

        // Skip batches where actual Q seq is 0 — no grad_Q/grad_K to compute.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_q_idx == 0

        // workspace layout: [total_padded_q, head_num, max_seq_kv]
        int gs_offset = (q_storage_offset * Config::head_num + head_idx) * max_seq_kv;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            grad_score_vals[i] = grad_scores[gs_offset + i];

        // Compute grad_Q = grad_scores @ K * scale
        uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            store_dwordx4_tmp_var[i].x = 0;
            store_dwordx4_tmp_var[i].y = 0;
            store_dwordx4_tmp_var[i].z = 0;
            store_dwordx4_tmp_var[i].w = 0;
        }

        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                int k_idx = (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                            head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
                load_dwordx4_tmp_var[i] = *((uint4*)&K[k_idx]);
            }
#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
                    grad_score_vals[j] *
                    ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
            }
        }

#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            // grad_Q layout: [total_padded_seq_q, head_num, head_dim]
            T* grad_Q_ptr = &grad_Q[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                                    thread_head_offset + i * dwordx4_load_elt];
            for(int b = 0; b < dwordx4_load_elt; b++)
            {
                grad_Q_ptr[b] = ((T*)&store_dwordx4_tmp_var[i])[b] * scale;
            }
        }

        // Compute grad_K = grad_scores^T @ Q * scale
        // Q layout: [total_padded_seq_q, head_num, head_dim]
#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            load_dwordx4_tmp_var[i] =
                *((uint4*)&Q[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                             thread_head_offset + i * dwordx4_load_elt]);
        }

        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                T val = grad_score_vals[j] *
                        ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] *
                        T(scale);
                int grad_k_idx =
                    (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                    head_idx * head_dim + thread_head_offset + b;
                grad_K[grad_k_idx] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AttnBackwardKernelLauncher
//
// Orchestrates the 4-kernel backward pipeline:
//   1. compute_grad_v_kernel    (attn_weights^T @ grad_O)
//   2. compute_grad_attn_kernel (grad_O @ V^T)
//   3. softmax_backward_kernel
//   4. compute_grad_qk_kernel   (grad_scores @ K, grad_scores^T @ Q)
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnBackwardKernelLauncher
{
    // workspace layout: [total_padded_q, head_num, max_seq_kv]
    static size_t calc_workspace_size(int total_padded_q)
    {
        constexpr int head_num   = Config::head_num;
        constexpr int max_seq_kv = Config::max_seq_kv;
        return (size_t)total_padded_q * head_num * max_seq_kv * sizeof(T);
    }

    static void run_attn_bwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* grad_O,
                                    const T* attn_weights,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* grad_Q,
                                    T* grad_K,
                                    T* grad_V,
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

        dim3 block(warp_size);

        // Step 1: Compute grad_V = attn_weights^T @ grad_O — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_v = 16;
        dim3 grid_v((bs * seq_q * head_num + tasks_per_block_v - 1) / tasks_per_block_v);
        compute_grad_v_kernel<T, Config, tasks_per_block_v><<<grid_v, block>>>(
            attn_weights, grad_O, grad_V, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded, batch);

        // Step 2: Compute grad_attn = grad_O @ V^T — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_attn  = 16;
        constexpr int process_head_per_warp = warp_size / (head_dim / 64);
        dim3 grid_grad_attn(
            (bs * seq_q * head_num + tasks_per_block_attn * process_head_per_warp - 1) /
            (tasks_per_block_attn * process_head_per_warp));
        compute_grad_attn_kernel<T, Config, tasks_per_block_attn><<<grid_grad_attn, block>>>(
            grad_O, V, workspace, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded, batch);

        // Step 3: Softmax backward — grid covers [total_padded_q, head_num, max_seq_kv] elements
        constexpr int work_thread_num = Config::step2_block_size / max_seq_kv * max_seq_kv;
        uint32_t total_elt = (uint32_t)total_padded_q * head_num * max_seq_kv;
        dim3 grid_softmax((total_elt + work_thread_num - 1) / work_thread_num);
        dim3 block_softmax(Config::step2_block_size);
        softmax_backward_kernel<T, Config><<<grid_softmax, block_softmax>>>(
            attn_weights, dropout_mask, workspace, dropout_scale, cu_seqlens_kv,
            padded_q_to_batch, total_elt);

        // Step 4: Compute grad_Q and grad_K — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_qk = 4;
        dim3 grid_qk((bs * seq_q * head_num + tasks_per_block_qk - 1) / tasks_per_block_qk);
        compute_grad_qk_kernel<T, Config, tasks_per_block_qk><<<grid_qk, block>>>(
            workspace, Q, K, grad_Q, grad_K, scale, cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded, batch);
    }
};

}  // namespace small_seq_kernels
