// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <iomanip>
#include <map>
#include <cstdlib>
#include <string>

// clang-format off
// /opt/rocm/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx950 -o attn_fwd attn_fwd.cpp && ./attn_fwd
// clang-format on

#define HIP_CHECK(call)                                                                    \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while(0)

enum class CausalMaskType
{
    DISABLE      = 0,
    TOP_LEFT     = 1,
    BOTTOM_RIGHT = 2
};

std::map<CausalMaskType, std::string> CausalMaskTypeName = {
    {CausalMaskType::DISABLE, "DISABLE"},
    {CausalMaskType::TOP_LEFT, "TOP_LEFT"},
    {CausalMaskType::BOTTOM_RIGHT, "BOTTOM_RIGHT"}};

template <int BS,
          int HEAD_NUM,
          int SEQ_Q,
          int MAX_SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          CausalMaskType MAKS_TYPE = CausalMaskType::DISABLE>
struct FmhaKernelConfig
{
    static constexpr int bs                        = BS;
    static constexpr int head_num                  = HEAD_NUM;
    static constexpr int seq_q                     = SEQ_Q;
    static constexpr int max_seq_kv                = MAX_SEQ_KV;
    static constexpr int head_dim                  = HEAD_DIM;
    static constexpr int step2_block_size          = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask      = ENABLE_DROPOUT_MASK;
    static constexpr enum CausalMaskType mask_type = MAKS_TYPE;
};

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_scores_kernel(const T* Q,
                                      const T* K,
                                      T* scores,
                                      float scale,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded)
{
    constexpr int seq_q = Config::seq_q;
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
        // Layout: [batch, seq_len, head_num, head_dim]
        // cur_batch_idx represents the combined index for (batch * seq_q(equal to 1) * head_num)
        int batch_idx    = cur_batch_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_batch_idx % (Config::seq_q * Config::head_num);
        int seq_idx      = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= Config::bs)
            continue;

        // Get actual sequence length for this batch
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];

        float results[max_seq_kv];
        T fetch_Q[block_k];
        T fetch_K[block_k];
        // Q: [batch, seq_q, head_num, head_dim]
        T* Q_ptr = (T*)&Q[(batch_idx * Config::seq_q * Config::head_num +
                           seq_idx * Config::head_num + head_idx) *
                          head_dim];
        // K: [total_padded_seq_kv, head_num, head_dim]
        T* K_ptr     = (T*)&K[(kv_offset * Config::head_num + head_idx) * head_dim];
        T* score_ptr = (T*)&scores[cur_batch_idx * max_seq_kv];
        uint4 ls_dwordx4_tmp_var;
        for(int i = 0; i < seq_kv; i++)
            results[i] = 0.0f;
        for(int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k)
        {
            if constexpr(std::is_same<T, hip_bfloat16>::value || std::is_same<T, __half>::value)
            {
                for(int k = 0; k < block_k / 8; k++)
                {
                    ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 8]);
                    fetch_Q[k * 8 + 0] = ((T*)&ls_dwordx4_tmp_var.x)[0];
                    fetch_Q[k * 8 + 1] = ((T*)&ls_dwordx4_tmp_var.x)[1];
                    fetch_Q[k * 8 + 2] = ((T*)&ls_dwordx4_tmp_var.y)[0];
                    fetch_Q[k * 8 + 3] = ((T*)&ls_dwordx4_tmp_var.y)[1];
                    fetch_Q[k * 8 + 4] = ((T*)&ls_dwordx4_tmp_var.z)[0];
                    fetch_Q[k * 8 + 5] = ((T*)&ls_dwordx4_tmp_var.z)[1];
                    fetch_Q[k * 8 + 6] = ((T*)&ls_dwordx4_tmp_var.w)[0];
                    fetch_Q[k * 8 + 7] = ((T*)&ls_dwordx4_tmp_var.w)[1];
                }
                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 8; k++)
                    {
                        // K layout: [batch, seq_kv, head_num, head_dim]
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * Config::head_num * head_dim +
                                                              dim_offset + k * 8]);
                        fetch_K[k * 8 + 0] = ((T*)&ls_dwordx4_tmp_var.x)[0];
                        fetch_K[k * 8 + 1] = ((T*)&ls_dwordx4_tmp_var.x)[1];
                        fetch_K[k * 8 + 2] = ((T*)&ls_dwordx4_tmp_var.y)[0];
                        fetch_K[k * 8 + 3] = ((T*)&ls_dwordx4_tmp_var.y)[1];
                        fetch_K[k * 8 + 4] = ((T*)&ls_dwordx4_tmp_var.z)[0];
                        fetch_K[k * 8 + 5] = ((T*)&ls_dwordx4_tmp_var.z)[1];
                        fetch_K[k * 8 + 6] = ((T*)&ls_dwordx4_tmp_var.w)[0];
                        fetch_K[k * 8 + 7] = ((T*)&ls_dwordx4_tmp_var.w)[1];
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

template <typename T, typename Config>
__global__ void apply_mask_and_softmax_kernel(T* scores,
                                              const T* dropout_mask,
                                              float dropout_scale,
                                              const int* cu_seqlens_kv)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int seq_q              = Config::seq_q;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_score_size     = seq_q * max_seq_kv;
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    constexpr uint32_t total_elt     = Config::bs * Config::head_num * seq_q * max_seq_kv;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_scores[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T row_max[row_num];
        __shared__ T row_sum[row_num];

        // Determine batch_idx for this thread
        int global_row_idx = cur_block_offset / max_seq_kv;
        int batch_idx      = global_row_idx / (Config::seq_q * Config::head_num);
        int k_idx          = cur_block_offset % max_seq_kv;

        // Get actual sequence length for this batch
        int seq_kv = (batch_idx < Config::bs)
                         ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx])
                         : max_seq_kv;

        T score_value         = scores[cur_block_offset];
        tmp_scores[thread_id] = score_value;

        // Apply causal mask before softmax
        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
            if(k_idx > q_idx || k_idx >= seq_kv)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
            if(k_idx < q_idx || k_idx >= seq_kv)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
        }
        else
        {
            // No causal mask, but still mask padding positions
            if(k_idx >= seq_kv)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
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

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 8>
__global__ void compute_output_kernel(const T* attn_weights,
                                      const T* V,
                                      T* O,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded)
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

        // Layout: [batch, seq_len, head_num, head_dim]
        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int seq_q_idx    = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= Config::bs)
            continue;

        // Get actual sequence length for this batch
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];

#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            store_dwordx4_tmp_var[i].x = 0;
            store_dwordx4_tmp_var[i].y = 0;
            store_dwordx4_tmp_var[i].z = 0;
            store_dwordx4_tmp_var[i].w = 0;
        }
        // ((T *)&store_dwordx4_tmp_var)[i] = 0.0f;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[cur_idx * max_seq_kv + i];
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
            // O layout: [batch, seq_q, head_num, head_dim]
            *((uint4*)&O[(batch_idx * Config::seq_q * Config::head_num +
                          seq_q_idx * Config::head_num + head_idx) *
                             head_dim +
                         thread_head_offset + i * dwordx4_load_elt]) = store_dwordx4_tmp_var[i];
    }
}

template <typename T, typename Config>
struct AttnForwardKernelLauncher
{
    static size_t calc_workspace_size()
    {
        constexpr int bs         = Config::bs;
        constexpr int head_num   = Config::head_num;
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;

        size_t workspace_size = bs * head_num * seq_q * max_seq_kv * sizeof(T);
        return workspace_size;
    }

    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
                                    T* workspace,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded)
    {
        constexpr int bs         = Config::bs;
        constexpr int head_num   = Config::head_num;
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;
        constexpr int head_dim   = Config::head_dim;
        constexpr int warp_size  = 64;

        constexpr int merge_bs = bs * head_num;
        float scale            = sqr_dk_scale;
        float dropout_scale    = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        constexpr int kernel1_threads = 64;

        dim3 block(kernel1_threads);
        dim3 grid(merge_bs / kernel1_threads);
        compute_scores_kernel<T, Config, 1>
            <<<grid, block>>>(Q, K, workspace, scale, cu_seqlens_kv, cu_seqlens_kv_padded);

        // Step 2: Apply mask and softmax (with dropout)
        constexpr int work_thread_num =
            Config::step2_block_size / (seq_q * max_seq_kv) * (seq_q * max_seq_kv);
        dim3 grid2((merge_bs * seq_q * max_seq_kv + work_thread_num - 1) / work_thread_num);
        dim3 block2(Config::step2_block_size);
        apply_mask_and_softmax_kernel<T, Config>
            <<<grid2, block2>>>(workspace, dropout_mask, dropout_scale, cu_seqlens_kv);

        constexpr int kernel3_block_k       = 8;
        constexpr int kernel3_threads       = 64;
        constexpr int process_head_per_warp = warp_size / (head_dim / kernel3_block_k);

        dim3 block3(kernel3_threads);
        dim3 grid3((merge_bs / process_head_per_warp + 2 - 1) / 2);
        compute_output_kernel<T, Config, 2, kernel3_block_k>
            <<<grid3, block3>>>(workspace, V, O, cu_seqlens_kv, cu_seqlens_kv_padded);
    }
};

/**
 * Multi-Head Attention Forward Pass (CPU Reference Implementation)
 */
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
                  int q_seq,
                  int max_kv_seq,
                  int head_dim,
                  CausalMaskType mask_type,
                  const int* cu_seqlens_kv,
                  const int* cu_seqlens_kv_padded)
{

    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Allocate temporary buffers
    std::vector<T> K_T(head_dim * max_kv_seq);
    std::vector<T> scores(q_seq * max_kv_seq);
    std::vector<T> attn_probs(q_seq * max_kv_seq);

    // Initialize output to zero
    std::memset(O, 0, batch * head_num * q_seq * head_dim * sizeof(T));
    if(attn_weights != nullptr)
    {
        std::memset(attn_weights, 0, batch * head_num * q_seq * max_kv_seq * sizeof(T));
    }

    // Process each batch and head
    // Layout: [batch, seq_len, head_num, head_dim]
    for(int b = 0; b < batch; b++)
    {
        // Get actual sequence length for this batch
        int kv_seq    = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];
        int kv_offset = cu_seqlens_kv_padded[b];

        for(int h = 0; h < head_num; h++)
        {
            // For each query position
            for(int q_idx = 0; q_idx < q_seq; q_idx++)
            {
                int offset_Q    = (b * q_seq * head_num + q_idx * head_num + h) * head_dim;
                int offset_O    = (b * q_seq * head_num + q_idx * head_num + h) * head_dim;
                int offset_attn = (b * q_seq * head_num + q_idx * head_num + h) * max_kv_seq;
                int offset_dropout =
                    dropout_mask ? (b * q_seq * head_num + q_idx * head_num + h) * max_kv_seq : 0;

                const T* Q_ptr       = Q + offset_Q;
                const T* dropout_ptr = dropout_mask ? dropout_mask + offset_dropout : nullptr;

                T* O_ptr    = O + offset_O;
                T* attn_ptr = attn_weights ? attn_weights + offset_attn : nullptr;

                // Step 1: Compute scores = Q @ K^T / sqrt(d_k)
                // Q: [1, head_dim], K: [kv_seq, head_dim] -> scores: [1, kv_seq]
                // Build K matrix for this head: K is [total_padded_seq_kv, head_num, head_dim]
                for(int kv_idx = 0; kv_idx < kv_seq; kv_idx++)
                {
                    int k_offset   = ((kv_offset + kv_idx) * head_num + h) * head_dim;
                    const T* K_ptr = K + k_offset;
                    float sum      = 0.0f;
                    for(int d = 0; d < head_dim; d++)
                    {
                        sum += float(Q_ptr[d]) * float(K_ptr[d]);
                    }
                    scores[kv_idx] = T(sum * scale);
                }

                // Step 2: Apply causal mask
                if(mask_type == CausalMaskType::TOP_LEFT)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j > q_idx)
                        {
                            scores[j] = T(-1e9f);
                        }
                    }
                }
                else if(mask_type == CausalMaskType::BOTTOM_RIGHT)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j < q_idx)
                        {
                            scores[j] = T(-1e9f);
                        }
                    }
                }

                // Step 3: Softmax
                // Find max for numerical stability
                float max_val = -1e9f;
                for(int j = 0; j < kv_seq; j++)
                {
                    max_val = std::max(max_val, float(scores[j]));
                }

                // Compute exp and sum
                float sum = 0.0f;
                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[j] = T(std::exp(float(scores[j]) - max_val));
                    sum += float(attn_probs[j]);
                }

                // Normalize
                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[j] = T(float(attn_probs[j]) / sum);
                }

                // Step 4: Apply dropout
                if(dropout_p > 0.0f && dropout_ptr != nullptr)
                {
                    for(int i = 0; i < kv_seq; i++)
                    {
                        attn_probs[i] =
                            T(float(attn_probs[i]) * float(dropout_ptr[i]) * dropout_scale);
                    }
                }

                // Save attention weights if requested
                if(attn_ptr != nullptr)
                {
                    std::memcpy(attn_ptr, attn_probs.data(), kv_seq * sizeof(T));
                }

                // Step 5: Compute output = attn_probs @ V
                // attn_probs: [1, kv_seq], V: [kv_seq, head_dim] -> O: [1, head_dim]
                for(int d = 0; d < head_dim; d++)
                {
                    float sum = 0.0f;
                    for(int kv_idx = 0; kv_idx < kv_seq; kv_idx++)
                    {
                        int v_offset = ((kv_offset + kv_idx) * head_num + h) * head_dim;
                        sum += float(attn_probs[kv_idx]) * float(V[v_offset + d]);
                    }
                    O_ptr[d] = T(sum);
                }
            }
        }
    }
}

/**
 * Test run_attn_fwd_kernel correctness and bandwidth
 */
template <typename DataType, typename Config>
void test_run_attn_fwd_kernel(
    float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnForwardKernelLauncher<DataType, Config>;

    constexpr int bs         = Config::bs;
    constexpr int head_num   = Config::head_num;
    constexpr int seq_q      = Config::seq_q;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;

    // Generate variable kv-seq lengths following normal distribution (mean=4, range 2-16)
    std::random_device rd;
    std::mt19937 gen(42);                                   // Fixed seed for reproducibility
    std::normal_distribution<float> normal_dis(4.0f, 2.0f); // mean=4, std=2
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<int> h_cu_seqlens_kv(bs + 1);
    std::vector<int> h_cu_seqlens_kv_padded(bs + 1);

    h_cu_seqlens_kv[0]        = 0;
    h_cu_seqlens_kv_padded[0] = 0;

    int total_actual_kv_seq = 0;
    int total_padded_kv_seq = 0;
    std::uniform_int_distribution<int> pad_dis(0, 5); // Random padding 0-5

    for(int b = 0; b < bs; b++)
    {
        // Generate kv-seq length from normal distribution, clamped to [2, max_seq_kv]
        int kv_len = static_cast<int>(std::round(normal_dis(gen)));
        kv_len     = std::max(2, std::min(max_seq_kv, kv_len));
        // Random padding (not aligned to any specific boundary)
        int random_pad = pad_dis(gen);
        int padded_len = kv_len + random_pad > max_seq_kv ? max_seq_kv : kv_len + random_pad;
        total_padded_kv_seq += padded_len;
        total_actual_kv_seq += kv_len;
        h_cu_seqlens_kv[b + 1]        = total_actual_kv_seq;
        h_cu_seqlens_kv_padded[b + 1] = total_padded_kv_seq;
    }

    // Calculate sizes
    size_t size_Q            = bs * head_num * seq_q * head_dim;
    size_t size_K            = total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = total_padded_kv_seq * head_num * head_dim;
    size_t size_O            = bs * head_num * seq_q * head_dim;
    size_t size_dropout_mask = bs * head_num * seq_q * max_seq_kv;

    // Allocate host memory
    std::vector<DataType> h_Q(size_Q);
    std::vector<DataType> h_K(size_K);
    std::vector<DataType> h_V(size_V);
    std::vector<DataType> h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_O_gpu(size_O);
    std::vector<DataType> h_O_cpu(size_O);

    // Initialize with random data
    for(size_t i = 0; i < size_Q; i++)
        h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++)
        h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++)
        h_V[i] = DataType(dis(gen));

    // Initialize dropout mask (1 = keep, 0 = drop)
    for(size_t i = 0; i < size_dropout_mask; i++)
    {
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);
    }

    // Compute CPU reference
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_forward(h_Q.data(),
                     h_K.data(),
                     h_V.data(),
                     Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                     dropout_p,
                     h_O_cpu.data(),
                     static_cast<DataType*>(nullptr),
                     bs,
                     head_num,
                     seq_q,
                     max_seq_kv,
                     head_dim,
                     Config::mask_type,
                     h_cu_seqlens_kv.data(),
                     h_cu_seqlens_kv_padded.data());

    // Allocate device memory
    DataType *d_Q, *d_K, *d_V;
    DataType* d_dropout_mask;
    DataType *d_O, *d_workspace;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv_padded, (bs + 1) * sizeof(int)));

    size_t workspace_size = Launcher::calc_workspace_size();
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask,
                        h_dropout_mask.data(),
                        size_dropout_mask * sizeof(DataType),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_cu_seqlens_kv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded,
                        h_cu_seqlens_kv_padded.data(),
                        (bs + 1) * sizeof(int),
                        hipMemcpyHostToDevice));

    // Warmup runs
    for(int i = 0; i < warmup_iters; i++)
    {
        Launcher::run_attn_fwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_O,
                                      d_workspace,
                                      d_cu_seqlens_kv,
                                      d_cu_seqlens_kv_padded);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < test_iters; i++)
    {
        Launcher::run_attn_fwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_O,
                                      d_workspace,
                                      d_cu_seqlens_kv,
                                      d_cu_seqlens_kv_padded);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    // Calculate TFLOPS
    // Main matrix operations in attention forward:
    // 1. scores = Q @ K^T: [q_seq, head_dim] @ [head_dim, kv_seq]
    // 2. O = attn_weights @ V: [q_seq, kv_seq] @ [kv_seq, head_dim]
    // Each matmul: FLOPs = 2 * M * N * K (multiply-add)
    // Calculate average kv_seq across batches
    double avg_kv_seq           = static_cast<double>(total_actual_kv_seq) / bs;
    double flops_per_batch_head = 2.0 * seq_q * avg_kv_seq * head_dim + // scores
                                  2.0 * seq_q * head_dim * avg_kv_seq;  // O
    double total_flops = flops_per_batch_head * bs * head_num;
    double tflops      = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    // Copy results back
    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));

    // Check correctness
    auto check_results = [&](const std::vector<DataType>& gpu,
                             const std::vector<DataType>& cpu,
                             const std::string& name,
                             float tolerance = 1e-1) {
        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t diff_count  = 0;

        for(size_t i = 0; i < gpu.size(); i++)
        {
            float diff     = std::abs(float(gpu[i]) - float(cpu[i]));
            float rel_diff = diff / (std::abs(float(cpu[i])) + 1e-6f);
            max_diff       = std::max(max_diff, diff);
            max_rel_diff   = std::max(max_rel_diff, rel_diff);
            if(rel_diff > tolerance)
            {
                if(dump_err && diff_count < 100)
                    std::cout << name << " mismatch at index " << i << ": GPU=" << std::fixed
                              << std::setprecision(10) << static_cast<float>(gpu[i])
                              << ", CPU=" << std::fixed << std::setprecision(10)
                              << static_cast<float>(cpu[i]) << ", abs_diff=" << diff
                              << ", rel_diff=" << rel_diff << std::endl;
                diff_count++;
            }
        }

        std::cout << name << " check:" << std::endl;
        std::cout << "  Max absolute diff: " << max_diff << std::endl;
        std::cout << "  Max relative diff: " << max_rel_diff << std::endl;
        std::cout << "  Elements exceeding tolerance: " << diff_count << " / " << gpu.size()
                  << std::endl;
        std::cout << "  Status: " << (max_rel_diff < tolerance ? "PASS" : "FAIL") << std::endl;
    };

    // Calculate bandwidth
    size_t bytes_read = (size_Q + size_K + size_V) * sizeof(DataType);
    if(Config::enable_dropout_mask)
        bytes_read += size_dropout_mask * sizeof(DataType);

    size_t bytes_write    = size_O * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // Print results
    std::cout << "\n===== run_attn_fwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Q sequence length: " << seq_q << std::endl;
    std::cout << "  KV sequence length (avg): " << std::fixed << std::setprecision(2) << avg_kv_seq
              << " (max: " << max_seq_kv << ")" << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Mask: " << (CausalMaskTypeName[Config::mask_type]) << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_results(h_O_gpu, h_O_cpu, "Output");
        std::cout << std::endl;
    }
    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read: " << std::fixed << std::setprecision(2) << bytes_read / 1e6
              << " MB" << std::endl;
    std::cout << "  Total data write: " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Workspace size: " << workspace_size / 1e6 << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
              << std::endl;
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << std::endl;
    std::cout << "VARLEN_ATTN_FWD_MS=" << std::fixed << avg_time_ms << std::endl;
    std::cout << "====================================\n" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_kv));
    HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

// Template metaprogramming: recursive template to iterate over SEQ_KV values
template <int SEQ_KV, int MAX_SEQ_KV>
struct TestRunner
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int SEQ_Q,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              SEQ_Q,
                                              SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_fwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);

        // Recursive call for next SEQ_KV value
        TestRunner<SEQ_KV + 1, MAX_SEQ_KV>::template run<DataType,
                                                         BS,
                                                         HEAD_NUM,
                                                         SEQ_Q,
                                                         HEAD_DIM,
                                                         STEP2_BLOCK_SIZE,
                                                         ENABLE_DROPOUT_MASK,
                                                         MASK_TYPE>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

// Termination condition: when SEQ_KV reaches MAX_SEQ_KV
template <int MAX_SEQ_KV>
struct TestRunner<MAX_SEQ_KV, MAX_SEQ_KV>
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int SEQ_Q,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              SEQ_Q,
                                              MAX_SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_fwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

// Dispatch helper: run one (s_kv) config for benchmark mode (same configs as profile_smallseq_rocm.py)
#define VARLEN_FWD_RUN(SKV)                                                                    \
    case SKV:                                                                                  \
        TestRunner<SKV, SKV>::run<DT, BS, 16, SQ, 128, 256, false, CausalMaskType::DISABLE>(   \
            0, 3, 100, 0, 0);                                                                  \
        break

int main(int argc, char const* argv[])
{
    // Benchmark mode: attn_fwd b s_q s_kv h_q h_kv d_qk d_v dtype  (same configs as profile_smallseq_rocm)
    if(argc >= 9)
    {
        int b     = std::atoi(argv[1]);
        int s_q   = std::atoi(argv[2]);
        int s_kv  = std::atoi(argv[3]);
        int h_q   = std::atoi(argv[4]);
        int h_kv  = std::atoi(argv[5]);
        int d_qk  = std::atoi(argv[6]);
        int d_v   = std::atoi(argv[7]);
        std::string dtype(argv[8]);
        if(h_q != 16 || h_kv != 16 || d_qk != 128 || d_v != 128 || s_q != 1 || s_kv < 2 || s_kv > 16)
        {
            std::cerr << "Unsupported config: s_q=1, h_q=h_kv=16, d_qk=d_v=128, s_kv in [2,16] required."
                      << std::endl;
            return 1;
        }
        if(b == 4000 && s_q == 1)
        {
#define BS 4000
#define SQ 1
            if(dtype == "bf16" || dtype == "bfloat16")
            {
                using DT = hip_bfloat16;
                switch(s_kv)
                {
                    VARLEN_FWD_RUN(2);
                    VARLEN_FWD_RUN(3);
                    VARLEN_FWD_RUN(4);
                    VARLEN_FWD_RUN(5);
                    VARLEN_FWD_RUN(6);
                    VARLEN_FWD_RUN(7);
                    VARLEN_FWD_RUN(8);
                    VARLEN_FWD_RUN(9);
                    VARLEN_FWD_RUN(10);
                    VARLEN_FWD_RUN(11);
                    VARLEN_FWD_RUN(12);
                    VARLEN_FWD_RUN(13);
                    VARLEN_FWD_RUN(14);
                    VARLEN_FWD_RUN(15);
                    VARLEN_FWD_RUN(16);
                default:
                    std::cerr << "s_kv must be 2..16" << std::endl;
                    return 1;
                }
            }
            else if(dtype == "fp16" || dtype == "float16")
            {
                using DT = __half;  // same as TE fused_attn_smallseq for apples-to-apples comparison
                switch(s_kv)
                {
                    VARLEN_FWD_RUN(2);
                    VARLEN_FWD_RUN(3);
                    VARLEN_FWD_RUN(4);
                    VARLEN_FWD_RUN(5);
                    VARLEN_FWD_RUN(6);
                    VARLEN_FWD_RUN(7);
                    VARLEN_FWD_RUN(8);
                    VARLEN_FWD_RUN(9);
                    VARLEN_FWD_RUN(10);
                    VARLEN_FWD_RUN(11);
                    VARLEN_FWD_RUN(12);
                    VARLEN_FWD_RUN(13);
                    VARLEN_FWD_RUN(14);
                    VARLEN_FWD_RUN(15);
                    VARLEN_FWD_RUN(16);
                default:
                    std::cerr << "s_kv must be 2..16" << std::endl;
                    return 1;
                }
            }
            else
            {
                std::cerr << "dtype must be bf16 or fp16" << std::endl;
                return 1;
            }
#undef BS
#undef SQ
        }
        else
        {
            std::cerr << "Unsupported (b,s_q,s_kv); only (4000,1,2..16) supported (s_q must be 1, s_kv in [2,16])."
                      << std::endl;
            return 1;
        }
        return 0;
    }

    // Default: correctness + performance tests (batch_size=4000, heads=16, same as profile_smallseq_rocm)
    std::cout << "\n========== Correctness Test ==========" << std::endl;
    TestRunner<2, 16>::run<float, 4000, 16, 1, 128, 256, false, CausalMaskType::DISABLE>(
        0, 1, 1, 1, 1);

    std::cout << "\n========== Performance Test ==========" << std::endl;
    TestRunner<2, 16>::run<hip_bfloat16, 4000, 16, 1, 128, 256, false, CausalMaskType::DISABLE>(
        0, 3, 5, 0, 0);

    return 0;
}
