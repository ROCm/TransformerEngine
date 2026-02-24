/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "ck_fused_attn/varlen_attn_common.hpp"

#define VARLEN_HIP_CHECK(call)                                                          \
    do {                                                                                \
        hipError_t err = (call);                                                        \
        if (err != hipSuccess) {                                                       \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
        }                                                                               \
    } while (0)

namespace ck_fused_attn {
namespace varlen {

// Kernel 1: Compute grad_V = attn_weights^T @ grad_O
template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_v_kernel(const T* attn_weights,
                                      const T* grad_O,
                                      T* grad_V,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int bs_rt,
                                      int hn_rt,
                                      int kv_stride)
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
    T attn[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        int batch_idx    = cur_idx / (seq_q * hn_rt);
        int seq_head_idx = cur_idx % (seq_q * hn_rt);
        int seq_q_idx    = seq_head_idx / hn_rt;
        int head_idx     = seq_head_idx % hn_rt;

        if(batch_idx >= bs_rt)
            continue;

        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[cur_idx * max_seq_kv + i];

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

#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                load_dwordx4_tmp_var[i] =
                    *((uint4*)&grad_O[(batch_idx * seq_q * hn_rt +
                                       seq_q_idx * hn_rt + head_idx) *
                                          head_dim +
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
                    (cu_seqlens_kv_padded[batch_idx] + j) * kv_stride +
                    head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
                *((uint4*)&grad_V[grad_v_idx]) = store_dwordx4_tmp_var[i];
            }
        }
    }
}

// Kernel 2: Compute grad_attn = grad_O @ V^T
template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_grad_attn_kernel(const T* grad_O,
                                         const T* V,
                                         T* grad_attn,
                                         const int* cu_seqlens_kv,
                                         const int* cu_seqlens_kv_padded,
                                         int bs_rt,
                                         int hn_rt,
                                         int kv_stride)
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
        int batch_idx     = cur_batch_idx / (seq_q * hn_rt);
        int seq_head_idx  = cur_batch_idx % (seq_q * hn_rt);
        int seq_idx       = seq_head_idx / hn_rt;
        int head_idx      = seq_head_idx % hn_rt;

        if(batch_idx >= bs_rt)
            continue;

        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

        float results[max_seq_kv];
        T fetch_grad_O[block_k];
        T fetch_V[block_k];

        T* grad_O_ptr = (T*)&grad_O[(batch_idx * seq_q * hn_rt +
                                     seq_idx * hn_rt + head_idx) *
                                    head_dim];

        const T* V_base =
            &V[cu_seqlens_kv_padded[batch_idx] * kv_stride + head_idx * head_dim];

        T* grad_attn_ptr = (T*)&grad_attn[cur_batch_idx * max_seq_kv];

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
                            *((uint4*)&V_base[kv_idx * kv_stride + dim_offset + k * 8]);
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
                            *((uint4*)&V_base[kv_idx * kv_stride + dim_offset + k * 4]);
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
        for(int i = seq_kv; i < max_seq_kv; i++)
        {
            grad_attn_ptr[i] = T(0.0f);
        }
    }
}

// Kernel 3: Apply softmax backward and mask
template <typename T, typename Config>
__global__ void softmax_backward_kernel(const T* attn_weights,
                                        const T* dropout_mask,
                                        T* grad_attn,
                                        float dropout_scale,
                                        const int* cu_seqlens_kv,
                                        int bs_rt,
                                        int hn_rt)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int seq_q              = Config::seq_q;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_grad_attn_size = seq_q * max_seq_kv;
    constexpr int valid_thread_range = block_size / per_grad_attn_size * per_grad_attn_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    const uint32_t total_elt         = (uint32_t)bs_rt * hn_rt * seq_q * max_seq_kv;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (int)(total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_grad_score[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T reduce_grad_score[row_num];

        int global_row_idx = cur_block_offset / max_seq_kv;
        int batch_idx      = global_row_idx / (seq_q * hn_rt);
        int k_idx          = cur_block_offset % max_seq_kv;

        int seq_kv = (batch_idx < bs_rt)
                         ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx])
                         : max_seq_kv;

        T grad_attn_value = grad_attn[cur_block_offset];
        if constexpr(Config::enable_dropout_mask)
        {
            grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
        }
        T attn_weight             = attn_weights[cur_block_offset];
        T grad_score              = grad_attn_value * attn_weight;
        tmp_grad_score[thread_id] = grad_score;
        __syncthreads();

        if(thread_id < real_row_num)
        {
            T sum = T(0.0f);
#pragma unroll
            for(int i = 0; i < max_seq_kv; i++)
            {
                sum += tmp_grad_score[thread_id * max_seq_kv + i];
            }
            reduce_grad_score[thread_id] = sum;
        }
        __syncthreads();

        grad_score -= attn_weight * reduce_grad_score[thread_id / max_seq_kv];

        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
            if(k_idx > q_idx || k_idx >= seq_kv)
            {
                grad_score = T(0.0f);
            }
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
            if(k_idx < q_idx || k_idx >= seq_kv)
            {
                grad_score = T(0.0f);
            }
        }
        else
        {
            if(k_idx >= seq_kv)
            {
                grad_score = T(0.0f);
            }
        }

        grad_attn[cur_block_offset] = grad_score;
    }
}

// Kernel 4: Compute grad_Q and grad_K
template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_qk_kernel(const T* grad_scores,
                                       const T* Q,
                                       const T* K,
                                       T* grad_Q,
                                       T* grad_K,
                                       float scale,
                                       const int* cu_seqlens_kv,
                                       const int* cu_seqlens_kv_padded,
                                       int bs_rt,
                                       int hn_rt,
                                       int kv_stride)
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

        int batch_idx    = cur_idx / (seq_q * hn_rt);
        int seq_head_idx = cur_idx % (seq_q * hn_rt);
        int seq_q_idx    = seq_head_idx / hn_rt;
        int head_idx     = seq_head_idx % hn_rt;

        if(batch_idx >= bs_rt)
            continue;

        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            grad_score_vals[i] = grad_scores[cur_idx * max_seq_kv + i];

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
                int k_idx = (cu_seqlens_kv_padded[batch_idx] + j) * kv_stride +
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
            T* grad_Q_ptr = &grad_Q[(batch_idx * seq_q * hn_rt +
                                     seq_q_idx * hn_rt + head_idx) *
                                        head_dim +
                                    thread_head_offset + i * dwordx4_load_elt];
            for(int b = 0; b < dwordx4_load_elt; b++)
            {
                grad_Q_ptr[b] = ((T*)&store_dwordx4_tmp_var[i])[b] * scale;
            }
        }

#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            load_dwordx4_tmp_var[i] = *((uint4*)&Q[(batch_idx * seq_q * hn_rt +
                                                    seq_q_idx * hn_rt + head_idx) *
                                                       head_dim +
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
                    (cu_seqlens_kv_padded[batch_idx] + j) * kv_stride +
                    head_idx * head_dim + thread_head_offset + b;
                grad_K[grad_k_idx] = val;
            }
        }
    }
}

template <typename T, typename Config>
struct AttnBackwardKernelLauncher
{
    static size_t calc_workspace_size(int bs_rt, int hn_rt)
    {
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;

        size_t workspace_size = (size_t)bs_rt * hn_rt * seq_q * max_seq_kv * sizeof(T);
        return workspace_size;
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
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    int bs_rt,
                                    int hn_rt,
                                    int kv_stride,
                                    hipStream_t stream = 0)
    {
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;
        constexpr int head_dim   = Config::head_dim;
        constexpr int warp_size  = 64;

        const int merge_bs = bs_rt * hn_rt;
        float scale            = sqr_dk_scale;
        float dropout_scale    = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        bool debug = std::getenv("NVTE_DEBUG_VARLEN_ATTN") != nullptr;

        dim3 block(warp_size);

        constexpr int tasks_per_block_v = 16;
        dim3 grid_v((merge_bs + tasks_per_block_v - 1) / tasks_per_block_v);
        if (debug) {
            std::cerr << "[run_attn_bwd_kernel] bs_rt=" << bs_rt << " hn_rt=" << hn_rt
                      << " merge_bs=" << merge_bs << " kv_stride=" << kv_stride << std::endl;
            std::cerr << "[run_attn_bwd_kernel] Launching compute_grad_v_kernel grid=" << grid_v.x << std::endl;
        }
        compute_grad_v_kernel<T, Config, tasks_per_block_v>
            <<<grid_v, block, 0, stream>>>(attn_weights, grad_O, grad_V,
                                           cu_seqlens_kv, cu_seqlens_kv_padded,
                                           bs_rt, hn_rt, kv_stride);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_bwd_kernel] FAULT after compute_grad_v_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_bwd_kernel] compute_grad_v_kernel completed OK" << std::endl;
        }

        constexpr int tasks_per_block_attn  = 16;
        constexpr int process_head_per_warp = warp_size / (head_dim / 64);
        dim3 grid_grad_attn(
            (merge_bs + tasks_per_block_attn * process_head_per_warp - 1) /
            (tasks_per_block_attn * process_head_per_warp));
        if (debug) {
            std::cerr << "[run_attn_bwd_kernel] Launching compute_grad_attn_kernel grid=" << grid_grad_attn.x << std::endl;
        }
        compute_grad_attn_kernel<T, Config, tasks_per_block_attn>
            <<<grid_grad_attn, block, 0, stream>>>(grad_O, V, workspace,
                                                    cu_seqlens_kv, cu_seqlens_kv_padded,
                                                    bs_rt, hn_rt, kv_stride);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_bwd_kernel] FAULT after compute_grad_attn_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_bwd_kernel] compute_grad_attn_kernel completed OK" << std::endl;
        }

        constexpr int work_thread_num =
            Config::step2_block_size / (seq_q * max_seq_kv) * (seq_q * max_seq_kv);
        dim3 grid_softmax(((size_t)merge_bs * seq_q * max_seq_kv + work_thread_num - 1) / work_thread_num);
        dim3 block_softmax(Config::step2_block_size);
        if (debug) {
            std::cerr << "[run_attn_bwd_kernel] Launching softmax_backward_kernel grid=" << grid_softmax.x << std::endl;
        }
        softmax_backward_kernel<T, Config><<<grid_softmax, block_softmax, 0, stream>>>(
            attn_weights, dropout_mask, workspace, dropout_scale,
            cu_seqlens_kv, bs_rt, hn_rt);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_bwd_kernel] FAULT after softmax_backward_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_bwd_kernel] softmax_backward_kernel completed OK" << std::endl;
        }

        constexpr int tasks_per_block_qk = 4;
        dim3 grid_qk((merge_bs + tasks_per_block_qk - 1) / tasks_per_block_qk);
        if (debug) {
            std::cerr << "[run_attn_bwd_kernel] Launching compute_grad_qk_kernel grid=" << grid_qk.x << std::endl;
        }
        compute_grad_qk_kernel<T, Config, tasks_per_block_qk><<<grid_qk, block, 0, stream>>>(
            workspace, Q, K, grad_Q, grad_K, scale,
            cu_seqlens_kv, cu_seqlens_kv_padded,
            bs_rt, hn_rt, kv_stride);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_bwd_kernel] FAULT after compute_grad_qk_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_bwd_kernel] compute_grad_qk_kernel completed OK" << std::endl;
        }
    }
};

template struct AttnBackwardKernelLauncher<hip_bfloat16, ConfigHeadDim64>;
template struct AttnBackwardKernelLauncher<hip_bfloat16, ConfigHeadDim128>;

}  // namespace varlen
}  // namespace ck_fused_attn

#include "ck_fused_attn/varlen_attn.hpp"

namespace ck_fused_attn {

size_t varlen_attn_bwd_workspace_size(size_t b, size_t h_q, size_t head_dim) {
    (void)head_dim;
    if (head_dim != 64 && head_dim != 128) {
        throw std::runtime_error("varlen_attn: unsupported head_dim (only 64 or 128)");
    }
    return b * h_q * 1 * 16 * sizeof(hip_bfloat16);
}

void run_varlen_attn_bwd(const void* Q,
                         const void* K,
                         const void* V,
                         const void* grad_O,
                         const void* attn_weights,
                         const void* dropout_mask,
                         float dropout_p,
                         float sqr_dk_scale,
                         void* grad_Q,
                         void* grad_K,
                         void* grad_V,
                         void* workspace,
                         const int* cu_seqlens_kv,
                         const int* cu_seqlens_kv_padded,
                         size_t b,
                         size_t h_q,
                         size_t head_dim,
                         size_t kv_stride,
                         hipStream_t stream) {
    const auto* q = static_cast<const hip_bfloat16*>(Q);
    const auto* k = static_cast<const hip_bfloat16*>(K);
    const auto* v = static_cast<const hip_bfloat16*>(V);
    const auto* go = static_cast<const hip_bfloat16*>(grad_O);
    const auto* attn = static_cast<const hip_bfloat16*>(attn_weights);
    const auto* mask = static_cast<const hip_bfloat16*>(dropout_mask);
    auto* gq = static_cast<hip_bfloat16*>(grad_Q);
    auto* gk = static_cast<hip_bfloat16*>(grad_K);
    auto* gv = static_cast<hip_bfloat16*>(grad_V);
    auto* wks = static_cast<hip_bfloat16*>(workspace);

    if (std::getenv("NVTE_DEBUG_VARLEN_ATTN")) {
        std::cerr << "[varlen_attn_bwd] ENTRY: b=" << b << " h_q=" << h_q
                  << " head_dim=" << head_dim << " kv_stride=" << kv_stride << std::endl;
    }

    const int bs_rt = static_cast<int>(b);
    const int hn_rt = static_cast<int>(h_q);
    const int kvs   = static_cast<int>(kv_stride);

    if (head_dim == 64) {
        varlen::AttnBackwardKernelLauncher<hip_bfloat16, varlen::ConfigHeadDim64>::run_attn_bwd_kernel(
            q, k, v, go, attn, mask, dropout_p, sqr_dk_scale,
            gq, gk, gv, wks, cu_seqlens_kv, cu_seqlens_kv_padded,
            bs_rt, hn_rt, kvs, stream);
        return;
    }
    if (head_dim == 128) {
        varlen::AttnBackwardKernelLauncher<hip_bfloat16, varlen::ConfigHeadDim128>::run_attn_bwd_kernel(
            q, k, v, go, attn, mask, dropout_p, sqr_dk_scale,
            gq, gk, gv, wks, cu_seqlens_kv, cu_seqlens_kv_padded,
            bs_rt, hn_rt, kvs, stream);
        return;
    }
    throw std::runtime_error("varlen_attn: unsupported head_dim (only 64 or 128)");
}

}  // namespace ck_fused_attn
