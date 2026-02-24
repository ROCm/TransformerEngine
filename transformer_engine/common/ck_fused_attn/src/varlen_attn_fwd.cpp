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
#include <algorithm>

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

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_scores_kernel(const T* Q,
                                      const T* K,
                                      T* scores,
                                      float scale,
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
        int batch_idx    = cur_batch_idx / (seq_q * hn_rt);
        int seq_head_idx = cur_batch_idx % (seq_q * hn_rt);
        int seq_idx      = seq_head_idx / hn_rt;
        int head_idx     = seq_head_idx % hn_rt;

        if(batch_idx >= bs_rt)
            continue;

        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];

        float results[max_seq_kv];
        T fetch_Q[block_k];
        T fetch_K[block_k];
        T* Q_ptr = (T*)&Q[(batch_idx * seq_q * hn_rt +
                           seq_idx * hn_rt + head_idx) *
                          head_dim];
        T* K_ptr     = (T*)&K[kv_offset * kv_stride + head_idx * head_dim];
        T* score_ptr = (T*)&scores[cur_batch_idx * max_seq_kv];
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
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * kv_stride +
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
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * kv_stride +
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
                                              const int* cu_seqlens_kv,
                                              int bs_rt,
                                              int hn_rt)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int seq_q              = Config::seq_q;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_score_size     = seq_q * max_seq_kv;
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    const uint32_t total_elt         = (uint32_t)bs_rt * hn_rt * seq_q * max_seq_kv;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (int)(total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_scores[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T row_max[row_num];
        __shared__ T row_sum[row_num];

        int global_row_idx = cur_block_offset / max_seq_kv;
        int batch_idx      = global_row_idx / (seq_q * hn_rt);
        int k_idx          = cur_block_offset % max_seq_kv;

        int seq_kv = (batch_idx < bs_rt)
                         ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx])
                         : max_seq_kv;

        T score_value         = scores[cur_block_offset];
        tmp_scores[thread_id] = score_value;

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
            if(k_idx >= seq_kv)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
        }
        __syncthreads();

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

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt],
        store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
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
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[cur_idx * max_seq_kv + i];
        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                load_dwordx4_tmp_var[i] =
                    *((uint4*)&V[(kv_offset + j) * kv_stride + head_idx * head_dim +
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
            *((uint4*)&O[(batch_idx * seq_q * hn_rt +
                          seq_q_idx * hn_rt + head_idx) *
                             head_dim +
                         thread_head_offset + i * dwordx4_load_elt]) = store_dwordx4_tmp_var[i];
    }
}

template <typename T, typename Config>
struct AttnForwardKernelLauncher
{
    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
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

        constexpr int kernel1_threads = 64;

        dim3 block(kernel1_threads);
        dim3 grid((merge_bs + kernel1_threads - 1) / kernel1_threads);
        if (debug) {
            std::cerr << "[run_attn_fwd_kernel] bs_rt=" << bs_rt << " hn_rt=" << hn_rt
                      << " merge_bs=" << merge_bs << " kv_stride=" << kv_stride << std::endl;
            std::cerr << "[run_attn_fwd_kernel] Kernel1 (compute_scores): grid=" << grid.x
                      << " block=" << block.x << std::endl;
            std::cerr << "[run_attn_fwd_kernel] Launching compute_scores_kernel..." << std::endl;
        }
        compute_scores_kernel<T, Config, 1>
            <<<grid, block, 0, stream>>>(Q, K, workspace, scale,
                                         cu_seqlens_kv, cu_seqlens_kv_padded,
                                         bs_rt, hn_rt, kv_stride);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_fwd_kernel] FAULT after compute_scores_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_fwd_kernel] compute_scores_kernel completed OK" << std::endl;
        }

        constexpr int work_thread_num =
            Config::step2_block_size / (seq_q * max_seq_kv) * (seq_q * max_seq_kv);
        dim3 grid2(((size_t)merge_bs * seq_q * max_seq_kv + work_thread_num - 1) / work_thread_num);
        dim3 block2(Config::step2_block_size);
        if (debug) {
            std::cerr << "[run_attn_fwd_kernel] Kernel2 (softmax): grid2=" << grid2.x
                      << " block2=" << block2.x << std::endl;
            std::cerr << "[run_attn_fwd_kernel] Launching apply_mask_and_softmax_kernel..." << std::endl;
        }
        apply_mask_and_softmax_kernel<T, Config>
            <<<grid2, block2, 0, stream>>>(workspace, dropout_mask, dropout_scale,
                                           cu_seqlens_kv, bs_rt, hn_rt);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_fwd_kernel] FAULT after apply_mask_and_softmax_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_fwd_kernel] apply_mask_and_softmax_kernel completed OK" << std::endl;
        }

        constexpr int kernel3_block_k       = 8;
        constexpr int kernel3_threads       = 64;
        constexpr int process_head_per_warp = warp_size / (head_dim / kernel3_block_k);

        dim3 block3(kernel3_threads);
        dim3 grid3((merge_bs + process_head_per_warp * 2 - 1) / (process_head_per_warp * 2));
        if (debug) {
            std::cerr << "[run_attn_fwd_kernel] Kernel3 (compute_output): grid3=" << grid3.x
                      << " block3=" << block3.x << std::endl;
            std::cerr << "[run_attn_fwd_kernel] Launching compute_output_kernel..." << std::endl;
        }
        compute_output_kernel<T, Config, 2, kernel3_block_k>
            <<<grid3, block3, 0, stream>>>(workspace, V, O,
                                           cu_seqlens_kv, cu_seqlens_kv_padded,
                                           bs_rt, hn_rt, kv_stride);
        if (debug) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[run_attn_fwd_kernel] FAULT after compute_output_kernel: "
                          << hipGetErrorString(err) << std::endl;
                return;
            }
            std::cerr << "[run_attn_fwd_kernel] compute_output_kernel completed OK" << std::endl;
        }
    }
};

template struct AttnForwardKernelLauncher<hip_bfloat16, ConfigHeadDim64>;
template struct AttnForwardKernelLauncher<hip_bfloat16, ConfigHeadDim128>;

}  // namespace varlen
}  // namespace ck_fused_attn

#include "ck_fused_attn/varlen_attn.hpp"

namespace ck_fused_attn {

size_t varlen_attn_fwd_workspace_size(size_t b, size_t h_q, size_t head_dim) {
    (void)head_dim;
    if (head_dim != 64 && head_dim != 128) {
        throw std::runtime_error("varlen_attn: unsupported head_dim (only 64 or 128)");
    }
    return b * h_q * 1 * 16 * sizeof(hip_bfloat16);
}

void run_varlen_attn_fwd(const void* Q,
                         const void* K,
                         const void* V,
                         const void* dropout_mask,
                         float dropout_p,
                         float sqr_dk_scale,
                         void* O,
                         void* output_attn_weights,
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
    const auto* mask = static_cast<const hip_bfloat16*>(dropout_mask);
    auto* o = static_cast<hip_bfloat16*>(O);
    auto* attn = static_cast<hip_bfloat16*>(output_attn_weights);

    if (std::getenv("NVTE_DEBUG_VARLEN_ATTN")) {
        std::cerr << "[varlen_attn_fwd] ENTRY: b=" << b << " h_q=" << h_q
                  << " head_dim=" << head_dim << " kv_stride=" << kv_stride << std::endl;
    }

    const int bs_rt = static_cast<int>(b);
    const int hn_rt = static_cast<int>(h_q);
    const int kvs   = static_cast<int>(kv_stride);

    if (head_dim == 64) {
        varlen::AttnForwardKernelLauncher<hip_bfloat16, varlen::ConfigHeadDim64>::run_attn_fwd_kernel(
            q, k, v, mask, dropout_p, sqr_dk_scale, o, attn,
            cu_seqlens_kv, cu_seqlens_kv_padded, bs_rt, hn_rt, kvs, stream);
        return;
    }
    if (head_dim == 128) {
        varlen::AttnForwardKernelLauncher<hip_bfloat16, varlen::ConfigHeadDim128>::run_attn_fwd_kernel(
            q, k, v, mask, dropout_p, sqr_dk_scale, o, attn,
            cu_seqlens_kv, cu_seqlens_kv_padded, bs_rt, hn_rt, kvs, stream);
        return;
    }
    throw std::runtime_error("varlen_attn: unsupported head_dim (only 64 or 128)");
}

}  // namespace ck_fused_attn
