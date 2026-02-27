/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file fused_attn_smallseq.cpp
 *  \brief Unfused small-seq (varlen) attention: seq_q=1, max_seqlen_kv<=16, THD only.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "fused_attn_smallseq.h"
#include "utils.h"

// Macros to avoid repeating dispatch switch cases for max_seqlen_kv in [2, 16].
// T, bi, hi and the pointer/scale args must be in scope where these are used.
#define SMALLSEQ_DISPATCH_FWD_CASE(N)                                      \
  case N:                                                                  \
    dispatch_fwd<N, T>(bi, hi, Q_ptr, K_ptr, V_ptr, dropout_mask, dropout, \
                       sqr_dk_scale, O_ptr, attn_workspace, cu_kv, cu_kv_p, \
                       hip_stream);                                        \
    break;
#define SMALLSEQ_DISPATCH_BWD_CASE(N)                                        \
  case N:                                                                    \
    dispatch_bwd<N, T>(bi, hi, Q_ptr, K_ptr, V_ptr, dO_ptr, attn_ptr,        \
                       dropout_mask, dropout, sqr_dk_scale, dQ_ptr, dK_ptr, \
                       dV_ptr, workspace_ptr, cu_kv, cu_kv_p, hip_stream);   \
    break;

namespace transformer_engine {
namespace fused_attn_rocm {

enum class CausalMaskType { DISABLE = 0, TOP_LEFT = 1, BOTTOM_RIGHT = 2 };

template <int MAX_SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = false,
          CausalMaskType MASK_TYPE = CausalMaskType::DISABLE>
struct SmallSeqConfig {
  static constexpr int seq_q                = 1;
  static constexpr int max_seq_kv           = MAX_SEQ_KV;
  static constexpr int head_dim             = HEAD_DIM;
  static constexpr int step2_block_size     = STEP2_BLOCK_SIZE;
  static constexpr bool enable_dropout_mask  = ENABLE_DROPOUT_MASK;
  static constexpr CausalMaskType mask_type = MASK_TYPE;
};

/* MAX_SEQ_KV and HEAD_DIM are compile-time so kernels can use fixed stack arrays
 * (e.g. float results[max_seq_kv], T attn[max_seq_kv]) and constexpr grid/block
 * sizes. This matches varlen_attn/attn_fwd.cpp (FmhaKernelConfig<..., MAX_SEQ_KV, HEAD_DIM>)
 * and INTEGRATION_TASK.md: seq_q==1, max_seq_kv<=16; head_dim=128 is the only
 * value tested in varlen_attn (main() uses TestRunner<2,16>::run<..., 128, ...>). */

// ----- Forward kernels (with runtime batch_size, head_num) -----

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_scores_kernel(const T* Q,
                                      const T* K,
                                      T* scores,
                                      float scale,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch_size,
                                      int head_num)
{
  constexpr int seq_q        = Config::seq_q;
  constexpr int max_seq_kv   = Config::max_seq_kv;
  constexpr int head_dim     = Config::head_dim;
  constexpr int block_k      = 64;
  constexpr int thread_block_size = 64;
  constexpr int tasks_per_block   = TASKS_PER_BLOCK;

  int base_block_offset = blockIdx.x * thread_block_size * tasks_per_block;
  int thread_id         = threadIdx.x;

  for (int task = 0; task < tasks_per_block; task++) {
    int cur_batch_idx = base_block_offset + task * thread_block_size + thread_id;
    int batch_idx     = cur_batch_idx / (seq_q * head_num);
    int seq_head_idx  = cur_batch_idx % (seq_q * head_num);
    int seq_idx       = seq_head_idx / head_num;
    int head_idx      = seq_head_idx % head_num;

    if (batch_idx >= batch_size)
      continue;

    int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
    int kv_offset = cu_seqlens_kv_padded[batch_idx];

    float results[max_seq_kv];
    T fetch_Q[block_k];
    T fetch_K[block_k];
    T* Q_ptr = (T*)&Q[(batch_idx * seq_q * head_num + seq_idx * head_num + head_idx) * head_dim];
    T* K_ptr = (T*)&K[(kv_offset * head_num + head_idx) * head_dim];
    T* score_ptr = (T*)&scores[cur_batch_idx * max_seq_kv];
    uint4 ls_dwordx4_tmp_var;
    for (int i = 0; i < seq_kv; i++)
      results[i] = 0.0f;
    for (int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k) {
      if constexpr (std::is_same<T, hip_bfloat16>::value || std::is_same<T, __half>::value) {
        for (int k = 0; k < block_k / 8; k++) {
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
        for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
          for (int k = 0; k < block_k / 8; k++) {
            ls_dwordx4_tmp_var =
                *((uint4*)&K_ptr[kv_idx * head_num * head_dim + dim_offset + k * 8]);
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
          for (int k = 0; k < block_k; k++)
            results[kv_idx] += static_cast<float>(fetch_Q[k]) * static_cast<float>(fetch_K[k]);
        }
      } else {
        for (int k = 0; k < block_k / 4; k++) {
          ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 4]);
          fetch_Q[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
          fetch_Q[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
          fetch_Q[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
          fetch_Q[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
        }
        for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
          for (int k = 0; k < block_k / 4; k++) {
            ls_dwordx4_tmp_var =
                *((uint4*)&K_ptr[kv_idx * head_num * head_dim + dim_offset + k * 4]);
            fetch_K[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
            fetch_K[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
            fetch_K[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
            fetch_K[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
          }
#pragma unroll
          for (int k = 0; k < block_k; k++)
            results[kv_idx] += fetch_Q[k] * fetch_K[k];
        }
      }
    }
    for (int i = 0; i < seq_kv; i++)
      score_ptr[i] = T(results[i] * scale);
    for (int i = seq_kv; i < max_seq_kv; i++)
      score_ptr[i] = T(-1e9f);
  }
}

template <typename T, typename Config>
__global__ void apply_mask_and_softmax_kernel(T* scores,
                                             const T* dropout_mask,
                                             float dropout_scale,
                                             const int* cu_seqlens_kv,
                                             int batch_size,
                                             int head_num)
{
  const uint32_t block_id  = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  constexpr int seq_q      = Config::seq_q;
  constexpr int max_seq_kv = Config::max_seq_kv;
  constexpr int block_size = Config::step2_block_size;
  constexpr int per_score_size     = seq_q * max_seq_kv;
  constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
  const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
  const uint32_t total_elt = static_cast<uint32_t>(batch_size) * head_num * seq_q * max_seq_kv;
  bool is_tail = block_id * valid_thread_range + block_size >= total_elt;
  int real_row_num =
      is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
              : valid_thread_range / max_seq_kv;

  if (cur_block_offset < total_elt && thread_id < valid_thread_range) {
    __shared__ T tmp_scores[valid_thread_range];
    constexpr int row_num = valid_thread_range / max_seq_kv;
    __shared__ T row_max[row_num];
    __shared__ T row_sum[row_num];

    int global_row_idx = cur_block_offset / max_seq_kv;
    int batch_idx      = global_row_idx / (seq_q * head_num);
    int k_idx          = cur_block_offset % max_seq_kv;

    int seq_kv = (batch_idx < batch_size)
                     ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx])
                     : max_seq_kv;

    T score_value          = scores[cur_block_offset];
    tmp_scores[thread_id]  = score_value;

    if constexpr (Config::mask_type == CausalMaskType::TOP_LEFT) {
      int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
      if (k_idx > q_idx || k_idx >= seq_kv)
        tmp_scores[thread_id] = T(-1e9f);
    } else if constexpr (Config::mask_type == CausalMaskType::BOTTOM_RIGHT) {
      int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
      if (k_idx < q_idx || k_idx >= seq_kv)
        tmp_scores[thread_id] = T(-1e9f);
    } else {
      if (k_idx >= seq_kv)
        tmp_scores[thread_id] = T(-1e9f);
    }
    __syncthreads();

    if (thread_id < real_row_num) {
      T max_val = T(-1e9f);
#pragma unroll
      for (int i = 0; i < max_seq_kv; i++)
        max_val = fmaxf(static_cast<float>(max_val),
                        static_cast<float>(tmp_scores[thread_id * max_seq_kv + i]));
      row_max[thread_id] = max_val;
    }
    __syncthreads();

    T exp_val = T(expf(static_cast<float>(tmp_scores[thread_id] -
                                          row_max[thread_id / max_seq_kv])));
    tmp_scores[thread_id] = exp_val;
    __syncthreads();

    if (thread_id < real_row_num) {
      T sum = T(0.0f);
#pragma unroll
      for (int i = 0; i < max_seq_kv; i++)
        sum += tmp_scores[thread_id * max_seq_kv + i];
      row_sum[thread_id] = sum;
    }
    __syncthreads();

    T attn_weight = tmp_scores[thread_id] / row_sum[thread_id / max_seq_kv];
    if constexpr (Config::enable_dropout_mask) {
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
                                     int batch_size,
                                     int head_num)
{
  constexpr int seq_q                 = Config::seq_q;
  constexpr int max_seq_kv            = Config::max_seq_kv;
  constexpr int head_dim              = Config::head_dim;
  constexpr int block_k               = BLOCK_K;
  constexpr int dwordx4_load_elt      = 16 / sizeof(T);
  constexpr int warp_size             = 64;
  constexpr int process_head_per_warp = warp_size / (head_dim / block_k);
  constexpr int tasks_per_block       = TASKS_PER_BLOCK;

  int base_block_offset  = blockIdx.x * process_head_per_warp * tasks_per_block;
  int thread_id         = threadIdx.x;
  int thread_batch_offset = thread_id / (head_dim / block_k);
  int thread_head_offset  = thread_id % (head_dim / block_k) * block_k;

  uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt],
      store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
  T attn[max_seq_kv];

  for (int task = 0; task < tasks_per_block; task++) {
    int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
    int cur_idx              = block_batch_head_idx + thread_batch_offset;

    int batch_idx    = cur_idx / (seq_q * head_num);
    int seq_head_idx = cur_idx % (seq_q * head_num);
    int seq_q_idx    = seq_head_idx / head_num;
    int head_idx     = seq_head_idx % head_num;

    if (batch_idx >= batch_size)
      continue;

    int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
    int kv_offset = cu_seqlens_kv_padded[batch_idx];

#pragma unroll
    for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
      store_dwordx4_tmp_var[i].x = 0;
      store_dwordx4_tmp_var[i].y = 0;
      store_dwordx4_tmp_var[i].z = 0;
      store_dwordx4_tmp_var[i].w = 0;
    }
#pragma unroll
    for (int i = 0; i < max_seq_kv; i++)
      attn[i] = attn_weights[cur_idx * max_seq_kv + i];
    for (int j = 0; j < seq_kv; j++) {
#pragma unroll
      for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
        load_dwordx4_tmp_var[i] =
            *((uint4*)&V[((kv_offset + j) * head_num + head_idx) * head_dim + thread_head_offset +
                         i * dwordx4_load_elt]);
      }
#pragma unroll
      for (int b = 0; b < block_k; b++)
        ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
            attn[j] *
            ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
    }
#pragma unroll
    for (int i = 0; i < block_k / dwordx4_load_elt; i++)
      *((uint4*)&O[(batch_idx * seq_q * head_num + seq_q_idx * head_num + head_idx) * head_dim +
                   thread_head_offset + i * dwordx4_load_elt]) = store_dwordx4_tmp_var[i];
  }
}

// ----- Forward launcher -----

template <typename T, typename Config>
void run_attn_fwd_impl(int b,
                       int head_num,
                       const T* Q,
                       const T* K,
                       const T* V,
                       const T* dropout_mask,
                       float dropout_p,
                       float sqr_dk_scale,
                       T* O,
                       T* workspace,
                       const int* cu_seqlens_kv,
                       const int* cu_seqlens_kv_padded,
                       hipStream_t stream)
{
  constexpr int seq_q      = Config::seq_q;
  constexpr int max_seq_kv = Config::max_seq_kv;
  constexpr int head_dim   = Config::head_dim;
  constexpr int warp_size  = 64;

  int merge_bs       = b * head_num;
  float scale         = sqr_dk_scale;
  float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

  constexpr int kernel1_threads = 64;
  dim3 block(kernel1_threads);
  dim3 grid((merge_bs + kernel1_threads - 1) / kernel1_threads);
  compute_scores_kernel<T, Config, 1><<<grid, block, 0, stream>>>(
      Q, K, workspace, scale, cu_seqlens_kv, cu_seqlens_kv_padded, b, head_num);

  constexpr int work_thread_num =
      Config::step2_block_size / (seq_q * max_seq_kv) * (seq_q * max_seq_kv);
  dim3 grid2((merge_bs * seq_q * max_seq_kv + work_thread_num - 1) / work_thread_num);
  dim3 block2(Config::step2_block_size);
  apply_mask_and_softmax_kernel<T, Config><<<grid2, block2, 0, stream>>>(
      workspace, dropout_mask, dropout_scale, cu_seqlens_kv, b, head_num);

  constexpr int kernel3_block_k       = 8;
  constexpr int kernel3_threads       = 64;
  constexpr int process_head_per_warp = warp_size / (head_dim / kernel3_block_k);

  dim3 block3(kernel3_threads);
  dim3 grid3((merge_bs / process_head_per_warp + 2 - 1) / 2);
  compute_output_kernel<T, Config, 2, kernel3_block_k><<<grid3, block3, 0, stream>>>(
      workspace, V, O, cu_seqlens_kv, cu_seqlens_kv_padded, b, head_num);
}

// ----- Backward kernels (with runtime batch_size, head_num) -----

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_v_kernel(const T* attn_weights,
                                      const T* grad_O,
                                      T* grad_V,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch_size,
                                      int head_num)
{
  constexpr int seq_q                 = Config::seq_q;
  constexpr int max_seq_kv            = Config::max_seq_kv;
  constexpr int head_dim              = Config::head_dim;
  constexpr int block_k               = BLOCK_K;
  constexpr int dwordx4_load_elt       = 16 / sizeof(T);
  constexpr int warp_size             = 64;
  constexpr int process_head_per_warp = warp_size / (head_dim / block_k);
  constexpr int tasks_per_block       = TASKS_PER_BLOCK;

  int base_block_offset   = blockIdx.x * process_head_per_warp * tasks_per_block;
  int thread_id           = threadIdx.x;
  int thread_batch_offset = thread_id / (head_dim / block_k);
  int thread_head_offset  = thread_id % (head_dim / block_k) * block_k;

  uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt];
  T attn[max_seq_kv];

  for (int task = 0; task < tasks_per_block; task++) {
    int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
    int cur_idx              = block_batch_head_idx + thread_batch_offset;

    int batch_idx    = cur_idx / (seq_q * head_num);
    int seq_head_idx = cur_idx % (seq_q * head_num);
    int seq_q_idx    = seq_head_idx / head_num;
    int head_idx     = seq_head_idx % head_num;

    if (batch_idx >= batch_size)
      continue;

    int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

#pragma unroll
    for (int i = 0; i < max_seq_kv; i++)
      attn[i] = attn_weights[cur_idx * max_seq_kv + i];

    for (int j = 0; j < seq_kv; j++) {
      uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
      for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
        store_dwordx4_tmp_var[i].x = 0;
        store_dwordx4_tmp_var[i].y = 0;
        store_dwordx4_tmp_var[i].z = 0;
        store_dwordx4_tmp_var[i].w = 0;
      }

#pragma unroll
      for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
        load_dwordx4_tmp_var[i] =
            *((uint4*)&grad_O[(batch_idx * seq_q * head_num + seq_q_idx * head_num + head_idx) *
                                  head_dim +
                              thread_head_offset + i * dwordx4_load_elt]);
      }

#pragma unroll
      for (int b = 0; b < block_k; b++) {
        ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
            attn[j] *
            ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
      }

#pragma unroll
      for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
        int grad_v_idx = (cu_seqlens_kv_padded[batch_idx] + j) * head_num * head_dim +
                         head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
        *((uint4*)&grad_V[grad_v_idx]) = store_dwordx4_tmp_var[i];
      }
    }
  }
}

template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_grad_attn_kernel(const T* grad_O,
                                         const T* V,
                                         T* grad_attn,
                                         const int* cu_seqlens_kv,
                                         const int* cu_seqlens_kv_padded,
                                         int batch_size,
                                         int head_num)
{
  constexpr int seq_q        = Config::seq_q;
  constexpr int max_seq_kv   = Config::max_seq_kv;
  constexpr int head_dim     = Config::head_dim;
  constexpr int block_k      = 64;
  constexpr int thread_block_size = 64;
  constexpr int tasks_per_block   = TASKS_PER_BLOCK;

  int base_block_offset = blockIdx.x * thread_block_size * tasks_per_block;
  int thread_id         = threadIdx.x;

  for (int task = 0; task < tasks_per_block; task++) {
    int cur_batch_idx = base_block_offset + task * thread_block_size + thread_id;
    int batch_idx     = cur_batch_idx / (seq_q * head_num);
    int seq_head_idx  = cur_batch_idx % (seq_q * head_num);
    int seq_idx       = seq_head_idx / head_num;
    int head_idx      = seq_head_idx % head_num;

    if (batch_idx >= batch_size)
      continue;

    int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

    float results[max_seq_kv];
    T fetch_grad_O[block_k];
    T fetch_V[block_k];

    T* grad_O_ptr = (T*)&grad_O[(batch_idx * seq_q * head_num + seq_idx * head_num + head_idx) *
                                head_dim];

    const T* V_base =
        &V[cu_seqlens_kv_padded[batch_idx] * head_num * head_dim + head_idx * head_dim];
    int V_stride = head_num * head_dim;

    T* grad_attn_ptr = (T*)&grad_attn[cur_batch_idx * max_seq_kv];

    uint4 ls_dwordx4_tmp_var;

    for (int i = 0; i < seq_kv; i++)
      results[i] = 0.0f;

    for (int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k) {
      if constexpr (std::is_same<T, hip_bfloat16>::value || std::is_same<T, __half>::value) {
        for (int k = 0; k < block_k / 8; k++) {
          ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 8]);
          fetch_grad_O[k * 8 + 0] = ((T*)&ls_dwordx4_tmp_var.x)[0];
          fetch_grad_O[k * 8 + 1] = ((T*)&ls_dwordx4_tmp_var.x)[1];
          fetch_grad_O[k * 8 + 2] = ((T*)&ls_dwordx4_tmp_var.y)[0];
          fetch_grad_O[k * 8 + 3] = ((T*)&ls_dwordx4_tmp_var.y)[1];
          fetch_grad_O[k * 8 + 4] = ((T*)&ls_dwordx4_tmp_var.z)[0];
          fetch_grad_O[k * 8 + 5] = ((T*)&ls_dwordx4_tmp_var.z)[1];
          fetch_grad_O[k * 8 + 6] = ((T*)&ls_dwordx4_tmp_var.w)[0];
          fetch_grad_O[k * 8 + 7] = ((T*)&ls_dwordx4_tmp_var.w)[1];
        }
        for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
          for (int k = 0; k < block_k / 8; k++) {
            ls_dwordx4_tmp_var =
                *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 8]);
            fetch_V[k * 8 + 0] = ((T*)&ls_dwordx4_tmp_var.x)[0];
            fetch_V[k * 8 + 1] = ((T*)&ls_dwordx4_tmp_var.x)[1];
            fetch_V[k * 8 + 2] = ((T*)&ls_dwordx4_tmp_var.y)[0];
            fetch_V[k * 8 + 3] = ((T*)&ls_dwordx4_tmp_var.y)[1];
            fetch_V[k * 8 + 4] = ((T*)&ls_dwordx4_tmp_var.z)[0];
            fetch_V[k * 8 + 5] = ((T*)&ls_dwordx4_tmp_var.z)[1];
            fetch_V[k * 8 + 6] = ((T*)&ls_dwordx4_tmp_var.w)[0];
            fetch_V[k * 8 + 7] = ((T*)&ls_dwordx4_tmp_var.w)[1];
          }
#pragma unroll
          for (int k = 0; k < block_k; k++)
            results[kv_idx] +=
                static_cast<float>(fetch_grad_O[k]) * static_cast<float>(fetch_V[k]);
        }
      } else {
        for (int k = 0; k < block_k / 4; k++) {
          ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 4]);
          fetch_grad_O[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
          fetch_grad_O[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
          fetch_grad_O[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
          fetch_grad_O[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
        }
        for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
          for (int k = 0; k < block_k / 4; k++) {
            ls_dwordx4_tmp_var =
                *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 4]);
            fetch_V[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
            fetch_V[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
            fetch_V[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
            fetch_V[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
          }
#pragma unroll
          for (int k = 0; k < block_k; k++)
            results[kv_idx] += fetch_grad_O[k] * fetch_V[k];
        }
      }
    }
    for (int i = 0; i < seq_kv; i++)
      grad_attn_ptr[i] = T(results[i]);
    for (int i = seq_kv; i < max_seq_kv; i++)
      grad_attn_ptr[i] = T(0.0f);
  }
}

template <typename T, typename Config>
__global__ void softmax_backward_kernel(const T* attn_weights,
                                        const T* dropout_mask,
                                        T* grad_attn,
                                        float dropout_scale,
                                        const int* cu_seqlens_kv,
                                        int batch_size,
                                        int head_num)
{
  const uint32_t block_id  = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  constexpr int seq_q      = Config::seq_q;
  constexpr int max_seq_kv = Config::max_seq_kv;
  constexpr int block_size = Config::step2_block_size;
  constexpr int per_grad_attn_size = seq_q * max_seq_kv;
  constexpr int valid_thread_range = block_size / per_grad_attn_size * per_grad_attn_size;
  const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
  const uint32_t total_elt = static_cast<uint32_t>(batch_size) * head_num * seq_q * max_seq_kv;
  bool is_tail = block_id * valid_thread_range + block_size >= total_elt;
  int real_row_num =
      is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
              : valid_thread_range / max_seq_kv;

  if (cur_block_offset < total_elt && thread_id < valid_thread_range) {
    __shared__ T tmp_grad_score[valid_thread_range];
    constexpr int row_num = valid_thread_range / max_seq_kv;
    __shared__ T reduce_grad_score[row_num];

    int global_row_idx = cur_block_offset / max_seq_kv;
    int batch_idx      = global_row_idx / (seq_q * head_num);
    int k_idx          = cur_block_offset % max_seq_kv;

    int seq_kv = (batch_idx < batch_size)
                     ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx])
                     : max_seq_kv;

    T grad_attn_value = grad_attn[cur_block_offset];
    if constexpr (Config::enable_dropout_mask)
      grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
    T attn_weight             = attn_weights[cur_block_offset];
    T grad_score              = grad_attn_value * attn_weight;
    tmp_grad_score[thread_id] = grad_score;
    __syncthreads();

    if (thread_id < real_row_num) {
      T sum = T(0.0f);
#pragma unroll
      for (int i = 0; i < max_seq_kv; i++)
        sum += tmp_grad_score[thread_id * max_seq_kv + i];
      reduce_grad_score[thread_id] = sum;
    }
    __syncthreads();

    grad_score -= attn_weight * reduce_grad_score[thread_id / max_seq_kv];

    if constexpr (Config::mask_type == CausalMaskType::TOP_LEFT) {
      int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
      if (k_idx > q_idx || k_idx >= seq_kv)
        grad_score = T(0.0f);
    } else if constexpr (Config::mask_type == CausalMaskType::BOTTOM_RIGHT) {
      int q_idx = (cur_block_offset % (seq_q * max_seq_kv)) / max_seq_kv;
      if (k_idx < q_idx || k_idx >= seq_kv)
        grad_score = T(0.0f);
    } else {
      if (k_idx >= seq_kv)
        grad_score = T(0.0f);
    }
    grad_attn[cur_block_offset] = grad_score;
  }
}

template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_qk_kernel(const T* grad_scores,
                                      const T* Q,
                                      const T* K,
                                      T* grad_Q,
                                      T* grad_K,
                                      float scale,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded,
                                      int batch_size,
                                      int head_num)
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

  for (int task = 0; task < tasks_per_block; task++) {
    int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
    int cur_idx              = block_batch_head_idx + thread_batch_offset;

    int batch_idx    = cur_idx / (seq_q * head_num);
    int seq_head_idx = cur_idx % (seq_q * head_num);
    int seq_q_idx    = seq_head_idx / head_num;
    int head_idx     = seq_head_idx % head_num;

    if (batch_idx >= batch_size)
      continue;

    int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

#pragma unroll
    for (int i = 0; i < max_seq_kv; i++)
      grad_score_vals[i] = grad_scores[cur_idx * max_seq_kv + i];

    uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
    for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
      store_dwordx4_tmp_var[i].x = 0;
      store_dwordx4_tmp_var[i].y = 0;
      store_dwordx4_tmp_var[i].z = 0;
      store_dwordx4_tmp_var[i].w = 0;
    }
    for (int j = 0; j < seq_kv; j++) {
#pragma unroll
      for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
        int k_idx = (cu_seqlens_kv_padded[batch_idx] + j) * head_num * head_dim +
                    head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
        load_dwordx4_tmp_var[i] = *((uint4*)&K[k_idx]);
      }
#pragma unroll
      for (int b = 0; b < block_k; b++) {
        ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
            grad_score_vals[j] *
            ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
      }
    }
#pragma unroll
    for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
      T* grad_Q_ptr = &grad_Q[(batch_idx * seq_q * head_num + seq_q_idx * head_num + head_idx) *
                                  head_dim +
                              thread_head_offset + i * dwordx4_load_elt];
      for (int b = 0; b < dwordx4_load_elt; b++)
        grad_Q_ptr[b] = ((T*)&store_dwordx4_tmp_var[i])[b] * T(scale);
    }
#pragma unroll
    for (int i = 0; i < block_k / dwordx4_load_elt; i++) {
      load_dwordx4_tmp_var[i] =
          *((uint4*)&Q[(batch_idx * seq_q * head_num + seq_q_idx * head_num + head_idx) * head_dim +
                       thread_head_offset + i * dwordx4_load_elt]);
    }
    for (int j = 0; j < seq_kv; j++) {
#pragma unroll
      for (int b = 0; b < block_k; b++) {
        T val = grad_score_vals[j] *
                ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] *
                T(scale);
        int grad_k_idx = (cu_seqlens_kv_padded[batch_idx] + j) * head_num * head_dim +
                         head_idx * head_dim + thread_head_offset + b;
        grad_K[grad_k_idx] = val;
      }
    }
  }
}

template <typename T, typename Config>
void run_attn_bwd_impl(int b,
                       int head_num,
                       const T* Q,
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
                       hipStream_t stream)
{
  constexpr int seq_q      = Config::seq_q;
  constexpr int max_seq_kv = Config::max_seq_kv;
  constexpr int head_dim   = Config::head_dim;
  constexpr int warp_size  = 64;

  int merge_bs       = b * head_num;
  float scale         = sqr_dk_scale;
  float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

  dim3 block(warp_size);
  constexpr int tasks_per_block_v = 16;
  dim3 grid_v((b * seq_q * head_num + tasks_per_block_v - 1) / tasks_per_block_v);
  compute_grad_v_kernel<T, Config, tasks_per_block_v><<<grid_v, block, 0, stream>>>(
      attn_weights, grad_O, grad_V, cu_seqlens_kv, cu_seqlens_kv_padded, b, head_num);

  constexpr int tasks_per_block_attn  = 16;
  constexpr int process_head_per_warp = warp_size / (head_dim / 64);
  dim3 grid_grad_attn((b * seq_q * head_num + tasks_per_block_attn * process_head_per_warp - 1) /
                      (tasks_per_block_attn * process_head_per_warp));
  compute_grad_attn_kernel<T, Config, tasks_per_block_attn><<<grid_grad_attn, block, 0, stream>>>(
      grad_O, V, workspace, cu_seqlens_kv, cu_seqlens_kv_padded, b, head_num);

  constexpr int work_thread_num =
      Config::step2_block_size / (seq_q * max_seq_kv) * (seq_q * max_seq_kv);
  dim3 grid_softmax((merge_bs * seq_q * max_seq_kv + work_thread_num - 1) / work_thread_num);
  dim3 block_softmax(Config::step2_block_size);
  softmax_backward_kernel<T, Config><<<grid_softmax, block_softmax, 0, stream>>>(
      attn_weights, dropout_mask, workspace, dropout_scale, cu_seqlens_kv, b, head_num);

  constexpr int tasks_per_block_qk = 4;
  dim3 grid_qk((b * seq_q * head_num + tasks_per_block_qk - 1) / tasks_per_block_qk);
  compute_grad_qk_kernel<T, Config, tasks_per_block_qk><<<grid_qk, block, 0, stream>>>(
      workspace, Q, K, grad_Q, grad_K, scale, cu_seqlens_kv, cu_seqlens_kv_padded, b, head_num);
}

size_t fused_attn_smallseq_bwd_workspace_size(size_t b,
                                              size_t h_q,
                                              size_t max_seqlen_kv,
                                              DType dtype) {
  constexpr size_t elt_size = 2u;  // BF16 and FP16 are 2 bytes
  return b * h_q * 1 * std::min(max_seqlen_kv, size_t(16)) * elt_size;
}

template <int MAX_KV, typename T>
static void dispatch_fwd(int b, int h_q, const T* Q, const T* K, const T* V, const T* dropout_mask,
                        float dropout, float scale, T* O, T* workspace, const int* cu_kv,
                        const int* cu_kv_p, hipStream_t stream) {
  run_attn_fwd_impl<T, SmallSeqConfig<MAX_KV, 128>>(
      b, h_q, Q, K, V, dropout_mask, dropout, scale, O, workspace, cu_kv, cu_kv_p, stream);
}

template <int MAX_KV, typename T>
static void dispatch_bwd(int b, int h_q, const T* Q, const T* K, const T* V, const T* grad_O,
                        const T* attn_weights, const T* dropout_mask, float dropout, float scale,
                        T* grad_Q, T* grad_K, T* grad_V, T* workspace, const int* cu_kv,
                        const int* cu_kv_p, hipStream_t stream) {
  run_attn_bwd_impl<T, SmallSeqConfig<MAX_KV, 128>>(
      b, h_q, Q, K, V, grad_O, attn_weights, dropout_mask, dropout, scale,
      grad_Q, grad_K, grad_V, workspace, cu_kv, cu_kv_p, stream);
}

void fused_attn_smallseq_fwd(size_t b,
                            size_t h_q,
                            size_t h_kv,
                            size_t max_seqlen_kv,
                            size_t d_qk,
                            size_t d_v,
                            bool is_training,
                            float attn_scale,
                            float dropout,
                            const void* devPtrQ,
                            const void* devPtrK,
                            const void* devPtrV,
                            void* devPtrO,
                            void* attn_weights_buffer,
                            const void* devPtrCuSeqlensKV,
                            const void* devPtrSeqOffsetsKV,
                            const void* rng_seed,
                            const void* rng_offset,
                            DType qkv_dtype,
                            void* workspace,
                            size_t* workspace_size,
                            cudaStream_t stream)
{
  const char* nvte_smallseq = std::getenv("NVTE_LOG_CK_CONFIG");
  if (nvte_smallseq && std::string(nvte_smallseq) == "1") {
    std::cout << std::endl << "attn_fwd(small-seq kernel): ";
    std::cout << "b: " << b << ", ";
    std::cout << "h_q: " << h_q << ", ";
    std::cout << "h_kv: " << h_kv << ", ";
    std::cout << "max_seqlen_kv: " << max_seqlen_kv << ", ";
    std::cout << "d_qk: " << d_qk << ", ";
    std::cout << "d_v: " << d_v << ", ";
    std::cout << "is_training: " << is_training << ", ";
    std::cout << "attn_scale: " << attn_scale << ", ";
    std::cout << "dropout: " << dropout << ", ";
    std::cout << "qkv_dtype: "
              << (qkv_dtype == DType::kBFloat16 ? "BF16" : qkv_dtype == DType::kFloat16 ? "FP16" : "?")
              << std::endl;
  }
  (void)h_kv;
  (void)d_qk;
  (void)d_v;
  (void)is_training;
  (void)rng_seed;
  (void)rng_offset;

  float sqr_dk_scale = attn_scale;
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

  if (qkv_dtype == DType::kBFloat16) {
    using T = hip_bfloat16;
    const T* Q_ptr         = static_cast<const T*>(devPtrQ);
    const T* K_ptr         = static_cast<const T*>(devPtrK);
    const T* V_ptr         = static_cast<const T*>(devPtrV);
    T* O_ptr               = static_cast<T*>(devPtrO);
    T* attn_workspace      = static_cast<T*>(attn_weights_buffer);
    const int* cu_kv       = static_cast<const int*>(devPtrCuSeqlensKV);
    const int* cu_kv_p     = static_cast<const int*>(devPtrSeqOffsetsKV);
    const T* dropout_mask  = nullptr;
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);

    switch (max_seqlen_kv) {
      SMALLSEQ_DISPATCH_FWD_CASE(2)
      SMALLSEQ_DISPATCH_FWD_CASE(3)
      SMALLSEQ_DISPATCH_FWD_CASE(4)
      SMALLSEQ_DISPATCH_FWD_CASE(5)
      SMALLSEQ_DISPATCH_FWD_CASE(6)
      SMALLSEQ_DISPATCH_FWD_CASE(7)
      SMALLSEQ_DISPATCH_FWD_CASE(8)
      SMALLSEQ_DISPATCH_FWD_CASE(9)
      SMALLSEQ_DISPATCH_FWD_CASE(10)
      SMALLSEQ_DISPATCH_FWD_CASE(11)
      SMALLSEQ_DISPATCH_FWD_CASE(12)
      SMALLSEQ_DISPATCH_FWD_CASE(13)
      SMALLSEQ_DISPATCH_FWD_CASE(14)
      SMALLSEQ_DISPATCH_FWD_CASE(15)
      SMALLSEQ_DISPATCH_FWD_CASE(16)
      default:
        NVTE_ERROR("Unsupported max_seqlen_kv for small-seq: max_seqlen_kv <= 16.");
    }
  } else if (qkv_dtype == DType::kFloat16) {
    using T = __half;
    const T* Q_ptr         = static_cast<const T*>(devPtrQ);
    const T* K_ptr         = static_cast<const T*>(devPtrK);
    const T* V_ptr         = static_cast<const T*>(devPtrV);
    T* O_ptr               = static_cast<T*>(devPtrO);
    T* attn_workspace      = static_cast<T*>(attn_weights_buffer);
    const int* cu_kv       = static_cast<const int*>(devPtrCuSeqlensKV);
    const int* cu_kv_p     = static_cast<const int*>(devPtrSeqOffsetsKV);
    const T* dropout_mask = nullptr;
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);

    switch (max_seqlen_kv) {
      SMALLSEQ_DISPATCH_FWD_CASE(2)
      SMALLSEQ_DISPATCH_FWD_CASE(3)
      SMALLSEQ_DISPATCH_FWD_CASE(4)
      SMALLSEQ_DISPATCH_FWD_CASE(5)
      SMALLSEQ_DISPATCH_FWD_CASE(6)
      SMALLSEQ_DISPATCH_FWD_CASE(7)
      SMALLSEQ_DISPATCH_FWD_CASE(8)
      SMALLSEQ_DISPATCH_FWD_CASE(9)
      SMALLSEQ_DISPATCH_FWD_CASE(10)
      SMALLSEQ_DISPATCH_FWD_CASE(11)
      SMALLSEQ_DISPATCH_FWD_CASE(12)
      SMALLSEQ_DISPATCH_FWD_CASE(13)
      SMALLSEQ_DISPATCH_FWD_CASE(14)
      SMALLSEQ_DISPATCH_FWD_CASE(15)
      SMALLSEQ_DISPATCH_FWD_CASE(16)
      default:
        NVTE_ERROR("Unsupported max_seqlen_kv for small-seq: max_seqlen_kv <= 16.");
    }
  } else {
    NVTE_ERROR("small-seq path supports only BF16 and FP16.");
  }

  if (workspace_size) {
    size_t bwd_ws = fused_attn_smallseq_bwd_workspace_size(b, h_q, max_seqlen_kv, qkv_dtype);
    *workspace_size = (bwd_ws > 8u) ? bwd_ws : 8u;
  }
}

void fused_attn_smallseq_bwd(size_t b,
                             size_t h_q,
                             size_t h_kv,
                             size_t max_seqlen_kv,
                             size_t d_qk,
                             size_t d_v,
                             float attn_scale,
                             float dropout,
                             const void* devPtrQ,
                             const void* devPtrK,
                             const void* devPtrV,
                             const void* devPtrO,
                             const void* devPtrdO,
                             const void* attn_weights,
                             void* devPtrdQ,
                             void* devPtrdK,
                             void* devPtrdV,
                             const void* devPtrCuSeqlensKV,
                             const void* devPtrSeqOffsetsKV,
                             DType qkv_dtype,
                             void* workspace,
                             size_t* workspace_size,
                             cudaStream_t stream)
{
  if (std::getenv("NVTE_FUSED_ATTN_CK_SMALLSEQ")) {
    std::cout << std::endl << "attn_bwd(ck small-seq kernel): ";
    std::cout << "b: " << b << ", ";
    std::cout << "h_q: " << h_q << ", ";
    std::cout << "h_kv: " << h_kv << ", ";
    std::cout << "max_seqlen_kv: " << max_seqlen_kv << ", ";
    std::cout << "d_qk: " << d_qk << ", ";
    std::cout << "d_v: " << d_v << ", ";
    std::cout << "attn_scale: " << attn_scale << ", ";
    std::cout << "dropout: " << dropout << ", ";
    std::cout << "qkv_dtype: "
              << (qkv_dtype == DType::kBFloat16 ? "BF16" : qkv_dtype == DType::kFloat16 ? "FP16" : "?")
              << std::endl;
  }
  (void)h_kv;
  (void)d_qk;
  (void)d_v;

  float sqr_dk_scale = attn_scale;
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

  if (qkv_dtype == DType::kBFloat16) {
    using T = hip_bfloat16;
    const T* Q_ptr      = static_cast<const T*>(devPtrQ);
    const T* K_ptr      = static_cast<const T*>(devPtrK);
    const T* V_ptr      = static_cast<const T*>(devPtrV);
    const T* O_ptr      = static_cast<const T*>(devPtrO);
    const T* dO_ptr     = static_cast<const T*>(devPtrdO);
    const T* attn_ptr   = static_cast<const T*>(attn_weights);
    T* dQ_ptr           = static_cast<T*>(devPtrdQ);
    T* dK_ptr           = static_cast<T*>(devPtrdK);
    T* dV_ptr           = static_cast<T*>(devPtrdV);
    T* workspace_ptr   = static_cast<T*>(workspace);
    const int* cu_kv    = static_cast<const int*>(devPtrCuSeqlensKV);
    const int* cu_kv_p  = static_cast<const int*>(devPtrSeqOffsetsKV);
    const T* dropout_mask = nullptr;
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);

    switch (max_seqlen_kv) {
      SMALLSEQ_DISPATCH_BWD_CASE(2)
      SMALLSEQ_DISPATCH_BWD_CASE(3)
      SMALLSEQ_DISPATCH_BWD_CASE(4)
      SMALLSEQ_DISPATCH_BWD_CASE(5)
      SMALLSEQ_DISPATCH_BWD_CASE(6)
      SMALLSEQ_DISPATCH_BWD_CASE(7)
      SMALLSEQ_DISPATCH_BWD_CASE(8)
      SMALLSEQ_DISPATCH_BWD_CASE(9)
      SMALLSEQ_DISPATCH_BWD_CASE(10)
      SMALLSEQ_DISPATCH_BWD_CASE(11)
      SMALLSEQ_DISPATCH_BWD_CASE(12)
      SMALLSEQ_DISPATCH_BWD_CASE(13)
      SMALLSEQ_DISPATCH_BWD_CASE(14)
      SMALLSEQ_DISPATCH_BWD_CASE(15)
      SMALLSEQ_DISPATCH_BWD_CASE(16)
      default:
        NVTE_ERROR("Unsupported max_seqlen_kv for small-seq: max_seqlen_kv <= 16.");
    }
  } else if (qkv_dtype == DType::kFloat16) {
    using T = __half;
    const T* Q_ptr      = static_cast<const T*>(devPtrQ);
    const T* K_ptr      = static_cast<const T*>(devPtrK);
    const T* V_ptr      = static_cast<const T*>(devPtrV);
    const T* O_ptr      = static_cast<const T*>(devPtrO);
    const T* dO_ptr     = static_cast<const T*>(devPtrdO);
    const T* attn_ptr   = static_cast<const T*>(attn_weights);
    T* dQ_ptr           = static_cast<T*>(devPtrdQ);
    T* dK_ptr           = static_cast<T*>(devPtrdK);
    T* dV_ptr           = static_cast<T*>(devPtrdV);
    T* workspace_ptr   = static_cast<T*>(workspace);
    const int* cu_kv    = static_cast<const int*>(devPtrCuSeqlensKV);
    const int* cu_kv_p  = static_cast<const int*>(devPtrSeqOffsetsKV);
    const T* dropout_mask = nullptr;
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);

    switch (max_seqlen_kv) {
      SMALLSEQ_DISPATCH_BWD_CASE(2)
      SMALLSEQ_DISPATCH_BWD_CASE(3)
      SMALLSEQ_DISPATCH_BWD_CASE(4)
      SMALLSEQ_DISPATCH_BWD_CASE(5)
      SMALLSEQ_DISPATCH_BWD_CASE(6)
      SMALLSEQ_DISPATCH_BWD_CASE(7)
      SMALLSEQ_DISPATCH_BWD_CASE(8)
      SMALLSEQ_DISPATCH_BWD_CASE(9)
      SMALLSEQ_DISPATCH_BWD_CASE(10)
      SMALLSEQ_DISPATCH_BWD_CASE(11)
      SMALLSEQ_DISPATCH_BWD_CASE(12)
      SMALLSEQ_DISPATCH_BWD_CASE(13)
      SMALLSEQ_DISPATCH_BWD_CASE(14)
      SMALLSEQ_DISPATCH_BWD_CASE(15)
      SMALLSEQ_DISPATCH_BWD_CASE(16)
      default:
        NVTE_ERROR("Unsupported max_seqlen_kv for small-seq: max_seqlen_kv <= 16.");
    }
  } else {
    NVTE_ERROR("small-seq path supports only BF16 and FP16.");
  }

  if (workspace_size)
    *workspace_size = fused_attn_smallseq_bwd_workspace_size(b, h_q, max_seqlen_kv, qkv_dtype);
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
