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
// T, bi, hi, d_qk and the pointer/scale args must be in scope where these are used.
#define SMALLSEQ_DISPATCH_FWD_CASE(N)                                      \
  case N:                                                                  \
    dispatch_fwd<N, T>(bi, hi, Q_ptr, K_ptr, V_ptr, dropout_mask, dropout, \
                       sqr_dk_scale, O_ptr, attn_workspace, cu_kv, cu_kv_p, \
                       d_qk, stream);                                      \
    break;
#define SMALLSEQ_DISPATCH_BWD_CASE(N)                                        \
  case N:                                                                    \
    dispatch_bwd<N, T>(bi, hi, Q_ptr, K_ptr, V_ptr, dO_ptr, attn_ptr,        \
                       dropout_mask, dropout, sqr_dk_scale, dQ_ptr, dK_ptr, \
                       dV_ptr, workspace_ptr, cu_kv, cu_kv_p, d_qk, stream); \
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
 * Dispatch supports head_dim 128, 256, 512 (d_qk == d_v). Varlen_attn tests */

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
                        const int* cu_kv_p, size_t d_qk, hipStream_t stream) {
  switch (d_qk) {
    case 128:
      run_attn_fwd_impl<T, SmallSeqConfig<MAX_KV, 128>>(
          b, h_q, Q, K, V, dropout_mask, dropout, scale, O, workspace, cu_kv, cu_kv_p, stream);
      break;
    case 256:
      run_attn_fwd_impl<T, SmallSeqConfig<MAX_KV, 256>>(
          b, h_q, Q, K, V, dropout_mask, dropout, scale, O, workspace, cu_kv, cu_kv_p, stream);
      break;
    case 512:
      run_attn_fwd_impl<T, SmallSeqConfig<MAX_KV, 512>>(
          b, h_q, Q, K, V, dropout_mask, dropout, scale, O, workspace, cu_kv, cu_kv_p, stream);
      break;
    default:
      NVTE_ERROR("Unsupported head dimension (d_qk) for small-seq attention: must be 128, 256, or "
                 "512.");
  }
}

template <int MAX_KV, typename T>
static void dispatch_bwd(int b, int h_q, const T* Q, const T* K, const T* V, const T* grad_O,
                        const T* attn_weights, const T* dropout_mask, float dropout, float scale,
                        T* grad_Q, T* grad_K, T* grad_V, T* workspace, const int* cu_kv,
                        const int* cu_kv_p, size_t d_qk, hipStream_t stream) {
  switch (d_qk) {
    case 128:
      run_attn_bwd_impl<T, SmallSeqConfig<MAX_KV, 128>>(
          b, h_q, Q, K, V, grad_O, attn_weights, dropout_mask, dropout, scale,
          grad_Q, grad_K, grad_V, workspace, cu_kv, cu_kv_p, stream);
      break;
    case 256:
      run_attn_bwd_impl<T, SmallSeqConfig<MAX_KV, 256>>(
          b, h_q, Q, K, V, grad_O, attn_weights, dropout_mask, dropout, scale,
          grad_Q, grad_K, grad_V, workspace, cu_kv, cu_kv_p, stream);
      break;
    case 512:
      run_attn_bwd_impl<T, SmallSeqConfig<MAX_KV, 512>>(
          b, h_q, Q, K, V, grad_O, attn_weights, dropout_mask, dropout, scale,
          grad_Q, grad_K, grad_V, workspace, cu_kv, cu_kv_p, stream);
      break;
    default:
      NVTE_ERROR("Unsupported head dimension (d_qk) for small-seq attention: must be 128, 256, or "
                 "512.");
  }
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

  NVTE_CHECK(d_qk == d_v,
             "Small-seq attention requires matching Q/K and V head sizes (d_qk == d_v).");
  NVTE_CHECK(d_qk == 128 || d_qk == 256 || d_qk == 512,
             "Small-seq attention supports head dimension (d_qk) 128, 256, or 512 only.");

  float sqr_dk_scale = attn_scale;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(qkv_dtype, T,
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
  );

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
  const char* nvte_smallseq = std::getenv("NVTE_LOG_CK_CONFIG");
  if (nvte_smallseq && std::string(nvte_smallseq) == "1") {
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

  NVTE_CHECK(d_qk == d_v,
             "Small-seq attention requires matching Q/K and V head sizes (d_qk == d_v).");
  NVTE_CHECK(d_qk == 128 || d_qk == 256 || d_qk == 512,
             "Small-seq attention supports head dimension (d_qk) 128, 256, or 512 only.");

  float sqr_dk_scale = attn_scale;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(qkv_dtype, T,
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
  );
}

// ===========================================================================
// Self-attention MFMA kernels (BSHD layout, s_q == s_kv <= 17)
//
// Ported from crossattn_hip_kernel MFMA 4x4 and 16x16 kernels.
// Adapted for:
//   - BSHD layout (no cu_seqlens indirection)
//   - Runtime batch_size and head_num (not compile-time)
// ===========================================================================

using bhalf_t = __bf16;
using bf16x4  = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8  = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#endif

// Config for self-attention MFMA kernels.
// Only max_seq and head_dim are compile-time (affect LDS sizing and loop bounds).
template <int MAX_SEQ, int HEAD_DIM,
          bool ENABLE_DROPOUT_MASK = false>
struct SelfAttnMfmaConfig {
  static constexpr int max_seq_q   = MAX_SEQ;
  static constexpr int max_seq_kv  = MAX_SEQ;
  static constexpr int head_dim    = HEAD_DIM;
  static constexpr bool enable_dropout_mask = ENABLE_DROPOUT_MASK;
  // Used by softmax kernels from the cross-attn path if needed
  static constexpr int step2_block_size = 256;
  static constexpr CausalMaskType mask_type = CausalMaskType::DISABLE;
};

template <typename T>
__device__ __forceinline__ bf16x8 self_load_cvt_bf16x8(const T* src)
{
    if constexpr(sizeof(T) == 2)
    {
        return *(const bf16x8*)src;
    }
    else
    {
        bf16x8 r;
        #pragma unroll
        for(int i = 0; i < 8; i++)
            r[i] = static_cast<bhalf_t>(src[i]);
        return r;
    }
}

// Helper: BSHD offset for Q/K/V/O: (batch * seq + seq_pos) * head_num * head_dim + head * head_dim
__device__ __forceinline__ size_t bshd_offset(int batch_idx, int seq_pos,
                                              int head_idx, int seq_len,
                                              int head_num, int head_dim)
{
    return ((size_t)batch_idx * seq_len + seq_pos) * head_num * head_dim +
           (size_t)head_idx * head_dim;
}

// ---------------------------------------------------------------------------
// MFMA 16x16x16 self-attention forward kernel (BSHD layout)
// Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void self_attn_fwd_mfma_16x16_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    T* workspace,
    float scale,
    int batch_size,
    int head_num,
    int seq_len)
{
    constexpr int head_dim      = Config::head_dim;
    constexpr int max_seq       = Config::max_seq_q;
    constexpr int hd_pad        = head_dim + 4;
    constexpr int q_tiles       = CEIL_DIV(max_seq, 16);
    constexpr int kv_tiles      = CEIL_DIV(max_seq, 16);
    constexpr int lds_q_rows    = q_tiles * 16;
    constexpr int lds_kv_rows   = kv_tiles * 16;
    constexpr int lds_sm_stride = lds_kv_rows + 4;

    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 64;
    const int lane_id   = tid % 64;
    const int lane_row  = lane_id / 16;
    const int lane_col  = lane_id % 16;

    if(batch_idx >= batch_size || head_idx >= head_num)
        return;

    const int actual_q = seq_len;
    const int seq_kv   = seq_len;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[lds_q_rows * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[lds_kv_rows * hd_pad];
    __shared__ float SM_lds[lds_q_rows * lds_sm_stride];

    // Load Q → Q_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* q_src = Q + bshd_offset(batch_idx, r, head_idx, seq_len, head_num, head_dim);
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(q_src + col);
            }
            else
            {
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    // Load K → KV_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            if(r < seq_kv)
            {
                const T* k_src = K + bshd_offset(batch_idx, r, head_idx, seq_len, head_num, head_dim);
                *(bf16x8*)(&KV_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(k_src + col);
            }
            else
            {
                *(bf16x8*)(&KV_lds[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    __syncthreads();

    // QK^T via MFMA
    float attn_weight[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            floatx4 acc = {0, 0, 0, 0};
            constexpr int total_hd_tiles = CEIL_DIV(head_dim, 16);

            #pragma unroll
            for(int k = 0; k < total_hd_tiles; ++k)
            {
                const int dim_off = k * 16;
                bf16x4 a = *(const bf16x4*)(&Q_lds[(qt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                bf16x4 b = *(const bf16x4*)(&KV_lds[(kvt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
                attn_weight[reg_base + i] = acc[i] * scale;
        }
    }

    // Softmax
    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int q_row = qt * 16 + lane_row * 4 + i;

            float row_max = -INFINITY;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                float val = masked ? -INFINITY : attn_weight[reg_idx];
                float tile_max = val;
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 16));
                row_max = fmaxf(row_max, tile_max);
            }

            float row_sum = 0.0f;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                float exp_val = masked ? 0.0f : expf(attn_weight[reg_idx] - row_max);
                attn_weight[reg_idx] = exp_val;
                float tile_sum = exp_val;
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    tile_sum += __shfl_xor(tile_sum, off, 16);
                row_sum += tile_sum;
            }

            float inv_sum = __builtin_amdgcn_rcpf(row_sum);
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                attn_weight[reg_idx] *= inv_sum;
            }
        }
    }

    // Write attention weights to workspace (for backward pass)
    if(warp_id == 0)
    {
        #pragma unroll
        for(int qt = 0; qt < q_tiles; qt++)
        {
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                #pragma unroll
                for(int i = 0; i < 4; i++)
                {
                    int q_row = qt * 16 + lane_row * 4 + i;
                    int kv_pos = kvt * 16 + lane_col;
                    if(q_row < actual_q && kv_pos < max_seq)
                    {
                        size_t ws_offset = bshd_offset(batch_idx, q_row, head_idx, seq_len, head_num, max_seq);
                        int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                        float w = (kv_pos < seq_kv) ? attn_weight[reg_idx] : 0.0f;
                        workspace[ws_offset + kv_pos] = static_cast<T>(w);
                    }
                }
            }
        }
    }

    // Write weights to SM_lds for Attn x V
    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                SM_lds[q_row * lds_sm_stride + kv_pos] = attn_weight[reg_idx];
            }
        }
    }

    __syncthreads();

    // Load V → KV_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int v_row = tid / threads_per_row;
        const int v_col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seq_kv - 1, 0);

        for(int r = v_row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            const int clamped_r = min(r, clamped_max);
            const T* v_src = V + bshd_offset(batch_idx, clamped_r, head_idx, seq_len, head_num, head_dim);
            *(bf16x8*)(&KV_lds[r * hd_pad + v_col]) = self_load_cvt_bf16x8(v_src + v_col);
        }
    }

    __syncthreads();

    // Attn x V via MFMA
    {
        constexpr int BK = 64;
        constexpr int total_d_tiles = CEIL_DIV(head_dim, BK);

        #pragma unroll
        for(int qt = 0; qt < q_tiles; qt++)
        {
            #pragma unroll
            for(int d = 0; d < total_d_tiles; d++)
            {
                const int dim_idx = d * BK + warp_id * 16;
                floatx4 acc = {0, 0, 0, 0};

                #pragma unroll
                for(int kvt = 0; kvt < kv_tiles; kvt++)
                {
                    bf16x4 a;
                    #pragma unroll
                    for(int k = 0; k < 4; k++)
                    {
                        int q_idx = qt * 16 + lane_col;
                        int kv_pos = kvt * 16 + lane_row * 4 + k;
                        a[k] = static_cast<bhalf_t>(SM_lds[q_idx * lds_sm_stride + kv_pos]);
                    }

                    bf16x4 b;
                    const int kv_base = kvt * 16;
                    #pragma unroll
                    for(int k = 0; k < 4; k++)
                    {
                        b[k] = KV_lds[(kv_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];
                    }

                    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
                }

                #pragma unroll
                for(int i = 0; i < 4; i++)
                {
                    int q_row = qt * 16 + lane_row * 4 + i;
                    if(q_row < actual_q)
                    {
                        O[bshd_offset(batch_idx, q_row, head_idx, seq_len, head_num, head_dim) + dim_idx + lane_col] =
                            static_cast<T>(acc[i]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MFMA 4x4x4 self-attention forward kernel (BSHD layout, seq <= 4)
// Grid: (1, ceil(head_num/16), bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, (Config::head_dim == 128) ? 3 : 1)
__global__ void self_attn_fwd_mfma_4x4_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    T* workspace,
    float scale,
    int batch_size,
    int head_num,
    int seq_len)
{
    constexpr int head_dim   = Config::head_dim;
    constexpr int max_seq    = Config::max_seq_q;
    constexpr int hd_pad     = head_dim + 4;
    constexpr int dims_per_warp  = head_dim / 4;
    constexpr int num_dim_groups = dims_per_warp / 4;

    static_assert(max_seq >= 1 && max_seq <= 4, "4x4x4 kernel supports max_seq 1..4");

    const int batch_idx  = blockIdx.z;
    const int head_group = blockIdx.y;
    const int tid        = threadIdx.x;
    const int warp_id    = tid / 64;
    const int lane_id    = tid % 64;
    const int mfma_block = lane_id / 4;
    const int mfma_tid   = lane_id % 4;

    const int head_base   = head_group * 16;
    const int head_idx    = head_base + mfma_block;
    const bool valid_head = (head_idx < head_num);

    if(batch_idx >= batch_size)
        return;

    const int actual_q = seq_len;
    const int seq_kv   = seq_len;
    const int warp_dim_start = warp_id * dims_per_warp;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_seq * 16 * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[4 * 16 * hd_pad];

    const int load_idx     = tid * 8;
    const int load_head    = load_idx / head_dim;
    const int load_dim     = load_idx % head_dim;
    const int load_lds_off = load_head * hd_pad + load_dim;

    const int q_lds_base = mfma_block * hd_pad;
    const int k_lds_base = mfma_tid * 16 * hd_pad + mfma_block * hd_pad;

    // Load Q → Q_lds
    #pragma unroll
    for(int qr = 0; qr < max_seq; qr++)
    {
        const int q_lds_offset = qr * 16 * hd_pad;
        if(qr < actual_q && head_base + load_head < head_num)
        {
            const T* q_src = Q + bshd_offset(batch_idx, qr, head_base, seq_len, head_num, head_dim);
            *(bf16x8*)(&Q_lds[q_lds_offset + load_lds_off]) = self_load_cvt_bf16x8(q_src + load_idx);
        }
        else
        {
            *(bf16x8*)(&Q_lds[q_lds_offset + load_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    float running_max[max_seq];
    float running_sum[max_seq];
    float v_acc[max_seq][num_dim_groups];

    #pragma unroll
    for(int m = 0; m < max_seq; m++)
    {
        running_max[m] = -INFINITY;
        running_sum[m] = 0.0f;
        #pragma unroll
        for(int dg = 0; dg < num_dim_groups; dg++)
            v_acc[m][dg] = 0.0f;
    }

    const int num_kv_groups = CEIL_DIV(seq_kv, 4);

    for(int kv_grp = 0; kv_grp < num_kv_groups; kv_grp++)
    {
        const int kv_base = kv_grp * 4;

        // Load K → KV_lds
        #pragma unroll
        for(int kv = 0; kv < 4; kv++)
        {
            const int kv_pos        = kv_base + kv;
            const int clamped_kv    = min(kv_pos, max(seq_kv - 1, 0));
            const int kv_lds_offset = kv * 16 * hd_pad;

            if(head_base + load_head < head_num)
            {
                const T* k_src = K + bshd_offset(batch_idx, clamped_kv, head_base, seq_len, head_num, head_dim);
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = self_load_cvt_bf16x8(k_src + load_idx);
            }
            else
            {
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }

        __syncthreads();

        // MFMA QK^T
        floatx4 qk_acc = {0, 0, 0, 0};

        #pragma unroll
        for(int k = 0; k < head_dim; k += 4)
        {
            bf16x4 q_a, k_b;
            if(mfma_tid < actual_q)
                q_a = *(const bf16x4*)(&Q_lds[mfma_tid * 16 * hd_pad + q_lds_base + k]);
            else
                q_a = bf16x4{0, 0, 0, 0};
            k_b = *(const bf16x4*)(&KV_lds[k_lds_base + k]);
            qk_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(q_a, k_b, qk_acc, 0, 0, 0);
        }

        // Online softmax
        float my_weights[4];

        #pragma unroll
        for(int m = 0; m < max_seq; m++)
        {
            float scores[4];
            #pragma unroll
            for(int s = 0; s < 4; s++)
            {
                int kv_pos = kv_base + s;
                bool masked = (kv_pos >= seq_kv) || (m >= actual_q);
                scores[s] = masked ? -INFINITY : (__shfl(qk_acc[m], s, 4) * scale);
            }

            float tile_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
            float new_max  = fmaxf(running_max[m], tile_max);

            if(running_max[m] > -INFINITY)
            {
                float rescale = expf(running_max[m] - new_max);
                running_sum[m] *= rescale;
                #pragma unroll
                for(int dg = 0; dg < num_dim_groups; dg++)
                    v_acc[m][dg] *= rescale;
            }
            running_max[m] = new_max;

            float weights[4];
            #pragma unroll
            for(int s = 0; s < 4; s++)
            {
                weights[s] = (running_max[m] > -INFINITY) ? expf(scores[s] - running_max[m]) : 0.0f;
                running_sum[m] += weights[s];
            }

            if(m == mfma_tid)
            {
                #pragma unroll
                for(int s = 0; s < 4; s++)
                    my_weights[s] = weights[s];
            }
        }

        bf16x4 weight_a;
        if(mfma_tid < actual_q)
        {
            #pragma unroll
            for(int i = 0; i < 4; i++)
                weight_a[i] = static_cast<bhalf_t>(my_weights[i]);
        }
        else
        {
            weight_a = bf16x4{0, 0, 0, 0};
        }

        __syncthreads();

        // Load V → KV_lds
        #pragma unroll
        for(int kv = 0; kv < 4; kv++)
        {
            const int kv_pos        = kv_base + kv;
            const int kv_lds_offset = kv * 16 * hd_pad;

            if(kv_pos < seq_kv && head_base + load_head < head_num)
            {
                const T* v_src = V + bshd_offset(batch_idx, kv_pos, head_base, seq_len, head_num, head_dim);
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = self_load_cvt_bf16x8(v_src + load_idx);
            }
            else
            {
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }

        __syncthreads();

        // MFMA weights x V
        #pragma unroll
        for(int dg = 0; dg < num_dim_groups; dg++)
        {
            const int out_d = warp_dim_start + dg * 4 + mfma_tid;

            bf16x4 v_b;
            #pragma unroll
            for(int i = 0; i < 4; i++)
                v_b[i] = KV_lds[i * 16 * hd_pad + mfma_block * hd_pad + out_d];

            floatx4 mfma_acc;
            #pragma unroll
            for(int m = 0; m < max_seq; m++)
                mfma_acc[m] = v_acc[m][dg];
            #pragma unroll
            for(int m = max_seq; m < 4; m++)
                mfma_acc[m] = 0.0f;

            mfma_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(weight_a, v_b, mfma_acc, 0, 0, 0);

            #pragma unroll
            for(int m = 0; m < max_seq; m++)
                v_acc[m][dg] = mfma_acc[m];
        }

        __syncthreads();
    }

    // Normalize
    #pragma unroll
    for(int m = 0; m < max_seq; m++)
    {
        float inv_sum = (running_sum[m] > 0.0f) ? (1.0f / running_sum[m]) : 0.0f;
        #pragma unroll
        for(int dg = 0; dg < num_dim_groups; dg++)
            v_acc[m][dg] *= inv_sum;
    }

    // Write O
    if(valid_head)
    {
        #pragma unroll
        for(int m = 0; m < max_seq; m++)
        {
            if(m < actual_q)
            {
                #pragma unroll
                for(int dg = 0; dg < num_dim_groups; dg++)
                {
                    const int out_d = warp_dim_start + dg * 4 + mfma_tid;
                    O[bshd_offset(batch_idx, m, head_idx, seq_len, head_num, head_dim) + out_d] =
                        static_cast<T>(v_acc[m][dg]);
                }
            }
        }
    }

    // Write attention weights to workspace (for backward pass) - only 4x4 kernel needs this
    // The 4x4 kernel does online softmax so we need to write the final weights after normalization
    // We write from warp 0 to avoid races
    if(valid_head && warp_id == 0)
    {
        #pragma unroll
        for(int m = 0; m < max_seq; m++)
        {
            if(m < actual_q && m == mfma_tid)
            {
                // Reconstruct normalized weights from running_sum
                // We already wrote output but need weights for backward pass
                // For 4x4 we can't easily extract them from online softmax
                // TODO: The 4x4 kernel doesn't write workspace like the 16x16 kernel does.
                // For now, skip workspace write for 4x4 and only use 16x16 for self-attention.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MFMA 16x16 self-attention backward: grad_V = attn^T @ grad_O
// Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void self_attn_bwd_grad_v_mfma_16x16_kernel(
    const T* attn_weights,
    const T* grad_O,
    T* grad_V,
    float scale,
    int batch_size,
    int head_num,
    int seq_len)
{
    constexpr int head_dim    = Config::head_dim;
    constexpr int max_seq     = Config::max_seq_q;
    constexpr int hd_pad      = head_dim + 4;
    constexpr int q_tiles     = CEIL_DIV(max_seq, 16);
    constexpr int kv_tiles    = CEIL_DIV(max_seq, 16);
    constexpr int lds_q_rows  = q_tiles * 16;
    constexpr int lds_kv_rows = kv_tiles * 16;
    constexpr int attn_pad    = lds_kv_rows + 4;

    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 64;
    const int lane_id   = tid % 64;
    const int lane_row  = lane_id / 16;
    const int lane_col  = lane_id % 16;

    if(batch_idx >= batch_size || head_idx >= head_num)
        return;

    const int actual_q = seq_len;
    const int seq_kv   = seq_len;

    __shared__ __attribute__((aligned(128))) float attn_lds[lds_q_rows * attn_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t dO_lds[lds_q_rows * hd_pad];

    // Load attn_weights → attn_lds
    {
        const int elts_per_thread = (lds_q_rows * attn_pad + 255) / 256;
        for(int e = 0; e < elts_per_thread; e++)
        {
            int idx = tid + e * 256;
            int row = idx / attn_pad;
            int col = idx % attn_pad;
            float val = 0.0f;
            if(row < actual_q && col < max_seq)
            {
                size_t attn_off = bshd_offset(batch_idx, row, head_idx, seq_len, head_num, max_seq) + col;
                val = static_cast<float>(attn_weights[attn_off]);
            }
            if(idx < lds_q_rows * attn_pad)
                attn_lds[idx] = val;
        }
    }

    // Load grad_O → dO_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int do_row = tid / threads_per_row;
        const int do_col = (tid % threads_per_row) * 8;

        for(int r = do_row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* do_src = grad_O + bshd_offset(batch_idx, r, head_idx, seq_len, head_num, head_dim);
                *(bf16x8*)(&dO_lds[r * hd_pad + do_col]) = self_load_cvt_bf16x8(do_src + do_col);
            }
            else
            {
                *(bf16x8*)(&dO_lds[r * hd_pad + do_col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    __syncthreads();

    // MFMA: grad_V = attn^T @ grad_O
    constexpr int BK = 64;

    #pragma unroll
    for(int kv_tile = 0; kv_tile < kv_tiles; kv_tile++)
    {
        constexpr int total_d_tiles = CEIL_DIV(head_dim, BK);

        #pragma unroll
        for(int d = 0; d < total_d_tiles; d++)
        {
            const int dim_idx = d * BK + warp_id * 16;
            floatx4 acc = {0, 0, 0, 0};

            #pragma unroll
            for(int q_tile = 0; q_tile < q_tiles; q_tile++)
            {
                bf16x4 a;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = q_tile * 16 + lane_row * 4 + k;
                    int kv_pos = kv_tile * 16 + lane_col;
                    float val = (q_row < actual_q && kv_pos < seq_kv)
                                    ? attn_lds[q_row * attn_pad + kv_pos] : 0.0f;
                    a[k] = static_cast<bhalf_t>(val);
                }

                bf16x4 b;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = q_tile * 16 + lane_row * 4 + k;
                    b[k] = dO_lds[q_row * hd_pad + dim_idx + lane_col];
                }

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int kv_pos = kv_tile * 16 + lane_row * 4 + i;
                if(kv_pos < seq_kv)
                {
                    size_t gv_idx = bshd_offset(batch_idx, kv_pos, head_idx, seq_len, head_num, head_dim)
                                    + dim_idx + lane_col;
                    grad_V[gv_idx] = static_cast<T>(acc[i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MFMA 16x16 self-attention backward: fused grad_attn/softmax_bwd/grad_Q/grad_K
// Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void self_attn_bwd_fused_mfma_16x16_kernel(
    const T* Q,
    const T* K,
    const T* V,
    const T* grad_O,
    const T* attn_weights,
    T* grad_Q,
    T* grad_K,
    float scale,
    int batch_size,
    int head_num,
    int seq_len)
{
    constexpr int head_dim      = Config::head_dim;
    constexpr int max_seq       = Config::max_seq_q;
    constexpr int hd_pad        = head_dim + 4;
    constexpr int q_tiles       = CEIL_DIV(max_seq, 16);
    constexpr int kv_tiles      = CEIL_DIV(max_seq, 16);
    constexpr int lds_q_rows    = q_tiles * 16;
    constexpr int lds_kv_rows   = kv_tiles * 16;
    constexpr int lds_sm_stride = lds_kv_rows + 4;

    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 64;
    const int lane_id   = tid % 64;
    const int lane_row  = lane_id / 16;
    const int lane_col  = lane_id % 16;

    if(batch_idx >= batch_size || head_idx >= head_num)
        return;

    const int actual_q = seq_len;
    const int seq_kv   = seq_len;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[lds_q_rows * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t dO_lds[lds_q_rows * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[lds_kv_rows * hd_pad];
    __shared__ float SM_lds[lds_q_rows * lds_sm_stride];

    // Load Q → Q_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* q_src = Q + bshd_offset(batch_idx, r, head_idx, seq_len, head_num, head_dim);
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(q_src + col);
            }
            else
            {
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    // Load dO → dO_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* do_src = grad_O + bshd_offset(batch_idx, r, head_idx, seq_len, head_num, head_dim);
                *(bf16x8*)(&dO_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(do_src + col);
            }
            else
            {
                *(bf16x8*)(&dO_lds[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    // Load V → KV_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seq_kv - 1, 0);

        for(int r = row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            const int clamped_r = min(r, clamped_max);
            const T* v_src = V + bshd_offset(batch_idx, clamped_r, head_idx, seq_len, head_num, head_dim);
            *(bf16x8*)(&KV_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(v_src + col);
        }
    }

    __syncthreads();

    // grad_attn = dO @ V^T via MFMA
    float grad_attn[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            floatx4 acc = {0, 0, 0, 0};
            constexpr int total_d_tiles = CEIL_DIV(head_dim, 16);

            #pragma unroll
            for(int dtile = 0; dtile < total_d_tiles; dtile++)
            {
                const int dim_off = dtile * 16;
                bf16x4 a = *(const bf16x4*)(&dO_lds[(qt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                bf16x4 b = *(const bf16x4*)(&KV_lds[(kvt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
                grad_attn[reg_base + i] = acc[i];
        }
    }

    // Load attn_weights into registers
    float attn_reg[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                float val = 0.0f;
                if(q_row < actual_q && kv_pos < seq_kv)
                {
                    size_t attn_off = bshd_offset(batch_idx, q_row, head_idx, seq_len, head_num, max_seq) + kv_pos;
                    val = static_cast<float>(attn_weights[attn_off]);
                }
                attn_reg[reg_base + i] = val;
            }
        }
    }

    // Softmax backward
    float grad_score[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int q_row = qt * 16 + lane_row * 4 + i;

            float dot_sum = 0.0f;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                float partial = grad_attn[reg_idx] * attn_reg[reg_idx];
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    partial += __shfl_xor(partial, off, 16);
                dot_sum += partial;
            }

            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                float gs = attn_reg[reg_idx] * (grad_attn[reg_idx] - dot_sum);
                if(q_row >= actual_q || kv_pos >= seq_kv)
                    gs = 0.0f;
                grad_score[reg_idx] = gs;
            }
        }
    }

    // Write grad_scores → SM_lds
    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                SM_lds[q_row * lds_sm_stride + kv_pos] = grad_score[reg_idx];
            }
        }
    }

    __syncthreads();

    // Load K → KV_lds (replaces V)
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seq_kv - 1, 0);

        for(int r = row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            const int clamped_r = min(r, clamped_max);
            const T* k_src = K + bshd_offset(batch_idx, clamped_r, head_idx, seq_len, head_num, head_dim);
            *(bf16x8*)(&KV_lds[r * hd_pad + col]) = self_load_cvt_bf16x8(k_src + col);
        }
    }

    __syncthreads();

    // grad_Q = grad_scores @ K * scale
    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        constexpr int BK = 64;
        constexpr int total_d_tiles = CEIL_DIV(head_dim, BK);

        #pragma unroll
        for(int d = 0; d < total_d_tiles; d++)
        {
            const int dim_idx = d * BK + warp_id * 16;
            floatx4 acc = {0, 0, 0, 0};

            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                bf16x4 a;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = qt * 16 + lane_col;
                    int kv_pos = kvt * 16 + lane_row * 4 + k;
                    a[k] = static_cast<bhalf_t>(SM_lds[q_row * lds_sm_stride + kv_pos]);
                }

                bf16x4 b;
                const int kv_base = kvt * 16;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                    b[k] = KV_lds[(kv_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                if(q_row < actual_q)
                {
                    size_t gq_idx = bshd_offset(batch_idx, q_row, head_idx, seq_len, head_num, head_dim)
                                    + dim_idx + lane_col;
                    grad_Q[gq_idx] = static_cast<T>(acc[i] * scale);
                }
            }
        }
    }

    // grad_K = grad_scores^T @ Q * scale
    #pragma unroll
    for(int kvt = 0; kvt < kv_tiles; kvt++)
    {
        constexpr int BK = 64;
        constexpr int total_d_tiles = CEIL_DIV(head_dim, BK);

        #pragma unroll
        for(int d = 0; d < total_d_tiles; d++)
        {
            const int dim_idx = d * BK + warp_id * 16;
            floatx4 acc = {0, 0, 0, 0};

            #pragma unroll
            for(int qt = 0; qt < q_tiles; qt++)
            {
                bf16x4 a;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = qt * 16 + lane_row * 4 + k;
                    int kv_pos = kvt * 16 + lane_col;
                    a[k] = static_cast<bhalf_t>(SM_lds[q_row * lds_sm_stride + kv_pos]);
                }

                bf16x4 b;
                const int q_base = qt * 16;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                    b[k] = Q_lds[(q_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int kv_pos = kvt * 16 + lane_row * 4 + i;
                if(kv_pos < seq_kv)
                {
                    size_t gk_idx = bshd_offset(batch_idx, kv_pos, head_idx, seq_len, head_num, head_dim)
                                    + dim_idx + lane_col;
                    grad_K[gk_idx] = static_cast<T>(acc[i] * scale);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Self-attention forward dispatcher
// ---------------------------------------------------------------------------

template <int MAX_SEQ, typename T>
static void dispatch_self_fwd(int b, int h_q, int seq_len,
                              const T* Q, const T* K, const T* V,
                              float scale, T* O, T* workspace,
                              hipStream_t stream) {
  using Cfg = SelfAttnMfmaConfig<MAX_SEQ, 128>;
  // Always use 16x16 for self-attention (4x4 doesn't write workspace for backward)
  dim3 grid(1, h_q, b);
  dim3 block(256);
  self_attn_fwd_mfma_16x16_kernel<T, Cfg><<<grid, block, 0, stream>>>(
      Q, K, V, O, workspace, scale, b, h_q, seq_len);
}

template <int MAX_SEQ, typename T>
static void dispatch_self_bwd(int b, int h_q, int seq_len,
                              const T* Q, const T* K, const T* V,
                              const T* grad_O, const T* attn_weights,
                              float scale, T* grad_Q, T* grad_K, T* grad_V,
                              hipStream_t stream) {
  using Cfg = SelfAttnMfmaConfig<MAX_SEQ, 128>;
  dim3 grid(1, h_q, b);
  dim3 block(256);

  self_attn_bwd_grad_v_mfma_16x16_kernel<T, Cfg><<<grid, block, 0, stream>>>(
      attn_weights, grad_O, grad_V, scale, b, h_q, seq_len);

  self_attn_bwd_fused_mfma_16x16_kernel<T, Cfg><<<grid, block, 0, stream>>>(
      Q, K, V, grad_O, attn_weights, grad_Q, grad_K, scale, b, h_q, seq_len);
}

// Dispatch macros for self-attention
#define SMALLSEQ_SELF_DISPATCH_FWD_CASE(N)                              \
  case N:                                                               \
    dispatch_self_fwd<N, T>(bi, hi, seq_len_i, Q_ptr, K_ptr, V_ptr,    \
                            sqr_dk_scale, O_ptr, attn_workspace, stream); \
    break;

#define SMALLSEQ_SELF_DISPATCH_BWD_CASE(N)                                    \
  case N:                                                                     \
    dispatch_self_bwd<N, T>(bi, hi, seq_len_i, Q_ptr, K_ptr, V_ptr,          \
                            dO_ptr, attn_ptr, sqr_dk_scale,                   \
                            dQ_ptr, dK_ptr, dV_ptr, stream);                  \
    break;

void fused_attn_smallseq_self_fwd(size_t b,
                                  size_t h_q,
                                  size_t h_kv,
                                  size_t max_seqlen,
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
                                  DType qkv_dtype,
                                  void* workspace,
                                  size_t* workspace_size,
                                  cudaStream_t stream)
{
  const char* nvte_log = std::getenv("NVTE_LOG_CK_CONFIG");
  if (nvte_log && std::string(nvte_log) == "1") {
    std::cout << std::endl << "attn_fwd(small-seq self-attn MFMA kernel): ";
    std::cout << "b: " << b << ", h_q: " << h_q << ", max_seqlen: " << max_seqlen;
    std::cout << ", d_qk: " << d_qk << ", d_v: " << d_v;
    std::cout << ", qkv_dtype: "
              << (qkv_dtype == DType::kBFloat16 ? "BF16" : qkv_dtype == DType::kFloat16 ? "FP16" : "?")
              << std::endl;
  }

  float sqr_dk_scale = attn_scale;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(qkv_dtype, T,
    const T* Q_ptr         = static_cast<const T*>(devPtrQ);
    const T* K_ptr         = static_cast<const T*>(devPtrK);
    const T* V_ptr         = static_cast<const T*>(devPtrV);
    T* O_ptr               = static_cast<T*>(devPtrO);
    T* attn_workspace      = static_cast<T*>(attn_weights_buffer);
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);
    int seq_len_i = static_cast<int>(max_seqlen);

    switch (max_seqlen) {
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(2)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(3)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(4)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(5)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(6)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(7)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(8)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(9)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(10)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(11)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(12)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(13)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(14)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(15)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(16)
      SMALLSEQ_SELF_DISPATCH_FWD_CASE(17)
      default:
        NVTE_ERROR("Unsupported max_seqlen for small-seq self-attn: must be 2..17.");
    }
  );
}

void fused_attn_smallseq_self_bwd(size_t b,
                                  size_t h_q,
                                  size_t h_kv,
                                  size_t max_seqlen,
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
                                  DType qkv_dtype,
                                  void* workspace,
                                  size_t* workspace_size,
                                  cudaStream_t stream)
{
  const char* nvte_log = std::getenv("NVTE_LOG_CK_CONFIG");
  if (nvte_log && std::string(nvte_log) == "1") {
    std::cout << std::endl << "attn_bwd(small-seq self-attn MFMA kernel): ";
    std::cout << "b: " << b << ", h_q: " << h_q << ", max_seqlen: " << max_seqlen;
    std::cout << ", d_qk: " << d_qk << ", d_v: " << d_v;
    std::cout << ", qkv_dtype: "
              << (qkv_dtype == DType::kBFloat16 ? "BF16" : qkv_dtype == DType::kFloat16 ? "FP16" : "?")
              << std::endl;
  }

  float sqr_dk_scale = attn_scale;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(qkv_dtype, T,
    const T* Q_ptr      = static_cast<const T*>(devPtrQ);
    const T* K_ptr      = static_cast<const T*>(devPtrK);
    const T* V_ptr      = static_cast<const T*>(devPtrV);
    const T* dO_ptr     = static_cast<const T*>(devPtrdO);
    const T* attn_ptr   = static_cast<const T*>(attn_weights);
    T* dQ_ptr           = static_cast<T*>(devPtrdQ);
    T* dK_ptr           = static_cast<T*>(devPtrdK);
    T* dV_ptr           = static_cast<T*>(devPtrdV);
    int bi = static_cast<int>(b);
    int hi = static_cast<int>(h_q);
    int seq_len_i = static_cast<int>(max_seqlen);

    switch (max_seqlen) {
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(2)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(3)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(4)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(5)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(6)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(7)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(8)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(9)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(10)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(11)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(12)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(13)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(14)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(15)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(16)
      SMALLSEQ_SELF_DISPATCH_BWD_CASE(17)
      default:
        NVTE_ERROR("Unsupported max_seqlen for small-seq self-attn bwd: must be 2..17.");
    }
  );
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
