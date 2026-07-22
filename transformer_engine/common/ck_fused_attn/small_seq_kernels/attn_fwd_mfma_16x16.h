/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once

#include "attn_common.h"
#include <type_traits>


#ifndef ATTN_MFMA_TYPES_DEFINED
#define ATTN_MFMA_TYPES_DEFINED
using bhalf_t = __bf16;
using bf16x4  = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8  = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#endif

namespace small_seq_kernels {

template <typename T>
__device__ __forceinline__ bf16x8 load_cvt_bf16x8_16(const T* src)
{
    if constexpr(sizeof(T) == 2)
    {
        return *(const bf16x8*)src;
    }
    else
    {
        // T = float
        bf16x8 r;
        #pragma unroll
        for(int i = 0; i < 8; i++)
        {
            r[i] = static_cast<bhalf_t>(src[i]);
        }
        return r;
    }
}

// ---------------------------------------------------------------------------
// MFMA 16x16x16 forward kernel (tiled Q and KV, 1 head/block)
//
// Thread: warp[0-3], lane_row=lane/16 [0,4), lane_col=lane%16 [0,16)
// LDS:    Q_lds[lds_q_rows × hd_pad], KV_lds[lds_kv_rows × hd_pad],
//         SM_lds[lds_q_rows × lds_sm_stride]
// Grid:   (1, head_num, bs), Block: 256
//
// softmax_lse (Option A / FA2-style aux): one float per (padded Q row, head),
//   index ((q_offset + q_row) * head_num + head_idx),
//   value log(sum_j exp(scale * QK^T_{row,j})) = row_max + log(row_sum_exp).
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void fmha_fwd_mfma_16x16_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    float* softmax_lse,
    const T* dropout_mask,
    float dropout_scale,
    float scale,
    const int* cu_seqlens_q,
    const int* cu_seqlens_q_padded,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded)
{
    // Compile-time constants
    constexpr int head_dim      = Config::head_dim;
    constexpr int head_num      = Config::head_num;
    constexpr int max_seq_kv    = Config::max_seq_kv;
    constexpr int max_seq_q     = Config::max_seq_q;
    constexpr int hd_pad        = head_dim + 4;
    constexpr int q_tiles       = CEIL_DIV(max_seq_q, 16);
    constexpr int kv_tiles      = CEIL_DIV(max_seq_kv, 16);
    constexpr int lds_q_rows    = q_tiles * 16;
    constexpr int lds_kv_rows   = kv_tiles * 16;
    constexpr int lds_sm_stride = lds_kv_rows + 4;

    static_assert(max_seq_q >= 1, "max_seq_q must be >= 1");

    // Thread mapping
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 64;
    const int lane_id   = tid % 64;
    const int lane_row  = lane_id / 16;
    const int lane_col  = lane_id % 16;

    const int actual_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
    if(actual_q == 0)
        return;

    const int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
    const int kv_offset = cu_seqlens_kv_padded[batch_idx];
    const int q_offset  = cu_seqlens_q_padded[batch_idx];

    // LDS
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
                const T* q_src = Q + ((size_t)(q_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = load_cvt_bf16x8_16(q_src + col);
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
                const T* k_src = K + ((size_t)(kv_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&KV_lds[r * hd_pad + col]) = load_cvt_bf16x8_16(k_src + col);
            }
            else
            {
                *(bf16x8*)(&KV_lds[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    __syncthreads();

    // QK^T via MFMA (all 4 warps redundant)
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

    // Softmax: two-pass across KV tiles per Q row
    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int q_row = qt * 16 + lane_row * 4 + i;

            // Pass 1: find global row_max across all KV tiles
            float row_max = -INFINITY;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;

                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
                {
                    if(kv_pos > q_row)
                        masked = true;
                }

                float val = masked ? -INFINITY : attn_weight[reg_idx];

                float tile_max = val;
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 16));

                row_max = fmaxf(row_max, tile_max);
            }

            // Pass 2: compute exp and sum across all KV tiles
            float row_sum = 0.0f;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;

                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
                {
                    if(kv_pos > q_row)
                        masked = true;
                }

                float exp_val = masked ? 0.0f : expf(attn_weight[reg_idx] - row_max);
                attn_weight[reg_idx] = exp_val;

                float tile_sum = exp_val;
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    tile_sum += __shfl_xor(tile_sum, off, 16);
                row_sum += tile_sum;
            }

            // Log-sum-exp per row (matches FlashAttention-style LSE; pre-dropout)
            float lse_row = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
            if(lane_col == 0 && q_row < actual_q)
            {
                softmax_lse[((size_t)(q_offset + q_row) * head_num + head_idx)] = lse_row;
            }

            // Normalize and apply dropout
            float inv_sum = __builtin_amdgcn_rcpf(row_sum);
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                attn_weight[reg_idx] *= inv_sum;

                if constexpr(Config::enable_dropout_mask)
                {
                    int kv_pos = kvt * 16 + lane_col;
                    if(q_row < actual_q && kv_pos < seq_kv)
                    {
                        const int ws_offset = ((q_offset + q_row) * head_num + head_idx) * max_seq_kv;
                        attn_weight[reg_idx] *= static_cast<float>(dropout_mask[ws_offset + kv_pos]) * dropout_scale;
                    }
                }
            }
        }
    }

    // Write weights to SM_lds for Attn×V
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

    // Load V → KV_lds (clamped; invalid positions zeroed by softmax weights)
    {
        constexpr int threads_per_row = head_dim / 8;
        const int v_row = tid / threads_per_row;
        const int v_col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seq_kv - 1, 0);

        for(int r = v_row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            const int clamped_r = min(r, clamped_max);
            const T* v_src = V + ((size_t)(kv_offset + clamped_r) * head_num + head_idx) * head_dim;
            *(bf16x8*)(&KV_lds[r * hd_pad + v_col]) = load_cvt_bf16x8_16(v_src + v_col);
        }
    }

    __syncthreads();

    // Attn×V via MFMA (4 warps split head_dim, tiled over Q and KV)
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
                    // A: softmax weights (transposed read from SM_lds)
                    bf16x4 a;
                    #pragma unroll
                    for(int k = 0; k < 4; k++)
                    {
                        int q_idx = qt * 16 + lane_col;
                        int kv_pos = kvt * 16 + lane_row * 4 + k;
                        a[k] = static_cast<bhalf_t>(SM_lds[q_idx * lds_sm_stride + kv_pos]);
                    }

                    // B: V[kv, d]
                    bf16x4 b;
                    const int kv_base = kvt * 16;
                    #pragma unroll
                    for(int k = 0; k < 4; k++)
                    {
                        b[k] = KV_lds[(kv_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];
                    }

                    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
                }

                // Write output
                #pragma unroll
                for(int i = 0; i < 4; i++)
                {
                    int q_row = qt * 16 + lane_row * 4 + i;
                    if(q_row < actual_q)
                    {
                        O[((size_t)(q_offset + q_row) * head_num + head_idx) * head_dim + dim_idx + lane_col] =
                            static_cast<T>(acc[i]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AttnForwardMfma16x16KernelLauncher — Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnForwardMfma16x16KernelLauncher
{
    using fwd_aux_buffer_scalar = float;

    /// Per-(padded Q row, head) softmax log-sum-exp (float), FA2-compatible aux.
    static size_t calc_workspace_size(int total_padded_q)
    {
        return (size_t)total_padded_q * Config::head_num * sizeof(float);
    }

    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
                                    float* softmax_lse,
                                    const int* cu_seqlens_q,
                                    const int* cu_seqlens_q_padded,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    const int* padded_q_to_batch,
                                    int total_padded_q,
                                    int batch,
                                    hipStream_t stream = 0)
    {
        float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;
        float scale         = sqr_dk_scale;

        // Batch is a runtime argument mapped to the grid z-dimension.
        dim3 grid(1, Config::head_num, batch);
        dim3 block(256);

        fmha_fwd_mfma_16x16_kernel<T, Config><<<grid, block, 0, stream>>>(
            Q, K, V, O, softmax_lse,
            dropout_mask, dropout_scale, scale,
            cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded);
    }
};

}  // namespace small_seq_kernels
