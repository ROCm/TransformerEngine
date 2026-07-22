// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "attn_common.h"
#include <cmath>
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
__device__ __forceinline__ bf16x8 bwd_load_cvt_bf16x8(const T* src)
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

// ---------------------------------------------------------------------------
// grad_V kernel: grad_V = attn^T @ grad_O
// Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void fmha_bwd_grad_v_mfma_16x16_kernel(
    const T* Q,
    const T* K,
    const float* softmax_lse,
    const T* grad_O,
    T* grad_V,
    float scale,
    const int* cu_seqlens_q,
    const int* cu_seqlens_q_padded,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded)
{
    constexpr int head_dim    = Config::head_dim;
    constexpr int head_num    = Config::head_num;
    constexpr int max_seq_kv  = Config::max_seq_kv;
    constexpr int max_seq_q   = Config::max_seq_q;
    constexpr int hd_pad      = head_dim + 4;
    constexpr int q_tiles     = CEIL_DIV(max_seq_q, 16);
    constexpr int kv_tiles    = CEIL_DIV(max_seq_kv, 16);
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

    const int actual_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
    if(actual_q == 0)
        return;

    const int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
    const int q_offset  = cu_seqlens_q_padded[batch_idx];
    const int kv_offset = cu_seqlens_kv_padded[batch_idx];

    __shared__ __attribute__((aligned(128))) float attn_lds[lds_q_rows * attn_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds_bwd[lds_q_rows * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t K_lds_bwd[lds_kv_rows * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t dO_lds[lds_q_rows * hd_pad];

    // Load Q → Q_lds_bwd
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* q_src = Q + ((size_t)(q_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&Q_lds_bwd[r * hd_pad + col]) = bwd_load_cvt_bf16x8(q_src + col);
            }
            else
                *(bf16x8*)(&Q_lds_bwd[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Load K → K_lds_bwd
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;

        for(int r = row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            if(r < seq_kv)
            {
                const T* k_src = K + ((size_t)(kv_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&K_lds_bwd[r * hd_pad + col]) = bwd_load_cvt_bf16x8(k_src + col);
            }
            else
                *(bf16x8*)(&K_lds_bwd[r * hd_pad + col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    __syncthreads();

    // QK^T (same MFMA tiling as forward) → exp(S - LSE) = P
    float P_reg[q_tiles * kv_tiles * 4];
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
                bf16x4 a = *(const bf16x4*)(&Q_lds_bwd[(qt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                bf16x4 b = *(const bf16x4*)(&K_lds_bwd[(kvt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
                {
                    if(kv_pos > q_row)
                        masked = true;
                }
                float S = acc[i] * scale;
                float lse =
                    softmax_lse[((size_t)(q_offset + q_row) * head_num + head_idx)];
                float pr = masked ? 0.0f : expf(S - lse);
                P_reg[reg_base + i] = pr;
            }
        }
    }

    // Scatter P_reg → attn_lds (same pattern as former workspace write)
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
                    if(q_row < actual_q && kv_pos < max_seq_kv)
                    {
                        int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                        float w = (kv_pos < seq_kv) ? P_reg[reg_idx] : 0.0f;
                        attn_lds[q_row * attn_pad + kv_pos] = w;
                    }
                }
            }
        }
    }

    __syncthreads();

    // Load grad_O → dO_lds
    {
        constexpr int threads_per_row = head_dim / 8;
        const int do_row = tid / threads_per_row;
        const int do_col = (tid % threads_per_row) * 8;

        for(int r = do_row; r < lds_q_rows; r += (256 / threads_per_row))
        {
            if(r < actual_q)
            {
                const T* do_src = grad_O + ((size_t)(q_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&dO_lds[r * hd_pad + do_col]) = bwd_load_cvt_bf16x8(do_src + do_col);
            }
            else
            {
                *(bf16x8*)(&dO_lds[r * hd_pad + do_col]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    __syncthreads();

    // MFMA: grad_V = attn^T @ grad_O (4 warps split head_dim)
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

                // B: dO[q, d]
                bf16x4 b;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = q_tile * 16 + lane_row * 4 + k;
                    b[k] = dO_lds[q_row * hd_pad + dim_idx + lane_col];
                }

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            // Write grad_V
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int kv_pos = kv_tile * 16 + lane_row * 4 + i;
                if(kv_pos < seq_kv)
                {
                    int gv_idx = (kv_offset + kv_pos) * head_num * head_dim +
                                 head_idx * head_dim + dim_idx + lane_col;
                    grad_V[gv_idx] = static_cast<T>(acc[i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused backward kernel: grad_attn → softmax_bwd → grad_Q + grad_K
// Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, 1)
__global__ void fmha_bwd_fused_mfma_16x16_kernel(
    const T* Q,
    const T* K,
    const T* V,
    const T* grad_O,
    const float* softmax_lse,
    T* grad_Q,
    T* grad_K,
    float scale,
    const int* cu_seqlens_q,
    const int* cu_seqlens_q_padded,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded)
{
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
    const int q_offset  = cu_seqlens_q_padded[batch_idx];
    const int kv_offset = cu_seqlens_kv_padded[batch_idx];

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
                const T* q_src = Q + ((size_t)(q_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&Q_lds[r * hd_pad + col]) = bwd_load_cvt_bf16x8(q_src + col);
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
                const T* do_src = grad_O + ((size_t)(q_offset + r) * head_num + head_idx) * head_dim;
                *(bf16x8*)(&dO_lds[r * hd_pad + col]) = bwd_load_cvt_bf16x8(do_src + col);
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
            const T* v_src = V + ((size_t)(kv_offset + clamped_r) * head_num + head_idx) * head_dim;
            *(bf16x8*)(&KV_lds[r * hd_pad + col]) = bwd_load_cvt_bf16x8(v_src + col);
        }
    }

    __syncthreads();

    // grad_attn = dO @ V^T via MFMA (all 4 warps redundant)
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
                // A: dO[q, d]
                bf16x4 a = *(const bf16x4*)(&dO_lds[(qt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                // B: V[kv, d]
                bf16x4 b = *(const bf16x4*)(&KV_lds[(kvt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
                grad_attn[reg_base + i] = acc[i];
        }
    }

    // Reload K into KV_lds (overwrite V) and recompute P_ij = exp(S_ij - LSE_i)
    {
        constexpr int threads_per_row = head_dim / 8;
        const int row = tid / threads_per_row;
        const int col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seq_kv - 1, 0);

        for(int r = row; r < lds_kv_rows; r += (256 / threads_per_row))
        {
            const int clamped_r = min(r, clamped_max);
            const T* k_src = K + ((size_t)(kv_offset + clamped_r) * head_num + head_idx) * head_dim;
            *(bf16x8*)(&KV_lds[r * hd_pad + col]) = bwd_load_cvt_bf16x8(k_src + col);
        }
    }

    __syncthreads();

    float attn_reg[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int kvt = 0; kvt < kv_tiles; kvt++)
        {
            floatx4 acc_s = {0, 0, 0, 0};
            constexpr int total_hd_tiles = CEIL_DIV(head_dim, 16);

            #pragma unroll
            for(int k = 0; k < total_hd_tiles; ++k)
            {
                const int dim_off = k * 16;
                bf16x4 a = *(const bf16x4*)(&Q_lds[(qt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                bf16x4 b = *(const bf16x4*)(&KV_lds[(kvt * 16 + lane_col) * hd_pad + dim_off + lane_row * 4]);
                acc_s = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc_s, 0, 0, 0);
            }

            int reg_base = (qt * kv_tiles + kvt) * 4;
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                bool masked = (kv_pos >= seq_kv) || (q_row >= actual_q);
                if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
                {
                    if(kv_pos > q_row)
                        masked = true;
                }
                float S = acc_s[i] * scale;
                float lse =
                    softmax_lse[((size_t)(q_offset + q_row) * head_num + head_idx)];
                attn_reg[reg_base + i] = masked ? 0.0f : expf(S - lse);
            }
        }
    }

    // Softmax backward: grad_score = attn * (grad_attn - dot_sum)
    float grad_score[q_tiles * kv_tiles * 4];

    #pragma unroll
    for(int qt = 0; qt < q_tiles; qt++)
    {
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int q_row = qt * 16 + lane_row * 4 + i;

            // dot_sum = sum_kv(grad_attn * attn)
            float dot_sum = 0.0f;
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                float partial = grad_attn[reg_idx] * attn_reg[reg_idx];

                // Reduce across lane_col
                #pragma unroll
                for(int off = 8; off > 0; off /= 2)
                    partial += __shfl_xor(partial, off, 16);

                dot_sum += partial;
            }

            // grad_score = attn * (grad_attn - dot_sum)
            #pragma unroll
            for(int kvt = 0; kvt < kv_tiles; kvt++)
            {
                int reg_idx = (qt * kv_tiles + kvt) * 4 + i;
                int kv_pos = kvt * 16 + lane_col;
                float gs = attn_reg[reg_idx] * (grad_attn[reg_idx] - dot_sum);

                // Zero invalid
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

    // K is already in KV_lds from P recomputation

    // grad_Q = grad_scores @ K * scale (4 warps split head_dim)
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
                // A: grad_scores (transposed SM_lds read)
                bf16x4 a;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = qt * 16 + lane_col;
                    int kv_pos = kvt * 16 + lane_row * 4 + k;
                    a[k] = static_cast<bhalf_t>(SM_lds[q_row * lds_sm_stride + kv_pos]);
                }

                // B: K[kv, d]
                bf16x4 b;
                const int kv_base = kvt * 16;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    b[k] = KV_lds[(kv_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];
                }

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            // Write grad_Q
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int q_row = qt * 16 + lane_row * 4 + i;
                if(q_row < actual_q)
                {
                    int gq_idx = ((size_t)(q_offset + q_row) * head_num + head_idx) * head_dim +
                                 dim_idx + lane_col;
                    grad_Q[gq_idx] = static_cast<T>(acc[i] * scale);
                }
            }
        }
    }

    // grad_K = grad_scores^T @ Q * scale (4 warps split head_dim)
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
                // A: grad_scores^T (direct SM_lds read)
                bf16x4 a;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    int q_row = qt * 16 + lane_row * 4 + k;
                    int kv_pos = kvt * 16 + lane_col;
                    a[k] = static_cast<bhalf_t>(SM_lds[q_row * lds_sm_stride + kv_pos]);
                }

                // B: Q[q, d]
                bf16x4 b;
                const int q_base = qt * 16;
                #pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    b[k] = Q_lds[(q_base + lane_row * 4 + k) * hd_pad + dim_idx + lane_col];
                }

                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }

            // Write grad_K
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                int kv_pos = kvt * 16 + lane_row * 4 + i;
                if(kv_pos < seq_kv)
                {
                    int gk_idx = (kv_offset + kv_pos) * head_num * head_dim +
                                 head_idx * head_dim + dim_idx + lane_col;
                    grad_K[gk_idx] = static_cast<T>(acc[i] * scale);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AttnBackwardMfma16x16KernelLauncher — Grid: (1, head_num, bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnBackwardMfma16x16KernelLauncher
{
    using bwd_softmax_aux_scalar = float;

    /// Option A: backward recomputes P from Q, K, and softmax_lse — no P workspace.
    static size_t calc_workspace_size(int total_padded_q)
    {
        (void)total_padded_q;
        return 0;
    }

    static void run_attn_bwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* grad_O,
                                    const float* softmax_lse,
                                    T* grad_Q,
                                    T* grad_K,
                                    T* grad_V,
                                    float sqr_dk_scale,
                                    const int* cu_seqlens_q,
                                    const int* cu_seqlens_q_padded,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    int batch,
                                    hipStream_t stream = 0)
    {
        float scale = sqr_dk_scale;

        // Batch is a runtime argument mapped to the grid z-dimension.
        dim3 grid(1, Config::head_num, batch);
        dim3 block(256);

        // Kernel B: grad_V = P^T @ grad_O (P recomputed from Q, K, LSE)
        fmha_bwd_grad_v_mfma_16x16_kernel<T, Config><<<grid, block, 0, stream>>>(
            Q, K, softmax_lse, grad_O, grad_V, scale,
            cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded);

        // Kernel A: fused grad_attn / softmax_bwd / grad_Q / grad_K
        fmha_bwd_fused_mfma_16x16_kernel<T, Config><<<grid, block, 0, stream>>>(
            Q, K, V, grad_O, softmax_lse,
            grad_Q, grad_K, scale,
            cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded);
    }
};

}  // namespace small_seq_kernels
