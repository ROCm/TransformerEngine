// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "attn_common.h"
#include <type_traits>

using bhalf_t = __bf16;
using bf16x4  = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8  = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#endif

template <typename T>
__device__ __forceinline__ bf16x8 load_cvt_bf16x8(const T* src)
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
// MFMA 4x4x4 forward kernel (seq_q ≤ 4, online softmax, 16 heads/wave)
//
// Thread: warp[0-3], lane[0-63], mfma_block=lane/4 (head), mfma_tid=lane%4 (Q row)
// LDS:    Q_lds[seq_q × 16 × hd_pad], KV_lds[4 × 16 × hd_pad] (reused K→V)
// Grid:   (1, ceil(heads/16), bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
__launch_bounds__(256, (Config::head_dim == 128) ? 3 : 1)
__global__ void fmha_fwd_mfma_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    T* workspace,
    const T* dropout_mask,
    float dropout_scale,
    float scale,
    const int* cu_seqlens_q,
    const int* cu_seqlens_q_padded,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded)
{
    // Compile-time constants
    constexpr int head_dim   = Config::head_dim;
    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int max_seq_q  = Config::max_seq_q;
    constexpr int hd_pad     = head_dim + 4;

    static_assert(max_seq_q >= 1 && max_seq_q <= 4, "4x4x4 kernel supports max_seq_q 1..4");

    // 4 warps split head_dim for Attn×V
    constexpr int dims_per_warp  = head_dim / 4;
    constexpr int num_dim_groups = dims_per_warp / 4;

    // Thread mapping
    const int batch_idx  = blockIdx.z;
    const int head_group = blockIdx.y;
    const int tid        = threadIdx.x;
    const int warp_id    = tid / 64;
    const int lane_id    = tid % 64;
    const int mfma_block = lane_id / 4;   // which head within group [0,16)
    const int mfma_tid   = lane_id % 4;   // Q-row worker within MFMA block [0,4)

    const int head_base   = head_group * 16;
    const int head_idx    = head_base + mfma_block;
    const bool valid_head = (head_idx < head_num);

    const int actual_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
    if(actual_q == 0)
        return;

    const int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
    const int kv_offset = cu_seqlens_kv_padded[batch_idx];
    const int q_offset  = cu_seqlens_q_padded[batch_idx];

    const int warp_dim_start = warp_id * dims_per_warp;

    // LDS
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_seq_q * 16 * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[4 * 16 * hd_pad];

    // Cooperative load: each thread loads 8 bf16 values
    const int load_idx     = tid * 8;
    const int load_head    = load_idx / head_dim;
    const int load_dim     = load_idx % head_dim;
    const int load_lds_off = load_head * hd_pad + load_dim;

    // MFMA LDS read offsets
    const int q_lds_base = mfma_block * hd_pad;
    const int k_lds_base = mfma_tid * 16 * hd_pad + mfma_block * hd_pad;

    // Load Q → Q_lds
    #pragma unroll
    for(int qr = 0; qr < max_seq_q; qr++)
    {
        const int q_lds_offset = qr * 16 * hd_pad;

        if(qr < actual_q && head_base + load_head < head_num)
        {
            const T* q_src = Q + ((size_t)(q_offset + qr) * head_num + head_base) * head_dim;
            *(bf16x8*)(&Q_lds[q_lds_offset + load_lds_off]) = load_cvt_bf16x8(q_src + load_idx);
        }
        else
        {
            *(bf16x8*)(&Q_lds[q_lds_offset + load_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Online attention: fused QK^T → softmax → Attn×V per KV group of 4
    float running_max[max_seq_q];
    float running_sum[max_seq_q];
    float v_acc[max_seq_q][num_dim_groups];

    #pragma unroll
    for(int m = 0; m < max_seq_q; m++)
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

        // Load K[4 positions] → KV_lds
        #pragma unroll
        for(int kv = 0; kv < 4; kv++)
        {
            const int kv_pos        = kv_base + kv;
            const int clamped_kv    = min(kv_pos, max(seq_kv - 1, 0));
            const T* k_src          = K + ((size_t)(kv_offset + clamped_kv) * head_num + head_base) * head_dim;
            const int kv_lds_offset = kv * 16 * hd_pad;

            if(head_base + load_head < head_num)
            {
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = load_cvt_bf16x8(k_src + load_idx);
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
            {
                q_a = *(const bf16x4*)(&Q_lds[mfma_tid * 16 * hd_pad + q_lds_base + k]);
            }
            else
            {
                q_a = bf16x4{0, 0, 0, 0};
            }

            k_b = *(const bf16x4*)(&KV_lds[k_lds_base + k]);

            qk_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(q_a, k_b, qk_acc, 0, 0, 0);
        }

        // Online softmax: extract scores, update running_max/sum, rescale v_acc
        float my_weights[4];

        #pragma unroll
        for(int m = 0; m < max_seq_q; m++)
        {
            float scores[4];
            #pragma unroll
            for(int s = 0; s < 4; s++)
            {
                int kv_pos = kv_base + s;
                bool masked = (kv_pos >= seq_kv) || (m >= actual_q);
                if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
                {
                    if(kv_pos > m)
                        masked = true;
                }
                scores[s] = masked ? -INFINITY : (__shfl(qk_acc[m], s, 4) * scale);
            }

            float tile_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
            float new_max  = fmaxf(running_max[m], tile_max);

            // Rescale previous accumulations (guard -inf - (-inf) = NaN)
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

        // Apply dropout
        if constexpr(Config::enable_dropout_mask)
        {
            if(valid_head && mfma_tid < actual_q)
            {
                const int ws_off = ((q_offset + mfma_tid) * head_num + head_idx) * max_seq_kv;
                #pragma unroll
                for(int s = 0; s < 4; s++)
                {
                    int kv_pos = kv_base + s;
                    if(kv_pos < seq_kv)
                    {
                        my_weights[s] *= static_cast<float>(dropout_mask[ws_off + kv_pos])
                                         * dropout_scale;
                    }
                }
            }
        }

        // Convert weights to bf16 for V MFMA
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

        // Load V[4 positions] → KV_lds
        #pragma unroll
        for(int kv = 0; kv < 4; kv++)
        {
            const int kv_pos        = kv_base + kv;
            const int kv_lds_offset = kv * 16 * hd_pad;

            if(kv_pos < seq_kv && head_base + load_head < head_num)
            {
                const T* v_src = V + ((size_t)(kv_offset + kv_pos) * head_num + head_base) * head_dim;
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = load_cvt_bf16x8(v_src + load_idx);
            }
            else
            {
                *(bf16x8*)(&KV_lds[kv_lds_offset + load_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }

        __syncthreads();

        // MFMA weights × V → accumulate v_acc
        #pragma unroll
        for(int dg = 0; dg < num_dim_groups; dg++)
        {
            const int out_d = warp_dim_start + dg * 4 + mfma_tid;

            bf16x4 v_b;
            #pragma unroll
            for(int i = 0; i < 4; i++)
            {
                v_b[i] = KV_lds[i * 16 * hd_pad + mfma_block * hd_pad + out_d];
            }

            floatx4 mfma_acc;
            #pragma unroll
            for(int m = 0; m < max_seq_q; m++)
                mfma_acc[m] = v_acc[m][dg];
            #pragma unroll
            for(int m = max_seq_q; m < 4; m++)
                mfma_acc[m] = 0.0f;

            mfma_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                weight_a, v_b, mfma_acc, 0, 0, 0);

            #pragma unroll
            for(int m = 0; m < max_seq_q; m++)
                v_acc[m][dg] = mfma_acc[m];
        }

        __syncthreads();
    }

    // Normalize: v_acc /= running_sum
    #pragma unroll
    for(int m = 0; m < max_seq_q; m++)
    {
        float inv_sum = (running_sum[m] > 0.0f) ? (1.0f / running_sum[m]) : 0.0f;
        #pragma unroll
        for(int dg = 0; dg < num_dim_groups; dg++)
            v_acc[m][dg] *= inv_sum;
    }

    // Write output O[total_padded_q, head_num, head_dim]
    if(valid_head)
    {
        #pragma unroll
        for(int m = 0; m < max_seq_q; m++)
        {
            if(m < actual_q)
            {
                #pragma unroll
                for(int dg = 0; dg < num_dim_groups; dg++)
                {
                    const int out_d = warp_dim_start + dg * 4 + mfma_tid;
                    O[((size_t)(q_offset + m) * head_num + head_idx) * head_dim + out_d] =
                        static_cast<T>(v_acc[m][dg]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AttnForwardMfmaKernelLauncher — Grid: (1, ceil(heads/16), bs), Block: 256
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnForwardMfmaKernelLauncher
{
    using fwd_aux_buffer_scalar = T;

    static size_t calc_workspace_size(int total_padded_q)
    {
        return (size_t)total_padded_q * Config::head_num * Config::max_seq_kv * sizeof(T);
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
                                    int total_padded_q)
    {
        float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        dim3 grid(1, CEIL_DIV(Config::head_num, 16), Config::bs);
        dim3 block(256);

        fmha_fwd_mfma_kernel<T, Config><<<grid, block>>>(
            Q, K, V, O, workspace,
            dropout_mask, dropout_scale, sqr_dk_scale,
            cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded);
    }
};
