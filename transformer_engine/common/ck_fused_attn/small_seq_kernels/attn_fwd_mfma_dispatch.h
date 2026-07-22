/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once

#include "attn_fwd_mfma.h"
#include "attn_fwd_mfma_16x16.h"

namespace small_seq_kernels {

// ---------------------------------------------------------------------------
// Dispatch: seq_q ≤ 4 → 4x4x4 (16 heads/wave), seq_q > 4 → 16x16x16
// ---------------------------------------------------------------------------

template <typename T, typename Config>
struct AttnForwardMfmaDispatchLauncher
{
    static_assert(Config::max_seq_q >= 1,
                  "max_seq_q must be >= 1");

    static size_t calc_workspace_size(int total_padded_q)
    {
        if constexpr(Config::max_seq_q <= 4)
            return AttnForwardMfmaKernelLauncher<T, Config>::calc_workspace_size(total_padded_q);
        else
            return AttnForwardMfma16x16KernelLauncher<T, Config>::calc_workspace_size(total_padded_q);
    }

    /// `aux`: 4x4 path = `T*` attention workspace; 16x16 path = `float*` softmax LSE (see
    /// AttnForwardMfma16x16KernelLauncher::calc_workspace_size).
    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
                                    void* aux,
                                    const int* cu_seqlens_q,
                                    const int* cu_seqlens_q_padded,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    const int* padded_q_to_batch,
                                    int total_padded_q,
                                    int batch,
                                    hipStream_t stream = 0)
    {
        if constexpr(Config::max_seq_q <= 4)
        {
            AttnForwardMfmaKernelLauncher<T, Config>::run_attn_fwd_kernel(
                Q, K, V, dropout_mask, dropout_p, sqr_dk_scale, O, static_cast<T*>(aux),
                cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv, cu_seqlens_kv_padded,
                padded_q_to_batch, total_padded_q, batch, stream);
        }
        else
        {
            AttnForwardMfma16x16KernelLauncher<T, Config>::run_attn_fwd_kernel(
                Q, K, V, dropout_mask, dropout_p, sqr_dk_scale, O, static_cast<float*>(aux),
                cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv, cu_seqlens_kv_padded,
                padded_q_to_batch, total_padded_q, batch, stream);
        }
    }
};

}  // namespace small_seq_kernels
