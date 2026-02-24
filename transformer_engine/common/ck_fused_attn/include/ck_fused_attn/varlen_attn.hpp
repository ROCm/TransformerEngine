/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_VARLEN_ATTN_H
#define CK_FUSED_ATTN_VARLEN_ATTN_H

#include <cstddef>
#include <hip/hip_runtime.h>

namespace ck_fused_attn {

/** Workspace size for varlen forward path (uses output_S as workspace; no extra). */
size_t varlen_attn_fwd_workspace_size(size_t b, size_t h_q, size_t head_dim);

/** Workspace size for varlen backward path. */
size_t varlen_attn_bwd_workspace_size(size_t b, size_t h_q, size_t head_dim);

/**
 * Run varlen unfused attention forward.
 * Constraints: seq_q == 1, max_seq_kv <= 16; Q/K/V in BF16.
 * Q layout: [batch, seq_q, head_num, head_dim]; K,V: [total_padded_seq_kv, head_num, head_dim].
 * output_attn_weights (workspace) used as scores then attn weights; must be size b*h_q*1*16*sizeof(T).
 * kv_stride: number of elements between consecutive KV tokens (h_q*head_dim for THD,
 *            2*h_q*head_dim for T2HD kvpacked).
 */
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
                         hipStream_t stream);

/**
 * Run varlen unfused attention backward.
 * attn_weights is the buffer from forward (output_S).
 */
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
                         hipStream_t stream);

}  // namespace ck_fused_attn

#endif  // CK_FUSED_ATTN_VARLEN_ATTN_H
