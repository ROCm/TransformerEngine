// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "attn_common.h"
#include <cmath>
#include <cstring>
#include <vector>

// CausalMaskType and friends live in namespace small_seq_kernels; pull them into scope
// for this standalone CPU reference.
using namespace small_seq_kernels;

/**
 * Multi-Head Attention Forward Pass (CPU Reference Implementation)
 *
 * Q layout: [total_padded_seq_q, head_num, head_dim]  (variable Q lengths, padded storage)
 * K layout: [total_padded_seq_kv, head_num, head_dim]
 * V layout: [total_padded_seq_kv, head_num, head_dim]
 * O layout: [total_padded_seq_q, head_num, head_dim]
 *
 * For each batch b, actual Q seq length is (cu_seqlens_q[b+1] - cu_seqlens_q[b]), which is 0 or 1.
 * Padded storage offset for Q in batch b starts at cu_seqlens_q_padded[b].
 * Batches with actual Q seq = 0 are skipped (their padded slot is unused).
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
                  int max_kv_seq,
                  int head_dim,
                  CausalMaskType mask_type,
                  const int* cu_seqlens_q,
                  const int* cu_seqlens_q_padded,
                  const int* cu_seqlens_kv,
                  const int* cu_seqlens_kv_padded,
                  bool bf16_weights = false);
