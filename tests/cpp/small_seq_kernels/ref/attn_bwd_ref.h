// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "attn_common.h"
#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// CPU helper functions used by attn_backward
// ---------------------------------------------------------------------------

// Matrix multiplication C = A @ B
// A: [rows_a, cols_a], B: [cols_a, cols_b], C: [rows_a, cols_b]
template <typename T>
void matmul(const T* A, const T* B, T* C, int rows_a, int cols_a, int cols_b);

// Matrix transpose: A_T = A^T
// A: [rows, cols], A_T: [cols, rows]
template <typename T>
void transpose(const T* A, T* A_T, int rows, int cols);

// Sum along last dimension: sums[i] = sum_j A[i, j]
template <typename T>
void sum_last_dim(const T* A, T* sums, int rows, int cols);

// ---------------------------------------------------------------------------
// CPU backward reference
// ---------------------------------------------------------------------------

/**
 * Multi-Head Attention Backward Pass (CPU Reference Implementation)
 *
 * Q/grad_O/grad_Q layout: [total_padded_seq_q, head_num, head_dim]
 * K/V/grad_K/grad_V layout: [total_padded_seq_kv, head_num, head_dim]
 * attn_weights/dropout_mask: [total_padded_q, head_num, max_kv_seq]
 *
 * Batches where actual Q seq = 0 are skipped.
 */
template <typename T>
void attn_backward(const T* Q,
                   const T* K,
                   const T* V,
                   const T* grad_O,
                   const T* attn_weights,
                   const T* dropout_mask,
                   float dropout_p,
                   T* grad_Q,
                   T* grad_K,
                   T* grad_V,
                   int batch,
                   int head_num,
                   int max_kv_seq,
                   int head_dim,
                   CausalMaskType mask_type,
                   const int* cu_seqlens_q,
                   const int* cu_seqlens_q_padded,
                   const int* cu_seqlens_kv,
                   const int* cu_seqlens_kv_padded,
                   int total_padded_q,
                   int total_padded_kv_seq,
                   int max_seq_q = 1,
                   bool bf16_weights = false);
