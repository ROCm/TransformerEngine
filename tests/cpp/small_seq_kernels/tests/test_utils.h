// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "attn_common.h"

#include <hip/hip_bfloat16.h>

using namespace small_seq_kernels;

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Build cu_seqlens for Q side
//
// max_seq_q == 1: Actual Q len per batch is 0 or 1 (bernoulli).
// max_seq_q > 1:  Actual Q len per batch is uniform in [0, max_seq_q].
//
// Returns total_padded_q (== total_actual_q in this scheme).
// ---------------------------------------------------------------------------

inline int build_cu_seqlens_q(int bs,
                               std::mt19937& gen,
                               std::vector<int>& cu_seqlens_q,
                               std::vector<int>& cu_seqlens_q_padded,
                               std::vector<int>& padded_q_to_batch,
                               int max_seq_q = 1)
{
    cu_seqlens_q.resize(bs + 1);
    cu_seqlens_q_padded.resize(bs + 1);
    cu_seqlens_q[0]        = 0;
    cu_seqlens_q_padded[0] = 0;
    int total_actual_q     = 0;

    // Different distributions based on max_seq_q
    std::bernoulli_distribution q_bernoulli(0.5);
    std::uniform_int_distribution<int> q_uniform(0, max_seq_q);

    for(int b = 0; b < bs; b++)
    {
        int q_len;
        if(max_seq_q == 1)
            q_len = q_bernoulli(gen) ? 1 : 0;
        else
            q_len = q_uniform(gen);

        total_actual_q += q_len;
        cu_seqlens_q[b + 1]        = total_actual_q;
        cu_seqlens_q_padded[b + 1] = cu_seqlens_q_padded[b] + q_len;
    }

    int total_padded_q = cu_seqlens_q_padded[bs];
    padded_q_to_batch.resize(total_padded_q);
    for(int b = 0; b < bs; b++)
    {
        int q_start = cu_seqlens_q_padded[b];
        int q_end   = cu_seqlens_q_padded[b + 1];
        for(int i = q_start; i < q_end; i++)
            padded_q_to_batch[i] = b;
    }

    return total_padded_q;
}

// ---------------------------------------------------------------------------
// Build cu_seqlens for KV side (random lengths with optional padding)
// ---------------------------------------------------------------------------

inline void build_cu_seqlens_kv(int bs,
                                 int max_seq_kv,
                                 std::mt19937& gen,
                                 std::vector<int>& cu_seqlens_kv,
                                 std::vector<int>& cu_seqlens_kv_padded,
                                 int& total_actual_kv_seq,
                                 int& total_padded_kv_seq)
{
    std::normal_distribution<float> normal_dis(4.0f, 2.0f);
    std::uniform_int_distribution<int> pad_dis(0, 5);

    cu_seqlens_kv.resize(bs + 1);
    cu_seqlens_kv_padded.resize(bs + 1);
    cu_seqlens_kv[0]        = 0;
    cu_seqlens_kv_padded[0] = 0;
    total_actual_kv_seq     = 0;
    total_padded_kv_seq     = 0;

    for(int b = 0; b < bs; b++)
    {
        int kv_len     = static_cast<int>(std::round(normal_dis(gen)));
        kv_len         = std::max(2, std::min(max_seq_kv, kv_len));
        int random_pad = pad_dis(gen);
        int padded_len = (kv_len + random_pad > max_seq_kv) ? max_seq_kv : kv_len + random_pad;
        total_actual_kv_seq += kv_len;
        total_padded_kv_seq += padded_len;
        cu_seqlens_kv[b + 1]        = total_actual_kv_seq;
        cu_seqlens_kv_padded[b + 1] = total_padded_kv_seq;
    }
}

// ---------------------------------------------------------------------------
// Reference softmax P + LSE (matches MFMA 16x16 forward numerics; mask-aware).
// Used by Option A backward tests (recompute P from Q, K, LSE on GPU).
// ---------------------------------------------------------------------------

template <typename T>
inline void reference_probs_and_lse_from_qk(
    const std::vector<T>& Q,
    const std::vector<T>& K,
    int bs,
    int head_num,
    int max_seq_kv,
    int head_dim,
    float scale,
    CausalMaskType mask_type,
    const std::vector<int>& cu_seqlens_q,
    const std::vector<int>& cu_seqlens_q_padded,
    const std::vector<int>& cu_seqlens_kv,
    const std::vector<int>& cu_seqlens_kv_padded,
    std::vector<float>& softmax_lse,
    std::vector<T>& attn_probs)
{
    int total_padded_q = cu_seqlens_q_padded.back();
    softmax_lse.resize((size_t)total_padded_q * head_num);
    attn_probs.assign((size_t)total_padded_q * head_num * max_seq_kv, T(0));

    for(int b = 0; b < bs; ++b)
    {
        int q_off    = cu_seqlens_q_padded[b];
        int kv_off   = cu_seqlens_kv_padded[b];
        int actual_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        int seq_kv   = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];

        for(int qi = 0; qi < actual_q; ++qi)
        {
            for(int h = 0; h < head_num; ++h)
            {
                int q_row_g = q_off + qi;
                size_t lse_i = (size_t)q_row_g * head_num + h;

                float row_max = -INFINITY;
                std::vector<float> scores((size_t)seq_kv);
                for(int j = 0; j < seq_kv; ++j)
                {
                    bool masked = false;
                    if(mask_type == CausalMaskType::TOP_LEFT)
                    {
                        if(j > qi)
                            masked = true;
                    }
                    float s = 0.0f;
                    if(!masked)
                    {
                        for(int d = 0; d < head_dim; ++d)
                        {
                            size_t qix = ((size_t)q_row_g * head_num + h) * head_dim + d;
                            size_t kix =
                                ((size_t)(kv_off + j) * head_num + h) * head_dim + d;
                            s += float(Q[qix]) * float(K[kix]);
                        }
                        s *= scale;
                    }
                    else
                        s = -INFINITY;
                    scores[(size_t)j] = s;
                    row_max           = std::max(row_max, s);
                }

                float row_sum = 0.0f;
                for(int j = 0; j < seq_kv; ++j)
                {
                    if(scores[(size_t)j] > -INFINITY / 4)
                        row_sum += std::exp(scores[(size_t)j] - row_max);
                }

                float lse = (row_sum > 0.0f) ? (row_max + std::log(row_sum)) : -INFINITY;
                softmax_lse[lse_i] = lse;

                for(int j = 0; j < seq_kv; ++j)
                {
                    size_t p_i = lse_i * (size_t)max_seq_kv + (size_t)j;
                    if(scores[(size_t)j] > -INFINITY / 4)
                        attn_probs[p_i] = T(std::exp(scores[(size_t)j] - lse));
                    else
                        attn_probs[p_i] = T(0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention probs P_ij = exp(S_ij - LSE_row) with **bf16** dot products for S_ij (matches
// MFMA forward/backward numerics better than float dots) and **given** per-row LSE — use
// LSE copied from the MFMA forward GPU pass so CPU backward matches Option A GPU backward.
// ---------------------------------------------------------------------------

template <typename T>
inline void reference_attn_probs_bf16_dots_with_given_lse(
    const std::vector<T>& Q,
    const std::vector<T>& K,
    int bs,
    int head_num,
    int max_seq_kv,
    int head_dim,
    float scale,
    CausalMaskType mask_type,
    const std::vector<int>& cu_seqlens_q,
    const std::vector<int>& cu_seqlens_q_padded,
    const std::vector<int>& cu_seqlens_kv,
    const std::vector<int>& cu_seqlens_kv_padded,
    const std::vector<float>& softmax_lse,
    std::vector<T>& attn_probs)
{
    int total_padded_q = cu_seqlens_q_padded.back();
    attn_probs.assign((size_t)total_padded_q * head_num * max_seq_kv, T(0));

    for(int b = 0; b < bs; ++b)
    {
        int q_off    = cu_seqlens_q_padded[b];
        int kv_off   = cu_seqlens_kv_padded[b];
        int actual_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        int seq_kv   = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];

        for(int qi = 0; qi < actual_q; ++qi)
        {
            for(int h = 0; h < head_num; ++h)
            {
                int         q_row_g = q_off + qi;
                size_t      lse_i   = (size_t)q_row_g * head_num + h;
                const float lse     = softmax_lse[lse_i];

                for(int j = 0; j < seq_kv; ++j)
                {
                    bool masked = false;
                    if(mask_type == CausalMaskType::TOP_LEFT)
                    {
                        if(j > qi)
                            masked = true;
                    }
                    size_t p_i = lse_i * (size_t)max_seq_kv + (size_t)j;
                    if(masked)
                    {
                        attn_probs[p_i] = T(0);
                        continue;
                    }
                    // Sum per 16-dim tile then add (matches MFMA k-loop float acc; not bf16-rounded P).
                    float s = 0.0f;
                    const int total_hd_tiles = (head_dim + 15) / 16;
                    for(int kt = 0; kt < total_hd_tiles; ++kt)
                    {
                        float partial = 0.0f;
                        const int d0 = kt * 16;
                        const int d1 = std::min(d0 + 16, head_dim);
                        for(int d = d0; d < d1; ++d)
                        {
                            size_t qix = ((size_t)q_row_g * head_num + h) * head_dim + d;
                            size_t kix =
                                ((size_t)(kv_off + j) * head_num + h) * head_dim + d;
                            float qf = static_cast<float>(Q[qix]);
                            float kf = static_cast<float>(K[kix]);
                            partial += float(hip_bfloat16(qf)) * float(hip_bfloat16(kf));
                        }
                        s += partial;
                    }
                    s *= scale;
                    attn_probs[p_i] = T(std::exp(s - lse));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Correctness check helpers
//
// Tolerance formula (numpy/PyTorch allclose, same as CK check_err):
//   PASS when |x_test − x_ref| ≤ atol + rtol × |x_ref|
//
// bf16 MFMA defaults: rtol=1e-2, atol=1e-2 (matches CK FmhaBwdBf16/FmhaFwdBf16)
// ---------------------------------------------------------------------------

// Check two arrays element-wise (all elements).
template <typename T>
void check_array(const std::vector<T>& gpu,
                 const std::vector<T>& cpu,
                 const std::string& name,
                 float rtol      = 1e-2f,
                 float atol      = 1e-2f,
                 bool dump_err   = false)
{
    float max_diff     = 0.0f;
    float max_rel_diff = 0.0f;
    size_t diff_count  = 0;

    for(size_t i = 0; i < gpu.size(); i++)
    {
        float ref_val  = float(cpu[i]);
        float diff     = std::abs(float(gpu[i]) - ref_val);
        float tol      = atol + rtol * std::abs(ref_val);
        float rel_diff = diff / (std::abs(ref_val) + 1e-12f);
        max_diff       = std::max(max_diff, diff);
        max_rel_diff   = std::max(max_rel_diff, rel_diff);
        if(diff > tol)
        {
            if(dump_err)
                std::cout << name << " mismatch at " << i
                          << ": GPU=" << float(gpu[i]) << " CPU=" << ref_val
                          << " abs=" << diff << " tol=" << tol << std::endl;
            diff_count++;
        }
    }

    bool pass = (diff_count == 0);
    std::cout << name << " check:" << std::endl;
    std::cout << "  Max abs diff: " << max_diff << "  Max rel diff: " << max_rel_diff << std::endl;
    std::cout << "  Exceeding tolerance (rtol=" << rtol << ", atol=" << atol
              << "): " << diff_count << " / " << gpu.size() << std::endl;
    std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << std::endl;
}

// Check grad_Q only on active-Q slots (skip empty-Q batches).
template <typename T>
void check_grad_q(const std::vector<T>& gpu,
                  const std::vector<T>& cpu,
                  int bs,
                  int head_num,
                  int head_dim,
                  const std::vector<int>& cu_seqlens_q,
                  const std::vector<int>& cu_seqlens_q_padded,
                  float rtol      = 1e-2f,
                  float atol      = 1e-2f,
                  bool dump_err   = false)
{
    float max_diff     = 0.0f;
    float max_rel_diff = 0.0f;
    size_t diff_count  = 0;
    size_t active_elems = 0;

    for(int b = 0; b < bs; b++)
    {
        int actual_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        if(actual_q == 0)
            continue;
        int q_off = cu_seqlens_q_padded[b];
        for(int q = 0; q < actual_q; q++)
        {
            for(int h = 0; h < head_num; h++)
            {
                int base = ((q_off + q) * head_num + h) * head_dim;
                for(int d = 0; d < head_dim; d++)
                {
                    size_t idx     = base + d;
                    float ref_val  = float(cpu[idx]);
                    float diff     = std::abs(float(gpu[idx]) - ref_val);
                    float tol      = atol + rtol * std::abs(ref_val);
                    float rel_diff = diff / (std::abs(ref_val) + 1e-12f);
                    max_diff       = std::max(max_diff, diff);
                    max_rel_diff   = std::max(max_rel_diff, rel_diff);
                    if(diff > tol)
                    {
                        if(dump_err)
                            std::cout << "grad_Q mismatch at [b=" << b << ",q=" << q
                                      << ",h=" << h << ",d=" << d
                                      << "]: GPU=" << float(gpu[idx]) << " CPU=" << ref_val
                                      << " abs=" << diff << " tol=" << tol << std::endl;
                        diff_count++;
                    }
                    active_elems++;
                }
            }
        }
    }

    bool pass = (diff_count == 0);
    std::cout << "grad_Q check (active slots only):" << std::endl;
    std::cout << "  Active Q elements: " << active_elems << std::endl;
    std::cout << "  Max abs diff: " << max_diff << "  Max rel diff: " << max_rel_diff << std::endl;
    std::cout << "  Exceeding tolerance (rtol=" << rtol << ", atol=" << atol
              << "): " << diff_count << " / " << active_elems << std::endl;
    std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << std::endl;
}

// Check output on active-Q batch positions (supports multi-Q).
//
// Tolerance: |diff| ≤ atol + rtol × |x_ref|
template <typename T>
void check_output(const std::vector<T>& gpu,
                  const std::vector<T>& cpu,
                  int bs,
                  int head_num,
                  int head_dim,
                  const std::vector<int>& cu_seqlens_q,
                  const std::vector<int>& cu_seqlens_q_padded,
                  const std::string& name,
                  float rtol      = 1e-2f,
                  float atol      = 1e-2f,
                  bool dump_err   = false)
{
    float max_diff     = 0.0f;
    float max_rel_diff = 0.0f;
    size_t diff_count  = 0;
    size_t total_elems = 0;

    for(int b = 0; b < bs; b++)
    {
        int actual_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        if(actual_q == 0)
            continue;
        int q_off = cu_seqlens_q_padded[b];
        for(int q = 0; q < actual_q; q++)
        {
            for(int h = 0; h < head_num; h++)
            {
                for(int d = 0; d < head_dim; d++)
                {
                    size_t idx     = ((size_t)(q_off + q) * head_num + h) * head_dim + d;
                    float ref_val  = float(cpu[idx]);
                    float diff     = std::abs(float(gpu[idx]) - ref_val);
                    float tol      = atol + rtol * std::abs(ref_val);
                    float rel_diff = diff / (std::abs(ref_val) + 1e-12f);
                    max_diff       = std::max(max_diff, diff);
                    max_rel_diff   = std::max(max_rel_diff, rel_diff);
                    total_elems++;
                    if(diff > tol)
                    {
                        if(dump_err)
                            std::cout << name << " mismatch at b=" << b << " q=" << q
                                      << " h=" << h << " d=" << d
                                      << ": GPU=" << float(gpu[idx]) << " CPU=" << ref_val
                                      << " abs=" << diff << " tol=" << tol << std::endl;
                        diff_count++;
                    }
                }
            }
        }
    }

    bool pass = (diff_count == 0);
    std::cout << name << " check:" << std::endl;
    std::cout << "  Max abs diff: " << max_diff << "  Max rel diff: " << max_rel_diff << std::endl;
    std::cout << "  Exceeding tolerance (rtol=" << rtol << ", atol=" << atol
              << "): " << diff_count << " / " << total_elems << std::endl;
    std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << std::endl;
}

// ---------------------------------------------------------------------------
// TestRunner: iterate over SEQ_KV values from SEQ_KV to MAX_SEQ_KV
//
// Usage:
//   TestRunner<START_SEQ, MAX_SEQ>::run<DataType, BS, HEAD_NUM, HEAD_DIM,
//                                       STEP2_BLOCK, DROPOUT, MASK, MAX_SEQ_Q>(fn, args...);
//
// The Func callable must have the signature:
//   template <typename DataType, typename Config>
//   void fn(Args...);
// ---------------------------------------------------------------------------

template <int SEQ_KV, int MAX_SEQ_KV>
struct TestRunner
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE,
              int MAX_SEQ_Q = 1,
              typename Func,
              typename... Args>
    static void run(Func fn, Args&&... args)
    {
        using KernelConfig = FmhaKernelConfig<HEAD_NUM,
                                              SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE,
                                              MAX_SEQ_Q>;
        fn.template operator()<DataType, KernelConfig>(BS, std::forward<Args>(args)...);

        TestRunner<SEQ_KV + 1, MAX_SEQ_KV>::template run<DataType,
                                                         BS,
                                                         HEAD_NUM,
                                                         HEAD_DIM,
                                                         STEP2_BLOCK_SIZE,
                                                         ENABLE_DROPOUT_MASK,
                                                         MASK_TYPE,
                                                         MAX_SEQ_Q>(fn, std::forward<Args>(args)...);
    }
};

// Termination specialisation
template <int MAX_SEQ_KV>
struct TestRunner<MAX_SEQ_KV, MAX_SEQ_KV>
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE,
              int MAX_SEQ_Q = 1,
              typename Func,
              typename... Args>
    static void run(Func fn, Args&&... args)
    {
        using KernelConfig = FmhaKernelConfig<HEAD_NUM,
                                              MAX_SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE,
                                              MAX_SEQ_Q>;
        fn.template operator()<DataType, KernelConfig>(BS, std::forward<Args>(args)...);
    }
};
