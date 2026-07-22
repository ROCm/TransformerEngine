// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unified varlen test for MFMA 16x16x16 forward + backward kernels.
//
// 4 test cases (each runs fwd + bwd):
//   1. sq∈[1,16], skv∈[2,16]; varlen + padding
//   2. sq=1 (fixed, no padding), skv∈[2,16] (varlen + padding)
//   3. sq=16, skv=16; fixed, no padding
//   4. sq=17, skv=17; fixed, no padding
//
// Build: cmake --build build --target test_varlen_mfma_16x16

#include "attn_fwd_mfma_16x16.h"
#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_ref.h"
#include "attn_bwd_ref.h"
#include "test_utils.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Q cu_seqlens builders
// ---------------------------------------------------------------------------

// Varlen Q: actual in [1, max_seq_q] with random padding [0, 3]
inline int build_varlen_cu_seqlens_q(int bs,
                                     int max_seq_q,
                                     std::mt19937& gen,
                                     std::vector<int>& cu_seqlens_q,
                                     std::vector<int>& cu_seqlens_q_padded,
                                     std::vector<int>& padded_q_to_batch)
{
    std::uniform_int_distribution<int> q_dist(1, max_seq_q);
    std::uniform_int_distribution<int> pad_dist(0, 3);

    cu_seqlens_q.resize(bs + 1);
    cu_seqlens_q_padded.resize(bs + 1);
    cu_seqlens_q[0] = cu_seqlens_q_padded[0] = 0;

    for(int b = 0; b < bs; b++)
    {
        int q_len      = q_dist(gen);
        int pad        = pad_dist(gen);
        int padded_len = std::min(q_len + pad, max_seq_q);
        cu_seqlens_q[b + 1]        = cu_seqlens_q[b] + q_len;
        cu_seqlens_q_padded[b + 1] = cu_seqlens_q_padded[b] + padded_len;
    }

    int total_padded_q = cu_seqlens_q_padded[bs];
    padded_q_to_batch.resize(total_padded_q);
    for(int b = 0; b < bs; b++)
        for(int q = cu_seqlens_q_padded[b]; q < cu_seqlens_q_padded[b + 1]; q++)
            padded_q_to_batch[q] = b;

    return total_padded_q;
}

// Fixed Q: all batches have exactly fix_sq tokens, no padding
inline int build_fixed_cu_seqlens_q(int bs,
                                    int fix_sq,
                                    std::vector<int>& cu_seqlens_q,
                                    std::vector<int>& cu_seqlens_q_padded,
                                    std::vector<int>& padded_q_to_batch)
{
    cu_seqlens_q.resize(bs + 1);
    cu_seqlens_q_padded.resize(bs + 1);
    cu_seqlens_q[0] = cu_seqlens_q_padded[0] = 0;

    for(int b = 0; b < bs; b++)
    {
        cu_seqlens_q[b + 1]        = cu_seqlens_q[b] + fix_sq;
        cu_seqlens_q_padded[b + 1] = cu_seqlens_q_padded[b] + fix_sq;
    }

    int total_padded_q = cu_seqlens_q_padded[bs];
    padded_q_to_batch.resize(total_padded_q);
    for(int b = 0; b < bs; b++)
        for(int q = cu_seqlens_q_padded[b]; q < cu_seqlens_q_padded[b + 1]; q++)
            padded_q_to_batch[q] = b;

    return total_padded_q;
}

// ---------------------------------------------------------------------------
// KV cu_seqlens builder (fixed)
// ---------------------------------------------------------------------------

inline void build_fixed_cu_seqlens_kv(int bs,
                                      int fix_skv,
                                      std::vector<int>& cu_seqlens_kv,
                                      std::vector<int>& cu_seqlens_kv_padded,
                                      int& total_padded_kv_seq)
{
    cu_seqlens_kv.resize(bs + 1);
    cu_seqlens_kv_padded.resize(bs + 1);
    cu_seqlens_kv[0] = cu_seqlens_kv_padded[0] = 0;

    for(int b = 0; b < bs; b++)
    {
        cu_seqlens_kv[b + 1]        = cu_seqlens_kv[b] + fix_skv;
        cu_seqlens_kv_padded[b + 1] = cu_seqlens_kv_padded[b] + fix_skv;
    }

    total_padded_kv_seq = cu_seqlens_kv_padded[bs];
}

// ---------------------------------------------------------------------------
// Forward test
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
bool test_fwd(int bs,
              bool varlen_q, int fix_sq,
              bool varlen_kv, int fix_skv,
              const std::string& label,
              const std::vector<int>& h_cu_seqlens_q,
              const std::vector<int>& h_cu_seqlens_q_padded,
              const std::vector<int>& h_padded_q_to_batch,
              const std::vector<int>& h_cu_seqlens_kv,
              const std::vector<int>& h_cu_seqlens_kv_padded,
              int total_padded_q,
              int total_padded_kv_seq)
{
    using Launcher = AttnForwardMfma16x16KernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int max_seq_q  = Config::max_seq_q;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    size_t size_Q = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V = size_K;
    size_t size_O = size_Q;

    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<DataType> h_O_gpu(size_O, DataType(0.0f));
    std::vector<DataType> h_O_cpu(size_O, DataType(0.0f));

    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));

    if constexpr(std::is_same<DataType, float>::value)
    {
        for(size_t i = 0; i < size_Q; i++) h_Q[i] = float(hip_bfloat16(h_Q[i]));
        for(size_t i = 0; i < size_K; i++) h_K[i] = float(hip_bfloat16(h_K[i]));
        for(size_t i = 0; i < size_V; i++) h_V[i] = float(hip_bfloat16(h_V[i]));
    }

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    attn_forward(h_Q.data(), h_K.data(), h_V.data(),
                 static_cast<const DataType*>(nullptr), 0.0f,
                 h_O_cpu.data(), static_cast<DataType*>(nullptr),
                 bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                 h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                 h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                 true);

    DataType *d_Q, *d_K, *d_V, *d_O;
    float* d_softmax_lse;
    int *d_cu_sq, *d_cu_sqp, *d_cu_skv, *d_cu_skvp;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
    size_t ws_size = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_softmax_lse, ws_size > 0 ? ws_size : sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cu_sq, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_sqp, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_skv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_skvp, (bs + 1) * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_O, 0, size_O * sizeof(DataType)));
    HIP_CHECK(hipMemcpy(d_cu_sq, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_sqp, h_cu_seqlens_q_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_skv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_skvp, h_cu_seqlens_kv_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));

    Launcher::run_attn_fwd_kernel(d_Q, d_K, d_V,
                                  static_cast<const DataType*>(nullptr),
                                  0.0f, sqr_dk_scale, d_O, d_softmax_lse,
                                  d_cu_sq, d_cu_sqp, d_cu_skv, d_cu_skvp,
                                  nullptr, total_padded_q, bs);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));

    std::cout << "  [FWD] ";
    check_output(h_O_gpu, h_O_cpu, bs, head_num, head_dim,
                 h_cu_seqlens_q, h_cu_seqlens_q_padded, "Output", 1e-2f, 1e-2f, false);

    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O)); HIP_CHECK(hipFree(d_softmax_lse));
    HIP_CHECK(hipFree(d_cu_sq)); HIP_CHECK(hipFree(d_cu_sqp));
    HIP_CHECK(hipFree(d_cu_skv)); HIP_CHECK(hipFree(d_cu_skvp));

    return true;
}

// ---------------------------------------------------------------------------
// Backward test
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
bool test_bwd(int bs,
              bool varlen_q, int fix_sq,
              bool varlen_kv, int fix_skv,
              const std::string& label,
              const std::vector<int>& h_cu_seqlens_q,
              const std::vector<int>& h_cu_seqlens_q_padded,
              const std::vector<int>& h_padded_q_to_batch,
              const std::vector<int>& h_cu_seqlens_kv,
              const std::vector<int>& h_cu_seqlens_kv_padded,
              int total_padded_q,
              int total_padded_kv_seq)
{
    using BwdLauncher = AttnBackwardMfma16x16KernelLauncher<DataType, Config>;
    using FwdLauncher = AttnForwardMfma16x16KernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int max_seq_q  = Config::max_seq_q;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    size_t size_Q  = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K  = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V  = size_K;
    size_t size_dO = size_Q;
    size_t size_P  = (size_t)total_padded_q * head_num * max_seq_kv;

    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<DataType> h_grad_O(size_dO);
    std::vector<DataType> h_P(size_P, DataType(0.0f));
    std::vector<DataType> h_grad_Q_gpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_gpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_gpu(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_Q_cpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_cpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_cpu(size_V, DataType(0.0f));

    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_dO; i++) h_grad_O[i] = DataType(dis(gen));

    if constexpr(std::is_same<DataType, float>::value)
    {
        for(size_t i = 0; i < size_Q; i++) h_Q[i] = float(hip_bfloat16(h_Q[i]));
        for(size_t i = 0; i < size_K; i++) h_K[i] = float(hip_bfloat16(h_K[i]));
        for(size_t i = 0; i < size_V; i++) h_V[i] = float(hip_bfloat16(h_V[i]));
        for(size_t i = 0; i < size_dO; i++) h_grad_O[i] = float(hip_bfloat16(h_grad_O[i]));
    }

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    DataType *d_Q, *d_K, *d_V, *d_O, *d_dO;
    float* d_softmax_lse;
    DataType *d_dQ, *d_dK, *d_dV;
    int *d_cu_sq, *d_cu_sqp, *d_cu_skv, *d_cu_skvp;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, total_padded_q * head_num * head_dim * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dO, size_dO * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_softmax_lse,
                        FwdLauncher::calc_workspace_size(total_padded_q) > 0
                            ? FwdLauncher::calc_workspace_size(total_padded_q)
                            : sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dQ, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dK, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dV, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_cu_sq, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_sqp, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_skv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_skvp, (bs + 1) * sizeof(int)));
    if(total_padded_q > 0)
        HIP_CHECK(hipMalloc(&d_padded_q_to_batch, total_padded_q * sizeof(int)));
    else
        d_padded_q_to_batch = nullptr;

    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dO, h_grad_O.data(), size_dO * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_sq, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_sqp, h_cu_seqlens_q_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_skv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_skvp, h_cu_seqlens_kv_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    FwdLauncher::run_attn_fwd_kernel(d_Q, d_K, d_V, static_cast<const DataType*>(nullptr), 0.0f,
                                     sqr_dk_scale, d_O, d_softmax_lse,
                                     d_cu_sq, d_cu_sqp, d_cu_skv, d_cu_skvp,
                                     d_padded_q_to_batch, total_padded_q, bs);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_softmax_lse(total_padded_q * head_num);
    HIP_CHECK(hipMemcpy(h_softmax_lse.data(), d_softmax_lse,
                        h_softmax_lse.size() * sizeof(float), hipMemcpyDeviceToHost));

    reference_attn_probs_bf16_dots_with_given_lse(
        h_Q, h_K, bs, head_num, max_seq_kv, head_dim, sqr_dk_scale, Config::mask_type,
        h_cu_seqlens_q, h_cu_seqlens_q_padded, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
        h_softmax_lse, h_P);

    // bf16_weights=false: MFMA bwd recomputes P in float (Option A).
    attn_backward(h_Q.data(), h_K.data(), h_V.data(), h_grad_O.data(),
                  h_P.data(), static_cast<const DataType*>(nullptr), 0.0f,
                  h_grad_Q_cpu.data(), h_grad_K_cpu.data(), h_grad_V_cpu.data(),
                  bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                  h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                  h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                  total_padded_q, total_padded_kv_seq,
                  max_seq_q, false);

    HIP_CHECK(hipMemset(d_dQ, 0, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMemset(d_dK, 0, size_K * sizeof(DataType)));
    HIP_CHECK(hipMemset(d_dV, 0, size_V * sizeof(DataType)));

    BwdLauncher::run_attn_bwd_kernel(d_Q, d_K, d_V, d_dO, d_softmax_lse,
                                  d_dQ, d_dK, d_dV, sqr_dk_scale,
                                  d_cu_sq, d_cu_sqp, d_cu_skv, d_cu_skvp, bs);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_grad_Q_gpu.data(), d_dQ, size_Q * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_K_gpu.data(), d_dK, size_K * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_V_gpu.data(), d_dV, size_V * sizeof(DataType), hipMemcpyDeviceToHost));

    std::cout << "  [BWD] ";
    check_output(h_grad_Q_gpu, h_grad_Q_cpu, bs, head_num, head_dim,
                 h_cu_seqlens_q, h_cu_seqlens_q_padded, "grad_Q", 1e-2f, 1e-2f, false);
    std::cout << "  [BWD] ";
    check_output(h_grad_K_gpu, h_grad_K_cpu, bs, head_num, head_dim,
                 h_cu_seqlens_kv, h_cu_seqlens_kv_padded, "grad_K", 1e-2f, 1e-2f, false);
    std::cout << "  [BWD] ";
    check_output(h_grad_V_gpu, h_grad_V_cpu, bs, head_num, head_dim,
                 h_cu_seqlens_kv, h_cu_seqlens_kv_padded, "grad_V", 1e-2f, 1e-2f, false);

    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O)); HIP_CHECK(hipFree(d_dO)); HIP_CHECK(hipFree(d_softmax_lse));
    HIP_CHECK(hipFree(d_dQ)); HIP_CHECK(hipFree(d_dK)); HIP_CHECK(hipFree(d_dV));
    HIP_CHECK(hipFree(d_cu_sq)); HIP_CHECK(hipFree(d_cu_sqp));
    HIP_CHECK(hipFree(d_cu_skv)); HIP_CHECK(hipFree(d_cu_skvp));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));

    return true;
}

// ---------------------------------------------------------------------------
// Run one test case (fwd + bwd)
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void run_test_case(int bs,
                   bool varlen_q, int fix_sq,
                   bool varlen_kv, int fix_skv,
                   const std::string& label)
{
    constexpr int max_seq_q  = Config::max_seq_q;
    constexpr int max_seq_kv = Config::max_seq_kv;

    std::mt19937 gen(42);

    std::vector<int> h_cu_sq, h_cu_sqp, h_q2b;
    std::vector<int> h_cu_skv, h_cu_skvp;
    int total_padded_kv_seq;
    int total_padded_q;
    int total_actual_kv_seq;

    if(varlen_q)
        total_padded_q = build_varlen_cu_seqlens_q(
            bs, max_seq_q, gen, h_cu_sq, h_cu_sqp, h_q2b);
    else
        total_padded_q = build_fixed_cu_seqlens_q(
            bs, fix_sq, h_cu_sq, h_cu_sqp, h_q2b);

    if(varlen_kv)
        build_cu_seqlens_kv(
            bs, max_seq_kv, gen, h_cu_skv, h_cu_skvp,
            total_actual_kv_seq, total_padded_kv_seq);
    else
        build_fixed_cu_seqlens_kv(
            bs, fix_skv, h_cu_skv, h_cu_skvp, total_padded_kv_seq);

    std::cout << "\n===== " << label << " =====" << std::endl;
    std::cout << "  bs=" << bs
              << "  max_sq=" << max_seq_q << "  max_skv=" << max_seq_kv
              << "  total_padded_q=" << total_padded_q
              << "  total_padded_kv=" << total_padded_kv_seq << std::endl;

    test_fwd<DataType, Config>(
        bs, varlen_q, fix_sq, varlen_kv, fix_skv, label,
        h_cu_sq, h_cu_sqp, h_q2b, h_cu_skv, h_cu_skvp,
        total_padded_q, total_padded_kv_seq);

    test_bwd<DataType, Config>(
        bs, varlen_q, fix_sq, varlen_kv, fix_skv, label,
        h_cu_sq, h_cu_sqp, h_q2b, h_cu_skv, h_cu_skvp,
        total_padded_q, total_padded_kv_seq);

    std::cout << "====================================\n" << std::endl;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    // Test 1: sq∈[1,16] varlen+pad, skv∈[2,16] varlen+pad
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 16>;
        run_test_case<float, Cfg>(2048, true, 0, true, 0,
            "Test 1: sq varlen+pad [1,16], skv varlen+pad [2,16]");
    }

    // Test 2: sq=1 fixed, skv∈[2,16] varlen+pad
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 1>;
        run_test_case<float, Cfg>(2048, false, 1, true, 0,
            "Test 2: sq=1 fixed, skv varlen+pad [2,16]");
    }

    // Test 3: sq=16, skv=16; fixed, no padding
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 16>;
        run_test_case<float, Cfg>(2048, false, 16, false, 16,
            "Test 3: sq=16 fixed, skv=16 fixed");
    }

    // Test 4: sq=17, skv=17; fixed, no padding
    {
        using Cfg = FmhaKernelConfig<8, 17, 128, 256, false, CausalMaskType::DISABLE, 17>;
        run_test_case<float, Cfg>(2048, false, 17, false, 17,
            "Test 4: sq=17 fixed, skv=17 fixed");
    }

    return 0;
}
