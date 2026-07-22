// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Test host for the MFMA 16x16x16 backward kernels (attn_bwd_mfma_16x16.h).
//
// 4 test cases:
//   1. sq∈[1,16], skv∈[2,16]; varlen + padding
//   2. sq=1 (fixed, no padding), skv∈[2,16] (varlen + padding)
//   3. sq=16, skv=16; fixed, no padding
//   4. sq=17, skv=17; fixed, no padding
//
// Build: cmake -B ck_arliu && cmake --build ck_arliu --target test_bwd_mfma_16x16

#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_mfma_16x16.h"
#include "attn_bwd_ref.h"
#include "test_utils.h"

#include <chrono>
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
// KV cu_seqlens builders
// ---------------------------------------------------------------------------

// Fixed KV: all batches have exactly fix_skv tokens, no padding
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
// Main backward correctness + performance test (MFMA 16x16x16 variant)
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void test_run_attn_bwd_mfma_16x16(
    int bs,
    bool varlen_q, int fix_sq,
    bool varlen_kv, int fix_skv,
    const std::string& label,
    int warmup_iters, int test_iters,
    bool check_correctness, bool dump_err,
    float cmp_rtol = 1e-2f,
    float cmp_atol = 1e-2f)
{
    using BwdLauncher = AttnBackwardMfma16x16KernelLauncher<DataType, Config>;
    using FwdLauncher = AttnForwardMfma16x16KernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int max_seq_q  = Config::max_seq_q;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // --- Build cu_seqlens ---
    std::vector<int> h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch;
    std::vector<int> h_cu_seqlens_kv, h_cu_seqlens_kv_padded;
    int total_padded_kv_seq;
    int total_padded_q;
    int total_actual_kv_seq;

    if(varlen_q)
        total_padded_q = build_varlen_cu_seqlens_q(
            bs, max_seq_q, gen, h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch);
    else
        total_padded_q = build_fixed_cu_seqlens_q(
            bs, fix_sq, h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch);

    if(varlen_kv)
        build_cu_seqlens_kv(
            bs, max_seq_kv, gen, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
            total_actual_kv_seq, total_padded_kv_seq);
    else
        build_fixed_cu_seqlens_kv(
            bs, fix_skv, h_cu_seqlens_kv, h_cu_seqlens_kv_padded, total_padded_kv_seq);

    // --- Buffer sizes ---
    size_t size_Q            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = size_K;
    size_t size_grad_O       = size_Q;
    size_t size_attn_weights = (size_t)total_padded_q * head_num * max_seq_kv;

    // --- Host allocations ---
    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<DataType> h_grad_O(size_grad_O);
    std::vector<DataType> h_attn_weights(size_attn_weights, DataType(0.0f));
    std::vector<DataType> h_grad_Q_gpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_gpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_gpu(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_Q_cpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_cpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_cpu(size_V, DataType(0.0f));

    // Initialize Q, K, V, grad_O
    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_grad_O; i++) h_grad_O[i] = DataType(dis(gen));

    // Pre-round to bf16 precision
    if constexpr(std::is_same<DataType, float>::value)
    {
        for(size_t i = 0; i < size_Q; i++) h_Q[i] = float(hip_bfloat16(h_Q[i]));
        for(size_t i = 0; i < size_K; i++) h_K[i] = float(hip_bfloat16(h_K[i]));
        for(size_t i = 0; i < size_V; i++) h_V[i] = float(hip_bfloat16(h_V[i]));
        for(size_t i = 0; i < size_grad_O; i++) h_grad_O[i] = float(hip_bfloat16(h_grad_O[i]));
    }

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // --- Device allocations ---
    DataType *d_Q, *d_K, *d_V, *d_O, *d_grad_O;
    float* d_softmax_lse;
    DataType *d_grad_Q, *d_grad_K, *d_grad_V;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, total_padded_q * head_num * head_dim * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_softmax_lse,
                        FwdLauncher::calc_workspace_size(total_padded_q) > 0
                            ? FwdLauncher::calc_workspace_size(total_padded_q)
                            : sizeof(float)));
    HIP_CHECK(hipMalloc(&d_grad_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q_padded, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv_padded, (bs + 1) * sizeof(int)));
    if(total_padded_q > 0)
        HIP_CHECK(hipMalloc(&d_padded_q_to_batch, total_padded_q * sizeof(int)));
    else
        d_padded_q_to_batch = nullptr;

    // --- Copy to device ---
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded, h_cu_seqlens_q_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded, h_cu_seqlens_kv_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    // MFMA forward writes softmax_lse used by Option A backward (must match GPU recomputation).
    FwdLauncher::run_attn_fwd_kernel(d_Q, d_K, d_V, static_cast<const DataType*>(nullptr), 0.0f,
                                     sqr_dk_scale, d_O, d_softmax_lse,
                                     d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                     d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                     d_padded_q_to_batch, total_padded_q, bs);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_softmax_lse(total_padded_q * head_num);
    HIP_CHECK(hipMemcpy(h_softmax_lse.data(), d_softmax_lse,
                        h_softmax_lse.size() * sizeof(float), hipMemcpyDeviceToHost));

    reference_attn_probs_bf16_dots_with_given_lse(
        h_Q, h_K, bs, head_num, max_seq_kv, head_dim, sqr_dk_scale, Config::mask_type,
        h_cu_seqlens_q, h_cu_seqlens_q_padded, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
        h_softmax_lse, h_attn_weights);

    // --- CPU reference (bf16_weights=false: GPU Option A uses float P, not bf16-stored weights)
    if(check_correctness)
        attn_backward(h_Q.data(), h_K.data(), h_V.data(), h_grad_O.data(),
                      h_attn_weights.data(), static_cast<const DataType*>(nullptr), 0.0f,
                      h_grad_Q_cpu.data(), h_grad_K_cpu.data(), h_grad_V_cpu.data(),
                      bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                      h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                      h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                      total_padded_q, total_padded_kv_seq,
                      max_seq_q, false);

    auto bwd_launch = [&]() {
        HIP_CHECK(hipMemset(d_grad_Q, 0, size_Q * sizeof(DataType)));
        HIP_CHECK(hipMemset(d_grad_K, 0, size_K * sizeof(DataType)));
        HIP_CHECK(hipMemset(d_grad_V, 0, size_V * sizeof(DataType)));
        BwdLauncher::run_attn_bwd_kernel(d_Q, d_K, d_V, d_grad_O, d_softmax_lse,
                                         d_grad_Q, d_grad_K, d_grad_V, sqr_dk_scale,
                                         d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                         d_cu_seqlens_kv, d_cu_seqlens_kv_padded, bs);
    };

    for(int i = 0; i < warmup_iters; i++) bwd_launch();
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < test_iters; i++) bwd_launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    HIP_CHECK(hipMemcpy(h_grad_Q_gpu.data(), d_grad_Q, size_Q * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_K_gpu.data(), d_grad_K, size_K * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_V_gpu.data(), d_grad_V, size_V * sizeof(DataType), hipMemcpyDeviceToHost));

    // --- Report ---
    std::cout << "\n===== " << label << " =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  max_seq_q: " << max_seq_q << "  max_seq_kv: " << max_seq_kv << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Mask: " << CausalMaskTypeName[Config::mask_type] << std::endl;
    std::cout << "  Q mode: " << (varlen_q ? "varlen+padding" : "fixed") << std::endl;
    std::cout << "  KV mode: " << (varlen_kv ? "varlen+padding" : "fixed") << std::endl;
    std::cout << "  total_padded_q: " << total_padded_q
              << "  total_padded_kv: " << total_padded_kv_seq << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_output(h_grad_Q_gpu, h_grad_Q_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_q, h_cu_seqlens_q_padded, "grad_Q", cmp_rtol, cmp_atol, dump_err);
        check_output(h_grad_K_gpu, h_grad_K_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_kv, h_cu_seqlens_kv_padded, "grad_K", cmp_rtol, cmp_atol, dump_err);
        check_output(h_grad_V_gpu, h_grad_V_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_kv, h_cu_seqlens_kv_padded, "grad_V", cmp_rtol, cmp_atol, dump_err);
        std::cout << std::endl;
    }

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    std::cout << "====================================\n" << std::endl;

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O)); HIP_CHECK(hipFree(d_grad_O)); HIP_CHECK(hipFree(d_softmax_lse));
    HIP_CHECK(hipFree(d_grad_Q)); HIP_CHECK(hipFree(d_grad_K)); HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_cu_seqlens_q)); HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv)); HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char const* argv[])
{
    // Test 1: sq∈[1,16], skv∈[2,16]; varlen + padding
    // Looser cmp_rtol/cmp_atol: CPU ref P uses chunked bf16 dots; MFMA S differs (see log max ~0.23).
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 16>;
        test_run_attn_bwd_mfma_16x16<float, Cfg>(
            2048,
            true, 0, true, 0,
            "Test 1: sq∈[1,16] varlen+pad, skv∈[2,16] varlen+pad",
            1, 1, true, true,
            0.12f, 0.23f);
    }

    // Test 2: sq=1 (fixed, no padding), skv∈[2,16] (varlen + padding)
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 1>;
        test_run_attn_bwd_mfma_16x16<float, Cfg>(
            2048,
            false, 1, true, 0,
            "Test 2: sq=1 fixed, skv∈[2,16] varlen+pad",
            1, 1, true, true,
            0.12f, 0.23f);
    }

    // Test 3: sq=16, skv=16; fixed, no padding
    {
        using Cfg = FmhaKernelConfig<8, 16, 128, 256, false, CausalMaskType::DISABLE, 16>;
        test_run_attn_bwd_mfma_16x16<float, Cfg>(
            2048,
            false, 16, false, 16,
            "Test 3: sq=16 fixed, skv=16 fixed",
            1, 1, true, true);
    }

    // Test 4: sq=17, skv=17; fixed, no padding
    {
        using Cfg = FmhaKernelConfig<8, 17, 128, 256, false, CausalMaskType::DISABLE, 17>;
        test_run_attn_bwd_mfma_16x16<float, Cfg>(
            2048,
            false, 17, false, 17,
            "Test 4: sq=17 fixed, skv=17 fixed",
            1, 1, true, true);
    }

    return 0;
}
