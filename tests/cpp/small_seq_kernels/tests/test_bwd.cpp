// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// clang-format off
// Build: cmake -B build && cmake --build build && ./build/test_bwd
// clang-format on

#include "attn_bwd.h"
#include "attn_bwd_ref.h"
#include "test_utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Main backward correctness + performance test
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void test_run_attn_bwd_kernel(
    int bs, float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnBackwardKernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::bernoulli_distribution q_present_dis(0.5);

    // --- Build cu_seqlens_q ---
    std::vector<int> h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch;
    int total_padded_q = build_cu_seqlens_q(bs, gen, h_cu_seqlens_q, h_cu_seqlens_q_padded,
                                            h_padded_q_to_batch);
    int total_actual_q = h_cu_seqlens_q[bs];

    // --- Build cu_seqlens_kv ---
    std::vector<int> h_cu_seqlens_kv, h_cu_seqlens_kv_padded;
    int total_actual_kv_seq, total_padded_kv_seq;
    build_cu_seqlens_kv(bs, max_seq_kv, gen, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
                        total_actual_kv_seq, total_padded_kv_seq);

    // --- Buffer sizes ---
    size_t size_Q            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_grad_O       = size_Q;
    size_t size_attn_weights = (size_t)total_padded_q * head_num * max_seq_kv;
    size_t size_dropout_mask = size_attn_weights;

    // --- Host allocations ---
    std::vector<DataType> h_Q(size_Q, DataType(0.0f));
    std::vector<DataType> h_K(size_K, DataType(0.0f));
    std::vector<DataType> h_V(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_O(size_grad_O, DataType(0.0f));
    std::vector<DataType> h_attn_weights(size_attn_weights, DataType(0.0f));
    std::vector<DataType> h_dropout_mask(size_dropout_mask, DataType(1.0f));
    std::vector<DataType> h_grad_Q_gpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_gpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_gpu(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_Q_cpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_cpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_cpu(size_V, DataType(0.0f));

    // Initialize Q and grad_O for active-Q batches
    for(int b = 0; b < bs; b++)
    {
        if(h_cu_seqlens_q[b + 1] == h_cu_seqlens_q[b]) continue;
        int q_off = h_cu_seqlens_q_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            int base = (q_off * head_num + h) * head_dim;
            for(int d = 0; d < head_dim; d++)
            {
                h_Q[base + d]      = DataType(dis(gen));
                h_grad_O[base + d] = DataType(dis(gen));
            }
        }
    }

    // Initialize K/V
    for(int b = 0; b < bs; b++)
    {
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int kv_off = h_cu_seqlens_kv_padded[b];
        for(int h = 0; h < head_num; h++)
            for(int s = 0; s < kv_seq; s++)
            {
                int base = (kv_off + s) * head_num * head_dim + h * head_dim;
                for(int d = 0; d < head_dim; d++)
                {
                    h_K[base + d] = DataType(dis(gen));
                    h_V[base + d] = DataType(dis(gen));
                }
            }
    }

    // Initialize attn_weights (normalized per row)
    for(int b = 0; b < bs; b++)
    {
        if(h_cu_seqlens_q[b + 1] == h_cu_seqlens_q[b]) continue;
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int q_off  = h_cu_seqlens_q_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            int base  = (q_off * head_num + h) * max_seq_kv;
            float sum = 0.0f;
            for(int j = 0; j < kv_seq; j++)
            {
                h_attn_weights[base + j] = DataType(std::abs(dis(gen)));
                sum += float(h_attn_weights[base + j]);
            }
            for(int j = kv_seq; j < max_seq_kv; j++)
                h_attn_weights[base + j] = DataType(0.0f);
            if(sum > 0.0f)
                for(int j = 0; j < kv_seq; j++)
                    h_attn_weights[base + j] =
                        DataType(float(h_attn_weights[base + j]) / sum);
        }
    }

    // Initialize dropout mask
    for(size_t i = 0; i < size_dropout_mask; i++)
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);

    // --- CPU reference ---
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_backward(h_Q.data(), h_K.data(), h_V.data(), h_grad_O.data(),
                      h_attn_weights.data(),
                      Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                      dropout_p, h_grad_Q_cpu.data(), h_grad_K_cpu.data(), h_grad_V_cpu.data(),
                      bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                      h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                      h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                      total_padded_q, total_padded_kv_seq);

    // --- Device allocations ---
    DataType *d_Q, *d_K, *d_V, *d_grad_O, *d_attn_weights, *d_dropout_mask;
    DataType *d_grad_Q, *d_grad_K, *d_grad_V, *d_workspace;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O > 0 ? size_grad_O * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights > 0 ? size_attn_weights * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask > 0 ? size_dropout_mask * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_grad_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
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

    size_t workspace_size = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size > 0 ? workspace_size : 1));

    // --- Copy to device ---
    if(size_Q > 0)
        HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_grad_O > 0)
        HIP_CHECK(hipMemcpy(d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_attn_weights > 0)
        HIP_CHECK(hipMemcpy(d_attn_weights, h_attn_weights.data(),
                            size_attn_weights * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_dropout_mask > 0)
        HIP_CHECK(hipMemcpy(d_dropout_mask, h_dropout_mask.data(),
                            size_dropout_mask * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded, h_cu_seqlens_q_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded, h_cu_seqlens_kv_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    auto bwd_launch = [&]() {
        Launcher::run_attn_bwd_kernel(d_Q, d_K, d_V, d_grad_O, d_attn_weights,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p, sqr_dk_scale,
                                      d_grad_Q, d_grad_K, d_grad_V, d_workspace,
                                      d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                      d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                      d_padded_q_to_batch, total_padded_q, bs);
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
    double avg_kv_seq        = total_actual_kv_seq / double(bs);
    double avg_padded_kv_seq = total_padded_kv_seq / double(bs);
    double flops_per_active  = 4.0 * 2.0 * avg_kv_seq * head_dim;
    double total_flops       = flops_per_active * total_actual_q * head_num;
    double tflops            = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    size_t bytes_read =
        (size_Q + size_K + size_V + size_grad_O + size_attn_weights) * sizeof(DataType);
    if(Config::enable_dropout_mask) bytes_read += size_dropout_mask * sizeof(DataType);
    size_t bytes_write    = (size_Q + size_K + size_V) * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    std::cout << "\n===== run_attn_bwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Active Q batches: " << total_actual_q << " / " << bs << std::endl;
    std::cout << "  KV max: " << max_seq_kv << "  KV avg: " << std::fixed << std::setprecision(2)
              << avg_kv_seq << "  KV avg padded: " << avg_padded_kv_seq << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled") << std::endl;
    std::cout << "  Mask: " << CausalMaskTypeName[Config::mask_type] << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_grad_q(h_grad_Q_gpu, h_grad_Q_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_q, h_cu_seqlens_q_padded, 1e-2f, 1e-2f, dump_err);
        check_array(h_grad_K_gpu, h_grad_K_cpu, "grad_K", 1e-2f, 1e-2f, dump_err);
        check_array(h_grad_V_gpu, h_grad_V_cpu, "grad_V", 1e-2f, 1e-2f, dump_err);
        std::cout << std::endl;
    }

    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read:     " << std::fixed << std::setprecision(2) << bytes_read / 1e6  << " MB" << std::endl;
    std::cout << "  Total data write:    " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Workspace:           " << workspace_size / 1e6 << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    std::cout << "  Bandwidth:    " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "  TFLOPS:       " << std::fixed << std::setprecision(2) << tflops << std::endl;
    std::cout << "====================================\n" << std::endl;

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_grad_O)); HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_grad_Q)); HIP_CHECK(hipFree(d_grad_K)); HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_q)); HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv)); HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
}

// ---------------------------------------------------------------------------
// Corner-case test: explicit Q seqlens provided by caller
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void test_run_attn_bwd_with_seqlens(const std::vector<int>& h_cu_seqlens_q,
                                    const std::vector<int>& h_cu_seqlens_q_padded,
                                    const std::vector<int>& h_padded_q_to_batch,
                                    int total_padded_q,
                                    float dropout_p,
                                    bool check_correctness,
                                    bool dump_err,
                                    const std::string& test_name)
{
    using Launcher = AttnBackwardKernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;
    int bs                   = static_cast<int>(h_cu_seqlens_q.size()) - 1;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<int> h_cu_seqlens_kv, h_cu_seqlens_kv_padded;
    int total_actual_kv_seq, total_padded_kv_seq;
    build_cu_seqlens_kv(bs, max_seq_kv, gen, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
                        total_actual_kv_seq, total_padded_kv_seq);

    size_t size_Q            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = size_K;
    size_t size_grad_O       = size_Q;
    size_t size_attn_weights = (size_t)total_padded_q * head_num * max_seq_kv;
    size_t size_dropout_mask = size_attn_weights;

    std::vector<DataType> h_Q(size_Q, DataType(0.0f)), h_K(size_K), h_V(size_K);
    std::vector<DataType> h_grad_O(size_grad_O), h_attn_weights(size_attn_weights);
    std::vector<DataType> h_dropout_mask(size_dropout_mask, DataType(1.0f));
    std::vector<DataType> h_grad_Q_gpu(size_Q), h_grad_K_gpu(size_K), h_grad_V_gpu(size_V);
    std::vector<DataType> h_grad_Q_cpu(size_Q), h_grad_K_cpu(size_K), h_grad_V_cpu(size_V);

    for(size_t i = 0; i < size_Q; i++)      h_Q[i]      = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++)      h_K[i]      = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++)      h_V[i]      = DataType(dis(gen));
    for(size_t i = 0; i < size_grad_O; i++) h_grad_O[i] = DataType(dis(gen));

    // Q and grad_O for active-Q batches only
    for(int b = 0; b < bs; b++)
    {
        if(h_cu_seqlens_q[b + 1] == h_cu_seqlens_q[b]) continue;
        int q_off = h_cu_seqlens_q_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            int base = (q_off * head_num + h) * head_dim;
            for(int d = 0; d < head_dim; d++)
            {
                h_Q[base + d]      = DataType(dis(gen));
                h_grad_O[base + d] = DataType(dis(gen));
            }
        }
    }

    // K/V
    for(int b = 0; b < bs; b++)
    {
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int kv_off = h_cu_seqlens_kv_padded[b];
        for(int h = 0; h < head_num; h++)
            for(int s = 0; s < kv_seq; s++)
            {
                int base = (kv_off + s) * head_num * head_dim + h * head_dim;
                for(int d = 0; d < head_dim; d++)
                {
                    h_K[base + d] = DataType(dis(gen));
                    h_V[base + d] = DataType(dis(gen));
                }
            }
    }

    // attn_weights (normalized per row)
    for(int b = 0; b < bs; b++)
    {
        if(h_cu_seqlens_q[b + 1] == h_cu_seqlens_q[b]) continue;
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int q_off  = h_cu_seqlens_q_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            int base  = (q_off * head_num + h) * max_seq_kv;
            float sum = 0.0f;
            for(int j = 0; j < kv_seq; j++)
            {
                h_attn_weights[base + j] = DataType(std::abs(dis(gen)));
                sum += float(h_attn_weights[base + j]);
            }
            for(int j = kv_seq; j < max_seq_kv; j++) h_attn_weights[base + j] = DataType(0.0f);
            if(sum > 0.0f)
                for(int j = 0; j < kv_seq; j++)
                    h_attn_weights[base + j] = DataType(float(h_attn_weights[base + j]) / sum);
        }
    }

    for(size_t i = 0; i < size_dropout_mask; i++)
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_backward(h_Q.data(), h_K.data(), h_V.data(), h_grad_O.data(),
                      h_attn_weights.data(),
                      Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                      dropout_p, h_grad_Q_cpu.data(), h_grad_K_cpu.data(), h_grad_V_cpu.data(),
                      bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                      h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                      h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                      total_padded_q, total_padded_kv_seq);

    DataType *d_Q, *d_K, *d_V, *d_grad_O, *d_attn_weights, *d_dropout_mask;
    DataType *d_grad_Q, *d_grad_K, *d_grad_V, *d_workspace;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O > 0 ? size_grad_O * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights > 0 ? size_attn_weights * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask > 0 ? size_dropout_mask * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_grad_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
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

    size_t workspace_size = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size > 0 ? workspace_size : 1));

    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_attn_weights, h_attn_weights.data(), size_attn_weights * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask, h_dropout_mask.data(), size_dropout_mask * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded, h_cu_seqlens_q_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded, h_cu_seqlens_kv_padded.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    if(workspace_size > 0) HIP_CHECK(hipMemset(d_workspace, 0, workspace_size));

    Launcher::run_attn_bwd_kernel(d_Q, d_K, d_V, d_grad_O, d_attn_weights,
                                  Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                  dropout_p, sqr_dk_scale,
                                  d_grad_Q, d_grad_K, d_grad_V, d_workspace,
                                  d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                  d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                  d_padded_q_to_batch, total_padded_q, bs);

    HIP_CHECK(hipMemcpy(h_grad_Q_gpu.data(), d_grad_Q, size_Q * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_K_gpu.data(), d_grad_K, size_K * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_V_gpu.data(), d_grad_V, size_V * sizeof(DataType), hipMemcpyDeviceToHost));

    if(check_correctness)
    {
        std::cout << "\n===== " << test_name << " =====" << std::endl;
        std::cout << "Correctness:" << std::endl;
        check_grad_q(h_grad_Q_gpu, h_grad_Q_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_q, h_cu_seqlens_q_padded, 1e-2f, 1e-2f, dump_err);
        check_array(h_grad_K_gpu, h_grad_K_cpu, "grad_K", 1e-2f, 1e-2f, dump_err);
        check_array(h_grad_V_gpu, h_grad_V_cpu, "grad_V", 1e-2f, 1e-2f, dump_err);
        std::cout << std::endl;
    }

    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_grad_O)); HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_grad_Q)); HIP_CHECK(hipFree(d_grad_K)); HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_q)); HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv)); HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char const* argv[])
{
    std::cout << "\n========== Correctness Test (bs=30720, SEQ_KV=16) ==========" << std::endl;
    using CorrConfig = FmhaKernelConfig<32, 16, 128, 128, false, CausalMaskType::DISABLE>;
    test_run_attn_bwd_kernel<float, CorrConfig>(30720, 0, 10, 10, true, true);

    std::cout << "\n========== Performance Test (bfloat16, TOP_LEFT mask) ==========" << std::endl;
    using PerfConfig = FmhaKernelConfig<32, 16, 128, 128, false, CausalMaskType::TOP_LEFT>;
    test_run_attn_bwd_kernel<hip_bfloat16, PerfConfig>(30720, 0, 3, 5, false, false);

    std::cout << "\n========== Mixed-Q Test (bs=128, 0/1 tokens) ==========" << std::endl;
    using MixedConfig = FmhaKernelConfig<4, 8, 64, 256, false, CausalMaskType::DISABLE>;
    test_run_attn_bwd_kernel<float, MixedConfig>(128, 0, 2, 5, true, true);

    std::cout << "\n========== Corner: Empty segments (even batches active, bs=128) =========="
              << std::endl;
    {
        const int corner_bs = 128;
        std::vector<int> h_cu_seqlens_q(corner_bs + 1);
        std::vector<int> h_cu_seqlens_q_padded(corner_bs + 1);
        std::vector<int> h_padded_q_to_batch(corner_bs / 2);
        h_cu_seqlens_q[0] = h_cu_seqlens_q_padded[0] = 0;
        for(int b = 0; b < corner_bs; b++)
        {
            int actual                    = (b % 2 == 0) ? 1 : 0;
            h_cu_seqlens_q[b + 1]        = h_cu_seqlens_q[b] + actual;
            h_cu_seqlens_q_padded[b + 1] = h_cu_seqlens_q_padded[b] + actual;
        }
        int total_padded_q = h_cu_seqlens_q_padded[corner_bs];
        for(int b = 0; b < corner_bs; b++)
            if(h_cu_seqlens_q_padded[b + 1] > h_cu_seqlens_q_padded[b])
                h_padded_q_to_batch[h_cu_seqlens_q_padded[b]] = b;

        using CornerConfig = FmhaKernelConfig<4, 8, 64, 256, false, CausalMaskType::DISABLE>;
        test_run_attn_bwd_with_seqlens<float, CornerConfig>(
            h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch,
            total_padded_q, 0.0f, true, true, "Empty segments");
    }

    std::cout << "\n========== Corner: Q padded > actual (2 slots per batch, bs=128) =========="
              << std::endl;
    {
        const int corner_bs = 128;
        std::vector<int> h_cu_seqlens_q(corner_bs + 1);
        std::vector<int> h_cu_seqlens_q_padded(corner_bs + 1);
        std::vector<int> h_padded_q_to_batch(256);
        for(int b = 0; b <= corner_bs; b++)
        {
            h_cu_seqlens_q[b]        = b;
            h_cu_seqlens_q_padded[b] = b * 2;
        }
        for(int i = 0; i < 256; i++) h_padded_q_to_batch[i] = i / 2;
        int total_padded_q = 256;

        using CornerConfig = FmhaKernelConfig<4, 8, 64, 256, false, CausalMaskType::DISABLE>;
        test_run_attn_bwd_with_seqlens<float, CornerConfig>(
            h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch,
            total_padded_q, 0.0f, true, true, "Q padded > actual");
    }

    return 0;
}
