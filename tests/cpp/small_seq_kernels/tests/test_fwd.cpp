/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
//
// clang-format off
// Build: cmake -B build && cmake --build build && ./build/test_fwd
// clang-format on

#include "attn_fwd.h"
#include "attn_fwd_ref.h"
#include "test_utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Main forward correctness + performance test
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void test_run_attn_fwd_kernel(
    int bs, float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnForwardKernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // --- Build cu_seqlens_q ---
    std::vector<int> h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch;
    int total_padded_q = build_cu_seqlens_q(bs, gen, h_cu_seqlens_q, h_cu_seqlens_q_padded,
                                            h_padded_q_to_batch);
    int total_actual_q_seq = h_cu_seqlens_q[bs];

    // --- Build cu_seqlens_kv ---
    std::vector<int> h_cu_seqlens_kv, h_cu_seqlens_kv_padded;
    int total_actual_kv_seq, total_padded_kv_seq;
    build_cu_seqlens_kv(bs, max_seq_kv, gen, h_cu_seqlens_kv, h_cu_seqlens_kv_padded,
                        total_actual_kv_seq, total_padded_kv_seq);

    // --- Buffer sizes ---
    size_t size_Q            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_O            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_dropout_mask = (size_t)bs * head_num * max_seq_kv;

    // --- Host allocations ---
    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<DataType> h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_O_gpu(size_O, DataType(0.0f));
    std::vector<DataType> h_O_cpu(size_O, DataType(0.0f));

    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_dropout_mask; i++)
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);

    // --- CPU reference ---
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_forward(h_Q.data(), h_K.data(), h_V.data(),
                     Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                     dropout_p, h_O_cpu.data(), static_cast<DataType*>(nullptr),
                     bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                     h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                     h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data());

    // --- Device allocations ---
    DataType *d_Q, *d_K, *d_V, *d_dropout_mask, *d_O, *d_workspace;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
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
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask, h_dropout_mask.data(),
                        size_dropout_mask * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded, h_cu_seqlens_q_padded.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded, h_cu_seqlens_kv_padded.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    auto launch = [&]() {
        Launcher::run_attn_fwd_kernel(d_Q, d_K, d_V,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p, sqr_dk_scale, d_O, d_workspace,
                                      d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                      d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                      d_padded_q_to_batch, total_padded_q, bs);
    };

    for(int i = 0; i < warmup_iters; i++) launch();
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < test_iters; i++) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));

    // --- Report ---
    double avg_kv_seq           = static_cast<double>(total_actual_kv_seq) / bs;
    double active_q             = static_cast<double>(total_actual_q_seq);
    double flops_per_batch_head = 2.0 * avg_kv_seq * head_dim + 2.0 * head_dim * avg_kv_seq;
    double total_flops          = flops_per_batch_head * active_q * head_num;
    double tflops               = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    size_t bytes_read = (size_Q + size_K + size_V) * sizeof(DataType);
    if(Config::enable_dropout_mask) bytes_read += size_dropout_mask * sizeof(DataType);
    size_t bytes_write    = size_O * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    std::cout << "\n===== run_attn_fwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Q seq (active/total): " << total_actual_q_seq << "/" << bs << std::endl;
    std::cout << "  KV seq (avg): " << std::fixed << std::setprecision(2) << avg_kv_seq
              << " (max: " << max_seq_kv << ")" << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled") << std::endl;
    std::cout << "  Mask: " << CausalMaskTypeName[Config::mask_type] << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_output(h_O_gpu, h_O_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_q, h_cu_seqlens_q_padded, "Output", 1e-2f, 1e-2f, dump_err);
        std::cout << std::endl;
    }

    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read:     " << std::fixed << std::setprecision(2)
              << bytes_read / 1e6 << " MB" << std::endl;
    std::cout << "  Total data write:    " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Workspace:           " << workspace_size / 1e6 << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms
              << " ms" << std::endl;
    std::cout << "  Bandwidth:    " << std::fixed << std::setprecision(2) << bandwidth_gbps
              << " GB/s" << std::endl;
    std::cout << "  TFLOPS:       " << std::fixed << std::setprecision(2) << tflops << std::endl;
    std::cout << "====================================\n" << std::endl;

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_dropout_mask)); HIP_CHECK(hipFree(d_O)); HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_q)); HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv)); HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
}

// ---------------------------------------------------------------------------
// Corner-case test: explicit Q seqlens provided by caller
// ---------------------------------------------------------------------------

template <typename DataType, typename Config>
void test_run_attn_fwd_with_seqlens(const std::vector<int>& h_cu_seqlens_q,
                                    const std::vector<int>& h_cu_seqlens_q_padded,
                                    const std::vector<int>& h_padded_q_to_batch,
                                    int total_padded_q,
                                    float dropout_p,
                                    bool check_correctness,
                                    bool dump_err,
                                    const std::string& test_name)
{
    using Launcher = AttnForwardKernelLauncher<DataType, Config>;

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
    size_t size_O            = size_Q;
    size_t size_dropout_mask = (size_t)total_padded_q * head_num * max_seq_kv;
    size_t size_workspace    = size_dropout_mask;

    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V), h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_O_gpu(size_O, DataType(0.0f)), h_O_cpu(size_O, DataType(0.0f));

    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_dropout_mask; i++)
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_forward(h_Q.data(), h_K.data(), h_V.data(),
                     Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                     dropout_p, h_O_cpu.data(), static_cast<DataType*>(nullptr),
                     bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                     h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                     h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data());

    DataType *d_Q, *d_K, *d_V, *d_dropout_mask, *d_O, *d_workspace;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
    size_t workspace_size = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size > 0 ? workspace_size : 1));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q_padded, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv_padded, (bs + 1) * sizeof(int)));
    if(total_padded_q > 0)
        HIP_CHECK(hipMalloc(&d_padded_q_to_batch, total_padded_q * sizeof(int)));
    else
        d_padded_q_to_batch = nullptr;

    // Init workspace to -1e9f so padding rows are defined for softmax
    std::vector<DataType> h_workspace_init(size_workspace, DataType(-1e9f));
    HIP_CHECK(hipMemcpy(d_workspace, h_workspace_init.data(),
                        size_workspace * sizeof(DataType), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask, h_dropout_mask.data(),
                        size_dropout_mask * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded, h_cu_seqlens_q_padded.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded, h_cu_seqlens_kv_padded.data(),
                        (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    Launcher::run_attn_fwd_kernel(d_Q, d_K, d_V,
                                  Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                  dropout_p, sqr_dk_scale, d_O, d_workspace,
                                  d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                  d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                  d_padded_q_to_batch, total_padded_q, bs);

    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));

    if(check_correctness)
    {
        check_output(h_O_gpu, h_O_cpu, bs, head_num, head_dim,
                     h_cu_seqlens_q, h_cu_seqlens_q_padded,
                     test_name + " Output", 1e-2f, 1e-2f, dump_err);
    }

    HIP_CHECK(hipFree(d_Q)); HIP_CHECK(hipFree(d_K)); HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_dropout_mask)); HIP_CHECK(hipFree(d_O)); HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_q)); HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv)); HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch) HIP_CHECK(hipFree(d_padded_q_to_batch));
}

// ---------------------------------------------------------------------------
// Functor for TestRunner
// ---------------------------------------------------------------------------

struct RunFwd {
    template <typename DataType, typename Config>
    void operator()(int bs, float dropout_p, int warmup, int iters, bool check, bool dump) const {
        test_run_attn_fwd_kernel<DataType, Config>(bs, dropout_p, warmup, iters, check, dump);
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char const* argv[])
{
    std::cout << "\n========== Correctness Test (SEQ_KV 2..16, bs=30720) ==========" << std::endl;

    TestRunner<2, 16>::run<float, 30720, 32, 128, 256, false, CausalMaskType::DISABLE>(
        RunFwd{}, 0.0f, 1, 1, true, true);

    std::cout << "\n========== Correctness Test (mixed Q=0/1, SEQ_KV=8, bs=128) ==========" << std::endl;
    {
        using MixedConfig = FmhaKernelConfig<8, 8, 128, 256, false, CausalMaskType::DISABLE>;
        test_run_attn_fwd_kernel<float, MixedConfig>(128, 0, 1, 1, true, true);
    }

    std::cout << "\n========== Performance Test (bfloat16, SEQ_KV 2..16) ==========" << std::endl;
    TestRunner<2, 16>::run<hip_bfloat16, 30720, 32, 128, 256, false, CausalMaskType::DISABLE>(
        RunFwd{}, 0.0f, 3, 5, false, false);

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

        using CornerConfig = FmhaKernelConfig<8, 8, 128, 256, false, CausalMaskType::DISABLE>;
        test_run_attn_fwd_with_seqlens<float, CornerConfig>(
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

        using CornerConfig = FmhaKernelConfig<8, 8, 128, 256, false, CausalMaskType::DISABLE>;
        test_run_attn_fwd_with_seqlens<float, CornerConfig>(
            h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch,
            total_padded_q, 0.0f, true, true, "Q padded > actual");
    }

    return 0;
}
