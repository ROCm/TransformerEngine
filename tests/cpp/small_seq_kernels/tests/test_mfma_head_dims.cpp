// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MFMA 16x16x16 forward smoke + correctness for head_dim 128, 256, 512.
// Uses AttnForwardMfma16x16KernelLauncher (same path as test_fwd_mfma_16x16).
//
// Build / run: see ../README.md

#include "attn_common.h"
#include "attn_fwd_mfma_16x16.h"
#include "attn_fwd_ref.h"
#include "test_utils.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Uniform Q length per batch, no extra padding (actual == padded).
static int build_uniform_q(int bs,
                           int sq,
                           std::vector<int>& csq,
                           std::vector<int>& csqp,
                           std::vector<int>& q2b)
{
    csq.resize(bs + 1);
    csqp.resize(bs + 1);
    csq[0] = csqp[0] = 0;
    for(int b = 0; b < bs; b++)
    {
        csq[b + 1]  = csq[b] + sq;
        csqp[b + 1] = csqp[b] + sq;
    }
    int tot = csqp[bs];
    q2b.resize(tot);
    for(int b = 0; b < bs; b++)
        for(int i = csqp[b]; i < csqp[b + 1]; i++)
            q2b[i] = b;
    return tot;
}

template <typename DataType, typename Config>
void run_fwd_head_dim_case(const std::string& label, float dropout_p, bool dump_err)
{
    using Launcher = AttnForwardMfma16x16KernelLauncher<DataType, Config>;

    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;
    const int bs             = Config::bs;

    std::vector<int> h_cu_seqlens_q, h_cu_seqlens_q_padded, h_padded_q_to_batch;
    const int sq         = Config::max_seq_q;
    int total_padded_q   = build_uniform_q(bs, sq, h_cu_seqlens_q, h_cu_seqlens_q_padded,
                                         h_padded_q_to_batch);

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

    std::vector<DataType> h_Q(size_Q), h_K(size_K), h_V(size_V), h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_O_gpu(size_O, DataType(0.0f)), h_O_cpu(size_O, DataType(0.0f));

    for(size_t i = 0; i < size_Q; i++) h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++) h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++) h_V[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_dropout_mask; i++)
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);

    if constexpr(std::is_same<DataType, float>::value)
    {
        for(size_t i = 0; i < size_Q; i++) h_Q[i] = float(hip_bfloat16(h_Q[i]));
        for(size_t i = 0; i < size_K; i++) h_K[i] = float(hip_bfloat16(h_K[i]));
        for(size_t i = 0; i < size_V; i++) h_V[i] = float(hip_bfloat16(h_V[i]));
    }

    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_forward(h_Q.data(), h_K.data(), h_V.data(),
                 Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                 dropout_p, h_O_cpu.data(), static_cast<DataType*>(nullptr),
                 bs, head_num, max_seq_kv, head_dim, Config::mask_type,
                 h_cu_seqlens_q.data(), h_cu_seqlens_q_padded.data(),
                 h_cu_seqlens_kv.data(), h_cu_seqlens_kv_padded.data(),
                 true);

    DataType *d_Q, *d_K, *d_V, *d_dropout_mask, *d_O;
    float* d_softmax_lse;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
    size_t lse_bytes = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_softmax_lse, lse_bytes > 0 ? lse_bytes : sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q_padded, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv_padded, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_padded_q_to_batch, total_padded_q * sizeof(int)));

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
    HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                        total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    std::cout << "\n===== " << label << " (MFMA 16x16 fwd) =====" << std::endl;
    std::cout << "  bs=" << bs << " heads=" << head_num << " sq=" << sq << " max_seq_kv=" << max_seq_kv
              << " head_dim=" << head_dim << std::endl;

    Launcher::run_attn_fwd_kernel(d_Q, d_K, d_V,
                                  Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                  dropout_p, sqr_dk_scale, d_O, d_softmax_lse,
                                  d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                  d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                  d_padded_q_to_batch, total_padded_q);

    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));

    check_output(h_O_gpu, h_O_cpu, bs, head_num, head_dim,
                 h_cu_seqlens_q, h_cu_seqlens_q_padded,
                 label + " Output", 1e-2f, 1e-2f, dump_err);
    std::cout << "  PASS (vs CPU ref)\n" << std::endl;

    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipFree(d_softmax_lse));
    HIP_CHECK(hipFree(d_cu_seqlens_q));
    HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv));
    HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    HIP_CHECK(hipFree(d_padded_q_to_batch));
}

int main()
{
    constexpr int bs           = 4;
    constexpr int heads        = 4;
    constexpr int max_seq_kv   = 16;
    constexpr int sq           = 8; // max_seq_q; >4 so tests 16x16 path, not 4x4 dispatch
    const float dropout_p      = 0.0f;
    const bool dump_err        = true;

    std::cout << "MFMA 16x16 forward head-dimension sweep (CPU reference vs GPU kernel)\n";

    run_fwd_head_dim_case<float,
                          FmhaKernelConfig<bs, heads, max_seq_kv, 128, 256, false,
                                           CausalMaskType::DISABLE, sq>>(
        "head_dim_128", dropout_p, dump_err);

    run_fwd_head_dim_case<float,
                          FmhaKernelConfig<bs, heads, max_seq_kv, 256, 256, false,
                                           CausalMaskType::DISABLE, sq>>(
        "head_dim_256", dropout_p, dump_err);

    run_fwd_head_dim_case<float,
                          FmhaKernelConfig<bs, heads, max_seq_kv, 512, 256, false,
                                           CausalMaskType::DISABLE, sq>>(
        "head_dim_512", dropout_p, dump_err);

    std::cout << "All head-dimension MFMA forward tests finished successfully.\n";
    return 0;
}
