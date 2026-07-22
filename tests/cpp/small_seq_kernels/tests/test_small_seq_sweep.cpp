// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Small-sequence sweep benchmark matching the TransformerEngine benchmark:
//   bs=2048, nheads=32, hdim=128, bfloat16
//
// 1) Self-attention: seqlen_q == seqlen_kv = 1..17
//    Forward: mfma_4x4 for seq<=4, mfma_16x16 for all; backward: mfma_16x16.
//
// 2) Cross-attention: seqlen_q = 1, seqlen_kv = 2..16 (uniform per batch)
//    Forward + backward: mfma_16x16 only (kernel compiled with max_seq_q=1,
//    max_seq_kv=16).
//
// Outputs results in CSV format compatible with the TE benchmark CSV.

#include "attn_fwd_mfma.h"
#include "attn_fwd_mfma_16x16.h"
#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_ref.h"
#include "attn_common.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Build cu_seqlens with UNIFORM lengths (every batch has exactly seq_len)
// ---------------------------------------------------------------------------

static void build_cu_seqlens_uniform(int bs, int seq_len,
                                     std::vector<int>& cu_seqlens,
                                     std::vector<int>& cu_seqlens_padded,
                                     std::vector<int>& padded_to_batch)
{
    cu_seqlens.resize(bs + 1);
    cu_seqlens_padded.resize(bs + 1);
    cu_seqlens[0] = cu_seqlens_padded[0] = 0;
    for(int b = 0; b < bs; b++)
    {
        cu_seqlens[b + 1]        = cu_seqlens[b] + seq_len;
        cu_seqlens_padded[b + 1] = cu_seqlens_padded[b] + seq_len;
    }
    int total = cu_seqlens_padded[bs];
    padded_to_batch.resize(total);
    for(int b = 0; b < bs; b++)
        for(int i = cu_seqlens_padded[b]; i < cu_seqlens_padded[b + 1]; i++)
            padded_to_batch[i] = b;
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

struct BenchResult {
    double min_ms;
    double median_ms;
    double mean_ms;
    double q1_ms;
    double q3_ms;
    double tflops;
};

static BenchResult compute_stats(std::vector<double>& timings, double total_flops)
{
    std::sort(timings.begin(), timings.end());
    int n = timings.size();
    BenchResult r;
    r.min_ms    = timings[0];
    r.median_ms = timings[n / 2];
    r.q1_ms     = timings[n / 4];
    r.q3_ms     = timings[3 * n / 4];
    double sum  = 0;
    for(auto t : timings) sum += t;
    r.mean_ms = sum / n;
    r.tflops  = (total_flops / 1e12) / (r.min_ms / 1000.0);
    return r;
}

static void print_csv_row(const char* mode, int bs, int sq, int skv,
                           int nheads, int hdim, const char* kernel,
                           const BenchResult& r)
{
    std::cout << mode << ",bshd,bfloat16," << bs << ","
              << sq << "," << skv << ","
              << nheads << "," << hdim << ",1," << kernel << ","
              << std::fixed << std::setprecision(3)
              << r.min_ms << "," << r.median_ms << ","
              << r.mean_ms << "," << r.q1_ms << "," << r.q3_ms << ","
              << r.tflops << std::endl;
}

// ---------------------------------------------------------------------------
// Forward benchmark
// ---------------------------------------------------------------------------

template <typename Launcher, typename DataType, typename Config>
BenchResult run_fwd_bench(int bs, int sq, int skv, int warmup, int niters)
{
    constexpr int nh   = Config::head_num;
    constexpr int hd   = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<int> csq, csqp, q2b, cskv, cskvp;
    build_cu_seqlens_uniform(bs, sq, csq, csqp, q2b);
    cskv.resize(bs + 1); cskvp.resize(bs + 1);
    cskv[0] = cskvp[0] = 0;
    for(int b = 0; b < bs; b++) {
        cskv[b+1]  = cskv[b]  + skv;
        cskvp[b+1] = cskvp[b] + skv;
    }
    int tot_q  = csqp[bs];
    int tot_kv = cskvp[bs];

    size_t sQ = (size_t)tot_q  * nh * hd;
    size_t sK = (size_t)tot_kv * nh * hd;
    size_t sO = sQ;

    std::vector<DataType> hQ(sQ), hK(sK), hV(sK);
    for(size_t i = 0; i < sQ; i++) hQ[i] = DataType(dis(gen));
    for(size_t i = 0; i < sK; i++) hK[i] = DataType(dis(gen));
    for(size_t i = 0; i < sK; i++) hV[i] = DataType(dis(gen));

    using FwdAuxEl = typename Launcher::fwd_aux_buffer_scalar;
    FwdAuxEl* dW;
    DataType *dQ, *dK, *dV, *dO;
    int *d_csq, *d_csqp, *d_cskv, *d_cskvp, *d_q2b;
    HIP_CHECK(hipMalloc(&dQ, sQ * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dK, sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dV, sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dO, sO * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_csq,  (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csqp, (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cskv, (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cskvp,(bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_q2b, tot_q * sizeof(int)));
    size_t ws = Launcher::calc_workspace_size(tot_q);
    HIP_CHECK(hipMalloc(&dW, ws > 0 ? ws : sizeof(FwdAuxEl)));

    HIP_CHECK(hipMemcpy(dQ, hQ.data(), sQ*sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, hK.data(), sK*sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, hV.data(), sK*sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csq,  csq.data(),  (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csqp, csqp.data(), (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cskv, cskv.data(), (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cskvp,cskvp.data(),(bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q2b, q2b.data(), tot_q*sizeof(int), hipMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)hd);
    auto launch = [&]() {
        Launcher::run_attn_fwd_kernel(dQ, dK, dV,
            static_cast<const DataType*>(nullptr), 0.0f, scale, dO, dW,
            d_csq, d_csqp, d_cskv, d_cskvp, d_q2b, tot_q, bs);
    };

    for(int i = 0; i < warmup; i++) launch();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> timings(niters);
    for(int i = 0; i < niters; i++) {
        hipEvent_t t0, t1;
        HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
        launch();
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        timings[i] = ms;
        HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    }

    double flops = 4.0 * sq * skv * hd * bs * nh;
    auto res = compute_stats(timings, flops);

    HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dK)); HIP_CHECK(hipFree(dV));
    HIP_CHECK(hipFree(dO)); HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(d_csq)); HIP_CHECK(hipFree(d_csqp));
    HIP_CHECK(hipFree(d_cskv)); HIP_CHECK(hipFree(d_cskvp));
    HIP_CHECK(hipFree(d_q2b));
    return res;
}

// ---------------------------------------------------------------------------
// Backward benchmark
// ---------------------------------------------------------------------------

template <typename Launcher, typename DataType, typename Config>
BenchResult run_bwd_bench(int bs, int sq, int skv, int warmup, int niters)
{
    using FwdLauncher = AttnForwardMfma16x16KernelLauncher<DataType, Config>;

    constexpr int nh  = Config::head_num;
    constexpr int hd  = Config::head_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<int> csq, csqp, q2b, cskv, cskvp;
    build_cu_seqlens_uniform(bs, sq, csq, csqp, q2b);
    cskv.resize(bs + 1); cskvp.resize(bs + 1);
    cskv[0] = cskvp[0] = 0;
    for(int b = 0; b < bs; b++) {
        cskv[b+1]  = cskv[b]  + skv;
        cskvp[b+1] = cskvp[b] + skv;
    }
    int tot_q  = csqp[bs];
    int tot_kv = cskvp[bs];

    size_t sQ  = (size_t)tot_q  * nh * hd;
    size_t sK  = (size_t)tot_kv * nh * hd;

    std::vector<DataType> hQ(sQ), hK(sK), hV(sK), hGO(sQ);
    for(size_t i = 0; i < sQ; i++) hQ[i]  = DataType(dis(gen));
    for(size_t i = 0; i < sK; i++) hK[i]  = DataType(dis(gen));
    for(size_t i = 0; i < sK; i++) hV[i]  = DataType(dis(gen));
    for(size_t i = 0; i < sQ; i++) hGO[i] = DataType(dis(gen));

    float scale = 1.0f / std::sqrt((float)hd);

    DataType *dQ, *dK, *dV, *dO, *dGO, *dGQ, *dGK, *dGV;
    float* d_lse;
    int *d_csq, *d_csqp, *d_cskv, *d_cskvp;
    int* d_q2b;
    HIP_CHECK(hipMalloc(&dQ,  sQ * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dK,  sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dV,  sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dO,  sQ * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dGO, sQ * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_lse,
                        FwdLauncher::calc_workspace_size(tot_q) > 0
                            ? FwdLauncher::calc_workspace_size(tot_q)
                            : sizeof(float)));
    HIP_CHECK(hipMalloc(&dGQ, sQ * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dGK, sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&dGV, sK * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_csq,  (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csqp, (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cskv, (bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cskvp,(bs+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_q2b, tot_q * sizeof(int)));

    HIP_CHECK(hipMemcpy(dQ,  hQ.data(),  sQ *sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK,  hK.data(),  sK *sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV,  hV.data(),  sK *sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dGO, hGO.data(), sQ *sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csq,  csq.data(),  (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csqp, csqp.data(), (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cskv, cskv.data(), (bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cskvp,cskvp.data(),(bs+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q2b, q2b.data(), tot_q * sizeof(int), hipMemcpyHostToDevice));

    FwdLauncher::run_attn_fwd_kernel(dQ, dK, dV, static_cast<const DataType*>(nullptr), 0.0f,
                                     scale, dO, d_lse,
                                     d_csq, d_csqp, d_cskv, d_cskvp, d_q2b, tot_q, bs);
    HIP_CHECK(hipDeviceSynchronize());

    auto launch = [&]() {
        HIP_CHECK(hipMemset(dGQ, 0, sQ * sizeof(DataType)));
        HIP_CHECK(hipMemset(dGK, 0, sK * sizeof(DataType)));
        HIP_CHECK(hipMemset(dGV, 0, sK * sizeof(DataType)));
        Launcher::run_attn_bwd_kernel(dQ, dK, dV, dGO, d_lse,
            dGQ, dGK, dGV, scale,
            d_csq, d_csqp, d_cskv, d_cskvp, bs);
    };

    for(int i = 0; i < warmup; i++) launch();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> timings(niters);
    for(int i = 0; i < niters; i++) {
        hipEvent_t t0, t1;
        HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
        launch();
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        timings[i] = ms;
        HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    }

    // Bwd FLOPS: ~8 * sq * skv * hd per batch per head (2x QK^T grad + 2x PV grad)
    double flops = 8.0 * sq * skv * hd * bs * nh;
    auto res = compute_stats(timings, flops);

    HIP_CHECK(hipFree(dQ));  HIP_CHECK(hipFree(dK));  HIP_CHECK(hipFree(dV));
    HIP_CHECK(hipFree(dO)); HIP_CHECK(hipFree(dGO)); HIP_CHECK(hipFree(d_lse));
    HIP_CHECK(hipFree(dGQ)); HIP_CHECK(hipFree(dGK)); HIP_CHECK(hipFree(dGV));
    HIP_CHECK(hipFree(d_csq)); HIP_CHECK(hipFree(d_csqp));
    HIP_CHECK(hipFree(d_cskv)); HIP_CHECK(hipFree(d_cskvp));
    HIP_CHECK(hipFree(d_q2b));
    return res;
}

// ---------------------------------------------------------------------------
// Recursive template to sweep seqlen from SEQ to MAX_SEQ
// ---------------------------------------------------------------------------

template <int SEQ, int MAX_SEQ>
struct SeqSweep
{
    template <typename DataType, int BS, int HEAD_NUM, int HEAD_DIM>
    static void run(int warmup, int iters)
    {
        // --- Forward: 4x4x4 (only for SEQ <= 4) ---
        if constexpr(SEQ <= 4)
        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, SEQ>;
            using L = AttnForwardMfmaKernelLauncher<DataType, Cfg>;
            auto r = run_fwd_bench<L, DataType, Cfg>(BS, SEQ, SEQ, warmup, iters);
            print_csv_row("fwd", BS, SEQ, SEQ, HEAD_NUM, HEAD_DIM, "mfma_4x4", r);
        }

        // --- Forward: 16x16x16 ---
        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, SEQ>;
            using L = AttnForwardMfma16x16KernelLauncher<DataType, Cfg>;
            auto r = run_fwd_bench<L, DataType, Cfg>(BS, SEQ, SEQ, warmup, iters);
            print_csv_row("fwd", BS, SEQ, SEQ, HEAD_NUM, HEAD_DIM, "mfma_16x16", r);
        }

        // --- Backward: 16x16x16 ---
        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, SEQ>;
            using L = AttnBackwardMfma16x16KernelLauncher<DataType, Cfg>;
            auto r = run_bwd_bench<L, DataType, Cfg>(BS, SEQ, SEQ, warmup, iters);
            print_csv_row("bwd", BS, SEQ, SEQ, HEAD_NUM, HEAD_DIM, "mfma_16x16", r);
        }

        SeqSweep<SEQ + 1, MAX_SEQ>::template run<DataType, BS, HEAD_NUM, HEAD_DIM>(warmup, iters);
    }
};

template <int MAX_SEQ>
struct SeqSweep<MAX_SEQ, MAX_SEQ>
{
    template <typename DataType, int BS, int HEAD_NUM, int HEAD_DIM>
    static void run(int warmup, int iters)
    {
        if constexpr(MAX_SEQ <= 4)
        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, MAX_SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, MAX_SEQ>;
            using L = AttnForwardMfmaKernelLauncher<DataType, Cfg>;
            auto r = run_fwd_bench<L, DataType, Cfg>(BS, MAX_SEQ, MAX_SEQ, warmup, iters);
            print_csv_row("fwd", BS, MAX_SEQ, MAX_SEQ, HEAD_NUM, HEAD_DIM, "mfma_4x4", r);
        }

        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, MAX_SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, MAX_SEQ>;
            using L = AttnForwardMfma16x16KernelLauncher<DataType, Cfg>;
            auto r = run_fwd_bench<L, DataType, Cfg>(BS, MAX_SEQ, MAX_SEQ, warmup, iters);
            print_csv_row("fwd", BS, MAX_SEQ, MAX_SEQ, HEAD_NUM, HEAD_DIM, "mfma_16x16", r);
        }

        {
            using Cfg = FmhaKernelConfig<HEAD_NUM, MAX_SEQ, HEAD_DIM, 256, false,
                                         CausalMaskType::DISABLE, MAX_SEQ>;
            using L = AttnBackwardMfma16x16KernelLauncher<DataType, Cfg>;
            auto r = run_bwd_bench<L, DataType, Cfg>(BS, MAX_SEQ, MAX_SEQ, warmup, iters);
            print_csv_row("bwd", BS, MAX_SEQ, MAX_SEQ, HEAD_NUM, HEAD_DIM, "mfma_16x16", r);
        }
    }
};

// ---------------------------------------------------------------------------
// Cross-attention sweep: sq=1, skv in [MIN_SKV, MAX_SKV] (MFMA 16x16 only)
// ---------------------------------------------------------------------------

template <int SKV, int MAX_SKV>
struct CrossSeqSweep
{
    template <typename DataType, int BS, int HEAD_NUM, int HEAD_DIM>
    static void run(int warmup, int iters)
    {
        using Cfg = FmhaKernelConfig<HEAD_NUM, 16, HEAD_DIM, 256, false,
                                     CausalMaskType::DISABLE, 1>;
        using FwdL = AttnForwardMfma16x16KernelLauncher<DataType, Cfg>;
        using BwdL = AttnBackwardMfma16x16KernelLauncher<DataType, Cfg>;

        auto rf = run_fwd_bench<FwdL, DataType, Cfg>(BS, 1, SKV, warmup, iters);
        print_csv_row("fwd", BS, 1, SKV, HEAD_NUM, HEAD_DIM, "mfma_16x16", rf);
        auto rb = run_bwd_bench<BwdL, DataType, Cfg>(BS, 1, SKV, warmup, iters);
        print_csv_row("bwd", BS, 1, SKV, HEAD_NUM, HEAD_DIM, "mfma_16x16", rb);

        CrossSeqSweep<SKV + 1, MAX_SKV>::template run<DataType, BS, HEAD_NUM, HEAD_DIM>(
            warmup, iters);
    }
};

template <int MAX_SKV>
struct CrossSeqSweep<MAX_SKV + 1, MAX_SKV>
{
    template <typename DataType, int BS, int HEAD_NUM, int HEAD_DIM>
    static void run(int, int)
    {
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    std::cout << "mode,layout,dtype,batch_size,seqlen_q,seqlen_kv,nheads,dim,"
              << "gqa_ratio,kernel,"
              << "min_steptime_ms,median_steptime_ms,mean_steptime_ms,"
              << "q1_steptime_ms,q3_steptime_ms,tflops"
              << std::endl;

    constexpr int warmup = 5;
    constexpr int niters = 20;

    SeqSweep<1, 17>::run<hip_bfloat16, 2048, 32, 128>(warmup, niters);
    CrossSeqSweep<2, 16>::run<hip_bfloat16, 2048, 32, 128>(warmup, niters);

    return 0;
}
