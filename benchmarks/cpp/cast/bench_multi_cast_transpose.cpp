/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "amd_detail/hip_float8.h"

#include "benchmark_utils.h"

#include "transformer_engine/padding_hip.h"
#include "transformer_engine/transpose_hip.h"
#include "transformer_engine/transformer_engine_hip.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace te_bench;
using namespace transformer_engine;
using fp8_e4m3 = test::fp8e4m3;

// MoE shapes from Qwen3-235B, DeepSeek-V3, and DeepSeek-V4
// Args: {total_tokens, cols, num_experts, top_k, routing_mode}
#define MOE_BALANCED                     \
  ->Args({4096,  4096, 128, 8, 0})       \
  ->Args({8192,  4096, 128, 8, 0})       \
  ->Args({16384, 4096, 128, 8, 0})       \
  ->Args({4096,  1536, 128, 8, 0})       \
  ->Args({8192,  1536, 128, 8, 0})       \
  ->Args({16384, 1536, 128, 8, 0})       \
  ->Args({4096,  3072, 128, 8, 0})       \
  ->Args({8192,  3072, 128, 8, 0})       \
  ->Args({16384, 3072, 128, 8, 0})       \
  ->Args({4096,  7168, 256, 8, 0})       \
  ->Args({8192,  7168, 256, 8, 0})       \
  ->Args({16384, 7168, 256, 8, 0})       \
  ->Args({4096,  2048, 256, 8, 0})       \
  ->Args({8192,  2048, 256, 8, 0})       \
  ->Args({16384, 2048, 256, 8, 0})       \
  ->Args({4096,  4096, 256, 8, 0})       \
  ->Args({8192,  4096, 256, 8, 0})       \
  ->Args({16384, 4096, 256, 8, 0})       \
  ->Args({4096,  7168, 384, 6, 0})       \
  ->Args({8192,  7168, 384, 6, 0})       \
  ->Args({16384, 7168, 384, 6, 0})       \
  ->Args({4096,  3072, 384, 6, 0})       \
  ->Args({8192,  3072, 384, 6, 0})       \
  ->Args({16384, 3072, 384, 6, 0})

#define MOE_SKEWED                       \
  ->Args({4096,  4096, 128, 8, 1})       \
  ->Args({8192,  4096, 128, 8, 1})       \
  ->Args({16384, 4096, 128, 8, 1})       \
  ->Args({4096,  1536, 128, 8, 1})       \
  ->Args({8192,  1536, 128, 8, 1})       \
  ->Args({16384, 1536, 128, 8, 1})       \
  ->Args({4096,  3072, 128, 8, 1})       \
  ->Args({8192,  3072, 128, 8, 1})       \
  ->Args({16384, 3072, 128, 8, 1})       \
  ->Args({4096,  7168, 256, 8, 1})       \
  ->Args({8192,  7168, 256, 8, 1})       \
  ->Args({16384, 7168, 256, 8, 1})       \
  ->Args({4096,  2048, 256, 8, 1})       \
  ->Args({8192,  2048, 256, 8, 1})       \
  ->Args({16384, 2048, 256, 8, 1})       \
  ->Args({4096,  4096, 256, 8, 1})       \
  ->Args({8192,  4096, 256, 8, 1})       \
  ->Args({16384, 4096, 256, 8, 1})       \
  ->Args({4096,  7168, 384, 6, 1})       \
  ->Args({8192,  7168, 384, 6, 1})       \
  ->Args({16384, 7168, 384, 6, 1})       \
  ->Args({4096,  3072, 384, 6, 1})       \
  ->Args({8192,  3072, 384, 6, 1})       \
  ->Args({16384, 3072, 384, 6, 1})

namespace {

static const uint64_t kRunSeed = std::random_device{}();
static constexpr size_t kPadMultiple = 16;

static uint64_t derive_seed(size_t a, size_t b, size_t c, size_t d, size_t e) {
  uint64_t h = kRunSeed;
  h ^= a; h *= 1099511628211ULL;
  h ^= b; h *= 1099511628211ULL;
  h ^= c; h *= 1099511628211ULL;
  h ^= d; h *= 1099511628211ULL;
  h ^= e; h *= 1099511628211ULL;
  return h;
}

static std::vector<size_t> simulate_topk_balanced(
    size_t total_tokens, size_t num_experts, size_t top_k, uint64_t seed)
{
  std::vector<size_t> counts(num_experts, 0);
  std::mt19937_64 gen(seed);

  std::vector<size_t> experts(num_experts);
  std::iota(experts.begin(), experts.end(), 0);

  std::vector<std::uniform_int_distribution<size_t>> dists;
  for (size_t k = 0; k < top_k; k++) {
    dists.push_back(std::uniform_int_distribution<size_t>(k, num_experts - 1));
  }

  for (size_t t = 0; t < total_tokens; t++) {
    for (size_t k = 0; k < top_k; k++) {
      std::swap(experts[k], experts[dists[k](gen)]);
      counts[experts[k]]++;
    }
  }
  return counts;
}

static std::vector<size_t> simulate_topk_skewed(
    size_t total_tokens, size_t num_experts, size_t top_k, uint64_t seed)
{
  std::vector<size_t> counts(num_experts, 0);
  std::mt19937_64 gen(seed);

  std::vector<double> weights(num_experts);
  for (size_t i = 0; i < num_experts; i++)
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 0.7);

  std::shuffle(weights.begin(), weights.end(), gen);
  std::discrete_distribution<size_t> wdist(weights.begin(), weights.end());

  std::vector<bool> used(num_experts, false);
  std::vector<size_t> used_list;
  used_list.reserve(top_k);

  for (size_t t = 0; t < total_tokens; t++) {
    used_list.clear();
    for (size_t k = 0; k < top_k; k++) {
      size_t e;
      do { e = wdist(gen); } while (used[e]);
      used[e] = true;
      used_list.push_back(e);
      counts[e]++;
    }
    for (size_t e : used_list) used[e] = false;
  }
  return counts;
}

template <typename IType, int EP = 1>
static void BM_MultiCastTranspose(benchmark::State &state) {
  const size_t total_tokens = state.range(0) / EP;
  const size_t cols         = state.range(1);
  const size_t num_experts  = state.range(2) / EP;
  const size_t top_k        = state.range(3);
  const size_t routing_mode = state.range(4);

  uint64_t seed = derive_seed(total_tokens, cols, num_experts, top_k, routing_mode);

  auto counts = (routing_mode == 0)
      ? simulate_topk_balanced(total_tokens, num_experts, top_k, seed)
      : simulate_topk_skewed(total_tokens, num_experts, top_k, seed);

  size_t min_tok = *std::min_element(counts.begin(), counts.end());
  size_t max_tok = *std::max_element(counts.begin(), counts.end());
  size_t sum_tok = std::accumulate(counts.begin(), counts.end(), size_t(0));

  DType itype = std::is_same_v<IType, float>       ? DType::kFloat32 :
                std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 :
                                                       DType::kFloat16;

  std::string pfx = "mct_" + std::to_string(total_tokens) + "_"
                   + std::to_string(cols) + "_" + std::to_string(num_experts)
                   + "_" + std::to_string(routing_mode);

  std::vector<NVTETensor> nvte_in(num_experts), nvte_out(num_experts);

  for (size_t e = 0; e < num_experts; e++) {
    size_t rows = ((std::max(counts[e], size_t(1)) + kPadMultiple - 1)
                   / kPadMultiple) * kPadMultiple;
    std::string in_name  = pfx + "_in_"  + std::to_string(e);
    std::string out_name = pfx + "_out_" + std::to_string(e);

    auto &input = TensorCache::get_or_create(
        in_name, {rows, cols}, itype,
        true, false, NVTE_DELAYED_TENSOR_SCALING, true);

    auto &output = TensorCache::get_or_create(
        out_name, {rows, cols}, DType::kFloat8E4M3,
        true, true, NVTE_DELAYED_TENSOR_SCALING, false);

    output.set_scale(1.0f);

    nvte_in[e]  = input.data();
    nvte_out[e] = output.data();
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_multi_cast_transpose(num_experts, nvte_in.data(), nvte_out.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t total_bytes = 0;
  for (size_t e = 0; e < num_experts; e++) {
    size_t rows = ((std::max(counts[e], size_t(1)) + kPadMultiple - 1)
                   / kPadMultiple) * kPadMultiple;
    total_bytes += rows * cols * sizeof(IType);
    total_bytes += rows * cols * sizeof(fp8_e4m3) * 2;
  }
  set_bytes_processed(state, total_bytes);

  state.counters["experts"]  = num_experts;
  state.counters["cols"]     = cols;
  state.counters["avg_tok"]  = static_cast<double>(sum_tok) / num_experts;
  state.counters["min_tok"]  = min_tok;
  state.counters["max_tok"]  = max_tok;

  HIP_CHECK(hipStreamDestroy(stream));
}

// Unfused baseline: separate padding kernel + cast_transpose
template <typename IType, int EP = 1>
static void BM_PaddingThenMCT(benchmark::State &state) {
  const size_t total_tokens = state.range(0) / EP;
  const size_t cols         = state.range(1);
  const size_t num_experts  = state.range(2) / EP;
  const size_t top_k        = state.range(3);
  const size_t routing_mode = state.range(4);

  uint64_t seed = derive_seed(total_tokens, cols, num_experts, top_k, routing_mode);

  auto counts = (routing_mode == 0)
      ? simulate_topk_balanced(total_tokens, num_experts, top_k, seed)
      : simulate_topk_skewed(total_tokens, num_experts, top_k, seed);

  size_t sum_tok = std::accumulate(counts.begin(), counts.end(), size_t(0));

  DType itype = std::is_same_v<IType, float>        ? DType::kFloat32  :
                std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 :
                                                      DType::kFloat16;

  std::string pfx = "pad_mct_" + std::to_string(total_tokens) + "_" + std::to_string(cols) + "_" 
                  + std::to_string(num_experts) + "_" + std::to_string(routing_mode);

  std::vector<NVTETensor> nvte_in(num_experts), nvte_pad_out(num_experts), 
      nvte_mct_in(num_experts), nvte_mct_out(num_experts);

  std::vector<int> padded_rows_list(num_experts);

  for (size_t e = 0; e < num_experts; e++) {
    size_t actual = std::max(counts[e], size_t(1));
    size_t padded = ((actual + kPadMultiple - 1) / kPadMultiple) * kPadMultiple;
    padded_rows_list[e] = static_cast<int>(padded);

    auto &input = TensorCache::get_or_create(pfx + "_in_" + std::to_string(e), {actual, cols}, itype, 
                                             true, false, NVTE_DELAYED_TENSOR_SCALING, true);

    auto &pad_out = TensorCache::get_or_create(pfx + "_pad_" + std::to_string(e), {padded, cols}, itype,
                                               true, false, NVTE_DELAYED_TENSOR_SCALING, false);

    auto &mct_out = TensorCache::get_or_create(pfx + "_out_" + std::to_string(e), {padded, cols}, DType::kFloat8E4M3,
                                               true, true, NVTE_DELAYED_TENSOR_SCALING, false);

    mct_out.set_scale(1.0f);

    nvte_in[e]      = input.data();
    nvte_pad_out[e] = pad_out.data();
    nvte_mct_in[e]  = pad_out.data();
    nvte_mct_out[e] = mct_out.data();
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_multi_padding(num_experts, nvte_in.data(), nvte_pad_out.data(), padded_rows_list.data(), stream);
    nvte_multi_cast_transpose(num_experts, nvte_mct_in.data(), nvte_mct_out.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t total_bytes = 0;
  for (size_t e = 0; e < num_experts; e++) {
    size_t actual = std::max(counts[e], size_t(1));
    size_t padded = padded_rows_list[e];
    total_bytes += actual * cols * sizeof(IType);
    total_bytes += padded * cols * sizeof(IType);
    total_bytes += padded * cols * sizeof(IType);
    total_bytes += padded * cols * sizeof(fp8_e4m3) * 2;
  }
  set_bytes_processed(state, total_bytes);

  state.counters["experts"] = num_experts;
  state.counters["cols"]    = cols;
  state.counters["avg_tok"] = static_cast<double>(sum_tok) / num_experts;

  HIP_CHECK(hipStreamDestroy(stream));
}

// Fused: single cast_transpose kernel with built-in padding
template <typename IType, int EP = 1>
static void BM_FusedPaddingMCT(benchmark::State &state) {
  const size_t total_tokens = state.range(0) / EP;
  const size_t cols         = state.range(1);
  const size_t num_experts  = state.range(2) / EP;
  const size_t top_k        = state.range(3);
  const size_t routing_mode = state.range(4);

  uint64_t seed = derive_seed(total_tokens, cols, num_experts, top_k, routing_mode);

  auto counts = (routing_mode == 0)
      ? simulate_topk_balanced(total_tokens, num_experts, top_k, seed)
      : simulate_topk_skewed(total_tokens, num_experts, top_k, seed);

  size_t sum_tok = std::accumulate(counts.begin(), counts.end(), size_t(0));

  DType itype = std::is_same_v<IType, float>        ? DType::kFloat32  :
                std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 :
                                                      DType::kFloat16;

  std::string pfx = "fused_mct_" + std::to_string(total_tokens) + "_" + std::to_string(cols) + "_" 
                  + std::to_string(num_experts) + "_" + std::to_string(routing_mode);

  std::vector<NVTETensor> nvte_in(num_experts), nvte_out(num_experts);
  std::vector<int> valid_rows_list(num_experts);

  for (size_t e = 0; e < num_experts; e++) {
    size_t actual = std::max(counts[e], size_t(1));
    size_t padded = ((actual + kPadMultiple - 1) / kPadMultiple) * kPadMultiple;
    valid_rows_list[e] = static_cast<int>(actual);

    auto &input = TensorCache::get_or_create(pfx + "_in_" + std::to_string(e), {actual, cols}, itype,
                                             true, false, NVTE_DELAYED_TENSOR_SCALING, true);

    auto &output = TensorCache::get_or_create(pfx + "_out_" + std::to_string(e), {padded, cols}, DType::kFloat8E4M3,
                                              true, true, NVTE_DELAYED_TENSOR_SCALING, false);

    output.set_scale(1.0f);

    nvte_in[e]  = input.data();
    nvte_out[e] = output.data();
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_multi_cast_transpose_with_padding(num_experts, nvte_in.data(), nvte_out.data(), valid_rows_list.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t total_bytes = 0;
  for (size_t e = 0; e < num_experts; e++) {
    size_t actual = std::max(counts[e], size_t(1));
    size_t padded = ((actual + kPadMultiple - 1) / kPadMultiple) * kPadMultiple;
    total_bytes += actual * cols * sizeof(IType);
    total_bytes += padded * cols * sizeof(fp8_e4m3) * 2;
  }
  set_bytes_processed(state, total_bytes);

  state.counters["experts"]  = num_experts;
  state.counters["cols"]     = cols;
  state.counters["avg_tok"]  = static_cast<double>(sum_tok) / num_experts;

  HIP_CHECK(hipStreamDestroy(stream));
}

}  // namespace

#define REGISTER_MCT(ITYPE, INAME)                                            \
  BENCHMARK_TEMPLATE(BM_MultiCastTranspose, ITYPE)                            \
    ->Name("BM_MultiCastTranspose/" INAME "_E4M3/moe")                        \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_MultiCastTranspose, ITYPE)                            \
    ->Name("BM_MultiCastTranspose/" INAME "_E4M3/moe_skewed")                 \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

#define REGISTER_PAD_MCT(ITYPE, INAME)                                        \
  BENCHMARK_TEMPLATE(BM_PaddingThenMCT, ITYPE)                                \
    ->Name("BM_PaddingThenMCT/" INAME "_E4M3/moe")                            \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_PaddingThenMCT, ITYPE)                                \
    ->Name("BM_PaddingThenMCT/" INAME "_E4M3/moe_skewed")                     \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_FusedPaddingMCT, ITYPE)                               \
    ->Name("BM_FusedPaddingMCT/" INAME "_E4M3/moe")                           \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_FusedPaddingMCT, ITYPE)                               \
    ->Name("BM_FusedPaddingMCT/" INAME "_E4M3/moe_skewed")                    \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

REGISTER_MCT(hip_bfloat16, "BF16")
REGISTER_PAD_MCT(hip_bfloat16, "BF16")

#ifdef NVTE_ROCM_EP8_BENCHMARKS
#define REGISTER_MCT_EP8(ITYPE, INAME)                                        \
  BENCHMARK_TEMPLATE(BM_MultiCastTranspose, ITYPE, 8)                         \
    ->Name("BM_MultiCastTranspose/" INAME "_E4M3/moe_ep8")                    \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_MultiCastTranspose, ITYPE, 8)                         \
    ->Name("BM_MultiCastTranspose/" INAME "_E4M3/moe_ep8_skewed")             \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

#define REGISTER_PAD_MCT_EP8(ITYPE, INAME)                                    \
  BENCHMARK_TEMPLATE(BM_PaddingThenMCT, ITYPE, 8)                             \
    ->Name("BM_PaddingThenMCT/" INAME "_E4M3/moe_ep8")                        \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_PaddingThenMCT, ITYPE, 8)                             \
    ->Name("BM_PaddingThenMCT/" INAME "_E4M3/moe_ep8_skewed")                 \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_FusedPaddingMCT, ITYPE, 8)                            \
    ->Name("BM_FusedPaddingMCT/" INAME "_E4M3/moe_ep8")                       \
    MOE_BALANCED                                                              \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_FusedPaddingMCT, ITYPE, 8)                            \
    ->Name("BM_FusedPaddingMCT/" INAME "_E4M3/moe_ep8_skewed")                \
    MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

REGISTER_MCT_EP8(hip_bfloat16, "BF16")
REGISTER_PAD_MCT_EP8(hip_bfloat16, "BF16")
#endif

BENCHMARK_MAIN();
