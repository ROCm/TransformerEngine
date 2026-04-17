/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "amd_detail/hip_float8.h"

#include "benchmark_utils.h"

#include "transformer_engine/cast_hip.h"
#include "transformer_engine/transpose_hip.h"
#include "transformer_engine/transformer_engine_hip.h"

#define NVTE_ROCM_EXTENDED_BENCHMARKS 1

using namespace te_bench;
using namespace transformer_engine;
using fp8_e4m3 = test::fp8e4m3;

#define GPT_OSS_COMMON_SHAPES      \
  ->Args({2880, 2880})             \
  ->Args({2880, 4096})             \
  ->Args({5120, 2880})             \
  ->Args({5760, 2880})             \
  ->Args({16384, 2880})            \
  ->Args({16384, 4096})            \
  ->Args({16384, 5120})

// GPT-OSS MoE per-expert shapes (hidden=2880, intermediate=5760)
#define GPT_OSS_MOE                \
  ->Args({64, 2880})               \
  ->Args({256, 2880})              \
  ->Args({320, 2880})              \
  ->Args({496, 2880})              \
  ->Args({1792, 2880})             \
  ->Args({64, 5760})               \
  ->Args({256, 5760})              \
  ->Args({320, 5760})              \
  ->Args({496, 5760})              \
  ->Args({1792, 5760})

// Tensor shapes from LLaMA (8B, 70B, 405B) and Qwen (7B, 72B)
#define COMMON_SHAPES              \
  ->Args({1024, 3584})             \
  ->Args({1024, 4096})             \
  ->Args({1024, 8192})             \
  ->Args({1024, 14336})            \
  ->Args({1024, 18944})            \
  ->Args({2048, 4096})             \
  ->Args({2048, 8192})             \
  ->Args({2048, 14336})            \
  ->Args({2048, 28672})            \
  ->Args({2048, 29568})            \
  ->Args({4096, 4096})             \
  ->Args({4096, 8192})             \
  ->Args({4096, 16384})            \
  ->Args({4096, 14336})            \
  ->Args({4096, 28672})            \
  ->Args({8192, 8192})             \
  ->Args({8192, 16384})            \
  ->Args({8192, 28672})            \
  ->Args({8192, 29568})            \
  ->Args({8192, 53248})            \
  ->Args({16384, 8192})            \
  ->Args({16384, 16384})           \
  ->Args({16384, 28672})           \
  ->Args({32768, 8192})            \
  ->Args({32768, 16384})

// Only used for specific benchmarks (older models, special cases, etc)
#define EXTENDED_SHAPES            \
  ->Args({2048, 12288})            \
  ->Args({256, 65536})             \
  ->Args({65536, 128})             \
  ->Args({1600, 1600})             \
  ->Args({1600, 6400})             \
  ->Args({4800, 1600})             \
  ->Args({56320 , 1600})           \
  ->Args({6400, 1600})             \
  ->Args({128256, 4096})           \
  ->Args({24576, 128256})          \
  ->Args({24576, 4096})            \
  ->Args({24576, 4096})            \
  ->Args({24576, 5120})            \
  ->Args({28672, 4096})            \
  ->Args({4096, 12288})            \
  ->Args({4096, 4096})             \
  ->Args({5120, 4096})             \
  ->Args({10240, 8192})            \
  ->Args({128256, 8192})           \
  ->Args({57344, 10240})           \
  ->Args({57344, 128256})          \
  ->Args({57344, 8192})            \
  ->Args({57344, 8192})            \
  ->Args({8192, 28672})            \
  ->Args({8192, 8192})             \
  ->Args({28672, 4096})            \
  ->Args({32000, 4096})            \
  ->Args({32768, 32000})           \
  ->Args({32768, 4096})            \
  ->Args({32768, 4096})            \
  ->Args({32768, 5120})            \
  ->Args({4096, 14336})            \
  ->Args({4096, 4096})             \
  ->Args({5120, 4096})

  



template <typename IType>
static void BM_CastOnly(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);
  std::vector<size_t> shape = {rows, cols};

  DType itype = std::is_same_v<IType, float>       ? DType::kFloat32 :
                std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 :
                                                      DType::kFloat16;

  test::Tensor &input  = TensorCache::get_or_create(
      "cast_input", shape, itype, true, false, NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &output = TensorCache::get_or_create(
      "cast_output", shape, DType::kFloat8E4M3, true, false, NVTE_DELAYED_TENSOR_SCALING, false);

  output.set_scale(1.0f);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();
  // Untimed call to trigger any RTC compilation before measurement
  nvte_quantize(input.data(), output.data(), stream);
  HIP_CHECK(hipDeviceSynchronize());

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_quantize(input.data(), output.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const size_t bytes_read  = rows * cols * sizeof(IType);
  const size_t bytes_write = rows * cols * sizeof(fp8_e4m3);
  set_bytes_processed(state, bytes_read + bytes_write);

  HIP_CHECK(hipStreamDestroy(stream));
}

template <typename IType>
static void BM_CastTranspose(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);
  std::vector<size_t> shape = {rows, cols};

  DType itype = std::is_same_v<IType, float>       ? DType::kFloat32 :
                std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 :
                                                      DType::kFloat16;

  test::Tensor &input  = TensorCache::get_or_create(
      "ct_input", shape, itype, true, false, NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &output = TensorCache::get_or_create(
      "ct_output", shape, DType::kFloat8E4M3, true, true, NVTE_DELAYED_TENSOR_SCALING, false);

  output.set_scale(1.0f);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();
  // Untimed call to trigger any RTC compilation before measurement
  nvte_quantize(input.data(), output.data(), stream);
  HIP_CHECK(hipDeviceSynchronize());

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_quantize(input.data(), output.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const size_t bytes_read  = rows * cols * sizeof(IType);
  const size_t bytes_write = rows * cols * sizeof(fp8_e4m3) * 2;
  set_bytes_processed(state, bytes_read + bytes_write);

  HIP_CHECK(hipStreamDestroy(stream));
}

#define REGISTER_CAST_ONLY(ITYPE, INAME)                                      \
  BENCHMARK_TEMPLATE(BM_CastOnly, ITYPE)                                      \
    ->Name("BM_CastOnly/" INAME "_E4M3/gpt_oss")                              \
    GPT_OSS_COMMON_SHAPES                                                     \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_CastOnly, ITYPE)                                      \
    ->Name("BM_CastOnly/" INAME "_E4M3/gpt_oss_moe")                          \
    GPT_OSS_MOE                                                               \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_CastOnly, ITYPE)                                      \
    ->Name("BM_CastOnly/" INAME "_E4M3/llm")                                  \
    COMMON_SHAPES                                                             \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

#define REGISTER_CAST_TRANSPOSE(ITYPE, INAME)                                 \
  BENCHMARK_TEMPLATE(BM_CastTranspose, ITYPE)                                 \
    ->Name("BM_CastTranspose/" INAME "_E4M3/gpt_oss")                         \
    GPT_OSS_COMMON_SHAPES                                                     \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_CastTranspose, ITYPE)                                 \
    ->Name("BM_CastTranspose/" INAME "_E4M3/gpt_oss_moe")                     \
    GPT_OSS_MOE                                                               \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();                                                        \
  BENCHMARK_TEMPLATE(BM_CastTranspose, ITYPE)                                 \
    ->Name("BM_CastTranspose/" INAME "_E4M3/llm")                             \
    COMMON_SHAPES                                                             \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

#ifdef NVTE_ROCM_EXTENDED_BENCHMARKS
#define REGISTER_EXTENDED_CAST_ONLY(ITYPE, INAME)                             \
  BENCHMARK_TEMPLATE(BM_CastOnly, ITYPE)                                      \
    ->Name("BM_CastOnlyExtended/" INAME "_E4M3/llm")                          \
    EXTENDED_SHAPES                                                           \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

#define REGISTER_EXTENDED_CAST_TRANSPOSE(ITYPE, INAME)                        \
  BENCHMARK_TEMPLATE(BM_CastTranspose, ITYPE)                                 \
    ->Name("BM_CastTransposeExtended/" INAME "_E4M3/llm")                     \
    EXTENDED_SHAPES                                                           \
    ->Unit(benchmark::kMicrosecond)                                           \
    ->UseManualTime();

REGISTER_EXTENDED_CAST_TRANSPOSE(float,        "FP32")
REGISTER_EXTENDED_CAST_TRANSPOSE(hip_bfloat16, "BF16")
#endif // #ifdef NVTE_ROCM_EXTENDED_BENCHMARKS

REGISTER_CAST_ONLY(float,        "FP32")
REGISTER_CAST_ONLY(hip_bfloat16, "BF16")

REGISTER_CAST_TRANSPOSE(float,        "FP32")
REGISTER_CAST_TRANSPOSE(hip_bfloat16, "BF16")

BENCHMARK_MAIN();
