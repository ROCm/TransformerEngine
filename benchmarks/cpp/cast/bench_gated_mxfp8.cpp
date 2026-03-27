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

#include <transformer_engine/cast_hip.h>
#include <transformer_engine/activation_hip.h>
#include <transformer_engine/transformer_engine_hip.h>

using namespace te_bench;
using namespace transformer_engine;
using fp8_e4m3 = test::fp8e4m3;

// SwiGLU shapes from LLaMA (8B, 70B, 405B) and Qwen (7B, 72B)
#define COMMON_SHAPES    \
  ->Args({1024, 14336})  \
  ->Args({1024, 18944})  \
  ->Args({1024, 28672})  \
  ->Args({2048, 14336})  \
  ->Args({2048, 28672})  \
  ->Args({2048, 29568})  \
  ->Args({4096, 14336})  \
  ->Args({4096, 28672})  \
  ->Args({4096, 53248})  \
  ->Args({8192, 14336})  \
  ->Args({8192, 28672})  \
  ->Args({8192, 29568})  \
  ->Args({8192, 53248})  \
  ->Args({16384, 28672}) \
  ->Args({16384, 53248}) \
  ->Args({32768, 28672}) \
  ->Args({32768, 53248})

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X>
static void BM_GatedMXFP8_Forward(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  const size_t input_cols  = cols * 2;
  const size_t output_cols = cols;

  const size_t scale_cols_row = USE_ROWWISE ? (output_cols + 31) / 32 : 0;
  const size_t scale_rows_col = USE_COLWISE ? (rows + 31) / 32 : 0;
  const size_t scale_cols_col = USE_COLWISE ? output_cols : 0;

  std::vector<size_t> input_shape  = {rows, input_cols};
  std::vector<size_t> output_shape = {rows, output_cols};

  DType itype = std::is_same_v<IType, __half> ? DType::kFloat16 :
                (std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);
  DType otype = std::is_same_v<OType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;

  test::Tensor &input_tensor  = TensorCache::get_or_create("input", input_shape, itype, true, false,
                                                           NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &output_tensor = TensorCache::get_or_create("output", output_shape, otype, USE_ROWWISE, USE_COLWISE,
                                                            NVTE_MXFP8_1D_SCALING, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_swiglu(input_tensor.data(), output_tensor.data(), stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const size_t bytes_write_data   = rows * output_cols * sizeof(OType) *
                                   ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  const size_t bytes_write_scales = (USE_ROWWISE ? rows * scale_cols_row : 0) +
                                     (USE_COLWISE ? scale_rows_col * scale_cols_col : 0);

  const size_t bytes_read  = rows * cols * sizeof(IType) * 2;
  const size_t total_bytes = bytes_read + bytes_write_data + bytes_write_scales;

  set_bytes_processed(state, total_bytes);

  HIP_CHECK(hipStreamDestroy(stream));
}

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X>
static void BM_GatedMXFP8_Backward(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  const size_t input_cols  = cols * 2;
  const size_t output_cols = cols * 2;

  const size_t scale_cols_row = USE_ROWWISE ? (output_cols + 31) / 32 : 0;
  const size_t scale_rows_col = USE_COLWISE ? (rows + 31) / 32 : 0;
  const size_t scale_cols_col = USE_COLWISE ? output_cols : 0;

  std::vector<size_t> grad_shape   = {rows, cols};
  std::vector<size_t> input_shape  = {rows, input_cols};
  std::vector<size_t> output_shape = {rows, output_cols};

  DType itype = std::is_same_v<IType, __half> ? DType::kFloat16 :
                (std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);
  DType otype = std::is_same_v<OType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;

  test::Tensor &grad_tensor   = TensorCache::get_or_create("grad", grad_shape, itype, true, false,
                                                          NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &input_tensor  = TensorCache::get_or_create("input", input_shape, itype, true, false,
                                                           NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &output_tensor = TensorCache::get_or_create("output", output_shape, otype, USE_ROWWISE, USE_COLWISE,
                                                            NVTE_MXFP8_1D_SCALING, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_dswiglu(grad_tensor.data(), input_tensor.data(), output_tensor.data(), stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const size_t bytes_write_data   = rows * output_cols * sizeof(OType) *
                                   ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  const size_t bytes_write_scales = (USE_ROWWISE ? rows * scale_cols_row : 0) +
                                     (USE_COLWISE ? scale_rows_col * scale_cols_col : 0);

  const size_t bytes_read  = rows * cols * sizeof(IType) * 3;
  const size_t total_bytes = bytes_read + bytes_write_data + bytes_write_scales;

  set_bytes_processed(state, total_bytes);

  HIP_CHECK(hipStreamDestroy(stream));
}

#define REGISTER_GATED_ALL_CONFIGS(ITYPE, OTYPE, INAME, ONAME) \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Forward, ITYPE, OTYPE, 1, 32) \
    ->Name("BM_GatedMXFP8_Forward/" INAME "_" ONAME "/rowwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Forward, ITYPE, OTYPE, 32, 1) \
    ->Name("BM_GatedMXFP8_Forward/" INAME "_" ONAME "/colwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Forward, ITYPE, OTYPE, 32, 32) \
    ->Name("BM_GatedMXFP8_Forward/" INAME "_" ONAME "/both") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Backward, ITYPE, OTYPE, 1, 32) \
    ->Name("BM_GatedMXFP8_Backward/" INAME "_" ONAME "/rowwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Backward, ITYPE, OTYPE, 32, 1) \
    ->Name("BM_GatedMXFP8_Backward/" INAME "_" ONAME "/colwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GatedMXFP8_Backward, ITYPE, OTYPE, 32, 32) \
    ->Name("BM_GatedMXFP8_Backward/" INAME "_" ONAME "/both") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime();

REGISTER_GATED_ALL_CONFIGS(__half, fp8_e4m3, "FP16", "E4M3")
REGISTER_GATED_ALL_CONFIGS(hip_bfloat16, fp8_e4m3, "BF16", "E4M3")
REGISTER_GATED_ALL_CONFIGS(float, fp8_e4m3, "FP32", "E4M3")

BENCHMARK_MAIN();
