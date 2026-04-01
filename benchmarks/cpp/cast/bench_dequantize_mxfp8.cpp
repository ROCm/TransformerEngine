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
#include "transformer_engine/transformer_engine_hip.h"

using namespace te_bench;
using namespace transformer_engine;
using fp8_e4m3 = test::fp8e4m3;

// Tensor shapes from LLaMA (8B, 70B, 405B) and Qwen (7B, 72B)
#define COMMON_SHAPES   \
  ->Args({1024, 3584})  \
  ->Args({1024, 4096})  \
  ->Args({1024, 8192})  \
  ->Args({1024, 14336}) \
  ->Args({2048, 4096})  \
  ->Args({2048, 8192})  \
  ->Args({2048, 14336}) \
  ->Args({2048, 28672}) \
  ->Args({4096, 4096})  \
  ->Args({4096, 8192})  \
  ->Args({4096, 16384}) \
  ->Args({4096, 28672}) \
  ->Args({8192, 8192})  \
  ->Args({8192, 16384}) \
  ->Args({8192, 28672}) \
  ->Args({8192, 53248}) \
  ->Args({16384, 8192}) \
  ->Args({16384, 16384})\
  ->Args({32768, 8192})

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X>
static void BM_DequantizeMXFP8(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  const size_t scale_cols_row = USE_ROWWISE ? (cols + 31) / 32 : 0;
  const size_t scale_rows_col = USE_COLWISE ? (rows + 31) / 32 : 0;
  const size_t scale_cols_col = USE_COLWISE ? cols : 0;

  std::vector<size_t> shape = {rows, cols};
  DType itype = std::is_same_v<IType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;
  DType otype = std::is_same_v<OType, __half> ? DType::kFloat16 :
                (std::is_same_v<OType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);

  test::Tensor &input_tensor  = TensorCache::get_or_create("input", shape, itype, USE_ROWWISE, USE_COLWISE,
                                                           NVTE_MXFP8_1D_SCALING, false);
  test::Tensor &output_tensor = TensorCache::get_or_create("output", shape, otype, true, false,
                                                            NVTE_DELAYED_TENSOR_SCALING, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  DeviceBuffer<float> temp_fp32(rows * cols);
  fill_random_uniform_gpu(temp_fp32.get(), rows * cols, -2.0f, 1.0f, stream);

  void *input_data_ptr = USE_ROWWISE ? input_tensor.rowwise_dptr() : input_tensor.columnwise_dptr();
  size_t threads = 256;
  size_t blocks = (rows * cols + threads - 1) / threads;
  cast_fp32_kernel<<<blocks, threads, 0, stream>>>(temp_fp32.get(), static_cast<IType*>(input_data_ptr), rows * cols);

  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_dequantize(input_tensor.data(), output_tensor.data(), stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const size_t bytes_read_data = rows * cols * sizeof(IType) *
                                  ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  // Scales are single byte, E8M0 type
  const size_t bytes_read_scales = (USE_ROWWISE ? rows * scale_cols_row : 0) +
                                    (USE_COLWISE ? scale_rows_col * scale_cols_col : 0);
  const size_t bytes_write = rows * cols * sizeof(OType);
  const size_t total_bytes = bytes_read_data + bytes_read_scales + bytes_write;

  set_bytes_processed(state, total_bytes);

  HIP_CHECK(hipStreamDestroy(stream));
}

#define REGISTER_DEQUANTIZE_ALL_CONFIGS(ITYPE, OTYPE, INAME, ONAME) \
  BENCHMARK_TEMPLATE(BM_DequantizeMXFP8, ITYPE, OTYPE, 1, 32) \
    ->Name("BM_DequantizeMXFP8/" INAME "_" ONAME "/rowwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_DequantizeMXFP8, ITYPE, OTYPE, 32, 1) \
    ->Name("BM_DequantizeMXFP8/" INAME "_" ONAME "/colwise") \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime();

REGISTER_DEQUANTIZE_ALL_CONFIGS(fp8_e4m3, __half, "E4M3", "FP16")
REGISTER_DEQUANTIZE_ALL_CONFIGS(fp8_e4m3, hip_bfloat16, "E4M3", "BF16")
REGISTER_DEQUANTIZE_ALL_CONFIGS(fp8_e4m3, float, "E4M3", "FP32")

BENCHMARK_MAIN();
