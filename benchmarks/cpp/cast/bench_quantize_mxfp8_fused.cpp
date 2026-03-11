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

enum ProcessingMethod {
    CAST_ONLY,
    CAST_DBIAS,
    CAST_DBIAS_DACT,
    CAST_DACT,
    CAST_ACT
};

// Tensor shapes from LLaMA (8B, 70B, 405B) and Qwen (7B, 72B)
#define COMMON_SHAPES   \
  ->Args({1024, 3584})  \
  ->Args({1024, 4096})  \
  ->Args({1024, 8192})  \
  ->Args({1024, 14336}) \
  ->Args({1024, 18944}) \
  ->Args({2048, 4096})  \
  ->Args({2048, 8192})  \
  ->Args({2048, 14336}) \
  ->Args({2048, 28672}) \
  ->Args({2048, 29568}) \
  ->Args({4096, 4096})  \
  ->Args({4096, 8192})  \
  ->Args({4096, 16384}) \
  ->Args({4096, 14336}) \
  ->Args({4096, 28672}) \
  ->Args({8192, 8192})  \
  ->Args({8192, 16384}) \
  ->Args({8192, 28672}) \
  ->Args({8192, 29568}) \
  ->Args({8192, 53248}) \
  ->Args({16384, 8192}) \
  ->Args({16384, 16384})\
  ->Args({16384, 28672})\
  ->Args({32768, 8192}) \
  ->Args({32768, 16384})

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X,
          ProcessingMethod PROC_METHOD>
static void BM_QuantizeMXFP8_Fused(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(1);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  const size_t scale_cols_row = USE_ROWWISE ? (cols + 31) / 32 : 0;
  const size_t scale_rows_col = USE_COLWISE ? (rows + 31) / 32 : 0;
  const size_t scale_cols_col = USE_COLWISE ? cols : 0;

  std::vector<size_t> shape = {rows, cols};

  DType itype = std::is_same_v<IType, __half> ? DType::kFloat16 :
                (std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);
  DType otype = std::is_same_v<OType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;

  test::Tensor &input_tensor  = TensorCache::get_or_create("input", shape, itype, true, false,
                                                           NVTE_DELAYED_TENSOR_SCALING, true);
  test::Tensor &output_tensor = TensorCache::get_or_create("output", shape, otype, USE_ROWWISE, USE_COLWISE,
                                                            NVTE_MXFP8_1D_SCALING, false);

  test::Tensor *grad_tensor_ptr = nullptr, *dbias_tensor_ptr = nullptr,  *workspace_tensor_ptr = nullptr;

  if constexpr (PROC_METHOD == CAST_DBIAS || PROC_METHOD == CAST_DBIAS_DACT) {
    std::vector<size_t> bias_shape = {cols};
    dbias_tensor_ptr = &TensorCache::get_or_create("dbias", bias_shape, itype, true, false,
                                                    NVTE_DELAYED_TENSOR_SCALING, false);
    workspace_tensor_ptr = &TensorCache::get_or_create("workspace", shape, itype, true, false,
                                                        NVTE_DELAYED_TENSOR_SCALING, false);
  }

  if constexpr (PROC_METHOD == CAST_DBIAS_DACT || PROC_METHOD == CAST_DACT) {
    grad_tensor_ptr = &TensorCache::get_or_create("grad", shape, itype, true, false,
                                                   NVTE_DELAYED_TENSOR_SCALING, true);
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    if constexpr (PROC_METHOD == CAST_ONLY) {
      nvte_quantize(input_tensor.data(), output_tensor.data(), stream);
    } else if constexpr (PROC_METHOD == CAST_DBIAS) {
      nvte_quantize_dbias(input_tensor.data(), output_tensor.data(), dbias_tensor_ptr->data(), workspace_tensor_ptr->data(), stream);
    } else if constexpr (PROC_METHOD == CAST_DBIAS_DACT) {
      nvte_quantize_dbias_dgelu(grad_tensor_ptr->data(), input_tensor.data(), output_tensor.data(), dbias_tensor_ptr->data(), workspace_tensor_ptr->data(), stream);
    } else if constexpr (PROC_METHOD == CAST_DACT) {
      nvte_dgelu(grad_tensor_ptr->data(), input_tensor.data(), output_tensor.data(), stream);
    } else if constexpr (PROC_METHOD == CAST_ACT) {
      nvte_gelu(input_tensor.data(), output_tensor.data(), stream);
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t bytes_write_data   = rows * cols * sizeof(OType) *
                             ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  size_t bytes_write_scales = (USE_ROWWISE ? rows * scale_cols_row : 0) +
                               (USE_COLWISE ? scale_rows_col * scale_cols_col : 0);

  size_t bytes_read = rows * cols * sizeof(IType);
  if constexpr (PROC_METHOD == CAST_DBIAS_DACT || PROC_METHOD == CAST_DACT) {
    bytes_read += rows * cols * sizeof(IType);
  }
  if constexpr (PROC_METHOD == CAST_DBIAS || PROC_METHOD == CAST_DBIAS_DACT) {
    bytes_write_data += cols * sizeof(IType);
  }

  const size_t total_bytes = bytes_read + bytes_write_data + bytes_write_scales;

  set_bytes_processed(state, total_bytes);

  HIP_CHECK(hipStreamDestroy(stream));
}

#define REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, METHOD, METHOD_NAME) \
  BENCHMARK_TEMPLATE(BM_QuantizeMXFP8_Fused, ITYPE, OTYPE, 1, 32, METHOD) \
    ->Name("BM_QuantizeMXFP8_" METHOD_NAME "/rowwise/" INAME "_" ONAME) \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_QuantizeMXFP8_Fused, ITYPE, OTYPE, 32, 1, METHOD) \
    ->Name("BM_QuantizeMXFP8_" METHOD_NAME "/colwise/" INAME "_" ONAME) \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_QuantizeMXFP8_Fused, ITYPE, OTYPE, 32, 32, METHOD) \
    ->Name("BM_QuantizeMXFP8_" METHOD_NAME "/both/" INAME "_" ONAME) \
    COMMON_SHAPES \
    ->Unit(benchmark::kMicrosecond) \
    ->UseManualTime();

#define REGISTER_ALL_METHODS(ITYPE, OTYPE, INAME, ONAME) \
  REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, CAST_ONLY, "CastOnly") \
  REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, CAST_DBIAS, "CastDBias") \
  REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, CAST_DBIAS_DACT, "CastDBiasDACT") \
  REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, CAST_DACT, "CastDACT") \
  REGISTER_QUANTIZE_FUSED(ITYPE, OTYPE, INAME, ONAME, CAST_ACT, "CastACT")

REGISTER_ALL_METHODS(__half, fp8_e4m3, "FP16", "E4M3")
REGISTER_ALL_METHODS(hip_bfloat16, fp8_e4m3, "BF16", "E4M3")
REGISTER_ALL_METHODS(float, fp8_e4m3, "FP32", "E4M3")

BENCHMARK_MAIN();
