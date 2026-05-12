/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include "benchmark_utils.h"

#include "transformer_engine/normalization_hip.h"
#include "transformer_engine/transformer_engine_hip.h"

using namespace te_bench;
using namespace transformer_engine;

#define RMSNORM_SHAPES \
  ->Args({8192, 128})  \
  ->Args({8192, 1536}) \
  ->Args({8192, 7168})

template <typename T>
constexpr DType dtype_of() {
  if constexpr (std::is_same_v<T, float>) {
    return DType::kFloat32;
  } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
    return DType::kBFloat16;
  } else {
    return DType::kFloat16;
  }
}

template <typename WType, typename IType, typename OType, typename CType>
static void BM_RMSNormForward(benchmark::State& state) {
  const size_t N = state.range(0);
  const size_t H = state.range(1);
  const float epsilon = 1e-5f;

  const DType wtype = dtype_of<WType>();
  const DType itype = dtype_of<IType>();
  const DType otype = dtype_of<OType>();

  test::Tensor input("input", std::vector<size_t>{N, H}, itype);
  test::Tensor output("output", std::vector<size_t>{N, H}, otype);
  test::Tensor gamma("gamma", std::vector<size_t>{H}, wtype);
  test::Tensor rsigma("rsigma", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor workspace;

  test::fillUniform(&input);
  test::fillUniform(&gamma);
  test::setRandomScale(&output);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(), workspace.data(),
                   prop.multiProcessorCount, false, stream);

  workspace = test::Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(), workspace.data(),
                   prop.multiProcessorCount, false, stream);

  HIP_CHECK(hipStreamSynchronize(stream));
  warmup_gpu();

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     output.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, false, stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));

  // Algorithmic byte traffic by tensor role:
  // read x + gamma, write z + rsigma.
  const size_t bytes_read =
      N * H * sizeof(IType) +
      H * sizeof(WType);

  const size_t bytes_write =
      N * H * sizeof(OType) +
      N * sizeof(CType);

  set_bytes_processed(state, bytes_read + bytes_write);
}

template <typename WType, typename IType, typename OType, typename CType>
static void BM_RMSNormBackward(benchmark::State& state) {
  const size_t N = state.range(0);
  const size_t H = state.range(1);
  const float epsilon = 1e-5f;

  const DType wtype = dtype_of<WType>();
  const DType itype = dtype_of<IType>();
  const DType otype = dtype_of<OType>();

  test::Tensor input("input", std::vector<size_t>{N, H}, itype);
  test::Tensor output("output", std::vector<size_t>{N, H}, otype);
  test::Tensor gamma("gamma", std::vector<size_t>{H}, wtype);
  test::Tensor rsigma("rsigma", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor dz("dz", std::vector<size_t>{N, H}, otype);
  test::Tensor dx("dx", std::vector<size_t>{N, H}, itype);
  test::Tensor dgamma("dgamma", std::vector<size_t>{H}, wtype);
  test::Tensor workspace_fwd;
  test::Tensor workspace_bwd;

  test::fillUniform(&input);
  test::fillUniform(&gamma);
  test::setRandomScale(&output);
  test::fillUniform(&dz);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(), workspace_fwd.data(),
                   prop.multiProcessorCount, false, stream);

  workspace_fwd = test::Tensor("workspace_fwd",
                               workspace_fwd.rowwise_shape(),
                               workspace_fwd.dtype());

  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(), workspace_fwd.data(),
                   prop.multiProcessorCount, false, stream);

  nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                   dx.data(), dgamma.data(), workspace_bwd.data(),
                   prop.multiProcessorCount, false, stream);

  workspace_bwd = test::Tensor("workspace_bwd",
                               workspace_bwd.rowwise_shape(),
                               workspace_bwd.dtype());

  nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                   dx.data(), dgamma.data(), workspace_bwd.data(),
                   prop.multiProcessorCount, false, stream);

  HIP_CHECK(hipStreamSynchronize(stream));
  warmup_gpu();

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                     dx.data(), dgamma.data(), workspace_bwd.data(),
                     prop.multiProcessorCount, false, stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));

  // Algorithmic byte traffic by tensor role:
  // read dz + x + rsigma + gamma, write dx + dgamma.
  const size_t bytes_read =
      N * H * sizeof(OType) +
      N * H * sizeof(IType) +
      N * sizeof(CType) +
      H * sizeof(WType);

  const size_t bytes_write =
      N * H * sizeof(IType) +
      H * sizeof(WType);

  set_bytes_processed(state, bytes_read + bytes_write);
}

#define REGISTER_RMSNORM(WTYPE, ITYPE, OTYPE, CTYPE, NAME)        \
  BENCHMARK_TEMPLATE(BM_RMSNormForward, WTYPE, ITYPE, OTYPE, CTYPE) \
    ->Name("BM_RMSNormForward/" NAME)                            \
    RMSNORM_SHAPES                                                \
    ->Unit(benchmark::kMicrosecond)                               \
    ->UseManualTime();                                            \
  BENCHMARK_TEMPLATE(BM_RMSNormBackward, WTYPE, ITYPE, OTYPE, CTYPE) \
    ->Name("BM_RMSNormBackward/" NAME)                           \
    RMSNORM_SHAPES                                                \
    ->Unit(benchmark::kMicrosecond)                               \
    ->UseManualTime();

REGISTER_RMSNORM(hip_bfloat16, hip_bfloat16, hip_bfloat16, float, "BF16_BF16_BF16_FP32")
REGISTER_RMSNORM(half,         half,         half,         float, "FP16_FP16_FP16_FP32")
REGISTER_RMSNORM(float,        float,        float,        float, "FP32_FP32_FP32_FP32")

BENCHMARK_MAIN();
