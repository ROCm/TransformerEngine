/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "benchmark_utils.h"

#include "transformer_engine/normalization_hip.h"
#include "transformer_engine/transformer_engine_hip.h"

using namespace te_bench;
using namespace transformer_engine;

#define NORM_SHAPES    \
  ->Args({8192, 128})  \
  ->Args({8192, 1536}) \
  ->Args({8192, 7168})

constexpr float kNormEpsilon = 1e-5f;

enum class BenchNormType {
  LayerNorm,
  RMSNorm,
};

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

template <BenchNormType Norm, typename WType, typename IType, typename OType, typename CType>
static void BM_NormForward(benchmark::State& state) {
  const size_t N = state.range(0);
  const size_t H = state.range(1);
  constexpr bool zero_centered_gamma = false;

  const DType wtype = dtype_of<WType>();
  const DType itype = dtype_of<IType>();
  const DType otype = dtype_of<OType>();

  test::Tensor input("input", std::vector<size_t>{N, H}, itype);
  test::Tensor output("output", std::vector<size_t>{N, H}, otype);
  test::Tensor gamma("gamma", std::vector<size_t>{H}, wtype);
  test::Tensor beta("beta", std::vector<size_t>{H}, wtype);
  test::Tensor mu("mu", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor rsigma("rsigma", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor workspace;

  test::fillUniform(&input);
  test::fillUniform(&gamma);
  test::fillUniform(&beta);
  test::setRandomScale(&output);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  if constexpr (Norm == BenchNormType::LayerNorm) {
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), kNormEpsilon,
                       output.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), kNormEpsilon,
                     output.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);
  }

  workspace = test::Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

  if constexpr (Norm == BenchNormType::LayerNorm) {
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), kNormEpsilon,
                       output.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), kNormEpsilon,
                     output.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);
  }

  warmup_gpu();

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    if constexpr (Norm == BenchNormType::LayerNorm) {
      nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), kNormEpsilon,
                         output.data(), mu.data(), rsigma.data(), workspace.data(),
                         prop.multiProcessorCount, zero_centered_gamma, stream);
    } else {
      nvte_rmsnorm_fwd(input.data(), gamma.data(), kNormEpsilon,
                       output.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));

  size_t bytes_read = N * H * sizeof(IType) + H * sizeof(WType);       

  size_t bytes_write = N * H * sizeof(OType) + N * sizeof(float);

  if constexpr (Norm == BenchNormType::LayerNorm) {
    bytes_read += H * sizeof(WType);   // beta
    bytes_write += N * sizeof(float);  // mu
  }

  set_bytes_processed(state, bytes_read + bytes_write);
}

template <BenchNormType Norm, typename WType, typename IType, typename OType, typename CType>
static void BM_NormBackward(benchmark::State& state) {
  const size_t N = state.range(0);
  const size_t H = state.range(1);
  constexpr bool zero_centered_gamma = false;

  const DType wtype = dtype_of<WType>();
  const DType itype = dtype_of<IType>();
  const DType otype = dtype_of<OType>();

  test::Tensor input("input", std::vector<size_t>{N, H}, itype);
  test::Tensor output("output", std::vector<size_t>{N, H}, otype);
  test::Tensor gamma("gamma", std::vector<size_t>{H}, wtype);
  test::Tensor beta("beta", std::vector<size_t>{H}, wtype);
  test::Tensor mu("mu", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor rsigma("rsigma", std::vector<size_t>{N}, DType::kFloat32);
  test::Tensor dz("dz", std::vector<size_t>{N, H}, otype);
  test::Tensor dx("dx", std::vector<size_t>{N, H}, itype);
  test::Tensor dgamma("dgamma", std::vector<size_t>{H}, wtype);
  test::Tensor dbeta("dbeta", std::vector<size_t>{H}, wtype);
  test::Tensor workspace_fwd;
  test::Tensor workspace_bwd;

  test::fillUniform(&input);
  test::fillUniform(&gamma);
  test::fillUniform(&beta);
  test::setRandomScale(&output);
  test::fillUniform(&dz);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  if constexpr (Norm == BenchNormType::LayerNorm) {
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), kNormEpsilon,
                       output.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), kNormEpsilon,
                     output.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);
  }

  workspace_fwd = test::Tensor("workspace_fwd",
                               workspace_fwd.rowwise_shape(),
                               workspace_fwd.dtype());

  if constexpr (Norm == BenchNormType::LayerNorm) {
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), kNormEpsilon,
                       output.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);

    nvte_layernorm_bwd(dz.data(), input.data(), mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(), workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), kNormEpsilon,
                     output.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);

    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                     dx.data(), dgamma.data(), workspace_bwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);
  }

  workspace_bwd = test::Tensor("workspace_bwd",
                               workspace_bwd.rowwise_shape(),
                               workspace_bwd.dtype());

  if constexpr (Norm == BenchNormType::LayerNorm) {
    nvte_layernorm_bwd(dz.data(), input.data(), mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(), workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
  } else {
    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                     dx.data(), dgamma.data(), workspace_bwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, stream);
  }

  warmup_gpu();

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    if constexpr (Norm == BenchNormType::LayerNorm) {
      nvte_layernorm_bwd(dz.data(), input.data(), mu.data(), rsigma.data(), gamma.data(),
                         dx.data(), dgamma.data(), dbeta.data(), workspace_bwd.data(),
                         prop.multiProcessorCount, zero_centered_gamma, stream);
    } else {
      nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, stream);
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));

  size_t bytes_read =
      N * H * sizeof(OType) +  // dz
      N * H * sizeof(IType) +  // x
      N * sizeof(float) +      // rsigma
      H * sizeof(WType);       // gamma

  size_t bytes_write =
      N * H * sizeof(IType) +  // dx
      H * sizeof(WType);       // dgamma

  if constexpr (Norm == BenchNormType::LayerNorm) {
    bytes_read += N * sizeof(float);   // mu
    bytes_write += H * sizeof(WType);  // dbeta
  }

  set_bytes_processed(state, bytes_read + bytes_write);
}

#define REGISTER_NORM_BENCH(NORM_ENUM, NORM_NAME, WTYPE, ITYPE, OTYPE, CTYPE, NAME) \
  BENCHMARK_TEMPLATE(BM_NormForward, NORM_ENUM, WTYPE, ITYPE, OTYPE, CTYPE)         \
    ->Name("BM_" NORM_NAME "Forward/" NAME)                                        \
    NORM_SHAPES                                                                     \
    ->Unit(benchmark::kMicrosecond)                                                 \
    ->UseManualTime();                                                              \
  BENCHMARK_TEMPLATE(BM_NormBackward, NORM_ENUM, WTYPE, ITYPE, OTYPE, CTYPE)        \
    ->Name("BM_" NORM_NAME "Backward/" NAME)                                       \
    NORM_SHAPES                                                                     \
    ->Unit(benchmark::kMicrosecond)                                                 \
    ->UseManualTime();

#define REGISTER_RMSNORM(WTYPE, ITYPE, OTYPE, CTYPE, NAME) \
  REGISTER_NORM_BENCH(BenchNormType::RMSNorm, "RMSNorm", WTYPE, ITYPE, OTYPE, CTYPE, NAME)

#define REGISTER_LAYERNORM(WTYPE, ITYPE, OTYPE, CTYPE, NAME) \
  REGISTER_NORM_BENCH(BenchNormType::LayerNorm, "LayerNorm", WTYPE, ITYPE, OTYPE, CTYPE, NAME)

REGISTER_RMSNORM(hip_bfloat16, hip_bfloat16, hip_bfloat16, float, "BF16_BF16_BF16_FP32")
REGISTER_RMSNORM(half,         half,         half,         float, "FP16_FP16_FP16_FP32")
REGISTER_RMSNORM(float,        float,        float,        float, "FP32_FP32_FP32_FP32")

REGISTER_LAYERNORM(hip_bfloat16, hip_bfloat16, hip_bfloat16, float, "BF16_BF16_BF16_FP32")
REGISTER_LAYERNORM(half,         half,         half,         float, "FP16_FP16_FP16_FP32")
REGISTER_LAYERNORM(float,        float,        float,        float, "FP32_FP32_FP32_FP32")

BENCHMARK_MAIN();
