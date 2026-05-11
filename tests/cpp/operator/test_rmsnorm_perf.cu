#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"
#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include <gtest/gtest.h>

using namespace transformer_engine;
using namespace test;

namespace {

std::vector<std::pair<size_t, size_t>> tensor_dims = {
    {8192, 1536},
    {8192, 7168},
};

template <typename Fn>
double time_ms(Fn&& fn, int warmup = 20, int iters = 100) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  NVTE_CHECK_CUDA(cudaEventCreate(&start));
  NVTE_CHECK_CUDA(cudaEventCreate(&stop));

  NVTE_CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  NVTE_CHECK_CUDA(cudaEventRecord(stop));
  NVTE_CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.f;
  NVTE_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  NVTE_CHECK_CUDA(cudaEventDestroy(start));
  NVTE_CHECK_CUDA(cudaEventDestroy(stop));

  return static_cast<double>(ms) / iters;
}

template <typename OutputType>
void performTest(const size_t N, const size_t H) {
    using InputType = OutputType;
    using WeightType = OutputType;
    const DType itype = TypeInfo<InputType>::dtype;
    const DType otype = TypeInfo<OutputType>::dtype;
    const DType wtype = TypeInfo<WeightType>::dtype;

    float epsilon = 1e-5;

    Tensor input("input", std::vector<size_t>{ N, H }, itype);
    Tensor z("z", std::vector<size_t>{ N, H }, otype);
    Tensor gamma("gamma", std::vector<size_t>{ H }, wtype);
    Tensor rsigma("rsigma", std::vector<size_t>{ N }, DType::kFloat32);
    Tensor dz("dz", std::vector<size_t>{ N, H }, wtype);
    Tensor dx("dx", std::vector<size_t>{ N, H }, itype);
    Tensor dgamma("dgamma", std::vector<size_t>{ H }, wtype);
    Tensor workspace_fwd, workspace_bwd;

    fillUniform(&input);
    fillUniform(&gamma);
    setRandomScale(&z);
    fillUniform(&dz);

    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, false, 0);
    workspace_fwd = Tensor("workspace", workspace_fwd.rowwise_shape(), workspace_fwd.dtype());
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    double fwd_ms = time_ms([&] {
        nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                        z.data(), rsigma.data(), workspace_fwd.data(),
                        prop.multiProcessorCount, false, 0);
    });

    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                    dx.data(), dgamma.data(), workspace_bwd.data(), prop.multiProcessorCount,
                    false, 0);
    workspace_bwd = Tensor("workspace", workspace_bwd.rowwise_shape(), workspace_bwd.dtype());
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    double bwd_ms = time_ms([&] {
        nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                        dx.data(), dgamma.data(), workspace_bwd.data(),
                        prop.multiProcessorCount, false, 0);
    });

    std::cout << "RMSNORM_PERF"
            << " N=" << N
            << " H=" << H
            << " dtype=" << typeName(otype)
            << " fwd_ms=" << fwd_ms
            << " bwd_ms=" << bwd_ms
            << std::endl;

}

}  // namespace

class RMSNormPerfTestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::pair<size_t, size_t>, transformer_engine::DType>> {};

TEST_P(RMSNormPerfTestSuite, TestRMSNormPerf) {
    const auto tensor_shape = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());

    const size_t N = tensor_shape.first;
    const size_t H = tensor_shape.second;

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest<OutputType>(N, H);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    RMSNormPerfTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<RMSNormPerfTestSuite::ParamType>& info) {
        std::string name =
            std::to_string(std::get<0>(info.param).first) + "X" +
            std::to_string(std::get<0>(info.param).second) + "X" +
            test::typeName(std::get<1>(info.param));
        return name;
    });