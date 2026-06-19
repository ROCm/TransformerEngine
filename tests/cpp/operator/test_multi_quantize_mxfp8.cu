/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename IType, typename OType>
void performTest(const std::vector<std::pair<size_t, size_t>> &tensor_dims,
                 bool rowwise, bool colwise) {
  const DType itype = TypeInfo<IType>::dtype;
  const DType otype = TypeInfo<OType>::dtype;
  const size_t num_tensors = tensor_dims.size();

  std::vector<Tensor> inputs, outputs_multi, outputs_ref;

  for (size_t i = 0; i < num_tensors; i++) {
    auto [rows, cols] = tensor_dims[i];
    inputs.emplace_back("input_" + std::to_string(i),
                        std::vector<size_t>{rows, cols}, itype);
    outputs_multi.emplace_back("output_multi_" + std::to_string(i),
                               std::vector<size_t>{rows, cols}, otype,
                               rowwise, colwise, NVTE_MXFP8_1D_SCALING);
    outputs_ref.emplace_back("output_ref_" + std::to_string(i),
                             std::vector<size_t>{rows, cols}, otype,
                             rowwise, colwise, NVTE_MXFP8_1D_SCALING);
    fillUniform(&inputs.back());
  }

  std::vector<NVTETensor> nvte_inputs, nvte_outputs_multi;
  for (auto &t : inputs) {
    nvte_inputs.push_back(t.data());
  }
  for (auto &t : outputs_multi) {
    nvte_outputs_multi.push_back(t.data());
  }

  nvte_multi_tensor_quantize(nvte_inputs.data(), nvte_outputs_multi.data(),
                             nullptr, num_tensors, 0);

  for (size_t i = 0; i < num_tensors; i++) {
    if (tensor_dims[i].first > 0 && tensor_dims[i].second > 0)
      nvte_quantize(inputs[i].data(), outputs_ref[i].data(), 0);
  }
  cudaDeviceSynchronize();

  for (size_t i = 0; i < num_tensors; i++) {
    auto [rows, cols] = tensor_dims[i];
    if (rows == 0 || cols == 0) continue;
    if (rowwise) {
      auto *multi_data = outputs_multi[i].rowwise_cpu_dptr<OType>();
      auto *ref_data   = outputs_ref[i].rowwise_cpu_dptr<OType>();
      for (size_t j = 0; j < rows * cols; j++) {
        ASSERT_EQ(static_cast<uint8_t>(multi_data[j]),
                  static_cast<uint8_t>(ref_data[j]))
            << "Mismatch at tensor " << i << " element " << j;
      }
      auto *multi_scales = outputs_multi[i].rowwise_cpu_scale_inv_ptr<uint8_t>();
      auto *ref_scales   = outputs_ref[i].rowwise_cpu_scale_inv_ptr<uint8_t>();
      size_t num_scales = rows * ((cols + 31) / 32);
      for (size_t j = 0; j < num_scales; j++) {
        ASSERT_EQ(multi_scales[j], ref_scales[j])
            << "Scale mismatch at tensor " << i << " scale " << j;
      }
    }
    if (colwise) {
      auto *multi_data = outputs_multi[i].columnwise_cpu_dptr<OType>();
      auto *ref_data   = outputs_ref[i].columnwise_cpu_dptr<OType>();
      for (size_t j = 0; j < rows * cols; j++) {
        ASSERT_EQ(static_cast<uint8_t>(multi_data[j]),
                  static_cast<uint8_t>(ref_data[j]))
            << "Colwise mismatch at tensor " << i << " element " << j;
      }
    }
  }
}

std::vector<std::pair<size_t, size_t>> getTestDims(int config) {
  switch (config) {
    case 0:
      return {{128, 128},
              {256, 256},
              {128, 512},
              {512, 128}};
    case 1:
      return {{128, 4096},
              {128, 4096},
              {384, 4096},
              {256, 4096},
              {256, 4096}};
    case 2:
      return {{0, 128},
              {128, 256},
              {256, 128}};
    default:
      return {};
  }
}

enum ScalingMode {
  Rowwise = 0,
  Colwise = 1,
  Both = 2
};

class MultiQuantizeMXFP8TestSuite
    : public ::testing::TestWithParam<
          std::tuple<transformer_engine::DType, transformer_engine::DType, int, ScalingMode>> {};

TEST_P(MultiQuantizeMXFP8TestSuite, Test) {
  auto [itype, otype, config, mode] = GetParam();
  auto dims = getTestDims(config);
  bool rowwise = (mode == Rowwise || mode == Both);
  bool colwise = (mode == Colwise || mode == Both);

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(itype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(otype, OType,
      performTest<IType, OType>(dims, rowwise, colwise);
    )
  )
}

static const char *scalingModeName(ScalingMode mode) {
  switch (mode) {
    case Rowwise: return "rowwise";
    case Colwise: return "colwise";
    case Both:    return "both";
    default:      return "unknown";
  }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, MultiQuantizeMXFP8TestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::Values(0, 1, 2),
        ::testing::Values(Rowwise, Colwise, Both)),
    [](const testing::TestParamInfo<MultiQuantizeMXFP8TestSuite::ParamType> &info) {
      return test::typeName(std::get<0>(info.param)) + "_" +
             test::typeName(std::get<1>(info.param)) + "_config" +
             std::to_string(std::get<2>(info.param)) + "_" +
             scalingModeName(std::get<3>(info.param));
    });

}  // namespace
