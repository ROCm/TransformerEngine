/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/swizzle.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;

constexpr int MAT_TILE_DIM_M = 128;
constexpr int MAT_TILE_DIM_K = 128;

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_swizzle(const uint8_t *h_input, uint8_t *h_output,
                         const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_output_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int out_index = tile_output_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[out_index] = h_input[k + m * K];
      else
        h_output[out_index] = h_input[k * M + m];
    }
  }
}

void performTestSwizzle1D(const int num_tiles_M, const int num_tiles_K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if ((rowwise && columnwise) || !(rowwise || columnwise)){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + "is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const size_t M = num_tiles_M * MAT_TILE_DIM_M;
  const size_t K = num_tiles_K * MAT_TILE_DIM_K;
  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  const auto scale_shape = std::vector<size_t>{data_shape[0] / SF_MODE_X, data_shape[1] /SF_MODE_Y};

  std::vector<int> scaling_mode = {SF_MODE_X, SF_MODE_Y, 0};
  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  output.set_with_gemm_swizzled_scales(true);

  fillUniform(&input);

  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(scale_shape[0] * scale_shape[1]);

  nvte_swizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_swizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0], scale_shape[1]);
  else
    compute_ref_swizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[1], scale_shape[0]);

  (void)cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_swizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  } else {
    compareResults("output_swizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  }
}

class SwizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, std::pair<bool, bool>, bool>> {};


TEST_P(SwizzleTestSuite, TestSwizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto num_tiles = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzle1D(num_tiles.first, num_tiles.second,
                       scaling_mode.first, scaling_mode.second,
                       transa);
}

namespace {

std::vector<std::pair<int, int>> num_tiles = {
  {1, 1},
  {1, 132},
  {132, 1},
  {65, 256},
  {65, 257},
  {65, 258},
  {65, 259},
};

std::vector<std::pair<bool, bool>> scaling_mode = {
  {true, false},
  {false, true}
};

std::vector<bool> transa = {true, false};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(num_tiles),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleTestSuite::ParamType>& info) {
    std::string name = "ntiles" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

#ifdef __HIP_PLATFORM_AMD__

// MX pre-swizzle test (gfx1250 Tensile 3D layout)
//
// Tensile 3D: {K_scale, M}.reshape({K_scale, padM/4, 4}).permute({1, 0, 2})
// For source (m, k): dst = (m/4) * (K*4) + k*4 + (m%4)

// CPU reference for Tensile 3D MX scale pre-swizzle.
// Row-major input [M, K], output is a flat permuted array.
void compute_ref_mx_swizzle_row(const uint8_t *h_input, uint8_t *h_output,
                                   const int M, const int K,
                                   const int orig_M, const int orig_K) {
  constexpr int GROUP = 4;
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      uint8_t val = 127;  // E8M0 identity: 2^0 = 1.0
      if (m < orig_M && k < orig_K) {
        val = h_input[m * orig_K + k];
      }
      int group = k / GROUP;
      int within = k % GROUP;
      int dst = group * (M * GROUP) + m * GROUP + within;
      h_output[dst] = val;
    }
  }
}

void compute_ref_mx_swizzle_col(const uint8_t *h_input, uint8_t *h_output,
                                   const int M, const int K,
                                   const int orig_M, const int orig_K) {
  constexpr int GROUP = 4;
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      uint8_t val = 127;
      if (m < orig_M && k < orig_K) {
        val = h_input[k * orig_M + m];
      }
      int group = k / GROUP;
      int within = k % GROUP;
      int dst = group * (M * GROUP) + m * GROUP + within;
      h_output[dst] = val;
    }
  }
}

static size_t roundup_sz(size_t val, size_t mult) {
  return ((val + mult - 1) / mult) * mult;
}

class MxSwizzleTestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::pair<int, int>, bool>> {};

TEST_P(MxSwizzleTestSuite, TestMxSwizzle) {
  using namespace transformer_engine;
  using namespace test;

  const auto dims = std::get<0>(GetParam());
  const bool rowwise = std::get<1>(GetParam());

  // Original (unpadded) scale dimensions
  const size_t orig_M = dims.first;
  const size_t orig_K = dims.second;

  // Padded dimensions: K-tiled layout requires K_scale padded to multiple of 4
  const size_t M = orig_M;
  const size_t K = roundup_sz(orig_K, 4);

  // Allocate host input (unpadded) and fill with random data
  const size_t input_size = orig_M * orig_K;
  std::unique_ptr<uint8_t[]> h_input(new uint8_t[input_size]);
  std::mt19937 rng(42);
  for (size_t i = 0; i < input_size; i++) {
    h_input[i] = static_cast<uint8_t>(rng() % 256);
  }

  // Allocate device input
  uint8_t *d_input = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&d_input, input_size));
  NVTE_CHECK_CUDA(cudaMemcpy(d_input, h_input.get(), input_size, cudaMemcpyHostToDevice));

  // Allocate device output (padded size)
  const size_t output_size = M * K;
  uint8_t *d_output = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&d_output, output_size));
  NVTE_CHECK_CUDA(cudaMemset(d_output, 0, output_size));

  // Build TensorWrapper for input and output
  TensorWrapper input_tw(NVTE_MXFP8_1D_SCALING);
  TensorWrapper output_tw(NVTE_MXFP8_1D_SCALING);
  output_tw.set_with_gemm_swizzled_scales(true);

  // Data shape must be consistent with scale shape for validation.
  // Scale shapes use padded K; data shapes use unpadded dims
  // (kernel derives original_M/K from them).
  if (rowwise) {
    std::vector<size_t> data_shape_in = {orig_M, orig_K * 32};
    std::vector<size_t> data_shape_out = {M, K * 32};
    std::vector<size_t> scale_shape_in = {M, K};
    std::vector<size_t> scale_shape_out = {M, K};
    input_tw.set_rowwise_data(nullptr, DType::kFloat8E4M3, data_shape_in);
    input_tw.set_rowwise_scale_inv(d_input, DType::kFloat8E8M0, scale_shape_in);
    output_tw.set_rowwise_data(nullptr, DType::kFloat8E4M3, data_shape_out);
    output_tw.set_rowwise_scale_inv(d_output, DType::kFloat8E8M0, scale_shape_out);
  } else {
    std::vector<size_t> data_shape_in = {orig_K * 32, orig_M};
    std::vector<size_t> data_shape_out = {K * 32, M};
    std::vector<size_t> scale_shape_in = {K, M};
    std::vector<size_t> scale_shape_out = {K, M};
    input_tw.set_columnwise_data(nullptr, DType::kFloat8E4M3, data_shape_in);
    input_tw.set_columnwise_scale_inv(d_input, DType::kFloat8E8M0, scale_shape_in);
    output_tw.set_columnwise_data(nullptr, DType::kFloat8E4M3, data_shape_out);
    output_tw.set_columnwise_scale_inv(d_output, DType::kFloat8E8M0, scale_shape_out);
  }

  nvte_swizzle_scaling_factors(input_tw.data(), output_tw.data(), 0);

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  // Copy output back to host
  std::unique_ptr<uint8_t[]> h_output(new uint8_t[output_size]);
  NVTE_CHECK_CUDA(cudaMemcpy(h_output.get(), d_output, output_size, cudaMemcpyDeviceToHost));

  // Compute reference
  std::unique_ptr<uint8_t[]> h_ref(new uint8_t[output_size]);
  memset(h_ref.get(), 0, output_size);
  if (rowwise) {
    compute_ref_mx_swizzle_row(h_input.get(), h_ref.get(), M, K, orig_M, orig_K);
  } else {
    compute_ref_mx_swizzle_col(h_input.get(), h_ref.get(), M, K, orig_M, orig_K);
  }

  // Compare
  compareResults("mx_swizzle", h_output.get(), h_ref.get(), output_size);

  cudaFree(d_input);
  cudaFree(d_output);
}

namespace {

// Scale dimensions (M_scale, K_scale).
// K_scale will be padded to multiple of 4 by the test.
std::vector<std::pair<int, int>> mx_scale_dims = {
  {4, 4},        // minimal
  {8, 4},        // small
  {32, 8},       // medium
  {64, 16},      // larger
  {96, 8},       // non-power-of-2 M
  {128, 32},     // big
  {256, 64},     // bigger
  {512, 128},    // stress inter-tile
  {1024, 256},   // large
  {4096, 256},   // max stress
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MxSwizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(mx_scale_dims),
    ::testing::Values(true, false)
  ),
  [](const testing::TestParamInfo<MxSwizzleTestSuite::ParamType>& info) {
    std::string name = "M" + std::to_string(std::get<0>(info.param).first) +
      "_K" + std::to_string(std::get<0>(info.param).second) +
      (std::get<1>(info.param) ? "_row" : "_col");
    return name;
  });

#endif  // __HIP_PLATFORM_AMD__
