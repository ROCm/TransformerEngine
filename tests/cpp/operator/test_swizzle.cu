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

// ============================================================================
// End-to-end MXFP8 GEMM test with pre-swizzled scales
//
// Verifies that the full pipeline works:
//   1. Create MXFP8 FP8 tensors with random data + scales
//   2. Run a reference GEMM (using un-swizzled scales)
//   3. Swizzle the scales via nvte_swizzle_scaling_factors
//   4. Run the actual hipBLASlt GEMM
//   5. Compare results
// ============================================================================

#include <transformer_engine/gemm.h>

// Helper: swizzle the MXFP8 scale_inv of a test::Tensor in-place.
// Allocates a temp device buffer, swizzles into it, copies back.
static void swizzle_tensor_scales(test::Tensor &t, bool rowwise) {
  using namespace transformer_engine;

  void *scale_ptr = rowwise ? t.rowwise_scale_inv_dptr()
                            : t.columnwise_scale_inv_dptr();
  if (!scale_ptr) return;

  const NVTEShape scale_shape = rowwise ? t.rowwise_scale_inv_shape()
                                        : t.columnwise_scale_inv_shape();
  const NVTEShape data_shape = rowwise ? t.rowwise_shape()
                                       : t.columnwise_shape();

  size_t num_scales = 1;
  for (size_t d = 0; d < scale_shape.ndim; d++) {
    num_scales *= scale_shape.data[d];
  }

  // Allocate temp buffer for swizzled output
  uint8_t *d_tmp = nullptr;
  ASSERT_EQ(cudaMalloc(&d_tmp, num_scales), cudaSuccess);

  // Build TensorWrapper pair for the swizzle call
  TensorWrapper input_tw(NVTE_MXFP8_1D_SCALING);
  TensorWrapper output_tw(NVTE_MXFP8_1D_SCALING);
  output_tw.set_with_gemm_swizzled_scales(true);

  if (rowwise) {
    input_tw.set_rowwise_data(nullptr, t.dtype(), data_shape);
    input_tw.set_rowwise_scale_inv(scale_ptr, DType::kFloat8E8M0, scale_shape);
    output_tw.set_rowwise_data(nullptr, t.dtype(), data_shape);
    output_tw.set_rowwise_scale_inv(d_tmp, DType::kFloat8E8M0, scale_shape);
  } else {
    input_tw.set_columnwise_data(nullptr, t.dtype(), data_shape);
    input_tw.set_columnwise_scale_inv(scale_ptr, DType::kFloat8E8M0, scale_shape);
    output_tw.set_columnwise_data(nullptr, t.dtype(), data_shape);
    output_tw.set_columnwise_scale_inv(d_tmp, DType::kFloat8E8M0, scale_shape);
  }

  nvte_swizzle_scaling_factors(input_tw.data(), output_tw.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Copy swizzled scales back over the original
  ASSERT_EQ(cudaMemcpy(scale_ptr, d_tmp, num_scales, cudaMemcpyDeviceToDevice), cudaSuccess);
  cudaFree(d_tmp);

  // Mark tensor as having swizzled scales
  t.set_with_gemm_swizzled_scales(true);
}

// Simple GPU reference kernel for MXFP8 GEMM: D = A * B^T  (TN layout)
// A is [M, K] row-major, B is [N, K] row-major, D is [M, N] column-major
// Scales are E8M0, one per group of 32 elements along K.
__global__ void mxfp8_gemm_ref_kernel(
    const test::fp8e4m3 *a_data, const uint8_t *a_scale, size_t a_scale_ld,
    const test::fp8e4m3 *b_data, const uint8_t *b_scale, size_t b_scale_ld,
    test::bf16 *d_data,
    size_t M, size_t K, size_t N) {
  const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;

  float acc = 0.0f;
  for (size_t kk = 0; kk < K; kk++) {
    size_t kc = kk / 32;
    float a_sinv = exp2f(static_cast<float>(a_scale[i * a_scale_ld + kc]) - 127.0f);
    float b_sinv = exp2f(static_cast<float>(b_scale[j * b_scale_ld + kc]) - 127.0f);
    float a_val = static_cast<float>(a_data[i * K + kk]);
    float b_val = static_cast<float>(b_data[j * K + kk]);
    acc += a_sinv * a_val * b_sinv * b_val;
  }
  d_data[i + j * M] = static_cast<test::bf16>(acc);
}

struct MxGemmParams {
  size_t m, k, n;
};

class MxGemmTestSuite
    : public ::testing::TestWithParam<MxGemmParams> {};

TEST_P(MxGemmTestSuite, TestMxfp8GemmE2E) {
  using namespace transformer_engine;
  using namespace test;

  const auto &p = GetParam();
  const size_t M = p.m;
  const size_t K = p.k;
  const size_t N = p.n;

  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

  // MXFP8 requires gfx950+ (MI350) or gfx1250 (MI450)
  bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5) ||
                          (prop.major >= 10);
  if (!mxfp8_supported) {
    GTEST_SKIP() << "MXFP8 GEMM not supported on this GPU";
  }

  // TN layout: A is [M, K], B is [N, K]
  const bool transa = true;
  const bool transb = false;

  Tensor A("A", std::vector<size_t>{M, K}, DType::kFloat8E4M3, true, false, NVTE_MXFP8_1D_SCALING);
  Tensor B("B", std::vector<size_t>{N, K}, DType::kFloat8E4M3, true, false, NVTE_MXFP8_1D_SCALING);
  Tensor D("D", std::vector<size_t>{N, M}, DType::kBFloat16);
  Tensor RefD("RefD", std::vector<size_t>{N, M}, DType::kBFloat16);
  Tensor bias;
  Tensor pre_gelu_out;

  fillUniform(&A);
  fillUniform(&B);

  // Override scales with values in [120,127] so layout errors are detectable.
  // Default random [0,127] produces mostly tiny scales (2^(-127)..2^0),
  // making the test insensitive to permutation errors.
  {
    auto fill_discriminating_scales = [](void *scale_ptr, size_t count) {
      std::vector<uint8_t> h(count);
      std::mt19937 rng(42);
      std::uniform_int_distribution<uint8_t> dist(120, 127);
      for (size_t i = 0; i < count; i++) h[i] = dist(rng);
      cudaMemcpy(scale_ptr, h.data(), count, cudaMemcpyHostToDevice);
    };
    auto a_sh = A.rowwise_scale_inv_shape();
    auto b_sh = B.rowwise_scale_inv_shape();
    fill_discriminating_scales(A.rowwise_scale_inv_dptr(), a_sh.data[0] * a_sh.data[1]);
    fill_discriminating_scales(B.rowwise_scale_inv_dptr(), b_sh.data[0] * b_sh.data[1]);
  }

  // GPU reference with un-swizzled (compact) scales
  const auto a_scale_shape = A.rowwise_scale_inv_shape();
  const auto b_scale_shape = B.rowwise_scale_inv_shape();

  std::cout << "  A_scale shape: [" << a_scale_shape.data[0] << ", " << a_scale_shape.data[1]
            << "], B_scale shape: [" << b_scale_shape.data[0] << ", " << b_scale_shape.data[1]
            << "]" << std::endl;

  {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    mxfp8_gemm_ref_kernel<<<grid, block>>>(
        static_cast<const fp8e4m3 *>(A.rowwise_dptr()),
        static_cast<const uint8_t *>(A.rowwise_scale_inv_dptr()),
        a_scale_shape.data[1],
        static_cast<const fp8e4m3 *>(B.rowwise_dptr()),
        static_cast<const uint8_t *>(B.rowwise_scale_inv_dptr()),
        b_scale_shape.data[1],
        static_cast<bf16 *>(RefD.rowwise_dptr()),
        M, K, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  // Swizzle scales to K-tiled layout for hipBLASlt BLK32_UE8M0_32_8_EXT on gfx1250.
  // Layout: {M, K_scale}.reshape({M, K_scale/4, 4}).permute({1,0,2})
  //   dst(m,k) = (k/4)*M*4 + m*4 + (k%4)
  if (prop.major >= 12) {
    swizzle_tensor_scales(A, true);
    swizzle_tensor_scales(B, true);
  }

  // Run actual GEMM
  size_t workspace_size = 134217728;  // 128MB
  Tensor Workspace("Workspace", std::vector<size_t>{workspace_size}, DType::kByte);

  nvte_cublas_gemm(A.data(), B.data(), D.data(),
                   bias.data(), pre_gelu_out.data(),
                   transa, transb,
                   /*grad=*/false,
                   Workspace.data(),
                   /*accumulate=*/false,
                   /*use_split_accumulator=*/false,
                   prop.multiProcessorCount,
                   0);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Compare
  D.to_cpu();
  RefD.to_cpu();

  const bf16 *d_ptr = D.rowwise_cpu_dptr<bf16>();
  const bf16 *ref_ptr = RefD.rowwise_cpu_dptr<bf16>();
  double max_atol = 0.0;
  double max_rtol = 0.0;
  int mismatch_count = 0;
  for (size_t i = 0; i < M * N; i++) {
    float actual = static_cast<float>(d_ptr[i]);
    float expected = static_cast<float>(ref_ptr[i]);
    double diff = std::abs(actual - expected);
    double denom = std::max(std::abs((double)expected), 1e-6);
    if (diff > 5e-2 && mismatch_count < 10) {
      size_t row = i / N;
      size_t col = i % N;
      std::cout << "  MISMATCH [" << row << "," << col << "]: actual=" << actual
                << " expected=" << expected << " diff=" << diff << std::endl;
      mismatch_count++;
    }
    max_atol = std::max(max_atol, diff);
    max_rtol = std::max(max_rtol, diff / denom);
  }

  // MXFP8 GEMM tolerance
  constexpr double ATOL = 5e-2;
  constexpr double RTOL = 5e-2;
  EXPECT_LE(max_atol, ATOL) << "Absolute error too large: " << max_atol;
  EXPECT_LE(max_rtol, RTOL) << "Relative error too large: " << max_rtol;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    MxGemmTestSuite,
    ::testing::Values(
        MxGemmParams{32, 128, 16},
        MxGemmParams{64, 128, 32},
        MxGemmParams{128, 128, 64},
        MxGemmParams{64, 256, 32},
        MxGemmParams{128, 384, 64}
    ),
    [](const testing::TestParamInfo<MxGemmTestSuite::ParamType> &info) {
      return "M" + std::to_string(info.param.m) +
             "_K" + std::to_string(info.param.k) +
             "_N" + std::to_string(info.param.n);
    });

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
  ASSERT_EQ(cudaMalloc(&d_input, input_size), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, h_input.get(), input_size, cudaMemcpyHostToDevice), cudaSuccess);

  // Allocate device output (padded size)
  const size_t output_size = M * K;
  uint8_t *d_output = nullptr;
  ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, output_size), cudaSuccess);

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

  nvte_swizzle_scaling_factors_mx(input_tw.data(), output_tw.data(), 0);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Copy output back to host
  std::unique_ptr<uint8_t[]> h_output(new uint8_t[output_size]);
  ASSERT_EQ(cudaMemcpy(h_output.get(), d_output, output_size, cudaMemcpyDeviceToHost),
            cudaSuccess);

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
