/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef __HIP_PLATFORM_AMD__

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include <cmath>

#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kFP4BlockSize1D = 16;

// Generates random FP8 (E4M3) scale values by sampling raw 8-bit patterns.
// Only finite, non-negative scales are allowed.
// Values are written using memcpy to preserve exact
// bit patterns rather than relying on numeric conversion.
void generate_1d_scales(fp8e4m3* scale_buffer,
                        const size_t unpadded_blocks_Y,
                        const size_t unpadded_blocks_X,
                        const size_t scales_stride,
                        std::mt19937& gen,
                        std::uniform_int_distribution<int>& finite_nonneg_e4m3_dis) {
    const size_t total_elems = unpadded_blocks_Y * scales_stride;
    std::memset(scale_buffer, 0, total_elems * sizeof(fp8e4m3));

    for (size_t row = 0; row < unpadded_blocks_Y; ++row) {
        for (size_t block = 0; block < unpadded_blocks_X; ++block) {
            const size_t scale_idx = row * scales_stride + block;
            const uint8_t scale = static_cast<uint8_t>(finite_nonneg_e4m3_dis(gen));
            std::memcpy(&scale_buffer[scale_idx], &scale, sizeof(scale));
        }
    }
}

// Write one mathematical FP4 E2M1 value, represented as a raw nibble [0, 15],
// into packed storage. Two mathematical FP4 values are packed per byte:
// even mathematical index -> low nibble, odd mathematical index -> high nibble.
void set_fp4_nibble(fp4e2m1* data, const size_t mathematical_idx, const uint8_t nibble) {
    ASSERT_TRUE(nibble < 16);
    auto* raw = reinterpret_cast<uint8_t*>(data);
    const size_t byte_idx = mathematical_idx / 2;
    const uint8_t val = nibble;

    if ((mathematical_idx % 2) == 0) {
        // set low nibble
        raw[byte_idx] = (raw[byte_idx] & 0xF0) | val;
    } else {
        // set high nibble
        raw[byte_idx] = (raw[byte_idx] & 0x0F) | (val << 4);
    }
}

// Populate FP4 (E2M1) tensor using packed 4-bit encoding.
void generate_data(fp4e2m1* data,
                   const size_t rows,
                   const size_t cols,
                   std::mt19937& gen,
                   std::uniform_int_distribution<int>& e2m1_dis) {
    const size_t packed_bytes = (rows * cols * BitsNumber<fp4e2m1>::num_bits) / 8;

    std::memset(data, 0, packed_bytes);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const uint8_t nibble = static_cast<uint8_t>(e2m1_dis(gen)) & 0xF;

            const size_t idx = i * cols + j;
            set_fp4_nibble(data, idx, nibble);
        }
    }
}

// Decode a single FP4 (E2M1) value from packed storage.
float get_fp4_value(const fp4e2m1* data, const size_t idx) {
    const auto* raw = reinterpret_cast<const uint8_t*>(data);
    const size_t byte_idx = idx / 2;
    const uint8_t packed = raw[byte_idx];
    const uint8_t nibble = (idx % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    return E2M1_LUT[nibble];
}

// Reference implementation: dequantize packed FP4 (E2M1) input using per-block FP8_E4M3 scales.
// Each block of 1x16 elements shares one scale; values are decoded to float and scaled.
template <typename OutputType>
void compute_ref(const fp4e2m1* input,
                 OutputType* output,
                 const fp8e4m3* scales,
                 const float amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scale_stride) {
#ifdef __HIP_PLATFORM_AMD__
    const float fp8_max = Numeric_Traits<fp8e4m3>::maxNorm;
    const float factor_inv = 1.0f / (6.0f * fp8_max);
#else
    constexpr float factor_inv = 1.0f / (6.0f * 448.0f);
#endif

    const size_t blocks_per_row = cols / kFP4BlockSize1D;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const float scale =
                static_cast<float>(scales[i * scale_stride + b]) * amax * factor_inv;

            for (size_t k = 0; k < kFP4BlockSize1D; ++k) {
                const size_t col = b * kFP4BlockSize1D + k;
                const size_t idx = i * cols + col;
                const float x = get_fp4_value(input, idx);
                output[idx] = static_cast<OutputType>(x * scale);
            }
        }
    }
}

template <typename OutputType>
void run_single_case(const std::string& case_name,
                     Tensor& input,
                     const size_t rows,
                     const size_t cols,
                     const size_t scale_stride,
                     const float amax,
                     DType otype) {
    Tensor output(case_name + "_output", std::vector<size_t>{rows, cols}, otype, true, false);

    std::unique_ptr<OutputType[]> ref_output =
        std::make_unique<OutputType[]>(rows * cols);

    input.from_cpu();
    nvte_dequantize(input.data(), output.data(), 0);

    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << case_name << ": " << cudaGetErrorString(err);

    output.to_cpu();

    compute_ref(input.rowwise_cpu_dptr<fp4e2m1>(),
                ref_output.get(),
                input.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                amax,
                rows,
                cols,
                scale_stride);

    auto [atol, rtol] = getTolerances(otype);
    compareResults(case_name, output, ref_output.get(), true, atol, rtol);
}

// End-to-end test: generate random FP4 input and FP8 scales.
// Only tests row-wise 1D dequant since the kernel is hardwired for that.
template <typename OutputType>
void performTest(const size_t rows, const size_t cols, DType otype) {
#ifdef __HIP_PLATFORM_AMD__
    const std::array<size_t, 4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16, NVTE_NVFP4_1D_SCALING);
#else
    const std::array<size_t, 4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);
#endif //#ifdef __HIP_PLATFORM_AMD__

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    const DType itype = DType::kFloat4E2M1;

    Tensor input("rowwise_1d_dequant_input",
                 std::vector<size_t>{rows, cols},
                 itype,
                 true, false,
                 NVTE_NVFP4_1D_SCALING);

    static std::mt19937 gen(42);
    std::uniform_int_distribution<int> e2m1_dis(0, 15);
    std::uniform_int_distribution<int> finite_nonneg_e4m3_dis(0, 126);

    generate_data(input.rowwise_cpu_dptr<fp4e2m1>(),
                  rows,
                  cols,
                  gen,
                  e2m1_dis);

    // Row-wise 1D scales on [rows, cols]
    generate_1d_scales(input.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                       unpadded_blocks_Y,
                       unpadded_blocks_X,
                       scales_stride,
                       gen,
                       finite_nonneg_e4m3_dis);

    // With the current test_common NVFP4 helper path on ROCm, there is no direct
    // way to populate a separate global amax buffer for dequant, so this test
    // explicitly covers the HIP nullptr -> 1.0f fallback path for now.
    const float amax = 1.0f;

    run_single_case<OutputType>("rowwise_1d_dequant",
                                input,
                                rows,
                                cols,
                                scales_stride,
                                amax,
                                otype);
}

std::vector<std::pair<size_t, size_t>> tensor_dims = {
    {32, 32},
    {32, 64},
    {64, 32},
    {64, 96},
    {128, 128},
    {256, 256},
    {512, 512},
    {1024, 1024},
    {2048, 2048},
};

}  // namespace

class DequantizeNVFP4TestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::pair<size_t, size_t>, transformer_engine::DType>> {};

TEST_P(DequantizeNVFP4TestSuite, TestDequantizeNVFP4) {
    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());

    const size_t rows = tensor_size.first;
    const size_t cols = tensor_size.second;

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
        output_type, OutputType,
        performTest<OutputType>(rows, cols, output_type););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<DequantizeNVFP4TestSuite::ParamType>& info) {
        std::string name =
            std::to_string(std::get<0>(info.param).first) + "X" +
            std::to_string(std::get<0>(info.param).second) + "X" +
            test::typeName(std::get<1>(info.param));
        return name;
    });

#else  // __HIP_PLATFORM_AMD__

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <transformer_engine/cast.h>
#include <transformer_engine/swizzle.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

#if FP4_TYPE_SUPPORTED

namespace {

float2 cvt_fp4x2_to_float2(fp4e2m1x2 fp4_pair) {
    const __half2_raw raw =
        __nv_cvt_fp4x2_to_halfraw2(
            *reinterpret_cast<__nv_fp4x2_storage_t *>(&fp4_pair), __NV_E2M1);
    const __half2 h2(raw);
    return {static_cast<float>(h2.x), static_cast<float>(h2.y)};
}

template <typename OType>
void compute_ref_dequantize_nvfp4(const uint8_t *packed_data,
                                  const fp8e4m3 *scales,
                                  const std::vector<float> &amax,
                                  OType *output,
                                  size_t rows,
                                  size_t cols,
                                  size_t scale_stride,
                                  int e4m3_max) {
    const float factor_inv = 1.0f / (6.0f * static_cast<float>(e4m3_max));
    constexpr size_t BLOCK_SIZE = 16;
    const size_t Mread = cols / BLOCK_SIZE;
    const size_t bytes_per_block = BLOCK_SIZE / 2;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < Mread; ++block) {
            const fp8e4m3 scale = scales[row * scale_stride + block];
            const float final_scale =
                static_cast<float>(scale) * (amax.size() == 1 ? amax[0] : amax[row]) * factor_inv;

            for (size_t pair_idx = 0; pair_idx < bytes_per_block; ++pair_idx) {
                const size_t byte_idx =
                    (row * Mread + block) * bytes_per_block + pair_idx;
                fp4e2m1x2 fp4_pair;
                std::memcpy(&fp4_pair, &packed_data[byte_idx], 1);
                const float2 values = cvt_fp4x2_to_float2(fp4_pair);

                const size_t col0 = block * BLOCK_SIZE + pair_idx * 2;
                output[row * cols + col0] =
                    static_cast<OType>(values.x * final_scale);
                output[row * cols + col0 + 1] =
                    static_cast<OType>(values.y * final_scale);
            }
        }
    }
}

template <typename OutputType>
float compute_amax(test::Tensor &t, size_t rows, size_t cols) {
    t.to_cpu();
    const auto *data = t.rowwise_cpu_dptr<OutputType>();
    float amax = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        amax = std::max(amax, std::abs(static_cast<float>(data[i])));
    }
    return amax;
}

struct NVFP4DequantizeTestConfig {
  NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled;
  int e4m3_max = 448;
};

// Quantize a high-precision input to NVFP4, then dequantize and compare
// against a CPU reference computed from the quantized data.
template <typename OutputType>
void performTest_dequantize_nvfp4(const size_t rows, const size_t cols,
                                  const bool row_scaled_nvfp4,
                                  const NVTENVFP44Over6Mode mode,
                                  const int e4m3_max) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    // Tensors
    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    Tensor quantized("quantized", std::vector<size_t>{rows, cols},
                     DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    Tensor output("output", std::vector<size_t>{rows, cols}, otype, true, false);

    // Fill input with random data
    fillCase<fp32>(&input, InputsFillCase::uniform);

    // Configure quantized tensor amax
    size_t amax_size = 1;
    quantized.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
      quantized.set_row_scaled_nvfp4(true);
      amax_size = rows;
    } else if (rows > 0 && cols > 0) {
      quantized.set_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
      quantized.set_amax(0.0f);
    }

    // Quantize
    if (rows > 0 && cols > 0) {
        QuantizationConfigWrapper quant_config;
        quant_config.set_nvfp4_4over6_mode(mode);
        nvte_quantize_v2(input.data(), quantized.data(), quant_config, 0);
        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    }

    // Dequantize
    nvte_dequantize(quantized.data(), output.data(), 0);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Nothing to be done if tensor is empty
    if (rows == 0 && cols == 0) {
      return;
    }

    // Dequantize reference implementation
    quantized.to_cpu();
    const uint8_t *fp4_data =
      reinterpret_cast<const uint8_t *>(quantized.rowwise_cpu_dptr<fp4e2m1>());
    const fp8e4m3 *scales = quantized.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
    const auto *amax = quantized.cpu_rowwise_amax_ptr<float>();
    const std::vector<float> amax_vals(amax, amax + amax_size);
    const NVTEShape scale_shape = quantized.rowwise_scale_inv_shape();
    const size_t scale_stride = scale_shape.data[scale_shape.ndim - 1];
    std::unique_ptr<OutputType[]> ref_output =
      std::make_unique<OutputType[]>(rows * cols);
    compute_ref_dequantize_nvfp4<OutputType>(
      fp4_data, scales, amax_vals, ref_output.get(),
      rows, cols, scale_stride, e4m3_max);

    // Compare results from TE and reference impls
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_nvfp4", output, ref_output.get(), true, atol, rtol);
}

// Dequantize NVFP4 with GEMM-swizzled scales and compare against compact path.
template <typename OutputType>
void performTest_dequantize_nvfp4_swizzled(const size_t rows, const size_t cols,
                                           const bool row_scaled_nvfp4,
                                           const NVTENVFP44Over6Mode mode,
                                           const int e4m3_max) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    Tensor quantized_compact("quantized_compact", std::vector<size_t>{rows, cols},
                             DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    quantized_compact.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized_compact.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
        quantized_compact.set_row_scaled_nvfp4(true);
    } else if (rows > 0 && cols > 0) {
        quantized_compact.set_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
        quantized_compact.set_amax(0.0f);
    }

    if (rows > 0 && cols > 0) {
        QuantizationConfigWrapper quant_config;
        quant_config.set_nvfp4_4over6_mode(mode);
        nvte_quantize_v2(input.data(), quantized_compact.data(), quant_config, 0);
        cudaDeviceSynchronize();
    }

    // Dequantize with compact scales to get the reference output.
    Tensor output_compact("output_compact", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized_compact.data(), output_compact.data(), 0);
    cudaDeviceSynchronize();

    // Create tensor with same FP4 data but swizzled scales
    Tensor quantized_swizzled("quantized_swizzled", std::vector<size_t>{rows, cols},
                              DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    quantized_swizzled.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized_swizzled.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
        quantized_swizzled.set_row_scaled_nvfp4(true);
    } else {
        quantized_swizzled.set_amax(0.0f);
    }
    quantized_swizzled.set_with_gemm_swizzled_scales(true);

    // Copy amax and scale from compact to swizzled before FP4 data,
    // since from_cpu() uploads all CPU buffers (including zero-init data).
    quantized_compact.to_cpu();
    if (row_scaled_nvfp4) {
        const auto *src = quantized_compact.cpu_rowwise_amax_ptr<float>();
        auto *dst = quantized_swizzled.cpu_rowwise_amax_ptr<float>();
        std::copy(src, src + rows, dst);
        quantized_swizzled.from_cpu();
    } else {
        quantized_swizzled.set_amax(quantized_compact.amax());
    }

    // Copy FP4 data after from_cpu() to avoid being overwritten
    const size_t data_bytes = rows * cols / 2;
    if (data_bytes > 0) {
        cudaMemcpy(quantized_swizzled.rowwise_dptr(), quantized_compact.rowwise_dptr(),
                   data_bytes, cudaMemcpyDeviceToDevice);
    }

    // Swizzle scales
    if (data_bytes > 0) {
        nvte_swizzle_scaling_factors(quantized_compact.data(), quantized_swizzled.data(), 0);
    }

    // Dequantize with swizzled scales
    Tensor output_swizzled("output_swizzled", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized_swizzled.data(), output_swizzled.data(), 0);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Read compact output as reference
    const size_t num_elems = rows * cols;
    std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(num_elems);
    if (num_elems > 0) {
        cudaMemcpy(ref_output.get(), output_compact.rowwise_dptr(),
                   num_elems * sizeof(OutputType), cudaMemcpyDeviceToHost);
    }

    auto [atol, rtol] = getTolerances(otype);
    if (num_elems > 0) {
        compareResults("output_nvfp4_swizzled", output_swizzled,
                       ref_output.get(), true, atol, rtol);
    }
}

std::vector<std::pair<size_t, size_t>> nvfp4_tensor_dims = {
    {0, 128},
    {0, 256},
    {32, 32},
    {32, 64},
    {64, 96},
    {128, 128},
    {128, 256},
    {256, 256},
    {256, 512},
    {512, 1024},
    {992, 512},
    {768, 1024},
};

}  // namespace

class DequantizeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                transformer_engine::DType,
                bool,
                NVFP4DequantizeTestConfig>> {};

TEST_P(DequantizeNVFP4TestSuite, TestDequantizeNVFP4)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const bool row_scaled_nvfp4 = std::get<2>(GetParam());
    const NVFP4DequantizeTestConfig config = std::get<3>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4<OutputType>(
            tensor_size.first, tensor_size.second, row_scaled_nvfp4, config.mode,
            config.e4m3_max);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Bool(),
        ::testing::Values(NVFP4DequantizeTestConfig{},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 448},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 256})),
    [](const testing::TestParamInfo<DequantizeNVFP4TestSuite::ParamType>& info)
    {
        const NVFP4DequantizeTestConfig config = std::get<3>(info.param);
        const bool use_4over6 = config.mode != kNVTENVFP44Over6Disabled;
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param)) + "X" +
                           (std::get<2>(info.param) ? "RowScaled" : "PerTensor") + "X" +
                           (use_4over6 ? "FourOverSix" : "Default") + "X" +
                           (config.e4m3_max == 256 ? "E4M3Max256" : "E4M3Max448");
        return name;
    }
);

class DequantizeNVFP4SwizzledTestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                transformer_engine::DType,
                bool,
                NVFP4DequantizeTestConfig>> {};

TEST_P(DequantizeNVFP4SwizzledTestSuite, TestDequantizeNVFP4Swizzled)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const bool row_scaled_nvfp4 = std::get<2>(GetParam());
    const NVFP4DequantizeTestConfig config = std::get<3>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4_swizzled<OutputType>(
            tensor_size.first, tensor_size.second, row_scaled_nvfp4, config.mode,
            config.e4m3_max);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4SwizzledTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Bool(),
        ::testing::Values(NVFP4DequantizeTestConfig{},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 448},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 256})),
    [](const testing::TestParamInfo<DequantizeNVFP4SwizzledTestSuite::ParamType>& info)
    {
        const NVFP4DequantizeTestConfig config = std::get<3>(info.param);
        const bool use_4over6 = config.mode != kNVTENVFP44Over6Disabled;
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param)) + "X" +
                           (std::get<2>(info.param) ? "RowScaled" : "PerTensor") + "X" +
                           (use_4over6 ? "FourOverSix" : "Default") + "X" +
                           (config.e4m3_max == 256 ? "E4M3Max256" : "E4M3Max448") + "X" +
                           "Swizzled";
        return name;
    }
);

#endif  // FP4_TYPE_SUPPORTED

#endif  // __HIP_PLATFORM_AMD__
