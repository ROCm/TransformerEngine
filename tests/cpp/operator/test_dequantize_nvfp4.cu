/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

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
    const std::array<size_t, 4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);

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
