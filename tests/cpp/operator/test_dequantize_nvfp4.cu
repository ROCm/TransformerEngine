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
constexpr size_t kFP4BlockSize2DY = 16;
constexpr size_t kFP4BlockSize2DX = 16;

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

// Generate compact 2D scales over 16x16 tiles, then replicate them row-wise
// into the physical scale layout expected by the existing 1D dequant kernel.
//
// replicated[row][block_x] = compact_2d[row / 16][block_x]
void generate_2d_scales_with_replication(fp8e4m3* scale_buffer,
                                         const size_t rows,
                                         const size_t cols,
                                         const size_t unpadded_blocks_Y,
                                         const size_t unpadded_blocks_X,
                                         const size_t scales_stride,
                                         std::mt19937& gen,
                                         std::uniform_int_distribution<int>& finite_nonneg_e4m3_dis) {
    const size_t total_elems = unpadded_blocks_Y * scales_stride;
    std::memset(scale_buffer, 0, total_elems * sizeof(fp8e4m3));

    const size_t blocks_y = divide_round_up(rows, kFP4BlockSize2DY);
    const size_t blocks_x = divide_round_up(cols, kFP4BlockSize2DX);

    std::vector<fp8e4m3> compact_2d(blocks_y * blocks_x);

    for (size_t by = 0; by < blocks_y; ++by) {
        for (size_t bx = 0; bx < blocks_x; ++bx) {
            const size_t compact_idx = by * blocks_x + bx;
            const uint8_t scale = static_cast<uint8_t>(finite_nonneg_e4m3_dis(gen));
            std::memcpy(&compact_2d[compact_idx], &scale, sizeof(scale));
        }
    }

    for (size_t row = 0; row < unpadded_blocks_Y; ++row) {
        const size_t by = row / kFP4BlockSize2DY;
        for (size_t bx = 0; bx < unpadded_blocks_X; ++bx) {
            const size_t scale_idx = row * scales_stride + bx;
            scale_buffer[scale_idx] = compact_2d[by * blocks_x + bx];
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

// Populate FP4 (E2M1) tensor using packed 4-bit encoding, and simultaneously
// populate its mathematical transpose in packed storage.
//
// data   has mathematical shape [rows, cols]
// data_t has mathematical shape [cols, rows]
void generate_data_and_transpose(fp4e2m1* data,
                                 fp4e2m1* data_t,
                                 const size_t rows,
                                 const size_t cols,
                                 std::mt19937& gen,
                                 std::uniform_int_distribution<int>& e2m1_dis) {
    const size_t packed_bytes = (rows * cols * BitsNumber<fp4e2m1>::num_bits) / 8;

    std::memset(data, 0, packed_bytes);
    std::memset(data_t, 0, packed_bytes);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const uint8_t nibble = static_cast<uint8_t>(e2m1_dis(gen)) & 0xF;

            const size_t idx = i * cols + j;
            set_fp4_nibble(data, idx, nibble);

            const size_t idx_t = j * rows + i;
            set_fp4_nibble(data_t, idx_t, nibble);
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
    constexpr float factor_inv = 1.0f / (6.0f * 448.0f);

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
                     const fp4e2m1* host_input,
                     const fp8e4m3* host_scales,
                     const size_t rows,
                     const size_t cols,
                     const size_t blocks_y,
                     const size_t blocks_x,
                     const size_t scale_stride,
                     const float amax,
                     DType otype) {
    const DType itype = DType::kFloat4E2M1;

    Tensor input(case_name + "_input", std::vector<size_t>{rows, cols}, itype,
                 true, false, NVTE_NVFP4_1D_SCALING);
    Tensor output(case_name + "_output", std::vector<size_t>{rows, cols}, otype, true, false);

    std::unique_ptr<OutputType[]> ref_output =
        std::make_unique<OutputType[]>(rows * cols);

    const size_t data_bytes = (rows * cols * BitsNumber<fp4e2m1>::num_bits) / 8;
    const size_t scale_bytes = blocks_y * blocks_x * sizeof(fp8e4m3);

    auto err = cudaMemcpy(input.rowwise_dptr(),
                          host_input,
                          data_bytes,
                          cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << case_name << ": " << cudaGetErrorString(err);

    err = cudaMemcpy(input.rowwise_scale_inv_dptr(),
                     host_scales,
                     scale_bytes,
                     cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << case_name << ": " << cudaGetErrorString(err);

    nvte_dequantize(input.data(), output.data(), 0);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << case_name << ": " << cudaGetErrorString(err);

    output.to_cpu();

    compute_ref(host_input,
                ref_output.get(),
                host_scales,
                amax,
                rows,
                cols,
                scale_stride);

    auto [atol, rtol] = getTolerances(otype);
    compareResults(case_name, output, ref_output.get(), true, atol, rtol);
}

// End-to-end test: generate random FP4 input and FP8 scales, then exercise
// 1) row-wise 1D dequant
// 2) col-wise 1D dequant (by running the same dequant kernel on transposed data)
// 3) 2D dequant semantics using row-wise replicated scales
template <typename OutputType>
void performTest(const size_t rows, const size_t cols, DType otype) {
    const std::array<size_t, 4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);
    const std::array<size_t, 4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t blocks_Y_t = scale_dims_t[2];
    const size_t blocks_X_t = scale_dims_t[3];
    const size_t scales_stride_t = blocks_X_t;

    std::unique_ptr<fp4e2m1[]> host_input =
        std::make_unique<fp4e2m1[]>(rows * cols);

    std::unique_ptr<fp4e2m1[]> host_input_t =
        std::make_unique<fp4e2m1[]>(rows * cols);

    std::unique_ptr<fp8e4m3[]> host_scales_rowwise_1d =
        std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);

    std::unique_ptr<fp8e4m3[]> host_scales_colwise_1d =
        std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);

    std::unique_ptr<fp8e4m3[]> host_scales_2d_replicated =
        std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);

    static std::mt19937 gen(42);
    std::uniform_int_distribution<int> e2m1_dis(0, 15);
    std::uniform_int_distribution<int> finite_nonneg_e4m3_dis(0, 126);

    generate_data_and_transpose(host_input.get(),
                                host_input_t.get(),
                                rows,
                                cols,
                                gen,
                                e2m1_dis);

    // Row-wise 1D scales on [rows, cols]
    generate_1d_scales(host_scales_rowwise_1d.get(),
                       unpadded_blocks_Y,
                       unpadded_blocks_X,
                       scales_stride,
                       gen,
                       finite_nonneg_e4m3_dis);

    // Col-wise 1D scales on [cols, rows]
    generate_1d_scales(host_scales_colwise_1d.get(),
                       unpadded_blocks_Y_t,
                       unpadded_blocks_X_t,
                       scales_stride_t,
                       gen,
                       finite_nonneg_e4m3_dis);

    // 2D scales replicated row-wise
    generate_2d_scales_with_replication(host_scales_2d_replicated.get(),
                                        rows,
                                        cols,
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
                                host_input.get(),
                                host_scales_rowwise_1d.get(),
                                rows,
                                cols,
                                blocks_Y,
                                blocks_X,
                                scales_stride,
                                amax,
                                otype);

    run_single_case<OutputType>("colwise_1d_dequant",
                                host_input_t.get(),
                                host_scales_colwise_1d.get(),
                                cols,
                                rows,
                                blocks_Y_t,
                                blocks_X_t,
                                scales_stride_t,
                                amax,
                                otype);

    run_single_case<OutputType>("replicated_2d_dequant",
                                host_input.get(),
                                host_scales_2d_replicated.get(),
                                rows,
                                cols,
                                blocks_Y,
                                blocks_X,
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
