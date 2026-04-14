/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include <transformer_engine/transpose.h>

namespace {

constexpr int MXFP4_BLOCK_SIZE = 32;

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

static const float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

float decode_fp4(uint8_t nibble) { return E2M1_LUT[nibble & 0xF]; }

bool is_gfx950() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    return prop.major == 9 && prop.minor == 5;
}

struct MXFP4TestParams {
    int M;
    int N;
    bool use_hadamard;
};

class CastTransposeMXFP4Test : public ::testing::TestWithParam<MXFP4TestParams> {};

TEST_P(CastTransposeMXFP4Test, SmokeAndNonZero) {
    if (!is_gfx950()) GTEST_SKIP() << "Requires gfx950";

    const auto& p = GetParam();
    const int M = p.M;
    const int N = p.N;
    const bool use_hadamard = p.use_hadamard;

    const int rowwise_scale_N = cdiv(N, MXFP4_BLOCK_SIZE);
    const int rowwise_scale_M_pad = cdiv(M, 256) * 256;
    const int rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    const int colwise_scale_M = N;
    const int colwise_scale_N_val = cdiv(M, MXFP4_BLOCK_SIZE);
    const int colwise_scale_M_pad = cdiv(N, 256) * 256;
    const int colwise_scale_N_pad = cdiv(colwise_scale_N_val, 8) * 8;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<hip_bfloat16> h_input(M * N);
    for (auto& v : h_input) v = hip_bfloat16(dist(rng));

    void *d_input = nullptr, *d_row_fp4 = nullptr, *d_row_scale = nullptr;
    void *d_col_fp4 = nullptr, *d_col_scale = nullptr;

    ASSERT_EQ(hipMalloc(&d_input, M * N * sizeof(hip_bfloat16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_row_fp4, M * N / 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_row_scale, rowwise_scale_M_pad * rowwise_scale_N_pad), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_col_fp4, N * M / 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_col_scale, colwise_scale_M_pad * colwise_scale_N_pad), hipSuccess);

    ASSERT_EQ(hipMemset(d_row_fp4, 0, M * N / 2), hipSuccess);
    ASSERT_EQ(hipMemset(d_col_fp4, 0, N * M / 2), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_input, h_input.data(), M * N * sizeof(hip_bfloat16),
                         hipMemcpyHostToDevice), hipSuccess);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    nvte_cast_transpose_mxfp4_fused_shuffle(
        d_input,
        d_row_fp4, d_row_scale,
        d_col_fp4, d_col_scale,
        M, N,
        true, true,
        false, use_hadamard,
        false, false,
        rowwise_scale_N_pad,
        colwise_scale_N_pad,
        rowwise_scale_N, rowwise_scale_M_pad, rowwise_scale_N_pad,
        colwise_scale_M, colwise_scale_N_val,
        colwise_scale_M_pad, colwise_scale_N_pad,
        stream);

    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipGetLastError(), hipSuccess);

    std::vector<uint8_t> h_row_fp4(M * N / 2);
    std::vector<uint8_t> h_col_fp4(N * M / 2);
    ASSERT_EQ(hipMemcpy(h_row_fp4.data(), d_row_fp4, M * N / 2, hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(h_col_fp4.data(), d_col_fp4, N * M / 2, hipMemcpyDeviceToHost), hipSuccess);

    int nonzero_row = 0, nonzero_col = 0;
    for (auto b : h_row_fp4) { if (b != 0) nonzero_row++; }
    for (auto b : h_col_fp4) { if (b != 0) nonzero_col++; }

    EXPECT_GT(nonzero_row, 0) << "Rowwise FP4 output is all zeros";
    EXPECT_GT(nonzero_col, 0) << "Colwise FP4 output is all zeros";

    hipStreamDestroy(stream);
    hipFree(d_input);
    hipFree(d_row_fp4);
    hipFree(d_row_scale);
    hipFree(d_col_fp4);
    hipFree(d_col_scale);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastTransposeMXFP4Test,
    ::testing::Values(
        MXFP4TestParams{128, 64, false},
        MXFP4TestParams{256, 256, false},
        MXFP4TestParams{512, 512, false},
        MXFP4TestParams{1024, 1024, false},
        MXFP4TestParams{128, 64, true},
        MXFP4TestParams{256, 256, true},
        MXFP4TestParams{512, 512, true}
    ),
    [](const ::testing::TestParamInfo<MXFP4TestParams>& info) {
        return std::to_string(info.param.M) + "x" + std::to_string(info.param.N) +
               (info.param.use_hadamard ? "_hadamard" : "_plain");
    });

}  // namespace
