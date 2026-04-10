/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace test;

namespace {

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

static constexpr float E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// ---------------------------------------------------------------------------
// CPU reference helpers (ported from test_cast_mxfp4.py / kernel logic)
// ---------------------------------------------------------------------------

uint8_t compute_e8m0_ref(float amax, float &native_scale) {
    if (amax == 0.0f) { native_scale = 1.0f; return 127; }
    uint32_t bits;
    std::memcpy(&bits, &amax, sizeof(bits));
    bits = (bits + 0x200000u) & 0xFF800000u;
    int exp = ((bits >> 23) & 0xFF) - 127;
    int s = std::max(-127, std::min(127, exp - 2));
    uint32_t sb = static_cast<uint32_t>(127 + s) << 23;
    std::memcpy(&native_scale, &sb, sizeof(native_scale));
    return static_cast<uint8_t>(s + 127);
}

uint8_t encode_fp4(float val) {
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float a = std::abs(val);
    uint8_t idx = 0;
    if (a >= 0.25f) idx = 1;
    if (a >= 0.75f) idx = 2;
    if (a >= 1.25f) idx = 3;
    if (a >= 1.75f) idx = 4;
    if (a >= 2.5f)  idx = 5;
    if (a >= 3.5f)  idx = 6;
    if (a >= 5.0f)  idx = 7;
    return (sign << 3) | idx;
}

// ---------------------------------------------------------------------------
// Shuffle index functions (same formulas as the kernel)
// ---------------------------------------------------------------------------

int shuffle_scale_index(int row, int col, int scale_n_pad) {
    int i0 = row >> 5;
    int i1 = (row >> 4) & 1;
    int i2 = row & 15;
    int i3 = col >> 3;
    int i4 = (col >> 2) & 1;
    int i5 = col & 3;
    return (i0 * (scale_n_pad >> 3) << 8) + (i3 << 8) + (i5 << 6) +
           (i2 << 2) + (i4 << 1) + i1;
}

int shuffled_fp4_index(int row, int col, int K_packed) {
    int N_block = row >> 4;
    int row_in_block = row & 15;
    int K_block = col >> 5;
    int col_in_block = col & 31;
    int sub_block = col_in_block >> 4;
    int k_elem = col_in_block & 15;
    return N_block * (K_packed << 4) + K_block * 512 +
           sub_block * 256 + row_in_block * 16 + k_elem;
}

void unshuffle_scales(const uint8_t *shuffled, uint8_t *out,
                      int rows, int num_blocks, int /*M_pad*/, int N_pad) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < num_blocks; c++)
            out[r * num_blocks + c] = shuffled[shuffle_scale_index(r, c, N_pad)];
}

void unshuffle_fp4(const uint8_t *shuffled, uint8_t *linear, int rows, int K_packed) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < K_packed; c += 2) {
            int idx = shuffled_fp4_index(r, c, K_packed);
            linear[r * K_packed + c]     = shuffled[idx];
            linear[r * K_packed + c + 1] = shuffled[idx + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// 16-point Hadamard (simulates 4 threads x 4 values with cross-lane swizzle)
// ---------------------------------------------------------------------------

void hadamard16_ref(float vals[16]) {
    for (int t = 0; t < 4; t++) {
        float *v = &vals[t * 4];
        float a0 = v[0] + v[1], a1 = v[0] - v[1];
        float a2 = v[2] + v[3], a3 = v[2] - v[3];
        v[0] = a0 + a2; v[2] = a0 - a2;
        v[1] = a1 + a3; v[3] = a1 - a3;
    }
    float tmp[16];
    std::memcpy(tmp, vals, sizeof(tmp));
    for (int t = 0; t < 4; t++) {
        int p = t ^ 1;
        bool neg = (t & 1);
        for (int k = 0; k < 4; k++)
            vals[t * 4 + k] = neg ? (tmp[p * 4 + k] - tmp[t * 4 + k])
                                  : (tmp[p * 4 + k] + tmp[t * 4 + k]);
    }
    std::memcpy(tmp, vals, sizeof(tmp));
    for (int t = 0; t < 4; t++) {
        int p = t ^ 2;
        bool neg = (t >> 1) & 1;
        for (int k = 0; k < 4; k++)
            vals[t * 4 + k] = neg ? (tmp[p * 4 + k] - tmp[t * 4 + k])
                                  : (tmp[p * 4 + k] + tmp[t * 4 + k]);
    }
    for (int i = 0; i < 16; i++) vals[i] *= 0.25f;
}

// ---------------------------------------------------------------------------
// CPU reference quantize / dequantize
// ---------------------------------------------------------------------------

void mxfp4_quantize_row(const bf16 *input, int N, bool use_hadamard,
                        uint8_t *fp4_out, uint8_t *scale_out) {
    constexpr int BLK = 32;
    int num_blocks = cdiv(N, BLK);

    for (int b = 0; b < num_blocks; b++) {
        float block[BLK] = {};
        for (int i = 0; i < BLK && b * BLK + i < N; i++)
            block[i] = static_cast<float>(input[b * BLK + i]);

        if (use_hadamard) {
            hadamard16_ref(&block[0]);
            hadamard16_ref(&block[16]);
        }

        float amax = 0.0f;
        for (int i = 0; i < BLK; i++)
            amax = std::max(amax, std::abs(block[i]));

        float native_scale;
        scale_out[b] = compute_e8m0_ref(amax, native_scale);

        for (int i = 0; i < BLK; i += 2) {
            uint8_t lo = encode_fp4(block[i] / native_scale);
            uint8_t hi = encode_fp4(block[i + 1] / native_scale);
            fp4_out[(b * BLK + i) / 2] = lo | (hi << 4);
        }
    }
}

void mxfp4_quantize_ref(const bf16 *input, int M, int N, bool use_hadamard,
                        uint8_t *fp4_out, uint8_t *scale_out) {
    int K_packed = N / 2;
    int num_blocks = cdiv(N, 32);
    for (int r = 0; r < M; r++)
        mxfp4_quantize_row(&input[r * N], N, use_hadamard,
                           &fp4_out[r * K_packed], &scale_out[r * num_blocks]);
}

void mxfp4_dequantize(const uint8_t *fp4, const uint8_t *scales,
                      float *output, int rows, int cols) {
    int num_blocks = cdiv(cols, 32);
    int K_packed = cols / 2;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c += 2) {
            uint8_t e8m0 = scales[r * num_blocks + c / 32];
            float sv = std::pow(2.0f, static_cast<float>(e8m0) - 127.0f);
            uint8_t packed = fp4[r * K_packed + c / 2];
            output[r * cols + c]     = E2M1_LUT[packed & 0xF] * sv;
            output[r * cols + c + 1] = E2M1_LUT[(packed >> 4) & 0xF] * sv;
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

void compare_e8m0(const std::string &name,
                  const uint8_t *test, const uint8_t *ref, int count,
                  int max_diff = 1) {
    int bad = 0;
    for (int i = 0; i < count; i++) {
        int d = std::abs(static_cast<int>(test[i]) - static_cast<int>(ref[i]));
        if (d > max_diff) {
            if (bad < 10)
                std::cout << name << " scale mismatch [" << i
                          << "]: got=" << (int)test[i]
                          << " ref=" << (int)ref[i] << std::endl;
            bad++;
        }
    }
    ASSERT_EQ(bad, 0) << name << ": " << bad << " scale outliers (tol=" << max_diff << ")";
}

void compare_deq(const std::string &name,
                 const float *test, const float *ref, int count,
                 float atol = 0.05f, float rtol = 0.1f) {
    int bad = 0;
    for (int i = 0; i < count; i++) {
        float ad = std::abs(test[i] - ref[i]);
        if (ad > atol && (ref[i] == 0.0f || ad / std::abs(ref[i]) > rtol)) {
            if (bad < 10)
                std::cout << name << " mismatch [" << i
                          << "]: got=" << test[i]
                          << " ref=" << ref[i] << std::endl;
            bad++;
        }
    }
    ASSERT_EQ(bad, 0) << name << ": " << bad << "/" << count << " mismatches";
}

// ---------------------------------------------------------------------------
// Hardware check
// ---------------------------------------------------------------------------

bool is_gfx950() {
#ifdef __HIP_PLATFORM_AMD__
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major == 9 && prop.minor == 5;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Test body
// ---------------------------------------------------------------------------

void performTest(int M, int N, bool use_hadamard, bool shuffle_fp4) {
    if (!is_gfx950()) GTEST_SKIP() << "Requires gfx950";

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 1.0f);
    std::vector<bf16> h_input(M * N);
    for (auto &v : h_input) v = static_cast<bf16>(dist(gen));

    // Scale dimensions (mirrors mxfp4_hip.cpp)
    int rsc_N   = cdiv(N, 32);
    int rsc_Mpad = cdiv(M, 256) * 256;
    int rsc_Npad = cdiv(rsc_N, 8) * 8;

    int csc_M    = N;
    int csc_N    = cdiv(M, 32);
    int csc_Mpad = cdiv(N, 256) * 256;
    int csc_Npad = cdiv(csc_N, 8) * 8;

    int K_packed = N / 2;
    int M_packed = M / 2;

    size_t in_bytes     = M * N * sizeof(bf16);
    size_t rfp4_bytes   = M * K_packed;
    size_t rscale_bytes = rsc_Mpad * rsc_Npad;
    size_t cfp4_bytes   = N * M_packed;
    size_t cscale_bytes = csc_Mpad * csc_Npad;

    void *d_in, *d_rfp4, *d_rsc, *d_cfp4, *d_csc;
    cudaMalloc(&d_in,   in_bytes);
    cudaMalloc(&d_rfp4, rfp4_bytes);
    cudaMalloc(&d_rsc,  rscale_bytes);
    cudaMalloc(&d_cfp4, cfp4_bytes);
    cudaMalloc(&d_csc,  cscale_bytes);
    cudaMemset(d_rsc, 0, rscale_bytes);
    cudaMemset(d_csc, 0, cscale_bytes);
    cudaMemcpy(d_in, h_input.data(), in_bytes, cudaMemcpyHostToDevice);

    nvte_cast_transpose_mxfp4_fused_shuffle(
        d_in,
        d_rfp4, d_rsc,
        d_cfp4, d_csc,
        M, N,
        /*use_rowwise=*/true, /*use_colwise=*/true,
        /*shuffle_scales=*/true, use_hadamard,
        /*shuffle_rowwise_fp4=*/shuffle_fp4,
        /*shuffle_colwise_fp4=*/shuffle_fp4,
        rsc_Npad, csc_Npad,
        rsc_N, rsc_Mpad, rsc_Npad,
        csc_M, csc_N, csc_Mpad, csc_Npad,
        0);

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<uint8_t> h_rfp4(rfp4_bytes), h_rsc(rscale_bytes);
    std::vector<uint8_t> h_cfp4(cfp4_bytes), h_csc(cscale_bytes);
    cudaMemcpy(h_rfp4.data(), d_rfp4, rfp4_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rsc.data(),  d_rsc,  rscale_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cfp4.data(), d_cfp4, cfp4_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csc.data(),  d_csc,  cscale_bytes, cudaMemcpyDeviceToHost);

    // Unshuffle scales
    std::vector<uint8_t> rsc_log(M * rsc_N), csc_log(N * csc_N);
    unshuffle_scales(h_rsc.data(), rsc_log.data(), M, rsc_N, rsc_Mpad, rsc_Npad);
    unshuffle_scales(h_csc.data(), csc_log.data(), N, csc_N, csc_Mpad, csc_Npad);

    // Unshuffle FP4 if needed
    const uint8_t *rfp4_ptr = h_rfp4.data();
    const uint8_t *cfp4_ptr = h_cfp4.data();
    std::vector<uint8_t> rfp4_lin, cfp4_lin;
    if (shuffle_fp4) {
        rfp4_lin.resize(rfp4_bytes);
        unshuffle_fp4(h_rfp4.data(), rfp4_lin.data(), M, K_packed);
        rfp4_ptr = rfp4_lin.data();
        cfp4_lin.resize(cfp4_bytes);
        unshuffle_fp4(h_cfp4.data(), cfp4_lin.data(), N, M_packed);
        cfp4_ptr = cfp4_lin.data();
    }

    // CPU reference — rowwise
    std::vector<uint8_t> ref_rfp4(rfp4_bytes), ref_rsc(M * rsc_N);
    mxfp4_quantize_ref(h_input.data(), M, N, use_hadamard,
                       ref_rfp4.data(), ref_rsc.data());

    // CPU reference — colwise (transpose input, then quantize as N x M)
    std::vector<bf16> h_input_t(M * N);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            h_input_t[c * M + r] = h_input[r * N + c];

    std::vector<uint8_t> ref_cfp4(cfp4_bytes), ref_csc(N * csc_N);
    mxfp4_quantize_ref(h_input_t.data(), N, M, use_hadamard,
                       ref_cfp4.data(), ref_csc.data());

    // Compare E8M0 scales (±1 tolerance)
    compare_e8m0("rowwise_scales", rsc_log.data(), ref_rsc.data(), M * rsc_N);
    compare_e8m0("colwise_scales", csc_log.data(), ref_csc.data(), N * csc_N);

    // Dequantize and compare
    std::vector<float> dq_kern_r(M * N), dq_ref_r(M * N);
    mxfp4_dequantize(rfp4_ptr,         rsc_log.data(), dq_kern_r.data(), M, N);
    mxfp4_dequantize(ref_rfp4.data(),  ref_rsc.data(), dq_ref_r.data(),  M, N);
    compare_deq("rowwise", dq_kern_r.data(), dq_ref_r.data(), M * N);

    std::vector<float> dq_kern_c(N * M), dq_ref_c(N * M);
    mxfp4_dequantize(cfp4_ptr,         csc_log.data(), dq_kern_c.data(), N, M);
    mxfp4_dequantize(ref_cfp4.data(),  ref_csc.data(), dq_ref_c.data(),  N, M);
    compare_deq("colwise", dq_kern_c.data(), dq_ref_c.data(), N * M);

    cudaFree(d_in);
    cudaFree(d_rfp4);
    cudaFree(d_rsc);
    cudaFree(d_cfp4);
    cudaFree(d_csc);
}

std::vector<std::pair<int, int>> test_sizes = {
    {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512},
    {128, 256}, {256, 128}, {1024, 1024}, {2048, 2048},
};

}  // namespace

class CastTransposeMXFP4TestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::pair<int, int>, bool, bool>> {};

TEST_P(CastTransposeMXFP4TestSuite, TestCastTransposeMXFP4) {
    auto [size, use_hadamard, shuffle_fp4] = GetParam();
    if (shuffle_fp4) {
        if (size.first % 16 != 0 || (size.second / 2) % 32 != 0 ||
            size.second % 16 != 0 || (size.first / 2) % 32 != 0)
            GTEST_SKIP() << "Shape doesn't meet shuffle alignment";
    }
    performTest(size.first, size.second, use_hadamard, shuffle_fp4);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastTransposeMXFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(test_sizes),
        ::testing::Values(false, true),
        ::testing::Values(false, true)),
    [](const testing::TestParamInfo<CastTransposeMXFP4TestSuite::ParamType>
           &info) {
        auto sz = std::get<0>(info.param);
        std::string name = std::to_string(sz.first) + "x" +
                           std::to_string(sz.second);
        name += std::get<1>(info.param) ? "_had" : "_nohad";
        name += std::get<2>(info.param) ? "_shuf" : "_lin";
        return name;
    });
