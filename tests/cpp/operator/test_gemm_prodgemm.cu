/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*
 * MXFP8 GEMM correctness tests for production LLM shapes.
 *
 * Tests forward, dgrad, and wgrad passes with appropriate FP8 type combos:
 *   Forward:  E4M3 x E4M3 -> BF16
 *   Dgrad:    E5M2 x E4M3 -> BF16
 *   Wgrad:    E4M3 x E5M2 -> BF16
 *
 * Each shape is tested with 3 transpose configs (TN, NN, NT) and
 * 3 micro-batch sizes (MBS = 1, 2, 4 -> tokens = 4096, 8192, 16384).
 */

#ifdef __HIP_PLATFORM_AMD__

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

using fp32 = float;
using fp8  = fp8e4m3;
using bf8  = fp8e5m2;

using TShape = std::vector<size_t>;
using Layout = std::pair<bool, bool>;  // {transa, transb}

static const Layout kTN{true, false};
static const Layout kNN{false, false};
static const Layout kNT{false, true};
static const std::vector<Layout> kLayouts = {kTN, kNN, kNT};

// ============================================================================
// GemmPass: determines A/B FP8 type combination
//   FWD:   fp8  x fp8  (E4M3 x E4M3)
//   DGRAD: bf8  x fp8  (E5M2 x E4M3)
//   WGRAD: fp8  x bf8  (E4M3 x E5M2)
// ============================================================================

enum class GemmPass { FWD, DGRAD, WGRAD };

// ============================================================================
// Shape definition: describes a GEMM from the model architecture.
//
//   Forward / Dgrad: M = tokens, dim1 = N, dim2 = K
//   Wgrad:           K = tokens, dim1 = M, dim2 = N
// ============================================================================

struct ShapeDef {
    const char* label;
    size_t dim1;
    size_t dim2;
    GemmPass pass;
};

// LLM1 (hidden=7168, MLA, seq=4096)

static const ShapeDef llm1_shapes[] = {
    // Forward (M=tokens, N, K)
    {"LLM1_Linear0_fwd",           1536,  7168, GemmPass::FWD},
    {"LLM1_Linear1_fwd",            576,  7168, GemmPass::FWD},
    {"LLM1_LNLinear0_fwd",        24576,  1536, GemmPass::FWD},
    {"LLM1_LNLinear1_fwd",        32768,   512, GemmPass::FWD},
    {"LLM1_Linear_attn_fwd",       7168, 16384, GemmPass::FWD},
    {"LLM1_LNMLP_gateup_fwd",     36864,  7168, GemmPass::FWD},
    {"LLM1_LNMLP_down_fwd",        7168, 18432, GemmPass::FWD},
    {"LLM1_SharedExp_gu_fwd",      4096,  7168, GemmPass::FWD},
    {"LLM1_SharedExp_dn_fwd",      7168,  2048, GemmPass::FWD},
    {"LLM1_TopKRouter_fwd",         256,  7168, GemmPass::FWD},
    // Dgrad (M=tokens, N, K)
    {"LLM1_attn_dgrad",           16384,  7168, GemmPass::DGRAD},
    {"LLM1_LNLinear1_dgrad",        512, 32768, GemmPass::DGRAD},
    {"LLM1_LNLinear0_dgrad",       1536, 24576, GemmPass::DGRAD},
    {"LLM1_SharedExp_dn_dgrad",    2048,  7168, GemmPass::DGRAD},
    {"LLM1_SharedExp_gu_dgrad",    7168,  4096, GemmPass::DGRAD},
    {"LLM1_TopKRouter_dgrad",      7168,   256, GemmPass::DGRAD},
    {"LLM1_MLP_post_dgrad",        7168, 14336, GemmPass::DGRAD},
    // Wgrad (M, N, K=tokens)
    {"LLM1_attn_wgrad",           16384,  7168, GemmPass::WGRAD},
    {"LLM1_LNLinear1_wgrad",        512, 32768, GemmPass::WGRAD},
    {"LLM1_LNLinear0_wgrad",       1536, 24576, GemmPass::WGRAD},
    {"LLM1_SharedExp_dn_wgrad",    2048,  7168, GemmPass::WGRAD},
    {"LLM1_SharedExp_gu_wgrad",    7168,  4096, GemmPass::WGRAD},
    {"LLM1_TopKRouter_wgrad",      7168,   256, GemmPass::WGRAD},
};

// LLM1 LM Head (large N, memory-intensive)
static const ShapeDef llm1_lm_head_shapes[] = {
    {"LLM1_LMHead_fwd",          129280,  7168, GemmPass::FWD},
    {"LLM1_LMHead_dgrad",          7168,129280, GemmPass::DGRAD},
    {"LLM1_LMHead_wgrad",          7168,129280, GemmPass::WGRAD},
};

// LLM2 (hidden=4096, GQA, seq=4096)

static const ShapeDef llm2_shapes[] = {
    // Forward (M=tokens, N, K)
    {"LLM2_LNLinear_QKV_fwd",   9216,  4096, GemmPass::FWD},
    {"LLM2_Linear_attn_fwd",    4096,  8192, GemmPass::FWD},
    {"LLM2_Router_fwd",          128,  4096, GemmPass::FWD},
    // Dgrad (M=tokens, N, K)
    {"LLM2_Router_dgrad",       4096,   128, GemmPass::DGRAD},
    {"LLM2_Linear_attn_dgrad",  8192,  4096, GemmPass::DGRAD},
    {"LLM2_LNLinear_dgrad",     4096,  9216, GemmPass::DGRAD},
    // Wgrad (M, N, K=tokens)
    {"LLM2_Router_wgrad",       4096,   128, GemmPass::WGRAD},
    {"LLM2_Linear_attn_wgrad",  8192,  4096, GemmPass::WGRAD},
    {"LLM2_LNLinear_wgrad",     4096,  9216, GemmPass::WGRAD},
};

// LLM2 LM Head (large N, memory-intensive)
static const ShapeDef llm2_lm_head_shapes[] = {
    {"LLM2_LMHead_fwd",        151936,  4096, GemmPass::FWD},
    {"LLM2_LMHead_dgrad",        4096,151936, GemmPass::DGRAD},
    {"LLM2_LMHead_wgrad",        4096,151936, GemmPass::WGRAD},
};

// ============================================================================
// Test case: a concrete (M, K, N) shape with pass info, ready for execution
// ============================================================================

struct ProdGemmTestCase {
    std::string label;
    size_t m, k, n;
    GemmPass pass;
};

std::ostream& operator<<(std::ostream& os, const ProdGemmTestCase& tc) {
    return os << tc.label;
}

static std::vector<ProdGemmTestCase> expand_shapes(const ShapeDef* defs, size_t count) {
    std::vector<ProdGemmTestCase> cases;
    for (size_t i = 0; i < count; ++i) {
        const auto& s = defs[i];
        for (size_t mbs : {1, 2, 4}) {
            size_t tokens = mbs * 4096;
            ProdGemmTestCase tc;
            tc.label = std::string(s.label) + "_mbs" + std::to_string(mbs);
            tc.pass = s.pass;
            switch (s.pass) {
                case GemmPass::FWD:
                case GemmPass::DGRAD:
                    tc.m = tokens;
                    tc.n = s.dim1;
                    tc.k = s.dim2;
                    break;
                case GemmPass::WGRAD:
                    tc.m = s.dim1;
                    tc.n = s.dim2;
                    tc.k = tokens;
                    break;
            }
            cases.push_back(std::move(tc));
        }
    }
    return cases;
}

static std::vector<ProdGemmTestCase> generate_model_test_cases() {
    auto v1   = expand_shapes(llm1_shapes,   std::size(llm1_shapes));
    auto v2   = expand_shapes(llm2_shapes,   std::size(llm2_shapes));
    v1.insert(v1.end(), std::make_move_iterator(v2.begin()),
                        std::make_move_iterator(v2.end()));
    return v1;
}

static std::vector<ProdGemmTestCase> generate_lm_head_test_cases() {
    auto v1   = expand_shapes(llm1_lm_head_shapes,   std::size(llm1_lm_head_shapes));
    auto v2   = expand_shapes(llm2_lm_head_shapes,   std::size(llm2_lm_head_shapes));
    v1.insert(v1.end(), std::make_move_iterator(v2.begin()),
                        std::make_move_iterator(v2.end()));
    return v1;
}

// ============================================================================
// Swizzle helper for gfx1250 MXFP8 scales (same as test_cublaslt_gemm.cu)
// ============================================================================

static void swizzle_mxfp8_scales(test::Tensor& t, bool rowwise) {
    void* scale_ptr = rowwise ? t.rowwise_scale_inv_dptr()
                              : t.columnwise_scale_inv_dptr();
    if (!scale_ptr) return;

    const NVTEShape scale_shape = rowwise ? t.rowwise_scale_inv_shape()
                                          : t.columnwise_scale_inv_shape();
    const NVTEShape data_shape  = rowwise ? t.rowwise_shape()
                                          : t.columnwise_shape();

    size_t num_scales = 1;
    for (size_t d = 0; d < scale_shape.ndim; d++) num_scales *= scale_shape.data[d];

    uint8_t* d_tmp = nullptr;
    NVTE_CHECK_CUDA(cudaMalloc(&d_tmp, num_scales));

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
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    NVTE_CHECK_CUDA(cudaMemcpy(scale_ptr, d_tmp, num_scales, cudaMemcpyDeviceToDevice));
    NVTE_CHECK_CUDA(cudaFree(d_tmp));
}

// ============================================================================
// MXFP8 dequantize-based GEMM correctness test
//
// 1. Create random source matrices A_src, B_src in D_Type (bf16)
// 2. Quantize:   A_src -> A_fp8, B_src -> B_fp8  (MXFP8 block scaling)
// 3. Dequantize: A_fp8 -> A_ref, B_fp8 -> B_ref  (back to D_Type)
// 4. Swizzle scales for gfx1250 (if needed)
// 5. MXFP8 GEMM:     D     = A_fp8 * B_fp8
// 6. Non-FP8 GEMM:   D_ref = A_ref * B_ref
// 7. Compare D vs D_ref
// ============================================================================

template <typename A_Type, typename B_Type, typename D_Type>
void performMxfp8DqTest(size_t m, size_t k, size_t n, bool transa, bool transb) {
    DType atype = TypeInfo<A_Type>::dtype;
    DType btype = TypeInfo<B_Type>::dtype;
    DType dtype = TypeInfo<D_Type>::dtype;

    ASSERT_TRUE(isFp8Type(atype) && isFp8Type(btype)) << "FP8/BF8 input types expected";
    ASSERT_FALSE(isFp8Type(dtype)) << "Non-FP8 output type expected";

    if (m % 16 || n % 16) {
        GTEST_SKIP() << "MXFP8 requires M & N to be multiples of 16";
    }
    if (k % 128) {
        GTEST_SKIP() << "MXFP8 requires K to be a multiple of 128";
    }

    cudaDeviceProp prop;
    (void)cudaGetDeviceProperties(&prop, 0);

    bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5) || prop.major >= 12;
    if (!mxfp8_supported) {
        GTEST_SKIP() << "MXFP8 is not supported on this GPU";
    }

    TShape a_shape = transa ? TShape{m, k} : TShape{k, m};
    TShape b_shape = transb ? TShape{k, n} : TShape{n, k};

    // 1. Create random source matrices
    Tensor A_src("A_src", a_shape, dtype);
    Tensor B_src("B_src", b_shape, dtype);
    fillUniform(&A_src);
    fillUniform(&B_src);

    // 2. Quantize to FP8 with MXFP8 scaling
    Tensor A_fp8("A_fp8", a_shape, atype, transa, !transa,
                 NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
    Tensor B_fp8("B_fp8", b_shape, btype, !transb, transb,
                 NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
    nvte_quantize(A_src.data(), A_fp8.data(), 0);
    nvte_quantize(B_src.data(), B_fp8.data(), 0);

    // 3. Dequantize back to reference type
    Tensor A_ref("A_ref", a_shape, dtype);
    Tensor B_ref("B_ref", b_shape, dtype);
    nvte_dequantize(A_fp8.data(), A_ref.data(), 0);
    nvte_dequantize(B_fp8.data(), B_ref.data(), 0);

    // 4. Swizzle scales for gfx1250
    if (prop.major == 12) {
        const bool a_colwise = !transa;
        const bool b_colwise = transb;
        if (!a_colwise) swizzle_mxfp8_scales(A_fp8, true);
        if (a_colwise)  swizzle_mxfp8_scales(A_fp8, false);
        if (!b_colwise) swizzle_mxfp8_scales(B_fp8, true);
        if (b_colwise)  swizzle_mxfp8_scales(B_fp8, false);
    }

    Tensor bias;
    Tensor pre_gelu_out;

    size_t workspace_size = 67108864;  // 64 MB
    Tensor Workspace("Workspace", TShape{workspace_size}, DType::kByte);

    // 5. MXFP8 GEMM
    Tensor D("D", TShape{n, m}, dtype);
    nvte_cublas_gemm(A_fp8.data(), B_fp8.data(), D.data(),
                     bias.data(), pre_gelu_out.data(),
                     transa, transb, false,
                     Workspace.data(), false, false,
                     prop.multiProcessorCount, 0);
    D.to_cpu();

    // 6. Non-FP8 reference GEMM
    Tensor D_ref("D_ref", TShape{n, m}, dtype);
    nvte_cublas_gemm(A_ref.data(), B_ref.data(), D_ref.data(),
                     bias.data(), pre_gelu_out.data(),
                     transa, transb, false,
                     Workspace.data(), false, false,
                     prop.multiProcessorCount, 0);
    D_ref.to_cpu();

    // Check for CUDA errors
    (void)cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // 7. Compare results
    auto [atol, rtol] = getTolerances(dtype);
    atol = std::max(atol, 5e-4);
    rtol = std::max(rtol, 1e-3);
    compareResults("D", D, D_ref.rowwise_cpu_dptr<D_Type>(), true, atol, rtol);
}

// ============================================================================
// Test suite
// ============================================================================

using ProdGemmParam = std::tuple<ProdGemmTestCase, Layout>;

class ProdGemmTestSuite : public ::testing::TestWithParam<ProdGemmParam> {};

TEST_P(ProdGemmTestSuite, TestMxfp8Dq) {
    const auto& tc = std::get<0>(GetParam());
    const auto& layout = std::get<1>(GetParam());
    bool transa = layout.first;
    bool transb = layout.second;

    switch (tc.pass) {
        case GemmPass::FWD:
            performMxfp8DqTest<fp8, fp8, bf16>(tc.m, tc.k, tc.n, transa, transb);
            break;
        case GemmPass::DGRAD:
            performMxfp8DqTest<bf8, fp8, bf16>(tc.m, tc.k, tc.n, transa, transb);
            break;
        case GemmPass::WGRAD:
            performMxfp8DqTest<fp8, bf8, bf16>(tc.m, tc.k, tc.n, transa, transb);
            break;
    }
}

static inline std::string TN(const Layout& layout) {
    static const char* map[2][2] = {{"NN", "NT"}, {"TN", "TT"}};
    return map[layout.first][layout.second];
}

// Regular model shapes (excluding LM Head)
INSTANTIATE_TEST_SUITE_P(
    ProdGemmModel, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(generate_model_test_cases()),
        ::testing::ValuesIn(kLayouts)),
    [](const testing::TestParamInfo<ProdGemmParam>& info) {
        return std::get<0>(info.param).label + "_" + TN(std::get<1>(info.param));
    });

// LM Head shapes (very large N, memory-intensive)
INSTANTIATE_TEST_SUITE_P(
    ProdGemmLMHead, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(generate_lm_head_test_cases()),
        ::testing::ValuesIn(kLayouts)),
    [](const testing::TestParamInfo<ProdGemmParam>& info) {
        return std::get<0>(info.param).label + "_" + TN(std::get<1>(info.param));
    });

}  // namespace

#endif  // __HIP_PLATFORM_AMD__
