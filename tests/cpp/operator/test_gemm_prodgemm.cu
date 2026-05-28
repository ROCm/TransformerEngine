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
#include <set>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

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

// DeepSeek3 (hidden=7168, MLA, seq=4096)

static const ShapeDef deepseek3_shapes[] = {
    // Forward (M=tokens, N, K)
    {"DeepSeek3_Linear0_fwd",        1536,  7168, GemmPass::FWD},
    {"DeepSeek3_Linear1_fwd",         576,  7168, GemmPass::FWD},
    {"DeepSeek3_LNLinear0_fwd",     24576,  1536, GemmPass::FWD},
    {"DeepSeek3_LNLinear1_fwd",     32768,   512, GemmPass::FWD},
    {"DeepSeek3_Linear_attn_fwd",    7168, 16384, GemmPass::FWD},
    {"DeepSeek3_LNMLP_gateup_fwd",  36864,  7168, GemmPass::FWD},
    {"DeepSeek3_LNMLP_down_fwd",     7168, 18432, GemmPass::FWD},
    {"DeepSeek3_SharedExp_gu_fwd",   4096,  7168, GemmPass::FWD},
    {"DeepSeek3_SharedExp_dn_fwd",   7168,  2048, GemmPass::FWD},
    {"DeepSeek3_TopKRouter_fwd",      256,  7168, GemmPass::FWD},
    // Dgrad (M=tokens, N, K)
    {"DeepSeek3_attn_dgrad",        16384,  7168, GemmPass::DGRAD},
    {"DeepSeek3_LNLinear1_dgrad",     512, 32768, GemmPass::DGRAD},
    {"DeepSeek3_LNLinear0_dgrad",    1536, 24576, GemmPass::DGRAD},
    {"DeepSeek3_SharedExp_dn_dgrad", 2048,  7168, GemmPass::DGRAD},
    {"DeepSeek3_SharedExp_gu_dgrad", 7168,  4096, GemmPass::DGRAD},
    {"DeepSeek3_TopKRouter_dgrad",   7168,   256, GemmPass::DGRAD},
    {"DeepSeek3_MLP_post_dgrad",     7168, 14336, GemmPass::DGRAD},
    // Wgrad (M, N, K=tokens)
    {"DeepSeek3_attn_wgrad",        16384,  7168, GemmPass::WGRAD},
    {"DeepSeek3_LNLinear1_wgrad",     512, 32768, GemmPass::WGRAD},
    {"DeepSeek3_LNLinear0_wgrad",    1536, 24576, GemmPass::WGRAD},
    {"DeepSeek3_SharedExp_dn_wgrad", 2048,  7168, GemmPass::WGRAD},
    {"DeepSeek3_SharedExp_gu_wgrad", 7168,  4096, GemmPass::WGRAD},
    {"DeepSeek3_TopKRouter_wgrad",   7168,   256, GemmPass::WGRAD},
};

// DeepSeek3 LM Head (large N, memory-intensive)
static const ShapeDef deepseek3_lm_head_shapes[] = {
    {"DeepSeek3_LMHead_fwd",     129280,   7168, GemmPass::FWD},
    {"DeepSeek3_LMHead_dgrad",     7168, 129280, GemmPass::DGRAD},
    {"DeepSeek3_LMHead_wgrad",     7168, 129280, GemmPass::WGRAD},
};

// Qwen3 (hidden=4096, GQA, seq=4096)

static const ShapeDef qwen3_shapes[] = {
    // Forward (M=tokens, N, K)
    {"Qwen3_LNLinear_QKV_fwd",  9216,  4096, GemmPass::FWD},
    {"Qwen3_Linear_attn_fwd",   4096,  8192, GemmPass::FWD},
    {"Qwen3_Router_fwd",         128,  4096, GemmPass::FWD},
    // Dgrad (M=tokens, N, K)
    {"Qwen3_Router_dgrad",      4096,   128, GemmPass::DGRAD},
    {"Qwen3_Linear_attn_dgrad", 8192,  4096, GemmPass::DGRAD},
    {"Qwen3_LNLinear_dgrad",    4096,  9216, GemmPass::DGRAD},
    // Wgrad (M, N, K=tokens)
    {"Qwen3_Router_wgrad",      4096,   128, GemmPass::WGRAD},
    {"Qwen3_Linear_attn_wgrad", 8192,  4096, GemmPass::WGRAD},
    {"Qwen3_LNLinear_wgrad",    4096,  9216, GemmPass::WGRAD},
};

// Qwen3 LM Head (large N, memory-intensive)
static const ShapeDef qwen3_lm_head_shapes[] = {
    {"Qwen3_LMHead_fwd",       151936,   4096, GemmPass::FWD},
    {"Qwen3_LMHead_dgrad",       4096, 151936, GemmPass::DGRAD},
    {"Qwen3_LMHead_wgrad",       4096, 151936, GemmPass::WGRAD},
};

// ====================================================
// Test case: a concrete (M, K, N) shape with pass info
// ====================================================

std::ostream& operator<<(std::ostream& os, const ShapeDef& s) {
    return os << s.label;
}

static void resolve_mkn(const ShapeDef& s, size_t mbs,
                         size_t& m, size_t& k, size_t& n) {
    size_t tokens = mbs * 4096;
    switch (s.pass) {
        case GemmPass::FWD: // Fallthrough, same as DGRAD
        case GemmPass::DGRAD:
            m = tokens;
            n = s.dim1;
            k = s.dim2;
            break;
        case GemmPass::WGRAD:
            m = s.dim1;
            n = s.dim2;
            k = tokens;
            break;
    }
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

using ProdGemmParam = std::tuple<ShapeDef, size_t, Layout>;

class ProdGemmTestSuite : public ::testing::TestWithParam<ProdGemmParam> {};

// Known-failing GEMM shapes on gfx950
static const std::set<std::string> kMI355XSkips = {
    // N=576 + NT: rocroller LDS stride mismatch
    "DeepSeek3_Linear1_fwd_mbs1_NT",
    "DeepSeek3_Linear1_fwd_mbs2_NT",
    "DeepSeek3_Linear1_fwd_mbs4_NT",
    // Sporadic kernel failures
    "DeepSeek3_LNLinear0_fwd_mbs2_NT",
    "DeepSeek3_Linear_attn_fwd_mbs4_NT",
    "DeepSeek3_LNMLP_gateup_fwd_mbs4_NN",
    "DeepSeek3_LNMLP_down_fwd_mbs2_NN",
    "DeepSeek3_attn_wgrad_mbs2_NN",
    "DeepSeek3_LNLinear0_dgrad_mbs2_NN",
    "DeepSeek3_LNLinear0_wgrad_mbs4_NN",
    "DeepSeek3_SharedExp_dn_wgrad_mbs4_NT",
    // K=128 (minimum for MXFP8): unreliable across layouts
    "Qwen3_Router_fwd_mbs1_NN",
    "Qwen3_Router_dgrad_mbs1_NN",
    "Qwen3_Router_dgrad_mbs1_NT",
    "Qwen3_Router_dgrad_mbs2_NT",
    "Qwen3_Router_dgrad_mbs4_TN",
    "Qwen3_Router_dgrad_mbs4_NT",
    // Other failures
    "Qwen3_Linear_attn_wgrad_mbs2_NT",
    "DeepSeek3_LMHead_fwd_mbs1_NT",
    "DeepSeek3_LMHead_fwd_mbs4_NN",
    // Qwen3 LM Head dgrad (N=151936): nearly all combos fail
    "Qwen3_LMHead_dgrad_mbs1_NN",
    "Qwen3_LMHead_dgrad_mbs1_NT",
    "Qwen3_LMHead_dgrad_mbs2_TN",
    "Qwen3_LMHead_dgrad_mbs2_NN",
    "Qwen3_LMHead_dgrad_mbs2_NT",
    "Qwen3_LMHead_dgrad_mbs4_TN",
    "Qwen3_LMHead_dgrad_mbs4_NN",
    "Qwen3_LMHead_dgrad_mbs4_NT",
    // Crash (likely OOM / kernel fault)
    "Qwen3_LMHead_fwd_mbs4_TN",
    "Qwen3_LMHead_fwd_mbs4_NN",
    "Qwen3_LMHead_fwd_mbs4_NT",
};

TEST_P(ProdGemmTestSuite, TestMxfp8Dq) {
    const auto& shape = std::get<0>(GetParam());
    size_t mbs = std::get<1>(GetParam());
    const auto& layout = std::get<2>(GetParam());
    bool transa = layout.first;
    bool transb = layout.second;

    static const char* tn_map[2][2] = {{"NN", "NT"}, {"TN", "TT"}};
    std::string name = std::string(shape.label) + "_mbs" + std::to_string(mbs)
                       + "_" + tn_map[transa][transb];
    if (kMI355XSkips.count(name)) {
        GTEST_SKIP() << "Known MI355X hipBLASLt failure: " << name;
    }

    size_t m, k, n;
    resolve_mkn(shape, mbs, m, k, n);

    switch (shape.pass) {
        case GemmPass::FWD:
            performMxfp8DqTest<fp8, fp8, bf16>(m, k, n, transa, transb);
            break;
        case GemmPass::DGRAD:
            performMxfp8DqTest<bf8, fp8, bf16>(m, k, n, transa, transb);
            break;
        case GemmPass::WGRAD:
            performMxfp8DqTest<fp8, bf8, bf16>(m, k, n, transa, transb);
            break;
    }
}

static inline std::string TN(const Layout& layout) {
    static const char* map[2][2] = {{"NN", "NT"}, {"TN", "TT"}};
    return map[layout.first][layout.second];
}

static inline auto testName(const testing::TestParamInfo<ProdGemmParam>& info) {
    const auto& shape = std::get<0>(info.param);
    size_t mbs = std::get<1>(info.param);
    const auto& layout = std::get<2>(info.param);
    return std::string(shape.label) + "_mbs" + std::to_string(mbs) + "_" + TN(layout);
}

// DeepSeek3 model shapes
INSTANTIATE_TEST_SUITE_P(
    ProdGemmDeepSeek3, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(deepseek3_shapes),
        ::testing::Values(size_t{1}, size_t{2}, size_t{4}),
        ::testing::ValuesIn(kLayouts)),
    testName);

// Qwen3 model shapes
INSTANTIATE_TEST_SUITE_P(
    ProdGemmQwen3, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(qwen3_shapes),
        ::testing::Values(size_t{1}, size_t{2}, size_t{4}),
        ::testing::ValuesIn(kLayouts)),
    testName);

// DeepSeek3 LM Head shapes (very large N, memory-intensive)
INSTANTIATE_TEST_SUITE_P(
    ProdGemmDeepSeek3LMHead, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(deepseek3_lm_head_shapes),
        ::testing::Values(size_t{1}, size_t{2}, size_t{4}),
        ::testing::ValuesIn(kLayouts)),
    testName);

// Qwen3 LM Head shapes (very large N, memory-intensive)
INSTANTIATE_TEST_SUITE_P(
    ProdGemmQwen3LMHead, ProdGemmTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(qwen3_lm_head_shapes),
        ::testing::Values(size_t{1}, size_t{2}, size_t{4}),
        ::testing::ValuesIn(kLayouts)),
    testName);

}  // namespace

#endif  // __HIP_PLATFORM_AMD__
