/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

// Forward-only TE CK grouped MXFP8 validation.
//
// Compares three paths for grouped MXFP8 forward GEMM:
//   1. TE nvte_multi_tensor_gemm grouped forward path (CK backend selected by env)
//   2. ck_tile::reference_mx_gemm host reference, using exact quantized operands/scales
//   3. TE HIP reference kernel adapted from test_cublaslt_gemm.cu compute_ref_kernel
//
// Intended drop-in location:
//   TransformerEngine/tests/cpp/operator/test_te_ck_grouped_mxfp8_forward_refs.cu

#ifndef CK_TILE_USE_OCP_FP8
#define CK_TILE_USE_OCP_FP8 1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace transformer_engine;
using namespace test;

using fp8 = fp8e4m3;
using bf16_t = bf16;
using e8m0_t_te = fp8e8m0;

namespace {

struct CaseConfig {
  size_t m_total;
  size_t n;
  size_t k;
  int experts;
  float scale;
  int seed;
  int ck_ref_groups;
};

static std::string case_name(const testing::TestParamInfo<CaseConfig>& info) {
  const auto& c = info.param;
  std::ostringstream os;
  os << "M" << c.m_total << "_N" << c.n << "_K" << c.k
     << "_E" << c.experts;
  return os.str();
}

static void set_env_defaults() {
  setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1", 1);
  setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1", 0);
  setenv("NVTE_ROCM_ENABLE_MXFP8", "1", 0);
}

static float to_float(float x) { return x; }
static float to_float(const bf16_t& x) { return static_cast<float>(x); }
static float to_float(const ck_tile::bfloat16_t& x) { return static_cast<float>(x); }

__device__ __host__ __forceinline__ float ref_gelu_unused(float x) {
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <typename A_Type, typename B_Type, typename Bias_Type,
          typename Gelu_Type, typename D_Type>
__global__ void compute_ref_kernel(
    const A_Type* __restrict__ a_data,
    const B_Type* __restrict__ b_data,
    float a_scale_inv_scalar,
    float b_scale_inv_scalar,
    const e8m0_t_te* __restrict__ a_scale_inv_mxfp8,
    const e8m0_t_te* __restrict__ b_scale_inv_mxfp8,
    size_t a_scale_ld,
    size_t b_scale_ld,
    bool a_scale_is_colwise,
    bool b_scale_is_colwise,
    const Bias_Type* __restrict__ bias_data,
    float d_scale,
    size_t m, size_t k, size_t n,
    D_Type* __restrict__ d_data,
    float* __restrict__ d_amax,
    Gelu_Type* __restrict__ gelu_data,
    bool transa,
    bool transb,
    bool is_fp8_output,
    bool a_is_colwise,
    bool b_is_colwise,
    bool use_mxfp8) {
  const size_t jj = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t ii = blockIdx.y * blockDim.y + threadIdx.y;
  const bool in_range = (ii < m) && (jj < n);

  float val = 0.0f;
  if (in_range) {
    for (size_t kk = 0; kk < k; ++kk) {
      size_t a_idx = 0;
      size_t b_idx = 0;

      if (use_mxfp8) {
        a_idx = transa ? (ii * k + kk) : (kk * m + ii);
        b_idx = transb ? (kk * n + jj) : (jj * k + kk);
      } else {
        a_idx = a_is_colwise ? (ii * k + kk)
                             : (transa ? (ii * k + kk) : (kk * m + ii));
        b_idx = b_is_colwise ? (jj * k + kk)
                             : (transb ? (kk * n + jj) : (jj * k + kk));
      }

      float a_scale_inv_val = a_scale_inv_scalar;
      float b_scale_inv_val = b_scale_inv_scalar;

      if (a_scale_inv_mxfp8) {
        const size_t kc = kk / 32;
        const size_t a_scale_idx =
            a_scale_is_colwise ? (kc * a_scale_ld + ii) : (ii * a_scale_ld + kc);
        const size_t b_scale_idx =
            b_scale_is_colwise ? (kc * b_scale_ld + jj) : (jj * b_scale_ld + kc);
        a_scale_inv_val = exp2f(a_scale_inv_mxfp8[a_scale_idx] - 127.0f);
        b_scale_inv_val = exp2f(b_scale_inv_mxfp8[b_scale_idx] - 127.0f);
      }

      const float a_val = static_cast<float>(a_data[a_idx]);
      const float b_val = static_cast<float>(b_data[b_idx]);
      val += a_scale_inv_val * a_val * b_scale_inv_val * b_val;
    }

    if (bias_data) val += static_cast<float>(bias_data[ii]);
    if (gelu_data) {
      gelu_data[ii + jj * m] = static_cast<Gelu_Type>(val);
      val = ref_gelu_unused(val);
    }

    const float scaled = val * d_scale;
    d_data[ii + jj * m] = static_cast<D_Type>(scaled);
  }

  if (is_fp8_output && d_amax) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    extern __shared__ float s_amax[];
    s_amax[tid] = in_range ? fabsf(val) : 0.0f;
    __syncthreads();
    for (int offset = nthreads / 2; offset > 0; offset /= 2) {
      if (tid < offset) s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + offset]);
      __syncthreads();
    }
    if (tid == 0) atomicMax(d_amax, s_amax[0]);
  }
}

template <typename T>
static void fill_randn_cpu(Tensor* t, float scale, int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  const size_t n = product(t->rowwise_shape());
  T* ptr = t->rowwise_cpu_dptr<T>();
  for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<T>(dist(gen));
  t->from_cpu();
}

static std::vector<size_t> split_even(size_t m_total, int experts) {
  NVTE_CHECK(experts > 0, "experts must be > 0");
  NVTE_CHECK(m_total % static_cast<size_t>(experts) == 0,
             "m_total must be divisible by experts");
  return std::vector<size_t>(experts, m_total / static_cast<size_t>(experts));
}

struct ErrorStats {
  size_t count = 0;
  double sum_abs = 0.0;
  double sum_rel = 0.0;
  double sum_ref_abs = 0.0;
  double sum_got_abs = 0.0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  std::vector<float> abs_errs;
};

static void add_err(ErrorStats& s, float got, float ref) {
  const float abs_err = std::abs(got - ref);
  const float rel_err = abs_err / std::max(std::abs(ref), 1.0e-12f);
  s.count++;
  s.sum_abs += abs_err;
  s.sum_rel += rel_err;
  s.sum_ref_abs += std::abs(ref);
  s.sum_got_abs += std::abs(got);
  s.max_abs = std::max(s.max_abs, abs_err);
  s.max_rel = std::max(s.max_rel, rel_err);
  s.abs_errs.push_back(abs_err);
}

static float quantile(std::vector<float>& values, double q) {
  if (values.empty()) return 0.0f;
  const size_t pos = std::min<size_t>(static_cast<size_t>(q * (values.size() - 1)), values.size() - 1);
  std::nth_element(values.begin(), values.begin() + pos, values.end());
  return values[pos];
}

static void print_stats(const std::string& label, ErrorStats s) {
  std::vector<float> v50 = s.abs_errs;
  std::vector<float> v90 = s.abs_errs;
  std::vector<float> v99 = s.abs_errs;
  const double denom = static_cast<double>(std::max<size_t>(s.count, 1));
  std::cout << std::fixed << std::setprecision(6)
            << label
            << " count=" << s.count
            << " max_abs=" << s.max_abs
            << " mean_abs=" << (s.sum_abs / denom)
            << " p50_abs=" << quantile(v50, 0.50)
            << " p90_abs=" << quantile(v90, 0.90)
            << " p99_abs=" << quantile(v99, 0.99)
            << " max_rel=" << s.max_rel
            << " mean_rel=" << (s.sum_rel / denom)
            << " ref_abs_mean=" << (s.sum_ref_abs / denom)
            << " got_abs_mean=" << (s.sum_got_abs / denom)
            << std::endl;
}

static void expect_reference_match(const std::string& label,
                                   const ErrorStats& stats,
                                   float max_abs_limit,
                                   float mean_abs_limit) {
  print_stats(label, stats);
  EXPECT_LE(stats.max_abs, max_abs_limit) << label;
  EXPECT_LE(stats.sum_abs / static_cast<double>(std::max<size_t>(stats.count, 1)),
            static_cast<double>(mean_abs_limit)) << label;
}

static void run_te_grouped_mxfp8_forward(const std::vector<Tensor>& weights_mx,
                                         const std::vector<Tensor>& inputs_mx,
                                         std::vector<Tensor>* outputs,
                                         Tensor* workspace,
                                         int math_sm_count) {
  const size_t groups = weights_mx.size();
  std::vector<NVTETensor> A(groups), B(groups), D(groups), Bias(groups), PreGelu(groups);
  std::vector<Tensor> empty_bias(groups), empty_pregelu(groups);

  // Match GroupedLinear forward / te_general_grouped_gemm:
  //   A = weight [N,K], transa=true
  //   B = input  [M,K], transb=false
  //   D = output [M,N]
  for (size_t i = 0; i < groups; ++i) {
    A[i] = const_cast<Tensor&>(weights_mx[i]).data();
    B[i] = const_cast<Tensor&>(inputs_mx[i]).data();
    D[i] = (*outputs)[i].data();
    Bias[i] = empty_bias[i].data();
    PreGelu[i] = empty_pregelu[i].data();
  }

  std::vector<NVTETensor> Workspaces(1);
  Workspaces[0] = workspace->data();

  nvte_multi_tensor_gemm(A.data(),
                         B.data(),
                         D.data(),
                         Bias.data(),
                         PreGelu.data(),
                         groups,
                         true,   // transa: weight [N,K] -> op(A) [K,N]
                         false,  // transb: input [M,K]  -> op(B) [M,K]
                         false,  // grad
                         Workspaces.data(),
                         false,  // accumulate
                         false,  // use_split_accumulator
                         math_sm_count,
                         0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename DTypeOut>
static void run_hip_ref_for_group(const Tensor& input_mx,
                                  const Tensor& weight_mx,
                                  Tensor* ref_d_colmajor,
                                  size_t m,
                                  size_t k,
                                  size_t n) {
  // compute_ref_kernel expects A=input [M,K], B=weight [N,K], transa=true, transb=false,
  // and writes D as column-major MxN into rowwise storage shaped [N,M].
  const auto a_s = input_mx.rowwise_scale_inv_shape();
  const auto b_s = weight_mx.rowwise_scale_inv_shape();
  NVTE_CHECK(a_s.ndim == 2 && b_s.ndim == 2, "Expected 2D MXFP8 scale_inv tensors");
  const size_t a_scale_ld = a_s.data[1];
  const size_t b_scale_ld = b_s.data[1];

  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
            static_cast<unsigned>((m + block.y - 1) / block.y));
  const size_t shmem_bytes = size_t(block.x) * size_t(block.y) * sizeof(float);

  compute_ref_kernel<fp8, fp8, bf16_t, bf16_t, DTypeOut>
      <<<grid, block, shmem_bytes, 0>>>(
          static_cast<const fp8*>(input_mx.rowwise_dptr()),
          static_cast<const fp8*>(weight_mx.rowwise_dptr()),
          1.0f,
          1.0f,
          static_cast<const e8m0_t_te*>(input_mx.rowwise_scale_inv_dptr()),
          static_cast<const e8m0_t_te*>(weight_mx.rowwise_scale_inv_dptr()),
          a_scale_ld,
          b_scale_ld,
          false,  // input scale rowwise [M,K/32]
          false,  // weight scale rowwise [N,K/32]
          nullptr,
          1.0f,
          m, k, n,
          static_cast<DTypeOut*>(ref_d_colmajor->rowwise_dptr()),
          nullptr,
          nullptr,
          true,   // transa for A=input in this reference-kernel convention
          false,  // transb for B=weight
          false,
          false,
          false,
          true);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
}

static ck_tile::HostTensor<ck_tile::bfloat16_t> run_ck_tile_reference_for_group(
    const Tensor& input_mx,
    const Tensor& weight_mx,
    size_t m,
    size_t k,
    size_t n) {
  using namespace ck_tile::literals;
  using AType = ck_tile::fp8_t;
  using BType = ck_tile::fp8_t;
  using CType = ck_tile::bfloat16_t;
  using ScaleType = ck_tile::e8m0_t;

  const size_t kscale = k / 32;

  ck_tile::HostTensor<AType> a_host(
      ck_tile::HostTensorDescriptor({m, k}, {k, 1_uz}));
  ck_tile::HostTensor<BType> b_host(
      ck_tile::HostTensorDescriptor({k, n}, {1_uz, k}));
  ck_tile::HostTensor<CType> c_ref(
      ck_tile::HostTensorDescriptor({m, n}, {n, 1_uz}));
  ck_tile::HostTensor<ScaleType> a_scale_ref(
      ck_tile::HostTensorDescriptor({m, kscale}, {kscale, 1_uz}));
  ck_tile::HostTensor<ScaleType> b_scale_ref(
      ck_tile::HostTensorDescriptor({kscale, n}, {1_uz, kscale}));

  c_ref.SetZero();

  NVTE_CHECK_CUDA(cudaMemcpy(a_host.data(),
                             input_mx.rowwise_dptr(),
                             a_host.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(b_host.data(),
                             weight_mx.rowwise_dptr(),
                             b_host.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(a_scale_ref.data(),
                             input_mx.rowwise_scale_inv_dptr(),
                             a_scale_ref.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(b_scale_ref.data(),
                             weight_mx.rowwise_scale_inv_dptr(),
                             b_scale_ref.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));

  ck_tile::reference_mx_gemm<AType, BType, ScaleType, ScaleType, float, CType>(
      a_host, b_host, c_ref, a_scale_ref, b_scale_ref);
  return c_ref;
}

static ErrorStats compare_te_vs_hip(const Tensor& te_out_rowmajor,
                                    const Tensor& hip_ref_colmajor,
                                    size_t m,
                                    size_t n) {
  ErrorStats stats;
  const bf16_t* te = te_out_rowmajor.rowwise_cpu_dptr<bf16_t>();
  const bf16_t* hip = hip_ref_colmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(te[i * n + j]), to_float(hip[j * m + i]));
    }
  }
  return stats;
}

static ErrorStats compare_te_vs_ck(const Tensor& te_out_rowmajor,
                                   const ck_tile::HostTensor<ck_tile::bfloat16_t>& ck_ref,
                                   size_t m,
                                   size_t n) {
  ErrorStats stats;
  const bf16_t* te = te_out_rowmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(te[i * n + j]), to_float(ck_ref(i, j)));
    }
  }
  return stats;
}

static ErrorStats compare_ck_vs_hip(const ck_tile::HostTensor<ck_tile::bfloat16_t>& ck_ref,
                                    const Tensor& hip_ref_colmajor,
                                    size_t m,
                                    size_t n) {
  ErrorStats stats;
  const bf16_t* hip = hip_ref_colmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(ck_ref(i, j)), to_float(hip[j * m + i]));
    }
  }
  return stats;
}

static void run_case(const CaseConfig& cfg) {
  set_env_defaults();

  ASSERT_EQ(cfg.k % 128, 0UL) << "K must be a multiple of 128 for MXFP8";
  ASSERT_EQ(cfg.m_total % static_cast<size_t>(cfg.experts), 0UL);

  cudaDeviceProp prop;
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
#ifdef __HIP_PLATFORM_AMD__
  const bool is_gfx950_or_newer_cdna = (prop.major == 9 && prop.minor >= 5);
  const bool is_gfx1250 = (prop.major == 12 && prop.minor == 5);

  if (!is_gfx950_or_newer_cdna && !is_gfx1250) {
    GTEST_SKIP() << "MXFP8 requires gfx950+ or gfx1250 in this test. GPU=" << prop.name
                 << " major=" << prop.major << " minor=" << prop.minor;
  }
#endif

  const auto m_splits = split_even(cfg.m_total, cfg.experts);
  const size_t per_m = m_splits[0];
  const int groups_to_ck = std::min(cfg.ck_ref_groups, cfg.experts);

  std::cout << "\n=== TE CK grouped MXFP8 forward reference comparison ===\n"
            << "M_total=" << cfg.m_total << " N=" << cfg.n << " K=" << cfg.k
            << " experts=" << cfg.experts << " per_expert_M=" << per_m
            << " scale=" << cfg.scale << " seed=" << cfg.seed << "\n"
            << "NVTE_USE_CUTLASS_GROUPED_GEMM=" << std::getenv("NVTE_USE_CUTLASS_GROUPED_GEMM") << "\n"
            << "NVTE_ROCM_ENABLE_MXFP8=" << std::getenv("NVTE_ROCM_ENABLE_MXFP8") << "\n"
            << "CK_TILE_USE_OCP_FP8=" << CK_TILE_USE_OCP_FP8 << "\n"
            << "GPU=" << prop.name << " SM/CU count=" << prop.multiProcessorCount << "\n";

  std::vector<Tensor> input_src;
  std::vector<Tensor> weight_src;
  std::vector<Tensor> input_mx;
  std::vector<Tensor> weight_mx;
  std::vector<Tensor> output_te;
  std::vector<Tensor> output_hip_colmajor;
  input_src.reserve(cfg.experts);
  weight_src.reserve(cfg.experts);
  input_mx.reserve(cfg.experts);
  weight_mx.reserve(cfg.experts);
  output_te.reserve(cfg.experts);
  output_hip_colmajor.reserve(cfg.experts);

  for (int g = 0; g < cfg.experts; ++g) {
    const size_t m = m_splits[g];
    input_src.emplace_back("input_src", std::vector<size_t>{m, cfg.k}, DType::kBFloat16);
    weight_src.emplace_back("weight_src", std::vector<size_t>{cfg.n, cfg.k}, DType::kBFloat16);

    fill_randn_cpu<bf16_t>(&input_src.back(), cfg.scale, cfg.seed + 1009 * g + 17);
    fill_randn_cpu<bf16_t>(&weight_src.back(), cfg.scale, cfg.seed + 1009 * g + 29);

    input_mx.emplace_back("input_mx", std::vector<size_t>{m, cfg.k}, DType::kFloat8E4M3,
                          true, false, NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
    weight_mx.emplace_back("weight_mx", std::vector<size_t>{cfg.n, cfg.k}, DType::kFloat8E4M3,
                           true, false, NVTEScalingMode::NVTE_MXFP8_1D_SCALING);

    nvte_quantize(input_src.back().data(), input_mx.back().data(), 0);
    nvte_quantize(weight_src.back().data(), weight_mx.back().data(), 0);

    output_te.emplace_back("output_te", std::vector<size_t>{m, cfg.n}, DType::kBFloat16);
    output_hip_colmajor.emplace_back("output_hip_colmajor", std::vector<size_t>{cfg.n, m}, DType::kBFloat16);
  }
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  Tensor workspace("workspace", std::vector<size_t>{67108864}, DType::kByte);

  run_te_grouped_mxfp8_forward(weight_mx, input_mx, &output_te, &workspace,
                               prop.multiProcessorCount);
  for (auto& out : output_te) out.to_cpu();

  for (int g = 0; g < cfg.experts; ++g) {
    run_hip_ref_for_group<bf16_t>(input_mx[g], weight_mx[g], &output_hip_colmajor[g],
                                  m_splits[g], cfg.k, cfg.n);
    output_hip_colmajor[g].to_cpu();
    expect_reference_match("group " + std::to_string(g) + " TE_vs_HIP_REF",
                           compare_te_vs_hip(output_te[g], output_hip_colmajor[g],
                                             m_splits[g], cfg.n),
                           0.25f,
                           0.03f);
  }

  for (int g = 0; g < groups_to_ck; ++g) {
    auto ck_ref = run_ck_tile_reference_for_group(input_mx[g], weight_mx[g],
                                                  m_splits[g], cfg.k, cfg.n);
    expect_reference_match("group " + std::to_string(g) + " TE_vs_CK_REF ",
                           compare_te_vs_ck(output_te[g], ck_ref, m_splits[g], cfg.n),
                           0.25f,
                           0.03f);
    expect_reference_match("group " + std::to_string(g) + " CK_vs_HIP_REF",
                           compare_ck_vs_hip(ck_ref, output_hip_colmajor[g],
                                             m_splits[g], cfg.n),
                           0.25f,
                           0.03f);
  }
}

}  // namespace

class GroupedMXFP8ForwardRefsTestSuite : public ::testing::TestWithParam<CaseConfig> {};

TEST_P(GroupedMXFP8ForwardRefsTestSuite, MatchesCKTileAndHIPReferences) {
  run_case(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GroupedMXFP8ForwardRefsTestSuite,
    ::testing::Values(
        // Small enough for quick CI-style sanity.
        CaseConfig{1024, 1024, 1024, 2, 0.25f, 1234, 1},
        // Reproduces the earlier forward-only "failure" scale/shape regime, but
        // validates against true MXFP8 references instead of BF16.
        CaseConfig{1536, 4096, 4096, 3, 0.25f, 1234, 1},
        // Llama-ish suspicious path. CK reference only group 0 to keep runtime sane;
        // HIP reference checks all groups.
        CaseConfig{4096, 12288, 4096, 4, 0.25f, 1234, 1}),
    case_name);
