/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

// TE CK grouped MXFP8 validation.
//
// Compares three paths for grouped MXFP8 GEMM across NN/NT/TN transpose layouts:
//   1. TE nvte_multi_tensor_gemm grouped path (CK backend selected by env)
//   2. ck_tile::reference_mx_gemm host reference, using exact quantized operands/scales
//   3. TE HIP reference kernel simplified from test_cublaslt_gemm.cu compute_ref_kernel

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

// CK defaults host-side reference code to non-OCP FP8, while TE and CK device kernels
// use OCP FP8/E8M0 on gfx950/gfx12. Force the host reference onto the same encoding.
#ifndef CK_TILE_USE_OCP_FP8
#define CK_TILE_USE_OCP_FP8 1
#endif

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

using namespace transformer_engine;
using namespace test;

using fp8 = fp8e4m3;
using bf8 = fp8e5m2;
using bf16_t = bf16;
using e8m0_t_te = fp8e8m0;

namespace {

enum class MXOperandDType {
  FP8,
  BF8,
};

struct DTypeConfig {
  const char* name;
  MXOperandDType a;
  MXOperandDType b;
};

static DType te_dtype(MXOperandDType t) {
  return t == MXOperandDType::FP8 ? DType::kFloat8E4M3 : DType::kFloat8E5M2;
}

struct LayoutConfig {
  const char* name;
  bool transa;
  bool transb;
};

struct CaseConfig {
  size_t m_total;
  size_t n;
  size_t k;
  int experts;
  LayoutConfig layout;
  DTypeConfig dtype;
};

static std::string case_name(const testing::TestParamInfo<CaseConfig>& info) {
  const auto& c = info.param;
  std::ostringstream os;
  os << "M" << c.m_total << "_N" << c.n << "_K" << c.k
     << "_E" << c.experts << "_" << c.layout.name << "_" << c.dtype.name;
  return os.str();
}

static void set_env_defaults() {
  setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1", 1);
  setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1", 0);
}

static float to_float(const bf16_t& x) { return static_cast<float>(x); }
static float to_float(const ck_tile::bfloat16_t& x) { return static_cast<float>(x); }

template <typename A_Type, typename B_Type, typename D_Type>
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
    size_t m, size_t k, size_t n,
    D_Type* __restrict__ d_data,
    bool transa,
    bool transb) {
  const size_t jj = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t ii = blockIdx.y * blockDim.y + threadIdx.y;
  const bool in_range = (ii < m) && (jj < n);

  float val = 0.0f;
  if (in_range) {
    for (size_t kk = 0; kk < k; ++kk) {
      const size_t a_idx = transa ? (ii * k + kk) : (kk * m + ii);
      const size_t b_idx = transb ? (kk * n + jj) : (jj * k + kk);

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

    d_data[ii + jj * m] = static_cast<D_Type>(val);
  }
}

static std::vector<size_t> split_even(size_t m_total, int experts) {
  NVTE_CHECK(experts > 0, "experts must be > 0");
  NVTE_CHECK(m_total % static_cast<size_t>(experts) == 0,
             "m_total must be divisible by experts");
  return std::vector<size_t>(experts, m_total / static_cast<size_t>(experts));
}

static std::vector<size_t> a_shape_for_te(size_t n, size_t k, bool transa) {
  // TE grouped GEMM computes output shape [M,N]. A contributes the N dimension.
  // transa=true means physical A is [N,K]; transa=false means physical A is [K,N].
  return transa ? std::vector<size_t>{n, k} : std::vector<size_t>{k, n};
}

static std::vector<size_t> b_shape_for_te(size_t m, size_t k, bool transb) {
  // B contributes the M dimension.
  // transb=false means physical B is [M,K]; transb=true means physical B is [K,M].
  return transb ? std::vector<size_t>{k, m} : std::vector<size_t>{m, k};
}

struct ErrorStats {
  size_t failures = 0;
  size_t first_i = 0;
  size_t first_j = 0;
  float first_got = 0.0f;
  float first_ref = 0.0f;
};

static void add_err(ErrorStats& s,
                    float got,
                    float ref,
                    size_t i,
                    size_t j,
                    float rtol,
                    float atol) {
  const bool failed =
      !std::isfinite(got) || !std::isfinite(ref) ||
      std::abs(got - ref) > atol + rtol * std::abs(ref);

  if (failed) {
    if (s.failures == 0) {
      s.first_i = i;
      s.first_j = j;
      s.first_got = got;
      s.first_ref = ref;
    }
    ++s.failures;
  }
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
static auto calculate_ck_internal_rtol_atol(const size_t K,
                                            const int kbatch,
                                            const float max_accumulated_value) {
  using ComputeType =
      std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

  const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
      ck_tile::integer_divide_ceil(static_cast<ck_tile::index_t>(K),
                                   static_cast<ck_tile::index_t>(kbatch)));
  auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
      max_accumulated_value / static_cast<float>(kbatch),
      ck_tile::integer_divide_ceil(static_cast<ck_tile::index_t>(K),
                                   static_cast<ck_tile::index_t>(kbatch)));

  const auto rtol_split_k =
      ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
  auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
      max_accumulated_value, kbatch);

  // Match CK internal mx grouped GEMM test: BF16 gets extra tolerance for HW vs SW conversion.
  if constexpr (std::is_same_v<CDataType, ck_tile::bf16_t> ||
                std::is_same_v<CDataType, ck_tile::bfloat16_t>) {
    atol += 0.6f;
    atol_split_k += 0.6f;
  }

  return ck_tile::make_tuple(std::max(rtol, rtol_split_k),
                             std::max(atol, atol_split_k));
}

static void expect_reference_match_ck_style(const std::string& label,
                                            const ErrorStats& stats) {
  EXPECT_EQ(stats.failures, 0UL)
      << label
      << " failures=" << stats.failures
      << " first_failure_coord=(" << stats.first_i << "," << stats.first_j << ")"
      << " got=" << stats.first_got
      << " ref=" << stats.first_ref;
}

static void run_te_grouped_mxfp8(const std::vector<Tensor>& a_mx,
                                 const std::vector<Tensor>& b_mx,
                                 std::vector<Tensor>* outputs,
                                 Tensor* workspace,
                                 bool transa,
                                 bool transb,
                                 int math_sm_count) {
  const size_t groups = a_mx.size();
  std::vector<NVTETensor> A(groups), B(groups), D(groups), Bias(groups), PreGelu(groups);
  std::vector<Tensor> empty_bias(groups), empty_pregelu(groups);

  for (size_t i = 0; i < groups; ++i) {
    A[i] = const_cast<Tensor&>(a_mx[i]).data();
    B[i] = const_cast<Tensor&>(b_mx[i]).data();
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
                         transa,
                         transb,
                         false,  // grad
                         Workspaces.data(),
                         false,  // accumulate
                         false,  // use_split_accumulator
                         math_sm_count,
                         0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename ATypeIn, typename BTypeIn, typename DTypeOut>
static void run_hip_ref_for_group(const Tensor& a_mx,
                                  const Tensor& b_mx,
                                  Tensor* ref_d_colmajor,
                                  size_t m,
                                  size_t k,
                                  size_t n,
                                  bool transa,
                                  bool transb) {
  // TE grouped GEMM output is op(B) [M,K] * op(A) [K,N] -> [M,N].
  // compute_ref_kernel convention is A_left [M,K] * B_right [K,N].
  // Therefore left operand is TE B and right operand is TE A.
  const bool left_transa = !transb;
  const bool right_transb = !transa;

  const bool left_use_colwise = !left_transa;
  const bool right_use_colwise = right_transb;

  const auto left_s = left_use_colwise ? b_mx.columnwise_scale_inv_shape()
                                       : b_mx.rowwise_scale_inv_shape();
  const auto right_s = right_use_colwise ? a_mx.columnwise_scale_inv_shape()
                                         : a_mx.rowwise_scale_inv_shape();
  NVTE_CHECK(left_s.ndim == 2 && right_s.ndim == 2, "Expected 2D MXFP8 scale_inv tensors");
  const size_t left_scale_ld = left_s.data[1];
  const size_t right_scale_ld = right_s.data[1];

  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
            static_cast<unsigned>((m + block.y - 1) / block.y));

  compute_ref_kernel<ATypeIn, BTypeIn, DTypeOut>
      <<<grid, block, 0, 0>>>(
          static_cast<const ATypeIn*>(left_use_colwise ? b_mx.columnwise_dptr() : b_mx.rowwise_dptr()),
          static_cast<const BTypeIn*>(right_use_colwise ? a_mx.columnwise_dptr() : a_mx.rowwise_dptr()),
          1.0f,
          1.0f,
          static_cast<const e8m0_t_te*>(left_use_colwise ? b_mx.columnwise_scale_inv_dptr()
                                                         : b_mx.rowwise_scale_inv_dptr()),
          static_cast<const e8m0_t_te*>(right_use_colwise ? a_mx.columnwise_scale_inv_dptr()
                                                          : a_mx.rowwise_scale_inv_dptr()),
          left_scale_ld,
          right_scale_ld,
          left_use_colwise,
          right_use_colwise,
          m, k, n,
          static_cast<DTypeOut*>(ref_d_colmajor->rowwise_dptr()),
          left_transa,
          right_transb);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename CkAType, typename CkBType>
static ck_tile::HostTensor<ck_tile::bfloat16_t> run_ck_tile_reference_for_group(
    const Tensor& a_mx,
    const Tensor& b_mx,
    size_t m,
    size_t k,
    size_t n,
    bool transa,
    bool transb) {
  using namespace ck_tile::literals;
  using AType = CkAType;
  using BType = CkBType;
  using CType = ck_tile::bfloat16_t;
  using ScaleType = ck_tile::e8m0_t;

  const size_t kscale = k / 32;

  // TE grouped GEMM computes op(B) [M,K] * op(A) [K,N].
  // CK host reference wants logical left=[M,K], right=[K,N].
  const bool left_transa = !transb;
  const bool right_transb = !transa;
  const bool left_use_colwise = !left_transa;
  const bool right_use_colwise = right_transb;

  ck_tile::HostTensor<AType> a_left(
      left_transa ? ck_tile::HostTensorDescriptor({m, k}, {k, 1_uz})
                  : ck_tile::HostTensorDescriptor({m, k}, {1_uz, m}));
  ck_tile::HostTensor<BType> b_right(
      right_transb ? ck_tile::HostTensorDescriptor({k, n}, {n, 1_uz})
                   : ck_tile::HostTensorDescriptor({k, n}, {1_uz, k}));
  ck_tile::HostTensor<CType> c_ref(
      ck_tile::HostTensorDescriptor({m, n}, {n, 1_uz}));

  // Match CK internal reference scale descriptors:
  //   A scale: logical [M, K/32], RowMajor
  //   B scale: logical [K/32, N], ColumnMajor
  ck_tile::HostTensor<ScaleType> a_scale_ref(
      ck_tile::HostTensorDescriptor({m, kscale}, {kscale, 1_uz}));
  ck_tile::HostTensor<ScaleType> b_scale_ref(
      ck_tile::HostTensorDescriptor({kscale, n}, {1_uz, kscale}));

  c_ref.SetZero();

  NVTE_CHECK_CUDA(cudaMemcpy(a_left.data(),
                             left_use_colwise ? b_mx.columnwise_dptr() : b_mx.rowwise_dptr(),
                             a_left.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(b_right.data(),
                             right_use_colwise ? a_mx.columnwise_dptr() : a_mx.rowwise_dptr(),
                             b_right.get_element_space_size_in_bytes(),
                             cudaMemcpyDeviceToHost));

  auto copy_device_scale_to_host = [](const void* dev_ptr, size_t count) {
    std::vector<ScaleType> host(count);
    NVTE_CHECK_CUDA(cudaMemcpy(host.data(),
                               dev_ptr,
                               count * sizeof(ScaleType),
                               cudaMemcpyDeviceToHost));
    return host;
  };

  if (!left_use_colwise) {
    const auto src = copy_device_scale_to_host(b_mx.rowwise_scale_inv_dptr(), m * kscale);
    for (size_t mi = 0; mi < m; ++mi) {
      for (size_t kc = 0; kc < kscale; ++kc) {
        a_scale_ref(mi, kc) = src[mi * kscale + kc];
      }
    }
  } else {
    const auto src = copy_device_scale_to_host(b_mx.columnwise_scale_inv_dptr(), kscale * m);
    for (size_t mi = 0; mi < m; ++mi) {
      for (size_t kc = 0; kc < kscale; ++kc) {
        a_scale_ref(mi, kc) = src[kc * m + mi];
      }
    }
  }

  if (!right_use_colwise) {
    const auto src = copy_device_scale_to_host(a_mx.rowwise_scale_inv_dptr(), n * kscale);
    for (size_t nj = 0; nj < n; ++nj) {
      for (size_t kc = 0; kc < kscale; ++kc) {
        b_scale_ref(kc, nj) = src[nj * kscale + kc];
      }
    }
  } else {
    const auto src = copy_device_scale_to_host(a_mx.columnwise_scale_inv_dptr(), kscale * n);
    for (size_t kc = 0; kc < kscale; ++kc) {
      for (size_t nj = 0; nj < n; ++nj) {
        b_scale_ref(kc, nj) = src[kc * n + nj];
      }
    }
  }

  ck_tile::reference_mx_gemm<AType, BType, ScaleType, ScaleType, float, CType>(
      a_left, b_right, c_ref, a_scale_ref, b_scale_ref);
  return c_ref;
}

static float max_abs_bf16_tensor(Tensor& t, size_t rows, size_t cols) {
  const bf16_t* p = t.rowwise_cpu_dptr<bf16_t>();
  float m = 0.0f;
  for (size_t i = 0; i < rows * cols; ++i) {
    const float v = static_cast<float>(p[i]);
    if (std::isfinite(v)) m = std::max(m, std::abs(v));
  }
  return m;
}

static float max_abs_ck_ref(const ck_tile::HostTensor<ck_tile::bfloat16_t>& t) {
  float m = 0.0f;
  for (const auto& x : t.mData) {
    const float v = static_cast<float>(x);
    if (std::isfinite(v)) m = std::max(m, std::abs(v));
  }
  return m;
}

static ErrorStats compare_te_vs_hip(Tensor& te_out_rowmajor,
                                    Tensor& hip_ref_colmajor,
                                    size_t m,
                                    size_t n,
                                    float rtol,
                                    float atol) {
  ErrorStats stats;
  const bf16_t* te = te_out_rowmajor.rowwise_cpu_dptr<bf16_t>();
  const bf16_t* hip = hip_ref_colmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(te[i * n + j]), to_float(hip[j * m + i]), i, j, rtol, atol);
    }
  }
  return stats;
}

static ErrorStats compare_te_vs_ck(Tensor& te_out_rowmajor,
                                   const ck_tile::HostTensor<ck_tile::bfloat16_t>& ck_ref,
                                   size_t m,
                                   size_t n,
                                   float rtol,
                                   float atol) {
  ErrorStats stats;
  const bf16_t* te = te_out_rowmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(te[i * n + j]), to_float(ck_ref(i, j)), i, j, rtol, atol);
    }
  }
  return stats;
}

static ErrorStats compare_ck_vs_hip(const ck_tile::HostTensor<ck_tile::bfloat16_t>& ck_ref,
                                    Tensor& hip_ref_colmajor,
                                    size_t m,
                                    size_t n,
                                    float rtol,
                                    float atol) {
  ErrorStats stats;
  const bf16_t* hip = hip_ref_colmajor.rowwise_cpu_dptr<bf16_t>();
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      add_err(stats, to_float(ck_ref(i, j)), to_float(hip[j * m + i]), i, j, rtol, atol);
    }
  }
  return stats;
}

template <typename ATypeIn, typename BTypeIn, typename CkAType, typename CkBType>
static void run_case_typed(const CaseConfig& cfg) {
  set_env_defaults();

  ASSERT_EQ(cfg.k % 128, 0UL) << "K must be a multiple of 128 for MXFP8";
  ASSERT_EQ(cfg.m_total % static_cast<size_t>(cfg.experts), 0UL);

  cudaDeviceProp prop;
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  const bool is_gfx1250 = (prop.major == 12 && prop.minor == 5);

  if (!is_gfx1250) {
    GTEST_SKIP() << "This MXFP8 grouped GEMM test currently exercises the gfx1250-compatible CK pipeline only. GPU="
                 << prop.name << " major=" << prop.major << " minor=" << prop.minor;
  }

  const auto m_splits = split_even(cfg.m_total, cfg.experts);

  std::vector<Tensor> a_src;
  std::vector<Tensor> b_src;
  std::vector<Tensor> a_mx;
  std::vector<Tensor> b_mx;
  std::vector<Tensor> output_te;
  std::vector<Tensor> output_hip_colmajor;
  a_src.reserve(cfg.experts);
  b_src.reserve(cfg.experts);
  a_mx.reserve(cfg.experts);
  b_mx.reserve(cfg.experts);
  output_te.reserve(cfg.experts);
  output_hip_colmajor.reserve(cfg.experts);

  for (int g = 0; g < cfg.experts; ++g) {
    const size_t m = m_splits[g];
    const auto a_shape = a_shape_for_te(cfg.n, cfg.k, cfg.layout.transa);
    const auto b_shape = b_shape_for_te(m, cfg.k, cfg.layout.transb);

    a_src.emplace_back("a_src" + std::to_string(g), a_shape, DType::kBFloat16);
    b_src.emplace_back("b_src" + std::to_string(g), b_shape, DType::kBFloat16);

    fillUniform(&a_src.back());
    fillUniform(&b_src.back());

    // Allocate both rowwise and columnwise MX views so the backend can canonicalize NN/NT/TN.
    a_mx.emplace_back("a_mx", a_shape, te_dtype(cfg.dtype.a),
                      true, true, NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
    b_mx.emplace_back("b_mx", b_shape, te_dtype(cfg.dtype.b),
                      true, true, NVTEScalingMode::NVTE_MXFP8_1D_SCALING);

    nvte_quantize(a_src.back().data(), a_mx.back().data(), 0);
    nvte_quantize(b_src.back().data(), b_mx.back().data(), 0);

    output_te.emplace_back("output_te", std::vector<size_t>{m, cfg.n}, DType::kBFloat16);
    output_hip_colmajor.emplace_back("output_hip_colmajor", std::vector<size_t>{cfg.n, m}, DType::kBFloat16);
  }
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  Tensor workspace("workspace", std::vector<size_t>{67108864}, DType::kByte);

  run_te_grouped_mxfp8(a_mx, b_mx, &output_te, &workspace,
                       cfg.layout.transa, cfg.layout.transb,
                       prop.multiProcessorCount);
  for (auto& out : output_te) out.to_cpu();

  for (int g = 0; g < cfg.experts; ++g) {
    run_hip_ref_for_group<BTypeIn, ATypeIn, bf16_t>(a_mx[g], b_mx[g], &output_hip_colmajor[g],
                                  m_splits[g], cfg.k, cfg.n,
                                  cfg.layout.transa, cfg.layout.transb);
    output_hip_colmajor[g].to_cpu();

    const float max_accumulated_value =
        max_abs_bf16_tensor(output_hip_colmajor[g], cfg.n, m_splits[g]);
    const auto rtol_atol =
        calculate_ck_internal_rtol_atol<CkBType, CkAType, ck_tile::bfloat16_t, float>(
            cfg.k, 1, max_accumulated_value);
    const float rtol = rtol_atol.at(ck_tile::number<0>{});
    const float atol = rtol_atol.at(ck_tile::number<1>{});

    expect_reference_match_ck_style("group " + std::to_string(g) + " TE_vs_HIP_REF",
                                    compare_te_vs_hip(output_te[g], output_hip_colmajor[g],
                                                      m_splits[g], cfg.n, rtol, atol));
  }

  for (int g = 0; g < cfg.experts; ++g) {
    auto ck_ref = run_ck_tile_reference_for_group<CkBType, CkAType>(a_mx[g], b_mx[g],
                                                  m_splits[g], cfg.k, cfg.n,
                                                  cfg.layout.transa, cfg.layout.transb);
    const float max_accumulated_value = max_abs_ck_ref(ck_ref);
    const auto rtol_atol =
        calculate_ck_internal_rtol_atol<CkBType, CkAType, ck_tile::bfloat16_t, float>(
            cfg.k, 1, max_accumulated_value);
    const float rtol = rtol_atol.at(ck_tile::number<0>{});
    const float atol = rtol_atol.at(ck_tile::number<1>{});

    expect_reference_match_ck_style("group " + std::to_string(g) + " TE_vs_CK_REF",
                                    compare_te_vs_ck(output_te[g], ck_ref, m_splits[g], cfg.n,
                                                     rtol, atol));
    expect_reference_match_ck_style("group " + std::to_string(g) + " CK_vs_HIP_REF",
                                    compare_ck_vs_hip(ck_ref, output_hip_colmajor[g],
                                                      m_splits[g], cfg.n, rtol, atol));
  }
}

static void run_case(const CaseConfig& cfg) {
  if (cfg.dtype.a == MXOperandDType::FP8 && cfg.dtype.b == MXOperandDType::FP8) {
    run_case_typed<fp8, fp8, ck_tile::fp8_t, ck_tile::fp8_t>(cfg);
  } else if (cfg.dtype.a == MXOperandDType::FP8 && cfg.dtype.b == MXOperandDType::BF8) {
    run_case_typed<fp8, bf8, ck_tile::fp8_t, ck_tile::bf8_t>(cfg);
  } else if (cfg.dtype.a == MXOperandDType::BF8 && cfg.dtype.b == MXOperandDType::FP8) {
    run_case_typed<bf8, fp8, ck_tile::bf8_t, ck_tile::fp8_t>(cfg);
  } else {
    run_case_typed<bf8, bf8, ck_tile::bf8_t, ck_tile::bf8_t>(cfg);
  }
}

}  // namespace

class GroupedMXFP8TestSuite : public ::testing::TestWithParam<CaseConfig> {};

TEST_P(GroupedMXFP8TestSuite, MatchesCKTileAndHIPReferences) {
  run_case(GetParam());
}

static constexpr LayoutConfig kNN{"NN", false, false};
static constexpr LayoutConfig kNT{"NT", false, true};
static constexpr LayoutConfig kTN{"TN", true, false};

static constexpr DTypeConfig kFP8FP8{"FP8xFP8", MXOperandDType::FP8, MXOperandDType::FP8};
static constexpr DTypeConfig kFP8BF8{"FP8xBF8", MXOperandDType::FP8, MXOperandDType::BF8};
static constexpr DTypeConfig kBF8FP8{"BF8xFP8", MXOperandDType::BF8, MXOperandDType::FP8};
static constexpr DTypeConfig kBF8BF8{"BF8xBF8", MXOperandDType::BF8, MXOperandDType::BF8};

static std::vector<CaseConfig> make_cases() {
  const std::vector<DTypeConfig> dtypes = {kFP8FP8, kFP8BF8, kBF8FP8, kBF8BF8};
  const std::vector<CaseConfig> base_cases = {
      // Small sanity across NN/NT/TN.
      CaseConfig{1024, 1024, 1024, 2, kNN, kFP8FP8},
      CaseConfig{1024, 1024, 1024, 2, kNT, kFP8FP8},
      CaseConfig{1024, 1024, 1024, 2, kTN, kFP8FP8},
      // Earlier failure regime across NN/NT/TN.
      CaseConfig{1536, 4096, 4096, 3, kNN, kFP8FP8},
      CaseConfig{1536, 4096, 4096, 3, kNT, kFP8FP8},
      CaseConfig{1536, 4096, 4096, 3, kTN, kFP8FP8},
      // Llama-ish suspicious path across NN/NT/TN.
      CaseConfig{4096, 12288, 4096, 4, kNN, kFP8FP8},
      CaseConfig{4096, 12288, 4096, 4, kNT, kFP8FP8},
      CaseConfig{4096, 12288, 4096, 4, kTN, kFP8FP8},
  };

  std::vector<CaseConfig> cases;
  cases.reserve(base_cases.size() * dtypes.size());
  for (const auto& base : base_cases) {
    for (const auto& dtype : dtypes) {
      CaseConfig c = base;
      c.dtype = dtype;
      cases.push_back(c);
    }
  }
  return cases;
}

static const std::vector<CaseConfig> kCases = make_cases();

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GroupedMXFP8TestSuite,
    ::testing::ValuesIn(kCases),
    case_name);
