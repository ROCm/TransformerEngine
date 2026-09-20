/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

#ifdef __HIP_PLATFORM_AMD__

// ============================================================================
// CK grouped GEMM validation.
//
// This is a ROCm-only, standalone test binary source (compiled into
// test_operator). It drives nvte_multi_tensor_gemm with
// NVTE_USE_CK_GROUPED_GEMM=1 so the CK grouped backend (ck_tile_grouped_gemm)
// is exercised, unlike the single-GEMM path (nvte_cublas_gemm).
//
// This suite validates the CK grouped-GEMM backend specifically: it captures
// stderr around the call and FAILS if the config falls back off CK to the
// multi-stream hipBLASLt path (the library emits "Fallback to cuBLAS grouped
// GEMM." when NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1).
//
// The reference-GEMM helpers below are intentionally self-contained (mirroring
// test_ck_grouped_mxfp8.cu) and live in an anonymous namespace so they do not
// clash with the similarly-named helpers in test_cublaslt_gemm.cu.
// ============================================================================

namespace {

using fp32 = float;
using fp8 = fp8e4m3;
using bf8 = fp8e5m2;
using TShape = std::vector<size_t>;

__device__ __host__ __forceinline__ float ref_gelu(float x) {
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <typename A_Type, typename B_Type, typename Bias_Type,
          typename Gelu_Type, typename D_Type>
__global__ void compute_ref_kernel(
  const A_Type* __restrict__ a_data,
  const B_Type* __restrict__ b_data,
  float a_scale_inv_scalar,                       // used when mxfp8 == false
  float b_scale_inv_scalar,
  const fp8e8m0* __restrict__ a_scale_inv_mxfp8,  // used when mxfp8 == true
  const fp8e8m0* __restrict__ b_scale_inv_mxfp8,
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
  bool use_mxfp8)
{
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
        // Non-MXFP8 FP8 path may use explicit transpose buffers (cpu_rowwise_to_columnwise),
        // so indexing depends on which backing buffer is passed in.
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

    if (bias_data) {
      val += static_cast<float>(bias_data[ii]);
    }

    if (gelu_data) {
      gelu_data[ii + jj * m] = static_cast<Gelu_Type>(val);
      val = ref_gelu(val);
    }

    const float scaled = val * d_scale;
    d_data[ii + jj * m] = static_cast<D_Type>(scaled);
  }

  // Blockwise reduction for amax
  if (is_fp8_output && d_amax) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    extern __shared__ float s_amax[];

    // Out-of-range threads contribute 0
    s_amax[tid] = in_range ? fabsf(val) : 0.0f;
    __syncthreads();

    for (int offset = nthreads / 2; offset > 0; offset /= 2) {
      if (tid < offset) {
        s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + offset]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float block_max = s_amax[0];
      atomicMax(d_amax, block_max);
    }
  }
}

struct TestParams {
  size_t m;
  size_t k;
  size_t n;
  bool use_bias;
  bool use_gelu;
  bool transa;
  bool transb;
  NVTEScalingMode scaling_mode;
  bool force_hipblaslt;
};

template <typename A_Type, typename B_Type, typename Bias_Type,
          typename Gelu_Type, typename D_Type>
static void run_reference(
    const TestParams& params,
    Tensor& A,
    Tensor& B,
    const Tensor* Bias,                 // nullable
    Tensor& D_for_scale,
    Tensor& RefD,
    Tensor* RefPreGeluOut)              // nullable
{
  const bool use_mxfp8 = (params.scaling_mode == NVTE_MXFP8_1D_SCALING);

  const bool is_fp8_output = test::isFp8Type(test::TypeInfo<D_Type>::dtype);

  // Only FP8 output has a scale (set via setRandomScale); non-FP8 output is unscaled.
  const float d_scale = is_fp8_output ? D_for_scale.scale() : 1.0f;

  const bool a_use_colwise = (!params.transa) && A.columnwise();
  const bool b_use_colwise = ( params.transb) && B.columnwise();

  const A_Type* a_dev = static_cast<const A_Type*>(
      a_use_colwise ? A.columnwise_dptr() : A.rowwise_dptr());

  const B_Type* b_dev = static_cast<const B_Type*>(
      b_use_colwise ? B.columnwise_dptr() : B.rowwise_dptr());

  // scaling inputs
  float a_scale_inv_scalar = 1.0f;
  float b_scale_inv_scalar = 1.0f;

  const fp8e8m0* a_scale_dev = nullptr;
  const fp8e8m0* b_scale_dev = nullptr;
  size_t a_scale_ld = 0;
  size_t b_scale_ld = 0;
  bool a_scale_is_colwise = !params.transa;
  bool b_scale_is_colwise =  params.transb;

  if (use_mxfp8) {
    a_scale_dev = static_cast<const fp8e8m0*>(
        a_scale_is_colwise ? A.columnwise_scale_inv_dptr() : A.rowwise_scale_inv_dptr());
    b_scale_dev = static_cast<const fp8e8m0*>(
        b_scale_is_colwise ? B.columnwise_scale_inv_dptr() : B.rowwise_scale_inv_dptr());

    const NVTEShape a_s = a_scale_is_colwise ? A.columnwise_scale_inv_shape() : A.rowwise_scale_inv_shape();
    const NVTEShape b_s = b_scale_is_colwise ? B.columnwise_scale_inv_shape() : B.rowwise_scale_inv_shape();
    NVTE_CHECK(a_s.ndim == 2 && b_s.ndim == 2, "Expected 2D MXFP8 scale_inv");
    a_scale_ld = a_s.data[1];
    b_scale_ld = b_s.data[1];
  } else {
    // Per-tensor scale_inv exists only for FP8 inputs; non-FP8 inputs stay at 1.0.
    if (test::isFp8Type(test::TypeInfo<A_Type>::dtype)) a_scale_inv_scalar = A.rowwise_scale_inv();
    if (test::isFp8Type(test::TypeInfo<B_Type>::dtype)) b_scale_inv_scalar = B.rowwise_scale_inv();
  }

  // optional bias device pointer
  const Bias_Type* bias_dev = nullptr;
  if (Bias) {
    bias_dev = static_cast<const Bias_Type*>(Bias->rowwise_dptr());
  }

  D_Type* d_refD = static_cast<D_Type*>(RefD.rowwise_dptr());

  Gelu_Type* d_refGelu = nullptr;
  float* d_refAmax = nullptr;

  if (RefPreGeluOut) {
    d_refGelu = static_cast<Gelu_Type*>(RefPreGeluOut->rowwise_dptr());
  }

  if (is_fp8_output) {
    d_refAmax = static_cast<float*>(RefD.amax_dptr());
    if (d_refAmax)
      NVTE_CHECK_CUDA(cudaMemset(d_refAmax, 0, sizeof(float)));
  }

  // Kernel launch
  dim3 block(16, 16);
  dim3 grid((unsigned)((params.n + block.x - 1) / block.x),
            (unsigned)((params.m + block.y - 1) / block.y));

  const size_t shmem_bytes = size_t(block.x) * size_t(block.y) * sizeof(float);

  compute_ref_kernel<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>
      <<<grid, block, shmem_bytes, 0>>>(
          a_dev,
          b_dev,
          a_scale_inv_scalar,
          b_scale_inv_scalar,
          a_scale_dev,
          b_scale_dev,
          a_scale_ld,
          b_scale_ld,
          a_scale_is_colwise,
          b_scale_is_colwise,
          bias_dev,
          d_scale,
          params.m, params.k, params.n,
          d_refD,
          d_refAmax,
          d_refGelu,
          params.transa,
          params.transb,
          is_fp8_output,
          a_use_colwise,
          b_use_colwise,
          use_mxfp8);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename Type>
void cpu_rowwise_to_columnwise(
  size_t m, size_t n,
  const Type* rowwise_ptr,
  Type* columnwise_ptr){

  for(size_t ii = 0; ii < m; ii++){
    for(size_t jj = 0; jj < n; jj++){
      columnwise_ptr[jj*m + ii] = rowwise_ptr[ii*n + jj];
    }
  }
}

std::pair<double, double> getTestTolerances(const DType type, bool use_fp8, bool use_mxfp8) {
  auto [atol, rtol] = getTolerances(type);

  //relax for certain prime number gemm
  if (type == DType::kFloat32) {
    atol = 1e-5;
  }
  // relax for certain FP8 gemm with hipblaslt
  if (use_mxfp8) {
    atol = 5e-4;
    rtol = std::max(rtol, 1e-3);
    // gfx950 MXFP8 GEMMs can show larger numerical variance
    // Relax tolerances to avoid flaky failures.
    cudaDeviceProp prop;
    (void)cudaGetDeviceProperties(&prop, 0);
    if (prop.major == 9 && prop.minor == 5) {
      rtol = std::max(rtol, 6e-2);
    }
  }
  else if (use_fp8) {
    atol = 1e-3;
    rtol = std::max(rtol, 1e-2);
    // gfx950 FP8 GEMMs can show larger numerical variance
    cudaDeviceProp prop;
    (void)cudaGetDeviceProperties(&prop, 0);
    if (prop.major == 9 && prop.minor == 5) {
      rtol = std::max(rtol, 2e-2);
    }
  }
  else if (type == DType::kBFloat16) {
    //relax for certain prime number TN gemm
    rtol = 5e-2;
  }
  else if (type == DType::kFloat32) {
    rtol = 1e-5;
  }
  return {atol, rtol};
}

}  // namespace

struct GroupedGemmCase {
  const char* label;
  size_t m;
  size_t k;
  size_t n;
  bool transa;
  bool transb;
  bool accumulate;
};

template <typename A_Type, typename B_Type, typename D_Type>
void performGroupedTest(const GroupedGemmCase& c) {
  const DType atype = TypeInfo<A_Type>::dtype;
  const DType btype = TypeInfo<B_Type>::dtype;
  const DType dtype = TypeInfo<D_Type>::dtype;
  const bool has_fp8 = isFp8Type(atype) || isFp8Type(btype);

  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, 0);

  if (has_fp8) {
    const bool fp8_supported = (prop.major == 9 && prop.minor >= 4) || prop.major >= 12;
    if (!fp8_supported) {
      GTEST_SKIP() << "FP8 is not supported in current config";
    }
    // TE convention: delayed-scaling FP8 grouped GEMM is TN (transa=T, transb=N).
    // After ck_grouped_gemm's A/B swap that becomes CK's internal NT presentation
    // with rowwise data only. Non-TN layouts (NT/NN/...) use the columnwise buffers
    // allocated below so CK can rewrite into its internal NT form.
  }

  // Route the grouped GEMM through the CK backend, and force the fallback
  // warning on so we can detect (and fail on) any fallback off CK.
  setenv("NVTE_USE_CK_GROUPED_GEMM", "1", 1);
  setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1", 1);

  // Fused wgrad accumulation (beta=1) always targets an FP32 main_grad buffer;
  // CK does not support accumulate on FP8, so restrict it to FP32-output 16-bit.
  const bool accumulate = c.accumulate && (dtype == DType::kFloat32) && !has_fp8;

  // TN FP8 uses rowwise only; non-TN FP8 allocates columnwise data for CK's layout
  // rewrite (same rule as performTest: "non TN layout" needs columnwise).
  const bool a_colwise = !c.transa && isFp8Type(atype);
  const bool b_colwise = c.transb && isFp8Type(btype);

  constexpr int num_gemms = 3;

  TestParams gp{};
  gp.m = c.m;
  gp.k = c.k;
  gp.n = c.n;
  gp.use_bias = false;
  gp.use_gelu = false;
  gp.transa = c.transa;
  gp.transb = c.transb;
  gp.scaling_mode = NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING;
  gp.force_hipblaslt = false;

  std::vector<Tensor> A_tensors, B_tensors, D_tensors, RefProd_tensors;
  A_tensors.reserve(num_gemms);
  B_tensors.reserve(num_gemms);
  D_tensors.reserve(num_gemms);
  RefProd_tensors.reserve(num_gemms);
  std::vector<std::vector<D_Type>> D_init(num_gemms);

  for (int i = 0; i < num_gemms; ++i) {
    A_tensors.emplace_back("gA" + std::to_string(i),
                           c.transa ? TShape{c.m, c.k} : TShape{c.k, c.m}, atype,
                           /*rowwise=*/true, a_colwise, NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING);
    B_tensors.emplace_back("gB" + std::to_string(i),
                           c.transb ? TShape{c.k, c.n} : TShape{c.n, c.k}, btype,
                           /*rowwise=*/true, b_colwise, NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING);
    D_tensors.emplace_back("gD" + std::to_string(i), TShape{c.n, c.m}, dtype);
    // Keep the reference product in FP32 so that, like the real GEMM, the
    // accumulate case rounds (D0 + product) to D_Type exactly once instead of
    // double-rounding an already-truncated product.
    RefProd_tensors.emplace_back("gRef" + std::to_string(i), TShape{c.n, c.m}, DType::kFloat32);

    fillUniform(&A_tensors[i]);
    if (a_colwise) {
      cpu_rowwise_to_columnwise(c.k, c.m, A_tensors[i].rowwise_cpu_dptr<A_Type>(),
                                A_tensors[i].columnwise_cpu_dptr<A_Type>());
      A_tensors[i].from_cpu();
    }
    fillUniform(&B_tensors[i]);
    if (b_colwise) {
      cpu_rowwise_to_columnwise(c.k, c.n, B_tensors[i].rowwise_cpu_dptr<B_Type>(),
                                B_tensors[i].columnwise_cpu_dptr<B_Type>());
      B_tensors[i].from_cpu();
    }
    // Seed the accumulator (D0) with data so the accumulate (beta=1) path is truly exercised
    const D_Type* d0 = D_tensors[i].rowwise_cpu_dptr<D_Type>();
    D_init[i].assign(d0, d0 + c.m * c.n);

    // Independent FP32-accumulate reference product op(A) x op(B) -> RefProd.
    run_reference<A_Type, B_Type, bf16, bf16, fp32>(
        gp, A_tensors[i], B_tensors[i], /*Bias=*/nullptr, D_tensors[i], RefProd_tensors[i],
        /*RefPreGeluOut=*/nullptr);
  }

  std::vector<NVTETensor> A_ptrs(num_gemms), B_ptrs(num_gemms), D_ptrs(num_gemms);
  std::vector<NVTETensor> bias_ptrs(num_gemms), gelu_ptrs(num_gemms);
  std::vector<Tensor> empty_bias(num_gemms), empty_gelu(num_gemms);
  for (int i = 0; i < num_gemms; ++i) {
    A_ptrs[i] = A_tensors[i].data();
    B_ptrs[i] = B_tensors[i].data();
    D_ptrs[i] = D_tensors[i].data();
    bias_ptrs[i] = empty_bias[i].data();
    gelu_ptrs[i] = empty_gelu[i].data();
  }

  // multi_stream_cublas_gemm (the fallback path) indexes workspace[i % num_streams],
  // so allocate one workspace per compute stream to stay in-bounds.
  const int num_streams = nvte_get_num_compute_streams();
  size_t workspace_size = 33'554'432;
  if ((prop.major == 9 && prop.minor == 5) || prop.major >= 12) {
    workspace_size = 67'108'864;
  }
  std::vector<Tensor> workspaces;
  workspaces.reserve(num_streams);
  std::vector<NVTETensor> workspace_ptrs(num_streams);
  for (int s = 0; s < num_streams; ++s) {
    workspaces.emplace_back("gws" + std::to_string(s), TShape{workspace_size}, DType::kByte);
    workspace_ptrs[s] = workspaces[s].data();
  }

  // Capture stderr across the call so we can assert CK handled the config and
  // did not fall back to the multi-stream hipBLASLt path. stderr is restored on
  // every path (including if the fallback GEMM throws) so a failure here does
  // not corrupt output for the rest of the test binary.
  std::fflush(stderr);
  std::cerr.flush();
  const int saved_stderr_fd = dup(fileno(stderr));
  std::FILE* captured_file = std::tmpfile();
  ASSERT_NE(captured_file, nullptr) << "Failed to create temp file for stderr capture";
  dup2(fileno(captured_file), fileno(stderr));

  bool threw = false;
  std::string exception_msg;
  try {
    nvte_multi_tensor_gemm(A_ptrs.data(), B_ptrs.data(), D_ptrs.data(), bias_ptrs.data(),
                           gelu_ptrs.data(), num_gemms, c.transa, c.transb, /*grad=*/false,
                           workspace_ptrs.data(), accumulate, /*use_split_accumulator=*/false,
                           prop.multiProcessorCount, 0);
    (void)cudaDeviceSynchronize();
  } catch (const std::exception& e) {
    threw = true;
    exception_msg = e.what();
  } catch (...) {
    threw = true;
    exception_msg = "unknown exception";
  }

  std::cerr.flush();
  std::fflush(stderr);
  dup2(saved_stderr_fd, fileno(stderr));
  close(saved_stderr_fd);

  std::string captured;
  std::rewind(captured_file);
  char rdbuf[4096];
  size_t nread;
  while ((nread = std::fread(rdbuf, 1, sizeof(rdbuf), captured_file)) > 0) {
    captured.append(rdbuf, nread);
  }
  std::fclose(captured_file);
  // Preserve the captured warnings in the test log.
  if (!captured.empty()) {
    std::cerr << captured;
  }

  // This suite is exclusively for the CK grouped-GEMM backend
  // Reject any fallback to multi-stream hipBLASLt
  ASSERT_EQ(captured.find("Fallback to cuBLAS grouped GEMM"), std::string::npos)
      << "Grouped GEMM fell back off the CK backend for this config:\n"
      << captured;

  ASSERT_FALSE(threw) << "Grouped GEMM threw for this config: " << exception_msg << "\n"
                      << captured;

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Tolerances are governed by the input precision, not the output dtype.
  auto [atol, rtol] = getTestTolerances(atype, /*use_fp8=*/has_fp8, /*use_mxfp8=*/false);
  for (int i = 0; i < num_gemms; ++i) {
    D_tensors[i].to_cpu();
    RefProd_tensors[i].to_cpu();
    const float* prod = RefProd_tensors[i].rowwise_cpu_dptr<float>();
    std::vector<D_Type> expected(c.m * c.n);
    for (size_t e = 0; e < c.m * c.n; ++e) {
      const float v = accumulate ? static_cast<float>(D_init[i][e]) + prod[e] : prod[e];
      expected[e] = static_cast<D_Type>(v);
    }
    compareResults("grouped_D" + std::to_string(i), D_tensors[i], expected.data(), true, atol,
                   rtol);
  }
}

class GroupedGEMMTestSuite : public ::testing::TestWithParam<GroupedGemmCase> {};

TEST_P(GroupedGEMMTestSuite, Testbf16xbf16xbf16) {
  performGroupedTest<bf16, bf16, bf16>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Testbf16xbf16xfp32) {
  performGroupedTest<bf16, bf16, fp32>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Testfp16xfp16xfp16) {
  performGroupedTest<fp16, fp16, fp16>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Testfp16xfp16xfp32) {
  performGroupedTest<fp16, fp16, fp32>(GetParam());
}
// FP8 inputs (delayed scaling). TN is the production default; non-TN layouts use
// columnwise buffers for CK's rewrite path. accumulate=false (CK FP8 overwrite only).
TEST_P(GroupedGEMMTestSuite, Teste4m3xe4m3xbf16) {
  performGroupedTest<fp8, fp8, bf16>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Teste4m3xe4m3xfp16) {
  performGroupedTest<fp8, fp8, fp16>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Teste4m3xe4m3xfp32) {
  performGroupedTest<fp8, fp8, fp32>(GetParam());
}
TEST_P(GroupedGEMMTestSuite, Teste5m2xe5m2xbf16) {
  performGroupedTest<bf8, bf8, bf16>(GetParam());
}
// {label, m, k, n, transa, transb, accumulate}.
// 16-bit: NT is fused-wgrad; TN/NN also covered. FP8: TN rowwise + non-TN columnwise.
static const std::vector<GroupedGemmCase> kGroupedGemmCases = {
    {"NT_accum", 256, 128, 256, false, true, true},
    {"NT_noaccum", 256, 128, 256, false, true, false},
    {"NN_accum", 256, 128, 256, false, false, true},
    {"TN_accum", 256, 128, 256, true, false, true},
    {"TN_noaccum", 256, 128, 256, true, false, false},
    {"NT_accum_384x256x512", 384, 256, 512, false, true, true},
    {"TN_noaccum_384x256x512", 384, 256, 512, true, false, false},
};

INSTANTIATE_TEST_SUITE_P(OperatorTest, GroupedGEMMTestSuite,
                         ::testing::ValuesIn(kGroupedGemmCases),
                         [](const testing::TestParamInfo<GroupedGEMMTestSuite::ParamType>& info) {
                           return std::string(info.param.label);
                         });

#endif  // __HIP_PLATFORM_AMD__
