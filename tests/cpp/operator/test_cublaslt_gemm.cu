/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include <cmath>
#include <iostream>
#include <string>
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
//m, k, n
std::vector<std::tuple<size_t, size_t, size_t>> test_case_sizes = {
  {2304, 768, 4096},
  {768, 768, 4096},
  {768, 3072, 4096},
  {229, 541, 541}, //primes
  {71, 71, 3571}, //primes
  {29, 29, 17389}, //primes
}; 

std::vector<std::tuple<size_t, size_t, size_t>> test_case_sizes_mxfp8 = {
  {2304, 768, 4096},
}; 

//  A, B, Bias, Gelu, D
//  Bias type choose as bf16 in use_fp8, D_type otherwise
//  Gelu type the same as Bias_Type

using fp32=float;
using fp8=fp8e4m3;
using bf8=fp8e5m2;

using Layout = std::pair<bool,bool>;// {transa, transb}
static const Layout kNN{false,false};
static const Layout kTN{true ,false};
static const Layout kNT{false,true };

static const std::vector<Layout> kLayouts = { kNN, kTN, kNT };

using TShape = std::vector<size_t>;
}  // namespace


float ref_gelu(float x){
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <typename A_Type, typename B_Type, typename Bias_Type, typename Gelu_Type, typename D_Type>
void compute_ref(
  const A_Type* a_data,
  const B_Type* b_data,
  const float a_scale_inv,
  const float b_scale_inv,
  const Bias_Type* bias_data, //bias is of dim m
  const float d_scale,
  size_t m, size_t k, size_t n,
  D_Type* ref_d_data,
  float* ref_d_amax_ptr,
  Gelu_Type* ref_gelu_data,
  bool transa,
  bool transb){

  float ref_d_amax = 0;

  #pragma omp parallel for schedule(static) collapse(2) reduction(max: ref_d_amax) proc_bind(spread)
  for(size_t ii = 0; ii < m; ii++){
    for(size_t jj = 0; jj < n; jj++){
      float val = 0;
      for(size_t kk = 0; kk < k; kk++){
        float a_val = transa ? a_data[kk + ii*k] : a_data[ii + kk*m];
        float b_val = transb ? b_data[jj + kk*n] : b_data[kk + jj*k];
        val += a_scale_inv*a_val*b_scale_inv*b_val;
      }
      if(bias_data){
        val += (float)bias_data[ii];
      }
      if(ref_gelu_data){
        ref_gelu_data[ii + jj*m] = (Gelu_Type)(val);
        val = ref_gelu(val);
      }
      ref_d_data[ii+jj*m] = (D_Type)(val*d_scale);
      // update ref_d_amax if in fp8
      DType dtype = TypeInfo<D_Type>::dtype;
      if(isFp8Type(dtype)){
        ref_d_amax = std::max<float>(ref_d_amax, std::fabs(val));
      }
    }
  }
  if (ref_d_amax_ptr)
  {
    *ref_d_amax_ptr = ref_d_amax;
  }
}

template <typename A_Type, typename B_Type, typename Bias_Type, typename Gelu_Type, typename D_Type>
void compute_mxfp8_ref(
  const A_Type* a_data,
  const B_Type* b_data,
  const fp8e8m0* a_scale_inv_data,
  const fp8e8m0* b_scale_inv_data,
  const Bias_Type* bias_data, //bias is of dim m
  const float d_scale,
  size_t m, size_t k, size_t n,
  D_Type* ref_d_data,
  float* ref_d_amax_ptr,
  Gelu_Type* ref_gelu_data,
  bool transa,
  bool transb){

  float ref_d_amax = 0;

  #pragma omp parallel for schedule(static) collapse(2) reduction(max: ref_d_amax) proc_bind(spread)
  for(size_t ii = 0; ii < m; ii++){
    for(size_t jj = 0; jj < n; jj++){
      float val = 0;
      for(size_t kk = 0; kk < k; kk++){
        size_t a_idx = transa ? (ii*k + kk) : (kk*m + ii);
        size_t b_idx = transb ? (kk*n + jj) : (jj*k + kk);
        float a_scale_inv_val = (float)std::pow(2,
          a_scale_inv_data[transa ? a_idx/32 : (kk/32 * m + ii)] - 127);
        float b_scale_inv_val = (float)std::pow(2,
          b_scale_inv_data[transb ? (kk/32 * n + jj) : b_idx/32] - 127);
        val += a_scale_inv_val * (float)a_data[a_idx] * b_scale_inv_val * (float)b_data[b_idx];
      }
      if(bias_data){
        val += (float)bias_data[ii];
      }
      if(ref_gelu_data){
        ref_gelu_data[ii + jj*m] = (Gelu_Type)(val);
        val = ref_gelu(val);
      }
      ref_d_data[ii+jj*m] = (D_Type)(val*d_scale);
      // update ref_d_amax if in fp8
      DType dtype = TypeInfo<D_Type>::dtype;
      if(isFp8Type(dtype)){
        ref_d_amax = std::max<float>(ref_d_amax, std::fabs(val));
      }
    }
  }
  if (ref_d_amax_ptr)
  {
    *ref_d_amax_ptr = ref_d_amax;
  }
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
    /*During hipifying std::max is converted to ::max
    to w/a HIP bug with using std:: in device functions.
    W/o explicitlit <double>, compiler uses non-templated int method variant from HIP headers
    TODO: remove when switch to new hipify version after fixing HIP bug */
    rtol = std::max<double>(rtol, 1e-3);
  }
  else if (use_fp8) {
    atol = 1e-3;
    //TODO: remove <double> (see comment above)
    rtol = std::max<double>(rtol, 5e-3);
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

struct TestParams {
  size_t m;
  size_t k;
  size_t n;
  bool use_bias;
  bool use_gelu;
  bool transa;
  bool transb;
  NVTEScalingMode scaling_mode;
};

template <typename A_Type, typename B_Type, typename Bias_Type, typename Gelu_Type, typename D_Type>
void performTest(const TestParams& params) {
  DType atype = TypeInfo<A_Type>::dtype;
  DType btype = TypeInfo<B_Type>::dtype;
  DType bias_type = TypeInfo<Bias_Type>::dtype;
  DType gelu_type = TypeInfo<Gelu_Type>::dtype;
  DType dtype = TypeInfo<D_Type>::dtype;

  const bool has_fp8 = isFp8Type(atype) || isFp8Type(btype);
  const bool use_mxfp8 = params.scaling_mode == NVTEScalingMode::NVTE_MXFP8_1D_SCALING;

  if (use_mxfp8)
  {
    if (!has_fp8) {
      GTEST_SKIP() << "MXFP8 scaling mode requires Float8 types";
    }
    if (params.m % 32 != 0 || params.n % 32 != 0 || params.k % 32 != 0) {
      GTEST_SKIP() << "MXFP8 requires M, N, K to be multiples of 32";
    }
  }

  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, 0);

#ifdef __HIP_PLATFORM_AMD__

  // Enable FP8 GEMM + GELU fusion tests only on MI300 (gfx942) with ROCm > 7.0.
  // hipBLASLt currently supports this config only
  bool fp8_gelu_fusion_config = false;
  #if HIP_VERSION >= 70000000
    if (prop.major == 9 && prop.minor == 4)
    {
      fp8_gelu_fusion_config = atype == DType::kFloat8E4M3 &&
                              btype == DType::kFloat8E4M3 &&
                              dtype == DType::kFloat8E4M3 &&
                              (params.use_gelu && gelu_type == DType::kFloat16) &&
                              (!params.use_bias || bias_type == DType::kFloat16);
    }
  #endif

  if (has_fp8)
  {
    bool fp8_supported = (prop.major == 9 && prop.minor >= 4);
    if (!fp8_supported) {
      GTEST_SKIP() << "FP8 is not supported in current config";
    }

    if (use_mxfp8)
    {
      bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5);
      if (!mxfp8_supported) {
        GTEST_SKIP() << "MXFP8 is not supported in current config";
      }
      if (params.use_bias) {
        GTEST_SKIP() << "MXFP8 GEMM with bias is not supported";
      }
    }

    if (params.use_gelu && !fp8_gelu_fusion_config) {
      GTEST_SKIP() << "FP8 GEMM with GELU is not supported in current config";
    }
    if (params.use_bias && dtype == DType::kFloat16) {
      GTEST_SKIP() << "FP8 GEMM with bias and FP16 output is not supported";
    }
  }

  if (prop.major == 9 && prop.minor == 5) //gfx950 specific hipblasLt limitations
  {
    if (isFp8Type(dtype)){
      GTEST_SKIP() << "GEMM with float8 output is not supported";
    }
    if (params.use_gelu && dtype == DType::kBFloat16) {
      GTEST_SKIP() << "BF16 GEMM with GELU is not supported in current config";
    }
    if constexpr ((std::is_same<A_Type, bf8>::value || std::is_same<B_Type, bf8>::value) &&
      std::is_same<D_Type, fp32>::value)
    {
      //GEMM with bias and fp32 output is not supported with bf8 A/B
      if (params.use_bias) {
        GTEST_SKIP() << "FP8 GEMM with bias is not supported in current config";
      }
    }
  }
  if (prop.major == 9 && prop.minor == 4) //gfx942 specific hipblasLt limitations
  {
    if (params.use_gelu && dtype == DType::kBFloat16 && !params.transa) {
      GTEST_SKIP() << "BF16 GEMM with GELU is not supported in current config";
    }
    if (has_fp8 && params.use_bias && dtype == DType::kFloat8E4M3 && !fp8_gelu_fusion_config) {
      GTEST_SKIP() << "FP8 GEMM with bias and FP8 output is not supported in current config";
    }
  }
#endif

  // FP8 GEMM path needs columnwise data for A/B tensor with non TN layout
  const bool a_colwise = !params.transa && isFp8Type(atype);
  const bool b_colwise = params.transb && isFp8Type(btype);
  Tensor A("A", params.transa ? TShape{ params.m, params.k } : TShape{ params.k, params.m },
    atype, (!a_colwise || !use_mxfp8), a_colwise, params.scaling_mode);
  Tensor B("B", params.transb ? TShape{ params.k, params.n } : TShape{ params.n, params.k },
    btype, (!b_colwise || !use_mxfp8), b_colwise, params.scaling_mode);

  Tensor D("D", TShape{ params.n, params.m }, dtype);
  Tensor bias;
  if(params.use_bias){
    bias = Tensor("bias", TShape{params.m}, bias_type);
  }
  Tensor pre_gelu_out;
  if(params.use_gelu){
    pre_gelu_out = Tensor("pre_gelu_out", TShape{ params.n, params.m }, gelu_type);
  }
  
  //initialize the data and scale inv of A, B
  //fillUniform does not initialize columnwise data if rowwise data exist
  fillUniform(&A);
  if (a_colwise && !use_mxfp8) {
    // A must be of shape k, m
    cpu_rowwise_to_columnwise(params.k, params.m,
      A.rowwise_cpu_dptr<A_Type>(), A.columnwise_cpu_dptr<A_Type>());
    // sync the columnwise data on GPU as well
    A.from_cpu();
  }
  fillUniform(&B);
  if (b_colwise && !use_mxfp8) {
    // B must be of shape k, n
    cpu_rowwise_to_columnwise(params.k, params.n,
      B.rowwise_cpu_dptr<B_Type>(), B.columnwise_cpu_dptr<B_Type>());
    // sync the columnwise data on GPU as well
    B.from_cpu();
  }
  if(params.use_bias){
    fillUniform(&bias);
  }
  //initialize the scale of D
  if(isFp8Type(dtype)){
    setRandomScale(&D);
  }
  bool grad = false;
  bool accumulate = false;

  size_t workspace_size = 33554432;
#ifdef __HIP_PLATFORM_AMD__
  if (prop.major == 9 && prop.minor == 5) {
    workspace_size = 67108864;
  }
#endif
  Tensor Workspace("Workspace", TShape{ workspace_size }, DType::kByte);

  //perform the gemm in GPU
  nvte_cublas_gemm(A.data(),
                   B.data(),
                   D.data(),
                   bias.data(),
                   pre_gelu_out.data(),
                   params.transa,
                   params.transb,
                   grad,
                   Workspace.data(),
                   accumulate,
                   false,
                   prop.multiProcessorCount,
                   //default stream
                   0);
  //copy the output results from GPU memory to CPU memory
  D.to_cpu();
  if(params.use_gelu){
    pre_gelu_out.to_cpu();
  }

  //perform the gemm in CPU
  std::unique_ptr<D_Type[]> ref_D = std::make_unique<D_Type[]>(params.m*params.n);
  std::unique_ptr<Gelu_Type[]> ref_pre_gelu_out;
  if(params.use_gelu){
    ref_pre_gelu_out = std::make_unique<Gelu_Type[]>(params.m*params.n);
  }

  float ref_amax_d;
  if (use_mxfp8) {
    const A_Type *a_data;
    const B_Type *b_data;
    const fp8e8m0 *a_scale_inv_data, *b_scale_inv_data;
    if (params.transa) {
      a_data = A.rowwise_cpu_dptr<A_Type>();
      a_scale_inv_data = A.rowwise_cpu_scale_inv_ptr<fp8e8m0>();
    } else {
      a_data = A.columnwise_cpu_dptr<A_Type>();
      a_scale_inv_data = A.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
    }
    if (params.transb) {
      b_data = B.columnwise_cpu_dptr<B_Type>();
      b_scale_inv_data = B.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
    } else {
      b_data = B.rowwise_cpu_dptr<B_Type>();
      b_scale_inv_data = B.rowwise_cpu_scale_inv_ptr<fp8e8m0>();
    }

    compute_mxfp8_ref<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(
        a_data, b_data, a_scale_inv_data, b_scale_inv_data,
        params.use_bias ? bias.rowwise_cpu_dptr<Bias_Type>() : nullptr,
        D.scale(), params.m, params.k, params.n, ref_D.get(), &ref_amax_d,
        params.use_gelu ? ref_pre_gelu_out.get() : nullptr,
        params.transa, params.transb);
  } else {
    compute_ref<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(
        A.rowwise_cpu_dptr<A_Type>(), B.rowwise_cpu_dptr<B_Type>(),
        A.rowwise_scale_inv(), B.rowwise_scale_inv(),
        params.use_bias ? bias.rowwise_cpu_dptr<Bias_Type>() : nullptr,
        D.scale(), params.m, params.k, params.n, ref_D.get(), &ref_amax_d,
        params.use_gelu ? ref_pre_gelu_out.get() : nullptr,
        params.transa, params.transb);
  }
  // check if error message happens in running                             
  (void)cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  //compare results
  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(dtype)) {
    compareResults("D_amax", D.amax(), ref_amax_d, atol_amax, rtol_amax);
  }

  auto [atol, rtol] = getTestTolerances(dtype, has_fp8, use_mxfp8);
  compareResults("D", D, ref_D.get(), true, atol, rtol);

  if(params.use_gelu){
    auto [atol, rtol] = getTestTolerances(gelu_type, false, false);
    compareResults("gelu", pre_gelu_out, ref_pre_gelu_out.get(), true, atol, rtol);
  }
}

#ifdef __HIP_PLATFORM_AMD__
template <typename A_Type, typename B_Type, typename D_Type>
void performDqTest(const TestParams &params) {
  DType atype = TypeInfo<A_Type>::dtype;
  DType btype = TypeInfo<B_Type>::dtype;
  DType dtype = TypeInfo<D_Type>::dtype;

  GTEST_ASSERT_TRUE(isFp8Type(atype) && isFp8Type(btype)) << "FP8/BF8 input datatype is expected";
  GTEST_ASSERT_FALSE(isFp8Type(dtype)) << "Non FP8/BF8 output datatype is expected";

  if (params.m % 32 != 0 || params.n % 32 != 0 || params.k % 32 != 0) {
    GTEST_SKIP() << "MXFP8 requires M, N, K to be multiples of 32";
  }

  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, 0);

  bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5);
  if (!mxfp8_supported) {
    GTEST_SKIP() << "MXFP8 is not supported in current config";
  }

  DType ref_type = dtype;
  TShape a_shape = params.transa ? TShape{params.m, params.k} : TShape{params.k, params.m};
  TShape b_shape = params.transb ? TShape{params.k, params.n} : TShape{params.n, params.k};

  Tensor A_src("A", a_shape, ref_type);
  Tensor B_src("B", b_shape, ref_type);
  //initialize A, B
  fillUniform(&A_src);
  fillUniform(&B_src);

  // FP8 GEMM path needs columnwise data for A/B tensor with non TN layout
  Tensor A_fp8("A_fp8", a_shape, atype, params.transa, !params.transa,
               NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
  Tensor B_fp8("B_fp8", b_shape, btype, !params.transb, params.transb,
               NVTEScalingMode::NVTE_MXFP8_1D_SCALING);
  nvte_quantize(A_src.data(), A_fp8.data(), 0);
  nvte_quantize(B_src.data(), B_fp8.data(), 0);

  Tensor A_ref("A_ref", a_shape, ref_type);
  Tensor B_ref("B_ref", b_shape, ref_type);
  nvte_dequantize(A_fp8.data(), A_ref.data(), 0);
  nvte_dequantize(B_fp8.data(), B_ref.data(), 0);

  Tensor bias;
  Tensor pre_gelu_out;

  size_t workspace_size = 67108864;
  Tensor Workspace("Workspace", TShape{workspace_size}, DType::kByte);

  //perform FP8 gemm and copy the output results from GPU memory to CPU memory
  Tensor D("D", TShape{params.n, params.m}, dtype);
  nvte_cublas_gemm(A_fp8.data(), B_fp8.data(), D.data(), bias.data(), pre_gelu_out.data(),
                   params.transa, params.transb, false, Workspace.data(), false, false,
                   prop.multiProcessorCount, 0);
  D.to_cpu();


  //perform non-FP8 gemm and copy the output results from GPU memory to CPU memory
  Tensor D_ref("D", TShape{params.n, params.m}, dtype);
  nvte_cublas_gemm(A_ref.data(), B_ref.data(), D_ref.data(), bias.data(), pre_gelu_out.data(),
                   params.transa, params.transb, false, Workspace.data(), false, false,
                   prop.multiProcessorCount, 0);
  D_ref.to_cpu();

  // check if error message happens in running
  (void)cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  //compare results
  auto [atol, rtol] = getTestTolerances(dtype, true, true);
  compareResults("D", D, D_ref.rowwise_cpu_dptr<D_Type>(), true, atol, rtol);
}
#endif // __HIP_PLATFORM_AMD__

#define MAKE_TEST_PARAMS(P_)                                                    \
  TestParams P_ = {.m = std::get<0>(std::get<0>(GetParam())),                   \
                   .k = std::get<1>(std::get<0>(GetParam())),                   \
                   .n = std::get<2>(std::get<0>(GetParam())),                   \
                   .use_bias = std::get<1>(GetParam()),                         \
                   .use_gelu = std::get<2>(GetParam()),                         \
                   .transa = std::get<3>(GetParam()).first,                     \
                   .transb = std::get<3>(GetParam()).second,                    \
                   .scaling_mode = std::get<4>(GetParam())                      \
                                       ? NVTEScalingMode::NVTE_MXFP8_1D_SCALING \
                                       : NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING}

// <m, k, n>, use_bias, use_gelu, Layout, fp8_scalinig
class GEMMTestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<size_t, size_t, size_t>, bool, bool, Layout, NVTEScalingMode>> {};

#define MAKE_GEMM_TEST(NAME_, A_, B_, BIAS_, GELU_, D_)                     \
  TEST_P(GEMMTestSuite, NAME_) {                                            \
    MAKE_TEST_PARAMS(test_params);                                          \
    using A_Type = A_;                                                      \
    using B_Type = B_;                                                      \
    using Bias_Type = BIAS_;                                                \
    using Gelu_Type = GELU_;                                                \
    using D_Type = D_;                                                      \
    performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(test_params); \
  }

MAKE_GEMM_TEST(Testfp32xfp32xfp32xfp32xfp32, fp32, fp32, fp32, fp32, fp32);

MAKE_GEMM_TEST(Testfp16xfp16xfp16xfp16xfp16, fp16, fp16, fp16, fp16, fp16);

MAKE_GEMM_TEST(Testbf16xbf16xbf16xbf16xbf16, bf16, bf16, bf16, bf16, bf16);

MAKE_GEMM_TEST(Testfp8xfp8xbf16xbf16xfp32, fp8, fp8, bf16, bf16, fp32);

MAKE_GEMM_TEST(Testfp8xfp8xbf16xbf16xfp16, fp8, fp8, bf16, bf16, fp16);

MAKE_GEMM_TEST(Testfp8xfp8xbf16xbf16xbf16, fp8, fp8, bf16, bf16, bf16);

MAKE_GEMM_TEST(Testfp8xfp8xbf16xbf16xfp8, fp8, fp8, bf16, bf16, fp8);

MAKE_GEMM_TEST(Testfp8xfp8xbf16xbf16xbf8, fp8, fp8, bf16, bf16, bf8);

MAKE_GEMM_TEST(Testfp8xbf8xbf16xbf16xfp32, fp8, bf8, bf16, bf16, fp32);

MAKE_GEMM_TEST(Testfp8xbf8xbf16xbf16xfp16, fp8, bf8, bf16, bf16, fp16);

MAKE_GEMM_TEST(Testfp8xbf8xbf16xbf16xbf16, fp8, bf8, bf16, bf16, bf16);

MAKE_GEMM_TEST(Testfp8xbf8xbf16xbf16xfp8, fp8, bf8, bf16, bf16, fp8);

MAKE_GEMM_TEST(Testfp8xbf8xbf16xbf16xbf8, fp8, bf8, bf16, bf16, bf8);

MAKE_GEMM_TEST(Testbf8xfp8xbf16xbf16xfp32, bf8, fp8, bf16, bf16, fp32);

MAKE_GEMM_TEST(Testbf8xfp8xbf16xbf16xfp16, bf8, fp8, bf16, bf16, fp16);

MAKE_GEMM_TEST(Testbf8xfp8xbf16xbf16xbf16, bf8, fp8, bf16, bf16, bf16);

MAKE_GEMM_TEST(Testbf8xfp8xbf16xbf16xfp8, bf8, fp8, bf16, bf16, fp8);

MAKE_GEMM_TEST(Testbf8xfp8xbf16xbf16xbf8, bf8, fp8, bf16, bf16, bf8);

MAKE_GEMM_TEST(Testfp8xfp8xfp16xfp16xfp8, fp8, fp8, fp16, fp16, fp8);

static inline auto TN(const Layout& layout) {
  static const char* map[2][2] = {{"NN", "NT"}, {"TN", "TT"}};
  return std::string(map[layout.first][layout.second]);
}

static inline auto MKN(const std::tuple<size_t, size_t, size_t>& shape) {
  return std::to_string(std::get<0>(shape)) + "x" + std::to_string(std::get<1>(shape)) + "x" +
         std::to_string(std::get<2>(shape));
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, GEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes),
                                            ::testing::Values(false, true),   //use bias
                                            ::testing::Values(false, true),   //use_gelu
                                            ::testing::ValuesIn(kLayouts),    //transa,transb
                                            ::testing::Values(false, true)),  //use mxfp8
                         [](const testing::TestParamInfo<GEMMTestSuite::ParamType>& info) {
                           return MKN(std::get<0>(info.param)) + "x" +
                                  std::to_string(std::get<1>(info.param)) + "x" +
                                  std::to_string(std::get<2>(info.param)) + "x" +
                                  TN(std::get<3>(info.param)) + "x" +
                                  (std::get<4>(info.param) ? "M" : "S");
                         });

#ifdef __HIP_PLATFORM_AMD__
class DqGEMMTestSuite: public GEMMTestSuite {};

#define MAKE_DQ_GEMM_TEST(NAME_, A_, B_, D_)            \
  TEST_P(DqGEMMTestSuite, NAME_) {                      \
    MAKE_TEST_PARAMS(test_params);                      \
    using A_Type = A_;                                  \
    using B_Type = B_;                                  \
    using D_Type = D_;                                  \
    performDqTest<A_Type, B_Type, D_Type>(test_params); \
  }

MAKE_DQ_GEMM_TEST(Testfp8xfp8xfp16, fp8, fp8, fp16)

INSTANTIATE_TEST_SUITE_P(OperatorTest, DqGEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes_mxfp8),
                                            ::testing::Values(false),       // bias - unused
                                            ::testing::Values(false),       // gelu - unused
                                            ::testing::ValuesIn(kLayouts),  //transa,transb
                                            ::testing::Values(true)),       //use mxfp8
                         [](const testing::TestParamInfo<DqGEMMTestSuite::ParamType>& info) {
                           return MKN(std::get<0>(info.param)) + "x" + TN(std::get<3>(info.param));
                         });
#endif  // __HIP_PLATFORM_AMD__
