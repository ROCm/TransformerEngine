/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>
#include <cmath>
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

//  A, B, Bias, Gelu, D
//  Bias type choose as bf16 in use_fp8, D_type otherwise
//  Gelu type the same as Bias_Type
//  {DType::kFloat32, DType::kFloat32, DType::kFloat32, DType::kFloat32, DType::kFloat32},
//  {DType::kFloat16, DType::kFloat16, DType::kFloat16, DType::kFloat16, DType::kFloat16},
//  {DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16},
//  {DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat32},
//  {DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat16},
//  {DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16},
//  {DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E4M3},
//  {DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E5M2},
//  {DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kBFloat16, DType::kBFloat16, DType::kFloat32},
//  {DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kBFloat16, DType::kBFloat16, DType::kFloat16},
//  {DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16},
//  {DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E4M3},
//  {DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E5M2},
//  {DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat32},
//  {DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat16},
//  {DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16},
//  {DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E4M3},
//  {DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kBFloat16, DType::kBFloat16, DType::kFloat8E5M2},
}  // namespace



// <A_type, B_type, Bias_Type, Gelu_Type D_type>, <m, k, n>
class GEMMTestSuite 
  :public ::testing::TestWithParam<std::tuple<
                                    std::tuple<size_t, size_t, size_t>, bool, bool>>{};

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
  float* ref_d_amax,
  Gelu_Type* ref_gelu_data){

  *ref_d_amax = 0;
  for(size_t ii = 0; ii < m; ii++){
    for(size_t jj = 0; jj < n; jj++){
      float val = 0;
      for(size_t kk = 0; kk < k; kk++){
        val += a_scale_inv*b_scale_inv*((float)a_data[ii + kk*m])*((float)b_data[kk + jj*k]);
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
        *ref_d_amax = std::max<float>(*ref_d_amax, std::fabs(val));
      }
    }
  }
}

template <typename A_Type, typename B_Type, typename Bias_Type, typename Gelu_Type, typename D_Type>
void performTest(bool use_bias, bool use_gelu, const size_t m, const size_t k, const size_t n) {
  DType atype = TypeInfo<A_Type>::dtype;
  DType btype = TypeInfo<B_Type>::dtype;
  DType bias_type = TypeInfo<Bias_Type>::dtype;
  DType gelu_type = TypeInfo<Gelu_Type>::dtype;
  DType dtype = TypeInfo<D_Type>::dtype;

  // pytorch tensor storage is row-major while cublas/rocblas is column-major
  Tensor A({ k, m }, atype);
  Tensor B({ n, k }, btype);
  Tensor D({ n, m }, dtype);
  Tensor bias;
  if(use_bias){
    bias = Tensor({m}, bias_type);
  }
  Tensor pre_gelu_out;
  if(use_gelu){
    pre_gelu_out = Tensor({ n, m }, gelu_type);
  }
  
  //initialize the data and scale inv of A, B
  fillUniform(&A);
  fillUniform(&B);
  if(use_bias){
    fillUniform(&bias);
  }
  //initialize the scale of D
  if(isFp8Type(dtype)){
    setRandomScale(&D);
  }
  bool transa = false;
  bool transb = false;
  bool grad = false;
  bool accumulate = false;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  //perform the gemm in GPU
  nvte_cublas_gemm(A.data(),
                   B.data(),
                   D.data(),
                   bias.data(),
                   pre_gelu_out.data(),
                   transa,
                   transb,
                   grad,
                   nullptr,
                   accumulate,
                   false,
                   prop.multiProcessorCount,
                   //default stream
                   0);
  //copy the output results from GPU memory to CPU memory
  D.to_cpu();
  if(use_gelu){
    pre_gelu_out.to_cpu();
  }

  //perform the gemm in CPU
  std::unique_ptr<D_Type[]> ref_D = std::make_unique<D_Type[]>(m*n);
  std::unique_ptr<Gelu_Type[]> ref_pre_gelu_out;
  if(use_gelu){
    ref_pre_gelu_out = std::make_unique<Gelu_Type[]>(m*n);
  }
  float ref_amax_d;
  compute_ref<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(
    A.cpu_dptr<A_Type>(), 
    B.cpu_dptr<B_Type>(), 
    A.scale_inv(),
    B.scale_inv(),
    use_bias? bias.cpu_dptr<Bias_Type>(): nullptr,
    D.scale(),
    m, k, n,
    ref_D.get(),
    &ref_amax_d,
    use_gelu? ref_pre_gelu_out.get(): nullptr);
  // check if error message happens in running                             
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  //compare results
  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(dtype)) {
    compareResults("D_amax", D.amax(), ref_amax_d, atol_amax, rtol_amax);
  }

  auto [atol, rtol] = getTolerances(dtype);
  //relax for certain prime number gemm
  if (dtype == DType::kFloat32) {
    atol = 1e-5;
  }
  compareResults("D", D, ref_D.get(), atol, rtol);

  if(use_gelu){
    auto [atol, rtol] = getTolerances(gelu_type);
    //relax for certain prime number gemm
    if (dtype == DType::kFloat32) {
      atol = 5e-6;
    }
    compareResults("gelu", pre_gelu_out, ref_pre_gelu_out.get(), atol, rtol);
  }
}

using fp32=float;
using fp8=fp8e4m3;
using bf8=fp8e5m2;
 
TEST_P(GEMMTestSuite, Testfp32xfp32xfp32xfp32xfp32) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp32;
  using B_Type = fp32;
  using Bias_Type = fp32;
  using Gelu_Type = fp32;
  using D_Type = fp32;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp16xfp16xfp16xfp16xfp16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp16;
  using B_Type = fp16;
  using Bias_Type = fp16;
  using Gelu_Type = fp16;
  using D_Type = fp16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf16xbf16xbf16xbf16xbf16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf16;
  using B_Type = bf16;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp32) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp32;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xfp8xbf16xbf16xbf16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xfp8xbf16xbf16xbf8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp32) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = bf8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp32;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = bf8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xbf8xbf16xbf16xbf16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = bf8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = bf8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testfp8xbf8xbf16xbf16xbf8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = fp8;
  using B_Type = bf8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp32) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp32;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf8xfp8xbf16xbf16xbf16) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf16;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = fp8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}

TEST_P(GEMMTestSuite, Testbf8xfp8xbf16xbf16xbf8) {
  using namespace transformer_engine;
  using namespace test;
 
  const size_t m = std::get<0>(std::get<0>(GetParam()));
  const size_t k = std::get<1>(std::get<0>(GetParam()));
  const size_t n = std::get<2>(std::get<0>(GetParam()));
  const bool use_bias = std::get<1>(GetParam());
  const bool use_gelu = std::get<2>(GetParam());

  using A_Type = bf8;
  using B_Type = fp8;
  using Bias_Type = bf16;
  using Gelu_Type = bf16;
  using D_Type = bf8;

  performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(use_bias, use_gelu, m, k, n);
}


INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GEMMTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(test_case_sizes),
        ::testing::Values(false, true), //use bias
        ::testing::Values(false, true)), //use_gelu
    [](const testing::TestParamInfo<GEMMTestSuite::ParamType>& info) {
      std::string name = std::to_string(std::get<0>(std::get<0>(info.param))) + "X" +
                         std::to_string(std::get<1>(std::get<0>(info.param))) + "X" +
                         std::to_string(std::get<2>(std::get<0>(info.param))) + "X" +
                         std::to_string(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param));
      return name;
    });
