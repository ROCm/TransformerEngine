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
  {768, 3072, 4096},
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


__device__ __host__ __forceinline__ float ref_gelu(float x){
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
  bool b_is_colwise)
{
  const size_t jj = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t ii = blockIdx.y * blockDim.y + threadIdx.y;

  const bool in_range = (ii < m) && (jj < n);

  float val = 0.0f;

  if (in_range) {
    for (size_t kk = 0; kk < k; ++kk) {
      // Indexing depends on which backing buffer we passed in
      const size_t a_idx =
          a_is_colwise ? (ii * k + kk)
                       : (transa ? (ii * k + kk) : (kk * m + ii));

      const size_t b_idx =
          b_is_colwise ? (jj * k + kk)
                       : (transb ? (kk * n + jj) : (jj * k + kk));

      float a_scale_inv_val = a_scale_inv_scalar;
      float b_scale_inv_val = b_scale_inv_scalar;

      if (a_scale_inv_mxfp8) {
        const size_t a_scale_idx =
            a_is_colwise ? (a_idx / 32)
                         : (transa ? (a_idx / 32) : ((kk / 32) * m + ii));

        const size_t b_scale_idx =
            b_is_colwise ? (b_idx / 32)
                         : (transb ? ((kk / 32) * n + jj) : (b_idx / 32));

        const float a_byte = static_cast<float>(a_scale_inv_mxfp8[a_scale_idx]);
        const float b_byte = static_cast<float>(b_scale_inv_mxfp8[b_scale_idx]);

        a_scale_inv_val = exp2f(a_byte - 127.0f);
        b_scale_inv_val = exp2f(b_byte - 127.0f);
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
};


template <typename A_Type, typename B_Type, typename Bias_Type,
          typename Gelu_Type, typename D_Type>
static void run_reference(
    const TestParams& params,
    const Tensor& A,
    const Tensor& B,
    const Tensor* Bias,                 // nullable
    float d_scale,
    std::unique_ptr<D_Type[]>& ref_D,   // m*n
    float* ref_amax_d,
    std::unique_ptr<Gelu_Type[]>& ref_pre_gelu_out) // nullable
{
  const bool use_mxfp8 = (params.scaling_mode == NVTE_MXFP8_1D_SCALING);

  Gelu_Type* ref_gelu_host = (params.use_gelu ? ref_pre_gelu_out.get() : nullptr);

  const bool is_fp8_output = test::isFp8Type(test::TypeInfo<D_Type>::dtype);

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

  if (use_mxfp8) {
    a_scale_dev = params.transa
        ? (const fp8e8m0*) A.rowwise_scale_inv_dptr()
        : (const fp8e8m0*) A.columnwise_scale_inv_dptr();

    b_scale_dev = params.transb
        ? (const fp8e8m0*) B.columnwise_scale_inv_dptr()
        : (const fp8e8m0*) B.rowwise_scale_inv_dptr();
  } else {
    a_scale_inv_scalar = A.rowwise_scale_inv();
    b_scale_inv_scalar = B.rowwise_scale_inv();
  }

  // optional bias device pointer
  const Bias_Type* bias_dev = nullptr;
  if (Bias) {
    bias_dev = static_cast<const Bias_Type*>(Bias->rowwise_dptr());
  }

  // allocate device outputs
  const size_t lenD = params.m * params.n;
  const size_t bytesD = lenD * sizeof(D_Type);

  D_Type* d_refD = nullptr;
  Gelu_Type* d_refGelu = nullptr;
  float* d_refAmax = nullptr;

  NVTE_CHECK_CUDA(cudaMalloc(&d_refD, bytesD));
  if (ref_gelu_host) {
    NVTE_CHECK_CUDA(cudaMalloc(&d_refGelu, lenD * sizeof(Gelu_Type)));
  }
  if (is_fp8_output && ref_amax_d) {
    NVTE_CHECK_CUDA(cudaMalloc(&d_refAmax, sizeof(float)));
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
          b_use_colwise);

  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  // copy outputs back
  NVTE_CHECK_CUDA(cudaMemcpy(ref_D.get(), d_refD, bytesD, cudaMemcpyDeviceToHost));

  if (ref_gelu_host) {
    NVTE_CHECK_CUDA(cudaMemcpy(ref_gelu_host, d_refGelu, lenD * sizeof(Gelu_Type),
                               cudaMemcpyDeviceToHost));
  }

  if (ref_amax_d) {
    if (is_fp8_output) {
      NVTE_CHECK_CUDA(cudaMemcpy(ref_amax_d, d_refAmax, sizeof(float),
                                 cudaMemcpyDeviceToHost));
    } else {
      *ref_amax_d = 0.0f;
    }
  }

  // cleanup
  NVTE_CHECK_CUDA(cudaFree(d_refD));
  if (d_refGelu)
    NVTE_CHECK_CUDA(cudaFree(d_refGelu));
  if (d_refAmax)
    NVTE_CHECK_CUDA(cudaFree(d_refAmax));
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
  }
  else if (use_fp8) {
    atol = 1e-3;
    rtol = std::max(rtol, 1e-2);
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

  #if HIP_VERSION < 70200000
    if (prop.major == 9 && prop.minor == 5 &&
        params.transa && !params.transb &&
        params.m == 2304 && params.k == 768 && params.n == 4096) {
      GTEST_SKIP() << "Skip TN 2304x768x4096 on gfx950 for ROCm < 7.2";
    }
  #endif

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
#if HIP_VERSION < 70100000
    if (params.use_gelu && dtype == DType::kBFloat16 && !params.transa) {
      GTEST_SKIP() << "BF16 GEMM with GELU is not supported in current config";
    }
#endif
    if constexpr (std::is_same<D_Type, fp8>::value && std::is_same<Bias_Type, bf16>::value) {
      if (params.use_bias && !fp8_gelu_fusion_config) {
        GTEST_SKIP() << "GEMM with BF16 bias and FP8 output is not supported in current config";
      }
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

  //perform the reference gemm on GPU
  std::unique_ptr<D_Type[]> ref_D = std::make_unique<D_Type[]>(params.m*params.n);
  std::unique_ptr<Gelu_Type[]> ref_pre_gelu_out;
  if(params.use_gelu){
    ref_pre_gelu_out = std::make_unique<Gelu_Type[]>(params.m*params.n);
  }

  float ref_amax_d;

  run_reference<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(
    params,
    A,
    B,
    params.use_bias ? &bias : nullptr,
    D.scale(),
    ref_D,
    &ref_amax_d,
    ref_pre_gelu_out);

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
