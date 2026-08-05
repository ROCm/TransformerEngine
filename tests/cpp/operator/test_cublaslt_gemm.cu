/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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
#include <transformer_engine/swizzle.h>
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
  {32, 128, 16},
  {256, 256, 256},
  {768, 3072, 4096},
  {4096, 16384, 4096},
};

// ============================================================================
// Production LLM MXFP8 GEMM shapes.
// ============================================================================

struct ProdGemmConfig {
  const char* label;
  size_t m;
  size_t n;
  size_t k;
  bool transa;
  bool transb;
};

static const ProdGemmConfig prod_gemm_sweep[] = {
    // Format: label, M, N, K, transa, transb
    // DeepSeek3
    {"DeepSeek3_Linear0_fwd_mbs1_TN", 4096, 1536, 7168, true, false},
    {"DeepSeek3_Linear0_fwd_mbs2_TN", 8192, 1536, 7168, true, false},
    {"DeepSeek3_Linear0_fwd_mbs4_TN", 16384, 1536, 7168, true, false},
    {"DeepSeek3_Linear1_fwd_mbs1_TN", 4096, 576, 7168, true, false},
    {"DeepSeek3_Linear1_fwd_mbs2_TN", 8192, 576, 7168, true, false},
    {"DeepSeek3_Linear1_fwd_mbs4_TN", 16384, 576, 7168, true, false},
    {"DeepSeek3_LNLinear0_fwd_mbs1_TN", 4096, 24576, 1536, true, false},
    {"DeepSeek3_LNLinear0_fwd_mbs2_TN", 8192, 24576, 1536, true, false},
    {"DeepSeek3_LNLinear0_fwd_mbs4_TN", 16384, 24576, 1536, true, false},
    {"DeepSeek3_LNLinear1_fwd_mbs1_TN", 4096, 32768, 512, true, false},
    {"DeepSeek3_LNLinear1_fwd_mbs2_TN", 8192, 32768, 512, true, false},
    {"DeepSeek3_LNLinear1_fwd_mbs4_TN", 16384, 32768, 512, true, false},
    {"DeepSeek3_Linear_attn_fwd_mbs1_TN", 4096, 7168, 16384, true, false},
    {"DeepSeek3_Linear_attn_fwd_mbs2_TN", 8192, 7168, 16384, true, false},
    {"DeepSeek3_Linear_attn_fwd_mbs4_TN", 16384, 7168, 16384, true, false},
    {"DeepSeek3_LNMLP_gateup_fwd_mbs1_TN", 4096, 36864, 7168, true, false},
    {"DeepSeek3_LNMLP_gateup_fwd_mbs2_TN", 8192, 36864, 7168, true, false},
    {"DeepSeek3_LNMLP_gateup_fwd_mbs4_TN", 16384, 36864, 7168, true, false},
    {"DeepSeek3_LNMLP_down_fwd_mbs1_TN", 4096, 7168, 18432, true, false},
    {"DeepSeek3_LNMLP_down_fwd_mbs2_TN", 8192, 7168, 18432, true, false},
    {"DeepSeek3_LNMLP_down_fwd_mbs4_TN", 16384, 7168, 18432, true, false},
    {"DeepSeek3_ExpertMLP_gu_fwd_mbs1_TN", 4096, 4096, 7168, true, false},
    {"DeepSeek3_ExpertMLP_gu_fwd_mbs2_TN", 8192, 4096, 7168, true, false},
    {"DeepSeek3_ExpertMLP_gu_fwd_mbs4_TN", 16384, 4096, 7168, true, false},
    {"DeepSeek3_ExpertMLP_dn_fwd_mbs1_TN", 4096, 7168, 2048, true, false},
    {"DeepSeek3_ExpertMLP_dn_fwd_mbs2_TN", 8192, 7168, 2048, true, false},
    {"DeepSeek3_ExpertMLP_dn_fwd_mbs4_TN", 16384, 7168, 2048, true, false},
    {"DeepSeek3_LNMLP_down_wgrad_mbs1_NT", 18432, 7168, 4096, false, true},
    {"DeepSeek3_LNMLP_down_wgrad_mbs2_NT", 18432, 7168, 8192, false, true},
    {"DeepSeek3_LNMLP_down_wgrad_mbs4_NT", 18432, 7168, 16384, false, true},
    {"DeepSeek3_LNMLP_gateup_wgrad_mbs1_NT", 7168, 36864, 4096, false, true},
    {"DeepSeek3_LNMLP_gateup_wgrad_mbs2_NT", 7168, 36864, 8192, false, true},
    {"DeepSeek3_LNMLP_gateup_wgrad_mbs4_NT", 7168, 36864, 16384, false, true},
    {"DeepSeek3_Linear_attn_wgrad_mbs1_NT", 16384, 7168, 4096, false, true},
    {"DeepSeek3_Linear_attn_wgrad_mbs2_NT", 16384, 7168, 8192, false, true},
    {"DeepSeek3_LNLinear1_wgrad_mbs1_NT", 512, 32768, 4096, false, true},
    {"DeepSeek3_LNLinear1_wgrad_mbs2_NT", 512, 32768, 8192, false, true},
    {"DeepSeek3_LNLinear1_wgrad_mbs4_NT", 512, 32768, 16384, false, true},
    {"DeepSeek3_LNLinear0_wgrad_mbs1_NT", 1536, 24576, 4096, false, true},
    {"DeepSeek3_LNLinear0_wgrad_mbs2_NT", 1536, 24576, 8192, false, true},
    {"DeepSeek3_LNLinear0_wgrad_mbs4_NT", 1536, 24576, 16384, false, true},
    {"DeepSeek3_Linear1_wgrad_mbs1_NT", 7168, 576, 4096, false, true},
    {"DeepSeek3_Linear1_wgrad_mbs2_NT", 7168, 576, 8192, false, true},
    {"DeepSeek3_Linear1_wgrad_mbs4_NT", 7168, 576, 16384, false, true},
    {"DeepSeek3_Linear0_wgrad_mbs1_NT", 7168, 1536, 4096, false, true},
    {"DeepSeek3_Linear0_wgrad_mbs2_NT", 7168, 1536, 8192, false, true},
    {"DeepSeek3_Linear0_wgrad_mbs4_NT", 7168, 1536, 16384, false, true},
    {"DeepSeek3_ExpertMLP_gu_dgrad_mbs1_NN", 4096, 7168, 4096, false, false},
    {"DeepSeek3_ExpertMLP_gu_dgrad_mbs2_NN", 8192, 7168, 4096, false, false},
    {"DeepSeek3_ExpertMLP_gu_wgrad_mbs1_NT", 7168, 4096, 4096, false, true},
    {"DeepSeek3_ExpertMLP_gu_wgrad_mbs2_NT", 7168, 4096, 8192, false, true},
    {"DeepSeek3_ExpertMLP_gu_wgrad_mbs4_NT", 7168, 4096, 16384, false, true},
    {"DeepSeek3_ExpertMLP_dn_dgrad_mbs1_NN", 4096, 2048, 7168, false, false},
    {"DeepSeek3_ExpertMLP_dn_dgrad_mbs2_NN", 8192, 2048, 7168, false, false},
    {"DeepSeek3_ExpertMLP_dn_dgrad_mbs4_NN", 16384, 2048, 7168, false, false},
    {"DeepSeek3_ExpertMLP_dn_wgrad_mbs1_NT", 2048, 7168, 4096, false, true},
    {"DeepSeek3_ExpertMLP_dn_wgrad_mbs2_NT", 2048, 7168, 8192, false, true},
    {"DeepSeek3_ExpertMLP_dn_wgrad_mbs4_NT", 2048, 7168, 16384, false, true},
    // DeepSeek4 (from https://amd-hub.atlassian.net/browse/AIHPBLAS-3861)
    {"DeepSeek4_M6144_N32_K7168_TN", 6144, 32, 7168, true, false},
    {"DeepSeek4_M6144_N64_K7168_TN", 6144, 64, 7168, true, false},
    {"DeepSeek4_M6144_N96_K7168_TN", 6144, 96, 7168, true, false},
    {"DeepSeek4_M6144_N128_K7168_TN", 6144, 128, 7168, true, false},
    {"DeepSeek4_M6144_N160_K7168_TN", 6144, 160, 7168, true, false},
    {"DeepSeek4_M6144_N192_K7168_TN", 6144, 192, 7168, true, false},
    {"DeepSeek4_M6144_N224_K7168_TN", 6144, 224, 7168, true, false},
    {"DeepSeek4_M6144_N256_K7168_TN", 6144, 256, 7168, true, false},
    {"DeepSeek4_M6144_N288_K7168_TN", 6144, 288, 7168, true, false},
    {"DeepSeek4_M6144_N320_K7168_TN", 6144, 320, 7168, true, false},
    {"DeepSeek4_M6144_N352_K7168_TN", 6144, 352, 7168, true, false},
    {"DeepSeek4_M6144_N384_K7168_TN", 6144, 384, 7168, true, false},
    {"DeepSeek4_M6144_N416_K7168_TN", 6144, 416, 7168, true, false},
    {"DeepSeek4_M6144_N448_K7168_TN", 6144, 448, 7168, true, false},
    {"DeepSeek4_M6144_N480_K7168_TN", 6144, 480, 7168, true, false},
    {"DeepSeek4_M6144_N512_K7168_TN", 6144, 512, 7168, true, false},
    {"DeepSeek4_M6144_N544_K7168_TN", 6144, 544, 7168, true, false},
    {"DeepSeek4_M6144_N576_K7168_TN", 6144, 576, 7168, true, false},
    {"DeepSeek4_M6144_N640_K7168_TN", 6144, 640, 7168, true, false},
    {"DeepSeek4_M6144_N800_K7168_TN", 6144, 800, 7168, true, false},
    {"DeepSeek4_M6144_N832_K7168_TN", 6144, 832, 7168, true, false},
    {"DeepSeek4_M7168_N32_K3072_TN", 7168, 32, 3072, true, false},
    {"DeepSeek4_M7168_N64_K3072_TN", 7168, 64, 3072, true, false},
    {"DeepSeek4_M7168_N96_K3072_TN", 7168, 96, 3072, true, false},
    {"DeepSeek4_M7168_N128_K3072_TN", 7168, 128, 3072, true, false},
    {"DeepSeek4_M7168_N160_K3072_TN", 7168, 160, 3072, true, false},
    {"DeepSeek4_M7168_N192_K3072_TN", 7168, 192, 3072, true, false},
    {"DeepSeek4_M7168_N224_K3072_TN", 7168, 224, 3072, true, false},
    {"DeepSeek4_M7168_N256_K3072_TN", 7168, 256, 3072, true, false},
    {"DeepSeek4_M7168_N288_K3072_TN", 7168, 288, 3072, true, false},
    {"DeepSeek4_M7168_N320_K3072_TN", 7168, 320, 3072, true, false},
    {"DeepSeek4_M7168_N352_K3072_TN", 7168, 352, 3072, true, false},
    {"DeepSeek4_M7168_N384_K3072_TN", 7168, 384, 3072, true, false},
    {"DeepSeek4_M7168_N416_K3072_TN", 7168, 416, 3072, true, false},
    {"DeepSeek4_M7168_N448_K3072_TN", 7168, 448, 3072, true, false},
    {"DeepSeek4_M7168_N480_K3072_TN", 7168, 480, 3072, true, false},
    {"DeepSeek4_M7168_N512_K3072_TN", 7168, 512, 3072, true, false},
    {"DeepSeek4_M7168_N544_K3072_TN", 7168, 544, 3072, true, false},
    {"DeepSeek4_M7168_N576_K3072_TN", 7168, 576, 3072, true, false},
    {"DeepSeek4_M7168_N640_K3072_TN", 7168, 640, 3072, true, false},
    {"DeepSeek4_M7168_N800_K3072_TN", 7168, 800, 3072, true, false},
    {"DeepSeek4_M7168_N832_K3072_TN", 7168, 832, 3072, true, false},
    // Qwen3
    {"Qwen3_LNLinear_QKV_fwd_mbs1_TN", 4096, 9216, 4096, true, false},
    {"Qwen3_LNLinear_QKV_fwd_mbs2_TN", 8192, 9216, 4096, true, false},
    {"Qwen3_LNLinear_QKV_fwd_mbs4_TN", 16384, 9216, 4096, true, false},
    {"Qwen3_Linear_attn_fwd_mbs1_TN", 4096, 4096, 8192, true, false},
    {"Qwen3_Linear_attn_fwd_mbs2_TN", 8192, 4096, 8192, true, false},
    {"Qwen3_Linear_attn_fwd_mbs4_TN", 16384, 4096, 8192, true, false},
    {"Qwen3_Linear_attn_wgrad_mbs1_NT", 8192, 4096, 4096, false, true},
    {"Qwen3_Linear_attn_wgrad_mbs4_NT", 8192, 4096, 16384, false, true},
    {"Qwen3_LNLinear_QKV_wgrad_mbs2_NT", 4096, 9216, 8192, false, true},
    {"Qwen3_LNLinear_QKV_wgrad_mbs4_NT", 4096, 9216, 16384, false, true},
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

// Swizzle MXFP8 scale_inv of a test::Tensor in-place for gfx1250.
static void swizzle_mxfp8_scales(test::Tensor &t, bool rowwise) {
  using namespace transformer_engine;
  void *scale_ptr = rowwise ? t.rowwise_scale_inv_dptr()
                            : t.columnwise_scale_inv_dptr();
  if (!scale_ptr) return;
  const NVTEShape scale_shape = rowwise ? t.rowwise_scale_inv_shape()
                                        : t.columnwise_scale_inv_shape();
  const NVTEShape data_shape = rowwise ? t.rowwise_shape()
                                       : t.columnwise_shape();
  size_t num_scales = 1;
  for (size_t d = 0; d < scale_shape.ndim; d++) num_scales *= scale_shape.data[d];
  uint8_t *d_tmp = nullptr;
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
  t.set_with_gemm_swizzled_scales(true);
  NVTE_CHECK_CUDA(cudaFree(d_tmp));
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


template <typename A_Type, typename B_Type, typename Bias_Type, typename Gelu_Type, typename D_Type>
void performTest(const TestParams& params) {
  DType atype = TypeInfo<A_Type>::dtype;
  DType btype = TypeInfo<B_Type>::dtype;
  DType bias_type = TypeInfo<Bias_Type>::dtype;
  DType gelu_type = TypeInfo<Gelu_Type>::dtype;
  DType dtype = TypeInfo<D_Type>::dtype;

  const bool has_fp8 = isFp8Type(atype) || isFp8Type(btype);
  const bool use_mxfp8 = params.scaling_mode == NVTEScalingMode::NVTE_MXFP8_1D_SCALING;
  const bool use_hipkittens_mxfp8 = use_mxfp8 && !params.force_hipblaslt;

  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, 0);

  if (use_mxfp8)
  {
    if (!has_fp8) {
      GTEST_SKIP() << "MXFP8 scaling mode requires Float8 types";
    }
    if (params.m % 16 || params.n % 16) {
      GTEST_SKIP() << "MXFP8 requires M & N to be multiples of 16";
    }
    size_t required_k_multiple = 128;
  #ifdef __HIP_PLATFORM_AMD__
    required_k_multiple = (prop.major == 12 && prop.minor == 5) ? 32 : 128;
  #endif
    if (params.k % required_k_multiple) {
      GTEST_SKIP() << "MXFP8 requires K to be a multiple of " << required_k_multiple;
    }
    if (use_hipkittens_mxfp8 && (params.m % 256 || params.n % 256 || params.k < 256)) {
      GTEST_SKIP() << "HipKittens requires M and N 256-aligned, K >= 256";
    }
  }

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
    const bool fp8_supported = (prop.major == 9 && prop.minor >= 4) || prop.major >= 12;
    if (!fp8_supported) {
      GTEST_SKIP() << "FP8 is not supported in current config";
    }
    const bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5) || prop.major >= 12;
    if (use_mxfp8 && !mxfp8_supported) {
      GTEST_SKIP() << "MXFP8 is not supported in current config";
    }
    if (!use_hipkittens_mxfp8 && params.use_bias) {
      GTEST_SKIP() << "MXFP8 GEMM with bias is not supported by hipBLASLt";
    }
    if (params.use_gelu && !fp8_gelu_fusion_config && !use_hipkittens_mxfp8) {
      GTEST_SKIP() << "FP8 GEMM with GELU is not supported in current config";
    }
    if (params.use_bias && dtype == DType::kFloat16) {
      GTEST_SKIP() << "FP8 GEMM with bias and FP16 output is not supported";
    }
  }

  if (prop.major == 9 && prop.minor == 5) //gfx950 specific hipblasLt limitations
  {
    if (isFp8Type(dtype)) {
      GTEST_SKIP() << "GEMM with float8 output is not supported";
    }
    if (params.use_gelu && dtype == DType::kBFloat16 && !use_hipkittens_mxfp8) {
      GTEST_SKIP() << "BF16 GEMM with GELU is not supported in current config";
    }
    if constexpr ((std::is_same_v<A_Type, bf8> || std::is_same_v<B_Type, bf8>) &&
                   std::is_same_v<D_Type, fp32>) {
      if (params.use_bias) {
        GTEST_SKIP() << "FP8 GEMM with bias is not supported in current config";
      }
    }
  }
  else if (prop.major == 9 && prop.minor == 4) //gfx942 specific hipblasLt limitations
  {
#if HIP_VERSION < 70100000
    if (params.use_gelu && dtype == DType::kBFloat16 && !params.transa) {
      GTEST_SKIP() << "BF16 GEMM with GELU is not supported in current config";
    }
#endif
    if constexpr (std::is_same_v<D_Type, fp8> && std::is_same_v<Bias_Type, bf16>) {
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

  size_t workspace_size = 33'554'432;
#ifdef __HIP_PLATFORM_AMD__
  if ((prop.major == 9 && prop.minor == 5) || prop.major >= 12) {
    workspace_size = 67'108'864;
  }
#endif
  Tensor Workspace("Workspace", TShape{ workspace_size }, DType::kByte);

  //perform the reference gemm on GPU (before swizzle, which modifies scales in-place)
  Tensor RefD("RefD", TShape{ params.n, params.m }, dtype);
  Tensor RefPreGeluOut;

  if (params.use_gelu) {
    RefPreGeluOut = Tensor("RefPreGeluOut", TShape{ params.n, params.m }, gelu_type);
  }

  run_reference<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(
    params,
    A,
    B,
    params.use_bias ? &bias : nullptr,
    D,
    RefD,
    params.use_gelu ? &RefPreGeluOut : nullptr);

  // On gfx1250, hipBLASLt MXFP8 kernels expect pre-swizzled scales.
  if (use_mxfp8 && prop.major == 12 && prop.minor == 5) {
    swizzle_mxfp8_scales(A, !a_colwise);
    swizzle_mxfp8_scales(B, !b_colwise);
  }

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

  // check if error message happens in running                             
  (void)cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  //compare results
  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(dtype)) {
    const float ref_amax_d = RefD.amax();
    compareResults("D_amax", D.amax(), ref_amax_d, atol_amax, rtol_amax);
  }

  auto [atol, rtol] = getTestTolerances(dtype, has_fp8, use_mxfp8);
  RefD.to_cpu();
  compareResults("D", D, RefD.rowwise_cpu_dptr<D_Type>(), true, atol, rtol);

  if(params.use_gelu){
    auto [atol, rtol] = getTestTolerances(gelu_type, has_fp8, use_mxfp8);
    RefPreGeluOut.to_cpu();
    compareResults("gelu", pre_gelu_out, RefPreGeluOut.rowwise_cpu_dptr<Gelu_Type>(), true, atol, rtol);
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

  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, 0);

  if (params.m % 16 || params.n % 16) {
    GTEST_SKIP() << "MXFP8 requires M & N to be multiples of 16";
  }
  size_t required_k_multiple = 128;
#ifdef __HIP_PLATFORM_AMD__
  required_k_multiple = (prop.major == 12 && prop.minor == 5) ? 32 : 128;
#endif
  if (params.k % required_k_multiple) {
    GTEST_SKIP() << "MXFP8 requires K to be a multiple of " << required_k_multiple;
  }

  bool mxfp8_supported = (prop.major == 9 && prop.minor >= 5) || prop.major >= 12;
  const bool use_hipkittens_mxfp8 = !params.force_hipblaslt;
  if (!mxfp8_supported) {
    GTEST_SKIP() << "MXFP8 is not supported in current config";
  }
  if (params.use_bias || params.use_gelu) {
    GTEST_SKIP() << "DqGEMMTestSuite does not yet have reference for bias/gelu epilogues";
  }
  if (use_hipkittens_mxfp8 && (params.m % 256 || params.n % 256 || params.k % 128 || params.k < 256)) {
    GTEST_SKIP() << "HipKittens requires M and N 256-aligned, K >= 256";
  }

  // hipBLASLt on gfx950 produces incorrect results for certain MXFP8
  // GEMMs with non-TN layouts.
  if (prop.major == 9 && prop.minor == 5) {
    const bool is_NT = !params.transa && params.transb;
    if (is_NT && params.m == 7168 && params.n == 576) {
      GTEST_SKIP() << "hipBLASLt MXFP8 non-TN GEMM with certain M/N is not supported on gfx950";
    }
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

  // On gfx1250, hipBLASLt MXFP8 kernels expect pre-swizzled scales.
  const bool a_colwise = !params.transa;
  const bool b_colwise = params.transb;
  if (prop.major == 12 && prop.minor == 5) {
    swizzle_mxfp8_scales(A_fp8, !a_colwise);
    swizzle_mxfp8_scales(B_fp8, !b_colwise);
  }

  Tensor bias;
  Tensor pre_gelu_out;

  Tensor Workspace("Workspace", TShape{67'108'864}, DType::kByte);

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
  bool force_hipblaslt_ = std::get<5>(GetParam());                              \
  if (force_hipblaslt_) {                                                       \
    setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP8", "1", 1);                            \
  } else {                                                                      \
    setenv("NVTE_ROCM_USE_HIPBLASLT_MXFP8", "0", 1);                            \
  }                                  \
  TestParams P_ = {.m = std::get<0>(std::get<0>(GetParam())),                   \
                   .k = std::get<1>(std::get<0>(GetParam())),                   \
                   .n = std::get<2>(std::get<0>(GetParam())),                   \
                   .use_bias = std::get<1>(GetParam()),                         \
                   .use_gelu = std::get<2>(GetParam()),                         \
                   .transa = std::get<3>(GetParam()).first,                     \
                   .transb = std::get<3>(GetParam()).second,                    \
                   .scaling_mode = std::get<4>(GetParam())                      \
                                 ? NVTEScalingMode::NVTE_MXFP8_1D_SCALING       \
                                 : NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING,\
                   .force_hipblaslt = force_hipblaslt_}

// <m, k, n>, use_bias, use_gelu, Layout, fp8_scaling, force_hipblaslt
class GEMMTestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<size_t, size_t, size_t>, bool, bool, Layout, NVTEScalingMode, bool>> {};

#define MAKE_GEMM_TEST(SUITE_, NAME_, A_, B_, BIAS_, GELU_, D_)              \
  TEST_P(SUITE_, NAME_) {                                                   \
    MAKE_TEST_PARAMS(test_params);                                          \
    using A_Type = A_;                                                      \
    using B_Type = B_;                                                      \
    using Bias_Type = BIAS_;                                                \
    using Gelu_Type = GELU_;                                                \
    using D_Type = D_;                                                      \
    performTest<A_Type, B_Type, Bias_Type, Gelu_Type, D_Type>(test_params); \
  }

// Non-FP8 types
MAKE_GEMM_TEST(GEMMTestSuite, Testfp32xfp32xfp32xfp32xfp32, fp32, fp32, fp32, fp32, fp32);
MAKE_GEMM_TEST(GEMMTestSuite, Testfp16xfp16xfp16xfp16xfp16, fp16, fp16, fp16, fp16, fp16);
MAKE_GEMM_TEST(GEMMTestSuite, Testbf16xbf16xbf16xbf16xbf16, bf16, bf16, bf16, bf16, bf16);

// FP8 types — used by both OperatorTest and OperatorTestMXFP8 suites
class FP8GEMMTestSuite
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<size_t, size_t, size_t>, bool, bool, Layout, NVTEScalingMode, bool>> {};

MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp32, fp8, fp8, bf16, bf16, fp32);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp16, fp8, fp8, bf16, bf16, fp16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xbf16xbf16xbf16, fp8, fp8, bf16, bf16, bf16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xbf16xbf16xfp8, fp8, fp8, bf16, bf16, fp8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xbf16xbf16xbf8, fp8, fp8, bf16, bf16, bf8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp32, fp8, bf8, bf16, bf16, fp32);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp16, fp8, bf8, bf16, bf16, fp16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xbf8xbf16xbf16xbf16, fp8, bf8, bf16, bf16, bf16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xbf8xbf16xbf16xfp8, fp8, bf8, bf16, bf16, fp8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xbf8xbf16xbf16xbf8, fp8, bf8, bf16, bf16, bf8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp32, bf8, fp8, bf16, bf16, fp32);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp16, bf8, fp8, bf16, bf16, fp16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testbf8xfp8xbf16xbf16xbf16, bf8, fp8, bf16, bf16, bf16);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testbf8xfp8xbf16xbf16xfp8, bf8, fp8, bf16, bf16, fp8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testbf8xfp8xbf16xbf16xbf8, bf8, fp8, bf16, bf16, bf8);
MAKE_GEMM_TEST(FP8GEMMTestSuite, Testfp8xfp8xfp16xfp16xfp8, fp8, fp8, fp16, fp16, fp8);

static inline auto TN(const Layout& layout) {
  static const char* map[2][2] = {{"NN", "NT"}, {"TN", "TT"}};
  return std::string(map[layout.first][layout.second]);
}

static inline auto MKN(const std::tuple<size_t, size_t, size_t>& shape) {
  return std::to_string(std::get<0>(shape)) + "x" + std::to_string(std::get<1>(shape)) + "x" +
         std::to_string(std::get<2>(shape));
}

static std::string GEMMTestName(const testing::TestParamInfo<GEMMTestSuite::ParamType>& info) {
  return MKN(std::get<0>(info.param)) + "x" +
         std::to_string(std::get<1>(info.param)) + "x" +
         std::to_string(std::get<2>(info.param)) + "x" +
         TN(std::get<3>(info.param)) + "x" +
         (std::get<4>(info.param) ? "M" : "S") + "x" +
         (std::get<5>(info.param) ? "HB" : "HK");
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, GEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes),
                                            ::testing::Values(false, true),   //use bias
                                            ::testing::Values(false, true),   //use_gelu
                                            ::testing::ValuesIn(kLayouts),    //transa,transb
                                            ::testing::Values(false),         //use mxfp8
                                            ::testing::Values(false)),        //force hipblaslt
                         GEMMTestName);

INSTANTIATE_TEST_SUITE_P(OperatorTestFP8, FP8GEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes),
                                            ::testing::Values(false, true),   //use bias
                                            ::testing::Values(false, true),   //use_gelu
                                            ::testing::ValuesIn(kLayouts),    //transa,transb
                                            ::testing::Values(false),         //use mxfp8
                                            ::testing::Values(false)),        //force hipblaslt
                         GEMMTestName);

INSTANTIATE_TEST_SUITE_P(OperatorTestMXFP8, FP8GEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes),
                                            ::testing::Values(false, true),   //use bias
                                            ::testing::Values(false, true),   //use_gelu
                                            ::testing::ValuesIn(kLayouts),    //transa,transb
                                            ::testing::Values(true),          //use mxfp8
                                            ::testing::Values(false, true)),  //force hipblaslt
                         GEMMTestName);

#ifdef __HIP_PLATFORM_AMD__
class DqGEMMTestSuite: public FP8GEMMTestSuite {};

#define MAKE_DQ_GEMM_TEST(NAME_, A_, B_, D_)            \
  TEST_P(DqGEMMTestSuite, NAME_) {                      \
    MAKE_TEST_PARAMS(test_params);                      \
    using A_Type = A_;                                  \
    using B_Type = B_;                                  \
    using D_Type = D_;                                  \
    performDqTest<A_Type, B_Type, D_Type>(test_params); \
  }

MAKE_DQ_GEMM_TEST(Testfp8xfp8xfp16, fp8, fp8, fp16)

INSTANTIATE_TEST_SUITE_P(OperatorTestMXFP8, DqGEMMTestSuite,
                         ::testing::Combine(::testing::ValuesIn(test_case_sizes_mxfp8),
                                            ::testing::Values(false),        // use bias
                                            ::testing::Values(false),        // use gelu
                                            ::testing::ValuesIn(kLayouts),   // transa,transb
                                            ::testing::Values(true),         // use mxfp8
                                            ::testing::Values(false, true)), // force hipblaslt
                         [](const testing::TestParamInfo<DqGEMMTestSuite::ParamType>& info) {
                           return MKN(std::get<0>(info.param)) + "x" +
                                  TN(std::get<3>(info.param)) + "x" +
                                  (std::get<5>(info.param) ? "HB" : "HK");
                         });

// ============================================================================
// Production GEMM shape instantiations (run with --gtest_filter='ProdGemm*')
// ============================================================================

class ProdGEMMTestSuite : public ::testing::TestWithParam<ProdGemmConfig> {};

TEST_P(ProdGEMMTestSuite, TestMxfp8Dq) {
  const auto& config = GetParam();

  TestParams params = {.m = config.m, .k = config.k, .n = config.n,
                       .use_bias = false, .use_gelu = false,
                       .transa = config.transa, .transb = config.transb,
                       .scaling_mode = NVTEScalingMode::NVTE_MXFP8_1D_SCALING};

  performDqTest<fp8, fp8, bf16>(params);
}

static auto prodTestName = [](const testing::TestParamInfo<ProdGemmConfig>& info) {
  return std::string(info.param.label);
};

INSTANTIATE_TEST_SUITE_P(ProdGemmSweep, ProdGEMMTestSuite,
    ::testing::ValuesIn(prod_gemm_sweep),
    prodTestName);

TEST(InputGenTest, FillUniform_DoesNotGetOverwrittenByFromCpu) {
  const size_t rows = 128;
  const size_t cols = 256;
  const size_t N = rows * cols;

  test::Tensor t("fillUniform_regression_fp32",
           std::vector<size_t>{rows, cols},
           transformer_engine::DType::kFloat32,
           /*rowwise=*/true,
           /*columnwise=*/false);

  // Tensor constructor initializes CPU mirror + device to zero.
  // If GPU generation happens but CPU mirror is not updated,
  // any later test::Tensor::from_cpu() will overwrite device back to zeros.
  fillUniform(&t);

  // Check the CPU mirror has *actual* generated values, not all zeros
  const float* cpu = t.rowwise_cpu_dptr<float>();

  bool any_nonzero = false;
  for (size_t i = 0; i < N; ++i) {
    any_nonzero |= (cpu[i] != 0.0f);
    if (any_nonzero)
      break;
  }

  ASSERT_TRUE(any_nonzero) << "CPU mirror is all zeros. "
                           << "Likely GPU-generated data got overwritten by from_cpu().";

  // Check device matches CPU mirror after fillUniform completes
  std::vector<float> dev(N, 0.0f);
  NVTE_CHECK_CUDA(cudaMemcpy(dev.data(),
                           t.rowwise_dptr(),
                           N * sizeof(float),
                           cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dev[i], cpu[i]) << "Mismatch at i=" << i
                              << " dev=" << dev[i] << " cpu=" << cpu[i];
  }
}

#endif  // __HIP_PLATFORM_AMD__
