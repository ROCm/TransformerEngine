/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "amd_detail/hip_float8.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "benchmark_utils.h"

#include "transformer_engine/cast_hip.h"
#include "transformer_engine/transformer_engine_hip.h"

using namespace te_bench;
using namespace transformer_engine;
using fp8_e4m3 = test::fp8e4m3;

constexpr int kPadMultiple = 128;

static std::vector<int> generate_routed_tokens(int total_tokens, int num_experts,
                                                std::mt19937 &rng, bool skewed,
                                                double zipf_s = 0.7) {
  std::vector<double> weights(num_experts);
  if (skewed) {
    for (int i = 0; i < num_experts; i++) {
      weights[i] = 1.0 / std::pow(i + 1, zipf_s);
    }
  } else {
    std::fill(weights.begin(), weights.end(), 1.0);
  }
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  for (auto &w : weights) {
    w /= sum;
  }
  std::shuffle(weights.begin(), weights.end(), rng);

  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  std::vector<int> tokens(num_experts, 0);
  for (int i = 0; i < total_tokens; i++) {
    tokens[dist(rng)]++;
  }

  for (auto &t : tokens) {
    t = std::max(kPadMultiple, ((t + kPadMultiple - 1) / kPadMultiple) * kPadMultiple);
  }
  return tokens;
}

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X>
static void BM_GroupQuantizeMXFP8(benchmark::State &state) {
  const int num_experts  = state.range(0);
  const int cols         = state.range(1);
  const int total_tokens = state.range(2);
  const int skewed       = state.range(3);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  DType itype = std::is_same_v<IType, __half> ? DType::kFloat16 :
                (std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);
  DType otype = std::is_same_v<OType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;

  unsigned int seed = std::random_device{}();
  uint64_t config_hash = seed ^ (uint64_t(num_experts) * 2654435761ULL)
                              ^ (uint64_t(cols) * 40503ULL)
                              ^ (uint64_t(total_tokens) * 12345ULL);
  std::mt19937 rng(config_hash);

  std::vector<int> token_counts = generate_routed_tokens(total_tokens, num_experts, rng, skewed);

  int min_tok = *std::min_element(token_counts.begin(), token_counts.end());
  int max_tok = *std::max_element(token_counts.begin(), token_counts.end());
  int sum_tok = std::accumulate(token_counts.begin(), token_counts.end(), 0);
  int avg_tok = sum_tok / num_experts;

  size_t total_elements = 0;
  std::vector<int64_t> first_dims_h(num_experts);
  std::vector<int64_t> offsets_h(num_experts + 1);
  offsets_h[0] = 0;
  for (int i = 0; i < num_experts; i++) {
    first_dims_h[i] = token_counts[i];
    total_elements += static_cast<size_t>(token_counts[i]) * cols;
    offsets_h[i + 1] = static_cast<int64_t>(total_elements);
  }

  size_t total_rowwise_scales = 0, total_colwise_scales = 0;
  for (int i = 0; i < num_experts; i++) {
    if (USE_ROWWISE) total_rowwise_scales += token_counts[i] * ((cols + 31) / 32);
    if (USE_COLWISE) total_colwise_scales += ((token_counts[i] + 31) / 32) * cols;
  }

  void *in_data_d = nullptr, *out_data_rw_d = nullptr, *out_data_cw_d = nullptr;
  void *scales_rw_d = nullptr, *scales_cw_d = nullptr;
  int64_t *first_dims_d = nullptr, *offsets_d = nullptr;
  float *amax_d = nullptr;

  HIP_CHECK(hipMalloc(&in_data_d, total_elements * sizeof(IType)));
  if (USE_ROWWISE) {
    HIP_CHECK(hipMalloc(&out_data_rw_d, total_elements * sizeof(OType)));
    HIP_CHECK(hipMalloc(&scales_rw_d, total_rowwise_scales));
  }
  if (USE_COLWISE) {
    HIP_CHECK(hipMalloc(&out_data_cw_d, total_elements * sizeof(OType)));
    HIP_CHECK(hipMalloc(&scales_cw_d, total_colwise_scales));
  }
  HIP_CHECK(hipMalloc(&amax_d, sizeof(float)));
  HIP_CHECK(hipMalloc(&first_dims_d, num_experts * sizeof(int64_t)));
  HIP_CHECK(hipMalloc(&offsets_d, (num_experts + 1) * sizeof(int64_t)));
  HIP_CHECK(hipMemcpy(first_dims_d, first_dims_h.data(), num_experts * sizeof(int64_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(offsets_d, offsets_h.data(), (num_experts + 1) * sizeof(int64_t), hipMemcpyHostToDevice));

  std::vector<size_t> logical_shape_vec = {static_cast<size_t>(sum_tok), static_cast<size_t>(cols)};
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), 2);
  NVTEShape first_dims_shape;
  first_dims_shape.ndim = 1;
  first_dims_shape.data[0] = num_experts;
  NVTEShape offsets_shape;
  offsets_shape.ndim = 1;
  offsets_shape.data[0] = num_experts + 1;

  NVTEGroupedTensor in_gt  = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_experts, logical_shape);
  NVTEGroupedTensor out_gt = nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_experts, logical_shape);

  NVTEBasicTensor in_bt = {in_data_d, std::is_same_v<IType, float> ? kNVTEFloat32 :
                            (std::is_same_v<IType, hip_bfloat16> ? kNVTEBFloat16 : kNVTEFloat16), logical_shape};
  nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedRowwiseData, &in_bt, sizeof(in_bt));

  NVTEBasicTensor fd_bt = {first_dims_d, kNVTEInt64, first_dims_shape};
  NVTEBasicTensor off_bt = {offsets_d, kNVTEInt64, offsets_shape};
  nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedFirstDims, &fd_bt, sizeof(fd_bt));
  nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedTensorOffsets, &off_bt, sizeof(off_bt));
  nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedFirstDims, &fd_bt, sizeof(fd_bt));
  nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedTensorOffsets, &off_bt, sizeof(off_bt));

  if (USE_ROWWISE) {
    NVTEBasicTensor rw_data_bt = {out_data_rw_d, std::is_same_v<OType, fp8_e4m3> ? kNVTEFloat8E4M3 : kNVTEFloat8E5M2, logical_shape};
    std::vector<size_t> scales_rw_shape = {total_rowwise_scales};
    NVTEShape scales_rw_nvshape = nvte_make_shape(scales_rw_shape.data(), 1);
    NVTEBasicTensor rw_scales_bt = {scales_rw_d, kNVTEFloat8E8M0, scales_rw_nvshape};
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedRowwiseData, &rw_data_bt, sizeof(rw_data_bt));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedRowwiseScaleInv, &rw_scales_bt, sizeof(rw_scales_bt));
  }
  if (USE_COLWISE) {
    NVTEBasicTensor cw_data_bt = {out_data_cw_d, std::is_same_v<OType, fp8_e4m3> ? kNVTEFloat8E4M3 : kNVTEFloat8E5M2, logical_shape};
    std::vector<size_t> scales_cw_shape = {total_colwise_scales};
    NVTEShape scales_cw_nvshape = nvte_make_shape(scales_cw_shape.data(), 1);
    NVTEBasicTensor cw_scales_bt = {scales_cw_d, kNVTEFloat8E8M0, scales_cw_nvshape};
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedColumnwiseData, &cw_data_bt, sizeof(cw_data_bt));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedColumnwiseScaleInv, &cw_scales_bt, sizeof(cw_scales_bt));
  }

  NVTEBasicTensor amax_bt = {amax_d, kNVTEFloat32, nvte_make_shape(std::vector<size_t>{1}.data(), 1)};
  nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedAmax, &amax_bt, sizeof(amax_bt));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));

    nvte_group_quantize(in_gt, out_gt, stream);

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t bytes_read  = total_elements * sizeof(IType);
  size_t bytes_write = total_elements * sizeof(OType) * ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  set_bytes_processed(state, bytes_read + bytes_write + total_rowwise_scales + total_colwise_scales);

  state.counters["experts"]  = num_experts;
  state.counters["cols"]     = cols;
  state.counters["avg_tok"]  = avg_tok;
  state.counters["min_tok"]  = min_tok;
  state.counters["max_tok"]  = max_tok;

  nvte_destroy_grouped_tensor(in_gt);
  nvte_destroy_grouped_tensor(out_gt);
  hipFree(in_data_d);
  if (out_data_rw_d) hipFree(out_data_rw_d);
  if (out_data_cw_d) hipFree(out_data_cw_d);
  if (scales_rw_d) hipFree(scales_rw_d);
  if (scales_cw_d) hipFree(scales_cw_d);
  hipFree(amax_d);
  hipFree(first_dims_d);
  hipFree(offsets_d);
  HIP_CHECK(hipStreamDestroy(stream));
}

template <typename IType, typename OType, int SCALE_DIM_Y, int SCALE_DIM_X>
static void BM_MultiQuantizeMXFP8(benchmark::State &state) {
  const int num_experts  = state.range(0);
  const int cols         = state.range(1);
  const int total_tokens = state.range(2);
  const int skewed       = state.range(3);

  constexpr bool USE_ROWWISE = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE = SCALE_DIM_Y > 1;

  DType itype = std::is_same_v<IType, __half> ? DType::kFloat16 :
                (std::is_same_v<IType, hip_bfloat16> ? DType::kBFloat16 : DType::kFloat32);
  DType otype = std::is_same_v<OType, fp8_e4m3> ? DType::kFloat8E4M3 : DType::kFloat8E5M2;

  unsigned int seed = std::random_device{}();
  uint64_t config_hash = seed ^ (uint64_t(num_experts) * 2654435761ULL)
                              ^ (uint64_t(cols) * 40503ULL)
                              ^ (uint64_t(total_tokens) * 12345ULL);
  std::mt19937 rng(config_hash);

  std::vector<int> token_counts = generate_routed_tokens(total_tokens, num_experts, rng, skewed);

  int min_tok = *std::min_element(token_counts.begin(), token_counts.end());
  int max_tok = *std::max_element(token_counts.begin(), token_counts.end());
  int sum_tok = std::accumulate(token_counts.begin(), token_counts.end(), 0);
  int avg_tok = sum_tok / num_experts;

  size_t total_elements = 0;
  for (int i = 0; i < num_experts; i++)
    total_elements += static_cast<size_t>(token_counts[i]) * cols;

  std::vector<void *> in_ptrs(num_experts), out_rw_ptrs(num_experts), out_cw_ptrs(num_experts);
  std::vector<void *> scales_rw_ptrs(num_experts), scales_cw_ptrs(num_experts);
  std::vector<float *> amax_ptrs(num_experts);
  std::vector<NVTETensor> nvte_inputs(num_experts), nvte_outputs(num_experts);

  NVTEDType nvte_itype = std::is_same_v<IType, float> ? kNVTEFloat32 :
                         (std::is_same_v<IType, hip_bfloat16> ? kNVTEBFloat16 : kNVTEFloat16);
  NVTEDType nvte_otype = std::is_same_v<OType, fp8_e4m3> ? kNVTEFloat8E4M3 : kNVTEFloat8E5M2;

  for (int i = 0; i < num_experts; i++) {
    size_t rows = token_counts[i];
    size_t elts = rows * cols;
    size_t rw_scales = rows * ((cols + 31) / 32);
    size_t cw_scales = ((rows + 31) / 32) * cols;

    HIP_CHECK(hipMalloc(&in_ptrs[i], elts * sizeof(IType)));
    if (USE_ROWWISE) {
      HIP_CHECK(hipMalloc(&out_rw_ptrs[i], elts * sizeof(OType)));
      HIP_CHECK(hipMalloc(&scales_rw_ptrs[i], rw_scales));
    }
    if (USE_COLWISE) {
      HIP_CHECK(hipMalloc(&out_cw_ptrs[i], elts * sizeof(OType)));
      HIP_CHECK(hipMalloc(&scales_cw_ptrs[i], cw_scales));
    }
    HIP_CHECK(hipMalloc(&amax_ptrs[i], sizeof(float)));

    std::vector<size_t> shape_vec = {rows, static_cast<size_t>(cols)};
    NVTEShape shape = nvte_make_shape(shape_vec.data(), 2);

    nvte_inputs[i] = nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING);
    NVTEBasicTensor in_bt = {in_ptrs[i], nvte_itype, shape};
    nvte_set_tensor_param(&nvte_inputs[i], kNVTERowwiseData, &in_bt);

    nvte_outputs[i] = nvte_create_tensor(NVTE_MXFP8_1D_SCALING);
    if (USE_ROWWISE) {
      NVTEBasicTensor rw_bt = {out_rw_ptrs[i], nvte_otype, shape};
      std::vector<size_t> srw = {rw_scales};
      NVTEBasicTensor srw_bt = {scales_rw_ptrs[i], kNVTEFloat8E8M0, nvte_make_shape(srw.data(), 1)};
      nvte_set_tensor_param(&nvte_outputs[i], kNVTERowwiseData, &rw_bt);
      nvte_set_tensor_param(&nvte_outputs[i], kNVTERowwiseScaleInv, &srw_bt);
    }
    if (USE_COLWISE) {
      NVTEBasicTensor cw_bt = {out_cw_ptrs[i], nvte_otype, shape};
      std::vector<size_t> scw = {cw_scales};
      NVTEBasicTensor scw_bt = {scales_cw_ptrs[i], kNVTEFloat8E8M0, nvte_make_shape(scw.data(), 1)};
      nvte_set_tensor_param(&nvte_outputs[i], kNVTEColumnwiseData, &cw_bt);
      nvte_set_tensor_param(&nvte_outputs[i], kNVTEColumnwiseScaleInv, &scw_bt);
    }
    NVTEBasicTensor amax_bt = {amax_ptrs[i], kNVTEFloat32, nvte_make_shape(std::vector<size_t>{1}.data(), 1)};
    nvte_set_tensor_param(&nvte_outputs[i], kNVTEAmax, &amax_bt);
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  warmup_gpu();

  for (auto _ : state) {
    HIP_CHECK(hipEventRecord(start, stream));
    nvte_multi_quantize_mxfp8(num_experts, nvte_inputs.data(), nvte_outputs.data(), stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    state.SetIterationTime(ms / 1000.0);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  size_t bytes_read  = total_elements * sizeof(IType);
  size_t bytes_write = total_elements * sizeof(OType) * ((USE_ROWWISE ?: 0) + (USE_COLWISE ?: 0));
  size_t total_rw_scales = 0, total_cw_scales = 0;
  for (int i = 0; i < num_experts; i++) {
    if (USE_ROWWISE) total_rw_scales += token_counts[i] * ((cols + 31) / 32);
    if (USE_COLWISE) total_cw_scales += ((token_counts[i] + 31) / 32) * cols;
  }
  set_bytes_processed(state, bytes_read + bytes_write + total_rw_scales + total_cw_scales);

  state.counters["experts"]  = num_experts;
  state.counters["cols"]     = cols;
  state.counters["avg_tok"]  = avg_tok;
  state.counters["min_tok"]  = min_tok;
  state.counters["max_tok"]  = max_tok;

  for (int i = 0; i < num_experts; i++) {
    nvte_destroy_tensor(nvte_inputs[i]);
    nvte_destroy_tensor(nvte_outputs[i]);
    HIP_CHECK(hipFree(in_ptrs[i]));
    if (out_rw_ptrs[i]) HIP_CHECK(hipFree(out_rw_ptrs[i]));
    if (out_cw_ptrs[i]) HIP_CHECK(hipFree(out_cw_ptrs[i]));
    if (scales_rw_ptrs[i]) HIP_CHECK(hipFree(scales_rw_ptrs[i]));
    if (scales_cw_ptrs[i]) HIP_CHECK(hipFree(scales_cw_ptrs[i]));
    HIP_CHECK(hipFree(amax_ptrs[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

//                                    experts, cols,  total_tokens, skewed
#define MOE_BALANCED                                              \
  ->Args({128,  4096, 65536,  0}) /* Qwen3 H=4096  */             \
  ->Args({128,  1536, 65536,  0}) /* Qwen3 I=1536  */             \
  ->Args({256,  7168, 131072, 0}) /* DeepSeek H=7168 */           \
  ->Args({256,  2048, 131072, 0}) /* DeepSeek I=2048 */

#define MOE_SKEWED                                                \
  ->Args({128,  4096, 65536,  1}) /* Qwen3 H=4096  */             \
  ->Args({128,  1536, 65536,  1}) /* Qwen3 I=1536  */             \
  ->Args({256,  7168, 131072, 1}) /* DeepSeek H=7168 */           \
  ->Args({256,  2048, 131072, 1}) /* DeepSeek I=2048 */

#define REGISTER_GROUP_QUANTIZE(ITYPE, OTYPE, INAME, ONAME)                                \
  BENCHMARK_TEMPLATE(BM_GroupQuantizeMXFP8, ITYPE, OTYPE, 1, 32)                           \
    ->Name("BM_GroupQuantizeMXFP8/rowwise/" INAME "_" ONAME)                               \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();                                     \
  BENCHMARK_TEMPLATE(BM_GroupQuantizeMXFP8, ITYPE, OTYPE, 32, 1)                           \
    ->Name("BM_GroupQuantizeMXFP8/colwise/" INAME "_" ONAME)                               \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();                                     \
  BENCHMARK_TEMPLATE(BM_GroupQuantizeMXFP8, ITYPE, OTYPE, 32, 32)                          \
    ->Name("BM_GroupQuantizeMXFP8/both/" INAME "_" ONAME)                                  \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();

REGISTER_GROUP_QUANTIZE(hip_bfloat16, fp8_e4m3, "BF16", "E4M3")

#define REGISTER_MULTI_QUANTIZE(ITYPE, OTYPE, INAME, ONAME)                                \
  BENCHMARK_TEMPLATE(BM_MultiQuantizeMXFP8, ITYPE, OTYPE, 1, 32)                           \
    ->Name("BM_MultiQuantizeMXFP8/rowwise/" INAME "_" ONAME)                               \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();                                     \
  BENCHMARK_TEMPLATE(BM_MultiQuantizeMXFP8, ITYPE, OTYPE, 32, 1)                           \
    ->Name("BM_MultiQuantizeMXFP8/colwise/" INAME "_" ONAME)                               \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();                                     \
  BENCHMARK_TEMPLATE(BM_MultiQuantizeMXFP8, ITYPE, OTYPE, 32, 32)                          \
    ->Name("BM_MultiQuantizeMXFP8/both/" INAME "_" ONAME)                                  \
    MOE_BALANCED MOE_SKEWED                                                                \
    ->Unit(benchmark::kMicrosecond) ->UseManualTime();

REGISTER_MULTI_QUANTIZE(hip_bfloat16, fp8_e4m3, "BF16", "E4M3")

BENCHMARK_MAIN();
