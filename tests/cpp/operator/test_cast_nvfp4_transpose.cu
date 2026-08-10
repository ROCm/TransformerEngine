/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ActivationType {
    Identity,
    GeLU,
    SiLU,
    ReLU,
    QGeLU,
    SReLU
};

double2 cvt_fp4x2_to_double2(fp4e2m1x2 fp4_pair) {
#ifdef __HIP_PLATFORM_AMD__
    uint8_t raw = *reinterpret_cast<uint8_t*>(&fp4_pair);
    // Decode manually
    float lo = E2M1_LUT[raw & 0xF];
    float hi = E2M1_LUT[(raw >> 4) & 0xF];
    return {static_cast<double>(lo), static_cast<double>(hi)};
#else
    const __half2_raw raw_truncated_to_fp4e2m1_pair =
        __nv_cvt_fp4x2_to_halfraw2(*reinterpret_cast<__nv_fp4x2_storage_t*>(&fp4_pair), __NV_E2M1);

    const __half2 truncated_to_fp4e2m1_pair(raw_truncated_to_fp4e2m1_pair);
    const double truncated_to_fp4e2m1_x = static_cast<double>(truncated_to_fp4e2m1_pair.x);
    const double truncated_to_fp4e2m1_y = static_cast<double>(truncated_to_fp4e2m1_pair.y);
    return {truncated_to_fp4e2m1_x, truncated_to_fp4e2m1_y};
#endif
}

template <typename InputType>
std::vector<InputType> create_transpose(const InputType* const input, const size_t rows, size_t cols) {
    std::vector<InputType> input_t(cols * rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            const size_t idx_t = j * rows + i;
            input_t[idx_t] = input[idx];
        }
    }
    return input_t;
}

// Compute the global encode scale factor for a given global amax
float compute_global_encode_scaling_factor_FP4(const float global_amax, const bool use_fast_math,
                                               const int e4m3_max = 448) {
  NVTE_CHECK(e4m3_max == 448 || e4m3_max == 256, "Unsupported NVFP4 E4M3 max.");
  const float fp8_max = static_cast<float>(e4m3_max);
  constexpr float fp4_max = 6.0f;       // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return the max normalized value
  const float max_norm_clamp = (use_fast_math && e4m3_max == 448)
                               ? Numeric_Traits<bf16>::maxNorm
                               : Numeric_Traits<float>::maxNorm;

  global_encode_scale = fminf(global_encode_scale, max_norm_clamp);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

struct NVFP4FourOverSixQuantization {
  fp8e4m3 scale_map4;
  fp8e4m3 scale_map6;
  float reciprocal_map4;
  float reciprocal_map6;
  fp4e2m1x2 quantized_map4;
  fp4e2m1x2 quantized_map6;
};

enum class NVFP4FourOverSixCandidate {
  Map4,
  Map6,
};

enum class NVFP4ScalingMode {
  Block1D,
  RowScaled1D,
  Block2D,
};

struct NVFP4FourOverSixTestConfig {
  NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled;
  int e4m3_max = 448;
  bool err_use_fast_math = false;
};

bool use_2d_quantization(const NVFP4ScalingMode scaling_mode) {
  return scaling_mode == NVFP4ScalingMode::Block2D;
}

NVFP4FourOverSixQuantization compute_4over6_quantization_scales(
    const float block_amax, const float global_encode_scale) {
  constexpr float fp4_max = 6.0f;
  constexpr float fp8_max = 448.0f;
  constexpr float scale_expansion_factor = 1.5f;
  const float base_sf_high_precision = block_amax / fp4_max * global_encode_scale;
  const float sf_high_precision_map4 =
      fminf(base_sf_high_precision * scale_expansion_factor, fp8_max);
  const float sf_high_precision_map6 = fminf(base_sf_high_precision, fp8_max);
  const fp8e4m3 scale_map4 = static_cast<fp8e4m3>(sf_high_precision_map4);
  const fp8e4m3 scale_map6 = static_cast<fp8e4m3>(sf_high_precision_map6);

  const float global_decode_scale = 1.0f / global_encode_scale;
  const float scale_map4_fp32 = static_cast<float>(scale_map4);
  const float reciprocal_map4 =
      fminf(1.0f / (scale_map4_fp32 * global_decode_scale), Numeric_Traits<float>::maxNorm);
  const float scale_map6_fp32 = static_cast<float>(scale_map6);
  const float reciprocal_map6 =
      fminf(1.0f / (scale_map6_fp32 * global_decode_scale), Numeric_Traits<float>::maxNorm);

  const float2 zero = {0.0f, 0.0f};
  return {
      scale_map4,
      scale_map6,
      reciprocal_map4,
      reciprocal_map6,
      fp4e2m1x2(zero),
      fp4e2m1x2(zero),
  };
}

fp8e4m3 select_4over6_scale(const NVFP4FourOverSixQuantization& quantization,
                            const NVFP4FourOverSixCandidate candidate) {
  if (candidate == NVFP4FourOverSixCandidate::Map4) {
    return quantization.scale_map4;
  }
  return quantization.scale_map6;
}

fp4e2m1x2 select_4over6_quantized_pair(const NVFP4FourOverSixQuantization& quantization,
                                       const NVFP4FourOverSixCandidate candidate) {
  if (candidate == NVFP4FourOverSixCandidate::Map4) {
    return quantization.quantized_map4;
  }
  return quantization.quantized_map6;
}

NVFP4FourOverSixQuantization quantize_4over6_pair(
    const float x, const float y, const NVFP4FourOverSixQuantization& quantization) {
  const float2 scaled_map4 = {x * quantization.reciprocal_map4,
                              y * quantization.reciprocal_map4};
  const fp4e2m1x2 quantized_map4(scaled_map4);

  const float2 scaled_map6 = {x * quantization.reciprocal_map6,
                              y * quantization.reciprocal_map6};
  const fp4e2m1x2 quantized_map6(scaled_map6);

  return {
      quantization.scale_map4,
      quantization.scale_map6,
      quantization.reciprocal_map4,
      quantization.reciprocal_map6,
      quantized_map4,
      quantized_map6,
  };
}

// CUDA keeps upstream's looser rule; its fast-math configs score in packed fp16.
#ifdef __HIP_PLATFORM_AMD__
constexpr bool kAcceptEitherCandidate = false;
#else
constexpr bool kAcceptEitherCandidate = true;
#endif

struct NVFP4FourOverSixDecision {
    NVFP4FourOverSixCandidate candidate = NVFP4FourOverSixCandidate::Map6;
    // Candidate errors too close for FP32 summation order to resolve.
    bool ambiguous = false;
};

// Mirrors fp4_roundtrip_err() in quantize_transpose_vector_blockwise_fp4.cu,
// operand order included.
float fp4_roundtrip_err_ref(const float x, const float block_scale_inverse, const float sf,
                            const float global_amax, const float err_denom, const bool use_mse) {
    const float2 scaled = {x * block_scale_inverse, 0.0f};
    const fp4e2m1x2 quantized(scaled);
    const float dequant = static_cast<float>(cvt_fp4x2_to_double2(quantized).x);
    const float val = dequant * sf * global_amax / err_denom;
    const float diff = val - x;
    return use_mse ? diff * diff : std::fabs(diff);
}

// Lower round-trip error wins; ties go to map-to-6.
NVFP4FourOverSixDecision decide_4over6_candidate(const std::vector<float>& block,
                                                 const float block_amax,
                                                 const float S_enc,
                                                 const int e4m3_max,
                                                 const bool use_mse) {
    const NVFP4FourOverSixQuantization quantization =
        compute_4over6_quantization_scales(block_amax, S_enc);
    const float err_denom = 6.0f * static_cast<float>(e4m3_max);
    const float global_amax = err_denom / S_enc;

    double err_map4 = 0.0;
    double err_map6 = 0.0;
    for (const float x : block) {
        err_map4 += fp4_roundtrip_err_ref(x, quantization.reciprocal_map4,
                                          static_cast<float>(quantization.scale_map4), global_amax,
                                          err_denom, use_mse);
        err_map6 += fp4_roundtrip_err_ref(x, quantization.reciprocal_map6,
                                          static_cast<float>(quantization.scale_map6), global_amax,
                                          err_denom, use_mse);
    }

    // Worst-case FP32 summation drift of the kernel's warp reduction.
    const double tie_band = 4.0 * static_cast<double>(block.size()) *
                            static_cast<double>(FLT_EPSILON) * std::max(err_map4, err_map6);

    NVFP4FourOverSixDecision decision;
    decision.ambiguous = std::fabs(err_map4 - err_map6) <= tie_band;
    decision.candidate =
        err_map4 < err_map6 ? NVFP4FourOverSixCandidate::Map4 : NVFP4FourOverSixCandidate::Map6;
    return decision;
}

// Laid out like the scale tensor so the checker can index it by scale_idx.
template <typename InputType>
std::vector<NVFP4FourOverSixDecision> compute_4over6_expected_decisions(
    float (*OP)(const float),
    const InputType* const input,
    const size_t rows,
    const size_t cols,
    const size_t scales_stride,
    const float* const amax,
    const bool use_fast_math,
    const bool use_2d_quantization,
    const bool row_scaled_nvfp4,
    const bool use_mse,
    const int e4m3_max) {

    constexpr size_t block_size_Y = 16;
    constexpr size_t block_size_X = 16;
    const size_t blocks_X = divide_round_up(cols, block_size_X);
    std::vector<NVFP4FourOverSixDecision> decisions(rows * scales_stride);

    // Same numerical truncation the reference quantizers apply.
    auto activated = [OP, input, cols](const size_t i, const size_t j) {
        const float act_elt = OP(static_cast<float>(input[i * cols + j]));
        return static_cast<float>(static_cast<InputType>(act_elt));
    };

    if (use_2d_quantization) {
        const float S_enc = compute_global_encode_scaling_factor_FP4(*amax, use_fast_math,
                                                                     e4m3_max);
        const size_t blocks_Y = divide_round_up(rows, block_size_Y);
        for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
            for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
                const size_t i_min = block_Y * block_size_Y;
                const size_t i_max = std::min(i_min + block_size_Y, rows);
                const size_t j_min = block_X * block_size_X;
                const size_t j_max = std::min(j_min + block_size_X, cols);

                std::vector<float> block;
                block.reserve(block_size_Y * block_size_X);
                float block_amax = 0.0f;
                for (size_t i = i_min; i < i_max; ++i) {
                    for (size_t j = j_min; j < j_max; ++j) {
                        const float elt = activated(i, j);
                        block.push_back(elt);
                        block_amax = std::max(block_amax, std::abs(elt));
                    }
                }

                const NVFP4FourOverSixDecision decision =
                    decide_4over6_candidate(block, block_amax, S_enc, e4m3_max, use_mse);
                // Block scale and its candidate replicate down the block's rows.
                for (size_t i = i_min; i < i_max; ++i) {
                    decisions[i * scales_stride + block_X] = decision;
                }
            }
        }
        return decisions;
    }

    for (size_t i = 0; i < rows; ++i) {
        const float S_enc = compute_global_encode_scaling_factor_FP4(
            row_scaled_nvfp4 ? amax[i] : *amax, use_fast_math, e4m3_max);
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = std::min(j_min + block_size_X, cols);

            std::vector<float> block;
            block.reserve(block_size_X);
            float block_amax = 0.0f;
            for (size_t j = j_min; j < j_max; ++j) {
                const float elt = activated(i, j);
                block.push_back(elt);
                block_amax = std::max(block_amax, std::abs(elt));
            }

            decisions[i * scales_stride + block_X] =
                decide_4over6_candidate(block, block_amax, S_enc, e4m3_max, use_mse);
        }
    }
    return decisions;
}

// 1D Scaling: Original implementation with 1x16 blocks
template <typename InputType>
void quantize_nvfp4_1d(float (*OP)(const float),
                       const InputType* const input,
                       fp4e2m1x2* const output,
                       fp8e4m3* const scales,
                       const size_t rows,
                       const size_t cols,
                       const size_t scales_stride,
                       const float global_amax,
                       const bool use_fast_math,
                       const bool use_4over6 = false,
                       const int e4m3_max = 448,
                       const NVFP4FourOverSixCandidate four_over_six_candidate =
                           NVFP4FourOverSixCandidate::Map6) {

    // Compute a global encoding/decoding scaling factor for all S_dec_b
    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);

    constexpr size_t block_size_X = 16;
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    std::array<float, block_size_X> cache_buffer;
    for (size_t i = 0; i < block_size_X; ++i) {
        cache_buffer[i] = 0.0f;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = j_min + block_size_X;

            // Find block amax
            float block_amax = 0.0f;
            for (size_t j = j_min; j < j_max; ++j) {
                const size_t idx = i * cols + j;
                const size_t cache_idx = j - j_min;

                const float input_elt = static_cast<float>(input[idx]);
                const float act_elt = OP(input_elt);

                // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                cache_buffer[cache_idx] = elt;
                block_amax = std::max(block_amax, std::abs(elt));
            }

            const size_t scale_idx = i * scales_stride + block_X;

            if (use_4over6) {
                const NVFP4FourOverSixQuantization quantization =
                    compute_4over6_quantization_scales(block_amax, S_enc);
                scales[scale_idx] = select_4over6_scale(quantization, four_over_six_candidate);

                for (size_t j = j_min; j < j_max; j += 2) {
                    const int idx_pair = (i * cols + j) / 2;
                    const int cache_idx_x = j - j_min;
                    const int cache_idx_y = cache_idx_x + 1;
                    const float cached_x = cache_buffer[cache_idx_x];
                    const float cached_y = cache_buffer[cache_idx_y];
                    const NVFP4FourOverSixQuantization pair_quantization =
                        quantize_4over6_pair(cached_x, cached_y, quantization);
                    output[idx_pair] =
                        select_4over6_quantized_pair(pair_quantization, four_over_six_candidate);
                }
                continue;
            }

            // Compute and store the per-block FP8 decode scale
            const float S_dec_b = block_amax * (S_enc * (1.0f / 6.0f));
            const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(fminf(S_dec_b, Numeric_Traits<float>::maxNorm));
            const float S_dec_b_fp32 = static_cast<float>(S_dec_b_fp8);

            // Compute "correct" per-block encoding scaling factor
            const float S_enc_b_fp8 = S_dec_b_fp32 == 0.f ? 0.f :
                fminf(1.0f / (S_dec_b_fp32 * (1.0f / S_enc)), Numeric_Traits<float>::maxNorm);

            scales[scale_idx] = S_dec_b_fp8;

            float scale_reciprocal = S_enc_b_fp8;
            if (use_fast_math) {
                // Numerical truncation to match GPU implementation, if mixed precision FMA instruction is used
                scale_reciprocal = static_cast<float>(static_cast<bf16>(scale_reciprocal));
            }

            for (size_t j = j_min; j < j_max; j += 2) {
                const int idx_pair = (i * cols + j) / 2;
                const int cache_idx_x = j - j_min;
                const int cache_idx_y = cache_idx_x + 1;
                const float cached_x = cache_buffer[cache_idx_x];
                const float cached_y = cache_buffer[cache_idx_y];
                const float scaled_elt_x = cached_x * scale_reciprocal;
                const float scaled_elt_y = cached_y * scale_reciprocal;
                const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                output[idx_pair] = casted_to_e2m1_pair;

                const double2 truncated_pair = cvt_fp4x2_to_double2(casted_to_e2m1_pair);
            }
        }
    }
}

// Compute 2D mathematical scaling factors (8x8 for 128x128 input)
template <typename InputType>
void compute_2d_mathematical_scales(float (*OP)(const float),
                                   const InputType* const input,
                                   const size_t rows,
                                   const size_t cols,
                                   const float global_amax,
                                   std::vector<std::vector<fp8e4m3>>& math_scales,
                                   const bool use_fast_math,
                                   const bool use_4over6 = false,
                                   const int e4m3_max = 448,
                                   const NVFP4FourOverSixCandidate four_over_six_candidate =
                                       NVFP4FourOverSixCandidate::Map6) {

    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);
    constexpr size_t block_size_Y = 16;
    constexpr size_t block_size_X = 16;
    const size_t blocks_Y = divide_round_up(rows, block_size_Y);
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    math_scales.resize(blocks_Y, std::vector<fp8e4m3>(blocks_X));

    for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t i_min = block_Y * block_size_Y;
            const size_t i_max = std::min(i_min + block_size_Y, rows);
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = std::min(j_min + block_size_X, cols);

            // Find 2D block amax over entire 16x16 region
            float block_amax = 0.0f;
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const float input_elt = static_cast<float>(input[idx]);
                    const float act_elt = OP(input_elt);
                    const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                    block_amax = std::max(block_amax, std::abs(elt));
                }
            }

            // Compute E4M3 scaling factor for this 16x16 block
            if (use_4over6) {
                const NVFP4FourOverSixQuantization quantization =
                    compute_4over6_quantization_scales(block_amax, S_enc);
                math_scales[block_Y][block_X] =
                    select_4over6_scale(quantization, four_over_six_candidate);
            } else {
                const float S_dec_b = block_amax / 6.0f * S_enc;
                const fp8e4m3 S_dec_b_fp8_map6 = static_cast<fp8e4m3>(S_dec_b);
                math_scales[block_Y][block_X] = S_dec_b_fp8_map6;
            }
        }
    }
}

// 2D Scaling: NEW implementation with proper replication
template <typename InputType>
void quantize_nvfp4_2d(float (*OP)(const float),
                       const InputType* const input,
                       fp4e2m1x2* const output,
                       fp8e4m3* const scales,
                       const size_t rows,
                       const size_t cols,
                       const size_t scales_stride,
                       const float global_amax,
                       const bool use_fast_math,
                       const bool use_4over6 = false,
                       const int e4m3_max = 448,
                       const NVFP4FourOverSixCandidate four_over_six_candidate =
                           NVFP4FourOverSixCandidate::Map6) {

    // Step 1: Compute mathematical 8x8 scaling factors
    std::vector<std::vector<fp8e4m3>> math_scales;
    compute_2d_mathematical_scales(OP, input, rows, cols, global_amax, math_scales, use_fast_math,
                                   use_4over6, e4m3_max, four_over_six_candidate);

    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);
    constexpr size_t block_size_Y = 16;
    constexpr size_t block_size_X = 16;
    const size_t blocks_Y = divide_round_up(rows, block_size_Y);
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    // Step 2: Replicate scaling factors row-wise (128×8 storage) - only if scales is not nullptr
    if (scales != nullptr) {
        // Each of the 128 rows gets scaling factors from its corresponding 16×16 block
        for (size_t i = 0; i < rows; ++i) {
            const size_t block_Y = i / block_size_Y;
            for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
                const size_t scale_idx = i * scales_stride + block_X;
                scales[scale_idx] = math_scales[block_Y][block_X];
            }
        }
    }

    // Step 3: Apply quantization using the mathematical scaling factors
    std::array<std::array<float, block_size_X>, block_size_Y> cache_buffer;

    for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t i_min = block_Y * block_size_Y;
            const size_t i_max = std::min(i_min + block_size_Y, rows);
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = std::min(j_min + block_size_X, cols);

            // Get the scaling factor for this block
            const float S_dec_b_fp8 = static_cast<float>(math_scales[block_Y][block_X]);
            const float S_enc_b_fp8 = S_dec_b_fp8 == 0.0f ? 0.0f : S_enc / S_dec_b_fp8;
            const float scale_reciprocal = S_enc_b_fp8;

            // Process and cache data for this 16x16 block
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const size_t cache_idx_y = i - i_min;
                    const size_t cache_idx_x = j - j_min;

                    const float input_elt = static_cast<float>(input[idx]);
                    const float act_elt = OP(input_elt);
                    const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                    cache_buffer[cache_idx_y][cache_idx_x] = elt;
                }
            }

            // Apply scaling to all elements in this 16x16 block
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; j += 2) {
                    const int idx_pair = (i * cols + j) / 2;
                    const size_t cache_idx_y = i - i_min;
                    const size_t cache_idx_x1 = j - j_min;
                    const size_t cache_idx_x2 = std::min(cache_idx_x1 + 1, block_size_X - 1);

                    const float cached_x = cache_buffer[cache_idx_y][cache_idx_x1];
                    const float cached_y = ((j + 1) < j_max && cache_idx_x2 < block_size_X) ?
                                          cache_buffer[cache_idx_y][cache_idx_x2] : 0.0f;

                    const float scaled_elt_x = cached_x * scale_reciprocal;
                    const float scaled_elt_y = cached_y * scale_reciprocal;
                    const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                    fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                    output[idx_pair] = casted_to_e2m1_pair;
                }
            }
        }
    }
}

// Wrapper function that calls appropriate implementation based on 2D flag
template <typename InputType>
void quantize_nvfp4(float (*OP)(const float),
                    const InputType* const input,
                    fp4e2m1x2* const output,
                    fp8e4m3* const scales,
                    const size_t rows,
                    const size_t cols,
                    const size_t scales_stride,
                    const float global_amax,
                    const bool use_fast_math,
                    const bool use_2d_quantization = false,
                    const bool use_4over6 = false,
                    const int e4m3_max = 448,
                    const NVFP4FourOverSixCandidate four_over_six_candidate =
                        NVFP4FourOverSixCandidate::Map6) {
    if (use_2d_quantization) {
        quantize_nvfp4_2d(OP, input, output, scales, rows, cols, scales_stride, global_amax,
                          use_fast_math, use_4over6, e4m3_max, four_over_six_candidate);
    } else {
        quantize_nvfp4_1d(OP, input, output, scales, rows, cols, scales_stride, global_amax,
                          use_fast_math, use_4over6, e4m3_max, four_over_six_candidate);
    }
}

template <typename InputType>
void compute_ref(float (*OP)(const float),
                 const InputType* input,
                 fp4e2m1x2* output,
                 fp4e2m1x2* output_t,
                 fp8e4m3* scales,
                 fp8e4m3* scales_t,
                 const float* amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const size_t scales_stride_t,
                 const bool use_fast_math,
                 const bool use_2d_quantization = false,
                 const bool row_scaled_nvfp4 = false,
                 const bool use_4over6 = false,
                 const int e4m3_max = 448,
                 const NVFP4FourOverSixCandidate four_over_six_candidate =
                     NVFP4FourOverSixCandidate::Map6)
{
    std::vector<InputType> input_t = create_transpose(input, rows, cols);
    NVTE_CHECK(!(use_2d_quantization && row_scaled_nvfp4),
               "2D quantization and row-scaling are not supported together.");

    // Ref impl for 2D quantization
    if (use_2d_quantization) {
        // Step 1: Compute mathematical 8×8 scaling factors
        std::vector<std::vector<fp8e4m3>> math_scales;
        compute_2d_mathematical_scales(OP, input, rows, cols, *amax, math_scales, use_fast_math,
                                       use_4over6, e4m3_max, four_over_six_candidate);

        constexpr size_t block_size_Y = 16;
        constexpr size_t block_size_X = 16;
        const size_t blocks_Y = divide_round_up(rows, block_size_Y);
        const size_t blocks_X = divide_round_up(cols, block_size_X);

        // Step 2: Generate scales (128×8) by replicating row-wise
        for (size_t i = 0; i < rows; ++i) {
            const size_t block_Y = i / block_size_Y;
            for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
                const size_t scale_idx = i * scales_stride + block_X;
                scales[scale_idx] = math_scales[block_Y][block_X];
            }
        }

        // Step 3: Generate scales_t (128×8) with proper transposed block mapping
        for (size_t i = 0; i < cols; ++i) {  // cols = 128, which becomes rows of transposed data
            const size_t block_X_orig = i / block_size_X;  // i was column index in original, so maps to block_X
            for (size_t block_Y_new = 0; block_Y_new < blocks_Y; ++block_Y_new) {  // block in transposed coordinate
                const size_t scale_idx = i * scales_stride_t + block_Y_new;
                scales_t[scale_idx] = math_scales[block_Y_new][block_X_orig];
            }
        }

        // Step 4: Process quantized outputs using the same algorithm as quantize_nvfp4_2d
        // (This part processes the actual FP4 data using the mathematical scaling factors)
        quantize_nvfp4_2d(OP, input, output, nullptr, rows, cols, scales_stride, *amax,
                          use_fast_math, use_4over6, e4m3_max,
                          four_over_six_candidate); // scales already filled
        quantize_nvfp4_2d(OP, input_t.data(), output_t, nullptr, cols, rows, scales_stride_t, *amax,
                          use_fast_math, use_4over6, e4m3_max,
                          four_over_six_candidate); // scales_t already filled

        return;
    }

    // Ref impl for row-scaling
    if (row_scaled_nvfp4) {
        for (size_t row = 0; row < rows; ++row) {
            quantize_nvfp4(OP,
                           input + row * cols,
                           output + row * (cols / 2),
                           scales + row * scales_stride,
                           1,
                           cols,
                           scales_stride,
                           amax[row],
                           use_fast_math,
                           use_2d_quantization,
                           use_4over6,
                           e4m3_max,
                           four_over_six_candidate);
        }
        return;
    }

    // Ref impl for basic NVFP4
    quantize_nvfp4(OP, input, output, scales, rows, cols, scales_stride, *amax,
                   use_fast_math, use_2d_quantization, use_4over6, e4m3_max,
                   four_over_six_candidate);
    quantize_nvfp4(OP, input_t.data(), output_t, scales_t, cols, rows, scales_stride_t, *amax,
                   use_fast_math, use_2d_quantization, use_4over6, e4m3_max,
                   four_over_six_candidate);
}

void compare_nvfp4_tensors(const std::string& name,
                           const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                           const int rows, const int cols,
                           double atol = 1e-5, double rtol = 1e-8) {
    constexpr int max_mismatches_to_print = 3;

    std::vector<std::string> mismatch_messages;
    size_t total_mismatches = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 2) {
            const int idx = i * cols + j;
            double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
            double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));

            for (int k = 0; k < 2; ++k) {
                const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);

#ifndef __HIP_PLATFORM_AMD__
                const bool mismatch = fabs(t - r) > (atol + fabs(r) * rtol);
#else
                bool mismatch = fabs(t - r) > (atol + fabs(r) * rtol);
                if (mismatch) {
                    /* Check if it is just a failure of round to nearest choosing different
                        side of the real value */
                    const double mean = (t + r) / 2;
                    const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
                    const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
                    const double cast_mean_p = static_cast<double>(static_cast<fp4e2m1>(mean_p));
                    const double cast_mean_m = static_cast<double>(static_cast<fp4e2m1>(mean_m));
                    mismatch = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
                }
#endif
                if (mismatch) {
                    total_mismatches++;
                    // Optional: limit number of detailed messages to avoid overwhelming output
                    if (total_mismatches <= max_mismatches_to_print) {
                        std::string msg = "Mismatch at place (" + std::to_string(idx + k) + "): " +
                                          std::to_string(t) + " vs " + std::to_string(r) +
                                          " (abs_diff: " + std::to_string(fabs(t - r)) +
                                          ", rel_diff: " + std::to_string(r == 0 ? 0.0 : fabs((t - r) / r)) + ")";
                        mismatch_messages.push_back(msg);
                        std::cout << "Error in tensor " << name << ": " << msg << std::endl;
                    }
                }
            }
        }
    }

    // Always report summary - either success or failure
    std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
    std::cout << "Total elements checked: " << (rows * cols) << std::endl;

    if (total_mismatches > 0) {
        std::cout << "STATUS: FAILED for output" << std::endl;
        std::cout << "Total mismatches found: " << total_mismatches << std::endl;
        std::cout << "Mismatch rate: " << (100.0 * total_mismatches) / (rows * cols) << "%" << std::endl;
        if (mismatch_messages.size() > max_mismatches_to_print) {
            std::cout << "... and " << (mismatch_messages.size() - max_mismatches_to_print)
            << " more mismatches (showing first " << max_mismatches_to_print << ")" << std::endl;
        }
        std::cout << "============================" << std::endl;

        GTEST_FAIL() << "Found " << total_mismatches << " mismatches in tensor " << name;
    } else {
        std::cout << "STATUS: PASSED for output" << std::endl;
        std::cout << "All elements match within tolerance!" << std::endl;
        std::cout << "Tensor " << name << " is IDENTICAL to reference" << std::endl;
        std::cout << "============================" << std::endl;
    }
}

// Optional: Function to dump tensor data to files for detailed analysis
void dump_nvfp4_tensor_data(const std::string& prefix,
                            const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                            const int rows, const int cols) {
    std::string test_file = prefix + "_test.txt";
    std::string ref_file = prefix + "_ref.txt";
    std::string diff_file = prefix + "_diff.txt";

    std::ofstream test_out(test_file);
    std::ofstream ref_out(ref_file);
    std::ofstream diff_out(diff_file);

    if (test_out.is_open() && ref_out.is_open() && diff_out.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; j += 2) {
                const int idx = i * cols + j;
                double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
                double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));

                for (int k = 0; k < 2; ++k) {
                    const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                    const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);
                    const int pos = idx + k;

                    test_out << "pos[" << pos << "] = " << t << std::endl;
                    ref_out << "pos[" << pos << "] = " << r << std::endl;
                    diff_out << "pos[" << pos << "] test=" << t << " ref=" << r
                            << " abs_diff=" << fabs(t - r)
                            << " rel_diff=" << (r == 0 ? 0.0 : fabs((t - r) / r)) << std::endl;
                }
            }
        }
        std::cout << "DEBUG: Dumped tensor data to files: " << test_file << ", " << ref_file << ", " << diff_file << std::endl;
    } else {
        std::cout << "WARNING: Could not open files for tensor data dump" << std::endl;
    }
}

void compareResults_nvfp4(Tensor &test,
                          const void *ref, const void *ref_t, const int rows, const int cols,
                          double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true,
                          bool dump_data = false, bool compare_columnwise = true) {
    if (if_on_gpus) test.to_cpu();

    const fp4e2m1 *test_data = test.rowwise_cpu_dptr<fp4e2m1>();
    const fp4e2m1 *ref_data = reinterpret_cast<const fp4e2m1*>(ref);

    // Optionally dump tensor data to files for detailed analysis
    if (dump_data) {
        dump_nvfp4_tensor_data("output", test_data, ref_data, rows, cols);
    }

    compare_nvfp4_tensors("output", test_data, ref_data, rows, cols, atol, rtol);
    if (compare_columnwise) {
        const fp4e2m1 *test_data_t = test.columnwise_cpu_dptr<fp4e2m1>();
        const fp4e2m1 *ref_data_t = reinterpret_cast<const fp4e2m1*>(ref_t);
        if (dump_data) {
            dump_nvfp4_tensor_data("output_t", test_data_t, ref_data_t, cols, rows);
        }
        compare_nvfp4_tensors("output_t", test_data_t, ref_data_t, cols, rows, atol, rtol);
    }
}

template <typename T>
bool bitwise_equal(const T& x, const T& y) {
    const auto *x_bytes = reinterpret_cast<const unsigned char*>(&x);
    const auto *y_bytes = reinterpret_cast<const unsigned char*>(&y);
    for (size_t i = 0; i < sizeof(T); ++i) {
        if (x_bytes[i] != y_bytes[i]) {
            return false;
        }
    }
    return true;
}

bool nvfp4_output_block_matches(const fp4e2m1x2* const test_data,
                                const fp4e2m1x2* const ref_data,
                                const size_t row,
                                const size_t cols,
                                const size_t block_x) {
    constexpr size_t block_size_X = 16;
    const size_t j_min = block_x * block_size_X;
    const size_t j_max = std::min(j_min + block_size_X, cols);
    for (size_t j = j_min; j < j_max; j += 2) {
        const size_t idx_pair = (row * cols + j) / 2;
        if (!bitwise_equal(test_data[idx_pair], ref_data[idx_pair])) {
            return false;
        }
    }
    return true;
}

void compare_nvfp4_4over6_candidates(const std::string& name,
                                     const fp4e2m1* const test_data,
                                     const fp8e4m3* const test_scales,
                                     const fp4e2m1x2* const ref_data_map4,
                                     const fp8e4m3* const ref_scales_map4,
                                     const fp4e2m1x2* const ref_data_map6,
                                     const fp8e4m3* const ref_scales_map6,
                                     const std::vector<NVFP4FourOverSixDecision>& decisions,
                                     const size_t rows,
                                     const size_t cols,
                                     const size_t blocks_X,
                                     const size_t scales_stride) {
    constexpr int max_mismatches_to_print = 3;
    const auto* const test_data_pairs = reinterpret_cast<const fp4e2m1x2*>(test_data);
    size_t total_mismatches = 0;
    size_t wrong_candidate = 0;
    size_t expected_map4 = 0;
    size_t expected_map6 = 0;
    // Blocks whose two candidates encode to the same bytes: a selection cannot be
    // read back out of them, so they neither pin the kernel nor weaken the check.
    size_t identical_candidates = 0;
    // Blocks that do encode differently but whose candidate errors are too close
    // for the host to call. These are the only ones where the kernel keeps any
    // latitude, so their count is the coverage this check gives up.
    size_t unpinned_blocks = 0;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block_x = 0; block_x < blocks_X; ++block_x) {
            const size_t scale_idx = row * scales_stride + block_x;
            const bool scale_matches_map4 =
                bitwise_equal(test_scales[scale_idx], ref_scales_map4[scale_idx]);
            const bool data_matches_map4 =
                nvfp4_output_block_matches(test_data_pairs, ref_data_map4, row, cols, block_x);
            const bool scale_matches_map6 =
                bitwise_equal(test_scales[scale_idx], ref_scales_map6[scale_idx]);
            const bool data_matches_map6 =
                nvfp4_output_block_matches(test_data_pairs, ref_data_map6, row, cols, block_x);
            const bool matches_map4 = scale_matches_map4 && data_matches_map4;
            const bool matches_map6 = scale_matches_map6 && data_matches_map6;

            const bool candidates_differ =
                !bitwise_equal(ref_scales_map4[scale_idx], ref_scales_map6[scale_idx]) ||
                !nvfp4_output_block_matches(ref_data_map4, ref_data_map6, row, cols, block_x);

            const NVFP4FourOverSixDecision& decision = decisions[scale_idx];
            bool matched = false;
            const char* expectation = nullptr;
            if (decision.ambiguous || kAcceptEitherCandidate) {
                // Too close to call from the host; either encoding is a correct answer.
                if (candidates_differ) {
                    ++unpinned_blocks;
                } else {
                    ++identical_candidates;
                }
                matched = matches_map4 || matches_map6;
                expectation = "map-to-4 or map-to-6 (candidate errors within the tie band)";
            } else if (decision.candidate == NVFP4FourOverSixCandidate::Map4) {
                ++expected_map4;
                matched = matches_map4;
                expectation = "map-to-4";
            } else {
                ++expected_map6;
                matched = matches_map6;
                expectation = "map-to-6";
            }

            if (matched) {
                continue;
            }

            ++total_mismatches;
            const bool matched_other = matches_map4 || matches_map6;
            if (matched_other) {
                ++wrong_candidate;
            }
            if (total_mismatches <= max_mismatches_to_print) {
                std::cout << "Error in tensor " << name << ": 4over6 block mismatch at row "
                          << row << ", block_x " << block_x << ". Expected " << expectation
                          << "; the output "
                          << (matched_other ? "matched the other candidate instead"
                                            : "matched neither candidate exactly")
                          << "." << std::endl;
            }
        }
    }

    std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
    std::cout << "Total 4over6 blocks checked: " << (rows * blocks_X) << std::endl;
    std::cout << "Blocks pinned to map-to-4: " << expected_map4
              << ", pinned to map-to-6: " << expected_map6
              << ", left unpinned by the tie band: " << unpinned_blocks
              << ", encoding identically either way: " << identical_candidates << std::endl;
    if (total_mismatches > 0) {
        std::cout << "STATUS: FAILED for output" << std::endl;
        std::cout << "Total mismatched 4over6 blocks found: " << total_mismatches
                  << " (" << wrong_candidate << " picked the wrong candidate)" << std::endl;
        std::cout << "============================" << std::endl;
        GTEST_FAIL() << "Found " << total_mismatches << " 4over6 block mismatches in tensor "
                     << name << " (" << wrong_candidate << " picked the wrong candidate)";
    }

    std::cout << "STATUS: PASSED for output" << std::endl;
    std::cout << "Each 4over6 block matched the candidate the reference selected" << std::endl;
    std::cout << "============================" << std::endl;
}

void compare_rowwise_amax(Tensor &output, const std::vector<float> &ref_amax) {
    ASSERT_EQ(output.rowwise_amax_size(), ref_amax.size());
    const auto *amax_ptr = output.cpu_rowwise_amax_ptr<float>();
    const std::vector<float> test_amax_data(amax_ptr, amax_ptr + ref_amax.size());
    for (size_t row = 0; row < ref_amax.size(); ++row) {
        ASSERT_EQ(test_amax_data[row], ref_amax[row])
            << "Row-scaled amax mismatch at row " << row;
    }
}

template <typename InputType>
void performTest(float (*OP)(const float),
                 const std::vector<size_t>& shape,
                 const bool use_fast_math,
                 const NVFP4ScalingMode scaling_mode = NVFP4ScalingMode::Block1D,
                 const NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled,
                 const int e4m3_max = 448,
                 const bool use_4over6_err_use_fast_math = false) {
    using namespace test;
    const bool use_4over6 = mode != kNVTENVFP44Over6Disabled;

    if (use_4over6 && use_fast_math) {
        std::cout << "WARNING: Plain NVFP4 fast math is ignored for 4over6. "
                     "Use use_4over6_err_use_fast_math to test the 4over6 candidate "
                     "error fast-math path."
                  << std::endl;
    }

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;

    const bool is_2d_quantization = use_2d_quantization(scaling_mode);
    const bool row_scaled_nvfp4 = scaling_mode == NVFP4ScalingMode::RowScaled1D;
    const bool rowwise = true;
    const bool columnwise = !row_scaled_nvfp4;

#ifdef __HIP_PLATFORM_AMD__
    if (te_fp8_fnuz()) GTEST_SKIP() << "NVFP4 not supported on gfx942 (fnuz)";
    if (use_4over6 && use_4over6_err_use_fast_math)
      GTEST_SKIP() << "NVFP4 4over6 fast-math error mode is not supported on ROCm";
#endif

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    // Use get_scale_tensor_dims for NVFP4 scale tensor dimensions
    // Now that CheckScaleTensorShape is fixed, this should work correctly
#ifdef __HIP_PLATFORM_AMD__
    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16, NVTE_NVFP4_1D_SCALING);
    const std::array<size_t,4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16, NVTE_NVFP4_1D_SCALING);
#else
    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);
    const std::array<size_t,4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);
#endif //#ifdef __HIP_PLATFORM_AMD__

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t blocks_Y_t = scale_dims_t[2];
    const size_t blocks_X_t = scale_dims_t[3];
    const size_t scales_stride_t = blocks_X_t;

    Tensor input("input", shape, itype);
    Tensor output("output", shape, otype, rowwise, columnwise, NVTE_NVFP4_1D_SCALING);
    output.set_nvfp4_e4m3_max(e4m3_max);

    std::unique_ptr<fp4e2m1x2[]> ref_output   = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
    std::unique_ptr<fp4e2m1x2[]> ref_output_t = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
    std::unique_ptr<fp8e4m3[]> ref_scales     = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);
    std::unique_ptr<fp8e4m3[]> ref_scales_t   = std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);
    std::unique_ptr<fp4e2m1x2[]> ref_output_map6;
    std::unique_ptr<fp4e2m1x2[]> ref_output_t_map6;
    std::unique_ptr<fp8e4m3[]> ref_scales_map6;
    std::unique_ptr<fp8e4m3[]> ref_scales_t_map6;
    std::vector<NVFP4FourOverSixDecision> expected_decisions;
    std::vector<NVFP4FourOverSixDecision> expected_decisions_t;

    fillCase<fp32>(&input, InputsFillCase::uniform);

    if (use_4over6 && row_scaled_nvfp4) {
        const float target_row_amax = static_cast<float>(e4m3_max) * 6.0f * 8.0f;
        auto *input_vals = input.rowwise_cpu_dptr<InputType>();
        for (size_t row = 0; row < rows; ++row) {
            float row_amax = 0.0f;
            size_t max_col = 0;
            for (size_t col = 0; col < cols; ++col) {
                const float val = static_cast<float>(input_vals[row * cols + col]);
                const float abs_val = fabsf(val);
                if (abs_val > row_amax) {
                    row_amax = abs_val;
                    max_col = col;
                }
            }

            if (row_amax == 0.0f) {
                continue;
            }

            const float row_scale = target_row_amax / row_amax;
            for (size_t col = 0; col < cols; ++col) {
                float scaled = static_cast<float>(input_vals[row * cols + col]) * row_scale;
                scaled = fminf(fmaxf(scaled, -target_row_amax), target_row_amax);
                input_vals[row * cols + col] = static_cast<InputType>(scaled);
            }

            const float max_val = static_cast<float>(input_vals[row * cols + max_col]);
            input_vals[row * cols + max_col] =
                static_cast<InputType>(max_val < 0.0f ? -target_row_amax : target_row_amax);
        }
        input.from_cpu();
    }

    // Compute 2nd stage NVFP4 scaling factor
    std::vector<float> ref_amax;
    if (row_scaled_nvfp4) {
        // Compute per-row amaxes
        const auto *input_vals = input.rowwise_cpu_dptr<InputType>();
        for (size_t row = 0; row < rows; ++row){
            float row_amax = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                row_amax = fmaxf(row_amax, fabsf(static_cast<float>(input_vals[row * cols + col])));
            }
            ref_amax.push_back(row_amax);
        }

        // Update tensor
        // Note: No need to update amax like standard NVFP4, amaxes
        // are computed during quantization.
        output.set_row_scaled_nvfp4(row_scaled_nvfp4);
    } else {
        // Golden value of amax chosen to make the 2nd-stage scaling mantissa zero and avoid rounding issues
        if (use_4over6) {
            ref_amax.assign(1, static_cast<float>(e4m3_max) * 6.0f * 8.0f);
        } else {
            ref_amax.assign(1, 448.0f * 6.0f * 8.0f);
        }

        // Update tensor
        if (rowwise) {
            std::copy(ref_amax.begin(), ref_amax.end(), output.cpu_rowwise_amax_ptr<float>());
        }
        if (columnwise) {
            std::copy(ref_amax.begin(), ref_amax.end(), output.cpu_columnwise_amax_ptr<float>());
        }
        output.from_cpu();
    }

    if (use_4over6) {
        ref_output_map6 = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
        ref_output_t_map6 = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
        ref_scales_map6 = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);
        ref_scales_t_map6 = std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);

        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output.get(),
                               ref_output_t.get(),
                               ref_scales.get(),
                               ref_scales_t.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6,
                               e4m3_max,
                               NVFP4FourOverSixCandidate::Map4);
        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output_map6.get(),
                               ref_output_t_map6.get(),
                               ref_scales_map6.get(),
                               ref_scales_t_map6.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6,
                               e4m3_max,
                               NVFP4FourOverSixCandidate::Map6);

        // The kernel is free to pick either candidate per block, but not freely:
        // it must pick the one with the smaller round-trip error. Derive that
        // choice here so the checker can hold it to exactly one reference.
        const bool use_mse = mode == kNVTENVFP44Over6MinMSE;
        expected_decisions = compute_4over6_expected_decisions<InputType>(
            OP, input.rowwise_cpu_dptr<InputType>(), rows, cols, scales_stride, ref_amax.data(),
            use_fast_math, is_2d_quantization, row_scaled_nvfp4, use_mse, e4m3_max);
        if (!row_scaled_nvfp4) {
            const std::vector<InputType> input_t =
                create_transpose(input.rowwise_cpu_dptr<InputType>(), rows, cols);
            expected_decisions_t = compute_4over6_expected_decisions<InputType>(
                OP, input_t.data(), cols, rows, scales_stride_t, ref_amax.data(), use_fast_math,
                is_2d_quantization, row_scaled_nvfp4, use_mse, e4m3_max);
        }
    } else {
        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output.get(),
                               ref_output_t.get(),
                               ref_scales.get(),
                               ref_scales_t.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6);
    }

#ifdef __HIP_PLATFORM_AMD__
    // Test stochastic rounding on gfx950.
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    const bool is_gfx950 = prop.major == 9 && prop.minor == 5;
    // 4over6 excluded: the quantize dispatch refuses it combined with stochastic
    // rounding, the same way this test already excludes it from fast math below.
    for (bool use_stochastic_rounding : (is_gfx950 && !use_4over6
                                             ? std::vector<bool>{false, true}
                                             : std::vector<bool>{false})) {
#endif

    // Initialize stochastic rounding
    Tensor rng_state("rng_state", std::vector<size_t>{2}, DType::kInt64);
    rng_state.rowwise_cpu_dptr<int64_t>()[0] = 123;  // rng_seed
    rng_state.rowwise_cpu_dptr<int64_t>()[1] = 321;  // rng_sequence
    rng_state.from_cpu();

    // Quantization options
    QuantizationConfigWrapper quant_config;
    quant_config.set_use_fast_math(use_fast_math && !use_4over6);
#ifdef __HIP_PLATFORM_AMD__
    quant_config.set_stochastic_rounding(use_stochastic_rounding);
#else
    quant_config.set_stochastic_rounding(false);
#endif
    quant_config.set_rng_state(rng_state.data());
    quant_config.set_nvfp4_2d_quantization(is_2d_quantization);
    quant_config.set_nvfp4_4over6_mode(mode);
    quant_config.set_nvfp4_4over6_err_use_fast_math(use_4over6 && use_4over6_err_use_fast_math);

    // Call appropriate function based on operation type
    // Activation functions take 3 parameters (input, output, stream)
    // nvte_quantize_v2 takes 4 parameters (input, output, quant_config, stream)
    if (OP == &gelu) {
        nvte_gelu(input.data(), output.data(), 0);
    } else if (OP == &silu) {
        nvte_silu(input.data(), output.data(), 0);
    } else if (OP == &relu) {
        nvte_relu(input.data(), output.data(), 0);
    } else if (OP == &qgelu) {
        nvte_qgelu(input.data(), output.data(), 0);
    } else if (OP == &srelu) {
        nvte_srelu(input.data(), output.data(), 0);
    } else {
        nvte_quantize_v2(input.data(), output.data(), quant_config, 0);
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("DEBUG: CUDA error detected: %s\n", cudaGetErrorString(err));
    }
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    const double atol = 1.0E-6;
    const double rtol = 1.0E-6;

    if (use_4over6) {
        output.to_cpu();
        compare_nvfp4_4over6_candidates("output",
                                        output.rowwise_cpu_dptr<fp4e2m1>(),
                                        output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                        ref_output.get(),
                                        ref_scales.get(),
                                        ref_output_map6.get(),
                                        ref_scales_map6.get(),
                                        expected_decisions,
                                        rows,
                                        cols,
                                        unpadded_blocks_X,
                                        scales_stride);
        if (!row_scaled_nvfp4) {
            compare_nvfp4_4over6_candidates("output_t",
                                            output.columnwise_cpu_dptr<fp4e2m1>(),
                                            output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                            ref_output_t.get(),
                                            ref_scales_t.get(),
                                            ref_output_t_map6.get(),
                                            ref_scales_t_map6.get(),
                                            expected_decisions_t,
                                            cols,
                                            rows,
                                            unpadded_blocks_X_t,
                                            scales_stride_t);
        }
    } else {
        // Set dump_data=true to enable dumping tensor data to files for analysis
        compareResults_nvfp4(output, ref_output.get(), ref_output_t.get(), rows, cols, atol, rtol,
                             true, false, !row_scaled_nvfp4);

        size_t scale_mismatches_num = 0;
#ifdef __HIP_PLATFORM_AMD__
        std::vector<size_t> mismatches_scales_indices;
#endif
        compare_scaling_factors<fp8e4m3>("scales", output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                          ref_scales.get(),
                                          unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
#ifdef __HIP_PLATFORM_AMD__
                                          mismatches_scales_indices,
#endif
                                          scale_mismatches_num);

        if (!row_scaled_nvfp4) {
            compare_scaling_factors<fp8e4m3>("scales_t",
                                              output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                              ref_scales_t.get(),
                                              unpadded_blocks_Y_t, unpadded_blocks_X_t,
                                              scales_stride_t,
#ifdef __HIP_PLATFORM_AMD__
                                              mismatches_scales_indices,
#endif
                                              scale_mismatches_num);
        }
    }

    compare_rowwise_amax(output, ref_amax);
#ifdef __HIP_PLATFORM_AMD__
    } // for (bool use_stochastic_rounding : ...)
#endif
}

// Columnwise-only 2D NVFP4 must match the columnwise half of both-directions output
template <typename InputType>
void performTestColumnwiseOnly2D(const std::vector<size_t>& shape) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    // Columnwise (transposed) scale-tensor dimensions.
    const std::array<size_t, 4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);
    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t scales_stride_t = scale_dims_t[3];

    Tensor input("input", shape, itype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    // Golden amax chosen so the 2nd-stage scaling mantissa is zero (avoids rounding noise).
    const float golden_amax = 448.0f * 6.0f * 8.0f;

    // Reference: both directions produced in a single kernel call (rowwise + columnwise).
    Tensor output_both("output_both", shape, otype, /*rowwise=*/true, /*columnwise=*/true,
                       NVTE_NVFP4_1D_SCALING);
    output_both.cpu_rowwise_amax_ptr<float>()[0] = golden_amax;
    output_both.cpu_columnwise_amax_ptr<float>()[0] = golden_amax;
    output_both.from_cpu();

    // System under test: columnwise-only output (no rowwise data allocated).
    Tensor output_col("output_col", shape, otype, /*rowwise=*/false, /*columnwise=*/true,
                      NVTE_NVFP4_1D_SCALING);
    output_col.cpu_columnwise_amax_ptr<float>()[0] = golden_amax;
    output_col.from_cpu();

    QuantizationConfigWrapper quant_config;
    quant_config.set_stochastic_rounding(false);
    quant_config.set_nvfp4_2d_quantization(true);

    nvte_quantize_v2(input.data(), output_both.data(), quant_config, 0);
    nvte_quantize_v2(input.data(), output_col.data(), quant_config, 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    output_both.to_cpu();
    output_col.to_cpu();

    // Columnwise FP4 data must match bitwise (atol = rtol = 0).
    compare_nvfp4_tensors("columnwise_only_data",
                          output_col.columnwise_cpu_dptr<fp4e2m1>(),
                          output_both.columnwise_cpu_dptr<fp4e2m1>(),
                          static_cast<int>(cols), static_cast<int>(rows),
                          /*atol=*/0.0, /*rtol=*/0.0);

    // Columnwise scale factors must match over the in-bounds region.
    size_t scale_mismatches = 0;
#ifdef __HIP_PLATFORM_AMD__
    std::vector<size_t> mismatches_scales_indices;
#endif
    compare_scaling_factors<fp8e4m3>("columnwise_only_scales",
                                     output_col.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     output_both.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     unpadded_blocks_Y_t, unpadded_blocks_X_t, scales_stride_t,
#ifdef __HIP_PLATFORM_AMD__
                                     mismatches_scales_indices,
#endif
                                     scale_mismatches);
    ASSERT_EQ(scale_mismatches, 0u);

    // The columnwise-only tensor must not allocate rowwise output.
    EXPECT_FALSE(output_col.rowwise());
}

std::vector<std::vector<size_t>> tensor_dims = {
    {32, 32},
    {32, 64},
    {64, 32},
    {64, 96},
    {128, 128},
    {256, 256},
    {512, 512},
    {1024, 1024},
    {2048, 2048},
    {128, 256},
    {8192, 128},
    {2048, 160},
    {8, 32, 1024},
    {16, 8, 4, 512},
    {1024, 16384},
    {4096, 13312},
};

// Only the Identity activation is currently supported.
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity
};

}  // namespace

class FusedCastTransposeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<ActivationType,
                std::vector<size_t>,
                transformer_engine::DType,
                bool,
                NVFP4ScalingMode,
                NVFP4FourOverSixTestConfig>> {};

TEST_P(FusedCastTransposeNVFP4TestSuite, TestFusedCastTransposeNVFP4) {
#ifndef __HIP_PLATFORM_AMD__
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
#endif

    using namespace transformer_engine;
    using namespace test;

    const ActivationType Act_type = std::get<0>(GetParam());
    const auto tensor_dims = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const bool use_fast_math = std::get<3>(GetParam());
    const NVFP4ScalingMode scaling_mode = std::get<4>(GetParam());
    const NVFP4FourOverSixTestConfig config = std::get<5>(GetParam());

    // Skip tests if the input tensor is 1D
    if (tensor_dims.size() < 2) {
        GTEST_SKIP();
    }

    // Forward activations
    auto OP = &identity;
    switch (Act_type) {
        case ActivationType::GeLU: OP = &gelu; break;
        case ActivationType::SiLU: OP = &silu; break;
        case ActivationType::ReLU: OP = &relu; break;
        case ActivationType::QGeLU: OP = &qgelu; break;
        case ActivationType::SReLU: OP = &srelu; break;
    }

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        performTest<InputType>(OP, tensor_dims, use_fast_math, scaling_mode, config.mode,
                               config.e4m3_max,
                               config.err_use_fast_math);
    );
}

std::string to_string(const ActivationType Act_type) {
    switch (Act_type) {
        case ActivationType::Identity:  return "CAST_ONLY";
        case ActivationType::GeLU:      return "GeLU";
        case ActivationType::SiLU:      return "SiLU";
        case ActivationType::ReLU:      return "ReLU";
        case ActivationType::QGeLU:     return "QGeLU";
        case ActivationType::SReLU:     return "SReLU";
        default: return "";
    }
}

std::string to_string(const NVFP4ScalingMode scaling_mode) {
    switch (scaling_mode) {
        case NVFP4ScalingMode::Block1D:     return "";
        case NVFP4ScalingMode::RowScaled1D: return "XROW_SCALED";
        case NVFP4ScalingMode::Block2D:     return "X2D";
        default: return "";
    }
}

std::string test_name(const FusedCastTransposeNVFP4TestSuite::ParamType& param) {
    std::string name = to_string(std::get<0>(param));
    const auto& shape = std::get<1>(param);
    for (const auto& s: shape) {
        name += "X" + std::to_string(s);
    }
    name += "X" + test::typeName(std::get<2>(param));
    if (std::get<3>(param)) {
        name += "X_FAST_SCALING";
    }
    name += to_string(std::get<4>(param));
    const NVFP4FourOverSixTestConfig& config = std::get<5>(param);
    if (config.mode != kNVTENVFP44Over6Disabled) {
        name += "X4OVER6";
        if (config.e4m3_max == 448) {
            name += "XE4M3_MAX_448";
        } else {
            name += "XE4M3_MAX_256";
        }
        if (config.mode == kNVTENVFP44Over6MinMSE) {
            name += "XMSE";
        } else if (config.mode == kNVTENVFP44Over6MinMAE) {
            name += "XMAE";
        } else {
            name += "XINVALID_MODE";
        }
        if (config.err_use_fast_math) {
            name += "XERR_USE_FAST_MATH";
        }
    }
    return name;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),           // activation_type
        ::testing::ValuesIn(tensor_dims),                // tensor_dims
        ::testing::Values(DType::kBFloat16),             // input_type
        ::testing::Values(false),                       // use_fast_math
        ::testing::Values(NVFP4ScalingMode::Block1D),   // scaling_mode
        ::testing::Values(NVFP4FourOverSixTestConfig{})), // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTestRowScaled,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),               // activation_type
        ::testing::ValuesIn(tensor_dims),                    // tensor_dims
        ::testing::Values(DType::kBFloat16, DType::kFloat32), // input_type
        ::testing::Values(false),                           // use_fast_math
        ::testing::Values(NVFP4ScalingMode::RowScaled1D),   // scaling_mode
        ::testing::Values(NVFP4FourOverSixTestConfig{})),   // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest4Over6,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),           // activation_type
        ::testing::ValuesIn(tensor_dims),                // tensor_dims
        ::testing::Values(DType::kBFloat16, DType::kFloat32), // input_type
        ::testing::Values(false),                       // use_fast_math
        ::testing::Values(NVFP4ScalingMode::Block1D,
                          NVFP4ScalingMode::RowScaled1D,
                          NVFP4ScalingMode::Block2D),   // scaling_mode
        ::testing::Values(
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 448, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 448, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 448, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 448, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 256, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 256, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 256, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 256, true})), // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

class CastNVFP4ColumnwiseOnly2DTestSuite : public ::testing::TestWithParam<std::vector<size_t>> {};

TEST_P(CastNVFP4ColumnwiseOnly2DTestSuite, ColumnwiseOnlyMatchesBothDirections) {
    // The optimized NVFP4 quantize-transpose kernel requires Blackwell.
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
    performTestColumnwiseOnly2D<bf16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastNVFP4ColumnwiseOnly2DTestSuite,
    // Include rectangular 128-multiple shapes to guard transposed data/scale indexing.
    ::testing::Values(
        std::vector<size_t>{128, 128},
        std::vector<size_t>{256, 512},
        std::vector<size_t>{384, 1024},
        std::vector<size_t>{2048, 256}),
    [](const testing::TestParamInfo<CastNVFP4ColumnwiseOnly2DTestSuite::ParamType>& info) {
        std::string name;
        for (const auto& s : info.param) {
            name += "X" + std::to_string(s);
        }
        return name;
    });
