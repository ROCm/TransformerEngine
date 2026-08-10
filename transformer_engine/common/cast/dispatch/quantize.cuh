/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize.cuh
 *  \brief Quantize dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"
#include "../../util/vectorized_pointwise.h"
#include "../core/common.cuh"
#include "../fp8/quantize_fp8.cuh"
#ifdef __HIP_PLATFORM_AMD__
#include "../fp8/rocm_cast.cuh"
#endif
#include "../mxfp8/group_quantize_mxfp8.cuh"
#include "../mxfp8/quantize_mxfp8.cuh"
#ifdef __HIP_PLATFORM_AMD__
#include "../mxfp4/quantize_mxfp4.cuh"
#endif //#ifdef __HIP_PLATFORM_AMD__
// The optimized NVFP4 kernels are Blackwell-only (CUtensorMap/tcgen05); ROCm uses
// the generic quantize_transpose_vector_blockwise_fp4 path plus this portable
// row-wise amax for row-scaled NVFP4.
#ifndef __HIP_PLATFORM_AMD__
#include "../nvfp4/group_quantize_transpose_nvfp4.cuh"
#include "../nvfp4/quantize_4over6_nvfp4.cuh"
#include "../nvfp4/quantize_transpose_nvfp4.cuh"
#else
#include "../nvfp4/rowwise_amax_nvfp4.cuh"
#endif //#ifndef __HIP_PLATFORM_AMD__

namespace transformer_engine {
namespace dispatch {

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_fwd_helper(const NVTETensor input, NVTETensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  const Tensor *input_tensor = convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const Tensor *dummy_input_tensor = nullptr;
      Tensor *dummy_dbias_tensor = nullptr;
      Tensor *dummy_workspace_tensor = nullptr;
      if (output_tensor->has_columnwise_data()) {
        NVTE_CHECK(output_tensor->has_data(),
                   "Quantizing in only the columnwise direction not supported yet!");
        if constexpr (!IS_ACT) {
          cast_transpose(*input_tensor, *noop_tensor, output_tensor, stream);
        } else {
          cast_transpose_fused</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, float, ParamOP, OP>(
              *input_tensor, dummy_input_tensor, output_tensor, dummy_dbias_tensor,
              dummy_workspace_tensor, stream);
        }
      } else if (output_tensor->has_data()) {
#ifdef __HIP_PLATFORM_AMD__
        if constexpr (!IS_ACT) {
          fp8::rocm_cast_only(*input_tensor, *noop_tensor, output_tensor, stream);
        } else
#endif
        {
          fp8::quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
              *input_tensor, dummy_input_tensor, noop_tensor, output_tensor, dummy_dbias_tensor,
              dummy_workspace_tensor, stream);
        }
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      const Tensor *dummy_input_tensor = nullptr;
      Tensor *dummy_dbias_tensor = nullptr;
      Tensor *dummy_workspace_tensor = nullptr;
      mxfp8::quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
          *input_tensor, dummy_input_tensor, noop_tensor, output_tensor, dummy_dbias_tensor,
          dummy_workspace_tensor, stream);
      break;
    }
#ifdef __HIP_PLATFORM_AMD__
    case NVTE_MXFP4_1D_SCALING: {
      const Tensor *dummy_input_tensor = nullptr;
      Tensor *dummy_dbias_tensor = nullptr;
      Tensor *dummy_workspace_tensor = nullptr;
      mxfp4::quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
          *input_tensor, dummy_input_tensor, noop_tensor, output_tensor, dummy_dbias_tensor,
          dummy_workspace_tensor, quant_config_cpp, stream);
      break;
    }
#endif //#ifdef __HIP_PLATFORM_AMD__
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK(!IS_ACT, "IS_ACT is not supported by FWD NVTE_NVFP4_1D_SCALING");

      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Choose kernel
      const auto [rows, cols] = input_tensor->flat_2d_dims();
      auto dtype = input_tensor->dtype();
      const bool row_scaled_nvfp4 = output_tensor->row_scaled_nvfp4;
      const bool nvfp4_use_4over6 = quant_config_cpp.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
      NVTE_CHECK(nvfp4_use_4over6 || output_tensor->nvfp4_e4m3_max == 448,
                 "Non-4over6 NVFP4 quantization requires E4M3 max 448.");
      NVTE_CHECK(!nvfp4_use_4over6 || !quant_config_cpp.stochastic_rounding,
                 "NVFP4 4over6 quantization does not support stochastic rounding.");
#ifdef __HIP_PLATFORM_AMD__
      // The fast-math error path is an optimisation of candidate selection, not a
      // capability; refuse it rather than silently scoring with the exact one.
      NVTE_CHECK(!nvfp4_use_4over6 || !quant_config_cpp.nvfp4_4over6_err_use_fast_math,
                 "NVFP4 4over6 fast-math error mode is not supported on ROCm.");
#endif
      if (row_scaled_nvfp4) {
        NVTE_CHECK(!quant_config_cpp.nvfp4_2d_quantization,
                   "Row-scaled NVFP4 quantization does not support 2D quantization.");
        NVTE_CHECK(!output_tensor->has_columnwise_data(),
                   "Row-scaled NVFP4 quantization does not produce columnwise output.");
        nvfp4::compute_rowwise_amax(*input_tensor, noop_tensor, output_tensor, stream);
      }
      // Columnwise-only is supported on the optimized path only for 2D scaling; rowwise-only and
      // both-directions keep their existing routing. Columnwise-only 1D and non-bf16 fall back to
      // quantize_transpose_vector_blockwise_fp4.
      bool use_optimized_kernel =
          (dtype == DType::kBFloat16) && (rows % 32 == 0) && (cols % 32 == 0) &&
          (output_tensor->has_data() ||
           (output_tensor->has_columnwise_data() && quant_config_cpp.nvfp4_2d_quantization));

      // Launch NVFP4 quantize kernel. Upstream's dedicated 4over6 and optimized
      // quantize_transpose kernels are CUDA-only (Blackwell); ROCm falls through to the portable
      // blockwise path below, which implements 4over6 and row-scaled NVFP4 itself.
#ifndef __HIP_PLATFORM_AMD__
      if (nvfp4_use_4over6) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_4over6</*use_2d_quantization=*/true>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_4over6</*use_2d_quantization=*/false>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else if (use_optimized_kernel) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization*/ false>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else {
#endif
        auto &global_amax = (output_tensor->amax.dptr != nullptr) ? output_tensor->amax
                                                                  : output_tensor->columnwise_amax;
        quantize_transpose_vector_blockwise_fp4(
            /*input=*/input_tensor->data, /*global_amax=*/global_amax,
            /*scale_inv=*/output_tensor->scale_inv,
            /*scale_inv_t=*/output_tensor->columnwise_scale_inv,
            /*output=*/output_tensor->data, /*output_t=*/output_tensor->columnwise_data,
            /*epsilon=*/0.0f, /*return_identity=*/output_tensor->has_data(),
            /*return_transpose=*/output_tensor->has_columnwise_data(), /*pow2_scale=*/false,
            /*swizzled_scale=*/false,
            /*use_stochastic_rounding=*/quant_config_cpp.stochastic_rounding,
            /*rng_state=*/quant_config_cpp.rng_state,
            /*use_2d_quantization=*/quant_config_cpp.nvfp4_2d_quantization,
            /*row_scaled_nvfp4=*/row_scaled_nvfp4,
            /*noop_tensor=*/noop_tensor->data,
            /*nvfp4_e4m3_max=*/output_tensor->nvfp4_e4m3_max,
            /*nvfp4_4over6_mode=*/quant_config_cpp.nvfp4_4over6_mode,
            /*stream=*/stream);
#ifndef __HIP_PLATFORM_AMD__
      }
#endif
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // TODO(kwyss): IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK(!IS_ACT, "IS_ACT is not implemented for FWD NVTE_BLOCK_SCALING_2D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      quantize_transpose_square_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon,
          /*return_transpose=*/output_tensor->has_columnwise_data(), force_pow_2_scales,
          /*noop_tensor=*/noop_tensor->data, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // TODO(kwyss): IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK(!IS_ACT, "IS_ACT is not implemented for FWD NVTE_BLOCK_SCALING_1D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;
      FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
      if (output_tensor->has_data()) {
        rowwise_option = FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
      }
      if (output_tensor->has_columnwise_data()) {
        columnwise_option = FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
      }
      quantize_transpose_vector_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon, rowwise_option,
          columnwise_option, force_pow_2_scales, noop_tensor->data, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_bwd_helper(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                         NVTETensor dbias, NVTETensor workspace,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  const Tensor *grad_tensor = convertNVTETensorCheck(grad);
  const Tensor *input_tensor = convertNVTETensor(input);

  Tensor *output_tensor = convertNVTETensorCheck(output);
  Tensor *dbias_tensor = convertNVTETensor(dbias);
  Tensor *workspace_tensor = convertNVTETensor(workspace);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (output_tensor->has_columnwise_data()) {
        NVTE_CHECK(output_tensor->has_data(),
                   "Quantizing in only the columnwise direction not supported yet!");
        if constexpr (!IS_DBIAS && !IS_DACT) {
          cast_transpose(*grad_tensor, *noop_tensor, output_tensor, stream);
        } else {
          cast_transpose_fused<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, float, ParamOP, OP>(
              *grad_tensor, input_tensor, output_tensor, dbias_tensor, workspace_tensor, stream);
        }
      } else if (output_tensor->has_data()) {
        fp8::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
            *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
            stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      mxfp8::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
          *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
          stream);
      break;
    }
#ifdef __HIP_PLATFORM_AMD__
    case NVTE_MXFP4_1D_SCALING: {
      mxfp4::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
          *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
          quant_config_cpp, stream);
      break;
    }
#endif
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not supported by BWD NVTE_NVFP4_1D_SCALING");

      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*grad_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Choose kernel
      const auto [rows, cols] = grad_tensor->flat_2d_dims();
      auto dtype = grad_tensor->dtype();
      const bool nvfp4_use_4over6 = quant_config_cpp.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
      NVTE_CHECK(nvfp4_use_4over6 || output_tensor->nvfp4_e4m3_max == 448,
                 "Non-4over6 NVFP4 quantization requires E4M3 max 448.");
      NVTE_CHECK(!nvfp4_use_4over6 || !quant_config_cpp.stochastic_rounding,
                 "NVFP4 4over6 quantization does not support stochastic rounding.");
#ifdef __HIP_PLATFORM_AMD__
      // The fast-math error path is an optimisation of candidate selection, not a
      // capability; refuse it rather than silently scoring with the exact one.
      NVTE_CHECK(!nvfp4_use_4over6 || !quant_config_cpp.nvfp4_4over6_err_use_fast_math,
                 "NVFP4 4over6 fast-math error mode is not supported on ROCm.");
#endif
      NVTE_CHECK(!output_tensor->row_scaled_nvfp4,
                 "Backward NVFP4 quantization does not support row-scaled outputs.");
      // Columnwise-only is supported on the optimized path only for 2D scaling; rowwise-only and
      // both-directions keep their existing routing. Columnwise-only 1D and non-bf16 fall back to
      // quantize_transpose_vector_blockwise_fp4.
      bool use_optimized_kernel =
          (dtype == DType::kBFloat16) && (rows % 32 == 0) && (cols % 32 == 0) &&
          (output_tensor->has_data() ||
           (output_tensor->has_columnwise_data() && quant_config_cpp.nvfp4_2d_quantization));

      // Launch NVFP4 quantize kernel. Upstream's dedicated 4over6 and optimized
      // quantize_transpose kernels are CUDA-only (Blackwell); ROCm falls through to the portable
      // blockwise path below, which implements 4over6 and row-scaled NVFP4 itself.
#ifndef __HIP_PLATFORM_AMD__
      if (nvfp4_use_4over6) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_4over6</*use_2d_quantization=*/true>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_4over6</*use_2d_quantization=*/false>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else if (use_optimized_kernel) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization*/ false>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else {
#endif
        auto &global_amax = (output_tensor->amax.dptr != nullptr) ? output_tensor->amax
                                                                  : output_tensor->columnwise_amax;
        quantize_transpose_vector_blockwise_fp4(
            /*input=*/grad_tensor->data, /*global_amax=*/global_amax,
            /*scale_inv=*/output_tensor->scale_inv,
            /*scale_inv_t=*/output_tensor->columnwise_scale_inv,
            /*output=*/output_tensor->data, /*output_t=*/output_tensor->columnwise_data,
            /*epsilon=*/0.0f, /*return_identity=*/output_tensor->has_data(),
            /*return_transpose=*/output_tensor->has_columnwise_data(), /*pow2_scale=*/false,
            /*swizzled_scale=*/false,
            /*use_stochastic_rounding=*/quant_config_cpp.stochastic_rounding,
            /*rng_state=*/quant_config_cpp.rng_state,
            /*use_2d_quantization=*/quant_config_cpp.nvfp4_2d_quantization,
            /*row_scaled_nvfp4=*/false,
            /*noop_tensor=*/noop_tensor->data,
            /*nvfp4_e4m3_max=*/output_tensor->nvfp4_e4m3_max,
            /*nvfp4_4over6_mode=*/quant_config_cpp.nvfp4_4over6_mode,
            /*stream=*/stream);
#ifndef __HIP_PLATFORM_AMD__
      }
#endif
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not implemented for BWD NVTE_BLOCK_SCALING_2D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      quantize_transpose_square_blockwise(
          grad_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon,
          /*return_transpose=*/output_tensor->has_columnwise_data(), force_pow_2_scales,
          /*noop_tensor=*/noop_tensor->data, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not implemented for BWD NVTE_BLOCK_SCALING_1D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;
      FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
      if (output_tensor->has_data()) {
        rowwise_option = FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
      }
      if (output_tensor->has_columnwise_data()) {
        columnwise_option = FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
      }
      quantize_transpose_vector_blockwise(
          grad_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon, rowwise_option,
          columnwise_option, force_pow_2_scales, noop_tensor->data, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}

// Host-aware and not graph-safe: group quantization with split section info from the host.
template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize_fwd_host_aware_helper(const NVTETensor input, NVTETensor *outputs,
                                          const size_t *split_sections, const size_t num_tensors,
                                          const NVTEQuantizationConfig quant_config,
                                          cudaStream_t stream) {
  using namespace detail;

  const Tensor *input_tensor = convertNVTETensorCheck(input);
  std::vector<Tensor *> output_tensors;
  for (size_t i = 0; i < num_tensors; ++i) {
    output_tensors.push_back(convertNVTETensorCheck(outputs[i]));
  }

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensors[0]->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  // Take the scaling mode of the first output tensor
  auto scaling_mode = output_tensors[0]->scaling_mode;

  // Dispatch to quantization kernel depending on data format
  switch (scaling_mode) {
#ifndef __HIP_PLATFORM_AMD__
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK(!IS_ACT, "IS_ACT is not supported by FWD NVTE_NVFP4_1D_SCALING");

      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      // Skip checking output tensor list
      // output list here is allowed to have empty tensor

      // Choose kernel
      const auto [rows, cols] = input_tensor->flat_2d_dims();
      auto dtype = input_tensor->dtype();

      const bool nvfp4_use_4over6 = quant_config_cpp.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
      for (const auto *output_tensor : output_tensors) {
        NVTE_CHECK(nvfp4_use_4over6 || output_tensor->nvfp4_e4m3_max == 448,
                   "Non-4over6 NVFP4 quantization requires E4M3 max 448.");
      }
      NVTE_CHECK(!quant_config_cpp.nvfp4_2d_quantization,
                 "2D quantization is not supported for group quantize.");
      NVTE_CHECK(!nvfp4_use_4over6,
                 "NVFP4 4over6 quantization is not supported for group quantize.");

      // Launch NVFP4 group quantize kernel
      nvfp4::group_quantize_transpose</*use_2d_quantization*/ false>(
          *input_tensor, noop_tensor, output_tensors, split_sections, num_tensors,
          &quant_config_cpp, stream);
      break;
    }
#endif //#ifndef __HIP_PLATFORM_AMD__
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(scaling_mode) + ".");
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize_fwd_helper(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                               const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  NVTEScalingMode scaling_mode = nvte_grouped_tensor_scaling_mode(output);

  const NVTEGroupedTensor activation = nullptr;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  const GroupedTensor *input_tensor = convertNVTEGroupedTensorCheck(input);
  GroupedTensor *output_tensor = convertNVTEGroupedTensorCheck(output);
  const GroupedTensor *activations_tensor = convertNVTEGroupedTensor(activation);
  GroupedTensor *dbias_tensor = convertNVTEGroupedTensor(dbias);
  Tensor *workspace_tensor = convertNVTETensor(workspace);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Dispatch to quantization kernel depending on data format
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      mxfp8::group_quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
          input_tensor, activations_tensor, noop_tensor, output_tensor, dbias_tensor,
          workspace_tensor, &quant_config_cpp, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize_bwd_helper(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                               NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                               NVTETensor workspace, const NVTEQuantizationConfig quant_config,
                               cudaStream_t stream) {
  using namespace detail;

  NVTEScalingMode scaling_mode = nvte_grouped_tensor_scaling_mode(output);

  const GroupedTensor *grad_tensor = convertNVTEGroupedTensorCheck(grad);
  const GroupedTensor *input_tensor = convertNVTEGroupedTensor(input);
  GroupedTensor *output_tensor = convertNVTEGroupedTensorCheck(output);
  GroupedTensor *dbias_tensor = convertNVTEGroupedTensor(dbias);
  Tensor *workspace_tensor = convertNVTETensor(workspace);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Dispatch to quantization kernel depending on data format
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      mxfp8::group_quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
          grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
          &quant_config_cpp, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(scaling_mode) + ".");
  }
}

#ifdef __HIP_PLATFORM_AMD__
inline void multi_quantize_mxfp8(const std::vector<Tensor *> &input_list,
                          std::vector<Tensor *> &output_list, cudaStream_t stream) {
  const size_t num_tensors = input_list.size();
  if (num_tensors == 0) return;
  NVTE_CHECK(num_tensors <= mxfp8::quantize_kernel::kMultiQuantizeMXFP8MaxTensors,
             "multi_quantize_mxfp8: num_tensors (", num_tensors, ") exceeds maximum (",
             mxfp8::quantize_kernel::kMultiQuantizeMXFP8MaxTensors, ").");

  DType itype = input_list[0]->data.dtype;
  DType otype = output_list[0]->dtype();
  const bool use_rowwise = output_list[0]->has_data();
  const bool use_colwise = output_list[0]->has_columnwise_data();

  constexpr size_t CDY = 64;   // tile height (rows)
  constexpr size_t CDX = 64;   // tile width (cols)
  constexpr size_t TPC = 128;  // threads per block

  mxfp8::quantize_kernel::MultiQuantizeMXFP8Args args;
  args.num_tensors = 0;
  args.block_range[0] = 0;
  int tiles_x = 0;

  for (size_t i = 0; i < num_tensors; i++) {
    const int rows = input_list[i]->flat_first_dim();
    const int cols = input_list[i]->flat_last_dim();
    const int row_tiles = DIVUP(static_cast<size_t>(rows), CDY);
    const int col_tiles = DIVUP(static_cast<size_t>(cols), CDX);
    if (col_tiles > tiles_x) {
      tiles_x = col_tiles;
    }
    const int pos = args.num_tensors;

    args.input_list[pos] = input_list[i]->data.dptr;
    args.output_rowwise_list[pos] = use_rowwise ? output_list[i]->data.dptr : nullptr;
    args.output_colwise_list[pos] = use_colwise ? output_list[i]->columnwise_data.dptr : nullptr;
    args.scales_rowwise_list[pos] = use_rowwise ? output_list[i]->scale_inv.dptr : nullptr;
    args.scales_colwise_list[pos] = use_colwise ? output_list[i]->columnwise_scale_inv.dptr : nullptr;
    args.amax_list[pos] = reinterpret_cast<float *>(output_list[i]->amax.dptr);
    args.rows_list[pos] = rows;
    args.cols_list[pos] = cols;
    args.block_range[pos + 1] = args.block_range[pos] + row_tiles;
    args.num_tensors++;
  }

  if (args.num_tensors == 0) return;

  bool is_aligned = true;
  for (int i = 0; i < args.num_tensors; i++) {
    if (args.cols_list[i] % (32 * typeToSize(itype)) != 0) {
      is_aligned = false;
      break;
    }
  }

  const dim3 grid(tiles_x, args.block_range[args.num_tensors]);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(itype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(otype, OType,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH((use_colwise ? 32 : 1), SCALE_DIM_Y,
        TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH((use_rowwise ? 32 : 1), SCALE_DIM_X,
          if (is_aligned) {
            mxfp8::quantize_kernel::multi_quantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, true, CDY, CDX, TPC>
                <<<grid, TPC, 0, stream>>>(args);
          } else {
            mxfp8::quantize_kernel::multi_quantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, false, CDY, CDX, TPC>
                <<<grid, TPC, 0, stream>>>(args);
          }
        ));
      ));
  NVTE_CHECK_CUDA(cudaGetLastError());
}
#endif  // __HIP_PLATFORM_AMD__

}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_
