/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp4.cuh
 *  \brief Dispatch for MXFP4 quantization.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_

#include "../../common.h"
#include "cast_transpose_mxfp4_shuffled.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {

constexpr int MXFP4_BLOCK_SIZE = 32;

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize(const Tensor &input, const Tensor *act_input, const Tensor *noop,
              Tensor *output, Tensor *dbias, Tensor *workspace,
              const QuantizationConfig &quant_config, cudaStream_t stream) {
  NVTE_CHECK(!IS_ACT, "IS_ACT is not supported by NVTE_MXFP4_1D_SCALING");
  NVTE_CHECK(!IS_DBIAS, "IS_DBIAS is not supported by NVTE_MXFP4_1D_SCALING");
  NVTE_CHECK(!IS_DACT, "IS_DACT is not supported by NVTE_MXFP4_1D_SCALING");

  {
    hipDeviceProp_t prop;
    int device;
    NVTE_CHECK_CUDA(hipGetDevice(&device));
    NVTE_CHECK_CUDA(hipGetDeviceProperties(&prop, device));
    NVTE_CHECK(prop.major == 9 && prop.minor == 5,
               "MXFP4 quantization requires gfx950 (detected gfx",
               prop.major, prop.minor * 10, ")");
  }

  int M = static_cast<int>(input.flat_first_dim());
  int N = static_cast<int>(input.flat_last_dim());

  bool use_rowwise = output->has_data();
  bool use_colwise = output->has_columnwise_data();

  bool use_hadamard = quant_config.mxfp4_use_hadamard;
  bool scale_shuffle = output->with_gemm_swizzled_scales;
  bool data_shuffle_rowwise_fp4 = output->mxfp4_shuffle_rowwise_data;
  bool data_shuffle_columnwise_fp4 = output->mxfp4_shuffle_columnwise_data;

  // The plain (non-shuffled) columnwise transpose store is flushed with
  // coalesced 128-bit (uint4) writes into each column's M/2 packed bytes, so it
  // requires M/2 to be a multiple of 16, i.e. M % 32 == 0. Enforce it
  // here. (Rowwise-only and shuffled-columnwise stores do not use this path.)
  const bool uses_coalesced_colwise_store = use_colwise && !data_shuffle_columnwise_fp4;
  NVTE_CHECK(!uses_coalesced_colwise_store || (M % MXFP4_BLOCK_SIZE == 0),
             "MXFP4 columnwise cast/transpose requires the first (token) dimension "
             "to be a multiple of ", MXFP4_BLOCK_SIZE, " (got M=", M, ")");

  auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
  auto rup = [](int a, int b) { return ((a + b - 1) / b) * b; };

  int rowwise_scale_N = cdiv(N, MXFP4_BLOCK_SIZE);
  int rowwise_scale_M_pad = rup(M, 256);
  int rowwise_scale_N_pad = rup(rowwise_scale_N, 8);
  int rowwise_scale_stride = rowwise_scale_N_pad;
  if (use_rowwise && output->scale_inv.has_data() &&
      output->scale_inv.shape.size() >= 2) {
    rowwise_scale_stride = static_cast<int>(output->scale_inv.shape[1]);
    rowwise_scale_M_pad = static_cast<int>(output->scale_inv.shape[0]);
    rowwise_scale_N_pad = static_cast<int>(output->scale_inv.shape[1]);
  }

  int colwise_scale_M = N;
  int colwise_scale_N = cdiv(M, MXFP4_BLOCK_SIZE);
  int colwise_scale_M_pad = rup(N, 256);
  int colwise_scale_N_pad = rup(colwise_scale_N, 8);
  int colwise_scale_stride = colwise_scale_N_pad;
  if (use_colwise && output->columnwise_scale_inv.has_data() &&
      output->columnwise_scale_inv.shape.size() >= 2) {
    colwise_scale_stride = static_cast<int>(output->columnwise_scale_inv.shape[1]);
    colwise_scale_M_pad = static_cast<int>(output->columnwise_scale_inv.shape[0]);
    colwise_scale_N_pad = static_cast<int>(output->columnwise_scale_inv.shape[1]);
  }

  nvte_cast_transpose_mxfp4_fused_shuffle(
      input.data.dptr,
      use_rowwise ? output->data.dptr : nullptr,
      use_rowwise ? output->scale_inv.dptr : nullptr,
      use_colwise ? output->columnwise_data.dptr : nullptr,
      use_colwise ? output->columnwise_scale_inv.dptr : nullptr,
      M, N,
      use_rowwise, use_colwise,
      scale_shuffle,
      use_hadamard,
      data_shuffle_rowwise_fp4,
      data_shuffle_columnwise_fp4,
      rowwise_scale_stride, colwise_scale_stride,
      rowwise_scale_N, rowwise_scale_M_pad, rowwise_scale_N_pad,
      colwise_scale_M, colwise_scale_N,
      colwise_scale_M_pad, colwise_scale_N_pad,
      stream);
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
