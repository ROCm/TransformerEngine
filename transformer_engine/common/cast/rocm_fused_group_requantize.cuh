/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// ROCm replacement for the TMA fused grouped MXFP8 requantize kernel.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

#include "mxfp8/rocm_colwise_mxfp8.cuh"

namespace transformer_engine {
namespace requantize {

// Rowwise MXFP8 grouped input -> columnwise MXFP8 output, plus an optional BF16 dequantized
// copy, in one pass over a 32 x 128 tile per block. ROCm GEMMs read compact scales, so unlike
// the CUDA kernel nothing is swizzled: the rowwise scales are copied verbatim (skipped when the
// output aliases them), and under the /128 contract every group's columnwise scales stack into
// one dense [rows / 32, cols] matrix, so no per-group lookup is needed. The arithmetic is that
// of the unfused group_dequantize -> group_quantize(columnwise) chain.
template <typename IType, typename OType, bool kUseFastMath, bool kReturnDequantized>
__global__ void __launch_bounds__(rocm_colwise_mxfp8::kThreads)
    rocm_fused_group_requantize_kernel(const IType *const input,
                                       const e8m0_t *const input_scale_inv,
                                       e8m0_t *const rowwise_scale_inv, OType *const colwise_data,
                                       e8m0_t *const colwise_scale_inv,
                                       bf16 *const dequantized_out,
                                       const int64_t *const tensor_offsets, const int num_groups,
                                       const int num_cols, const int input_scale_stride) {
  using namespace rocm_colwise_mxfp8;
  constexpr int kScaleDim = 32;
  constexpr int kScalesPerRow = kTileCols / kScaleDim;

  const size_t row_base = static_cast<size_t>(blockIdx.y) * kTileRows;
  const size_t col_base = static_cast<size_t>(blockIdx.x) * kTileCols;
  const size_t num_rows = static_cast<size_t>(gridDim.y) * kTileRows;

  // Rows at or past the live end (capacity mode / paged stashing) are left untouched.
  if (static_cast<int64_t>(row_base * num_cols) >= tensor_offsets[num_groups]) {
    return;
  }

  alignas(16) __shared__ IType in_sh[kTileRows][kTileCols];
  alignas(16) __shared__ OType out_sh[kTileRows][kTileCols];
  alignas(16) __shared__ bf16 dequantized_sh[kReturnDequantized ? kTileRows : 1][kTileCols];
  __shared__ e8m0_t scale_sh[kTileRows][kScalesPerRow];

  const int tid = threadIdx.x;
  load_tile<IType, true>(in_sh, input, row_base, col_base, num_cols, num_rows, num_cols);
  {
    static_assert(kThreads == kTileRows * kScalesPerRow);
    const int r = tid / kScalesPerRow;
    const int s = tid % kScalesPerRow;
    const size_t scale_col = col_base / kScaleDim + s;
    const e8m0_t scale = input_scale_inv[(row_base + r) * input_scale_stride + scale_col];
    scale_sh[r][s] = scale;
    if (rowwise_scale_inv != input_scale_inv) {
      rowwise_scale_inv[(row_base + r) * (num_cols / kScaleDim) + scale_col] = scale;
    }
  }
  __syncthreads();

  float values[kTileRows];
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    float value = ptx::exp2f(scale_sh[i][tid / kScaleDim]) * static_cast<float>(in_sh[i][tid]);
    if constexpr (kUseFastMath || kReturnDequantized) {
      const bf16 value_bf16 = static_cast<bf16>(value);
      if constexpr (kUseFastMath) {
        // The unfused chain materializes the dequantized tensor in BF16.
        value = static_cast<float>(value_bf16);
      }
      if constexpr (kReturnDequantized) {
        dequantized_sh[i][tid] = value_bf16;
      }
    }
    values[i] = value;
  }
  colwise_scale_inv[row_base / kTileRows * num_cols + col_base + tid] =
      quantize_column<OType>(values, &out_sh[0][tid], kTileCols);
  __syncthreads();

  store_tile<OType, true>(out_sh, colwise_data, row_base, col_base, num_rows, num_cols);
  if constexpr (kReturnDequantized) {
    store_tile<bf16, true>(dequantized_sh, dequantized_out, row_base, col_base, num_rows,
                           num_cols);
  }
}

template <typename IType, bool kUseFastMath, bool kReturnDequantized>
void rocm_launch_fused_group_requantize(const Tensor &input, Tensor *output,
                                        const Tensor &tensor_offsets, Tensor *dequantized,
                                        const int num_groups, const int num_rows,
                                        const int num_cols, const int input_scale_stride,
                                        cudaStream_t stream) {
  using OType = fp8e4m3;
  const dim3 grid(num_cols / rocm_colwise_mxfp8::kTileCols,
                  num_rows / rocm_colwise_mxfp8::kTileRows);
  bf16 *dequantized_ptr = nullptr;
  if constexpr (kReturnDequantized) {
    dequantized_ptr = reinterpret_cast<bf16 *>(dequantized->data.dptr);
  }
  rocm_fused_group_requantize_kernel<IType, OType, kUseFastMath, kReturnDequantized>
      <<<grid, rocm_colwise_mxfp8::kThreads, 0, stream>>>(
          reinterpret_cast<const IType *>(input.data.dptr),
          reinterpret_cast<const e8m0_t *>(input.scale_inv.dptr),
          reinterpret_cast<e8m0_t *>(output->scale_inv.dptr),
          reinterpret_cast<OType *>(output->columnwise_data.dptr),
          reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr), dequantized_ptr,
          reinterpret_cast<const int64_t *>(tensor_offsets.data.dptr), num_groups, num_cols,
          input_scale_stride);
}

}  // namespace requantize
}  // namespace transformer_engine
