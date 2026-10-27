/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// gfx1250 MXFP8 dequantize with the Tensor Data Mover, included into dequantize_kernel after
// rocm_dequantize_mxfp8.cuh, whose grid and scale indexing it shares.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Each wave moves its own TDM_BAND_Y-row band of the chunk global->LDS->global, so waves never
// wait for each other.
constexpr size_t TDM_WAVE_SIZE = 32;
constexpr size_t TDM_WAVES = THREADS_PER_CHUNK / TDM_WAVE_SIZE;  // 4
constexpr size_t TDM_BAND_Y = CHUNK_DIM_Y / TDM_WAVES;          // 32

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X,
          bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_tdm_kernel(const IType *input_ptr, OType *output_ptr,
                                const e8m0_t *const scales_ptr, const size_t rows,
                                const size_t cols, const size_t scales_stride,
                                const size_t mx_swizzle_padded_dim) {
#ifdef __gfx1250__
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  __shared__ alignas(128) IType in_sh[TDM_WAVES][TDM_BAND_Y][CHUNK_DIM_X];
  __shared__ alignas(128) OType out_sh[TDM_WAVES][TDM_BAND_Y][CHUNK_DIM_X];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / TDM_WAVE_SIZE);
  const int lane = threadIdx.x % TDM_WAVE_SIZE;
  const size_t band_offset_Y = blockIdx.y * CHUNK_DIM_Y + wave * TDM_BAND_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  if (band_offset_Y >= rows) {
    return;
  }
  tdm::load_2d(in_sh[wave], input_ptr, rows, cols, cols, band_offset_Y, chunk_offset_X);

  // Scales are fetched while the copy is in flight. Outside the scale tensor a lane reads a dummy
  // in-bounds scale and uses 2^0 instead, so the loads need no branches and are issued together.
  const size_t scales_rows = USE_ROWWISE_SCALING ? rows : DIVUP(rows, SCALE_DIM_Y);
  const size_t scales_cols = USE_ROWWISE_SCALING ? DIVUP(cols, SCALE_DIM_X) : cols;
  const auto load_scale = [&](const size_t scale_offset_Y, const size_t scale_offset_X) {
    const bool in_bounds = scale_offset_Y < scales_rows && scale_offset_X < scales_cols;
    size_t scale_idx;
    if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
      scale_idx = USE_ROWWISE_SCALING
                      ? mx_preswizzle_scale_idx(scale_offset_Y, scale_offset_X, mx_swizzle_padded_dim)
                      : mx_preswizzle_scale_idx(scale_offset_X, scale_offset_Y, mx_swizzle_padded_dim);
    } else {
      scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
    }
    const e8m0_t biased_exponent = scales_ptr[in_bounds ? scale_idx : 0];
    return in_bounds ? biased_exponent : static_cast<e8m0_t>(127);
  };

  if constexpr (USE_ROWWISE_SCALING) {
    // Per step, THREADS_PER_CHUNK_X_ROWWISE lanes cover one row of the band.
    constexpr size_t ROWS_PER_STEP = TDM_WAVE_SIZE / THREADS_PER_CHUNK_X_ROWWISE;  // 4
    constexpr size_t STEPS = TDM_BAND_Y / ROWS_PER_STEP;                           // 8
    const int lane_offset_Y = lane / THREADS_PER_CHUNK_X_ROWWISE;
    const int lane_offset_X = (lane % THREADS_PER_CHUNK_X_ROWWISE) * ELEMS_PER_THREAD;
    e8m0_t biased_exponent[STEPS];
#pragma unroll
    for (int step = 0; step < STEPS; step++) {
      biased_exponent[step] = load_scale(band_offset_Y + step * ROWS_PER_STEP + lane_offset_Y,
                                         (chunk_offset_X + lane_offset_X) / SCALE_DIM_X);
    }
    tdm::wait<0>();
#pragma unroll
    for (int step = 0; step < STEPS; step++) {
      const int y = step * ROWS_PER_STEP + lane_offset_Y;
      const float block_scale = ptx::exp2f(biased_exponent[step]);
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;
      in.load_from(&in_sh[wave][y][lane_offset_X]);
#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        out.data.elt[j] = cvt_from_float<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[wave][y][lane_offset_X]);
    }
  } else {
    // The band is one scale block tall; each lane owns COLS_PER_LANE adjacent columns.
    constexpr size_t COLS_PER_LANE = CHUNK_DIM_X / TDM_WAVE_SIZE;  // 4
    const int lane_offset_X = lane * COLS_PER_LANE;
    e8m0_t biased_exponent[COLS_PER_LANE];
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      biased_exponent[i] = load_scale(band_offset_Y / SCALE_DIM_Y, chunk_offset_X + lane_offset_X + i);
    }
    tdm::wait<0>();
    float block_scale[COLS_PER_LANE];
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      block_scale[i] = ptx::exp2f(biased_exponent[i]);
    }
#pragma unroll
    for (int y = 0; y < TDM_BAND_Y; y++) {
      Vec<IType, COLS_PER_LANE> in;
      Vec<OType, COLS_PER_LANE> out;
      in.load_from(&in_sh[wave][y][lane_offset_X]);
#pragma unroll
      for (int i = 0; i < COLS_PER_LANE; i++) {
        out.data.elt[i] = cvt_from_float<OType>(block_scale[i] * static_cast<float>(in.data.elt[i]));
      }
      out.store_to(&out_sh[wave][y][lane_offset_X]);
    }
  }

  tdm::store_2d(output_ptr, out_sh[wave], rows, cols, cols, band_offset_Y, chunk_offset_X);
  tdm::wait<0>();
#endif  // __gfx1250__
}

// Returns false, leaving the work to the generic kernel, where TDM does not apply. The scale
// arguments are the ones of the generic kernel.
inline bool dequantize_tdm(const Tensor &input, const SimpleTensor &input_data, Tensor *output,
                           const e8m0_t *const scales_ptr, const size_t scales_stride,
                           const size_t mx_swizzle_padded_dim, hipStream_t stream) {
  const auto [rows, cols] = input.flat_2d_dims();
  if (!tdm_kernel::use_tdm(rows, cols, {input_data.dptr, output->data.dptr})) {
    return false;
  }
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      (input.has_columnwise_data() ? 32 : 1), SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          (input.has_data() ? 32 : 1), SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
                  output->dtype(), OType,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      input.with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
                      dequantize_mxfp8_tdm_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X,
                                                  WITH_GEMM_SWIZZLED_SCALES>
                      <<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                          reinterpret_cast<const IType *>(input_data.dptr),
                          reinterpret_cast<OType *>(output->data.dptr), scales_ptr, rows, cols,
                          scales_stride, mx_swizzle_padded_dim););))));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}
