/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// gfx1250 gated MXFP8 quantize with the Tensor Data Mover, included into gated_kernel after
// rocm_gated_mxfp8.cuh for its compute_gated_activation.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Each wave moves its own band of a 128x128 chunk of the act (and gate) half global->LDS->global,
// so the waves never synchronize.
template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(tdm_kernel::THREADS_PER_CHUNK)
    quantize_gated_mxfp8_tdm_kernel(
      const IType *grad_ptr, const IType *input_act, const IType *input_gate,
      OType *output_act_rowwise, OType *output_gate_rowwise, OType *output_act_colwise,
      OType *output_gate_colwise, e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
      const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
      const size_t scale_stride_colwise, const ParamOP p) {
#ifdef __gfx1250__
  // These hide the chunk constants of the generic gated kernel.
  using tdm_kernel::BAND_Y;
  using tdm_kernel::band_t;
  using tdm_kernel::CHUNK_DIM_X;
  using tdm_kernel::CHUNK_DIM_Y;
  using tdm_kernel::COLS_PER_LANE;
  using tdm_kernel::WAVE_SIZE;
  using tdm_kernel::WAVES;
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr size_t HALVES = IS_DGATED ? 2 : 1;
  constexpr size_t OUT_BAND_BYTES = sizeof(band_t<OType>);
  constexpr size_t HALF_OUT_BYTES = (USE_ROWWISE_SCALING + USE_COLWISE_SCALING) * OUT_BAND_BYTES;
  constexpr size_t AUX_BYTES = std::max(sizeof(band_t<IType>), HALVES * HALF_OUT_BYTES);
  static_assert(!USE_COLWISE_SCALING || SCALE_DIM_Y == BAND_Y,
                "A band must be one colwise scale block");

  // act (and for dgated gate) bands, which hold the activations after the first pass
  __shared__ alignas(128) band_t<IType> in_sh[WAVES][HALVES];
  // The band only the first pass reads (gate, or grad for dgated), then the outputs
  __shared__ alignas(128) unsigned char aux_sh[WAVES][AUX_BYTES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane_offset_X = (threadIdx.x % WAVE_SIZE) * COLS_PER_LANE;
  const size_t band_offset_Y = blockIdx.y * CHUNK_DIM_Y + wave * BAND_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  if (band_offset_Y >= rows) {
    return;
  }
  auto &act = in_sh[wave][0];
  auto &aux_in = *reinterpret_cast<band_t<IType> *>(aux_sh[wave]);
  auto &gate = IS_DGATED ? in_sh[wave][HALVES - 1] : aux_in;

  tdm::load_2d(act, input_act, rows, cols, 2 * cols, band_offset_Y, chunk_offset_X);
  tdm::load_2d(gate, input_gate, rows, cols, 2 * cols, band_offset_Y, chunk_offset_X);
  if constexpr (IS_DGATED) {
    tdm::load_2d(aux_in, grad_ptr, rows, cols, cols, band_offset_Y, chunk_offset_X);
  }
  tdm::wait<0>();

  // Out-of-bounds elements are zero and every gated activation keeps them zero.
#pragma unroll
  for (int y = 0; y < BAND_Y; y++) {
    Vec<IType, COLS_PER_LANE> a, g, d;
    a.load_from(&act[y][lane_offset_X]);
    g.load_from(&gate[y][lane_offset_X]);
    if constexpr (IS_DGATED) {
      d.load_from(&aux_in[y][lane_offset_X]);
    }
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      float grad_elt = 0.0f;
      if constexpr (IS_DGATED) {
        grad_elt = static_cast<float>(d.data.elt[i]);
      }
      float result_act, result_gate;
      compute_gated_activation<IS_DGATED, ParamOP, ActOP, DActOP, float>(
          static_cast<float>(a.data.elt[i]), static_cast<float>(g.data.elt[i]), grad_elt, p,
          result_act, result_gate);
      a.data.elt[i] = tdm_kernel::cvt_from_float<IType>(result_act);
      g.data.elt[i] = tdm_kernel::cvt_from_float<IType>(result_gate);
    }
    a.store_to(&act[y][lane_offset_X]);
    if constexpr (IS_DGATED) {
      g.store_to(&gate[y][lane_offset_X]);
    }
  }
  asm volatile("" ::: "memory");  // the outputs overwrite aux_in through another type

  const size_t out_ld = HALVES * cols;
  const size_t scale_cols = DIVUP(cols, SCALE_DIM_X);
  // Gate scales follow the act scales of each row (rowwise) or block row (colwise).
  auto quantize_half = [&](const band_t<IType> &in, OType *out_rowwise, OType *out_colwise,
                           const size_t scale_offset_rowwise, const size_t scale_offset_colwise,
                           unsigned char *out_sh) {
    if constexpr (USE_ROWWISE_SCALING) {
      auto &out = *reinterpret_cast<band_t<OType> *>(out_sh);
      const size_t scale_offset_X = scale_offset_rowwise + chunk_offset_X / SCALE_DIM_X;
      tdm_kernel::quantize_band_rowwise<IType, OType, SCALE_DIM_X>(
          in, out, scales_rowwise + band_offset_Y * scale_stride_rowwise + scale_offset_X,
          scale_stride_rowwise, rows - band_offset_Y, scale_cols - chunk_offset_X / SCALE_DIM_X);
      tdm::store_2d(out_rowwise, out, rows, cols, out_ld, band_offset_Y, chunk_offset_X);
    }
    if constexpr (USE_COLWISE_SCALING) {
      auto &out = *reinterpret_cast<band_t<OType> *>(
          out_sh + (USE_ROWWISE_SCALING ? OUT_BAND_BYTES : 0));
      tdm_kernel::quantize_band_colwise<IType, OType>(
          in, out,
          scales_colwise + band_offset_Y / SCALE_DIM_Y * scale_stride_colwise +
              scale_offset_colwise + chunk_offset_X,
          cols - chunk_offset_X);
      tdm::store_2d(out_colwise, out, rows, cols, out_ld, band_offset_Y, chunk_offset_X);
    }
  };
  quantize_half(act, output_act_rowwise, output_act_colwise, 0, 0, aux_sh[wave]);
  if constexpr (IS_DGATED) {
    quantize_half(gate, output_gate_rowwise, output_gate_colwise, scale_cols, cols,
                  aux_sh[wave] + HALF_OUT_BYTES);
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

// Returns false, leaving the work to the generic kernel, where TDM does not apply.
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
bool quantize_gated_tdm(const Tensor &gated_input, const Tensor &grad, Tensor *output, ParamOP &p,
                        hipStream_t stream) {
  const bool rowwise = output->has_data();
  const bool colwise = output->has_columnwise_data();
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  if (!tdm_kernel::use_tdm(rows, cols,
                           {gated_input.data.dptr, IS_BWD ? grad.data.dptr : nullptr,
                            output->data.dptr, output->columnwise_data.dptr})) {
    return false;
  }
  const dim3 grid(DIVUP(cols, tdm_kernel::CHUNK_DIM_X), DIVUP(rows, tdm_kernel::CHUNK_DIM_Y));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
              (colwise ? 32 : 1), SCALE_DIM_Y,
              TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                  (rowwise ? 32 : 1), SCALE_DIM_X,
                  // The gate half follows the act half in every row.
                  const IType *const input = reinterpret_cast<const IType *>(gated_input.data.dptr);
                  OType *const output_rowwise = reinterpret_cast<OType *>(output->data.dptr);
                  OType *const output_colwise =
                      reinterpret_cast<OType *>(output->columnwise_data.dptr);
                  quantize_gated_mxfp8_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                  SCALE_DIM_Y, SCALE_DIM_X>
                  <<<grid, tdm_kernel::THREADS_PER_CHUNK, 0, stream>>>(
                      IS_BWD ? reinterpret_cast<const IType *>(grad.data.dptr) : nullptr, input,
                      input + cols, output_rowwise, rowwise ? output_rowwise + cols : nullptr,
                      output_colwise, colwise ? output_colwise + cols : nullptr,
                      reinterpret_cast<e8m0_t *>(output->scale_inv.dptr),
                      reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr), rows, cols,
                      rowwise ? output->scale_inv.shape[1] : 1,
                      colwise ? output->columnwise_scale_inv.shape[1] : 1, p);))));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}
