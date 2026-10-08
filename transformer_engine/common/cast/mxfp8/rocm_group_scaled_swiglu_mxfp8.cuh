/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// ROCm replacement for the TMA grouped scaled SwiGLU -> columnwise MXFP8 kernel.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

#include "rocm_colwise_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_scaled_swiglu_kernel {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = rocm_colwise_mxfp8::kTileCols;
constexpr size_t THREADS_PER_CHUNK = rocm_colwise_mxfp8::kThreads;

// One block covers CHUNK_DIM_Y tokens by CHUNK_DIM_X output columns, a 32-row scale block
// per stage. ROCm scales are compact and unpadded, and every expert's token count is a
// multiple of 128, so the experts' data and scales stack into one dense [N, H] tensor and
// one dense [N / 32, H] scale matrix: no per-expert bookkeeping is needed.
template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType,
          bool IS_ALIGNED>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    rocm_group_scaled_swiglu_mxfp8_kernel(const IType *const input, const IType *const prob,
                                          OType *const output, e8m0_t *const scales,
                                          const float *const noop, const int64_t *const offsets,
                                          const size_t num_tensors, const size_t rows,
                                          const size_t cols, const ParamOP p) {
  using namespace rocm_colwise_mxfp8;

  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;
  // Rows past the last expert (paged stashing) hold no tokens.
  if (offsets != nullptr && block_offset_Y * cols >= static_cast<size_t>(offsets[num_tensors])) {
    return;
  }

  alignas(16) __shared__ IType act_sh[kTileRows][kTileCols];
  alignas(16) __shared__ IType gate_sh[kTileRows][kTileCols];
  alignas(16) __shared__ OType out_sh[kTileRows][kTileCols];
  // Holds prob/2 to pair with the doubled activation OP returns; exact, as 0.5 is a power of 2.
  __shared__ float half_prob_sh[CHUNK_DIM_Y];

  for (size_t row = threadIdx.x; row < CHUNK_DIM_Y; row += blockDim.x) {
    half_prob_sh[row] = 0.5f * static_cast<float>(prob[block_offset_Y + row]);
  }

  const int tid = threadIdx.x;
  const size_t col = block_offset_X + tid;

  for (size_t stage = 0; stage < CHUNK_DIM_Y / kTileRows; ++stage) {
    const size_t row_base = block_offset_Y + stage * kTileRows;

    // act is the first half of each [act|gate] row, gate the second.
    load_tile<IType, IS_ALIGNED>(act_sh, input, row_base, block_offset_X, 2 * cols, rows, cols);
    load_tile<IType, IS_ALIGNED>(gate_sh, input + cols, row_base, block_offset_X, 2 * cols, rows,
                                 cols);
    __syncthreads();

    if (col < cols) {
      float values[kTileRows];
#pragma unroll
      for (int i = 0; i < kTileRows; ++i) {
        const float act_elt = static_cast<float>(act_sh[i][tid]);
        float gate_elt = static_cast<float>(gate_sh[i][tid]);
        if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
          gate_elt = clamp_symmetric(gate_elt, p.limit) + p.glu_linear_offset;
        }
        float elt = OP(act_elt, p) * gate_elt * half_prob_sh[stage * kTileRows + i];
        // Match round-trip precision of the plain quantize path (cast through IType).
        if constexpr (!std::is_same_v<IType, float>) {
          elt = static_cast<float>(static_cast<IType>(elt));
        }
        values[i] = elt;
      }
      scales[row_base / kTileRows * cols + col] =
          quantize_column<OType>(values, &out_sh[0][tid], kTileCols);
    }
    __syncthreads();

    store_tile<OType, IS_ALIGNED>(out_sh, output, row_base, block_offset_X, rows, cols);
    __syncthreads();
  }
}

template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType>
void rocm_group_scaled_swiglu(const IType *const input, const IType *const prob,
                              OType *const output, e8m0_t *const scales, const float *const noop,
                              const int64_t *const offsets, const size_t num_tensors,
                              const size_t rows, const size_t cols, const ParamOP &p,
                              cudaStream_t stream) {
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), rows / CHUNK_DIM_Y);
  // Vector access needs whole 16-byte vectors in every row, act and gate halves included.
  const bool aligned = cols % ROCM_VEC_BYTES == 0 && is_aligned_ptr(input, ROCM_VEC_BYTES) &&
                       is_aligned_ptr(output, ROCM_VEC_BYTES);
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      aligned, IS_ALIGNED,
      rocm_group_scaled_swiglu_mxfp8_kernel<ParamOP, OP, IType, OType, IS_ALIGNED>
      <<<grid, THREADS_PER_CHUNK, 0, stream>>>(input, prob, output, scales, noop, offsets,
                                               num_tensors, rows, cols, p););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace group_scaled_swiglu_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
