/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// ROCm replacement for the TMA grouped scaled SwiGLU -> columnwise MXFP8 kernel.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

#include "rocm_colwise_mxfp8.cuh"
#include "../../util/system.h"
// After the util headers it uses.
#include "../core/rocm_tdm_common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_scaled_swiglu_kernel {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = rocm_colwise_mxfp8::kTileCols;
constexpr size_t THREADS_PER_CHUNK = rocm_colwise_mxfp8::kThreads;

// The gfx1250 TDM kernel gives each wave of a block its own kWaveSize columns of the chunk.
constexpr int kWaveSize = 32;
constexpr int kWaves = CHUNK_DIM_X / kWaveSize;  // 4
static_assert(kWaves * kWaveSize == THREADS_PER_CHUNK, "A wave per kWaveSize columns");

// rocm_colwise_mxfp8::store_tile<T, true> of a wave's columns of a tile.
template <typename T>
__device__ __forceinline__ void store_wave_tile(
    const T (&src)[rocm_colwise_mxfp8::kTileRows][kWaveSize], T *dst, const size_t row_base,
    const size_t col_base, const size_t rows, const size_t cols) {
  constexpr int kVec = ROCM_VEC_BYTES / sizeof(T);
  constexpr int kVecsPerRow = kWaveSize / kVec;
  for (int i = threadIdx.x % kWaveSize; i < rocm_colwise_mxfp8::kTileRows * kVecsPerRow;
       i += kWaveSize) {
    const int r = i / kVecsPerRow;
    const int c = (i % kVecsPerRow) * kVec;
    const size_t row = row_base + r;
    const size_t col = col_base + c;
    if (row < rows && col < cols) {
      *reinterpret_cast<uint4 *>(dst + row * cols + col) =
          *reinterpret_cast<const uint4 *>(&src[r][c]);
    }
  }
}

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

// rocm_group_scaled_swiglu_mxfp8_kernel on gfx1250: each wave moves its own kWaveSize columns of
// the chunk with the TDM, a stage ahead, and quantizes them as that kernel does, a column per lane,
// without waiting for the other waves.
template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    rocm_group_scaled_swiglu_mxfp8_tdm_kernel(const IType *const input, const IType *const prob,
                                              OType *const output, e8m0_t *const scales,
                                              const float *const noop,
                                              const int64_t *const offsets,
                                              const size_t num_tensors, const size_t rows,
                                              const size_t cols, const ParamOP p) {
#ifdef __gfx1250__
  using namespace rocm_colwise_mxfp8;
  constexpr int kStages = CHUNK_DIM_Y / kTileRows;

  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  // Rows past the last expert (paged stashing) hold no tokens.
  if (offsets != nullptr && block_offset_Y * cols >= static_cast<size_t>(offsets[num_tensors])) {
    return;
  }
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const int lane = threadIdx.x % kWaveSize;
  const size_t col_base = blockIdx.x * CHUNK_DIM_X + wave * kWaveSize;
  if (col_base >= cols) {
    return;
  }

  // Two stages of the wave's columns, the next loading while the current one is quantized.
  __shared__ alignas(128) IType act_sh[kWaves][2][kTileRows][kWaveSize];
  __shared__ alignas(128) IType gate_sh[kWaves][2][kTileRows][kWaveSize];
  __shared__ alignas(128) OType out_sh[kWaves][kTileRows][kWaveSize];
  // Holds prob/2 to pair with the doubled activation OP returns; exact, as 0.5 is a power of 2.
  __shared__ float half_prob_sh[kWaves][CHUNK_DIM_Y];

  const auto load_stage = [&](const int stage) {
    const size_t row = block_offset_Y + stage * kTileRows;
    // act is the first half of each [act|gate] row, gate the second.
    tdm::load_2d(act_sh[wave][stage % 2], input, rows, cols, 2 * cols, row, col_base);
    tdm::load_2d(gate_sh[wave][stage % 2], input + cols, rows, cols, 2 * cols, row, col_base);
  };
  load_stage(0);
  for (int row = lane; row < CHUNK_DIM_Y; row += kWaveSize) {
    half_prob_sh[wave][row] = 0.5f * static_cast<float>(prob[block_offset_Y + row]);
  }

  const size_t col = col_base + lane;
  for (int stage = 0; stage < kStages; ++stage) {
    if (stage + 1 < kStages) {
      load_stage(stage + 1);
      tdm::wait<2>();
    } else {
      tdm::wait<0>();
    }
    __builtin_amdgcn_wave_barrier();
    const size_t row_base = block_offset_Y + stage * kTileRows;
    const auto &act = act_sh[wave][stage % 2];
    const auto &gate = gate_sh[wave][stage % 2];
    if (col < cols) {
      float values[kTileRows];
#pragma unroll
      for (int i = 0; i < kTileRows; ++i) {
        const float act_elt = static_cast<float>(act[i][lane]);
        float gate_elt = static_cast<float>(gate[i][lane]);
        if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
          gate_elt = clamp_symmetric(gate_elt, p.limit) + p.glu_linear_offset;
        }
        float elt = OP(act_elt, p) * gate_elt * half_prob_sh[wave][stage * kTileRows + i];
        // Match round-trip precision of the plain quantize path (cast through IType).
        if constexpr (!std::is_same_v<IType, float>) {
          elt = static_cast<float>(static_cast<IType>(elt));
        }
        values[i] = elt;
      }
      scales[row_base / kTileRows * cols + col] =
          quantize_column<OType>(values, &out_sh[wave][0][lane], kWaveSize);
    }
    __builtin_amdgcn_wave_barrier();
    store_wave_tile(out_sh[wave], output, row_base, col_base, rows, cols);
  }
#endif  // __gfx1250__
}

template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType>
void rocm_group_scaled_swiglu(const IType *const input, const IType *const prob,
                              OType *const output, e8m0_t *const scales, const float *const noop,
                              const int64_t *const offsets, const size_t num_tensors,
                              const size_t rows, const size_t cols, const ParamOP &p,
                              cudaStream_t stream) {
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), rows / CHUNK_DIM_Y);
  if (common::use_tdm(rows, cols, {input, output})) {
    rocm_group_scaled_swiglu_mxfp8_tdm_kernel<ParamOP, OP, IType, OType>
        <<<grid, THREADS_PER_CHUNK, 0, stream>>>(input, prob, output, scales, noop, offsets,
                                                 num_tensors, rows, cols, p);
    NVTE_CHECK_CUDA(cudaGetLastError());
    return;
  }
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
