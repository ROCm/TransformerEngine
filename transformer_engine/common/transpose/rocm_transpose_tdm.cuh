/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" // prevent hipification of this rocm_ file

// gfx1250 transpose of 1-, 2- and 4-byte elements with the Tensor Data Mover. A workgroup transposes a chunk of
// CHUNK_ROWS rows by CHUNK_BYTES bytes: each wave TDM-loads a BAND_ROWS-row band, its lanes transpose 4x4-byte
// blocks into the chunk's transpose in LDS, and after a barrier each wave TDM-stores its share of it. Expects its
// includer to have included the HIP runtime and common.h.

#include <cstddef>
#include <cstdint>

#include "../util/rocm_device_utils.cuh"

namespace transformer_engine {
namespace detail {
namespace transpose_tdm {

constexpr uint32_t CHUNK_ROWS = 128;
constexpr uint32_t CHUNK_BYTES = 128;
constexpr uint32_t WAVE_SIZE = 32;
constexpr uint32_t WAVES = 4;
constexpr uint32_t THREADS = WAVES * WAVE_SIZE;
constexpr uint32_t BAND_ROWS = CHUNK_ROWS / WAVES;  // 32
constexpr uint32_t QUADS = BAND_ROWS / 4;           // 8

// Block b transposes the chunk at row chunk b / col_chunks and column chunk b % col_chunks of the rows x cols input.
template <typename T>
__global__ void __launch_bounds__(THREADS)
    transpose_tdm_kernel(const T *input, T *output, const float *noop, const uint32_t rows, const uint32_t cols,
                         const uint32_t col_chunks) {
#ifdef __GFX12__
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  constexpr uint32_t COLS   = CHUNK_BYTES / sizeof(T);
  constexpr uint32_t ELEMS  = 4 / sizeof(T);  // elements of a band row per lane
  constexpr uint32_t T_ROWS = COLS / WAVES;   // rows of the chunk's transpose each wave stores
  __shared__ alignas(128) T in_sh[WAVES][BAND_ROWS][COLS];
  // TDM stores read LDS densely, so the transpose cannot be padded against bank conflicts.
  __shared__ alignas(128) T t_sh[WAVES][T_ROWS][CHUNK_ROWS];

  const int wave           = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane           = threadIdx.x % WAVE_SIZE;
  const uint32_t chunk_row = blockIdx.x / col_chunks * CHUNK_ROWS;
  const uint32_t chunk_col = blockIdx.x % col_chunks * COLS;
  const uint32_t band_row  = chunk_row + wave * BAND_ROWS;

  if (band_row < rows) {
    tdm::load_2d(in_sh[wave], input, rows, cols, cols, band_row, chunk_col);
    tdm::wait<0>();
    // A transposed word's bank depends only on its quad of rows, so the lanes start at different quads.
    const int first_quad = lane % QUADS;
#pragma unroll
    for (int step = 0; step < QUADS; step++) {
      const int quad = (first_quad + step) % QUADS;
      uint32_t packed[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        packed[i] = *reinterpret_cast<const uint32_t *>(&in_sh[wave][quad * 4 + i][lane * ELEMS]);
      }
      tdm::store_transposed_quad(t_sh, packed, lane * ELEMS, wave * BAND_ROWS + quad * 4);
    }
  }
  __syncthreads();
  // Columns past the input's rows are garbage in t_sh but fall outside the output.
  const uint32_t t_row = chunk_col + wave * T_ROWS;
  if (t_row < cols) {
    tdm::store_2d(output, t_sh[wave], cols, rows, rows, t_row, chunk_row);
  }
  tdm::wait<0>();
#endif  // __GFX12__
}

// Transposes the rows x cols row-major input into the cols x rows output on gfx1250, unless noop is set. The TDM needs
// 16-byte aligned tensors with rows and cols divisible by 16.
template <typename T>
void launch(const T *input, T *output, const size_t rows, const size_t cols, const float *noop,
            hipStream_t stream) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                "TDM transpose supports 1-, 2- and 4-byte elements");
  constexpr size_t COLS   = CHUNK_BYTES / sizeof(T);
  const size_t col_chunks = (cols + COLS - 1) / COLS;
  const size_t blocks     = (rows + CHUNK_ROWS - 1) / CHUNK_ROWS * col_chunks;
  transpose_tdm_kernel<T><<<blocks, THREADS, 0, stream>>>(input, output, noop, static_cast<uint32_t>(rows),
                                                          static_cast<uint32_t>(cols),
                                                          static_cast<uint32_t>(col_chunks));
}

}  // namespace transpose_tdm
}  // namespace detail
}  // namespace transformer_engine
