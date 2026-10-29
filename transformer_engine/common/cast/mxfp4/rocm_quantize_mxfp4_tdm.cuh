/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 MXFP4 quantize with the Tensor Data Mover: like the MXFP8 TDM path, each wave moves its
// own 32-row band of a 128x128 chunk global->LDS->global. It computes the scales and the Hadamard
// transform of cast_transpose_mxfp4_shuffled, whose header the includer must include first, and
// rounds to nearest even with the hardware conversion, as that kernel does on gfx950.

#include "../../util/rocm_device_utils.cuh"
#include "../../util/rocm_tdm.cuh"
#include "../core/rocm_tdm_common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace tdm_kernel {

using common::store_bytes;
using common::use_tdm;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t WAVE_SIZE = 32;
constexpr size_t WAVES = THREADS_PER_CHUNK / WAVE_SIZE;    // 4
constexpr size_t BAND_Y = CHUNK_DIM_Y / WAVES;             // 32
constexpr size_t COLS_PER_LANE = CHUNK_DIM_X / WAVE_SIZE;  // 4
constexpr size_t BLOCK_SIZE = 32;
constexpr size_t SCALES_PER_ROW = CHUNK_DIM_X / BLOCK_SIZE;  // 4
constexpr size_t BAND_BYTES = BAND_Y * CHUNK_DIM_X / 2;  // packed FP4 of a band
constexpr size_t CHUNK_BYTES = CHUNK_DIM_Y * CHUNK_DIM_X / 2;
// The shuffled FP4 layouts store each 16-row group contiguously, in 16-byte column slices.
constexpr size_t SHUFFLE_ROWS = 16;
constexpr size_t SHUFFLE_SLICE = 16;
static_assert(BAND_Y == BLOCK_SIZE, "A band must be one colwise scale block");
static_assert(BAND_Y == WAVE_SIZE, "A lane per band row writes the rowwise scales");

// Where the E8M0 scales of one direction go: scale (row, col) for col < valid_cols, at
// row * stride + col or, with shuffle, at its compute_scale_shuffle_index within the padding.
struct ScaleLayout {
  uint8_t *ptr;
  size_t stride;
  int valid_cols;
  int pad_rows;
  int pad_cols;
  bool shuffle;
};

#ifdef __gfx1250__
using common::cvt_8x_to_fp4;
using common::hadamard16;

__device__ __forceinline__ void write_scale(const ScaleLayout &layout, const int row, const int col,
                                            const uint8_t e8m0) {
  if (col >= layout.valid_cols) {
    return;
  }
  if (!layout.shuffle) {
    layout.ptr[row * layout.stride + col] = e8m0;
  } else if (row < layout.pad_rows && col < layout.pad_cols) {
    layout.ptr[te_mxfp4::compute_scale_shuffle_index(row, col, layout.pad_cols)] = e8m0;
  }
}

// Scales a 32-element block for FP4 (x holds n of its values; a lane group of 32 / n shares it)
// and returns its E8M0 scale. As fmaxf skips NaN, only an all-NaN block has a NaN amax.
template <int LANES>
__device__ __forceinline__ uint8_t block_scale(const float *x, const int n, float &scale) {
  float amax = fabsf(x[0]);
  for (int i = 1; i < n; i++) {
    amax = fmaxf(amax, fabsf(x[i]));
  }
  amax = rocm_subwarp_allreduce<LANES>(amax, rocm_op::max{});
  return te_mxfp4::compute_e8m0_scale(amax, scale);
}

__device__ __forceinline__ float bf16_bits_to_float(const uint32_t bits) {
  return __uint_as_float(bits << 16);
}
#endif  // __gfx1250__

// Quantizes chunk (blockIdx.y, blockIdx.x) of the rows x cols BF16 input to the rowwise FP4 data,
// rows x cols / 2 bytes, and the columnwise FP4 data of its transpose, cols x rows / 2 bytes.
template <bool ROWWISE, bool COLWISE, bool HADAMARD>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_mxfp4_tdm_kernel(const uint16_t *input, uint8_t *rowwise_fp4, uint8_t *colwise_fp4,
                              const ScaleLayout rowwise_scales, const ScaleLayout colwise_scales,
                              const uint32_t rows, const uint32_t cols, const bool shuffle_rowwise,
                              const bool shuffle_colwise) {
#ifdef __gfx1250__
  __shared__ alignas(128) uint16_t in_sh[WAVES][BAND_Y][CHUNK_DIM_X];
  __shared__ alignas(128) uint8_t rowwise_sh[ROWWISE ? WAVES : 1][BAND_BYTES];
  __shared__ alignas(4) uint8_t rowwise_scale_sh[ROWWISE ? WAVES : 1][BAND_Y][SCALES_PER_ROW];
  // The transpose of the chunk, so that its rows go out in CHUNK_DIM_Y / 2-byte segments.
  __shared__ alignas(128) uint8_t colwise_sh[COLWISE ? CHUNK_BYTES : 1];
  __shared__ alignas(4) uint8_t colwise_scale_sh[COLWISE ? CHUNK_DIM_X : 1][WAVES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const uint32_t band_offset_Y = blockIdx.y * CHUNK_DIM_Y + wave * BAND_Y;
  const uint32_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  // The columnwise stores need every wave at the barrier.
  const bool active = band_offset_Y < rows;
  if (!COLWISE && !active) {
    return;
  }
  auto &in = in_sh[wave];
  if (active) {
    tdm::load_2d(in, input, rows, cols, cols, band_offset_Y, chunk_offset_X);
    tdm::wait<0>();
  }

  if (ROWWISE && active) {
    // Per step, 8 lanes cover a row with 16 elements, one Hadamard half, each.
    constexpr int ELEMS = 16;
    constexpr int LANES_PER_ROW = CHUNK_DIM_X / ELEMS;  // 8
    constexpr int ROWS_PER_STEP = WAVE_SIZE / LANES_PER_ROW;  // 4
    const int lane_offset_Y = lane / LANES_PER_ROW;
    const int lane_offset_X = (lane % LANES_PER_ROW) * ELEMS;
    const int packed_X = lane_offset_X / 2;
    uint8_t *const out = rowwise_sh[wave];
    auto &scales = rowwise_scale_sh[wave];
#pragma unroll
    for (int step = 0; step < BAND_Y / ROWS_PER_STEP; step++) {
      const int y = step * ROWS_PER_STEP + lane_offset_Y;
      uint4 bits[2];
      bits[0] = *reinterpret_cast<const uint4 *>(&in[y][lane_offset_X]);
      bits[1] = *reinterpret_cast<const uint4 *>(&in[y][lane_offset_X + 8]);
      const uint32_t *const words = reinterpret_cast<const uint32_t *>(bits);
      float x[ELEMS];
#pragma unroll
      for (int i = 0; i < ELEMS / 2; i++) {
        x[2 * i] = bf16_bits_to_float(words[i] & 0xFFFF);
        x[2 * i + 1] = bf16_bits_to_float(words[i] >> 16);
      }
      if constexpr (HADAMARD) {
        hadamard16(x);
      }
      float scale;
      const uint8_t e8m0 = block_scale<BLOCK_SIZE / ELEMS>(x, ELEMS, scale);
      const uint2 packed = make_uint2(cvt_8x_to_fp4(x, scale), cvt_8x_to_fp4(x + 8, scale));
      const int offset = shuffle_rowwise
                             ? y / SHUFFLE_ROWS * (SHUFFLE_ROWS * CHUNK_DIM_X / 2) +
                                   packed_X / SHUFFLE_SLICE * (SHUFFLE_ROWS * SHUFFLE_SLICE) +
                                   y % SHUFFLE_ROWS * SHUFFLE_SLICE + packed_X % SHUFFLE_SLICE
                             : y * (CHUNK_DIM_X / 2) + packed_X;
      *reinterpret_cast<uint2 *>(out + offset) = packed;
      if (lane % 2 == 0) {
        scales[y][lane_offset_X / BLOCK_SIZE] = e8m0;
      }
    }
    const uint32_t packed_cols = cols / 2;
    if (shuffle_rowwise) {
      tdm::store_2d(rowwise_fp4,
                    *reinterpret_cast<uint8_t(*)[BAND_Y / SHUFFLE_ROWS]
                                                [SHUFFLE_ROWS * CHUNK_DIM_X / 2]>(out),
                    rows / SHUFFLE_ROWS, packed_cols * SHUFFLE_ROWS, packed_cols * SHUFFLE_ROWS,
                    band_offset_Y / SHUFFLE_ROWS, chunk_offset_X / 2 * SHUFFLE_ROWS);
    } else {
      tdm::store_2d(rowwise_fp4, *reinterpret_cast<uint8_t(*)[BAND_Y][CHUNK_DIM_X / 2]>(out), rows,
                    packed_cols, packed_cols, band_offset_Y, chunk_offset_X / 2);
    }
    // A lane per band row writes the row's scales.
    const uint32_t row = band_offset_Y + lane;
    const int first_scale = chunk_offset_X / BLOCK_SIZE;
    if (row < rows) {
      if (!rowwise_scales.shuffle) {
        store_bytes(rowwise_scales.ptr + row * rowwise_scales.stride + first_scale, scales[lane],
                    rowwise_scales.valid_cols - first_scale);
      } else {
#pragma unroll
        for (int s = 0; s < SCALES_PER_ROW; s++) {
          write_scale(rowwise_scales, row, first_scale + s, scales[lane][s]);
        }
      }
    }
  }

  if constexpr (COLWISE) {
    // A lane quantizes its COLS_PER_LANE columns one at a time; a band is one block of each. Its
    // 16 packed bytes are the band's segment of a row of the transpose: (c, wave) goes to
    // c * CHUNK_DIM_Y / 2 + wave * 16 or, shuffled, to its 16-column group's slice wave.
    if (active) {
      const int lane_offset_X = lane * COLS_PER_LANE;
      uint2 bits[BAND_Y];
#pragma unroll
      for (int y = 0; y < BAND_Y; y++) {
        bits[y] = *reinterpret_cast<const uint2 *>(&in[y][lane_offset_X]);
      }
#pragma unroll
      for (int i = 0; i < COLS_PER_LANE; i++) {
        float x[BAND_Y];
#pragma unroll
        for (int y = 0; y < BAND_Y; y++) {
          const uint32_t word = i < 2 ? bits[y].x : bits[y].y;
          x[y] = bf16_bits_to_float(i % 2 == 0 ? word & 0xFFFF : word >> 16);
        }
        if constexpr (HADAMARD) {
          hadamard16(x);
          hadamard16(x + 16);
        }
        float scale;
        const uint8_t e8m0 = block_scale<1>(x, BAND_Y, scale);
        const uint4 packed = make_uint4(cvt_8x_to_fp4(x, scale), cvt_8x_to_fp4(x + 8, scale),
                                        cvt_8x_to_fp4(x + 16, scale), cvt_8x_to_fp4(x + 24, scale));
        const int c = lane_offset_X + i;
        const int offset = shuffle_colwise
                               ? c / SHUFFLE_ROWS * (SHUFFLE_ROWS * CHUNK_DIM_Y / 2) +
                                     wave * (SHUFFLE_ROWS * SHUFFLE_SLICE) +
                                     c % SHUFFLE_ROWS * SHUFFLE_SLICE
                               : c * (CHUNK_DIM_Y / 2) + wave * (BAND_Y / 2);
        *reinterpret_cast<uint4 *>(colwise_sh + offset) = packed;
        colwise_scale_sh[c][wave] = e8m0;
      }
    }
    __syncthreads();

    // A thread per column writes its scales, at once where they are contiguous and aligned.
    const uint32_t col = chunk_offset_X + threadIdx.x;
    const int first_scale = blockIdx.y * WAVES;
    if (col < cols) {
      const ScaleLayout &layout = colwise_scales;
      if (!layout.shuffle) {
        store_bytes(layout.ptr + col * layout.stride + first_scale, colwise_scale_sh[threadIdx.x],
                    layout.valid_cols - first_scale);
      } else {
#pragma unroll
        for (int w = 0; w < WAVES; w++) {
          write_scale(layout, col, first_scale + w, colwise_scale_sh[threadIdx.x][w]);
        }
      }
    }
    if (wave == 0) {
      const uint32_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
      const uint32_t packed_rows = rows / 2;
      if (shuffle_colwise) {
        tdm::store_2d(colwise_fp4,
                      *reinterpret_cast<uint8_t(*)[CHUNK_DIM_X / SHUFFLE_ROWS]
                                                  [SHUFFLE_ROWS * CHUNK_DIM_Y / 2]>(colwise_sh),
                      cols / SHUFFLE_ROWS, packed_rows * SHUFFLE_ROWS, packed_rows * SHUFFLE_ROWS,
                      chunk_offset_X / SHUFFLE_ROWS, chunk_offset_Y / 2 * SHUFFLE_ROWS);
      } else {
        tdm::store_2d(colwise_fp4,
                      *reinterpret_cast<uint8_t(*)[CHUNK_DIM_X][CHUNK_DIM_Y / 2]>(colwise_sh),
                      cols, packed_rows, packed_rows, chunk_offset_X, chunk_offset_Y / 2);
      }
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

}  // namespace tdm_kernel

// MXFP4 quantize of input into output, with the scale layouts of quantize. Returns false, leaving
// the work to cast_transpose_mxfp4_shuffled, where TDM does not apply. A template, so that only the
// files calling it compile its kernels.
template <typename = void>
bool quantize_tdm(const Tensor &input, Tensor *output, const bool use_hadamard,
                  const tdm_kernel::ScaleLayout &rowwise_scales,
                  const tdm_kernel::ScaleLayout &colwise_scales, hipStream_t stream) {
  using namespace tdm_kernel;
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const bool rowwise = output->has_data();
  const bool colwise = output->has_columnwise_data();
  const bool shuffle_rowwise = rowwise && output->mxfp4_shuffle_rowwise_data;
  const bool shuffle_colwise = colwise && output->mxfp4_shuffle_columnwise_data;
  // Every block and, in the shuffled layouts, 16-row group of 32 packed bytes is complete.
  if (!(rowwise || colwise) || cols % BLOCK_SIZE != 0 || (colwise && rows % BLOCK_SIZE != 0) ||
      (shuffle_rowwise && (rows % SHUFFLE_ROWS != 0 || cols % (4 * SHUFFLE_SLICE) != 0)) ||
      (shuffle_colwise && (cols % SHUFFLE_ROWS != 0 || rows % (4 * SHUFFLE_SLICE) != 0)) ||
      !use_tdm(rows, cols,
               {input.data.dptr, rowwise ? output->data.dptr : nullptr,
                colwise ? output->columnwise_data.dptr : nullptr})) {
    return false;
  }
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  const auto launch = [&](auto kernel) {
    kernel<<<grid, THREADS_PER_CHUNK, 0, stream>>>(
        reinterpret_cast<const uint16_t *>(input.data.dptr),
        reinterpret_cast<uint8_t *>(output->data.dptr),
        reinterpret_cast<uint8_t *>(output->columnwise_data.dptr), rowwise_scales, colwise_scales,
        rows, cols, shuffle_rowwise, shuffle_colwise);
  };
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_hadamard, HADAMARD,
      if (rowwise && colwise) {
        launch(quantize_mxfp4_tdm_kernel<true, true, HADAMARD>);
      } else if (rowwise) {
        launch(quantize_mxfp4_tdm_kernel<true, false, HADAMARD>);
      } else {
        launch(quantize_mxfp4_tdm_kernel<false, true, HADAMARD>);
      });  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine
