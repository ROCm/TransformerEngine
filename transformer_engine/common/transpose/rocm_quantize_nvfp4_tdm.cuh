/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 NVFP4 quantize with the Tensor Data Mover, included in quantize_transpose_nvfp4 after the
// scale helpers it shares with block_scaled_1d_cast_transpose_kernel, whose numerics it reproduces.
// As in the MXFP8 and MXFP4 TDM paths, each wave moves its own 32-row band of a 128x128 chunk
// global->LDS->global, and the transpose is assembled in LDS for the whole chunk.

namespace nvfp4_tdm {

constexpr int kChunkDim = 128;
constexpr int kThreads = 128;
constexpr int kWaveSize = 32;
constexpr int kWaves = kThreads / kWaveSize;            // 4
constexpr int kBandY = kChunkDim / kWaves;              // 32
constexpr int kColsPerLane = kChunkDim / kWaveSize;     // 4
constexpr int kBlockDim = 16;                           // values per scale
constexpr int kBlocks = kChunkDim / kBlockDim;          // 8 per chunk row or column
constexpr int kBandBlocks = kBandY / kBlockDim;         // 2
static_assert(kBandY == kWaveSize, "A lane per band row writes the rowwise scales");
static_assert(kThreads == kChunkDim, "A thread per chunk column writes the columnwise scales");

using Philox =
    transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>;

#ifdef __gfx1250__
// src[0..N) as floats, read up to 16 bytes at a time.
template <int N, typename IType>
__device__ __forceinline__ void load_floats(const IType *src, float *x) {
  constexpr int kVec = N < 16 / sizeof(IType) ? N : 16 / sizeof(IType);
#pragma unroll
  for (int i = 0; i < N; i += kVec) {
    Vec<IType, kVec> v;
    v.load_from(src + i);
#pragma unroll
    for (int j = 0; j < kVec; j++) {
      x[i + j] = static_cast<float>(v.data.elt[j]);
    }
  }
}

// The 16 values x with amax as NVFP4, the first in the low nibble of packed; returns the E4M3
// scale. Stochastic rounding takes a byte of one draw of rng per value, as the generic kernel does.
template <bool kSR>
__device__ __forceinline__ uint8_t quantize_block(const float *x, const float amax,
                                                  const float encode_multiplier,
                                                  const float decode_scale, Philox &rng,
                                                  uint2 &packed) {
  const fp8e4m3 scale_inv = ComputeDecodeScaleFP4<fp8e4m3>(amax, encode_multiplier);
  const float encode_scale = ComputeEncodeScaleFP4<fp8e4m3>(scale_inv, decode_scale);
  float y[kBlockDim];
#pragma unroll
  for (int i = 0; i < kBlockDim; i++) {
    y[i] = x[i] * encode_scale;
  }
  if constexpr (kSR) {
    const uint4 rbits = rng.generate4();
    packed.x = dispatch::common::cvt_8x_to_fp4_sr(y, make_uint2(rbits.x, rbits.y), 1.0f);
    packed.y = dispatch::common::cvt_8x_to_fp4_sr(y + 8, make_uint2(rbits.z, rbits.w), 1.0f);
  } else {
    packed.x = dispatch::common::cvt_8x_to_fp4(y, 1.0f);
    packed.y = dispatch::common::cvt_8x_to_fp4(y + 8, 1.0f);
  }
  return __builtin_bit_cast(uint8_t, scale_inv);
}
#endif  // __gfx1250__

// Quantizes chunk (blockIdx.y, blockIdx.x) of the rows x cols input to the rowwise NVFP4 data,
// rows x cols / 2 bytes, and the columnwise data of its transpose, cols x rows / 2 bytes.
template <bool kRowwise, bool kColwise, bool k2D, bool kSR, typename IType>
__global__ void __launch_bounds__(kThreads)
    quantize_kernel(const IType *input, const float *global_amax, uint8_t *output_c,
                    uint8_t *output_t, uint8_t *scales_c, uint8_t *scales_t, const uint32_t rows,
                    const uint32_t cols, const size_t scale_stride, const size_t scale_t_stride,
                    const bool row_scaled, const size_t *rng_state, const float *noop,
                    const float e4m3_max) {
#ifdef __gfx1250__
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  __shared__ alignas(128) IType in_sh[kWaves][kBandY][kChunkDim];
  __shared__ alignas(128) uint8_t rowwise_sh[kRowwise ? kWaves : 1][kBandY][kChunkDim / 2];
  __shared__ alignas(8) uint8_t rowwise_scale_sh[kRowwise ? kWaves : 1][kBandY][kBlocks];
  // The transpose of the chunk, so that its rows go out in kChunkDim / 2-byte segments.
  __shared__ alignas(128) uint8_t colwise_sh[kColwise ? kChunkDim : 1][kChunkDim / 2];
  __shared__ alignas(8) uint8_t colwise_scale_sh[kColwise ? kChunkDim : 1][kBlocks];
  __shared__ float block_amax_sh[k2D ? kWaves : 1][kBandBlocks][kBlocks];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const int lane = threadIdx.x % kWaveSize;
  const uint32_t chunk_offset_Y = blockIdx.y * kChunkDim;
  const uint32_t chunk_offset_X = blockIdx.x * kChunkDim;
  const uint32_t band_offset_Y = chunk_offset_Y + wave * kBandY;
  // The columnwise stores need every wave at the barrier.
  const bool active = band_offset_Y < rows;
  if (!kColwise && !active) {
    return;
  }
  auto &in = in_sh[wave];
  if (active) {
    tdm::load_2d(in, input, rows, cols, cols, band_offset_Y, chunk_offset_X);
  }

  constexpr float fp4_max_inv = 1.0f / TypeExtrema<fp4e2m1>::max;
  const float global_encode_scale =
      global_amax == nullptr ? 1.0f : ComputeGlobalEncodeScaleFP4(global_amax[0], e4m3_max);
  const float global_encode_multiplier = global_encode_scale * fp4_max_inv;
  const float global_decode_scale = 1.0 / global_encode_scale;
  Philox rng;
  if constexpr (kSR) {
    const size_t sequence =
        threadIdx.x + (blockIdx.x + static_cast<size_t>(blockIdx.y) * gridDim.x) * kThreads;
    rng.init(rng_state != nullptr ? rng_state[0] : 0, sequence,
             rng_state != nullptr ? rng_state[1] : 0);
  }
  if (active) {
    tdm::wait<0>();
  }

  if constexpr (k2D) {
    // The amaxes of the band's 16x16 blocks, each from 4 lanes.
    constexpr int kLanesPerBlock = kBlockDim / kColsPerLane;
    if (active) {
#pragma unroll
      for (int b = 0; b < kBandBlocks; b++) {
        float amax = 0.0f;
#pragma unroll
        for (int y = 0; y < kBlockDim; y++) {
          float x[kColsPerLane];
          load_floats<kColsPerLane>(&in[b * kBlockDim + y][lane * kColsPerLane], x);
#pragma unroll
          for (int i = 0; i < kColsPerLane; i++) {
            amax = fmaxf(amax, fabsf(x[i]));
          }
        }
        amax = rocm_subwarp_allreduce<kLanesPerBlock>(amax, rocm_op::max{});
        if (lane % kLanesPerBlock == 0) {
          block_amax_sh[wave][b][lane / kLanesPerBlock] = amax;
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }

  if (kRowwise && active) {
    // Per step, a lane quantizes a block of a row, kBlocks lanes the whole row.
    constexpr int kRowsPerStep = kWaveSize / kBlocks;  // 4
    const int block = lane % kBlocks;
    auto &out = rowwise_sh[wave];
    auto &scales = rowwise_scale_sh[wave];
#pragma unroll
    for (int step = 0; step < kBandY / kRowsPerStep; step++) {
      const int y = step * kRowsPerStep + lane / kBlocks;
      const uint32_t row = band_offset_Y + y;
      float x[kBlockDim];
      load_floats<kBlockDim>(&in[y][block * kBlockDim], x);
      float amax = 0.0f;
      if constexpr (k2D) {
        amax = block_amax_sh[wave][y / kBlockDim][block];
      } else {
#pragma unroll
        for (int i = 0; i < kBlockDim; i++) {
          amax = fmaxf(amax, fabsf(x[i]));
        }
      }
      float encode_multiplier = global_encode_multiplier;
      float decode_scale = global_decode_scale;
      if (row_scaled) {
        const float row_encode_scale =
            row < rows ? ComputeGlobalEncodeScaleFP4(global_amax[row], e4m3_max) : 1.0f;
        encode_multiplier = row_encode_scale * fp4_max_inv;
        decode_scale = 1.0f / row_encode_scale;
      }
      uint2 packed;
      scales[y][block] = quantize_block<kSR>(x, amax, encode_multiplier, decode_scale, rng, packed);
      *reinterpret_cast<uint2 *>(&out[y][block * kBlockDim / 2]) = packed;
    }
    tdm::store_2d(output_c, out, rows, cols / 2, cols / 2, band_offset_Y, chunk_offset_X / 2);
    // A lane per band row writes the row's scales.
    const uint32_t row = band_offset_Y + lane;
    const uint32_t first_scale = chunk_offset_X / kBlockDim;
    if (row < rows) {
      dispatch::common::store_bytes(scales_c + row * scale_stride + first_scale, scales[lane],
                                    DIVUP(cols, uint32_t{kBlockDim}) - first_scale);
    }
  }

  if constexpr (kColwise) {
    // A lane quantizes the blocks of its kColsPerLane columns; block b of column c is segment
    // wave * kBandBlocks + b of row c of the transpose.
    if (active) {
      const int lane_offset_X = lane * kColsPerLane;
#pragma unroll
      for (int b = 0; b < kBandBlocks; b++) {
        float x[kColsPerLane][kBlockDim];
#pragma unroll
        for (int y = 0; y < kBlockDim; y++) {
          float v[kColsPerLane];
          load_floats<kColsPerLane>(&in[b * kBlockDim + y][lane_offset_X], v);
#pragma unroll
          for (int i = 0; i < kColsPerLane; i++) {
            x[i][y] = v[i];
          }
        }
#pragma unroll
        for (int i = 0; i < kColsPerLane; i++) {
          const int c = lane_offset_X + i;
          const int segment = wave * kBandBlocks + b;
          float amax = 0.0f;
          if constexpr (k2D) {
            amax = block_amax_sh[wave][b][c / kBlockDim];
          } else {
#pragma unroll
            for (int y = 0; y < kBlockDim; y++) {
              amax = fmaxf(amax, fabsf(x[i][y]));
            }
          }
          uint2 packed;
          colwise_scale_sh[c][segment] = quantize_block<kSR>(
              x[i], amax, global_encode_multiplier, global_decode_scale, rng, packed);
          *reinterpret_cast<uint2 *>(&colwise_sh[c][segment * kBlockDim / 2]) = packed;
        }
      }
    }
    __syncthreads();
    if (wave == 0) {
      tdm::store_2d(output_t, colwise_sh, cols, rows / 2, rows / 2, chunk_offset_X,
                    chunk_offset_Y / 2);
    }
    // A thread per column writes its scales.
    const uint32_t col = chunk_offset_X + threadIdx.x;
    const uint32_t first_scale = chunk_offset_Y / kBlockDim;
    if (col < cols) {
      dispatch::common::store_bytes(scales_t + col * scale_t_stride + first_scale,
                                    colwise_scale_sh[threadIdx.x],
                                    DIVUP(rows, uint32_t{kBlockDim}) - first_scale);
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

// NVFP4 quantize as quantize_transpose_vector_blockwise_fp4 does it. Returns false, leaving the
// work to block_scaled_1d_cast_transpose_kernel, where TDM does not apply; that includes swizzled
// scales, which only GPUs without TDM request. Stochastic rounding draws its own random bits, so
// it matches that kernel in distribution rather than bitwise.
inline bool quantize(const SimpleTensor &input, const SimpleTensor &global_amax,
                     const SimpleTensor &scale_inv, const SimpleTensor &scale_inv_t,
                     const SimpleTensor &output, const SimpleTensor &output_t, const size_t rows,
                     const size_t cols, const bool rowwise, const bool colwise,
                     const bool swizzled, const bool use_sr, const size_t *rng_state,
                     const bool use_2d, const bool row_scaled, const bool use_4over6,
                     const float *noop, const float e4m3_max, cudaStream_t stream) {
  // Packed rows must be whole 16-byte TDM units.
  if (use_4over6 || swizzled || is_fp8_dtype(input.dtype) || (rowwise && cols % 32 != 0) ||
      (colwise && rows % 32 != 0) ||
      !dispatch::common::use_tdm(rows, cols, {input.dptr, rowwise ? output.dptr : nullptr,
                                              colwise ? output_t.dptr : nullptr})) {
    return false;
  }
  const dim3 grid(DIVUP(cols, static_cast<size_t>(kChunkDim)),
                  DIVUP(rows, static_cast<size_t>(kChunkDim)));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype, IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_2d, k2D,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              use_sr, kSR,
              const auto launch = [&](auto kernel) {
                kernel<<<grid, kThreads, 0, stream>>>(
                    reinterpret_cast<const IType *>(input.dptr),
                    reinterpret_cast<const float *>(global_amax.dptr),
                    reinterpret_cast<uint8_t *>(output.dptr),
                    reinterpret_cast<uint8_t *>(output_t.dptr),
                    reinterpret_cast<uint8_t *>(scale_inv.dptr),
                    reinterpret_cast<uint8_t *>(scale_inv_t.dptr), rows, cols,
                    rowwise ? scale_inv.shape[1] : 0, colwise ? scale_inv_t.shape[1] : 0,
                    row_scaled, rng_state, noop, e4m3_max);
              };
              if (rowwise && colwise) {
                launch(quantize_kernel<true, true, k2D, kSR, IType>);
              } else if (rowwise) {
                launch(quantize_kernel<true, false, k2D, kSR, IType>);
              } else {
                launch(quantize_kernel<false, true, k2D, kSR, IType>);
              });););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
  return true;
}

}  // namespace nvfp4_tdm
