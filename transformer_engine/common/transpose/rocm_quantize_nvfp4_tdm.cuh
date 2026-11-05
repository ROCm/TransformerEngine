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

// x times encode_scale as NVFP4, the first value in the low nibble of the result. Stochastic
// rounding takes a byte of one draw of rng per value, as the generic kernel does.
template <bool kSR>
__device__ __forceinline__ uint2 encode_block(const float *x, const float encode_scale,
                                              Philox &rng) {
  float y[kBlockDim];
#pragma unroll
  for (int i = 0; i < kBlockDim; i++) {
    y[i] = x[i] * encode_scale;
  }
  if constexpr (kSR) {
    const uint4 rbits = rng.generate4();
    return make_uint2(
        dispatch::common::cvt_8x_to_fp4_sr(y, make_uint2(rbits.x, rbits.y), 1.0f),
        dispatch::common::cvt_8x_to_fp4_sr(y + 8, make_uint2(rbits.z, rbits.w), 1.0f));
  } else {
    return make_uint2(dispatch::common::cvt_8x_to_fp4(y, 1.0f),
                      dispatch::common::cvt_8x_to_fp4(y + 8, 1.0f));
  }
}
#endif  // __gfx1250__

// The tensor-level scales of an encoding, as block_scaled_1d_cast_transpose_kernel derives them.
struct GlobalScales {
  float encode;
  float encode_multiplier;
  float decode;
  float amax;  // what 4over6 dequantizes with
};

__device__ __forceinline__ GlobalScales global_scales(const float *global_amax,
                                                      const float e4m3_max) {
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;
  GlobalScales s;
  s.encode = global_amax == nullptr ? 1.0f : ComputeGlobalEncodeScaleFP4(global_amax[0], e4m3_max);
  s.encode_multiplier = s.encode * (1.0f / fp4_max);
  s.decode = 1.0 / s.encode;
  s.amax = global_amax == nullptr ? fp4_max * e4m3_max : global_amax[0];
  return s;
}

// global_scales of row row of a row-scaled tensor, whose global_amax has an amax per row.
__device__ __forceinline__ GlobalScales row_global_scales(const float *global_amax,
                                                          const uint32_t row, const uint32_t rows,
                                                          const float e4m3_max) {
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;
  GlobalScales s;
  s.encode = row < rows ? ComputeGlobalEncodeScaleFP4(global_amax[row], e4m3_max) : 1.0f;
  s.encode_multiplier = s.encode * (1.0f / fp4_max);
  s.decode = 1.0f / s.encode;
  s.amax = row < rows ? global_amax[row] : fp4_max * e4m3_max;
  return s;
}

// The E4M3 scale of a block of x with amax, and its encode scale. With 4over6, the candidate of
// lower error, from err6 and err4 of the whole 16x16 block with k2D and from x otherwise.
template <bool k4Over6, bool k2D>
__device__ __forceinline__ fp8e4m3 block_scale(const float *x, const float amax,
                                               const GlobalScales &g, const bool use_mse,
                                               const float err_denom, float err6, float err4,
                                               float &encode_scale) {
  if constexpr (!k4Over6) {
    const fp8e4m3 scale_inv = ComputeDecodeScaleFP4<fp8e4m3>(amax, g.encode_multiplier);
    encode_scale = ComputeEncodeScaleFP4<fp8e4m3>(scale_inv, g.decode);
    return scale_inv;
  } else {
    fp8e4m3 sf6, sf4;
    float enc6, enc4;
    if constexpr (k2D) {
      Compute4Over6Scales<fp8e4m3>(amax, g.decode, g.encode, &sf6, &sf4, &enc6, &enc4);
    } else {
      Compute4Over6Candidates<fp8e4m3>(x, kBlockDim, amax, g.decode, use_mse, g.encode, g.amax,
                                       err_denom, &sf6, &sf4, &enc6, &enc4, &err6, &err4);
    }
    // A tie keeps map-to-6, as upstream.
    const bool pick4 = err4 < err6;
    encode_scale = pick4 ? enc4 : enc6;
    return pick4 ? sf4 : sf6;
  }
}

// Quantizes chunk (chunk_Y, chunk_X) of the rows x cols input to the rowwise NVFP4 data, rows x
// cols / 2 bytes, and the columnwise data of its transpose, cols x rows / 2 bytes. With kRHT, the
// columnwise data is that of the random Hadamard transform of the transpose with sign_mask, rounded
// to BF16 as hadamard_transform stores it.
template <bool kRowwise, bool kColwise, bool k2D, bool kSR, bool kRHT, bool k4Over6,
          typename IType>
__device__ __forceinline__ void quantize_chunk(
    const IType *input, const float *global_amax, const float *global_amax_t,
    const uint32_t sign_mask, uint8_t *output_c, uint8_t *output_t, uint8_t *scales_c,
    uint8_t *scales_t, const uint32_t rows, const uint32_t cols, const size_t scale_stride,
    const size_t scale_t_stride, const bool row_scaled, const size_t *rng_state,
    const float e4m3_max, const bool use_mse, const uint32_t chunk_Y, const uint32_t chunk_X) {
#ifdef __gfx1250__
  // 4over6 scales 2D blocks by their error, summed per row (rowwise) or column (colwise) first.
  constexpr bool k2DErr = k4Over6 && k2D;
  __shared__ alignas(128) IType in_sh[kWaves][kBandY][kChunkDim];
  __shared__ alignas(128) uint8_t rowwise_sh[kRowwise ? kWaves : 1][kBandY][kChunkDim / 2];
  __shared__ alignas(8) uint8_t rowwise_scale_sh[kRowwise ? kWaves : 1][kBandY][kBlocks];
  // The transpose of the chunk, so that its rows go out in kChunkDim / 2-byte segments.
  __shared__ alignas(128) uint8_t colwise_sh[kColwise ? kChunkDim : 1][kChunkDim / 2];
  __shared__ alignas(8) uint8_t colwise_scale_sh[kColwise ? kChunkDim : 1][kBlocks];
  __shared__ float block_amax_sh[k2D ? kWaves : 1][kBandBlocks][kBlocks];
  __shared__ float row_err_sh[k2DErr && kRowwise ? kWaves : 1][2][kBandY][kBlocks];
  __shared__ float col_err_sh[k2DErr && kColwise ? kWaves : 1][2][kBandBlocks][kChunkDim];
  // [wave][rowwise, colwise][map-to-6, map-to-4][block row][block column]
  __shared__ float block_err_sh[k2DErr ? kWaves : 1][2][2][kBandBlocks][kBlocks];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const int lane = threadIdx.x % kWaveSize;
  const uint32_t chunk_offset_Y = chunk_Y * kChunkDim;
  const uint32_t chunk_offset_X = chunk_X * kChunkDim;
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

  const GlobalScales g = global_scales(global_amax, e4m3_max);
  const GlobalScales g_t = global_scales(global_amax_t, e4m3_max);
  const float err_denom = TypeExtrema<fp4e2m1>::max * e4m3_max;
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

  if constexpr (k2DErr) {
    if (active) {
      if constexpr (kRowwise) {
        constexpr int kRowsPerStep = kWaveSize / kBlocks;
        const int block = lane % kBlocks;
#pragma unroll
        for (int step = 0; step < kBandY / kRowsPerStep; step++) {
          const int y = step * kRowsPerStep + lane / kBlocks;
          float x[kBlockDim];
          load_floats<kBlockDim>(&in[y][block * kBlockDim], x);
          fp8e4m3 sf6, sf4;
          float enc6, enc4;
          Compute4Over6Candidates<fp8e4m3>(x, kBlockDim, block_amax_sh[wave][y / kBlockDim][block],
                                           g.decode, use_mse, g.encode, g.amax, err_denom, &sf6,
                                           &sf4, &enc6, &enc4, &row_err_sh[wave][0][y][block],
                                           &row_err_sh[wave][1][y][block]);
        }
      }
      if constexpr (kColwise) {
#pragma unroll
        for (int b = 0; b < kBandBlocks; b++) {
#pragma unroll
          for (int i = 0; i < kColsPerLane; i++) {
            const int c = lane * kColsPerLane + i;
            float x[kBlockDim];
#pragma unroll
            for (int y = 0; y < kBlockDim; y++) {
              x[y] = static_cast<float>(in[b * kBlockDim + y][c]);
            }
            fp8e4m3 sf6, sf4;
            float enc6, enc4;
            Compute4Over6Candidates<fp8e4m3>(x, kBlockDim, block_amax_sh[wave][b][c / kBlockDim],
                                             g_t.decode, use_mse, g_t.encode, g_t.amax, err_denom,
                                             &sf6, &sf4, &enc6, &enc4, &col_err_sh[wave][0][b][c],
                                             &col_err_sh[wave][1][b][c]);
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
      // A lane per (candidate, block): upstream's reduction tree over the block's 16 rows or
      // columns.
      const int candidate = lane / (kBandBlocks * kBlocks);
      const int b = lane / kBlocks % kBandBlocks;
      const int block = lane % kBlocks;
      float err[kBlockDim];
      if constexpr (kRowwise) {
#pragma unroll
        for (int r = 0; r < kBlockDim; r++) {
          err[r] = row_err_sh[wave][candidate][b * kBlockDim + r][block];
        }
        block_err_sh[wave][0][candidate][b][block] = reduceBlockError<kBlockDim>(err);
      }
      if constexpr (kColwise) {
#pragma unroll
        for (int c = 0; c < kBlockDim; c++) {
          err[c] = col_err_sh[wave][candidate][b][block * kBlockDim + c];
        }
        block_err_sh[wave][1][candidate][b][block] = reduceBlockError<kBlockDim>(err);
      }
      __builtin_amdgcn_wave_barrier();
    }
  }

  if (kRowwise && active) {
    // Per step, a lane quantizes a block of a row, kBlocks lanes the whole row.
    constexpr int kRowsPerStep = kWaveSize / kBlocks;  // 4
    // 1D 4over6 scores every block here, so a step at a time keeps its code and registers small.
    constexpr bool kOneAtATime = k4Over6 && !k2D;
    constexpr int kUnroll = kOneAtATime ? 1 : kBandY / kRowsPerStep;
    const int block = lane % kBlocks;
    auto &out = rowwise_sh[wave];
    auto &scales = rowwise_scale_sh[wave];
#pragma unroll kUnroll
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
      float err6 = 0.0f;
      float err4 = 0.0f;
      if constexpr (k2DErr) {
        err6 = block_err_sh[wave][0][0][y / kBlockDim][block];
        err4 = block_err_sh[wave][0][1][y / kBlockDim][block];
      }
      float encode_scale;
      const fp8e4m3 scale_inv = block_scale<k4Over6, k2D>(
          x, amax, row_scaled ? row_global_scales(global_amax, row, rows, e4m3_max) : g, use_mse,
          err_denom, err6, err4, encode_scale);
      scales[y][block] = __builtin_bit_cast(uint8_t, scale_inv);
      *reinterpret_cast<uint2 *>(&out[y][block * kBlockDim / 2]) =
          encode_block<kSR>(x, encode_scale, rng);
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
    const auto quantize_column = [&](float *x, const int b, const int c) {
      const int segment = wave * kBandBlocks + b;
      if constexpr (kRHT) {
#pragma unroll
        for (int y = 0; y < kBlockDim; y++) {
          x[y] = (sign_mask >> y) & 1 ? -x[y] : x[y];
        }
        dispatch::common::hadamard16(x);
#pragma unroll
        for (int y = 0; y < kBlockDim; y++) {
          x[y] = static_cast<float>(static_cast<__bf16>(x[y]));
        }
      }
      float amax = 0.0f;
      if constexpr (k2D) {
        amax = block_amax_sh[wave][b][c / kBlockDim];
      } else {
#pragma unroll
        for (int y = 0; y < kBlockDim; y++) {
          amax = fmaxf(amax, fabsf(x[y]));
        }
      }
      float err6 = 0.0f;
      float err4 = 0.0f;
      if constexpr (k2DErr) {
        err6 = block_err_sh[wave][1][0][b][c / kBlockDim];
        err4 = block_err_sh[wave][1][1][b][c / kBlockDim];
      }
      float encode_scale;
      const fp8e4m3 scale_inv =
          block_scale<k4Over6, k2D>(x, amax, g_t, use_mse, err_denom, err6, err4, encode_scale);
      colwise_scale_sh[c][segment] = __builtin_bit_cast(uint8_t, scale_inv);
      *reinterpret_cast<uint2 *>(&colwise_sh[c][segment * kBlockDim / 2]) =
          encode_block<kSR>(x, encode_scale, rng);
    };
    if (active) {
      const int lane_offset_X = lane * kColsPerLane;
#pragma unroll
      for (int b = 0; b < kBandBlocks; b++) {
        if constexpr (k4Over6 && !k2D) {
          // A column at a time, as the rowwise steps.
#pragma unroll 1
          for (int i = 0; i < kColsPerLane; i++) {
            float x[kBlockDim];
#pragma unroll
            for (int y = 0; y < kBlockDim; y++) {
              x[y] = static_cast<float>(in[b * kBlockDim + y][lane_offset_X + i]);
            }
            quantize_column(x, b, lane_offset_X + i);
          }
        } else {
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
            quantize_column(x[i], b, lane_offset_X + i);
          }
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

template <bool kRowwise, bool kColwise, bool k2D, bool kSR, bool kRHT, bool k4Over6,
          typename IType>
__global__ void __launch_bounds__(kThreads)
    quantize_kernel(const IType *input, const float *global_amax, const float *global_amax_t,
                    const uint16_t *rht_matrix, uint8_t *output_c, uint8_t *output_t,
                    uint8_t *scales_c, uint8_t *scales_t, const uint32_t rows, const uint32_t cols,
                    const size_t scale_stride, const size_t scale_t_stride, const bool row_scaled,
                    const size_t *rng_state, const float *noop, const float e4m3_max,
                    const bool use_mse) {
#ifdef __gfx1250__
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  // Row i of the RHT matrix is row i of the Hadamard matrix, whose first entry is positive, times
  // sign i.
  const int lane = threadIdx.x % kWaveSize;
  const uint32_t sign_mask =
      kRHT ? __builtin_amdgcn_ballot_w32(lane < kBlockDim &&
                                         (rht_matrix[lane * kBlockDim] & 0x8000) != 0)
           : 0;
  quantize_chunk<kRowwise, kColwise, k2D, kSR, kRHT, k4Over6>(
      input, global_amax, global_amax_t, sign_mask, output_c, output_t, scales_c, scales_t, rows,
      cols, scale_stride, scale_t_stride, row_scaled, rng_state, e4m3_max, use_mse, blockIdx.y,
      blockIdx.x);
#endif  // __gfx1250__
}

constexpr int kGroupMaxTensors = 256;

// Splits of a tensor along dimension 0, each quantized into its own NVFP4 tensor.
struct GroupArgs {
  const void *input[kGroupMaxTensors];
  const float *amax[kGroupMaxTensors];
  const float *amax_t[kGroupMaxTensors];
  uint8_t *output_c[kGroupMaxTensors];
  uint8_t *output_t[kGroupMaxTensors];
  uint8_t *scales_c[kGroupMaxTensors];
  uint8_t *scales_t[kGroupMaxTensors];
  uint32_t scale_stride[kGroupMaxTensors];
  uint32_t scale_t_stride[kGroupMaxTensors];
  uint32_t rows[kGroupMaxTensors];
  // Split t owns chunk rows [first_chunk_Y[t], first_chunk_Y[t + 1]).
  uint32_t first_chunk_Y[kGroupMaxTensors + 1];
  int num_tensors;
};

// quantize_kernel of each split, with a chunk row of the grid per chunk row of a split.
template <bool kRowwise, bool kColwise, bool kSR, bool kRHT, bool k4Over6, typename IType>
__global__ void __launch_bounds__(kThreads)
    group_quantize_kernel(const GroupArgs args, const uint32_t sign_mask, const uint32_t cols,
                          const size_t *rng_state, const float e4m3_max, const bool use_mse) {
  int t = 0;
  int hi = args.num_tensors - 1;
  while (t < hi) {
    const int mid = t + (hi - t) / 2;
    if (args.first_chunk_Y[mid + 1] <= blockIdx.y) {
      t = mid + 1;
    } else {
      hi = mid;
    }
  }
  quantize_chunk<kRowwise, kColwise, false, kSR, kRHT, k4Over6>(
      static_cast<const IType *>(args.input[t]), args.amax[t], args.amax_t[t], sign_mask,
      args.output_c[t], args.output_t[t], args.scales_c[t], args.scales_t[t], args.rows[t], cols,
      args.scale_stride[t], args.scale_t_stride[t], false, rng_state, e4m3_max, use_mse,
      blockIdx.y - args.first_chunk_Y[t], blockIdx.x);
}

// Whether quantize_kernel takes an NVFP4 quantize: TDM applies, packed rows are whole 16-byte TDM
// units, and it is not one of the modes left to block_scaled_1d_cast_transpose_kernel; swizzled
// scales are requested only on GPUs without TDM, and 4over6 never with stochastic rounding.
inline bool supported(const SimpleTensor &input, const SimpleTensor &output,
                      const SimpleTensor &output_t, const size_t rows, const size_t cols,
                      const bool rowwise, const bool colwise, const bool swizzled,
                      const bool use_4over6, const bool use_sr) {
  return !(use_4over6 && use_sr) && !swizzled && !is_fp8_dtype(input.dtype) &&
         (!rowwise || cols % 32 == 0) && (!colwise || rows % 32 == 0) &&
         dispatch::common::use_tdm(rows, cols, {input.dptr, rowwise ? output.dptr : nullptr,
                                                colwise ? output_t.dptr : nullptr});
}

template <bool k2D, bool kSR, bool kRHT, bool k4Over6, typename IType>
void launch(const SimpleTensor &input, const float *global_amax, const float *global_amax_t,
            const uint16_t *rht_matrix, const SimpleTensor &scale_inv,
            const SimpleTensor &scale_inv_t, const SimpleTensor &output,
            const SimpleTensor &output_t, const size_t rows, const size_t cols, const bool rowwise,
            const bool colwise, const size_t *rng_state, const bool row_scaled, const float *noop,
            const float e4m3_max, const bool use_mse, cudaStream_t stream) {
  const dim3 grid(DIVUP(cols, static_cast<size_t>(kChunkDim)),
                  DIVUP(rows, static_cast<size_t>(kChunkDim)));
  const auto run = [&](auto kernel) {
    kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const IType *>(input.dptr), global_amax, global_amax_t, rht_matrix,
        reinterpret_cast<uint8_t *>(output.dptr), reinterpret_cast<uint8_t *>(output_t.dptr),
        reinterpret_cast<uint8_t *>(scale_inv.dptr), reinterpret_cast<uint8_t *>(scale_inv_t.dptr),
        rows, cols, rowwise ? scale_inv.shape[1] : 0, colwise ? scale_inv_t.shape[1] : 0,
        row_scaled, rng_state, noop, e4m3_max, use_mse);
  };
  if (rowwise && colwise) {
    run(quantize_kernel<true, true, k2D, kSR, kRHT, k4Over6, IType>);
  } else if (rowwise) {
    run(quantize_kernel<true, false, k2D, kSR, false, k4Over6, IType>);
  } else {
    run(quantize_kernel<false, true, k2D, kSR, kRHT, k4Over6, IType>);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// NVFP4 quantize as quantize_transpose_vector_blockwise_fp4 does it. Returns false, leaving the
// work to block_scaled_1d_cast_transpose_kernel, where quantize_kernel does not apply. Stochastic
// rounding draws its own random bits, so it matches that kernel in distribution, not bitwise.
inline bool quantize(const SimpleTensor &input, const SimpleTensor &global_amax,
                     const SimpleTensor &scale_inv, const SimpleTensor &scale_inv_t,
                     const SimpleTensor &output, const SimpleTensor &output_t, const size_t rows,
                     const size_t cols, const bool rowwise, const bool colwise,
                     const bool swizzled, const bool use_sr, const size_t *rng_state,
                     const bool use_2d, const bool row_scaled,
                     const NVTENVFP44Over6Mode mode_4over6, const float *noop,
                     const float e4m3_max, cudaStream_t stream) {
  const bool use_4over6 = mode_4over6 != kNVTENVFP44Over6Disabled;
  if (!supported(input, output, output_t, rows, cols, rowwise, colwise, swizzled, use_4over6,
                 use_sr)) {
    return false;
  }
  const auto *amax = reinterpret_cast<const float *>(global_amax.dptr);
  const bool use_mse = mode_4over6 == kNVTENVFP44Over6MinMSE;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype, IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_2d, k2D,
          if (use_4over6) {
            launch<k2D, false, false, true, IType>(input, amax, amax, nullptr, scale_inv,
                                                   scale_inv_t, output, output_t, rows, cols,
                                                   rowwise, colwise, rng_state, row_scaled, noop,
                                                   e4m3_max, use_mse, stream);
          } else {
            TRANSFORMER_ENGINE_SWITCH_CONDITION(
                use_sr, kSR,
                launch<k2D, kSR, false, false, IType>(input, amax, amax, nullptr, scale_inv,
                                                      scale_inv_t, output, output_t, rows, cols,
                                                      rowwise, colwise, rng_state, row_scaled,
                                                      noop, e4m3_max, false, stream););
          }));
  return true;
}

// Whether quantize_rht applies: BF16 input, 1D scaling, and what quantize_kernel takes.
inline bool supported_rht(const SimpleTensor &input, const SimpleTensor &output,
                          const SimpleTensor &output_t, const size_t rows, const size_t cols,
                          const bool use_2d, const bool swizzled) {
  const bool rowwise = output.dptr != nullptr;
  const bool colwise = output_t.dptr != nullptr;
  return input.dtype == DType::kBFloat16 && !use_2d && (rowwise || colwise) &&
         supported(input, output, output_t, rows, cols, rowwise, colwise, swizzled, false, false);
}

// The rowwise NVFP4 of the BF16 input and the columnwise NVFP4 of the RHT of its transpose, as
// hadamard_transform followed by quantize_transpose_vector_blockwise_fp4 of the BF16 result does
// it, where supported_rht holds.
inline void quantize_rht(const SimpleTensor &input, const SimpleTensor &amax,
                         const SimpleTensor &amax_t, const SimpleTensor &rht_matrix,
                         const SimpleTensor &scale_inv, const SimpleTensor &scale_inv_t,
                         const SimpleTensor &output, const SimpleTensor &output_t,
                         const size_t rows, const size_t cols, const bool use_sr,
                         const size_t *rng_state, const float *noop, const float e4m3_max,
                         cudaStream_t stream) {
  const bool rowwise = output.dptr != nullptr;
  const bool colwise = output_t.dptr != nullptr;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_sr, kSR,
      launch<false, kSR, true, false, bf16>(input, reinterpret_cast<const float *>(amax.dptr),
                                            reinterpret_cast<const float *>(amax_t.dptr),
                                            reinterpret_cast<const uint16_t *>(rht_matrix.dptr),
                                            scale_inv, scale_inv_t, output, output_t, rows,
                                            cols, rowwise, colwise, rng_state, false, noop,
                                            e4m3_max, false, stream););
}

template <bool kSR, bool kRHT, bool k4Over6>
void launch_group(const GroupArgs &args, const uint32_t sign_mask, const size_t cols,
                  const bool rowwise, const bool colwise, const size_t *rng_state,
                  const float e4m3_max, const bool use_mse, cudaStream_t stream) {
  const dim3 grid(DIVUP(cols, static_cast<size_t>(kChunkDim)),
                  args.first_chunk_Y[args.num_tensors]);
  const auto run = [&](auto kernel) {
    kernel<<<grid, kThreads, 0, stream>>>(args, sign_mask, cols, rng_state, e4m3_max, use_mse);
  };
  if (rowwise && colwise) {
    run(group_quantize_kernel<true, true, kSR, kRHT, k4Over6, bf16>);
  } else if (rowwise) {
    run(group_quantize_kernel<true, false, kSR, false, k4Over6, bf16>);
  } else {
    run(group_quantize_kernel<false, true, kSR, kRHT, k4Over6, bf16>);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// The NVFP4 of each split of the BF16 input along its rows, amaxes included, as quantizing the
// split on its own does it, in one amax and one quantize launch; with RHT, the columnwise outputs
// are of the RHT of the transposes with sign_mask, as quantize_rht does them. Returns false,
// launching nothing, for 2D or row-scaled NVFP4, swizzled scales, splits that quantize_kernel does
// not take, or outputs that differ in directions or E4M3 bound. Empty splits keep their amaxes.
inline bool group_quantize(const Tensor &input, const std::vector<Tensor *> &outputs,
                           const size_t *split_sections, const bool with_rht,
                           const uint16_t sign_mask, const QuantizationConfig &config,
                           cudaStream_t stream) {
  using namespace dispatch::common::tdm_amax_kernel;
  const int num_tensors = outputs.size();
  const bool use_4over6 = config.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
  const bool use_sr = config.stochastic_rounding;
  if (num_tensors == 0 || num_tensors > kGroupMaxTensors || input.dtype() != DType::kBFloat16 ||
      config.nvfp4_2d_quantization || config.noop_tensor != nullptr ||
      (use_4over6 && (use_sr || with_rht))) {
    return false;
  }
  const auto [total_rows, cols] = input.flat_2d_dims();
  GroupArgs args{};
  SplitAmaxArgs<kGroupMaxTensors> amax_args{};
  args.num_tensors = num_tensors;
  bool rowwise = false;
  bool colwise = false;
  float e4m3_max = 0;
  size_t row = 0;
  uint32_t chunk_rows = 0;
  for (int t = 0; t < num_tensors; t++) {
    const size_t rows = split_sections[t];
    const Tensor &output = *outputs[t];
    const void *split = static_cast<const bf16 *>(input.data.dptr) + row * cols;
    row += rows;
    args.first_chunk_Y[t] = chunk_rows;
    chunk_rows += DIVUP(rows, static_cast<size_t>(kChunkDim));
    args.rows[t] = amax_args.rows[t] = rows;
    args.input[t] = amax_args.input[t] = split;
    if (rows == 0) {
      amax_args.amax[t] = amax_args.amax_t[t] = nullptr;
      continue;
    }
    NVTE_CHECK(output.scaling_mode == NVTE_NVFP4_1D_SCALING, "Expected NVFP4 outputs.");
    const float output_e4m3_max =
        te_fp8_fnuz() ? 240.0f : static_cast<float>(output.nvfp4_e4m3_max);
    if (e4m3_max == 0) {
      rowwise = output.has_data();
      colwise = output.has_columnwise_data();
      e4m3_max = output_e4m3_max;
    }
    if (output.has_data() != rowwise || output.has_columnwise_data() != colwise ||
        output_e4m3_max != e4m3_max || output.row_scaled_nvfp4 ||
        !supported(SimpleTensor(const_cast<void *>(split), {rows, cols}, DType::kBFloat16),
                   output.data, output.columnwise_data, rows, cols, rowwise, colwise,
                   output.with_gemm_swizzled_scales, use_4over6, use_sr) ||
        (with_rht && rows % kBlockDim != 0)) {
      return false;
    }
    auto *const amax = reinterpret_cast<float *>(output.amax.dptr);
    auto *const amax_t = reinterpret_cast<float *>(output.columnwise_amax.dptr);
    NVTE_CHECK((amax != nullptr || !rowwise) && (amax_t != nullptr || !colwise),
               "NVFP4 outputs need the amaxes of their directions.");
    amax_args.amax[t] = amax;
    amax_args.amax_t[t] = amax_t;
    // Without RHT, both directions scale by the amax quantize_transpose_vector_blockwise_fp4 takes.
    args.amax[t] = amax != nullptr ? amax : amax_t;
    args.amax_t[t] = with_rht ? amax_t : args.amax[t];
    args.output_c[t] = reinterpret_cast<uint8_t *>(output.data.dptr);
    args.output_t[t] = reinterpret_cast<uint8_t *>(output.columnwise_data.dptr);
    args.scales_c[t] = reinterpret_cast<uint8_t *>(output.scale_inv.dptr);
    args.scales_t[t] = reinterpret_cast<uint8_t *>(output.columnwise_scale_inv.dptr);
    args.scale_stride[t] = rowwise ? output.scale_inv.shape[1] : 0;
    args.scale_t_stride[t] = colwise ? output.columnwise_scale_inv.shape[1] : 0;
  }
  NVTE_CHECK(row == total_rows, "Split sections sum to ", row, " rows, but the input has ",
             total_rows, ".");
  args.first_chunk_Y[num_tensors] = chunk_rows;

  const size_t *rng_state = nullptr;
  if (use_sr) {
    NVTE_CHECK(config.rng_state != nullptr, "Stochastic rounding needs an RNG state.");
    const Tensor &rng_state_tensor = *convertNVTETensorCheck(config.rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64 && rng_state_tensor.numel() == 2,
               "RNG state should contain 2 64-bit values.");
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }
  if (with_rht) {
    launch_split_amax_tdm<true>(amax_args, num_tensors, cols, sign_mask, true, stream);
  } else {
    launch_split_amax_tdm<false>(amax_args, num_tensors, cols, 0, true, stream);
  }
  if (chunk_rows == 0) {
    return true;
  }
  const bool use_mse = config.nvfp4_4over6_mode == kNVTENVFP44Over6MinMSE;
  if (use_4over6) {
    launch_group<false, false, true>(args, 0, cols, rowwise, colwise, nullptr, e4m3_max, use_mse,
                                     stream);
  } else {
    TRANSFORMER_ENGINE_SWITCH_CONDITION(
        use_sr, kSR,
        TRANSFORMER_ENGINE_SWITCH_CONDITION(
            with_rht, kRHT,
            launch_group<kSR, kRHT, false>(args, sign_mask, cols, rowwise, colwise, rng_state,
                                           e4m3_max, false, stream);););
  }
  return true;
}

}  // namespace nvfp4_tdm
