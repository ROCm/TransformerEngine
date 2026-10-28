/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 MXFP8 quantize with the Tensor Data Mover: each wave moves its own BAND_Y-row band of a
// 128x128 chunk global->LDS->global and derives both scalings from LDS, so the waves only
// synchronize to combine dbias. The launchers at the end return false where TDM does not apply,
// leaving the work to the generic kernels. Like the other rocm_ headers, this expects its includer
// to have included the common headers.

#include "../../util/rocm_device_utils.cuh"
#include "../../util/rocm_tdm.cuh"
#include "../core/rocm_tdm_common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace tdm_kernel {

using common::find_group_chunk;
using common::find_tensor;
using common::reduce_dbias_tdm;
using common::use_tdm;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t WAVE_SIZE = 32;
constexpr size_t WAVES = THREADS_PER_CHUNK / WAVE_SIZE;    // 4
constexpr size_t BAND_Y = CHUNK_DIM_Y / WAVES;             // 32
constexpr size_t COLS_PER_LANE = CHUNK_DIM_X / WAVE_SIZE;  // 4
constexpr size_t TILE_2D = 32;  // 2D scaling shares a scale per TILE_2D x TILE_2D tile
constexpr size_t MULTI_MAX_TENSORS = 1024;
static_assert(CHUNK_DIM_Y == common::tdm_dbias_kernel::CHUNK_DIM_Y,
              "A dbias partial row covers a chunk");

template <typename T>
using band_t = T[BAND_Y][CHUNK_DIM_X];

#ifdef __gfx1250__
using common::cvt_from_float;

// The e8m0 scale rounds up, so scaled finite values stay within the FP8 range and need no clamp:
// this matches static_cast<OType> bit for bit, including Inf and NaN, at two values per instruction.
template <typename OType>
__device__ __forceinline__ uint32_t cvt_4x_to_fp8(const float x0, const float x1, const float x2,
                                                  const float x3) {
  int packed = 0;
  if constexpr (std::is_same_v<OType, transformer_engine::fp8e4m3>) {
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(x0, x1, packed, false);
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(x2, x3, packed, true);
  } else {
    packed = __builtin_amdgcn_cvt_pk_bf8_f32(x0, x1, packed, false);
    packed = __builtin_amdgcn_cvt_pk_bf8_f32(x2, x3, packed, true);
  }
  return static_cast<uint32_t>(packed);
}

// With 2D scaling, byte t of tile_exponents is the biased exponent of the band's tile t; this
// returns the one of band column x.
__device__ __forceinline__ e8m0_t tile_exponent(const uint32_t tile_exponents, const int x) {
  return static_cast<e8m0_t>(tile_exponents >> (x / static_cast<int>(TILE_2D) * 8));
}

// Rowwise MXFP8 of a band in LDS: per step, 8 lanes cover a row with 16 elements each and a lane
// pair shares a scale, or with IS_2D the one of its tile. Writes the scales of the first valid_rows
// rows and valid_scales scale columns, and returns the lane's amax, which IS_2D leaves at 0.
template <typename IType, typename OType, size_t SCALE_DIM_X, bool IS_2D = false>
__device__ __forceinline__ float quantize_band_rowwise(const band_t<IType> &in, band_t<OType> &out,
                                                       e8m0_t *const scales,
                                                       const size_t scale_stride,
                                                       const size_t valid_rows,
                                                       const size_t valid_scales,
                                                       const uint32_t tile_exponents = 0) {
  constexpr size_t ELEMS = 16;
  constexpr size_t LANES_PER_ROW = CHUNK_DIM_X / ELEMS;        // 8
  constexpr size_t ROWS_PER_STEP = WAVE_SIZE / LANES_PER_ROW;  // 4
  constexpr size_t LANES_PER_SCALE = SCALE_DIM_X / ELEMS;      // 2
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_offset_Y = lane / LANES_PER_ROW;
  const int lane_offset_X = (lane % LANES_PER_ROW) * ELEMS;
  const size_t scale_offset_X = lane_offset_X / SCALE_DIM_X;
  float amax = 0;
#pragma unroll
  for (int step = 0; step < BAND_Y / ROWS_PER_STEP; step++) {
    const size_t y = step * ROWS_PER_STEP + lane_offset_Y;
    Vec<IType, ELEMS> x;
    x.load_from(&in[y][lane_offset_X]);
    e8m0_t biased_exponent;
    if constexpr (IS_2D) {
      biased_exponent = tile_exponent(tile_exponents, lane_offset_X);
    } else {
      float thread_amax = 0;
#pragma unroll
      for (int j = 0; j < ELEMS; j++) {
        thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(x.data.elt[j])));
      }
      amax = fmaxf(amax, thread_amax);
      const float scale_amax = rocm_subwarp_allreduce<LANES_PER_SCALE>(thread_amax, rocm_op::max{});
      biased_exponent = ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
    }
    if (lane % LANES_PER_SCALE == 0 && y < valid_rows && scale_offset_X < valid_scales) {
      scales[y * scale_stride + scale_offset_X] = biased_exponent;
    }
    const float block_scale_inverse = ptx::exp2f_rcp<float>(biased_exponent);
    uint32_t packed[ELEMS / 4];
#pragma unroll
    for (int p = 0; p < ELEMS / 4; p++) {
      packed[p] = cvt_4x_to_fp8<OType>(static_cast<float>(x.data.elt[4 * p]) * block_scale_inverse,
                                       static_cast<float>(x.data.elt[4 * p + 1]) * block_scale_inverse,
                                       static_cast<float>(x.data.elt[4 * p + 2]) * block_scale_inverse,
                                       static_cast<float>(x.data.elt[4 * p + 3]) * block_scale_inverse);
    }
    Vec<OType, ELEMS> o;
    memcpy(&o.data, packed, sizeof(packed));
    o.store_to(&out[y][lane_offset_X]);
  }
  return amax;
}

// Raises col_amax to the amaxes of the lane's COLS_PER_LANE adjacent columns of a band in LDS.
template <typename IType>
__device__ __forceinline__ void band_column_amaxes(const band_t<IType> &in,
                                                   float (&col_amax)[COLS_PER_LANE]) {
  const int lane_offset_X = (threadIdx.x % WAVE_SIZE) * COLS_PER_LANE;
#pragma unroll
  for (int y = 0; y < BAND_Y; y++) {
    Vec<IType, COLS_PER_LANE> x;
    x.load_from(&in[y][lane_offset_X]);
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      col_amax[i] = fmaxf(col_amax[i], fabsf(static_cast<float>(x.data.elt[i])));
    }
  }
}

// Colwise MXFP8 of a band in LDS: each lane owns COLS_PER_LANE adjacent columns, with one pass for
// the amaxes and one to convert, or with IS_2D only the latter, with the scales of the tiles.
// Writes the scales of the first valid_cols columns and returns the lane's amax, which IS_2D leaves
// at 0.
template <typename IType, typename OType, bool IS_2D = false>
__device__ __forceinline__ float quantize_band_colwise(const band_t<IType> &in, band_t<OType> &out,
                                                       e8m0_t *const scales,
                                                       const size_t valid_cols,
                                                       const uint32_t tile_exponents = 0) {
  const int lane_offset_X = (threadIdx.x % WAVE_SIZE) * COLS_PER_LANE;
  float col_amax[COLS_PER_LANE] = {};
  if constexpr (!IS_2D) {
    band_column_amaxes(in, col_amax);
  }
  float amax = 0;
  float block_scale_inverse[COLS_PER_LANE];
#pragma unroll
  for (int i = 0; i < COLS_PER_LANE; i++) {
    amax = fmaxf(amax, col_amax[i]);
    const e8m0_t biased_exponent =
        IS_2D ? tile_exponent(tile_exponents, lane_offset_X)
              : ptx::float_to_e8m0(col_amax[i] * Quantized_Limits<OType>::max_norm_rcp);
    if (static_cast<size_t>(lane_offset_X + i) < valid_cols) {
      scales[lane_offset_X + i] = biased_exponent;
    }
    block_scale_inverse[i] = ptx::exp2f_rcp<float>(biased_exponent);
  }
#pragma unroll
  for (int y = 0; y < BAND_Y; y++) {
    Vec<IType, COLS_PER_LANE> x;
    x.load_from(&in[y][lane_offset_X]);
    Vec<OType, COLS_PER_LANE> o;
    o.data.vec = cvt_4x_to_fp8<OType>(static_cast<float>(x.data.elt[0]) * block_scale_inverse[0],
                                      static_cast<float>(x.data.elt[1]) * block_scale_inverse[1],
                                      static_cast<float>(x.data.elt[2]) * block_scale_inverse[2],
                                      static_cast<float>(x.data.elt[3]) * block_scale_inverse[3]);
    o.store_to(&out[y][lane_offset_X]);
  }
  return amax;
}
#endif  // __gfx1250__

// Quantizes chunk (chunk_Y, chunk_X) of a rows x cols tensor and writes its dbias partials to row
// dbias_row of dbias_workspace. IS_2D scales TILE_2D x TILE_2D tiles instead of rows and columns.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_2D>
__device__ __forceinline__ void quantize_chunk(
    const IType *input_ptr, const IType *act_input_ptr, OType *output_rowwise,
    OType *output_colwise, e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
    float *dbias_workspace, float *amax_ptr, const size_t rows, const size_t cols,
    const size_t scale_stride_rowwise, const size_t scale_stride_colwise, const size_t chunk_Y,
    const size_t chunk_X, const size_t dbias_row) {
#ifdef __gfx1250__
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_ACTIVATIONS = IS_ACT || IS_DACT;
  constexpr size_t OUT_BAND_BYTES = sizeof(band_t<OType>);
  constexpr size_t OUT_BYTES = (USE_ROWWISE_SCALING + USE_COLWISE_SCALING) * OUT_BAND_BYTES;
  constexpr size_t ACT_BYTES = IS_DACT ? sizeof(band_t<IType>) : 0;
  constexpr size_t AUX_BYTES = std::max({OUT_BYTES, ACT_BYTES, size_t{1}});
  static_assert(!USE_COLWISE_SCALING || SCALE_DIM_Y == BAND_Y,
                "A band must be one colwise scale block");
  static_assert(!IS_2D || TILE_2D == BAND_Y, "A band must be one row of tiles");

  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  // Stages the activation input of DACT, then holds the outputs once in_sh has the activations.
  __shared__ alignas(128) unsigned char aux_sh[WAVES][AUX_BYTES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_offset_X = lane * COLS_PER_LANE;
  const size_t band_offset_Y = chunk_Y * CHUNK_DIM_Y + wave * BAND_Y;
  const size_t chunk_offset_X = chunk_X * CHUNK_DIM_X;
  auto &in = in_sh[wave];
  auto &act_in = *reinterpret_cast<band_t<IType> *>(aux_sh[wave]);
  auto &out_rowwise = *reinterpret_cast<band_t<OType> *>(aux_sh[wave]);
  auto &out_colwise =
      *reinterpret_cast<band_t<OType> *>(aux_sh[wave] + (USE_ROWWISE_SCALING ? OUT_BAND_BYTES : 0));
  // Out-of-bounds elements are zero in LDS and every activation (or derivative times a zero
  // gradient) keeps them zero, so they leave the amaxes and dbias unchanged.
  float amax = 0;
  float partial_dbias[COLS_PER_LANE] = {};

  if (band_offset_Y < rows) {
    tdm::load_2d(in, input_ptr, rows, cols, cols, band_offset_Y, chunk_offset_X);
    if constexpr (IS_DACT) {
      tdm::load_2d(act_in, act_input_ptr, rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
    tdm::wait<0>();

    if constexpr (COMPUTE_ACTIVATIONS || IS_DBIAS) {
      // Activations in place; a lane owns COLS_PER_LANE columns, so it also sums their dbias.
#pragma unroll
      for (int y = 0; y < BAND_Y; y++) {
        Vec<IType, COLS_PER_LANE> x;
        Vec<IType, COLS_PER_LANE> a;
        x.load_from(&in[y][lane_offset_X]);
        if constexpr (IS_DACT) {
          a.load_from(&act_in[y][lane_offset_X]);
        }
#pragma unroll
        for (int i = 0; i < COLS_PER_LANE; i++) {
          float elt = static_cast<float>(x.data.elt[i]);
          if constexpr (IS_ACT) {
            elt = OP(elt, {});
          }
          if constexpr (IS_DACT) {
            elt *= OP(static_cast<float>(a.data.elt[i]), {});
          }
          if constexpr (IS_DBIAS) {
            partial_dbias[i] += elt;
          }
          if constexpr (COMPUTE_ACTIVATIONS) {
            x.data.elt[i] = cvt_from_float<IType>(elt);
          }
        }
        if constexpr (COMPUTE_ACTIVATIONS) {
          x.store_to(&in[y][lane_offset_X]);
        }
      }
      if constexpr (IS_DACT) {
        asm volatile("" ::: "memory");  // the outputs overwrite act_in through another type
      }
    }

    // The amax of a tile scales all its elements; in the colwise layout, its lanes are adjacent.
    uint32_t tile_exponents = 0;
    if constexpr (IS_2D) {
      constexpr int TILE_LANES = TILE_2D / COLS_PER_LANE;  // 8
      float col_amax[COLS_PER_LANE] = {};
      band_column_amaxes(in, col_amax);
      float lane_amax = 0;
#pragma unroll
      for (int i = 0; i < COLS_PER_LANE; i++) {
        lane_amax = fmaxf(lane_amax, col_amax[i]);
      }
      amax = fmaxf(amax, lane_amax);
      const int biased_exponent =
          ptx::float_to_e8m0(rocm_subwarp_allreduce<TILE_LANES>(lane_amax, rocm_op::max{}) *
                             Quantized_Limits<OType>::max_norm_rcp);
#pragma unroll
      for (int t = 0; t < CHUNK_DIM_X / TILE_2D; t++) {
        tile_exponents |=
            static_cast<uint32_t>(__builtin_amdgcn_readlane(biased_exponent, t * TILE_LANES))
            << (8 * t);
      }
    }

    if constexpr (USE_ROWWISE_SCALING) {
      const size_t scale_offset_X = chunk_offset_X / SCALE_DIM_X;
      // Like the 128-wide chunks of the generic kernel, this also fills the scale padding.
      amax = fmaxf(amax, quantize_band_rowwise<IType, OType, SCALE_DIM_X, IS_2D>(
                             in, out_rowwise,
                             scales_rowwise + band_offset_Y * scale_stride_rowwise + scale_offset_X,
                             scale_stride_rowwise, rows - band_offset_Y,
                             scale_stride_rowwise - scale_offset_X, tile_exponents));
      tdm::store_2d(output_rowwise, out_rowwise, rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
    if constexpr (USE_COLWISE_SCALING) {
      amax = fmaxf(amax, quantize_band_colwise<IType, OType, IS_2D>(
                             in, out_colwise,
                             scales_colwise + band_offset_Y / SCALE_DIM_Y * scale_stride_colwise +
                                 chunk_offset_X,
                             cols - chunk_offset_X, tile_exponents));
      tdm::store_2d(output_colwise, out_colwise, rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
  }

  if constexpr (IS_DBIAS) {
    // Every wave leaves its column sums in its consumed in_sh band.
    asm volatile("" ::: "memory");  // in_sh is reused as float
    Vec<float, COLS_PER_LANE> partial;
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      partial.data.elt[i] = partial_dbias[i];
    }
    partial.store_to(reinterpret_cast<float *>(in_sh[wave]) + lane_offset_X);
    __syncthreads();
    const size_t col = chunk_offset_X + threadIdx.x;
    if (col < cols) {
      float dbias = 0;
#pragma unroll
      for (int w = 0; w < WAVES; w++) {
        dbias += reinterpret_cast<const float *>(in_sh[w])[threadIdx.x];
      }
      dbias_workspace[dbias_row * cols + col] = dbias;
    }
  }

  if (amax_ptr != nullptr) {
    const float wave_amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
    // amax only grows, so a stale read can only cause a redundant atomic.
    if (lane == 0 && wave_amax > *amax_ptr) {
      atomicMaxFloat(amax_ptr, wave_amax);
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_2D>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_mxfp8_tdm_kernel(const IType *input, const IType *act_input, OType *output_rowwise,
                              OType *output_colwise, e8m0_t *scales_rowwise,
                              e8m0_t *scales_colwise, const float *noop, float *dbias_workspace,
                              float *amax, const size_t rows, const size_t cols,
                              const size_t scale_stride_rowwise,
                              const size_t scale_stride_colwise) {
  if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
  quantize_chunk<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, SCALE_DIM_Y, SCALE_DIM_X,
                 IS_2D>(input, act_input, output_rowwise, output_colwise, scales_rowwise,
                        scales_colwise, dbias_workspace, amax, rows, cols, scale_stride_rowwise,
                        scale_stride_colwise, blockIdx.y, blockIdx.x, blockIdx.y);
}

// With a common last dimension, the tensors each have a multiple of CHUNK_DIM_Y rows, so no chunk
// of the 2D grid straddles two of them, and the dbias partials of chunk row blockIdx.y go to that
// workspace row. Otherwise, the 1D grid has a block per chunk of find_group_chunk and no dbias.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_2D>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    grouped_quantize_mxfp8_tdm_kernel(const IType *input, const IType *act_input,
                                      OType *output_rowwise, OType *output_colwise,
                                      e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
                                      const float *noop, float *dbias_workspace, float *amax,
                                      const size_t first_logical_dim, const size_t last_logical_dim,
                                      const size_t num_tensors, const int64_t *offsets_ptr,
                                      const int64_t *first_dims_ptr,
                                      const int64_t *last_dims_ptr) {
  if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
  size_t tensor_offset;
  size_t rows;
  size_t cols;
  size_t chunk_Y;
  size_t chunk_X;
  size_t scales_offset_rowwise;
  size_t scales_offset_colwise;
  if (last_dims_ptr == nullptr) {
    cols = last_logical_dim;
    const size_t chunk_offset_Y = static_cast<size_t>(blockIdx.y) * CHUNK_DIM_Y;
    size_t tensor_id;
    if (offsets_ptr == nullptr) {
      const size_t rows_per_tensor = first_logical_dim / num_tensors;
      tensor_id = chunk_offset_Y / rows_per_tensor;
      tensor_offset = tensor_id * rows_per_tensor * cols;
    } else {
      tensor_id = find_tensor(offsets_ptr, num_tensors, chunk_offset_Y * cols);
      tensor_offset = static_cast<size_t>(offsets_ptr[tensor_id]);
    }
    rows = first_dims_ptr != nullptr ? static_cast<size_t>(first_dims_ptr[tensor_id])
                                     : first_logical_dim / num_tensors;
    const size_t tensor_offset_Y = tensor_offset / cols;
    chunk_Y = blockIdx.y - tensor_offset_Y / CHUNK_DIM_Y;
    chunk_X = blockIdx.x;
    scales_offset_rowwise = tensor_offset_Y * DIVUP(cols, SCALE_DIM_X);
    scales_offset_colwise = tensor_offset_Y / SCALE_DIM_Y * cols;
  } else {
    const auto chunk = find_group_chunk<CHUNK_DIM_Y, CHUNK_DIM_X>(
        blockIdx.x, num_tensors, first_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr);
    tensor_offset = chunk.offset;
    rows = chunk.rows;
    cols = chunk.cols;
    chunk_Y = chunk.chunk_Y;
    chunk_X = chunk.chunk_X;
    // Both dimensions are divisible by the scale blocks.
    scales_offset_rowwise = tensor_offset / SCALE_DIM_X;
    scales_offset_colwise = tensor_offset / SCALE_DIM_Y;
  }
  if (chunk_Y * CHUNK_DIM_Y >= rows) {
    return;
  }
  constexpr bool ROWWISE = SCALE_DIM_X > 1;
  constexpr bool COLWISE = SCALE_DIM_Y > 1;
  quantize_chunk<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, SCALE_DIM_Y, SCALE_DIM_X,
                 IS_2D>(input + tensor_offset, IS_DACT ? act_input + tensor_offset : nullptr,
                        ROWWISE ? output_rowwise + tensor_offset : nullptr,
                        COLWISE ? output_colwise + tensor_offset : nullptr,
                        ROWWISE ? scales_rowwise + scales_offset_rowwise : nullptr,
                        COLWISE ? scales_colwise + scales_offset_colwise : nullptr,
                        dbias_workspace, amax, rows, cols, DIVUP(cols, SCALE_DIM_X), cols, chunk_Y,
                        chunk_X, blockIdx.y);
}

struct MultiQuantizeArgs {
  const void *input[MULTI_MAX_TENSORS];
  void *output_rowwise[MULTI_MAX_TENSORS];
  void *output_colwise[MULTI_MAX_TENSORS];
  void *scales_rowwise[MULTI_MAX_TENSORS];
  void *scales_colwise[MULTI_MAX_TENSORS];
  float *amax[MULTI_MAX_TENSORS];
  uint32_t rows[MULTI_MAX_TENSORS];
  uint32_t cols[MULTI_MAX_TENSORS];
  // Tensor t owns chunk rows [first_chunk_Y[t], first_chunk_Y[t + 1]).
  uint32_t first_chunk_Y[MULTI_MAX_TENSORS + 1];
  int num_tensors;
};

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    multi_quantize_mxfp8_tdm_kernel(const MultiQuantizeArgs args) {
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
  const size_t rows = args.rows[t];
  const size_t cols = args.cols[t];
  const size_t chunk_Y = blockIdx.y - args.first_chunk_Y[t];
  if (chunk_Y * CHUNK_DIM_Y >= rows || blockIdx.x * CHUNK_DIM_X >= cols) {
    return;
  }
  quantize_chunk<false, false, false, Empty, nullptr, IType, OType, SCALE_DIM_Y, SCALE_DIM_X,
                 false>(
      static_cast<const IType *>(args.input[t]), nullptr,
      static_cast<OType *>(args.output_rowwise[t]), static_cast<OType *>(args.output_colwise[t]),
      static_cast<e8m0_t *>(args.scales_rowwise[t]), static_cast<e8m0_t *>(args.scales_colwise[t]),
      nullptr, args.amax[t], rows, cols, DIVUP(cols, SCALE_DIM_X), cols, chunk_Y, blockIdx.x, 0);
}

}  // namespace tdm_kernel

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
bool quantize_tdm(const Tensor &input, const Tensor *act_input, const Tensor *noop, Tensor *output,
                  Tensor *dbias, Tensor *workspace, const bool use_2d_quantization,
                  hipStream_t stream) {
  using namespace tdm_kernel;
  const bool rowwise = output->has_data();
  const bool colwise = output->has_columnwise_data();
  const auto [rows, cols] = input.flat_2d_dims();
  // The dbias workspace query takes the generic path, whose workspace has at least as many rows.
  if (!(rowwise || colwise) ||
      (IS_DBIAS && (workspace == nullptr || workspace->data.dptr == nullptr)) ||
      !use_tdm(rows, cols,
               {input.data.dptr, IS_DACT ? act_input->data.dptr : nullptr, output->data.dptr,
                output->columnwise_data.dptr, IS_DBIAS ? workspace->data.dptr : nullptr})) {
    return false;
  }
  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
  }
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
              (colwise ? 32 : 1), SCALE_DIM_Y,
              TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                  (rowwise ? 32 : 1), SCALE_DIM_X,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      use_2d_quantization, IS_2D,
                      quantize_mxfp8_tdm_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType,
                                                OType, SCALE_DIM_Y, SCALE_DIM_X, IS_2D>
                      <<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                          reinterpret_cast<const IType *>(input.data.dptr),
                          IS_DACT ? reinterpret_cast<const IType *>(act_input->data.dptr)
                                  : nullptr,
                          reinterpret_cast<OType *>(output->data.dptr),
                          reinterpret_cast<OType *>(output->columnwise_data.dptr),
                          reinterpret_cast<e8m0_t *>(output->scale_inv.dptr),
                          reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr),
                          reinterpret_cast<const float *>(noop->data.dptr), workspace_ptr,
                          reinterpret_cast<float *>(output->amax.dptr), rows, cols,
                          rowwise ? output->scale_inv.shape[1] : 1,
                          colwise ? output->columnwise_scale_inv.shape[1] : 1);)))));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  if constexpr (IS_DBIAS) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input.dtype(), IType,
        reduce_dbias_tdm<IType>(reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, cols,
                                1, grid.y, nullptr, nullptr, stream););
  }
  return true;
}

// Called after the dbias workspace query: the generic chunks it sized for are not taller than TDM's.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
bool group_quantize_tdm(const GroupedTensor *input, const GroupedTensor *activations,
                        const Tensor *noop, GroupedTensor *output, GroupedTensor *dbias,
                        Tensor *workspace, const bool use_2d_quantization, hipStream_t stream) {
  using namespace tdm_kernel;
  const bool rowwise = output->has_data();
  const bool colwise = output->has_columnwise_data();
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t num_tensors = input->num_tensors;
  const bool same_shape = output->all_same_shape();
  const bool same_last_dim = output->all_same_last_dim();
  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  // Varying dimensions are device-side and, as upstream requires, divisible by 128. Upstream also
  // has no dbias for varying last dimensions.
  const size_t tensor_rows = same_shape ? first_logical_dim / num_tensors : first_logical_dim;
  if ((!output->varying_both_dims() && tensor_rows % CHUNK_DIM_Y != 0) ||
      (!same_last_dim && (IS_DBIAS || offsets_ptr == nullptr)) ||
      !use_tdm(first_logical_dim, last_logical_dim,
               {input->data.dptr, IS_DACT ? activations->data.dptr : nullptr,
                rowwise ? output->data.dptr : nullptr,
                colwise ? output->columnwise_data.dptr : nullptr,
                IS_DBIAS ? workspace->data.dptr : nullptr})) {
    return false;
  }
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  const dim3 grid = same_last_dim ? dim3(DIVUP(last_logical_dim, CHUNK_DIM_X),
                                         DIVUP(first_logical_dim, CHUNK_DIM_Y))
                                  : dim3(DIVUP(first_logical_dim * last_logical_dim,
                                               CHUNK_DIM_Y * CHUNK_DIM_X));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
              (colwise ? 32 : 1), SCALE_DIM_Y,
              TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                  (rowwise ? 32 : 1), SCALE_DIM_X,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      use_2d_quantization, IS_2D,
                      grouped_quantize_mxfp8_tdm_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP,
                                                        IType, OType, SCALE_DIM_Y, SCALE_DIM_X,
                                                        IS_2D>
                      <<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                          reinterpret_cast<const IType *>(input->data.dptr),
                          IS_DACT ? reinterpret_cast<const IType *>(activations->data.dptr)
                                  : nullptr,
                          reinterpret_cast<OType *>(output->data.dptr),
                          reinterpret_cast<OType *>(output->columnwise_data.dptr),
                          reinterpret_cast<e8m0_t *>(output->scale_inv.dptr),
                          reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr),
                          reinterpret_cast<const float *>(noop->data.dptr), workspace_ptr,
                          reinterpret_cast<float *>(output->amax.dptr), first_logical_dim,
                          last_logical_dim, num_tensors, offsets_ptr, first_dims_ptr,
                          last_dims_ptr);)))));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  if constexpr (IS_DBIAS) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input->dtype(), IType,
        reduce_dbias_tdm<IType>(reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr,
                                last_logical_dim, num_tensors, grid.y,
                                same_shape ? nullptr : offsets_ptr,
                                same_shape ? nullptr : first_dims_ptr, stream););
  }
  return true;
}

// TDM only if every tensor qualifies, as one launch covers them all.
inline bool multi_quantize_tdm(const std::vector<Tensor *> &input_list,
                               std::vector<Tensor *> &output_list, hipStream_t stream) {
  using namespace tdm_kernel;
  const size_t num_tensors = input_list.size();
  const bool rowwise = output_list[0]->has_data();
  const bool colwise = output_list[0]->has_columnwise_data();
  if (num_tensors > MULTI_MAX_TENSORS) {
    return false;
  }
  for (size_t i = 0; i < num_tensors; i++) {
    if (!use_tdm(input_list[i]->flat_first_dim(), input_list[i]->flat_last_dim(),
                 {input_list[i]->data.dptr, rowwise ? output_list[i]->data.dptr : nullptr,
                  colwise ? output_list[i]->columnwise_data.dptr : nullptr})) {
      return false;
    }
  }
  MultiQuantizeArgs args;
  args.num_tensors = num_tensors;
  args.first_chunk_Y[0] = 0;
  size_t chunks_X = 0;
  for (size_t i = 0; i < num_tensors; i++) {
    const size_t rows = input_list[i]->flat_first_dim();
    const size_t cols = input_list[i]->flat_last_dim();
    args.input[i] = input_list[i]->data.dptr;
    args.output_rowwise[i] = rowwise ? output_list[i]->data.dptr : nullptr;
    args.output_colwise[i] = colwise ? output_list[i]->columnwise_data.dptr : nullptr;
    args.scales_rowwise[i] = rowwise ? output_list[i]->scale_inv.dptr : nullptr;
    args.scales_colwise[i] = colwise ? output_list[i]->columnwise_scale_inv.dptr : nullptr;
    args.amax[i] = reinterpret_cast<float *>(output_list[i]->amax.dptr);
    args.rows[i] = rows;
    args.cols[i] = cols;
    args.first_chunk_Y[i + 1] = args.first_chunk_Y[i] + DIVUP(rows, CHUNK_DIM_Y);
    chunks_X = std::max(chunks_X, DIVUP(cols, CHUNK_DIM_X));
  }
  const dim3 grid(chunks_X, args.first_chunk_Y[num_tensors]);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input_list[0]->data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output_list[0]->dtype(), OType,
          TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
              (colwise ? 32 : 1), SCALE_DIM_Y,
              TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                  (rowwise ? 32 : 1), SCALE_DIM_X,
                  multi_quantize_mxfp8_tdm_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X>
                  <<<grid, THREADS_PER_CHUNK, 0, stream>>>(args);))));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
