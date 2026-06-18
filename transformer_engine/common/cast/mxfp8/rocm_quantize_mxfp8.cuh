/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// drop-in replacement for rocm quantize_mxfp8 kernels
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

#include "../../util/rocm_device_utils.cuh"

constexpr size_t MXFP8_CHUNK_DIM_Y = 64;
constexpr size_t MXFP8_CHUNK_DIM_X = 64;
constexpr size_t MXFP8_THREADS_PER_CHUNK = 64;

constexpr int kMultiQuantizeMXFP8MaxTensors = 1024;

struct MultiQuantizeMXFP8Args {
  const void *input_list[kMultiQuantizeMXFP8MaxTensors];
  void *output_rowwise_list[kMultiQuantizeMXFP8MaxTensors];
  void *output_colwise_list[kMultiQuantizeMXFP8MaxTensors];
  void *scales_rowwise_list[kMultiQuantizeMXFP8MaxTensors];
  void *scales_colwise_list[kMultiQuantizeMXFP8MaxTensors];
  float *amax_list[kMultiQuantizeMXFP8MaxTensors];
  int rows_list[kMultiQuantizeMXFP8MaxTensors];
  int cols_list[kMultiQuantizeMXFP8MaxTensors];
  int block_range[kMultiQuantizeMXFP8MaxTensors + 1];
  int num_tensors;
};


#ifdef HAS_CVT_4xFLOAT8
typedef short mxfp8_v2i16_t __attribute__((ext_vector_type(2)));
#endif  // #ifdef HAS_CVT_4xFLOAT8

// Single tensor quantize
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_ALIGNED,
          size_t CHUNK_DIM_Y = 64, size_t CHUNK_DIM_X = 64, size_t THREADS_PER_CHUNK = 64>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_mxfp8_kernel(
      const IType *input_ptr, const IType *act_input_ptr,
      OType *output_rowwise, OType *output_colwise,
      e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
      const float *noop, float *const dbias_workspace, float *amax_ptr,
      size_t rows, size_t cols, size_t scale_stride_rowwise, size_t scale_stride_colwise) {
  if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
    if (noop != nullptr && noop[0] == 1.0f) return;
  }
  const int block_id_Y = blockIdx.y;
  const int block_id_X = blockIdx.x;
  const int dbias_y_offset = blockIdx.y;
#include "rocm_quantize_mxfp8_body.inc"
}

// Grouped quantize (contiguous buffer + offsets)
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_ALIGNED,
          size_t CHUNK_DIM_Y = 64, size_t CHUNK_DIM_X = 64, size_t THREADS_PER_CHUNK = 64>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    grouped_quantize_mxfp8_kernel(
      const IType *input_ptr, const IType *act_input_ptr,
      OType *output_rowwise, OType *output_colwise,
      e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
      const float *noop, float *const dbias_workspace, float *amax_ptr,
      size_t rows, size_t cols, size_t scale_stride_rowwise, size_t scale_stride_colwise,
      const size_t num_tensors, const int64_t *const offsets_ptr,
      const int64_t *const first_dims_ptr, const int64_t *const last_dims_ptr,
      const size_t first_logical_dim) {
  if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
    if (noop != nullptr && noop[0] == 1.0f) return;
  }
  int block_id_Y = blockIdx.y;
  int block_id_X = blockIdx.x;
  size_t tensor_id;
  size_t tensor_base_elts;

  if (last_dims_ptr == nullptr) {
    const size_t global_row = static_cast<size_t>(blockIdx.y) * CHUNK_DIM_Y;
    size_t current_offset = global_row * cols;
    if (offsets_ptr == nullptr) {
      const size_t rows_per_tensor = first_logical_dim / num_tensors;
      tensor_id = global_row / rows_per_tensor;
      tensor_base_elts = tensor_id * rows_per_tensor * cols;
    } else {
      size_t lo = 1, hi = num_tensors;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (static_cast<size_t>(offsets_ptr[mid]) <= current_offset) lo = mid + 1;
        else hi = mid;
      }
      tensor_id = lo - 1;
      tensor_base_elts = static_cast<size_t>(offsets_ptr[tensor_id]);
    }
    rows = (first_dims_ptr != nullptr)
        ? static_cast<size_t>(first_dims_ptr[tensor_id]) : first_logical_dim / num_tensors;
    const size_t tensor_base_rows = tensor_base_elts / cols;
    block_id_Y = blockIdx.y - static_cast<int>(tensor_base_rows / CHUNK_DIM_Y);
  } else {
    size_t block_tile = static_cast<size_t>(blockIdx.x);
    size_t tiles_before = 0;
    tensor_base_elts = 0;
    for (tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
      size_t t_rows = first_dims_ptr ? static_cast<size_t>(first_dims_ptr[tensor_id]) : first_logical_dim;
      size_t t_cols = static_cast<size_t>(last_dims_ptr[tensor_id]);
      size_t t_tiles = DIVUP(t_rows, CHUNK_DIM_Y) * DIVUP(t_cols, CHUNK_DIM_X);
      if (block_tile < tiles_before + t_tiles) {
        rows = t_rows;
        cols = t_cols;
        size_t local_tile = block_tile - tiles_before;
        size_t tiles_x = DIVUP(t_cols, CHUNK_DIM_X);
        block_id_Y = static_cast<int>(local_tile / tiles_x);
        block_id_X = static_cast<int>(local_tile % tiles_x);
        break;
      }
      tiles_before += t_tiles;
      tensor_base_elts += t_rows * t_cols;
    }
    if (tensor_id >= num_tensors) return;
  }

  if (static_cast<size_t>(block_id_Y) * CHUNK_DIM_Y >= rows) return;
  if (static_cast<size_t>(block_id_X) * CHUNK_DIM_X >= cols) return;

  input_ptr      += tensor_base_elts;
  if (act_input_ptr) act_input_ptr += tensor_base_elts;
  if (output_rowwise) output_rowwise += tensor_base_elts;
  if (output_colwise) output_colwise += tensor_base_elts;

  scale_stride_rowwise = DIVUP(cols, SCALE_DIM_X);
  scale_stride_colwise = cols;

  if (last_dims_ptr == nullptr) {
    const size_t tensor_base_rows = tensor_base_elts / cols;
    if (scales_rowwise) scales_rowwise += tensor_base_rows * scale_stride_rowwise;
    if (scales_colwise) scales_colwise += (tensor_base_rows / SCALE_DIM_Y) * scale_stride_colwise;
  } else {
    size_t rowwise_offset = 0, colwise_offset = 0;
    for (size_t t = 0; t < tensor_id; t++) {
      size_t t_rows = first_dims_ptr ? static_cast<size_t>(first_dims_ptr[t]) : first_logical_dim;
      size_t t_cols = static_cast<size_t>(last_dims_ptr[t]);
      rowwise_offset += t_rows * DIVUP(t_cols, SCALE_DIM_X);
      colwise_offset += DIVUP(t_rows, SCALE_DIM_Y) * t_cols;
    }
    if (scales_rowwise) scales_rowwise += rowwise_offset;
    if (scales_colwise) scales_colwise += colwise_offset;
  }

  const int dbias_y_offset = blockIdx.y;
#include "rocm_quantize_mxfp8_body.inc"
}

// Multi-tensor quantize (per-tensor pointers)
template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED,
          size_t CHUNK_DIM_Y = 64, size_t CHUNK_DIM_X = 64, size_t THREADS_PER_CHUNK = 64>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    multi_quantize_mxfp8_kernel(MultiQuantizeMXFP8Args args) {
  const int row_tile = blockIdx.y;
  int lo = 0, hi = args.num_tensors - 1;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (args.block_range[mid + 1] <= row_tile) lo = mid + 1;
    else hi = mid;
  }
  const int tensor_id = lo;
  const size_t rows = args.rows_list[tensor_id];
  const size_t cols = args.cols_list[tensor_id];
  if (rows == 0) return;

  const int block_id_Y = row_tile - args.block_range[tensor_id];
  const int block_id_X = blockIdx.x;
  if (static_cast<size_t>(block_id_Y) * CHUNK_DIM_Y >= rows) return;
  if (static_cast<size_t>(block_id_X) * CHUNK_DIM_X >= cols) return;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT  = false;
  constexpr bool IS_ACT   = false;
  using ParamOP = Empty;
  constexpr float (*OP)(float, const Empty &) = nullptr;

  const IType *input_ptr       = reinterpret_cast<const IType *>(args.input_list[tensor_id]);
  const IType *act_input_ptr   = nullptr;
  OType *output_rowwise        = reinterpret_cast<OType *>(args.output_rowwise_list[tensor_id]);
  OType *output_colwise        = reinterpret_cast<OType *>(args.output_colwise_list[tensor_id]);
  e8m0_t *scales_rowwise       = reinterpret_cast<e8m0_t *>(args.scales_rowwise_list[tensor_id]);
  e8m0_t *scales_colwise       = reinterpret_cast<e8m0_t *>(args.scales_colwise_list[tensor_id]);
  float *dbias_workspace       = nullptr;
  float *amax_ptr              = args.amax_list[tensor_id];
  const size_t scale_stride_rowwise = DIVUP(cols, SCALE_DIM_X);
  const size_t scale_stride_colwise = cols;
  const int dbias_y_offset     = block_id_Y;
#include "rocm_quantize_mxfp8_body.inc"
}
