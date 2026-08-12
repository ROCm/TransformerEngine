/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// drop-in rocm replacement for mxfp8 grouped dequantize kernel
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 16;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //  8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     //  128
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //    8 = 128 / 16
static_assert(ITERATIONS >= 1);

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    grouped_dequantize_mxfp8_kernel(const IType *input_ptr, OType *output_ptr,
                                    const e8m0_t *scales_ptr, const size_t first_logical_dim,
                                    const size_t last_logical_dim, const size_t num_tensors,
                                    const int64_t *const offsets_ptr,
                                    const int64_t *const first_dims_ptr,
                                    const int64_t *const last_dims_ptr) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);  // 2 = 32 / 16
  constexpr size_t VECTOR_WIDTH = IS_ALIGNED ? 8 : 16;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;

  alignas(128) __shared__ IType in_sh[SHMEM_DIM_Y][SHMEM_DIM_X];
  alignas(128) __shared__ OType out_sh[SHMEM_DIM_Y][SHMEM_DIM_X];

  // Mirrors the launcher's is_single_tensor: null last_dims means a 2D grid, not a tile stride.
  const bool uniform_last_dim = (last_dims_ptr == nullptr);
  size_t total_tiles = 1;
  size_t tiles_stride = 1;
  if (!uniform_last_dim) {
    total_tiles = 0;
    for (size_t t = 0; t < num_tensors; t++) {
      const size_t t_rows =
          (first_dims_ptr != nullptr) ? static_cast<size_t>(first_dims_ptr[t]) : first_logical_dim;
      const size_t t_cols = static_cast<size_t>(last_dims_ptr[t]);
      total_tiles += DIVUP(t_rows, CHUNK_DIM_Y) * DIVUP(t_cols, CHUNK_DIM_X);
    }
    tiles_stride = gridDim.x;
  }

  for (size_t block_tile = uniform_last_dim ? 0 : blockIdx.x; block_tile < total_tiles;
       block_tile += tiles_stride) {
    size_t rows = 0;
    size_t cols = 0;
    int block_id_Y = 0;
    int block_id_X = 0;
    size_t tensor_base_elts = 0;
    size_t scales_base_offset = 0;

    if (uniform_last_dim) {
      // Uniform last dim: the group is one contiguous 2D tensor and, because ROCm scale
      // tensors are unpadded, the per-tensor scale blocks stack into one contiguous array.
      rows = first_logical_dim;
      cols = last_logical_dim;
      block_id_Y = blockIdx.y;
      block_id_X = blockIdx.x;
    } else {
      // Find the tensor owning this tile.
      size_t tiles_before = 0;
      size_t tensor_id = 0;
      for (; tensor_id < num_tensors; tensor_id++) {
        const size_t t_rows = (first_dims_ptr != nullptr)
                                  ? static_cast<size_t>(first_dims_ptr[tensor_id])
                                  : first_logical_dim;
        const size_t t_cols = static_cast<size_t>(last_dims_ptr[tensor_id]);
        const size_t t_tiles = DIVUP(t_rows, CHUNK_DIM_Y) * DIVUP(t_cols, CHUNK_DIM_X);
        if (block_tile < tiles_before + t_tiles) {
          const size_t local_tile = block_tile - tiles_before;
          const size_t tiles_x = DIVUP(t_cols, CHUNK_DIM_X);
          rows = t_rows;
          cols = t_cols;
          block_id_Y = static_cast<int>(local_tile / tiles_x);
          block_id_X = static_cast<int>(local_tile % tiles_x);
          break;
        }
        tiles_before += t_tiles;
        tensor_base_elts += t_rows * t_cols;
        scales_base_offset += USE_ROWWISE_SCALING ? t_rows * DIVUP(t_cols, SCALE_DIM_X)
                                                  : DIVUP(t_rows, SCALE_DIM_Y) * t_cols;
      }
      if (tensor_id >= num_tensors) continue;
      if (offsets_ptr != nullptr) tensor_base_elts = static_cast<size_t>(offsets_ptr[tensor_id]);
    }

    if (rows == 0 || cols == 0) continue;

    // Per-tile views of the group buffers; the kernel arguments stay untouched so that the
    // bases do not accumulate across tiles.
    const IType *const tile_input_ptr = input_ptr + tensor_base_elts;
    OType *const tile_output_ptr = output_ptr + tensor_base_elts;
    const e8m0_t *const tile_scales_ptr = scales_ptr + scales_base_offset;

    const size_t scales_stride = USE_ROWWISE_SCALING ? DIVUP(cols, SCALE_DIM_X) : cols;

    const int chunk_offset_Y = block_id_Y * CHUNK_DIM_Y;
    const int chunk_offset_X = block_id_X * CHUNK_DIM_X;

    const int scales_rowwise_chunk_offset_Y = block_id_Y * SCALES_ROWWISE_PER_CHUNK_Y;
    const int scales_rowwise_chunk_offset_X = block_id_X * SCALES_ROWWISE_PER_CHUNK_X;
    const int scales_colwise_chunk_offset_Y = block_id_Y * SCALES_COLWISE_PER_CHUNK_Y;
    const int scales_colwise_chunk_offset_X = block_id_X * SCALES_COLWISE_PER_CHUNK_X;

    for (int iter = 0; iter < ITERATIONS; iter++) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_sh[0][0], tile_input_ptr,
                                                         chunk_it_offset_x, chunk_it_offset_y, cols,
                                                         SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      __syncthreads();

      const int scale_offset_Y =
          USE_ROWWISE_SCALING
              ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
              : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

      const int scale_offset_X =
          USE_ROWWISE_SCALING
              ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
              : (scales_colwise_chunk_offset_X + tid_colwise_X);

      const size_t scales_rows = USE_ROWWISE_SCALING ? rows : DIVUP(rows, SCALE_DIM_Y);
      const size_t scales_cols = USE_ROWWISE_SCALING ? DIVUP(cols, SCALE_DIM_X) : cols;

      e8m0_t biased_exponent = static_cast<e8m0_t>(127);
      if (static_cast<size_t>(scale_offset_Y) < scales_rows &&
          static_cast<size_t>(scale_offset_X) < scales_cols) {
        const size_t scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
        biased_exponent = tile_scales_ptr[scale_idx];
      }
      const float block_scale = ptx::exp2f(biased_exponent);

      if constexpr (USE_ROWWISE_SCALING) {
        Vec<IType, ELEMS_PER_THREAD> in;
        Vec<OType, ELEMS_PER_THREAD> out;

        const int shmem_offset_y = thread_offset_Y;
        const int shmem_offset_x = thread_offset_X_rowwise;
        in.load_from(&in_sh[shmem_offset_y][shmem_offset_x]);

#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
#if defined(__gfx1250__)
          // FIXME: Force E4M3 OCP interpretation because HIP headers do not declare
          // which type gfx1250 supports. This can be removed once HIP headers are updated.
          const float elt =
              std::is_same_v<IType, fp8e4m3>
                  ? static_cast<float>(*reinterpret_cast<__hip_fp8_e4m3 *>(&in.data.elt[j]))
                  : static_cast<float>(in.data.elt[j]);
          out.data.elt[j] = static_cast<OType>(block_scale * elt);
#else
          out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
#endif
        }
        out.store_to(&out_sh[shmem_offset_y][shmem_offset_x]);
      } else {
#pragma unroll
        for (int i = 0; i < BUFFER_DIM_Y; i++) {
#if defined(__gfx1250__)
          // FIXME: Force E4M3 OCP interpretation because HIP headers do not declare
          // which type gfx1250 supports. This can be removed once HIP headers are updated.
          const float elt = std::is_same_v<IType, fp8e4m3>
                                ? static_cast<float>(
                                      *reinterpret_cast<__hip_fp8_e4m3 *>(&in_sh[i][tid_colwise_X]))
                                : static_cast<float>(in_sh[i][tid_colwise_X]);
#else
          const float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
#endif
          out_sh[i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
        }
      }

      __syncthreads();

      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(
          &out_sh[0][0], tile_output_ptr, chunk_it_offset_x, chunk_it_offset_y, cols, SHMEM_DIM_Y,
          SHMEM_DIM_X, rows, cols);

      __syncthreads();
    }
  }
}
