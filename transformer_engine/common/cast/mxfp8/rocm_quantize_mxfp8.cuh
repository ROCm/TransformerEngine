/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// drop-in replacement for rocm quantize_mxfp8 kernels

constexpr size_t MXFP8_CHUNK_DIM_Y = 64;
constexpr size_t MXFP8_CHUNK_DIM_X = 64;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK = MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNKS_PER_BLOCK_X;
constexpr size_t MXFP8_THREADS_PER_CHUNK = 64;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t MXFP8_BUFFER_DIM_Y = 32;                 // only 32 is supported
constexpr size_t MXFP8_BUFFER_DIM_X = MXFP8_CHUNK_DIM_X;  // 64
constexpr size_t MXFP8_SHMEM_DIM_Y = MXFP8_BUFFER_DIM_Y;  // 32
constexpr size_t MXFP8_SHMEM_DIM_X = MXFP8_BUFFER_DIM_X;  // 64

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE =
    MXFP8_CHUNK_DIM_X / ELEMS_PER_THREAD;  //   4 = 64 / 16
constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE =
    MXFP8_THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;         //  16 = 64 / 4
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = MXFP8_CHUNK_DIM_X;  //  64
constexpr size_t MXFP8_BUFF_STAGES_NUM =
    MXFP8_BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;                        //   2 = 32 / 16
constexpr size_t MXFP8_ITERATIONS = MXFP8_CHUNK_DIM_Y / MXFP8_BUFFER_DIM_Y;  //   2 = 64 / 32

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, size_t SCALE_DIM_Y,
          size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(MXFP8_THREADS_PER_CHUNK)
    quantize_mxfp8_kernel(
      const IType *input_ptr,
      const IType *act_input_ptr,
      OType *output_rowwise,
      OType *output_colwise,
      e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
      const float *noop, float *const dbias_workspace, float *const amax_ptr,
      const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
      const size_t scale_stride_colwise) {
  if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
    if (noop != nullptr && noop[0] == 1.0f) return;
  }
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_DBIAS_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = MXFP8_CHUNK_DIM_Y;                //   2 = 64 / 32
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = MXFP8_CHUNK_DIM_X / SCALE_DIM_X;  //  64 = 64 / 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_Y =
      SCALES_ROWWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_X =
      SCALES_ROWWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = MXFP8_CHUNK_DIM_Y / SCALE_DIM_Y;  //   2 = 64 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = MXFP8_CHUNK_DIM_X;                //  64 = 64 / 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_Y =
      SCALES_COLWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_X =
      SCALES_COLWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      //   2 = 32 / 16
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE;  //   2
  constexpr size_t VECTOR_WIDTH = (IS_ALIGNED ?: 2) * 8 / sizeof(OType);

  const int block_offset_Y = blockIdx.y * MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X;
  const int scales_rowwise_block_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_BLOCK_Y;
  const int scales_rowwise_block_offset_X = blockIdx.x * SCALES_ROWWISE_PER_BLOCK_X;
  const int scales_colwise_block_offset_Y = blockIdx.y * SCALES_COLWISE_PER_BLOCK_Y;
  const int scales_colwise_block_offset_X = blockIdx.x * SCALES_COLWISE_PER_BLOCK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  const int dbias_rowwise_offset_Y = blockIdx.y * MXFP8_CHUNKS_PER_BLOCK_Y + tid_rowwise_Y;
  const int dbias_rowwise_block_offset_X =
      blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X + thread_offset_X_rowwise;
  const int dbias_colwise_offset_Y = blockIdx.y;
  const int dbias_colwise_block_offset_X =
      blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X + tid_colwise_X;
  const int dbias_stride = cols;

  Vec<float, ELEMS_PER_THREAD> partial_dbias_rowwise[MXFP8_CHUNKS_PER_BLOCK_X];
  float partial_dbias_colwise[MXFP8_CHUNKS_PER_BLOCK_X];
  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; i++) {
        partial_dbias_rowwise[i].clear();
      }
    } else {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; i++) {
        partial_dbias_colwise[i] = 0;
      }
    }
  }

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  alignas(128) __shared__ IType in_sh[MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  alignas(128) __shared__ IType act_in_sh[MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  alignas(128) __shared__ OType out_rowwise_sh[MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  alignas(128) __shared__ OType out_colwise_sh[MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];

  float block_amax = 0;

#pragma unroll
  for (int chunk = 0; chunk < MXFP8_CHUNKS_PER_BLOCK; chunk++) {
    const int chunk_Y = chunk / MXFP8_CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % MXFP8_CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * MXFP8_CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * MXFP8_CHUNK_DIM_X;

    const int dbias_rowwise_offset_X = dbias_rowwise_block_offset_X + chunk_X * MXFP8_CHUNK_DIM_X;
    const int dbias_colwise_offset_X = dbias_colwise_block_offset_X + chunk_X * MXFP8_CHUNK_DIM_X;

    const int scales_rowwise_chunk_offset_Y =
        scales_rowwise_block_offset_Y + chunk_Y * SCALES_ROWWISE_PER_CHUNK_Y;
    const int scales_rowwise_chunk_offset_X =
        scales_rowwise_block_offset_X + chunk_X * SCALES_ROWWISE_PER_CHUNK_X;
    const int scales_colwise_chunk_offset_Y =
        scales_colwise_block_offset_Y + chunk_Y * SCALES_COLWISE_PER_CHUNK_Y;
    const int scales_colwise_chunk_offset_X =
        scales_colwise_block_offset_X + chunk_X * SCALES_COLWISE_PER_CHUNK_X;

    __syncthreads();

#pragma unroll
    for (int iter = 0; iter < MXFP8_ITERATIONS; iter++) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      const size_t row_base = chunk_it_offset_y;
      if constexpr (IS_DACT) {
        copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&act_in_sh[0][0], act_input_ptr, 
                          chunk_it_offset_x, chunk_it_offset_y, cols, 
                          MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X, rows, cols);
      }
      copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_sh[0][0], input_ptr, chunk_it_offset_x, 
                        chunk_it_offset_y, cols, MXFP8_SHMEM_DIM_Y,
                        MXFP8_SHMEM_DIM_X, rows, cols);
      __syncthreads();

      if constexpr (USE_ROWWISE_SCALING) {
        Vec<IType, ELEMS_PER_THREAD> in;
        Vec<IType, ELEMS_PER_THREAD> act_in;
        Vec<OType, ELEMS_PER_THREAD> out_c;

        const int iteration_scale_rowwise_offset_Y =
            scales_rowwise_chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;

#pragma unroll
        for (int stage = 0; stage < MXFP8_BUFF_STAGES_NUM; stage++) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_ROWWISE;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X_rowwise;

          const size_t row = row_base + shmem_offset_y;
          const bool row_out_of_bounds = (row >= rows);

          in.load_from(&in_sh[shmem_offset_y][shmem_offset_x]);
          if constexpr (IS_DACT) {
            act_in.load_from(&act_in_sh[shmem_offset_y][shmem_offset_x]);
          }

          float thread_amax = 0;
          float in_compute[ELEMS_PER_THREAD];

#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            const bool col_out_of_bounds = (dbias_rowwise_offset_X + j >= cols);
            const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

            float elt = static_cast<float>(in.data.elt[j]);
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            if constexpr (IS_DACT) {
              float act_in_elt = static_cast<float>(act_in.data.elt[j]);
              elt *= OP(act_in_elt, {});
            }
            if constexpr (IS_DBIAS && COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
              if (!out_of_bounds) {
                partial_dbias_rowwise[chunk_X].data.elt[j] += elt;
              }
            }
            // Numerical truncation: downcast to IType (BF16/FP16) and upcast back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            in_compute[j] = elt;
            if (!out_of_bounds) {
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
          }

          __builtin_assume(block_amax >= 0);
          __builtin_assume(thread_amax >= 0);
          block_amax = fmaxf(block_amax, thread_amax);

          const float subwarp_amax = subwarp_reduce_max_broadcast<SUBWARP_WIDTH>(thread_amax);
          const e8m0_t biased_exponent =
              ptx::float_to_e8m0(subwarp_amax * Quantized_Limits<OType>::max_norm_rcp);
          // Only single thread writes the computed scaling factor
          if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0) {
            const int global_scales_offset_Y =
                iteration_scale_rowwise_offset_Y + stage_offset_Y + tid_rowwise_Y;
            const int global_scales_offset_X =
                scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
            const int scale_idx =
                global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
            scales_rowwise[scale_idx] = biased_exponent;
          }

          const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);

#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            out_c.data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
          }
          out_c.store_to(&out_rowwise_sh[shmem_offset_y][shmem_offset_x]);
        }
      }

      if constexpr (USE_COLWISE_SCALING) {
        const bool col_out_of_bounds = (dbias_colwise_offset_X >= cols);
        float in_compute[SCALE_DIM_Y];

        float amax = 0;
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; i++) {
          const size_t row = row_base + i;
          const bool row_out_of_bounds = (row >= rows);
          const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

          float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
          if constexpr (IS_ACT) {
            elt = OP(elt, {});
          }
          if constexpr (IS_DACT) {
            float act_in_elt = static_cast<float>(act_in_sh[i][tid_colwise_X]);
            elt *= OP(act_in_elt, {});
          }
          if constexpr (IS_DBIAS) {
            if (!out_of_bounds) {
              partial_dbias_colwise[chunk_X] += elt;
            }
          }
          // Numerical truncation: downcast to IType (BF16/FP16) and upcast back to FP32
          if constexpr (!std::is_same_v<IType, float>) {
            elt = static_cast<float>(static_cast<IType>(elt));
          }
          in_compute[i] = elt;
          if (!out_of_bounds) {
            amax = fmaxf(amax, fabsf(elt));
          }
        }

        __builtin_assume(block_amax >= 0);
        __builtin_assume(amax >= 0);
        block_amax = fmaxf(block_amax, amax);

        const e8m0_t biased_exponent = ptx::float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);

        const int global_scales_offset_Y = scales_colwise_chunk_offset_Y + iter;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;

        const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; i++) {
          out_colwise_sh[i][tid_colwise_X] =
              static_cast<OType>(in_compute[i] * block_scale_inverse);
        }
      }
      
      __syncthreads();

      if constexpr (USE_ROWWISE_SCALING) {
        bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_rowwise_sh[0][0], output_rowwise, chunk_it_offset_x,
                                        chunk_it_offset_y, cols, MXFP8_SHMEM_DIM_Y,
                                        MXFP8_SHMEM_DIM_X, rows, cols);
      }
      if constexpr (USE_COLWISE_SCALING) {
        bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_colwise_sh[0][0], output_colwise, chunk_it_offset_x,
                                        chunk_it_offset_y, cols, MXFP8_SHMEM_DIM_Y,
                                        MXFP8_SHMEM_DIM_X, rows, cols);
      }

      __syncthreads();
    }
  }

  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
      constexpr size_t CZ = MXFP8_CHUNKS_PER_BLOCK_X;
      constexpr size_t Y = THREADS_PER_CHUNK_Y_ROWWISE - 1;
      constexpr size_t X = THREADS_PER_CHUNK_X_ROWWISE;
      __shared__ float shmem_partial_dbias_rowwise[CZ][Y][X][ELEMS_PER_THREAD];

      if (tid_rowwise_Y > 0) {
#pragma unroll
        for (int c = 0; c < MXFP8_CHUNKS_PER_BLOCK_X; c++) {
          partial_dbias_rowwise[c].store_to(
              &shmem_partial_dbias_rowwise[c][tid_rowwise_Y - 1][tid_rowwise_X]);
        }
      }
      __syncthreads();

      if (tid_rowwise_Y == 0) {
#pragma unroll
        for (int c = 0; c < MXFP8_CHUNKS_PER_BLOCK_X; c++) {
          Vec<float, ELEMS_PER_THREAD> other_row_dbias;
          const int dbias_rowwise_offset_X = dbias_rowwise_block_offset_X + c * MXFP8_CHUNK_DIM_X;
          const int dbias_offset = dbias_rowwise_offset_Y * dbias_stride + dbias_rowwise_offset_X;

          const int left_bound = dbias_rowwise_offset_X;
          const int right_bound = dbias_rowwise_offset_X + ELEMS_PER_THREAD - 1;

#pragma unroll
          for (int i = 0; i < Y; i++) {
            other_row_dbias.load_from(&shmem_partial_dbias_rowwise[c][i][tid_rowwise_X]);
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              partial_dbias_rowwise[c].data.elt[j] += other_row_dbias.data.elt[j];
            }
          }

          // Vectorized store when all elements are inside the boundaries
          if (right_bound < cols) {
            partial_dbias_rowwise[c].store_to(&dbias_workspace[dbias_offset]);
          } else if (left_bound < cols && right_bound >= cols) {
            // Element-by-element store when some elements cross the boundaries
            const int in_bound_elts_count = cols - left_bound;
            partial_dbias_rowwise[c].store_to_elts(&dbias_workspace[dbias_offset], 0,
                                                   in_bound_elts_count);
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; i++) {
        const int dbias_colwise_offset_X = dbias_colwise_block_offset_X + i * MXFP8_CHUNK_DIM_X;
        const int dbias_offset = dbias_colwise_offset_Y * dbias_stride + dbias_colwise_offset_X;
        const bool col_out_of_bounds = (dbias_colwise_offset_X >= cols);
        if (!col_out_of_bounds) {
          dbias_workspace[dbias_offset] = partial_dbias_colwise[i];
        }
      }
    }
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    block_amax = reduce_max<MXFP8_THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (threadIdx.x == 0 && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }
}
