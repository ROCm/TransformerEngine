/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <cfloat>
#include <hip/hip_runtime.h>
#include <limits>

#include "../common.h"
#include "math.h"
#include "ptx.cuh"
#include "rocm_vectorized_2d.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/cast.h"
#include "../transpose/cast_transpose.h"
#include "transformer_engine/transpose.h"
#include "../utils.cuh"
#include "vectorized_pointwise.h"

namespace transformer_engine {

namespace dequantization {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t BUFFERS_NUM = 2;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 16;           // only 32 is supported
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //  8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     //  128
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //    8 = 128 / 16
static_assert(ITERATIONS >= 1);

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_kernel(const IType *input_ptr,
                            OType *output_ptr,
                            const e8m0_t *const scales_ptr, const size_t rows, const size_t cols,
                            const size_t scales_stride) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      // 2 = 32 / 16
  constexpr size_t VECTOR_WIDTH = (IS_ALIGNED ?: 2) * 8 / sizeof(IType);

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  alignas(128) __shared__ IType in_sh[SHMEM_DIM_Y][SHMEM_DIM_X];
  alignas(128) __shared__ OType out_sh[SHMEM_DIM_Y][SHMEM_DIM_X];

  for (int iter = 0; iter < ITERATIONS; iter++) {
    const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
    const int chunk_it_offset_x = chunk_offset_X;

    copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_sh[0][0], input_ptr, chunk_it_offset_x, 
                      chunk_it_offset_y, cols, SHMEM_DIM_Y,
                      SHMEM_DIM_X, rows, cols);
    __syncthreads();

    const int scale_offset_Y =
        USE_ROWWISE_SCALING ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
                            : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

    const int scale_offset_X =
        USE_ROWWISE_SCALING
            ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
            : (scales_colwise_chunk_offset_X + tid_colwise_X);

    const int scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
    const e8m0_t biased_exponent = scales_ptr[scale_idx];
    const float block_scale = ptx::exp2f(biased_exponent);

    if constexpr (USE_ROWWISE_SCALING) {
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;

      const int shmem_offset_y = thread_offset_Y;
      const int shmem_offset_x = thread_offset_X_rowwise;
      in.load_from(&in_sh[shmem_offset_y][shmem_offset_x]);

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y; i++) {
        const float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
        out_sh[i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
      }
    }

    __syncthreads();

    bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_sh[0][0], output_ptr, chunk_it_offset_x,
                                    chunk_it_offset_y, cols, SHMEM_DIM_Y,
                                    SHMEM_DIM_X, rows, cols);

    __syncthreads();
  }
}
} // namespace dequantization

static void rocm_mxfp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  using namespace dequantization;

  bool use_rowwise_scaling = input.has_data();
  bool use_colwise_scaling = input.has_columnwise_data();

  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? 32 : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? 32 : 1;

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);

  const size_t scales_X_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t scales_X_colwise = cols;

  const e8m0_t *const scales_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(input.scale_inv.dptr)
                          : reinterpret_cast<e8m0_t *>(input.columnwise_scale_inv.dptr);

  const size_t scales_stride = use_rowwise_scaling ? scales_X_rowwise : scales_X_colwise;

  const SimpleTensor &input_data = use_rowwise_scaling ? input.data : input.columnwise_data;

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(chunks_X, chunks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
                  output->dtype(), OType,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  !(cols % (32 * sizeof(OType))), IS_ALIGNED,
                  {
                    dequantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED>
                        <<<grid, block, 0, stream>>>(
                            reinterpret_cast<const IType *>(input_data.dptr),
                            reinterpret_cast<OType *>(output->data.dptr),
                            scales_ptr, rows, cols, scales_stride);
                    NVTE_CHECK_CUDA(cudaGetLastError());
                  });  // NOLINT(*)
          );            // NOLINT(*)
      );                // NOLINT(*)
  );                    // NOLINT(*)
  );                    // NOLINT(*)
}

} // namespace transformer_engine
