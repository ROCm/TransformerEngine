/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// drop-in rocm replacement for mxfp8 dequantize kernel
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

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

// MX pre-swizzle inverse index (matches swizzle_scaling_factors_mx in swizzle.cu):
//   dst = (j / 4) * (padded_dim * 4) + i * 4 + (j % 4)
// The grouped-by-4 dimension is j; the contiguous dimension is i.
__device__ __forceinline__ size_t mx_preswizzle_scale_idx(size_t i, size_t j, size_t padded_dim) {
  constexpr size_t GROUP = 4;  // MX_PRESWIZZLE_GROUP_SIZE
  return (j / GROUP) * (padded_dim * GROUP) + i * GROUP + (j % GROUP);
}

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED,
          bool WITH_GEMM_SWIZZLED_SCALES>
__device__ __forceinline__ void
    dequantize_mxfp8_chunk(const IType *input_ptr,
                           OType *output_ptr,
                           const e8m0_t *const scales_ptr, const size_t rows, const size_t cols,
                           const size_t scales_stride, const size_t mx_swizzle_padded_dim,
                           const int block_id_Y, const int block_id_X) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      // 2 = 32 / 16
  constexpr size_t VECTOR_WIDTH = IS_ALIGNED ? 8 : 16;

  const int chunk_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const int chunk_offset_X = block_id_X * CHUNK_DIM_X;

  const int scales_rowwise_chunk_offset_Y = block_id_Y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = block_id_X * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = block_id_Y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = block_id_X * SCALES_COLWISE_PER_CHUNK_X;

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

    const size_t scales_rows = USE_ROWWISE_SCALING ? rows : DIVUP(rows, SCALE_DIM_Y);
    const size_t scales_cols = USE_ROWWISE_SCALING ? DIVUP(cols, SCALE_DIM_X) : cols;

    e8m0_t biased_exponent = static_cast<e8m0_t>(127);
    if (static_cast<size_t>(scale_offset_Y) < scales_rows &&
      static_cast<size_t>(scale_offset_X) < scales_cols) {
      size_t scale_idx;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = USE_ROWWISE_SCALING
                        ? mx_preswizzle_scale_idx(scale_offset_Y, scale_offset_X,
                                                  mx_swizzle_padded_dim)
                        : mx_preswizzle_scale_idx(scale_offset_X, scale_offset_Y,
                                                  mx_swizzle_padded_dim);
      } else {
        scale_idx = static_cast<size_t>(scale_offset_Y) * scales_stride + scale_offset_X;
      }
      biased_exponent = scales_ptr[scale_idx];
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
        const float elt = std::is_same_v<IType, fp8e4m3>
                              ? static_cast<float>(*reinterpret_cast<__hip_fp8_e4m3 *>(
                                    &in.data.elt[j]))
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
          ? static_cast<float>(*reinterpret_cast<__hip_fp8_e4m3 *>(
            &in_sh[i][tid_colwise_X]))
          : static_cast<float>(in_sh[i][tid_colwise_X]);
#else
        const float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
#endif
        out_sh[i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
      }
    }

    __syncthreads();

    bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_sh[0][0], output_ptr, chunk_it_offset_x,
                                    chunk_it_offset_y, cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);

    __syncthreads();
  }
}

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED,
          bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_kernel(const IType *input_ptr,
                            OType *output_ptr,
                            const e8m0_t *const scales_ptr, const size_t rows, const size_t cols,
                            const size_t scales_stride, const size_t mx_swizzle_padded_dim) {
  dequantize_mxfp8_chunk<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED,
                         WITH_GEMM_SWIZZLED_SCALES>(input_ptr, output_ptr, scales_ptr, rows, cols,
                                                    scales_stride, mx_swizzle_padded_dim,
                                                    blockIdx.y, blockIdx.x);
}

