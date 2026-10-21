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

// hip_bfloat16 rounds with per-element branches; __bf16 is branch-free (v_cvt_pk_bf16_f32 on gfx950+).
template <typename T>
__device__ __forceinline__ T cvt_from_float(const float x) {
  if constexpr (std::is_same_v<T, bf16>) {
    return __builtin_bit_cast(T, static_cast<__bf16>(x));
  } else {
    return static_cast<T>(x);
  }
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
      // dequantize() only reaches this branch on gfx1250 (it rejects GEMM-swizzled scales on
      // other archs), so a swizzled buffer here is always the gfx1250 MX pre-swizzle layout.
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
        out.data.elt[j] = cvt_from_float<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y; i++) {
        const float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
        out_sh[i][tid_colwise_X] = cvt_from_float<OType>(block_scale * elt);
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

// gfx1250 variant with the same grid: each wave moves its own TDM_BAND_Y-row band of the chunk
// global->LDS->global with the Tensor Data Mover, so waves never wait for each other.
constexpr size_t TDM_WAVE_SIZE = 32;
constexpr size_t TDM_WAVES = THREADS_PER_CHUNK / TDM_WAVE_SIZE;  // 4
constexpr size_t TDM_BAND_Y = CHUNK_DIM_Y / TDM_WAVES;          // 32

// TDM tiles need 16-byte aligned rows and 32-bit tensor extents.
inline bool use_tdm_dequantize(const void *input, const void *output, size_t rows, size_t cols) {
  static const bool enabled = getenv<bool>("NVTE_USE_TDM", true);
  return enabled && cuda::sm_arch() == 125 && cols % 16 == 0 && rows <= UINT32_MAX &&
         cols <= UINT32_MAX && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(output) % 16 == 0;
}

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X,
          bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_tdm_kernel(const IType *input_ptr, OType *output_ptr,
                                const e8m0_t *const scales_ptr, const size_t rows,
                                const size_t cols, const size_t scales_stride,
                                const size_t mx_swizzle_padded_dim) {
#ifdef NVTE_ROCM_HAS_TDM
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  __shared__ alignas(128) IType in_sh[TDM_WAVES][TDM_BAND_Y][CHUNK_DIM_X];
  __shared__ alignas(128) OType out_sh[TDM_WAVES][TDM_BAND_Y][CHUNK_DIM_X];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / TDM_WAVE_SIZE);
  const int lane = threadIdx.x % TDM_WAVE_SIZE;
  const size_t band_offset_Y = blockIdx.y * CHUNK_DIM_Y + wave * TDM_BAND_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  if (band_offset_Y >= rows) {
    return;
  }
  tdm::load_2d(in_sh[wave], input_ptr, rows, cols, cols, band_offset_Y, chunk_offset_X);

  // Scales are fetched while the copy is in flight. Outside the scale tensor a lane reads a dummy
  // in-bounds scale and uses 2^0 instead, so the loads need no branches and are issued together.
  const size_t scales_rows = USE_ROWWISE_SCALING ? rows : DIVUP(rows, SCALE_DIM_Y);
  const size_t scales_cols = USE_ROWWISE_SCALING ? DIVUP(cols, SCALE_DIM_X) : cols;
  const auto load_scale = [&](const size_t scale_offset_Y, const size_t scale_offset_X) {
    const bool in_bounds = scale_offset_Y < scales_rows && scale_offset_X < scales_cols;
    size_t scale_idx;
    if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
      scale_idx = USE_ROWWISE_SCALING
                      ? mx_preswizzle_scale_idx(scale_offset_Y, scale_offset_X, mx_swizzle_padded_dim)
                      : mx_preswizzle_scale_idx(scale_offset_X, scale_offset_Y, mx_swizzle_padded_dim);
    } else {
      scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
    }
    const e8m0_t biased_exponent = scales_ptr[in_bounds ? scale_idx : 0];
    return in_bounds ? biased_exponent : static_cast<e8m0_t>(127);
  };

  if constexpr (USE_ROWWISE_SCALING) {
    // Per step, THREADS_PER_CHUNK_X_ROWWISE lanes cover one row of the band.
    constexpr size_t ROWS_PER_STEP = TDM_WAVE_SIZE / THREADS_PER_CHUNK_X_ROWWISE;  // 4
    constexpr size_t STEPS = TDM_BAND_Y / ROWS_PER_STEP;                           // 8
    const int lane_offset_Y = lane / THREADS_PER_CHUNK_X_ROWWISE;
    const int lane_offset_X = (lane % THREADS_PER_CHUNK_X_ROWWISE) * ELEMS_PER_THREAD;
    e8m0_t biased_exponent[STEPS];
#pragma unroll
    for (int step = 0; step < STEPS; step++) {
      biased_exponent[step] = load_scale(band_offset_Y + step * ROWS_PER_STEP + lane_offset_Y,
                                         (chunk_offset_X + lane_offset_X) / SCALE_DIM_X);
    }
    tdm::wait<0>();
#pragma unroll
    for (int step = 0; step < STEPS; step++) {
      const int y = step * ROWS_PER_STEP + lane_offset_Y;
      const float block_scale = ptx::exp2f(biased_exponent[step]);
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;
      in.load_from(&in_sh[wave][y][lane_offset_X]);
#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        out.data.elt[j] = cvt_from_float<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[wave][y][lane_offset_X]);
    }
  } else {
    // The band is one scale block tall; each lane owns COLS_PER_LANE adjacent columns.
    constexpr size_t COLS_PER_LANE = CHUNK_DIM_X / TDM_WAVE_SIZE;  // 4
    const int lane_offset_X = lane * COLS_PER_LANE;
    e8m0_t biased_exponent[COLS_PER_LANE];
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      biased_exponent[i] = load_scale(band_offset_Y / SCALE_DIM_Y, chunk_offset_X + lane_offset_X + i);
    }
    tdm::wait<0>();
    float block_scale[COLS_PER_LANE];
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      block_scale[i] = ptx::exp2f(biased_exponent[i]);
    }
#pragma unroll
    for (int y = 0; y < TDM_BAND_Y; y++) {
      Vec<IType, COLS_PER_LANE> in;
      Vec<OType, COLS_PER_LANE> out;
      in.load_from(&in_sh[wave][y][lane_offset_X]);
#pragma unroll
      for (int i = 0; i < COLS_PER_LANE; i++) {
        out.data.elt[i] = cvt_from_float<OType>(block_scale[i] * static_cast<float>(in.data.elt[i]));
      }
      out.store_to(&out_sh[wave][y][lane_offset_X]);
    }
  }

  tdm::store_2d(output_ptr, out_sh[wave], rows, cols, cols, band_offset_Y, chunk_offset_X);
  tdm::wait<0>();
#elif defined(__gfx1250__)
  // Compiler without the TDM builtins
  dequantize_mxfp8_chunk<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, false, WITH_GEMM_SWIZZLED_SCALES>(
      input_ptr, output_ptr, scales_ptr, rows, cols, scales_stride, mx_swizzle_padded_dim, blockIdx.y,
      blockIdx.x);
#endif  // NVTE_ROCM_HAS_TDM
}

