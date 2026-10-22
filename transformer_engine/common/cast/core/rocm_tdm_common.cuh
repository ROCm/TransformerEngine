/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 Tensor Data Mover pieces shared by the TDM quantize paths: the check whether TDM applies,
// the reduction of per-chunk dbias partials, and amax. This expects its includer to have included
// the common headers, util/system.h and util/cuda_runtime.h.

#include <algorithm>
#include <initializer_list>

#include "../../util/rocm_device_utils.cuh"
#include "../../util/rocm_tdm.cuh"

namespace transformer_engine {
namespace dispatch {
namespace common {

// TDM tiles need 16-byte aligned rows and 32-bit tensor extents.
inline bool use_tdm(const size_t rows, const size_t cols,
                    const std::initializer_list<const void *> tensors) {
  static const bool enabled = getenv<bool>("NVTE_USE_TDM", true);
  const auto aligned = [](const void *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
  return enabled && cuda::sm_arch() == 125 && cols % 16 == 0 && rows <= UINT32_MAX &&
         cols <= UINT32_MAX && std::all_of(tensors.begin(), tensors.end(), aligned);
}

namespace tdm_dbias_kernel {

constexpr size_t CHUNK_DIM_Y = 128;  // input rows per partial row
constexpr size_t SLAB_DIM_X = 128;
constexpr size_t BAND_Y = 32;
constexpr size_t WAVE_SIZE = 32;
constexpr size_t WAVES = 8;
constexpr size_t COLS_PER_LANE = SLAB_DIM_X / WAVE_SIZE;  // 4

// Sums each tensor's dbias partials, one row per chunk, into its row of dbias. A block covers
// WAVES / waves_per_slab 128-column slabs, whose waves split the BAND_Y-row bands.
template <typename OType>
__global__ void __launch_bounds__(WAVES * WAVE_SIZE)
    reduce_dbias_tdm_kernel(OType *const dbias, const float *const partial, const size_t cols,
                            const size_t partial_rows, const int64_t *const offsets_ptr,
                            const int64_t *const first_dims_ptr, const int waves_per_slab) {
#ifdef __gfx1250__
  __shared__ alignas(128) float band_sh[WAVES][BAND_Y][SLAB_DIM_X];

  const size_t tensor_id = blockIdx.y;
  size_t tensor_rows = partial_rows / gridDim.y;
  size_t tensor_offset_Y = tensor_id * tensor_rows;
  if (offsets_ptr != nullptr) {
    tensor_rows = static_cast<size_t>(first_dims_ptr[tensor_id]) / CHUNK_DIM_Y;
    tensor_offset_Y = static_cast<size_t>(offsets_ptr[tensor_id]) / cols / CHUNK_DIM_Y;
  }
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const size_t block_cols = WAVES / waves_per_slab * SLAB_DIM_X;
  const size_t block_offset_X = blockIdx.x * block_cols;
  const size_t slab_offset_X = block_offset_X + wave / waves_per_slab * SLAB_DIM_X;
  const int lane_offset_X = (threadIdx.x % WAVE_SIZE) * COLS_PER_LANE;
  const float *const tensor_partial = partial + tensor_offset_Y * cols;
  auto &band = band_sh[wave];

  Vec<float, COLS_PER_LANE> sum;
  sum.clear();
  if (slab_offset_X < cols) {
    for (size_t band_offset_Y = wave % waves_per_slab * BAND_Y; band_offset_Y < tensor_rows;
         band_offset_Y += waves_per_slab * BAND_Y) {
      tdm::load_2d(band, tensor_partial, tensor_rows, cols, cols, band_offset_Y, slab_offset_X);
      tdm::wait<0>();
#pragma unroll
      for (int y = 0; y < BAND_Y; y++) {
        Vec<float, COLS_PER_LANE> x;
        x.load_from(&band[y][lane_offset_X]);
#pragma unroll
        for (int i = 0; i < COLS_PER_LANE; i++) {
          sum.data.elt[i] += x.data.elt[i];
        }
      }
    }
  }
  sum.store_to(&band[0][lane_offset_X]);
  __syncthreads();
  for (size_t x = threadIdx.x; x < block_cols; x += blockDim.x) {
    const size_t col = block_offset_X + x;
    if (col < cols) {
      const int first_wave = x / SLAB_DIM_X * waves_per_slab;
      float dbias_sum = 0;
      for (int w = first_wave; w < first_wave + waves_per_slab; w++) {
        dbias_sum += band_sh[w][0][x % SLAB_DIM_X];
      }
      dbias[tensor_id * cols + col] = static_cast<OType>(dbias_sum);
    }
  }
#endif  // __gfx1250__
}

}  // namespace tdm_dbias_kernel

// Reduces the dbias partials of TDM chunks, one row per CHUNK_DIM_Y input rows. partial_rows counts
// the partials of all num_tensors tensors. Without offsets, the tensors share them equally;
// otherwise tensor t has first_dims[t] / CHUNK_DIM_Y of them.
template <typename OType>
void reduce_dbias_tdm(OType *dbias, const float *partial, const size_t cols,
                      const size_t num_tensors, const size_t partial_rows,
                      const int64_t *offsets_ptr, const int64_t *first_dims_ptr,
                      hipStream_t stream) {
  using namespace tdm_dbias_kernel;
  // About one band per wave; short tensors leave the spare waves to more slabs.
  const size_t bands = DIVUP(partial_rows / num_tensors, BAND_Y);
  size_t waves_per_slab = 1;
  while (waves_per_slab < WAVES && waves_per_slab < bands) {
    waves_per_slab *= 2;
  }
  const dim3 grid(DIVUP(cols, WAVES / waves_per_slab * SLAB_DIM_X), num_tensors);
  reduce_dbias_tdm_kernel<OType><<<grid, WAVES * WAVE_SIZE, 0, stream>>>(
      dbias, partial, cols, partial_rows, offsets_ptr, first_dims_ptr,
      static_cast<int>(waves_per_slab));
  NVTE_CHECK_CUDA(hipGetLastError());
}

namespace tdm_amax_kernel {

constexpr size_t TILE_BYTES = 16384;
constexpr size_t WAVE_SIZE = 32;
constexpr size_t WAVES = 8;

// Max |x| over the n elements of input, which the waves stream a row tile at a time. Atomics on
// one address serialize, so a block reduces its waves' amaxes first and grid has a block per CU.
template <typename IType>
__global__ void __launch_bounds__(WAVES * WAVE_SIZE)
    amax_tdm_kernel(const IType *input, float *amax_ptr, const size_t n, const float *noop) {
#ifdef __gfx1250__
  constexpr size_t TILE = TILE_BYTES / sizeof(IType);
  constexpr size_t VEC = 16 / sizeof(IType);
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  __shared__ alignas(128) IType tile_sh[WAVES][1][TILE];
  __shared__ float amax_sh[WAVES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  auto &tile = tile_sh[wave];
  const size_t step = static_cast<size_t>(gridDim.x) * WAVES * TILE;
  // The single-row tiles never step by the pitch.
  const size_t ld = DIVUP(n, size_t{16}) * 16;
  float amax = 0;
  for (size_t offset = (blockIdx.x * WAVES + wave) * TILE; offset < n; offset += step) {
    tdm::load_2d(tile, input, 1, n, ld, 0, offset);
    tdm::wait<0>();
#pragma unroll 4
    for (size_t i = lane * VEC; i < TILE; i += WAVE_SIZE * VEC) {
      Vec<IType, VEC> x;
      x.load_from(&tile[0][i]);
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(fabsf(static_cast<float>(x.data.elt[j])), amax);
      }
    }
  }
  const float wave_amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
  if (lane == 0) {
    amax_sh[wave] = wave_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_amax = 0;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      block_amax = fmaxf(block_amax, amax_sh[w]);
    }
    rocm_atomicMaxFloat(amax_ptr, block_amax);
  }
#endif  // __gfx1250__
}

}  // namespace tdm_amax_kernel

// Raises *amax to the max |x| of input's n elements, so the caller zeroes it first.
template <typename IType>
bool compute_amax_tdm(const IType *input, float *amax, const size_t n, const float *noop,
                      hipStream_t stream) {
  using namespace tdm_amax_kernel;
  // A single row has no row alignment to keep.
  if (n > UINT32_MAX || !use_tdm(1, 0, {input})) {
    return false;
  }
  constexpr size_t TILE = TILE_BYTES / sizeof(IType);
  const size_t blocks =
      std::min(DIVUP(n, TILE * WAVES), static_cast<size_t>(cuda::sm_count()));
  amax_tdm_kernel<IType><<<blocks, WAVES * WAVE_SIZE, 0, stream>>>(input, amax, n, noop);
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine
