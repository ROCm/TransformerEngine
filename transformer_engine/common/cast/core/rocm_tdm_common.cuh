/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 Tensor Data Mover pieces shared by the TDM quantize paths: the check whether TDM applies,
// stores, the chunks of grouped tensors, the reduction of per-chunk dbias partials, and amax. This
// expects its includer to have included the common headers, util/system.h and util/cuda_runtime.h.

#include <algorithm>
#include <initializer_list>

#include "../../util/rocm_device_utils.cuh"
#include "grouped_layout.cuh"
#include "rocm_quantize_common.cuh"

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

// Stores the first n of the N bytes of src to dst, at once when that is all of them and dst is
// aligned for it.
template <int N, typename T>
__device__ __forceinline__ void store_bytes(T *dst, const T (&src)[N], const int n) {
  static_assert(sizeof(T) == 1 && (N == 4 || N == 8), "Stores 4 or 8 bytes");
  using Word = std::conditional_t<N == 8, uint64_t, uint32_t>;
  if (n >= N && reinterpret_cast<uintptr_t>(dst) % N == 0) {
    *reinterpret_cast<Word *>(dst) = *reinterpret_cast<const Word *>(src);
  } else {
    for (int i = 0; i < min(n, N); i++) {
      dst[i] = src[i];
    }
  }
}

struct GroupChunk {
  size_t offset;  // of the tensor's first element
  size_t rows;    // of the tensor, or 0 past the last one
  size_t cols;
  size_t chunk_Y;
  size_t chunk_X;
};

// Chunk `chunk` of a grouped tensor whose tensors have dimensions divisible by the chunk's, so that
// tensor t owns the chunks from offsets[t] / (CHUNK_DIM_Y * CHUNK_DIM_X) on, row-major. Without
// first_dims, every tensor has first_logical_dim rows.
template <size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X>
__device__ __forceinline__ GroupChunk find_group_chunk(const size_t chunk, const size_t num_tensors,
                                                       const size_t first_logical_dim,
                                                       const int64_t *offsets,
                                                       const int64_t *first_dims,
                                                       const int64_t *last_dims) {
  constexpr size_t CHUNK_ELTS = CHUNK_DIM_Y * CHUNK_DIM_X;
  const size_t chunk_offset = chunk * CHUNK_ELTS;
  const size_t tensor_id    = find_tensor_from_offsets(offsets, num_tensors, chunk_offset);
  GroupChunk c{};
  c.offset = static_cast<size_t>(offsets[tensor_id]);
  c.rows = first_dims != nullptr ? static_cast<size_t>(first_dims[tensor_id]) : first_logical_dim;
  c.cols = static_cast<size_t>(last_dims[tensor_id]);
  if (chunk_offset >= c.offset + c.rows * c.cols) {
    c.rows = 0;
    return c;
  }
  const size_t chunks_X     = c.cols / CHUNK_DIM_X;
  const size_t tensor_chunk = (chunk_offset - c.offset) / CHUNK_ELTS;
  c.chunk_Y = tensor_chunk / chunks_X;
  c.chunk_X = tensor_chunk % chunks_X;
  return c;
}

namespace tdm_dbias_kernel {

constexpr size_t CHUNK_DIM_Y   = 128;  // input rows per partial row
constexpr size_t SLAB_DIM_X    = 128;
constexpr size_t BAND_Y        = 32;
constexpr size_t WAVES         = 8;
constexpr size_t COLS_PER_LANE = SLAB_DIM_X / ROCM_CT_WARP_SIZE;  // 4

// Sums each tensor's dbias partials, one row per chunk, into its row of dbias. A block covers
// WAVES / waves_per_slab 128-column slabs, whose waves split the BAND_Y-row bands.
template <typename OType>
__global__ void __launch_bounds__(WAVES * ROCM_CT_WARP_SIZE)
    reduce_dbias_tdm_kernel(OType *const dbias, const float *const partial, const size_t cols,
                            const size_t partial_rows, const int64_t *const offsets_ptr,
                            const int64_t *const first_dims_ptr, const int waves_per_slab) {
#ifdef __gfx1250__
  __shared__ alignas(128) float band_sh[WAVES][BAND_Y][SLAB_DIM_X];

  const size_t tensor_id = blockIdx.y;
  size_t tensor_rows     = partial_rows / gridDim.y;
  size_t tensor_offset_Y = tensor_id * tensor_rows;
  if (offsets_ptr != nullptr) {
    tensor_rows = static_cast<size_t>(first_dims_ptr[tensor_id]) / CHUNK_DIM_Y;
    tensor_offset_Y = static_cast<size_t>(offsets_ptr[tensor_id]) / cols / CHUNK_DIM_Y;
  }
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / ROCM_CT_WARP_SIZE);
  const size_t block_cols     = WAVES / waves_per_slab * SLAB_DIM_X;
  const size_t block_offset_X = blockIdx.x * block_cols;
  const size_t slab_offset_X  = block_offset_X + wave / waves_per_slab * SLAB_DIM_X;
  const int lane_offset_X = (threadIdx.x % ROCM_CT_WARP_SIZE) * COLS_PER_LANE;
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
  reduce_dbias_tdm_kernel<OType><<<grid, WAVES * ROCM_CT_WARP_SIZE, 0, stream>>>(
      dbias, partial, cols, partial_rows, offsets_ptr, first_dims_ptr,
      static_cast<int>(waves_per_slab));
  NVTE_CHECK_CUDA(hipGetLastError());
}

namespace tdm_amax_kernel {

constexpr size_t TILE_BYTES = 16384;
constexpr size_t WAVES      = 8;

// Max |x| over the n elements of input, which the waves stream a row tile at a time. Atomics on
// one address serialize, so a block reduces its waves' amaxes first and grid has a block per CU.
template <typename IType>
__global__ void __launch_bounds__(WAVES * ROCM_CT_WARP_SIZE)
    amax_tdm_kernel(const IType *input, float *amax_ptr, const size_t n, const float *noop) {
#ifdef __gfx1250__
  constexpr size_t TILE = TILE_BYTES / sizeof(IType);
  constexpr size_t VEC  = 16 / sizeof(IType);
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  __shared__ alignas(128) IType tile_sh[WAVES][1][TILE];
  __shared__ float amax_sh[WAVES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / ROCM_CT_WARP_SIZE);
  const int lane = threadIdx.x % ROCM_CT_WARP_SIZE;
  auto &tile = tile_sh[wave];
  const size_t step = static_cast<size_t>(gridDim.x) * WAVES * TILE;
  // The single-row tiles never step by the pitch.
  const size_t ld = DIVUP(n, size_t{16}) * 16;
  float amax = 0;
  for (size_t offset = (blockIdx.x * WAVES + wave) * TILE; offset < n; offset += step) {
    tdm::load_2d(tile, input, 1, n, ld, 0, offset);
    tdm::wait<0>();
#pragma unroll 4
    for (size_t i = lane * VEC; i < TILE; i += ROCM_CT_WARP_SIZE * VEC) {
      Vec<IType, VEC> x;
      x.load_from(&tile[0][i]);
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(fabsf(static_cast<float>(x.data.elt[j])), amax);
      }
    }
  }
  const float wave_amax = rocm_subwarp_allreduce<ROCM_CT_WARP_SIZE>(amax, rocm_op::max{});
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
  amax_tdm_kernel<IType><<<blocks, WAVES * ROCM_CT_WARP_SIZE, 0, stream>>>(input, amax, n, noop);
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

namespace tdm_amax_kernel {

constexpr int RHT_TILE_Y        = 64;
constexpr int RHT_TILE_X        = 128;
constexpr int RHT_DIM           = 16;
constexpr int RHT_COLS_PER_LANE = RHT_TILE_X / ROCM_CT_WARP_SIZE;  // 4

// The splits of a tensor along dimension 0 whose amaxes split_amax_tdm_kernel computes.
template <int kMaxSplits>
struct SplitAmaxArgs {
  const void *input[kMaxSplits];
  float *amax[kMaxSplits];
  float *amax_t[kMaxSplits];
  uint32_t rows[kMaxSplits];
};

// For split blockIdx.y of the cols-wide BF16 input: max |x| into *amax and, with kRHT, into
// *amax_t the max |y| over the transposed RHT y of its 16-row column blocks, rounded to BF16 as
// hadamard_transform stores it, else max |x| as well. Null pointers are skipped. Grid and reduction
// as amax_tdm_kernel, with 2D tiles.
template <bool kRHT, int kMaxSplits>
__global__ void __launch_bounds__(WAVES * ROCM_CT_WARP_SIZE)
    split_amax_tdm_kernel(const SplitAmaxArgs<kMaxSplits> args, const uint32_t cols,
                          const uint32_t sign_mask) {
#ifdef __gfx1250__
  __shared__ alignas(128) uint16_t tile_sh[WAVES][RHT_TILE_Y][RHT_TILE_X];
  __shared__ float amax_sh[2][WAVES];

  const uint16_t *const input = static_cast<const uint16_t *>(args.input[blockIdx.y]);
  const uint32_t rows = args.rows[blockIdx.y];
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / ROCM_CT_WARP_SIZE);
  const int lane = threadIdx.x % ROCM_CT_WARP_SIZE;
  auto &tile = tile_sh[wave];
  const uint32_t tiles_x = DIVUP(cols, static_cast<uint32_t>(RHT_TILE_X));
  const uint32_t tiles   = DIVUP(rows, static_cast<uint32_t>(RHT_TILE_Y)) * tiles_x;
  float amax   = 0;
  float amax_t = 0;
  for (uint32_t t = blockIdx.x * WAVES + wave; t < tiles; t += gridDim.x * WAVES) {
    tdm::load_2d(tile, input, rows, cols, cols, t / tiles_x * RHT_TILE_Y,
                 t % tiles_x * RHT_TILE_X);
    tdm::wait<0>();
#pragma unroll
    for (int b = 0; b < RHT_TILE_Y / RHT_DIM; b++) {
      float x[RHT_COLS_PER_LANE][RHT_DIM];
#pragma unroll
      for (int y = 0; y < RHT_DIM; y++) {
        const uint2 bits =
            *reinterpret_cast<const uint2 *>(&tile[b * RHT_DIM + y][lane * RHT_COLS_PER_LANE]);
        x[0][y] = __uint_as_float(bits.x << 16);
        x[1][y] = __uint_as_float(bits.x & 0xFFFF0000);
        x[2][y] = __uint_as_float(bits.y << 16);
        x[3][y] = __uint_as_float(bits.y & 0xFFFF0000);
      }
#pragma unroll
      for (int i = 0; i < RHT_COLS_PER_LANE; i++) {
#pragma unroll
        for (int y = 0; y < RHT_DIM; y++) {
          amax = fmaxf(amax, fabsf(x[i][y]));
        }
        if constexpr (kRHT) {
#pragma unroll
          for (int y = 0; y < RHT_DIM; y++) {
            x[i][y] = (sign_mask >> y) & 1 ? -x[i][y] : x[i][y];
          }
          hadamard16(x[i]);
#pragma unroll
          for (int y = 0; y < RHT_DIM; y++) {
            amax_t = fmaxf(amax_t, fabsf(static_cast<float>(static_cast<__bf16>(x[i][y]))));
          }
        }
      }
    }
  }
  amax = rocm_subwarp_allreduce<ROCM_CT_WARP_SIZE>(amax, rocm_op::max{});
  if constexpr (kRHT) {
    amax_t = rocm_subwarp_allreduce<ROCM_CT_WARP_SIZE>(amax_t, rocm_op::max{});
  } else {
    amax_t = amax;
  }
  if (lane == 0) {
    amax_sh[0][wave] = amax;
    amax_sh[1][wave] = amax_t;
  }
  __syncthreads();
  if (threadIdx.x < 2) {
    const float *const wave_amax = amax_sh[threadIdx.x];
    float block_amax = 0;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      block_amax = fmaxf(block_amax, wave_amax[w]);
    }
    float *const ptr = threadIdx.x == 0 ? args.amax[blockIdx.y] : args.amax_t[blockIdx.y];
    if (ptr != nullptr) {
      rocm_atomicMaxFloat(ptr, block_amax);
    }
  }
#endif  // __gfx1250__
}

template <int kMaxSplits>
__global__ void zero_split_amax_kernel(const SplitAmaxArgs<kMaxSplits> args, const int splits) {
  for (int t = threadIdx.x; t < splits; t += blockDim.x) {
    if (args.amax[t] != nullptr) {
      *args.amax[t] = 0;
    }
    if (args.amax_t[t] != nullptr) {
      *args.amax_t[t] = 0;
    }
  }
}

// Sets the amaxes of the first splits of args as split_amax_tdm_kernel defines them, spreading a
// block per CU over the splits by size.
template <bool kRHT, int kMaxSplits>
void launch_split_amax_tdm(const SplitAmaxArgs<kMaxSplits> &args, const int splits,
                           const size_t cols, const uint16_t sign_mask, const bool zero,
                           hipStream_t stream) {
  size_t max_rows   = 0;
  size_t total_rows = 0;
  for (int t = 0; t < splits; t++) {
    max_rows = std::max<size_t>(max_rows, args.rows[t]);
    total_rows += args.rows[t];
  }
  if (zero) {
    zero_split_amax_kernel<kMaxSplits><<<1, 256, 0, stream>>>(args, splits);
  }
  if (total_rows == 0) {
    NVTE_CHECK_CUDA(hipGetLastError());
    return;
  }
  // The largest split's tiles over its share of the CUs, as each split gets as many blocks.
  const size_t tiles = DIVUP(max_rows, static_cast<size_t>(RHT_TILE_Y)) *
                       DIVUP(cols, static_cast<size_t>(RHT_TILE_X));
  const size_t share = std::max<size_t>(1, cuda::sm_count() * max_rows / total_rows);
  const dim3 grid(std::min(DIVUP(tiles, WAVES), share), splits);
  split_amax_tdm_kernel<kRHT, kMaxSplits>
      <<<grid, WAVES * ROCM_CT_WARP_SIZE, 0, stream>>>(args, cols, sign_mask);
  NVTE_CHECK_CUDA(hipGetLastError());
}

}  // namespace tdm_amax_kernel

// Raises *amax to the max |x| of the rows x cols BF16 input and *amax_t to that of the RHT of its
// transpose with sign_mask, as hadamard_transform_amax computes them; null pointers are skipped.
template <typename = void>
bool compute_rht_amax_tdm(const void *input, float *amax, float *amax_t, const size_t rows,
                          const size_t cols, const uint16_t sign_mask, hipStream_t stream) {
  using namespace tdm_amax_kernel;
  if (rows % RHT_DIM != 0 || !use_tdm(rows, cols, {input})) {
    return false;
  }
  SplitAmaxArgs<1> args;
  args.input[0] = input;
  args.amax[0] = amax;
  args.amax_t[0] = amax_t;
  args.rows[0] = rows;
  launch_split_amax_tdm<true>(args, 1, cols, sign_mask, false, stream);
  return true;
}

namespace tdm_amax_kernel {

// amax[r] = max |x| over row r of the rows x cols input. A wave owns whole rows, streamed in
// single-row tiles, so it writes each amax without atomics.
template <typename IType>
__global__ void __launch_bounds__(WAVES * ROCM_CT_WARP_SIZE)
    rowwise_amax_tdm_kernel(const IType *input, float *amax_ptr, const uint32_t rows,
                            const uint32_t cols, const float *noop) {
#ifdef __gfx1250__
  constexpr uint32_t TILE = TILE_BYTES / sizeof(IType);
  constexpr uint32_t VEC  = 16 / sizeof(IType);
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  __shared__ alignas(128) IType tile_sh[WAVES][1][TILE];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / ROCM_CT_WARP_SIZE);
  const int lane = threadIdx.x % ROCM_CT_WARP_SIZE;
  auto &tile = tile_sh[wave];
  for (uint32_t row = blockIdx.x * WAVES + wave; row < rows; row += gridDim.x * WAVES) {
    float amax = 0;
    for (uint32_t col = 0; col < cols; col += TILE) {
      tdm::load_2d(tile, input, rows, cols, cols, row, col);
      tdm::wait<0>();
      const uint32_t n = min(TILE, cols - col);
      for (uint32_t i = lane * VEC; i < n; i += ROCM_CT_WARP_SIZE * VEC) {
        Vec<IType, VEC> x;
        x.load_from(&tile[0][i]);
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(fabsf(static_cast<float>(x.data.elt[j])), amax);
        }
      }
    }
    amax = rocm_subwarp_allreduce<ROCM_CT_WARP_SIZE>(amax, rocm_op::max{});
    if (lane == 0) {
      amax_ptr[row] = amax;
    }
  }
#endif  // __gfx1250__
}

}  // namespace tdm_amax_kernel

// amax[r] = max |x| over row r of the rows x cols input, as the row-scaled NVFP4 quantize needs.
template <typename IType>
bool compute_rowwise_amax_tdm(const IType *input, float *amax, const size_t rows,
                              const size_t cols, const float *noop, hipStream_t stream) {
  using namespace tdm_amax_kernel;
  if (!use_tdm(rows, cols, {input})) {
    return false;
  }
  const size_t blocks = std::min(DIVUP(rows, WAVES), static_cast<size_t>(cuda::sm_count()));
  rowwise_amax_tdm_kernel<IType>
      <<<blocks, WAVES * ROCM_CT_WARP_SIZE, 0, stream>>>(input, amax, rows, cols, noop);
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine
