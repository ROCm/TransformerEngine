/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*************************************************************************
 * Common cast+transpose tiled implementation shared between:
 *  - transpose/rtc/cast_transpose.cu   (CUDA NVRTC / hipRTC)
 *  - transpose/rocm_cast_transpose.cuh (ROCm static HIP compilation)
 *
 * Requires the including translation unit to have already included:
 *  - utils.cuh (CUDA) or utils_hip.cuh (hipified) for Vec, reduce_max,
 *    atomicMaxFloat, reciprocal, THREADS_PER_WARP
 *  - util/rocm_device_utils.cuh (ROCm only) for NTVec,
 *    rocm_block_reduce_max, rocm_atomicMaxFloat, ROCM_CT_WARP_SIZE
 ************************************************************************/
#pragma once

namespace transformer_engine {
namespace transpose {

// -----------------------------------------------------------------------
// NvOps: Ops policy for CUDA/NVRTC (and hipified hipRTC) path.
// Uses Vec<T,N> from utils.cuh.
// Note: Vec uses uint32_t for its size parameter.
// -----------------------------------------------------------------------
struct NvOps {
  template <typename T, size_t N>
  using VecT = Vec<T, N>;

  template <typename T, uint32_t N>
  static __device__ __forceinline__ void vec_load(Vec<T, N> &v, const T *ptr) {
    v.load_from(ptr);
  }

  template <typename T, uint32_t N>
  static __device__ __forceinline__ float vec_elt_as_float(const Vec<T, N> &v,
                                                           int j) {
    return static_cast<float>(v.data.elt[j]);
  }

  template <typename IType, typename OType, size_t NVEC_IN, size_t NVEC_OUT,
            size_t NUM_ITERS>
  static __device__ __forceinline__ void pack_store_c_fill_t(
      const Vec<IType, NVEC_IN> &in_vec, float scale,
      Vec<OType, NVEC_IN> &out_c,
      Vec<OType, NVEC_OUT> (&local_t)[NVEC_IN][NUM_ITERS],
      size_t iter, size_t i2) {
#pragma unroll
    for (size_t j2 = 0; j2 < NVEC_IN; ++j2) {
      const float v = static_cast<float>(in_vec.data.elt[j2]);
      const OType o = OType(v * scale);
      out_c.data.elt[j2] = o;
      local_t[j2][iter].data.elt[i2] = o;
    }
  }

  template <typename T, uint32_t N>
  static __device__ __forceinline__ void vec_store_c(const Vec<T, N> &v,
                                                     T *ptr) {
    v.store_to(ptr);
  }

  template <typename T, uint32_t N>
  static __device__ __forceinline__ void vec_store_t(const Vec<T, N> &v,
                                                     T *ptr) {
    v.store_to(ptr);
  }

  template <size_t WARPS>
  static __device__ __forceinline__ float block_reduce_max(float val,
                                                           int warpid) {
    return reduce_max<WARPS>(val, warpid);
  }

  static __device__ __forceinline__ void atomic_max_float(float *addr,
                                                          float val) {
    atomicMaxFloat(addr, val);
  }

  static __device__ __forceinline__ void write_scale_inv(float *ptr,
                                                         float scale) {
    reciprocal<float>(ptr, scale);
  }
};

// -----------------------------------------------------------------------
// RocmOps: Ops policy for ROCm static HIP compilation path.
// Uses NTVec<T,N> from rocm_device_utils.cuh with non-temporal stores.
// Note: NTVec uses int for its size parameter.
// -----------------------------------------------------------------------
#ifdef __HIP_PLATFORM_AMD__
struct RocmOps {
  template <typename T, size_t N>
  using VecT = NTVec<T, static_cast<int>(N)>;

  template <typename T, int N>
  static __device__ __forceinline__ void vec_load(NTVec<T, N> &v,
                                                  const T *ptr) {
    v.load(ptr);
  }

  template <typename T, int N>
  static __device__ __forceinline__ float vec_elt_as_float(
      const NTVec<T, N> &v, int j) {
    return static_cast<float>(v.val[j]);
  }

  template <typename IType, typename OType, size_t NVEC_IN, size_t NVEC_OUT,
            size_t NUM_ITERS>
  static __device__ __forceinline__ void pack_store_c_fill_t(
      const NTVec<IType, static_cast<int>(NVEC_IN)> &in_vec, float scale,
      NTVec<OType, static_cast<int>(NVEC_IN)> &out_c,
      NTVec<OType, static_cast<int>(NVEC_OUT)> (&local_t)[NVEC_IN][NUM_ITERS],
      size_t iter, size_t i2) {
#if defined(__gfx950__) && __HIP_DEVICE_COMPILE__
    if constexpr (sizeof(OType) == 1) {
#pragma unroll
      for (size_t j2 = 0; j2 < NVEC_IN; j2 += 4) {
        uint32_t packed = rocm_cvt_4xfloat8<OType>(
            static_cast<float>(in_vec.val[j2]) * scale,
            (j2 + 1 < NVEC_IN) ? static_cast<float>(in_vec.val[j2 + 1]) * scale
                                : 0.0f,
            (j2 + 2 < NVEC_IN) ? static_cast<float>(in_vec.val[j2 + 2]) * scale
                                : 0.0f,
            (j2 + 3 < NVEC_IN) ? static_cast<float>(in_vec.val[j2 + 3]) * scale
                                : 0.0f);
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&packed);
#pragma unroll
        for (size_t k = 0; k < 4 && j2 + k < NVEC_IN; ++k) {
          out_c.val[j2 + k] = reinterpret_cast<OType &>(bytes[k]);
          local_t[j2 + k][iter].val[i2] = out_c.val[j2 + k];
        }
      }
    } else
#endif  // #if defined(__gfx950__)
    {
#pragma unroll
      for (size_t j2 = 0; j2 < NVEC_IN; ++j2) {
        const OType o =
            static_cast<OType>(static_cast<float>(in_vec.val[j2]) * scale);
        out_c.val[j2] = o;
        local_t[j2][iter].val[i2] = o;
      }
    }
  }

  template <typename T, int N>
  static __device__ __forceinline__ void vec_store_c(
      const NTVec<T, N> &v, T *ptr) {
    v.nt_store(ptr);
  }

  template <typename T, int N>
  static __device__ __forceinline__ void vec_store_t(
      const NTVec<T, N> &v, T *ptr) {
    v.nt_store(ptr);
  }

  template <size_t WARPS>
  static __device__ __forceinline__ float block_reduce_max(float val,
                                                           int warpid) {
    return rocm_block_reduce_max<static_cast<int>(WARPS)>(val, warpid);
  }

  static __device__ __forceinline__ void atomic_max_float(float *addr,
                                                          float val) {
    rocm_atomicMaxFloat(addr, val);
  }

  static __device__ __forceinline__ void write_scale_inv(float *ptr,
                                                         float scale) {
    *ptr = __frcp_rn(scale);
  }
};
#endif  // __HIP_PLATFORM_AMD__

// -----------------------------------------------------------------------
// cast_transpose_tile_impl: unified tiled cast+transpose device function.
//
// Template parameters:
//   Ops            - NvOps or RocmOps policy struct
//   load_size      - vectorized load width in bytes
//   store_size     - vectorized store width in bytes
//   warps_per_tile - number of warps collaborating on each tile
//   warp_size      - threads per warp (avoids dependence on platform constant names)
//   IType          - input element type
//   OType          - output element type
//
// The caller is responsible for:
//   - Checking noop before calling
//   - Computing grid dimensions (num_blocks)
//   - Passing correct stride_row/stride_col
// -----------------------------------------------------------------------
template <typename Ops, size_t load_size, size_t store_size,
          size_t warps_per_tile, size_t warp_size, typename IType,
          typename OType>
__device__ __forceinline__ void cast_transpose_tile_impl(
    const IType *__restrict__ const input,
    OType *__restrict__ const output_c,
    OType *__restrict__ const output_t,
    const float *__restrict__ const scale_ptr,
    float *__restrict__ const amax_ptr,
    float *__restrict__ const scale_inv_ptr,
    const size_t rows,
    const size_t stride_row,
    const size_t stride_col) {
  // Vectorized load/store sizes
  constexpr size_t nvec_in = load_size / sizeof(IType);
  constexpr size_t nvec_out = store_size / sizeof(OType);
  using IVec = typename Ops::template VecT<IType, nvec_in>;
  using OVecC = typename Ops::template VecT<OType, nvec_in>;
  using OVecT = typename Ops::template VecT<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr size_t bdimx = warp_size;
  constexpr size_t bdimy = warps_per_tile;
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;
  const size_t bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr size_t tile_dim_m = warp_size * nvec_out;
  constexpr size_t tile_dim_n = warp_size * nvec_in;

  // Position of tile within tensor
  const size_t num_tiles_m = rows / tile_dim_m;
  const size_t tile_id_m = bid % num_tiles_m;
  const size_t tile_id_n = bid / num_tiles_m;
  const size_t tile_row = tile_id_m * tile_dim_m;
  const size_t tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to load/store
  constexpr size_t num_iterations = warp_size / warps_per_tile;

  // FP8 factors
  const float scale = scale_ptr == nullptr ? 1.0f : *scale_ptr;
  float amax = 0.0f;

  // Load input to registers and transpose
  // Note: Each thread loads num_iterations subtiles, computes amax,
  // casts type, and transposes in registers.
  OVecT local_output_t[nvec_in][num_iterations];
#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * bdimy;
    const size_t j1 = tidx;
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = tile_row + i1 * nvec_out + i2;
      const size_t col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output_c;
      Ops::vec_load(local_input, &input[row * stride_row + col]);

      // Compute amax
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        const float v = Ops::vec_elt_as_float(local_input, static_cast<int>(j2));
        __builtin_assume(amax >= 0);
        amax = fmaxf(fabsf(v), amax);
      }

      // Cast, pack into output_c, and fill transposed register tile
      Ops::template pack_store_c_fill_t<IType, OType, nvec_in, nvec_out,
                                        num_iterations>(
          local_input, scale, local_output_c, local_output_t, iter, i2);

      Ops::vec_store_c(local_output_c, &output_c[row * stride_row + col]);
    }
  }

  // Copy from registers to shared memory to global memory
  __shared__ OVecT shared_output_t[warp_size][warp_size + 1];
#pragma unroll
  for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
      const size_t j1 = tidx;
      shared_output_t[j1][i1] = local_output_t[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidx;
      const size_t j1 = tidy + iter * bdimy;
      const size_t row = tile_row + i1 * nvec_out;
      const size_t col = tile_col + j1 * nvec_in + j2;
      Ops::vec_store_t(shared_output_t[j1][i1],
                       &output_t[col * stride_col + row]);
    }
    if (j2 + 1 < nvec_in) {
      __syncthreads();
    }
  }

  // Reduce amax over block
  if (amax_ptr != nullptr) {
    amax = Ops::template block_reduce_max<warps_per_tile>(
        amax, static_cast<int>(tidy));
    if (threadIdx.x == 0) {
      Ops::atomic_max_float(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv_ptr != nullptr) {
    Ops::write_scale_inv(scale_inv_ptr, scale);
  }
}

}  // namespace transpose
}  // namespace transformer_engine
