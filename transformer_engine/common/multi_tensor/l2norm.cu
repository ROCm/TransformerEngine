/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../utils.cuh"
#include "multi_tensor_apply.cuh"
#ifdef __HIP_PLATFORM_AMD__
#include "../util/rocm_device_utils.cuh"
#endif

namespace transformer_engine {
namespace multi_tensor_l2norm {

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T *p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T *dst, T *src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT *)dst)[dst_offset] = ((LT *)src)[src_offset];  // NOLINT(*)
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes(T *x, T val, int lanes = 1,
                        bool share_result = false) {  // lanes is intended to be <= 32.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1) 
#ifdef __HIP_PLATFORM_AMD__
    final = final + __shfl_down(final, i, THREADS_PER_WARP);
#else
    final = final + __shfl_down_sync(0xffffffff, final, i);
#endif
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
  }
  __syncthreads();
  // Avoid potential write before read race when reduce_block_into_lanes is called back to back

  return final;
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes_max_op(T *x, T val, int lanes = 1,
                               bool share_result = false) {  // lanes is intended to be <= 32.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid + i]));
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + 32]));
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
#ifdef __HIP_PLATFORM_AMD__
      final = fmaxf(fabsf(final), fabsf(__shfl_down(final, i, THREADS_PER_WARP)));
#else
      final = fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
#endif
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template <typename x_t>
struct L2NormFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             float *output, float *output_per_tensor,
                                             bool per_tensor, int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0.f;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

template <typename x_t>
struct UnscaleL2NormFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             const float *inv_scale, float *output,
                                             float *output_per_tensor, bool per_tensor,
                                             int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0.f;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]) * (*inv_scale);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]) * (*inv_scale);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

#ifdef __HIP_PLATFORM_AMD__

template <typename x_t, bool UNSCALE>
__global__ void custom_multi_tensor_l2norm_kernel(
    int chunk_size, volatile int * __restrict__ noop_gmem,
    const int64_t * __restrict__ addresses, const int * __restrict__ sizes,
    const int * __restrict__ block_to_tensor, const int * __restrict__ chunk_offsets,
    int total_chunks, const float * __restrict__ inv_scale,
    float * __restrict__ output) {
  constexpr int CILP = 8;
  const int global_chunk = blockIdx.x;
  __shared__ float s_vals[512];

  const float inv_scale_val = UNSCALE ? *inv_scale : 0.f;

  int n = 0;
  x_t *x = nullptr;
  if (global_chunk < total_chunks) {
    const int tensor_loc = block_to_tensor[global_chunk];
    const int chunk_idx = global_chunk - chunk_offsets[tensor_loc];
    n = sizes[tensor_loc];

    x = reinterpret_cast<x_t *>(static_cast<uintptr_t>(addresses[tensor_loc]));
    x += static_cast<size_t>(chunk_idx) * chunk_size;
    n -= chunk_idx * chunk_size;
  }

  float vals[CILP];
  x_t r_x[CILP];
  for (int i = 0; i < CILP; i++) {
    vals[i] = 0.f;
    r_x[i] = 0.f;
  }

  if (global_chunk < total_chunks) {
    if (n % CILP == 0 && chunk_size % CILP == 0 && is_aligned_n<CILP>(x)) {
      for (int i_start = threadIdx.x; i_start * CILP < n && i_start * CILP < chunk_size;
           i_start += blockDim.x) {
        load_store_n<CILP>(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          if constexpr (UNSCALE) {
            next *= inv_scale_val;
          }
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * CILP) {
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            if constexpr (UNSCALE) {
              next *= inv_scale_val;
            }
            vals[ii] += next * next;
          }
        }
      }
    }
  }

  float val = 0.f;
  for (int i = 0; i < CILP; i++) val += vals[i];

  float result = reduce_block_into_lanes(s_vals, val);

  if (threadIdx.x == 0) {
    if (!isfinite(result)) {
      *noop_gmem = 1;
    }
    output[global_chunk] = result;
  }
}

__global__ void custom_multi_tensor_l2norm_reduce(
    const float * __restrict__ output, float * __restrict__ ret, int total_chunks) {
  __shared__ float s_vals[512];

  float val = 0.f;
  for (int i = threadIdx.x; i < total_chunks; i += blockDim.x) {
    val += output[i];
  }

  float result = reduce_block_into_lanes(s_vals, val);

  if (threadIdx.x == 0) {
    *ret = sqrtf(result);
  }
}

void multi_tensor_l2norm_cuda_custom(int chunk_size, NVTETensor noop_flag_tensor,
                                     DType input_dtype, const int64_t *addresses,
                                     const int *sizes, const int *block_to_tensor,
                                     const int *chunk_offsets, int total_chunks,
                                     NVTETensor output_tensor,
                                     NVTETensor ret_tensor, cudaStream_t stream) {
  auto *noop_flag = convertNVTETensorCheck(noop_flag_tensor);
  auto *output = convertNVTETensorCheck(output_tensor);
  auto *ret = convertNVTETensorCheck(ret_tensor);

  if (total_chunks == 0) {
    nvte_memset(ret->data.dptr, 0, sizeof(float), stream);
    return;
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input_dtype, dtype,
      custom_multi_tensor_l2norm_kernel<dtype, false><<<total_chunks, BLOCK_SIZE, 0, stream>>>(
          chunk_size, reinterpret_cast<int *>(noop_flag->data.dptr), addresses, sizes,
          block_to_tensor, chunk_offsets, total_chunks, nullptr,
          reinterpret_cast<float *>(output->data.dptr));)
  NVTE_CHECK_CUDA(cudaGetLastError());

  custom_multi_tensor_l2norm_reduce<<<1, BLOCK_SIZE, 0, stream>>>(
      reinterpret_cast<float *>(output->data.dptr),
      reinterpret_cast<float *>(ret->data.dptr), total_chunks);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_unscale_l2norm_cuda_custom(int chunk_size, NVTETensor noop_flag_tensor,
                                             DType input_dtype, const int64_t *addresses,
                                             const int *sizes, const int *block_to_tensor,
                                             const int *chunk_offsets, int total_chunks,
                                             NVTETensor output_tensor, NVTETensor ret_tensor,
                                             NVTETensor inv_scale_tensor, cudaStream_t stream) {
  auto *noop_flag = convertNVTETensorCheck(noop_flag_tensor);
  auto *output = convertNVTETensorCheck(output_tensor);
  auto *ret = convertNVTETensorCheck(ret_tensor);
  auto *inv_scale = convertNVTETensorCheck(inv_scale_tensor);

  if (total_chunks == 0) {
    nvte_memset(ret->data.dptr, 0, sizeof(float), stream);
    return;
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input_dtype, dtype,
      custom_multi_tensor_l2norm_kernel<dtype, true><<<total_chunks, BLOCK_SIZE, 0, stream>>>(
          chunk_size, reinterpret_cast<int *>(noop_flag->data.dptr), addresses, sizes,
          block_to_tensor, chunk_offsets, total_chunks,
          reinterpret_cast<const float *>(inv_scale->data.dptr),
          reinterpret_cast<float *>(output->data.dptr));)
  NVTE_CHECK_CUDA(cudaGetLastError());

  custom_multi_tensor_l2norm_reduce<<<1, BLOCK_SIZE, 0, stream>>>(
      reinterpret_cast<float *>(output->data.dptr),
      reinterpret_cast<float *>(ret->data.dptr), total_chunks);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

#endif  // __HIP_PLATFORM_AMD__

// Probably better to template, but since we are not likely to support other norm
template <typename x_t>
struct MaxNormFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*),
                                             float *output, float *output_per_tensor,
                                             bool per_tensor, int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val = fmaxf(fabsf(val), fabsf(vals[i]));

    float final = reduce_block_into_lanes_max_op(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] = fmaxf(fabsf(output[blockIdx.x]), fabsf(final));
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

__global__ void cleanup(float *output, float *output_per_tensor, float *ret, float *ret_per_tensor,
                        bool per_tensor, int max_chunks_per_tensor) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) *ret = sqrt(final);
  }

  if (per_tensor) {
    float *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    float val = 0;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
      val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}

__global__ void cleanup_v2(float *output, float *output_per_tensor, float *ret,
                           float *ret_per_tensor, bool per_tensor, int max_chunks_per_tensor,
                           int norm_type, float alpha, float beta) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    if (norm_type == 0) {
      float final = reduce_block_into_lanes_max_op(vals, val);
      if (threadIdx.x == 0) *ret = alpha * (*ret) + beta * final;
    } else {
      float final = reduce_block_into_lanes(vals, val);
      if (threadIdx.x == 0) *ret = sqrt(alpha * (*ret) * (*ret) + beta * final);
    }
  }

  if (per_tensor) {
    float *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    if (norm_type == 0) {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val = fmaxf(fabsf(val), fabsf(output_this_tensor[i]));

      float final = reduce_block_into_lanes_max_op(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] = alpha * ret_per_tensor[blockIdx.x] + beta * final;
    } else {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val += output_this_tensor[i];

      float final = reduce_block_into_lanes(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] =
            sqrt(alpha * ret_per_tensor[blockIdx.x] * ret_per_tensor[blockIdx.x] + beta * final);
    }
  }
}

void multi_tensor_l2norm_cuda(int chunk_size, Tensor noop_flag,
                              std::vector<std::vector<Tensor *>> tensor_lists, Tensor output,
                              Tensor output_per_tensor, Tensor ret, Tensor ret_per_tensor,
                              bool per_tensor, int max_chunks_per_tensor, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, L2NormFunctor<dtype>(), stream,
          reinterpret_cast<float *>(output.data.dptr),
          per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr, per_tensor,
          max_chunks_per_tensor);)

  NVTE_CHECK_CUDA(cudaGetLastError());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  cleanup<<<per_tensor ? tensor_lists[0].size() : 1, 512, 0, stream>>>(
      reinterpret_cast<float *>(output.data.dptr),
      per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr,
      reinterpret_cast<float *>(ret.data.dptr),
      per_tensor ? reinterpret_cast<float *>(ret_per_tensor.data.dptr) : nullptr, per_tensor,
      max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_unscale_l2norm_cuda(int chunk_size, Tensor noop_flag,
                                      std::vector<std::vector<Tensor *>> tensor_lists,
                                      Tensor output, Tensor output_per_tensor, Tensor ret,
                                      Tensor ret_per_tensor, Tensor inv_scale, bool per_tensor,
                                      int max_chunks_per_tensor, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, UnscaleL2NormFunctor<dtype>(), stream,
          reinterpret_cast<float *>(inv_scale.data.dptr),
          reinterpret_cast<float *>(output.data.dptr),
          per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr, per_tensor,
          max_chunks_per_tensor);)

  NVTE_CHECK_CUDA(cudaGetLastError());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  cleanup<<<per_tensor ? tensor_lists[0].size() : 1, 512, 0, stream>>>(
      reinterpret_cast<float *>(output.data.dptr),
      per_tensor ? reinterpret_cast<float *>(output_per_tensor.data.dptr) : nullptr,
      reinterpret_cast<float *>(ret.data.dptr),
      per_tensor ? reinterpret_cast<float *>(ret_per_tensor.data.dptr) : nullptr, per_tensor,
      max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_l2norm
}  // namespace transformer_engine

#ifdef __HIP_PLATFORM_AMD__
void nvte_multi_tensor_l2norm_cuda_custom(int chunk_size, NVTETensor noop_flag,
                      const NVTEDType input_dtype, const int64_t * const addresses,
                      const int * const sizes, const int * const block_to_tensor,
                      const int * const chunk_offsets, int total_chunks,
                      NVTETensor output,
                      NVTETensor ret, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_l2norm_cuda_custom);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_l2norm_cuda_custom(
    chunk_size, noop_flag, static_cast<DType>(input_dtype), addresses, sizes, block_to_tensor,
    chunk_offsets, total_chunks, output, ret, stream);
}

void nvte_multi_tensor_unscale_l2norm_cuda_custom(
  int chunk_size, NVTETensor noop_flag, const NVTEDType input_dtype,
  const int64_t * const addresses, const int * const sizes, const int * const block_to_tensor,
  const int * const chunk_offsets, int total_chunks,
  NVTETensor output, NVTETensor ret, NVTETensor inv_scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_unscale_l2norm_cuda_custom);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_unscale_l2norm_cuda_custom(
    chunk_size, noop_flag, static_cast<DType>(input_dtype), addresses, sizes, block_to_tensor,
    chunk_offsets, total_chunks, output, ret, inv_scale, stream);
}
#endif  // __HIP_PLATFORM_AMD__

void nvte_multi_tensor_l2norm_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                   const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                   NVTETensor output, NVTETensor output_per_tensor, NVTETensor ret,
                                   NVTETensor ret_per_tensor, int per_tensor,
                                   int max_chunks_per_tensor, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_l2norm_cuda);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_l2norm_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(output_per_tensor),
      *convertNVTETensorCheck(ret), *convertNVTETensorCheck(ret_per_tensor), per_tensor,
      max_chunks_per_tensor, stream);
}

void nvte_multi_tensor_unscale_l2norm_cuda(int chunk_size, NVTETensor noop_flag,
                                           NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                           const size_t num_tensors_per_list, NVTETensor output,
                                           NVTETensor output_per_tensor, NVTETensor ret,
                                           NVTETensor ret_per_tensor, NVTETensor inv_scale,
                                           int per_tensor, int max_chunks_per_tensor,
                                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_unscale_l2norm_cuda);
  using namespace transformer_engine;

  multi_tensor_l2norm::multi_tensor_unscale_l2norm_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(output_per_tensor),
      *convertNVTETensorCheck(ret), *convertNVTETensorCheck(ret_per_tensor),
      *convertNVTETensorCheck(inv_scale), per_tensor, max_chunks_per_tensor, stream);
}
