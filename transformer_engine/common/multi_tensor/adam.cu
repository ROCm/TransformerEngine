/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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
namespace multi_tensor_adam {

#define BLOCK_SIZE 512
#define ILP 4
#define THREADS_PER_WARP 32

typedef enum {
  ADAM_MODE_0 = 0,  // L2 regularization mode
  ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;
#ifndef __HIP_PLATFORM_AMD__
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
#else
using fp8e4m3 = te_hip_fp8_e4m3;
using fp8e5m2 = te_hip_fp8_e5m2;
#endif //__HIP_PLATFORM_AMD__

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

template <bool is_fp8>
struct FP8Data {
  float scale;
  float *amax_ptr;
  float *scale_inv_ptr;
  float max;
  int warp_id;
};

template <>
struct FP8Data<false> {};

template <typename PARAM_T, typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctorMaster {
  static constexpr bool is_fp8_type = is_fp8<PARAM_T>::value;

  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5, is_fp8_type> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    FP8Data<is_fp8_type> fp8_data;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    PARAM_T *p = reinterpret_cast<PARAM_T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    FULL_T *p_master = reinterpret_cast<FULL_T *>(tl.addresses[4][tensor_loc]);
    p_master += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    if constexpr (is_fp8_type) {
      float *scale_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[0][tensor_loc]);
      fp8_data.scale = scale_ptr != nullptr ? *scale_ptr : 1;
      fp8_data.amax_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[1][tensor_loc]);
      fp8_data.scale_inv_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[2][tensor_loc]);
      fp8_data.warp_id = threadIdx.x / THREADS_PER_WARP;
      fp8_data.max = 0;
    }

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
          if constexpr (is_fp8_type) {
            __builtin_assume(fp8_data.max >= 0);
            fp8_data.max = fmaxf(fabsf(r_p[ii]), fp8_data.max);
            p[i] = static_cast<PARAM_T>(r_p[ii] * fp8_data.scale);
          } else {
            p[i] = static_cast<PARAM_T>(r_p[ii]);
          }
        }
      }
    }

    if constexpr (is_fp8_type) {
      fp8_data.max = transformer_engine::reduce_max<BLOCK_SIZE / THREADS_PER_WARP>(
          fp8_data.max, fp8_data.warp_id);
      if (threadIdx.x == 0) {
        if (fp8_data.amax_ptr != nullptr) {
          transformer_engine::atomicMaxFloat(fp8_data.amax_ptr, fp8_data.max);
        }
        if (fp8_data.scale_inv_ptr != nullptr) {
          *fp8_data.scale_inv_ptr = __frcp_rn(fp8_data.scale);
        }
      }
    }
  }
};

template <typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctorMasterParamRemainder {
  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    int16_t *p = reinterpret_cast<int16_t *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    int16_t *p_remainder = reinterpret_cast<int16_t *>(tl.addresses[4][tensor_loc]);
    p_remainder += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      union fp32_or_int162 {
        float fp32;
        int16_t int16[2];
      };
      fp32_or_int162 local_master_param[ILP];
      int16_t local_p[ILP];
      int16_t local_p_rem[ILP];
      MATH_T r_g[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);

          local_p[ii] = p[i];
          local_p_rem[ii] = p_remainder[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);

          local_p[ii] = int16_t(0);
          local_p_rem[ii] = int16_t(0);
        }
      }
// Reconstruct FP32 params
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (local_p_rem[ii] < 0) local_p[ii]--;  // Undo rounding
        local_master_param[ii].int16[1] = local_p[ii];
        local_master_param[ii].int16[0] = local_p_rem[ii];
      }

      MATH_T *r_p = reinterpret_cast<MATH_T *>(local_master_param);

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }

// Split into BF16 params (rounded-to-nearest) and remainders
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        local_p[ii] = local_master_param[ii].int16[1];
        local_p_rem[ii] = local_master_param[ii].int16[0];
        if (local_p_rem[ii] < 0) local_p[ii]++;  // Round up
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p_remainder[i] = local_p_rem[ii];
          p[i] = local_p[ii];

          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename PARAM_T, typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctor {
  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<4> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    PARAM_T *p = reinterpret_cast<PARAM_T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<PARAM_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename T, typename FULL_T, typename MOMENT_T>
struct AdamCapturableFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<4> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2, const int *step,
                                             const int bias_correction, const float epsilon,
                                             const float *lr, adamMode_t mode, const float decay,
                                             const float *inv_scale) {
    if (*noop_gmem == 1) return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T *g = reinterpret_cast<T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    T *p = reinterpret_cast<T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename T, typename FULL_T, typename MOMENT_T>
struct AdamCapturableMasterFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2, const int *step,
                                             const int bias_correction, const float epsilon,
                                             const float *lr, adamMode_t mode, const float decay,
                                             const float *inv_scale) {
    if (*noop_gmem == 1) return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T *g = reinterpret_cast<T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    T *p = reinterpret_cast<T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    FULL_T *p_master = reinterpret_cast<FULL_T *>(tl.addresses[4][tensor_loc]);
    p_master += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<T>(r_p[ii]);
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

void multi_tensor_adam_cuda(int chunk_size, Tensor noop_flag,
                            std::vector<std::vector<Tensor *>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay, cudaStream_t stream) {
  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v
  // 5 tensor lists: g, p, m, v, p_master
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 4 || num_tensor_lists == 5,
             "Expected 4 or 5 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  const auto p_in_type_te = tensor_lists[1][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == p_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(p_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    if (num_tensor_lists == 5) {
      NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
                 " has dtype=", to_string(tensor_lists[4][j]->dtype()),
                 ", but expected dtype=", to_string(DType::kFloat32));
    }
  }

  // Check if 64-bit indices are required
  bool requires_64bit_indexing = false;
  for (size_t i = 0; i < num_tensor_lists; i++) {
    for (size_t j = 0; j < num_tensors_per_list; j++) {
      if (tensor_lists[i][j]->numel() >= INT_MAX) {
        requires_64bit_indexing = true;
        break;
      }
    }
    if (requires_64bit_indexing) {
      break;
    }
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  if (requires_64bit_indexing) {
    if (num_tensor_lists == 4) {
      // g, p, m, v
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<4>(
                      (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                      AdamFunctor<p_in_type, g_in_type, float, moment_type, int64_t>(), stream,
                      beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    } else {
      // g, p, m, v, p_master
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<5>(
                      (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                      AdamFunctorMaster<p_in_type, g_in_type, float, moment_type, int64_t>(),
                      stream, beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    }
  } else {
    if (num_tensor_lists == 4) {
      // g, p, m, v
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<4>(
                      BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                      AdamFunctor<p_in_type, g_in_type, float, moment_type, int32_t>(), stream,
                      beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    } else {
      // g, p, m, v, p_master
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<5>(
                      BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                      AdamFunctorMaster<p_in_type, g_in_type, float, moment_type, int32_t>(),
                      stream, beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, Tensor noop_flag,
                                            std::vector<std::vector<Tensor *>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay,
                                            cudaStream_t stream) {
  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Check tensor list sizes
  // 5 tensor lists: g, p, m, v, p_remainder
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 5, "Expected 5 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == DType::kBFloat16, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(DType::kBFloat16));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kInt16, "Param remainder tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kInt16));
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      g_in_type_te, g_in_type,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<5>(
              (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
              AdamFunctorMasterParamRemainder<g_in_type, float, moment_type, int64_t>(), stream,
              beta1, beta2, bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode,
              weight_decay);));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_fp8_cuda(int chunk_size, Tensor noop_flag,
                                std::vector<std::vector<Tensor *>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, const DType fp8_dtype,
                                cudaStream_t stream) {
  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Check tensor list sizes
  // 8 tensor lists: g, p_fp8, m, v, p_master, scale, amax, scale_inv
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 8, "Expected 8 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(
        tensor_lists[1][j]->dtype() == fp8_dtype || tensor_lists[1][j]->dtype() == DType::kByte,
        "Param tensor ", j, " has dtype=", to_string(tensor_lists[1][j]->dtype()),
        ", but expected dtype=", to_string(fp8_dtype));
    NVTE_CHECK(tensor_lists[2][j]->dtype() == DType::kFloat32, "First moment tensor ", j,
               " has dtype=", to_string(tensor_lists[2][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[3][j]->dtype() == DType::kFloat32, "Second moment tensor ", j,
               " has dtype=", to_string(tensor_lists[3][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[5][j]->dtype() == DType::kFloat32, "Scale tensor ", j,
               " has dtype=", to_string(tensor_lists[5][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[6][j]->dtype() == DType::kFloat32, "Absmax tensor ", j,
               " has dtype=", to_string(tensor_lists[6][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[7][j]->dtype() == DType::kFloat32, "Scale-inverse tensor ", j,
               " has dtype=", to_string(tensor_lists[7][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
  }

  // Check if 64-bit indices are required
  bool requires_64bit_indexing = false;
  for (size_t i = 0; i < num_tensor_lists; i++) {
    for (size_t j = 0; j < num_tensors_per_list; j++) {
      if (tensor_lists[i][j]->numel() >= INT_MAX) {
        requires_64bit_indexing = true;
        break;
      }
    }
    if (requires_64bit_indexing) {
      break;
    }
  }

  // Launch kernel
  if (requires_64bit_indexing) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        fp8_dtype, FP8_T,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            multi_tensor_apply<5, true>(
                (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                AdamFunctorMaster<FP8_T, g_in_type, float, float, int64_t>(), stream, beta1, beta2,
                bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode, weight_decay);));
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        fp8_dtype, FP8_T,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            multi_tensor_apply<5, true>(
                BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                AdamFunctorMaster<FP8_T, g_in_type, float, float, int32_t>(), stream, beta1, beta2,
                bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode, weight_decay);));
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_capturable_cuda(int chunk_size, Tensor noop_flag,
                                       std::vector<std::vector<Tensor *>> tensor_lists, Tensor lr,
                                       const float beta1, const float beta2, const float epsilon,
                                       Tensor step, const int mode, const int bias_correction,
                                       const float weight_decay, Tensor inv_scale,
                                       cudaStream_t stream) {
  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 4, "Expected 4 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == g_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<4>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                                AdamCapturableFunctor<dtype, float, moment_type>(), stream, beta1,
                                beta2, reinterpret_cast<int *>(step.data.dptr), bias_correction,
                                epsilon, reinterpret_cast<float *>(lr.data.dptr), (adamMode_t)mode,
                                weight_decay, reinterpret_cast<float *>(inv_scale.data.dptr));))

  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, Tensor noop_flag,
                                              std::vector<std::vector<Tensor *>> tensor_lists,
                                              Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              Tensor inv_scale, cudaStream_t stream) {
  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v, p_master
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 5, "Expected 4 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == g_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<5>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                                AdamCapturableMasterFunctor<dtype, float, moment_type>(), stream,
                                beta1, beta2, reinterpret_cast<int *>(step.data.dptr),
                                bias_correction, epsilon, reinterpret_cast<float *>(lr.data.dptr),
                                (adamMode_t)mode, weight_decay,
                                reinterpret_cast<float *>(inv_scale.data.dptr));))

  NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Custom adam param-remainder kernel using device-side arrays instead of
// TensorListMetadata.  Removes the 320-block limit and avoids packing the
// metadata struct on each launch.
// ---------------------------------------------------------------------------
#ifdef __HIP_PLATFORM_AMD__

static constexpr int CILP = 8;

template <typename GRAD_T, typename MOMENT_T, adamMode_t MODE>
__global__ __launch_bounds__(BLOCK_SIZE)
void custom_adam_param_remainder_kernel(
    const int chunk_size,
    volatile int * __restrict__ noop_gmem,
    const int64_t * __restrict__ addresses,
    const int64_t * __restrict__ sizes,
    const int * __restrict__ block_to_tensor,
    const int * __restrict__ chunk_offsets,
    const int total_chunks,
    const float beta1, const float beta2,
    const float step_size, const float beta2_corr_inv,
    const float epsilon,
    const float lr, const float decay) {
  const int global_chunk = blockIdx.x;
  if (global_chunk >= total_chunks) return;

  const int tensor_loc = block_to_tensor[global_chunk];
  const int chunk_idx = global_chunk - chunk_offsets[tensor_loc];

  // addresses layout: [tensor_idx * 5 + list_idx]
  // list 0 = grads, 1 = params(int16), 2 = exp_avg, 3 = exp_avg_sq, 4 = remainders
  GRAD_T * __restrict__ g =
      reinterpret_cast<GRAD_T *>(addresses[tensor_loc * 5 + 0]);
  int16_t * __restrict__ p =
      reinterpret_cast<int16_t *>(addresses[tensor_loc * 5 + 1]);
  MOMENT_T * __restrict__ m =
      reinterpret_cast<MOMENT_T *>(addresses[tensor_loc * 5 + 2]);
  MOMENT_T * __restrict__ v =
      reinterpret_cast<MOMENT_T *>(addresses[tensor_loc * 5 + 3]);
  int16_t * __restrict__ p_remainder =
      reinterpret_cast<int16_t *>(addresses[tensor_loc * 5 + 4]);

  const int64_t elem_offset = (int64_t)chunk_idx * chunk_size;
  g += elem_offset;
  p += elem_offset;
  m += elem_offset;
  v += elem_offset;
  p_remainder += elem_offset;

  const int n_this = static_cast<int>(
      min(sizes[tensor_loc] - elem_offset, (int64_t)chunk_size));

  // Contiguous access: each thread processes CILP adjacent elements.
  // This enables 128-bit vectorized loads for 16-bit types (CILP*2 = 16 bytes)
  // and 2x 128-bit loads for float types (CILP*4 = 32 bytes).
  for (int i_start = threadIdx.x * CILP; i_start < n_this;
       i_start += blockDim.x * CILP) {
    union fp32_or_int162 {
      float fp32;
      int16_t int16[2];
    };
    GRAD_T g_raw[CILP];
    int16_t local_p[CILP];
    int16_t local_p_rem[CILP];
    MATH_T r_m[CILP];
    MATH_T r_v[CILP];

    if (i_start + CILP <= n_this && is_aligned_n<CILP>(g + i_start)) {
      // Vectorized loads: 128-bit for 16-bit types
      load_store_n<CILP>(g_raw, g, 0, i_start / CILP);
      load_store_n<CILP>(local_p, p, 0, i_start / CILP);
      load_store_n<CILP>(local_p_rem, p_remainder, 0, i_start / CILP);
      // Vectorized m/v loads
      if constexpr (sizeof(MOMENT_T) == sizeof(MATH_T)) {
        load_store_n<4>(r_m, reinterpret_cast<MATH_T *>(m), 0, i_start / 4);
        load_store_n<4>(r_m, reinterpret_cast<MATH_T *>(m), 1, i_start / 4 + 1);
        load_store_n<4>(r_v, reinterpret_cast<MATH_T *>(v), 0, i_start / 4);
        load_store_n<4>(r_v, reinterpret_cast<MATH_T *>(v), 1, i_start / 4 + 1);
      } else {
        MOMENT_T m_raw[CILP], v_raw[CILP];
        load_store_n<CILP>(m_raw, m, 0, i_start / CILP);
        load_store_n<CILP>(v_raw, v, 0, i_start / CILP);
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          r_m[ii] = static_cast<MATH_T>(m_raw[ii]);
          r_v[ii] = static_cast<MATH_T>(v_raw[ii]);
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < CILP; ii++) {
        int i = i_start + ii;
        if (i < n_this) {
          g_raw[ii] = g[i];
          local_p[ii] = p[i];
          local_p_rem[ii] = p_remainder[i];
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          g_raw[ii] = GRAD_T(0);
          local_p[ii] = int16_t(0);
          local_p_rem[ii] = int16_t(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
    }

    // Convert grads bf16 -> float
    MATH_T r_g[CILP];
#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      r_g[ii] = static_cast<MATH_T>(g_raw[ii]);
    }

    // Reconstruct FP32 master params from BF16 + int16 remainder
    fp32_or_int162 local_master_param[CILP];
#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      if (local_p_rem[ii] < 0) local_p[ii]--;
      local_master_param[ii].int16[1] = local_p[ii];
      local_master_param[ii].int16[0] = local_p_rem[ii];
    }

    MATH_T *r_p = reinterpret_cast<MATH_T *>(local_master_param);

#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      if (MODE == ADAM_MODE_0) {  // L2
        r_g[ii] += decay * r_p[ii];
      }
      r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
      r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
      MATH_T denom = sqrtf(r_v[ii] * beta2_corr_inv) + epsilon;
      if (MODE == ADAM_MODE_0) {  // L2
        r_p[ii] -= step_size * (r_m[ii] / denom);
      } else {  // weight decay
        r_p[ii] = r_p[ii] - step_size * (r_m[ii] / denom) - lr * decay * r_p[ii];
      }
    }

    // Split into BF16 params (rounded-to-nearest) and remainders
#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      local_p[ii] = local_master_param[ii].int16[1];
      local_p_rem[ii] = local_master_param[ii].int16[0];
      if (local_p_rem[ii] < 0) local_p[ii]++;  // Round up
    }

    // Store
    if (i_start + CILP <= n_this && is_aligned_n<CILP>(p + i_start)) {
      load_store_n<CILP>(p, local_p, i_start / CILP, 0);
      load_store_n<CILP>(p_remainder, local_p_rem, i_start / CILP, 0);
      // Vectorized m/v stores
      if constexpr (sizeof(MOMENT_T) == sizeof(MATH_T)) {
        load_store_n<4>(reinterpret_cast<MATH_T *>(m), r_m, i_start / 4, 0);
        load_store_n<4>(reinterpret_cast<MATH_T *>(m), r_m, i_start / 4 + 1, 1);
        load_store_n<4>(reinterpret_cast<MATH_T *>(v), r_v, i_start / 4, 0);
        load_store_n<4>(reinterpret_cast<MATH_T *>(v), r_v, i_start / 4 + 1, 1);
      } else {
        MOMENT_T m_out[CILP], v_out[CILP];
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          m_out[ii] = static_cast<MOMENT_T>(r_m[ii]);
          v_out[ii] = static_cast<MOMENT_T>(r_v[ii]);
        }
        load_store_n<CILP>(m, m_out, i_start / CILP, 0);
        load_store_n<CILP>(v, v_out, i_start / CILP, 0);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < CILP; ii++) {
        int i = i_start + ii;
        if (i < n_this) {
          p[i] = local_p[ii];
          p_remainder[i] = local_p_rem[ii];
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
}

void multi_tensor_adam_param_remainder_cuda_custom(
    int chunk_size, Tensor noop_flag, DType grad_dtype, DType moment_dtype,
    int64_t *addresses, int64_t *sizes, int *block_to_tensor, int *chunk_offsets,
    int total_chunks,
    const float lr, const float beta1, const float beta2, const float epsilon,
    const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  const float step_size = lr / bias_correction1;
  const float beta2_corr_inv = 1.0f / bias_correction2;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      grad_dtype, grad_type,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_dtype, moment_type,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(mode == ADAM_MODE_0, IS_MODE_0,
              constexpr adamMode_t ADAM_MODE = IS_MODE_0 ? ADAM_MODE_0 : ADAM_MODE_1;
              custom_adam_param_remainder_kernel<grad_type, moment_type, ADAM_MODE>
                  <<<total_chunks, BLOCK_SIZE, 0, stream>>>(
                      chunk_size, reinterpret_cast<int *>(noop_flag.data.dptr), addresses, sizes,
                      block_to_tensor, chunk_offsets, total_chunks, beta1, beta2, step_size,
                      beta2_corr_inv, epsilon, lr, weight_decay);
          );););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Custom adam kernel (4-list: g, p, m, v) or (5-list: g, p, m, v, p_master)
// using device-side arrays.
// ---------------------------------------------------------------------------
template <typename GRAD_T, typename PARAM_T, typename MOMENT_T, adamMode_t MODE,
          bool HAS_MASTER = false>
__global__ __launch_bounds__(BLOCK_SIZE)
void custom_adam_kernel(
    const int chunk_size,
    volatile int * __restrict__ noop_gmem,
    const int64_t * __restrict__ addresses,
    const int64_t * __restrict__ sizes,
    const int * __restrict__ block_to_tensor,
    const int * __restrict__ chunk_offsets,
    const int total_chunks,
    const float beta1, const float beta2,
    const float step_size, const float beta2_corr_inv,
    const float epsilon,
    const float lr, const float decay) {
  constexpr int kDepth = HAS_MASTER ? 5 : 4;
  const int global_chunk = blockIdx.x;
  if (global_chunk >= total_chunks) return;

  const int tensor_loc = block_to_tensor[global_chunk];
  const int chunk_idx = global_chunk - chunk_offsets[tensor_loc];

  GRAD_T * __restrict__ g =
      reinterpret_cast<GRAD_T *>(addresses[tensor_loc * kDepth + 0]);
  PARAM_T * __restrict__ p =
      reinterpret_cast<PARAM_T *>(addresses[tensor_loc * kDepth + 1]);
  MOMENT_T * __restrict__ m =
      reinterpret_cast<MOMENT_T *>(addresses[tensor_loc * kDepth + 2]);
  MOMENT_T * __restrict__ v =
      reinterpret_cast<MOMENT_T *>(addresses[tensor_loc * kDepth + 3]);
  float * __restrict__ p_master = nullptr;
  if constexpr (HAS_MASTER) {
    p_master = reinterpret_cast<float *>(addresses[tensor_loc * kDepth + 4]);
  }

  const int64_t elem_offset = (int64_t)chunk_idx * chunk_size;
  g += elem_offset;
  p += elem_offset;
  m += elem_offset;
  v += elem_offset;
  if constexpr (HAS_MASTER) {
    p_master += elem_offset;
  }

  const int n_this = static_cast<int>(
      min(sizes[tensor_loc] - elem_offset, (int64_t)chunk_size));

  for (int i_start = threadIdx.x * CILP; i_start < n_this;
       i_start += blockDim.x * CILP) {
    GRAD_T g_raw[CILP];
    MATH_T r_p[CILP];
    MATH_T r_m[CILP];
    MATH_T r_v[CILP];

    if (i_start + CILP <= n_this && is_aligned_n<CILP>(g + i_start)) {
      // Vectorized loads
      if constexpr (sizeof(GRAD_T) == 2) {
        load_store_n<CILP>(g_raw, g, 0, i_start / CILP);
      } else {
        load_store_n<4>(g_raw, g, 0, i_start / 4);
        load_store_n<4>(g_raw, g, 1, i_start / 4 + 1);
      }
      if constexpr (HAS_MASTER) {
        // Load from FP32 master params
        load_store_n<4>(r_p, p_master, 0, i_start / 4);
        load_store_n<4>(r_p, p_master, 1, i_start / 4 + 1);
      } else {
        PARAM_T p_raw[CILP];
        if constexpr (sizeof(PARAM_T) == 2) {
          load_store_n<CILP>(p_raw, p, 0, i_start / CILP);
        } else {
          load_store_n<4>(p_raw, p, 0, i_start / 4);
          load_store_n<4>(p_raw, p, 1, i_start / 4 + 1);
        }
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          r_p[ii] = static_cast<MATH_T>(p_raw[ii]);
        }
      }
      // Vectorized m/v loads
      if constexpr (sizeof(MOMENT_T) == sizeof(MATH_T)) {
        load_store_n<4>(r_m, reinterpret_cast<MATH_T *>(m), 0, i_start / 4);
        load_store_n<4>(r_m, reinterpret_cast<MATH_T *>(m), 1, i_start / 4 + 1);
        load_store_n<4>(r_v, reinterpret_cast<MATH_T *>(v), 0, i_start / 4);
        load_store_n<4>(r_v, reinterpret_cast<MATH_T *>(v), 1, i_start / 4 + 1);
      } else {
        MOMENT_T m_raw[CILP], v_raw[CILP];
        load_store_n<CILP>(m_raw, m, 0, i_start / CILP);
        load_store_n<CILP>(v_raw, v, 0, i_start / CILP);
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          r_m[ii] = static_cast<MATH_T>(m_raw[ii]);
          r_v[ii] = static_cast<MATH_T>(v_raw[ii]);
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < CILP; ii++) {
        int i = i_start + ii;
        if (i < n_this) {
          g_raw[ii] = g[i];
          if constexpr (HAS_MASTER) {
            r_p[ii] = p_master[i];
          } else {
            r_p[ii] = static_cast<MATH_T>(p[i]);
          }
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          g_raw[ii] = GRAD_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
    }

    MATH_T r_g[CILP];
#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      r_g[ii] = static_cast<MATH_T>(g_raw[ii]);
    }

#pragma unroll
    for (int ii = 0; ii < CILP; ii++) {
      if (MODE == ADAM_MODE_0) {  // L2
        r_g[ii] += decay * r_p[ii];
      }
      r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
      r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
      MATH_T denom = sqrtf(r_v[ii] * beta2_corr_inv) + epsilon;
      if (MODE == ADAM_MODE_0) {  // L2
        r_p[ii] -= step_size * (r_m[ii] / denom);
      } else {  // weight decay
        r_p[ii] = r_p[ii] - step_size * (r_m[ii] / denom) - lr * decay * r_p[ii];
      }
    }

    // Store
    if (i_start + CILP <= n_this && is_aligned_n<CILP>(p + i_start)) {
      // Write p (PARAM_T)
      PARAM_T p_out[CILP];
#pragma unroll
      for (int ii = 0; ii < CILP; ii++) {
        p_out[ii] = static_cast<PARAM_T>(r_p[ii]);
      }
      if constexpr (sizeof(PARAM_T) == 2) {
        load_store_n<CILP>(p, p_out, i_start / CILP, 0);
      } else {
        load_store_n<4>(p, p_out, i_start / 4, 0);
        load_store_n<4>(p, p_out, i_start / 4 + 1, 1);
      }
      if constexpr (HAS_MASTER) {
        load_store_n<4>(p_master, r_p, i_start / 4, 0);
        load_store_n<4>(p_master, r_p, i_start / 4 + 1, 1);
      }
      // Vectorized m/v stores
      if constexpr (sizeof(MOMENT_T) == sizeof(MATH_T)) {
        load_store_n<4>(reinterpret_cast<MATH_T *>(m), r_m, i_start / 4, 0);
        load_store_n<4>(reinterpret_cast<MATH_T *>(m), r_m, i_start / 4 + 1, 1);
        load_store_n<4>(reinterpret_cast<MATH_T *>(v), r_v, i_start / 4, 0);
        load_store_n<4>(reinterpret_cast<MATH_T *>(v), r_v, i_start / 4 + 1, 1);
      } else {
        MOMENT_T m_out[CILP], v_out[CILP];
#pragma unroll
        for (int ii = 0; ii < CILP; ii++) {
          m_out[ii] = static_cast<MOMENT_T>(r_m[ii]);
          v_out[ii] = static_cast<MOMENT_T>(r_v[ii]);
        }
        load_store_n<CILP>(m, m_out, i_start / CILP, 0);
        load_store_n<CILP>(v, v_out, i_start / CILP, 0);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < CILP; ii++) {
        int i = i_start + ii;
        if (i < n_this) {
          p[i] = static_cast<PARAM_T>(r_p[ii]);
          if constexpr (HAS_MASTER) {
            p_master[i] = r_p[ii];
          }
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
}

void multi_tensor_adam_cuda_custom(
    int chunk_size, Tensor noop_flag, DType grad_dtype, DType param_dtype,
    DType moment_dtype, int64_t *addresses, int64_t *sizes, int *block_to_tensor,
    int *chunk_offsets, int total_chunks, bool has_master,
    const float lr, const float beta1, const float beta2, const float epsilon,
    const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  const float step_size = lr / bias_correction1;
  const float beta2_corr_inv = 1.0f / bias_correction2;

#define LAUNCH_CUSTOM_ADAM(g_type, p_type, m_type, adam_mode, master_flag) \
  custom_adam_kernel<g_type, p_type, m_type, adam_mode, master_flag> \
      <<<total_chunks, BLOCK_SIZE, 0, stream>>>( \
          chunk_size, reinterpret_cast<int *>(noop_flag.data.dptr), \
          addresses, sizes, block_to_tensor, chunk_offsets, total_chunks, \
          beta1, beta2, step_size, beta2_corr_inv, epsilon, lr, \
          weight_decay)

  if (has_master) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        param_dtype, p_type,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            grad_dtype, g_type,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                moment_dtype, m_type,
                TRANSFORMER_ENGINE_SWITCH_CONDITION(mode == ADAM_MODE_0, IS_MODE_0,
                    constexpr adamMode_t ADAM_MODE = IS_MODE_0 ? ADAM_MODE_0 : ADAM_MODE_1;
                    LAUNCH_CUSTOM_ADAM(g_type, p_type, m_type, ADAM_MODE, true);
                ););););  // NOLINT(*)
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        param_dtype, p_type,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            grad_dtype, g_type,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                moment_dtype, m_type,
                TRANSFORMER_ENGINE_SWITCH_CONDITION(mode == ADAM_MODE_0, IS_MODE_0,
                    constexpr adamMode_t ADAM_MODE = IS_MODE_0 ? ADAM_MODE_0 : ADAM_MODE_1;
                    LAUNCH_CUSTOM_ADAM(g_type, p_type, m_type, ADAM_MODE, false);
                ););););  // NOLINT(*)
  }

#undef LAUNCH_CUSTOM_ADAM
  NVTE_CHECK_CUDA(cudaGetLastError());
}

#endif  // __HIP_PLATFORM_AMD__

}  // namespace multi_tensor_adam
}  // namespace transformer_engine

void nvte_multi_tensor_adam_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                 const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                 const float lr, const float beta1, const float beta2,
                                 const float epsilon, const int step, const int mode,
                                 const int bias_correction, const float weight_decay,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, stream);
}

void nvte_multi_tensor_adam_param_remainder_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_param_remainder_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_param_remainder_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, stream);
}

void nvte_multi_tensor_adam_fp8_cuda(int chunk_size, NVTETensor noop_flag,
                                     NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                     const size_t num_tensors_per_list, const float lr,
                                     const float beta1, const float beta2, const float epsilon,
                                     const int step, const int mode, const int bias_correction,
                                     const float weight_decay, const NVTEDType fp8_dtype,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_fp8_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_fp8_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, static_cast<DType>(fp8_dtype), stream);
}

void nvte_multi_tensor_adam_capturable_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_capturable_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_capturable_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(lr), beta1, beta2, epsilon, *convertNVTETensorCheck(step), mode,
      bias_correction, weight_decay, *convertNVTETensorCheck(inv_scale), stream);
}

void nvte_multi_tensor_adam_capturable_master_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_capturable_master_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_capturable_master_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(lr), beta1, beta2, epsilon, *convertNVTETensorCheck(step), mode,
      bias_correction, weight_decay, *convertNVTETensorCheck(inv_scale), stream);
}

#ifdef __HIP_PLATFORM_AMD__
void nvte_multi_tensor_adam_param_remainder_cuda_custom(
    int chunk_size, NVTETensor noop_flag, NVTEDType grad_dtype, NVTEDType moment_dtype,
    int64_t *addresses, int64_t *sizes, int *block_to_tensor, int *chunk_offsets,
    int total_chunks,
    const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_param_remainder_cuda_custom);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_param_remainder_cuda_custom(
      chunk_size, *convertNVTETensorCheck(noop_flag), static_cast<DType>(grad_dtype),
      static_cast<DType>(moment_dtype),
      addresses, sizes, block_to_tensor, chunk_offsets, total_chunks,
      lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay, stream);
}

void nvte_multi_tensor_adam_cuda_custom(
    int chunk_size, NVTETensor noop_flag, NVTEDType grad_dtype, NVTEDType param_dtype,
    NVTEDType moment_dtype, int64_t *addresses, int64_t *sizes,
    int *block_to_tensor, int *chunk_offsets, int total_chunks, int has_master,
    const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_cuda_custom);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_cuda_custom(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      static_cast<DType>(grad_dtype), static_cast<DType>(param_dtype),
      static_cast<DType>(moment_dtype),
      addresses, sizes, block_to_tensor, chunk_offsets, total_chunks,
      has_master != 0,
      lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay, stream);
}
#endif  // __HIP_PLATFORM_AMD__
