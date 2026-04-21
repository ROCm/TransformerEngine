/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated_mxfp8.cuh
 *  \brief CUDA kernels to cast to MXFP8 with gated activations.
 */

#ifndef TRANSFORMER_ENGINE_GATED_MXFP8_CUH_
#define TRANSFORMER_ENGINE_GATED_MXFP8_CUH_

#include <cuda.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cudaTypedefs.h>
#endif //#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "swizzle.cuh"

#ifdef __HIP_PLATFORM_AMD__
#include "./rocm_vectorized_2d.cuh"
#endif

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace gated_kernel {
#ifdef __HIP_PLATFORM_AMD__
#include "rocm_gated_mxfp8.cuh"

// ---------------------------------------------------------------------------
// TDM (Tensor Data Mover) gated MXFP8 kernel for gfx1250
// ---------------------------------------------------------------------------
#if defined(__gfx1250__)
}  // temporarily close namespace gated_kernel
}  // temporarily close namespace mxfp8
}  // temporarily close namespace dispatch
#include "../../util/tdm.cuh"
namespace dispatch {
namespace mxfp8 {
namespace gated_kernel {

namespace gated_mxfp8_tdm_kernel {

constexpr size_t TDM_GATED_MXFP8_CHUNK_DIM_Y   = 64;
constexpr size_t TDM_GATED_MXFP8_CHUNK_DIM_X   = 64;
constexpr size_t TDM_GATED_MXFP8_THREADS       = 256;
constexpr size_t TDM_GATED_MXFP8_BUFFERS_NUM   = 2;
constexpr size_t TDM_GATED_MXFP8_BUFFER_DIM_Y  = 32;
constexpr size_t TDM_GATED_MXFP8_BUFFER_DIM_X  = TDM_GATED_MXFP8_CHUNK_DIM_X;   // 64
constexpr size_t TDM_GATED_MXFP8_SHMEM_DIM_Y   = TDM_GATED_MXFP8_BUFFER_DIM_Y;  // 32
constexpr size_t TDM_GATED_MXFP8_SHMEM_DIM_X   = TDM_GATED_MXFP8_BUFFER_DIM_X;  // 64
constexpr size_t TDM_GATED_MXFP8_ITERATIONS    = TDM_GATED_MXFP8_CHUNK_DIM_Y / TDM_GATED_MXFP8_BUFFER_DIM_Y;  // 2

constexpr size_t TDM_GATED_MXFP8_ELEMS_PER_THREAD = 8;
constexpr size_t TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_ROWWISE =
    TDM_GATED_MXFP8_CHUNK_DIM_X / TDM_GATED_MXFP8_ELEMS_PER_THREAD;  // 8
constexpr size_t TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_ROWWISE =
    TDM_GATED_MXFP8_THREADS / TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_ROWWISE;  // 32
constexpr size_t TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_COLWISE = TDM_GATED_MXFP8_CHUNK_DIM_X;  // 64
constexpr size_t TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE =
    TDM_GATED_MXFP8_THREADS / TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_COLWISE;  // 4
constexpr size_t TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE =
    TDM_GATED_MXFP8_BUFFER_DIM_Y / TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE;  // 8

constexpr size_t TDM_GATED_MXFP8_SUBWARP_WIDTH =
    DIVUP(32, TDM_GATED_MXFP8_ELEMS_PER_THREAD);  // 4

// IS_BWD=false: loads act + gate (2 loads/iter)
// IS_BWD=true : loads grad + act + gate (1 + 2 = 3 loads/iter, separate strides)
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          bool ROWWISE_SCALING, bool COLWISE_SCALING>
__global__ void __launch_bounds__(TDM_GATED_MXFP8_THREADS)
    quantize_gated_mxfp8_tdm_kernel(
        const IType *__restrict__ grad_ptr,
        const IType *__restrict__ act_ptr,
        const IType *__restrict__ gate_ptr,
        OType *__restrict__ out_act_rowwise_ptr,
        OType *__restrict__ out_gate_rowwise_ptr,
        OType *__restrict__ out_act_colwise_ptr,
        OType *__restrict__ out_gate_colwise_ptr,
        e8m0_t *const scales_rowwise,
        e8m0_t *const scales_colwise,
        const size_t rows,
        const size_t cols,
        const size_t input_stride,
        const size_t grad_stride,
        const size_t output_stride,
        const size_t scale_stride_rowwise,
        const size_t scale_stride_colwise,
        const ParamOP p) {
  using namespace transformer_engine::tdm;

  constexpr size_t SCALE_DIM_X = 32;
  constexpr size_t SCALE_DIM_Y = 32;

  const size_t chunk_offset_Y = blockIdx.y * TDM_GATED_MXFP8_CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * TDM_GATED_MXFP8_CHUNK_DIM_X;

  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t scales_rowwise_chunk_offset_X = blockIdx.x * (TDM_GATED_MXFP8_CHUNK_DIM_X / SCALE_DIM_X);
  const size_t scales_colwise_chunk_offset_Y = blockIdx.y * (TDM_GATED_MXFP8_CHUNK_DIM_Y / SCALE_DIM_Y);
  const size_t scales_colwise_chunk_offset_X = blockIdx.x * TDM_GATED_MXFP8_CHUNK_DIM_X;

  const int tid_rowwise_Y = threadIdx.x / TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_ROWWISE;
  const int thread_offset_X_rowwise = tid_rowwise_X * TDM_GATED_MXFP8_ELEMS_PER_THREAD;

  const int tid_colwise_Y = threadIdx.x / TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % TDM_GATED_MXFP8_THREADS_PER_CHUNK_X_COLWISE;

  constexpr uint32_t input_data_size  = get_data_size_from_bits(sizeof(IType) * 8);
  constexpr uint32_t output_data_size = get_data_size_from_bits(sizeof(OType) * 8);

  const uint32_t tensor_w_in  = static_cast<uint32_t>(cols);
  const uint32_t tensor_h     = static_cast<uint32_t>(rows);
  const uint32_t stride_in    = static_cast<uint32_t>(input_stride);
  const uint32_t stride_grad  = static_cast<uint32_t>(grad_stride);
  const uint32_t stride_out   = static_cast<uint32_t>(output_stride);
  const uint32_t tensor_w_out = static_cast<uint32_t>(cols);

  // Shared memory buffers: double-buffered
  __shared__ alignas(128) IType in_act_sh [TDM_GATED_MXFP8_BUFFERS_NUM][TDM_GATED_MXFP8_SHMEM_DIM_Y][TDM_GATED_MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) IType in_gate_sh[TDM_GATED_MXFP8_BUFFERS_NUM][TDM_GATED_MXFP8_SHMEM_DIM_Y][TDM_GATED_MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) IType in_grad_sh[IS_BWD ? TDM_GATED_MXFP8_BUFFERS_NUM : 1][TDM_GATED_MXFP8_SHMEM_DIM_Y][TDM_GATED_MXFP8_SHMEM_DIM_X];

  // Output shmem: for colwise store via TDM, for rowwise direct global writes
  __shared__ alignas(128) OType out_act_colwise_sh [COLWISE_SCALING ? TDM_GATED_MXFP8_BUFFERS_NUM : 1][TDM_GATED_MXFP8_SHMEM_DIM_Y][TDM_GATED_MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) OType out_gate_colwise_sh[COLWISE_SCALING && IS_BWD ? TDM_GATED_MXFP8_BUFFERS_NUM : 1][TDM_GATED_MXFP8_SHMEM_DIM_Y][TDM_GATED_MXFP8_SHMEM_DIM_X];

  // For colwise cross-thread Y reduction
  __shared__ float stage_amax_sh[TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE][TDM_GATED_MXFP8_CHUNK_DIM_X];

  // --- Prologue: prefetch iteration 0 into buffer 0 ---
  {
    const uint32_t cy = static_cast<uint32_t>(chunk_offset_Y);
    const uint32_t cx = static_cast<uint32_t>(chunk_offset_X);
    if constexpr (IS_BWD) {
      // grad has stride_grad, act/gate have stride_in -- cannot use x3
      copy_2d_to_shared(
          &in_grad_sh[0][0][0], grad_ptr, cx, cy,
          TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_grad, input_data_size);
      copy_2d_to_shared_x2(
          &in_act_sh [0][0][0], act_ptr,  cx, cy,
          &in_gate_sh[0][0][0], gate_ptr, cx, cy,
          TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_in, input_data_size);
    } else {
      copy_2d_to_shared_x2(
          &in_act_sh [0][0][0], act_ptr,  cx, cy,
          &in_gate_sh[0][0][0], gate_ptr, cx, cy,
          TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_in, input_data_size);
    }
  }

  // --- Main loop ---
#pragma unroll
  for (int iter = 0; iter < (int)TDM_GATED_MXFP8_ITERATIONS; ++iter) {
    const int buff      = iter % TDM_GATED_MXFP8_BUFFERS_NUM;
    const int next_iter = iter + 1;
    const int next_buff = next_iter % TDM_GATED_MXFP8_BUFFERS_NUM;

    // Prefetch next iteration if available
    if (next_iter < (int)TDM_GATED_MXFP8_ITERATIONS) {
      const uint32_t ncy = static_cast<uint32_t>(chunk_offset_Y + next_iter * TDM_GATED_MXFP8_BUFFER_DIM_Y);
      const uint32_t ncx = static_cast<uint32_t>(chunk_offset_X);
      if constexpr (IS_BWD) {
        copy_2d_to_shared(
            &in_grad_sh[next_buff][0][0], grad_ptr, ncx, ncy,
            TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_grad, input_data_size);
        copy_2d_to_shared_x2(
            &in_act_sh [next_buff][0][0], act_ptr,  ncx, ncy,
            &in_gate_sh[next_buff][0][0], gate_ptr, ncx, ncy,
            TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_in, input_data_size);
      } else {
        copy_2d_to_shared_x2(
            &in_act_sh [next_buff][0][0], act_ptr,  ncx, ncy,
            &in_gate_sh[next_buff][0][0], gate_ptr, ncx, ncy,
            TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_in, input_data_size);
      }
    }

    // Wait for current buffer's loads to complete.
    // IS_BWD: 3 TDM ops/iter (1 grad + 2 act/gate), FWD: 2 TDM ops/iter.
    // When next prefetch is live, keep those N ops in-flight; otherwise drain all.
    if (is_tdm_wave()) {
      if (next_iter < (int)TDM_GATED_MXFP8_ITERATIONS) {
        if constexpr (IS_BWD) {
          wait_tensorcnt_3();
        } else {
          wait_tensorcnt_2();
        }
      } else {
        wait_tensorcnt_0();
      }
    }
    __syncthreads();

    // --- Compute: rowwise path (direct global writes) ---
    if constexpr (ROWWISE_SCALING) {
      const size_t row = chunk_offset_Y + iter * TDM_GATED_MXFP8_BUFFER_DIM_Y + tid_rowwise_Y;
      const size_t col_start = chunk_offset_X + thread_offset_X_rowwise;
      const bool row_valid = (row < rows);
      const bool col_valid = (col_start < cols);

      const int shmem_base_y = tid_rowwise_Y;
      const int shmem_base_x = thread_offset_X_rowwise;

      // Load from shmem
      float computed_act[TDM_GATED_MXFP8_ELEMS_PER_THREAD];
      float computed_gate[TDM_GATED_MXFP8_ELEMS_PER_THREAD];
      float act_amax = 0.0f;
      float gate_amax = 0.0f;

#pragma unroll
      for (int j = 0; j < (int)TDM_GATED_MXFP8_ELEMS_PER_THREAD; ++j) {
        float act_elt  = static_cast<float>(in_act_sh [buff][shmem_base_y][shmem_base_x + j]);
        float gate_elt = static_cast<float>(in_gate_sh[buff][shmem_base_y][shmem_base_x + j]);
        float grad_elt = 0.0f;
        if constexpr (IS_BWD) {
          grad_elt = static_cast<float>(in_grad_sh[buff][shmem_base_y][shmem_base_x + j]);
        }

        compute_gated_activation<IS_BWD, ParamOP, ActOP, DActOP, IType>(
            act_elt, gate_elt, grad_elt, p, computed_act[j], computed_gate[j]);

        if (row_valid && col_valid && (col_start + j < cols)) {
          act_amax = fmaxf(act_amax, fabsf(computed_act[j]));
          if constexpr (IS_BWD) {
            gate_amax = fmaxf(gate_amax, fabsf(computed_gate[j]));
          }
        }
      }

      // Act rowwise quantization
      {
        __builtin_assume(act_amax >= 0);
        const float scale_amax = subwarp_reduce_max_broadcast<TDM_GATED_MXFP8_SUBWARP_WIDTH>(act_amax);
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp(biased_exp);

        if (row_valid && col_valid) {
#pragma unroll
          for (int j = 0; j < (int)TDM_GATED_MXFP8_ELEMS_PER_THREAD; ++j) {
            if (col_start + j < cols) {
              out_act_rowwise_ptr[row * output_stride + col_start + j] =
                  static_cast<OType>(computed_act[j] * scale_inv);
            }
          }
        }

        constexpr size_t THREADS_PER_SCALE_X = DIVUP(SCALE_DIM_X, TDM_GATED_MXFP8_ELEMS_PER_THREAD);
        if (tid_rowwise_X % (int)THREADS_PER_SCALE_X == 0 && row_valid && col_valid) {
          const size_t scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X;
          scales_rowwise[scale_idx] = biased_exp;
        }
      }

      // Gate rowwise quantization (BWD only)
      if constexpr (IS_BWD) {
        __builtin_assume(gate_amax >= 0);
        const float scale_amax = subwarp_reduce_max_broadcast<TDM_GATED_MXFP8_SUBWARP_WIDTH>(gate_amax);
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp(biased_exp);

        if (row_valid && col_valid) {
#pragma unroll
          for (int j = 0; j < (int)TDM_GATED_MXFP8_ELEMS_PER_THREAD; ++j) {
            if (col_start + j < cols) {
              out_gate_rowwise_ptr[row * output_stride + col_start + j] =
                  static_cast<OType>(computed_gate[j] * scale_inv);
            }
          }
        }

        constexpr size_t THREADS_PER_SCALE_X = DIVUP(SCALE_DIM_X, TDM_GATED_MXFP8_ELEMS_PER_THREAD);
        if (tid_rowwise_X % (int)THREADS_PER_SCALE_X == 0 && row_valid && col_valid) {
          const size_t scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X +
              DIVUP(cols, SCALE_DIM_X);
          scales_rowwise[scale_idx] = biased_exp;
        }
      }
    }

    // --- Compute: colwise path ---
    if constexpr (COLWISE_SCALING) {
      const bool col_out_of_bounds = (chunk_offset_X + tid_colwise_X >= cols);
      const size_t row_base = chunk_offset_Y + iter * TDM_GATED_MXFP8_BUFFER_DIM_Y;
      const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + iter;

      float after_dact_reg[TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE];
      float after_dgate_reg[TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE];

      float thread_Y_mx_block_amax      = 0.0f;
      float thread_Y_mx_block_amax_gate = 0.0f;

      // Compute activation and accumulate column amax
      for (int stage = 0; stage < (int)TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE; ++stage) {
        const int stage_offset_Y = stage * TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE;
        const int shmem_y = tid_colwise_Y + stage_offset_Y;

        float act_elt  = static_cast<float>(in_act_sh [buff][shmem_y][tid_colwise_X]);
        float gate_elt = static_cast<float>(in_gate_sh[buff][shmem_y][tid_colwise_X]);
        float grad_elt = 0.0f;
        if constexpr (IS_BWD) {
          grad_elt = static_cast<float>(in_grad_sh[buff][shmem_y][tid_colwise_X]);
        }

        compute_gated_activation<IS_BWD, ParamOP, ActOP, DActOP, IType>(
            act_elt, gate_elt, grad_elt, p, after_dact_reg[stage], after_dgate_reg[stage]);

        // Cache in shmem for potential rowwise reuse (IS_CACHED_ACT_OP)
        if constexpr (ROWWISE_SCALING) {
          in_act_sh [buff][shmem_y][tid_colwise_X] = static_cast<IType>(after_dact_reg[stage]);
          if constexpr (IS_BWD) {
            in_gate_sh[buff][shmem_y][tid_colwise_X] = static_cast<IType>(after_dgate_reg[stage]);
          }
        }

        __builtin_assume(thread_Y_mx_block_amax >= 0);
        thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, fabsf(after_dact_reg[stage]));
        if constexpr (IS_BWD) {
          __builtin_assume(thread_Y_mx_block_amax_gate >= 0);
          thread_Y_mx_block_amax_gate =
              fmaxf(thread_Y_mx_block_amax_gate, fabsf(after_dgate_reg[stage]));
        }
      }

      const bool row_out_of_bounds = (row_base >= rows);
      const bool out_of_bounds     = (col_out_of_bounds || row_out_of_bounds);

      // Gate colwise reduction and quantization (BWD only, do first to reuse stage_amax_sh)
      if constexpr (IS_BWD) {
        if (tid_colwise_Y > 0) {
          stage_amax_sh[tid_colwise_Y][tid_colwise_X] = thread_Y_mx_block_amax_gate;
        }
        __syncthreads();
        if (tid_colwise_Y == 0) {
#pragma unroll
          for (int y = 1; y < (int)TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE; ++y) {
            thread_Y_mx_block_amax_gate =
                fmaxf(thread_Y_mx_block_amax_gate, stage_amax_sh[y][tid_colwise_X]);
          }
          stage_amax_sh[0][tid_colwise_X] = thread_Y_mx_block_amax_gate;
        }
        __syncthreads();

        const float mx_block_Y_amax = stage_amax_sh[0][tid_colwise_X];
        __builtin_assume(mx_block_Y_amax >= 0);

        const e8m0_t biased_exponent =
            ptx::float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal = ptx::exp2f_rcp(biased_exponent);

        if ((tid_colwise_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X + cols;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < (int)TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE; ++stage) {
          const int stage_offset_Y = stage * TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE;
          out_gate_colwise_sh[buff][tid_colwise_Y + stage_offset_Y][tid_colwise_X] =
              static_cast<OType>(scale_reciprocal * after_dgate_reg[stage]);
        }
      }

      // Act colwise reduction and quantization
      {
        if (tid_colwise_Y > 0) {
          stage_amax_sh[tid_colwise_Y][tid_colwise_X] = thread_Y_mx_block_amax;
        }
        __syncthreads();
        if (tid_colwise_Y == 0) {
#pragma unroll
          for (int y = 1; y < (int)TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE; ++y) {
            thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, stage_amax_sh[y][tid_colwise_X]);
          }
          stage_amax_sh[0][tid_colwise_X] = thread_Y_mx_block_amax;
        }
        __syncthreads();

        const float mx_block_Y_amax = stage_amax_sh[0][tid_colwise_X];
        __builtin_assume(mx_block_Y_amax >= 0);

        const e8m0_t biased_exponent =
            ptx::float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal = ptx::exp2f_rcp(biased_exponent);

        if ((tid_colwise_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < (int)TDM_GATED_MXFP8_BUFFER_STAGES_NUM_COLWISE; ++stage) {
          const int stage_offset_Y = stage * TDM_GATED_MXFP8_THREADS_PER_CHUNK_Y_COLWISE;
          out_act_colwise_sh[buff][tid_colwise_Y + stage_offset_Y][tid_colwise_X] =
              static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
        }
      }
    }

    // Ensure all threads finished writing shmem before TDM reads it
    __syncthreads();

    // TDM store: colwise output shmem -> global
    if constexpr (COLWISE_SCALING) {
      const uint32_t scy = static_cast<uint32_t>(chunk_offset_Y + iter * TDM_GATED_MXFP8_BUFFER_DIM_Y);
      const uint32_t scx = static_cast<uint32_t>(chunk_offset_X);

      store_2d_to_global(
          &out_act_colwise_sh[buff][0][0], out_act_colwise_ptr,
          scx, scy,
          TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
          tensor_w_out, tensor_h, stride_out, output_data_size);

      if constexpr (IS_BWD) {
        store_2d_to_global(
            &out_gate_colwise_sh[buff][0][0], out_gate_colwise_ptr,
            scx, scy,
            TDM_GATED_MXFP8_SHMEM_DIM_X, TDM_GATED_MXFP8_SHMEM_DIM_Y,
            tensor_w_out, tensor_h, stride_out, output_data_size);
      }
    }
  }

  // Drain all remaining TDM ops (stores)
  if (is_tdm_wave()) {
    wait_tensorcnt_0();
  }
  __syncthreads();
}

}  // namespace gated_mxfp8_tdm_kernel

// ---------------------------------------------------------------------------
// TDM launcher for gated MXFP8 cast
// ---------------------------------------------------------------------------

// Runtime check for gfx1250 TDM support.
#ifndef TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_
#define TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_
inline bool is_gfx1250() {
  static int result = -1;
  if (result < 0) {
    int device;
    (void)hipGetDevice(&device);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, device);
    result = (strncmp(prop.gcnArchName, "gfx1250", 7) == 0) ? 1 : 0;
  }
  return result == 1;
}
#endif  // TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_mxfp8_tdm(const Tensor &gated_input, const Tensor &grad, Tensor *output,
                               ParamOP &p, cudaStream_t stream) {
  using namespace gated_mxfp8_tdm_kernel;

  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, TDM_GATED_MXFP8_CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, TDM_GATED_MXFP8_CHUNK_DIM_X);

  const dim3 block_dim(TDM_GATED_MXFP8_THREADS);
  const dim3 grid_dim(blocks_X, blocks_Y);

  size_t scale_stride_rowwise = USE_ROWWISE_SCALING ? output->scale_inv.shape[1] : 1;
  size_t scale_stride_colwise = USE_COLWISE_SCALING ? output->columnwise_scale_inv.shape[1] : 1;

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          const IType *act_ptr  = reinterpret_cast<const IType *>(gated_input.data.dptr);
          const IType *gate_ptr = act_ptr + cols;  // gate follows act in interleaved layout
          const IType *grad_ptr = IS_BWD ? reinterpret_cast<const IType *>(grad.data.dptr)
                                         : nullptr;

          OType *out_act_rowwise_ptr  = USE_ROWWISE_SCALING
              ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
          OType *out_gate_rowwise_ptr = (USE_ROWWISE_SCALING && IS_BWD)
              ? reinterpret_cast<OType *>(output->data.dptr) + cols : nullptr;
          OType *out_act_colwise_ptr  = USE_COLWISE_SCALING
              ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;
          OType *out_gate_colwise_ptr = (USE_COLWISE_SCALING && IS_BWD)
              ? reinterpret_cast<OType *>(output->columnwise_data.dptr) + cols : nullptr;

          const size_t input_stride  = cols * 2;   // gated_input: [rows][cols*2]
          const size_t grad_stride   = cols;        // grad:        [rows][cols]
          const size_t output_stride = output_cols; // output:      [rows][output_cols]

          if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
            quantize_gated_mxfp8_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                             true, true>
                <<<grid_dim, block_dim, 0, stream>>>(
                    grad_ptr, act_ptr, gate_ptr,
                    out_act_rowwise_ptr, out_gate_rowwise_ptr,
                    out_act_colwise_ptr, out_gate_colwise_ptr,
                    scales_rowwise_ptr, scales_colwise_ptr,
                    rows, cols,
                    input_stride, grad_stride, output_stride,
                    scale_stride_rowwise, scale_stride_colwise, p);
          } else if (USE_ROWWISE_SCALING) {
            quantize_gated_mxfp8_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                             true, false>
                <<<grid_dim, block_dim, 0, stream>>>(
                    grad_ptr, act_ptr, gate_ptr,
                    out_act_rowwise_ptr, out_gate_rowwise_ptr,
                    out_act_colwise_ptr, out_gate_colwise_ptr,
                    scales_rowwise_ptr, scales_colwise_ptr,
                    rows, cols,
                    input_stride, grad_stride, output_stride,
                    scale_stride_rowwise, scale_stride_colwise, p);
          } else if (USE_COLWISE_SCALING) {
            quantize_gated_mxfp8_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                             false, true>
                <<<grid_dim, block_dim, 0, stream>>>(
                    grad_ptr, act_ptr, gate_ptr,
                    out_act_rowwise_ptr, out_gate_rowwise_ptr,
                    out_act_colwise_ptr, out_gate_colwise_ptr,
                    scales_rowwise_ptr, scales_colwise_ptr,
                    rows, cols,
                    input_stride, grad_stride, output_stride,
                    scale_stride_rowwise, scale_stride_colwise, p);
          }
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                              // NOLINT(*)
}

#endif  // defined(__gfx1250__)

#else
constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t THREADS_PER_CHUNK_COLWISE = 128;
constexpr size_t THREADS_PER_CHUNK_NON_COLWISE = CHUNK_DIM_X;

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = 32;
constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;
constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
static_assert(BUFF_DIM_Y == 32);

constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

// Number of 1-byte elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4) / 1;  // 128

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 4 = 128 / 32

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          bool ROWWISE_SCALING, bool COLWISE_SCALING, bool WITH_GEMM_SWIZZLED_SCALES,
          size_t THREADS_PER_CHUNK>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_gated_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                                const __grid_constant__ CUtensorMap tensor_map_input_act,
                                const __grid_constant__ CUtensorMap tensor_map_input_gate,
                                const __grid_constant__ CUtensorMap tensor_map_output_act_rowwise,
                                const __grid_constant__ CUtensorMap tensor_map_output_gate_rowwise,
                                const __grid_constant__ CUtensorMap tensor_map_output_act_colwise,
                                const __grid_constant__ CUtensorMap tensor_map_output_gate_colwise,
                                e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                                const size_t rows, const size_t cols,
                                const size_t scale_stride_rowwise,
                                const size_t scale_stride_colwise, const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;

  using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
  static_assert(STAGES >= 1);

  constexpr bool IS_CACHED_ACT_OP = ROWWISE_SCALING && COLWISE_SCALING;
  constexpr bool ONLY_COLWISE_SCALING = COLWISE_SCALING && (!ROWWISE_SCALING);

  // # of rows covered by one wave. Equal to the # of columnwise threads in Y dimension.
  constexpr size_t COLWISE_WAVEFRONT_SIZE = DIVUP(THREADS_PER_CHUNK, CHUNK_DIM_X);

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * CHUNK_DIM_X / SCALE_DIM_X;
  const size_t scales_block_offset_Y_colwise = blockIdx.y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const size_t scales_block_offset_X_colwise = blockIdx.x * CHUNK_DIM_X;

  constexpr size_t THREADS_X_ROWWISE = CHUNK_DIM_X / SCALE_DIM_X;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_Y_colwise = threadIdx.x / CHUNK_DIM_X;
  const size_t tid_X_colwise = threadIdx.x % CHUNK_DIM_X;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;
  const size_t thread_offset_Y_colwise = tid_Y_colwise;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t col_base_rowwise = block_offset_X + thread_offset_X_rowwise;
  const size_t row_base_colwise = block_offset_Y + thread_offset_Y_colwise;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_rowwise = (col_base_rowwise >= cols);
  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
  const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

  const size_t gate_scale_idx_offset_rowwise = (cols + SCALE_DIM_X - 1) / SCALE_DIM_X;
  const size_t gate_scale_idx_offset_colwise = cols;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t SUBAMAX_BUFF_DIM_Y = ONLY_COLWISE_SCALING ? COLWISE_WAVEFRONT_SIZE - 1 : 1;
  __shared__ float subamax_colwise_buff[SUBAMAX_BUFF_DIM_Y][CHUNK_DIM_X];

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  const size_t grad_mem = (IS_BWD ? buff_size_aligned_in : 0);

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = (IS_BWD ? buff_size_aligned_out : 0);
  const size_t out_mem = out_act_mem + out_gate_mem;

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  OType *out_act_colwise_sh = out_act_rowwise_sh;
  OType *out_gate_colwise_sh = out_gate_rowwise_sh;

  if constexpr (ROWWISE_SCALING && COLWISE_SCALING) {
    out_act_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem);
    out_gate_colwise_sh =
        reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem + out_act_mem);
  }

  IType *cached_act_sh = in_act_sh;    // in_act_sh is used as a cache buffer for activations
  IType *cached_gate_sh = in_gate_sh;  // in_gate_sh is used as a cache buffer for gated values

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  if constexpr (IS_BWD) {
    copy_2d_to_sharedx3(&in_grad_sh[0], &tensor_map_grad, block_offset_X, block_offset_Y,
                        &in_act_sh[0], &tensor_map_input_act, block_offset_X, block_offset_Y,
                        &in_gate_sh[0], &tensor_map_input_gate, block_offset_X, block_offset_Y,
                        shmem_buff_size, &mbar[0], is_master_thread);
  } else {
    copy_2d_to_sharedx2(&in_act_sh[0], &tensor_map_input_act, block_offset_X, block_offset_Y,
                        &in_gate_sh[0], &tensor_map_input_gate, block_offset_X, block_offset_Y,
                        shmem_buff_size, &mbar[0], is_master_thread);
  }

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_DIM;
      if constexpr (IS_BWD) {
        copy_2d_to_sharedx3(&in_grad_sh[next_buff_offset], &tensor_map_grad, global_offset_X,
                            global_offset_Y, &in_act_sh[next_buff_offset], &tensor_map_input_act,
                            global_offset_X, global_offset_Y, &in_gate_sh[next_buff_offset],
                            &tensor_map_input_gate, global_offset_X, global_offset_Y,
                            shmem_buff_size, &mbar[next_stage], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff_offset], &tensor_map_input_act, global_offset_X,
                            global_offset_Y, &in_gate_sh[next_buff_offset], &tensor_map_input_gate,
                            global_offset_X, global_offset_Y, shmem_buff_size, &mbar[next_stage],
                            is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], parity);

    if constexpr (COLWISE_SCALING) {
      const size_t shmem_offset_base_colwise =
          buff * BUFF_DIM + tid_Y_colwise * BUFF_DIM_X + tid_X_colwise;
      float thread_amax_act = 0.0f;
      float thread_amax_gate = 0.0f;
      float after_act_colwise[BUFF_DIM_Y / COLWISE_WAVEFRONT_SIZE];
      float after_gate_colwise[BUFF_DIM_Y / COLWISE_WAVEFRONT_SIZE];

// 1. Read/Compute elements. Find MXFP8-block AMAX
#pragma unroll
      for (int i = 0; i < SCALE_DIM_Y / COLWISE_WAVEFRONT_SIZE; ++i) {
        const size_t shmem_offset_colwise =
            shmem_offset_base_colwise + i * COLWISE_WAVEFRONT_SIZE * BUFF_DIM_X;

        float act_elt = static_cast<float>(in_act_sh[shmem_offset_colwise]);
        float gate_elt = static_cast<float>(in_gate_sh[shmem_offset_colwise]);
        float after_act_elt;
        float after_gate_elt;
        bool dgate_elt = true;  // gating is ideally an identity function
        if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
          // In case of GPT OSS, clamp the activation and gate values
          dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;  // Derivative of clamp
          gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
        }
        if constexpr (IS_BWD) {
          float grad_elt = static_cast<float>(in_grad_sh[shmem_offset_colwise]);
          const float x = act_elt;
          float act_x;
          float dact_x;
          if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
            const float x = min(act_elt, p.limit);
            const float s = sigmoidf(p.alpha * x);
            act_x = x * s;
            dact_x = act_elt <= p.limit ? s + s * (1 - s) * p.alpha * x : 0.0f;
          } else {
            if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
              const float s = sigmoidf(x);
              act_x = x * s;
              dact_x = x * s * (1 - s) + s;
            } else {
              act_x = ActOP(x, p);
              dact_x = DActOP(x, p);
            }
          }

          after_act_elt = dact_x * grad_elt * gate_elt;
          after_gate_elt = dgate_elt ? act_x * grad_elt : 0.0f;
        } else {
          after_act_elt = ActOP(act_elt, p) * gate_elt;
        }
        // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
        if constexpr (!std::is_same_v<IType, float>) {
          after_act_elt = static_cast<float>(static_cast<IType>(after_act_elt));
          if constexpr (IS_BWD) {
            after_gate_elt = static_cast<float>(static_cast<IType>(after_gate_elt));
          }
        }

        after_act_colwise[i] = after_act_elt;
        if constexpr (IS_BWD) {
          after_gate_colwise[i] = after_gate_elt;
        }

        // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
        if constexpr (IS_CACHED_ACT_OP) {
          cached_act_sh[shmem_offset_colwise] = static_cast<IType>(after_act_elt);
          if constexpr (IS_BWD) {
            cached_gate_sh[shmem_offset_colwise] = static_cast<IType>(after_gate_elt);
          }
        }

        const bool row_out_of_bounds_colwise = (row_base_colwise + stage_offset_Y + i >= rows);
        const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);

        if (!out_of_bounds) {
          thread_amax_act = fmaxf(thread_amax_act, fabsf(after_act_elt));
          if constexpr (IS_BWD) {
            thread_amax_gate = fmaxf(thread_amax_gate, fabsf(after_gate_elt));
          }
        }
      }

      if constexpr (ONLY_COLWISE_SCALING) {
        // Threads, whose id along Y-dim is 0, don't need to store to shared memory,
        // as they manage the columwise reduction of the amax
        if (tid_Y_colwise > 0) {
          subamax_colwise_buff[tid_Y_colwise - 1][tid_X_colwise] = thread_amax_act;
        }
        __syncthreads();
        if (tid_Y_colwise == 0) {
#pragma unroll
          for (int t = 0; t < SUBAMAX_BUFF_DIM_Y; ++t) {
            const float other_thread_amax = subamax_colwise_buff[t][tid_X_colwise];
            __builtin_assume(thread_amax_act >= 0);
            __builtin_assume(other_thread_amax >= 0);

            thread_amax_act = fmaxf(thread_amax_act, other_thread_amax);
          }
          subamax_colwise_buff[0][tid_X_colwise] = thread_amax_act;
        }
        __syncthreads();

        // All threads read the reduced amax (ACT)
        thread_amax_act = subamax_colwise_buff[0][tid_X_colwise];

        if constexpr (IS_BWD) {
          // Make sure the previous read of the ACT values has been completed,
          // so the data are not rewritten
          __syncthreads();
          if (tid_Y_colwise > 0) {
            subamax_colwise_buff[tid_Y_colwise - 1][tid_X_colwise] = thread_amax_gate;
          }
          __syncthreads();
          if (tid_Y_colwise == 0) {
#pragma unroll
            for (int t = 0; t < SUBAMAX_BUFF_DIM_Y; ++t) {
              const float other_thread_amax = subamax_colwise_buff[t][tid_X_colwise];
              __builtin_assume(thread_amax_gate >= 0);
              __builtin_assume(other_thread_amax >= 0);

              thread_amax_gate = fmaxf(thread_amax_gate, other_thread_amax);
            }
            subamax_colwise_buff[0][tid_X_colwise] = thread_amax_gate;
          }
          __syncthreads();

          // All threads read the reduced amax (GATE)
          thread_amax_gate = subamax_colwise_buff[0][tid_X_colwise];
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent_act =
          ptx::float_to_e8m0(thread_amax_act * Quantized_Limits<OType>::max_norm_rcp);
      const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const size_t global_scales_offset_X = scales_offset_X_colwise;
      size_t scale_idx;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = gemm_swizzled_scale_idx(global_scales_offset_X, global_scales_offset_Y,
                                            DIVUP(rows, static_cast<size_t>(128)));
      } else {
        scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      }
      const bool row_out_of_bounds_colwise = (row_base_colwise + stage_offset_Y) >= rows;
      const bool out_of_bounds_colwise = row_out_of_bounds_colwise || col_out_of_bounds_colwise;
      if (tid_Y_colwise == 0 && (!out_of_bounds_colwise)) {
        scales_colwise[scale_idx] = biased_exponent_act;
      }

      float block_scale_inverse_act = ptx::exp2f_rcp(biased_exponent_act);
      float block_scale_inverse_gate;

      if constexpr (IS_BWD) {
        const e8m0_t biased_exponent_gate =
            ptx::float_to_e8m0(thread_amax_gate * Quantized_Limits<OType>::max_norm_rcp);

        size_t scale_idx_gate;
        if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
          scale_idx_gate = gemm_swizzled_scale_idx(
              global_scales_offset_X + gate_scale_idx_offset_colwise, global_scales_offset_Y,
              DIVUP(rows, static_cast<size_t>(128)));
        } else {
          scale_idx_gate = scale_idx + gate_scale_idx_offset_colwise;
        }
        if (tid_Y_colwise == 0 && (!out_of_bounds_colwise)) {
          scales_colwise[scale_idx_gate] = biased_exponent_gate;
        }
        block_scale_inverse_gate = ptx::exp2f_rcp(biased_exponent_gate);
      }

// 3. Scale elements
#pragma unroll
      for (int i = 0; i < SCALE_DIM_Y / COLWISE_WAVEFRONT_SIZE; ++i) {
        const size_t shmem_offset_elt =
            shmem_offset_base_colwise + i * COLWISE_WAVEFRONT_SIZE * BUFF_DIM_X;
        if constexpr (IS_BWD) {
          OType2 out_pair;
          ptx::floatx2 in_pair = {after_act_colwise[i], after_gate_colwise[i]};
          const ptx::floatx2 block_scale_inverse_2x_pair = {block_scale_inverse_act,
                                                            block_scale_inverse_gate};
          ptx::mul_cvt_2x(out_pair, in_pair, block_scale_inverse_2x_pair);
          out_act_colwise_sh[shmem_offset_elt] = out_pair.x;
          out_gate_colwise_sh[shmem_offset_elt] = out_pair.y;
        } else {
          const float scaled_out_act = block_scale_inverse_act * after_act_colwise[i];
          out_act_colwise_sh[shmem_offset_elt] = static_cast<OType>(scaled_out_act);
        }
      }
    }

    if constexpr (ROWWISE_SCALING) {
      const size_t shmem_offset_base_rowwise =
          buff * BUFF_DIM + thread_offset_Y_rowwise * BUFF_DIM_X;

      float thread_amax_act = 0.0f;
      float thread_amax_gate = 0.0f;

      Vec<IType, PACK_SIZE> in_cached_act[WAVES];
      Vec<IType, PACK_SIZE> in_cached_gate[WAVES];

      float after_act_rowwise[SCALE_DIM_X];
      float after_gate_rowwise[SCALE_DIM_X];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      if constexpr (IS_CACHED_ACT_OP) {
        // ensures that all writes to cache made in the section above are visible to all threads
        __syncthreads();
        IType2 thread_amax_2x_act = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
        IType2 thread_amax_2x_gate = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
          const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
          const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

          // Load cached elements
          in_cached_act[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          if constexpr (IS_BWD) {
            in_cached_gate[w].load_from(&cached_gate_sh[shmem_offset_rowwise]);
          }
          // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
          // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
          if (!out_of_bounds) {
            if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
              for (int e = 0; e < PACK_SIZE; ++e) {
                thread_amax_act = fmaxf(thread_amax_act, fabsf(in_cached_act[w].data.elt[e]));
                if constexpr (IS_BWD) {
                  thread_amax_gate = fmaxf(thread_amax_gate, fabsf(in_cached_gate[w].data.elt[e]));
                }
              }
            } else {
#pragma unroll
              for (int e = 0; e < PACK_SIZE; e += 2) {
                const IType2 in_cached_2x_act = {in_cached_act[w].data.elt[e],
                                                 in_cached_act[w].data.elt[e + 1]};
                ptx::abs_max_2x(thread_amax_2x_act, thread_amax_2x_act, in_cached_2x_act);
                if constexpr (IS_BWD) {
                  const IType2 in_cached_2x_gate = {in_cached_gate[w].data.elt[e],
                                                    in_cached_gate[w].data.elt[e + 1]};
                  ptx::abs_max_2x(thread_amax_2x_gate, thread_amax_2x_gate, in_cached_2x_gate);
                }
              }
            }
          }
        }
        if constexpr (!std::is_same_v<IType, float>) {
          thread_amax_act = static_cast<float>(
              __hmax(__habs(thread_amax_2x_act.x), __habs(thread_amax_2x_act.y)));
          if constexpr (IS_BWD) {
            thread_amax_gate = static_cast<float>(
                __hmax(__habs(thread_amax_2x_gate.x), __habs(thread_amax_2x_gate.y)));
          }
        }
      } else {
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          Vec<IType, PACK_SIZE> in_grad;
          Vec<IType, PACK_SIZE> in_act;
          Vec<IType, PACK_SIZE> in_gate;

          in_act.load_from(&in_act_sh[shmem_offset_rowwise]);
          in_gate.load_from(&in_gate_sh[shmem_offset_rowwise]);
          if constexpr (IS_BWD) {
            in_grad.load_from(&in_grad_sh[shmem_offset_rowwise]);
          }

#pragma unroll
          for (int e = 0; e < PACK_SIZE; ++e) {
            const int j = w * PACK_SIZE + e;

            float act_elt = static_cast<float>(in_act.data.elt[e]);
            float gate_elt = static_cast<float>(in_gate.data.elt[e]);
            float after_act_elt;
            float after_gate_elt;
            bool dgate_elt = true;
            if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
              // In case of GPT OSS, clamp the activation and gate values
              dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;  // Derivative of clamp
              gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
            }
            if constexpr (IS_BWD) {
              float grad_elt = static_cast<float>(in_grad.data.elt[e]);
              const float x = act_elt;
              float act_x;
              float dact_x;
              if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
                const float x = min(act_elt, p.limit);
                const float s = sigmoidf(p.alpha * x);
                act_x = x * s;
                dact_x = act_elt <= p.limit ? s + s * (1 - s) * p.alpha * x : 0.0f;
              } else {
                if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
                  const float s = sigmoidf(x);
                  act_x = x * s;
                  dact_x = x * s * (1 - s) + s;
                } else {
                  act_x = ActOP(x, p);
                  dact_x = DActOP(x, p);
                }
              }

              after_act_elt = dact_x * grad_elt * gate_elt;
              after_gate_elt = dgate_elt ? act_x * grad_elt : 0.0f;
              after_act_rowwise[j] = after_act_elt;
              after_gate_rowwise[j] = after_gate_elt;
            } else {
              after_act_elt = ActOP(act_elt, p) * gate_elt;
              after_act_rowwise[j] = after_act_elt;
            }

            // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              after_act_elt = static_cast<float>(static_cast<IType>(after_act_elt));
              if constexpr (IS_BWD) {
                after_gate_elt = static_cast<float>(static_cast<IType>(after_gate_elt));
              }
            }

            const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
            if (!out_of_bounds) {
              thread_amax_act = fmaxf(thread_amax_act, fabsf(after_act_elt));
              if constexpr (IS_BWD) {
                thread_amax_gate = fmaxf(thread_amax_gate, fabsf(after_gate_elt));
              }
            }
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent_act =
          ptx::float_to_e8m0(thread_amax_act * Quantized_Limits<OType>::max_norm_rcp);
      const size_t stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
      const size_t stage_scales_offset_X = scales_offset_X_rowwise;
      size_t scale_idx;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        const size_t output_cols = (IS_BWD ? 2 : 1) * cols;
        scale_idx = gemm_swizzled_scale_idx(stage_scales_offset_Y, stage_scales_offset_X,
                                            DIVUP(output_cols, static_cast<size_t>(128)));
      } else {
        scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
      }
      const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y) >= rows;
      const bool out_of_bounds_rowwise = row_out_of_bounds_rowwise || col_out_of_bounds_rowwise;
      if (!out_of_bounds_rowwise) {
        scales_rowwise[scale_idx] = biased_exponent_act;
      }

      const float block_scale_inverse_act = ptx::exp2f_rcp(biased_exponent_act);
      const ptx::floatx2 block_scale_inverse_2x_act = {block_scale_inverse_act,
                                                       block_scale_inverse_act};

      float block_scale_inverse_gate;
      ptx::floatx2 block_scale_inverse_2x_gate;
      if constexpr (IS_BWD) {
        const e8m0_t biased_exponent_gate =
            ptx::float_to_e8m0(thread_amax_gate * Quantized_Limits<OType>::max_norm_rcp);

        size_t scale_idx_gate;
        if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
          const size_t output_cols = (IS_BWD ? 2 : 1) * cols;
          scale_idx_gate = gemm_swizzled_scale_idx(
              stage_scales_offset_Y, stage_scales_offset_X + gate_scale_idx_offset_rowwise,
              DIVUP(output_cols, static_cast<size_t>(128)));
        } else {
          scale_idx_gate = scale_idx + gate_scale_idx_offset_rowwise;
        }
        if (!out_of_bounds_rowwise) {
          scales_rowwise[scale_idx_gate] = biased_exponent_gate;
        }
        block_scale_inverse_gate = ptx::exp2f_rcp(biased_exponent_gate);
        block_scale_inverse_2x_gate = {block_scale_inverse_gate, block_scale_inverse_gate};
      }

// 3. Scale elements
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        Vec<OType2, PACK_SIZE / 2> out_act;
        Vec<OType2, PACK_SIZE / 2> out_gate;
#pragma unroll
        for (int e = 0; e < PACK_SIZE / 2; ++e) {
          IType2 in_act;
          OType2 &out_act_pair = reinterpret_cast<OType2 &>(out_act.data.elt[e]);

          if constexpr (IS_CACHED_ACT_OP) {
            in_act.x = in_cached_act[w].data.elt[2 * e];
            in_act.y = in_cached_act[w].data.elt[2 * e + 1];
          } else {
            const int j = w * PACK_SIZE + 2 * e;
            in_act.x = after_act_rowwise[j];
            in_act.y = after_act_rowwise[j + 1];
          }
          ptx::mul_cvt_2x(out_act_pair, in_act, block_scale_inverse_2x_act);

          if constexpr (IS_BWD) {
            IType2 in_gate;
            OType2 &out_gate_pair = reinterpret_cast<OType2 &>(out_gate.data.elt[e]);

            if constexpr (IS_CACHED_ACT_OP) {
              in_gate.x = in_cached_gate[w].data.elt[2 * e];
              in_gate.y = in_cached_gate[w].data.elt[2 * e + 1];
            } else {
              const int j = w * PACK_SIZE + 2 * e;
              in_gate.x = after_gate_rowwise[j];
              in_gate.y = after_gate_rowwise[j + 1];
            }
            ptx::mul_cvt_2x(out_gate_pair, in_gate, block_scale_inverse_2x_gate);
          }
        }

        const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
        const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
        const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_idx;
        out_act.store_to(&out_act_rowwise_sh[shmem_offset_rowwise]);
        if constexpr (IS_BWD) {
          out_gate.store_to(&out_gate_rowwise_sh[shmem_offset_rowwise]);
        }
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t buff_offset = buff * BUFF_DIM;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_act_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_act_rowwise_sh[buff_offset]));
        if constexpr (IS_BWD) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_rowwise), global_offset_X,
              global_offset_Y, reinterpret_cast<uint64_t *>(&out_gate_rowwise_sh[buff_offset]));
        }
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_act_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_act_colwise_sh[buff_offset]));
        if constexpr (IS_BWD) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_colwise), global_offset_X,
              global_offset_Y, reinterpret_cast<uint64_t *>(&out_gate_colwise_sh[buff_offset]));
        }
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }

  parity ^= 1;
  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}  // NOLINT(readability/fn_size)
#endif //#ifndef __HIP_PLATFORM_AMD__

}  // namespace gated_kernel

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated(const Tensor &gated_input, const Tensor &grad, Tensor *output, ParamOP &p,
                    cudaStream_t stream) {
  using namespace gated_kernel;
  checkCuDriverContext(stream);

  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();
  const bool with_gemm_swizzled_scales = output->with_gemm_swizzled_scales;
#ifdef __HIP_PLATFORM_AMD__
  // TODO: rocm TE should not need swizzle
  // ensure upstream does not pass swizzle=true down here
  NVTE_CHECK(with_gemm_swizzled_scales != true, "ROCm TE does not support swizzling for gemm");
#endif

  if (USE_ROWWISE_SCALING) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (USE_COLWISE_SCALING) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

#ifndef __HIP_PLATFORM_AMD__
  ScalingType scaling_type;
  if (USE_ROWWISE_SCALING && (!USE_COLWISE_SCALING)) {
    scaling_type = ScalingType::ROWWISE;
  } else if ((!USE_ROWWISE_SCALING) && USE_COLWISE_SCALING) {
    scaling_type = ScalingType::COLWISE;
  } else if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
    scaling_type = ScalingType::BIDIMENSIONAL;
  }
#endif

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

#ifdef __HIP_PLATFORM_AMD__
  constexpr size_t TMA_SHMEM_ALIGNMENT = ALIGNMENT_SIZE;

  constexpr size_t BUFF_DIM_Y = BUFFER_DIM_Y;
  constexpr size_t BUFF_DIM_X = BUFFER_DIM_X;
  constexpr size_t BUFFS_NUM = BUFFERS_NUM;
#endif

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

#ifndef __HIP_PLATFORM_AMD__
  const size_t THREADS_PER_CHUNK = (scaling_type == ScalingType::COLWISE)
                                       ? THREADS_PER_CHUNK_COLWISE
                                       : THREADS_PER_CHUNK_NON_COLWISE;
#endif

  const dim3 grid(blocks_X, blocks_Y);
  const dim3 block_size(THREADS_PER_CHUNK);

  size_t scale_stride_rowwise = USE_ROWWISE_SCALING ? output->scale_inv.shape[1] : 1;
  size_t scale_stride_colwise = USE_COLWISE_SCALING ? output->columnwise_scale_inv.shape[1] : 1;

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
#ifdef __HIP_PLATFORM_AMD__
#if defined(__gfx1250__)
              // gfx1250 TDM path -- use TDM-accelerated kernel when cols is aligned
              if (is_gfx1250() && (cols % gated_mxfp8_tdm_kernel::TDM_GATED_MXFP8_CHUNK_DIM_X == 0)) {
                quantize_gated_mxfp8_tdm<IS_BWD, ParamOP, ActOP, DActOP>(
                    gated_input, grad, output, p, stream);
                return;
              }
#endif  // defined(__gfx1250__)
              const IType *tensor_map_grad = IS_BWD ? reinterpret_cast<const IType *>(grad.data.dptr) : nullptr;
              const IType *tensor_map_input_act = reinterpret_cast<const IType *>(gated_input.data.dptr);
              const IType *tensor_map_input_gate = reinterpret_cast<const IType *>(gated_input.data.dptr) + cols;
              OType *tensor_map_output_act_rowwise = USE_ROWWISE_SCALING ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
              OType *tensor_map_output_gate_rowwise = USE_ROWWISE_SCALING ? reinterpret_cast<OType *>(output->data.dptr) + cols : nullptr;
              OType *tensor_map_output_act_colwise = USE_COLWISE_SCALING ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;
              OType *tensor_map_output_gate_colwise = USE_COLWISE_SCALING ? reinterpret_cast<OType *>(output->columnwise_data.dptr) + cols : nullptr;

              constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
              constexpr size_t output_type_bit_size = TypeInfo<OType>::size;
#else // #ifdef __HIP_PLATFORM_AMD__
              alignas(64) CUtensorMap tensor_map_grad{};
              alignas(64) CUtensorMap tensor_map_input_act{};
              alignas(64) CUtensorMap tensor_map_input_gate{};
              alignas(64) CUtensorMap tensor_map_output_act_rowwise{};
              alignas(64) CUtensorMap tensor_map_output_gate_rowwise{};
              alignas(64) CUtensorMap tensor_map_output_act_colwise{};
              alignas(64) CUtensorMap tensor_map_output_gate_colwise{};

              constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
              constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

              if constexpr (IS_BWD) {
                create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                                     cols, 0, input_type_bit_size);
              }

              const uint32_t tensor_stride_elems = output_cols;
              create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, BUFF_DIM_Y,
                                   BUFF_DIM_X, cols * 2, 0, input_type_bit_size);
              create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, BUFF_DIM_Y,
                                   BUFF_DIM_X, cols * 2, cols, input_type_bit_size);

              if (USE_ROWWISE_SCALING) {
                create_2D_tensor_map(tensor_map_output_act_rowwise, output->data, rows, cols,
                                     BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, 0,
                                     output_type_bit_size);
                create_2D_tensor_map(tensor_map_output_gate_rowwise, output->data, rows, cols,
                                     BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, cols,
                                     output_type_bit_size);
              }
              if (USE_COLWISE_SCALING) {
                create_2D_tensor_map(tensor_map_output_act_colwise, output->columnwise_data, rows,
                                     cols, BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, 0,
                                     output_type_bit_size);
                create_2D_tensor_map(tensor_map_output_gate_colwise, output->columnwise_data, rows,
                                     cols, BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, cols,
                                     output_type_bit_size);
              }

#endif // #ifdef __HIP_PLATFORM_AMD__

              const size_t buff_elems_total = BUFFS_NUM * BUFF_DIM_Y * BUFF_DIM_X;
              const size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
              const size_t output_buff_size = (buff_elems_total * output_type_bit_size) / 8;
              const size_t buff_size_aligned_in =
                  DIVUP_TO_MULTIPLE(input_buff_size, TMA_SHMEM_ALIGNMENT);
              const size_t buff_size_aligned_out =
                  DIVUP_TO_MULTIPLE(output_buff_size, TMA_SHMEM_ALIGNMENT);

              const size_t grad_mem = (IS_BWD ? buff_size_aligned_in : 0);
              const size_t in_act_mem = buff_size_aligned_in;
              const size_t in_gate_mem = buff_size_aligned_in;
              const size_t in_mem = grad_mem + in_act_mem + in_gate_mem;

              const size_t out_act_mem = buff_size_aligned_out;
#ifdef __HIP_PLATFORM_AMD__
              const size_t out_gate_mem = buff_size_aligned_out;
#else
              const size_t out_gate_mem = (IS_BWD ? buff_size_aligned_out : 0);
#endif
              size_t out_mem = out_act_mem + out_gate_mem;

              if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) { out_mem *= 2; }

              const size_t shmem_size = in_mem + out_mem + TMA_SHMEM_ALIGNMENT;

#ifdef __HIP_PLATFORM_AMD__
              TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                (USE_COLWISE_SCALING ? 32 : 1), SCALE_DIM_Y,
                TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
                  (USE_ROWWISE_SCALING ? 32 : 1), SCALE_DIM_X,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(!(cols % (32 * sizeof(IType))), IS_ALIGNED, {
                    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                        quantize_gated_mxfp8_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                    SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

                        quantize_gated_mxfp8_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                    SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED>
                            <<<grid, block_size, shmem_size, stream>>>(
                                tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                                tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                                tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                                scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                                scale_stride_colwise, p);
                        NVTE_CHECK_CUDA(cudaGetLastError());
              })));  // NOLINT(*)
#else
              // Zero out swizzled scales if padding is needed
              /// TODO (tmoon) Handle this within the cast kernel
              if (with_gemm_swizzled_scales) {
                constexpr size_t TILE_DIM_X = 128;  // Tile dim in data buffer
                constexpr size_t TILE_DIM_Y = 128;
                if (cols % TILE_DIM_X != 0 || rows % TILE_DIM_Y != 0) {
                  if (USE_ROWWISE_SCALING) {
                    NVTE_CHECK_CUDA(cudaMemsetAsync(output->scale_inv.dptr, 0,
                                                    output->scale_inv.buffer_size_bytes(), stream));
                  }
                  if (USE_COLWISE_SCALING) {
                    NVTE_CHECK_CUDA(
                        cudaMemsetAsync(output->columnwise_scale_inv.dptr, 0,
                                        output->columnwise_scale_inv.buffer_size_bytes(), stream));
                  }
                }
              }

              switch (scaling_type) {
                case ScalingType::ROWWISE: {
                  auto kernel =
                      quantize_gated_mxfp8_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                  true, false, WITH_GEMM_SWIZZLED_SCALES,
                                                  THREADS_PER_CHUNK_NON_COLWISE>;
                  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

                  kernel<<<grid, block_size, shmem_size, stream>>>(
                      tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                      tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                      tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                      scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                      scale_stride_colwise, p);
                  break;
                }
                case ScalingType::COLWISE: {
                  auto kernel =
                      quantize_gated_mxfp8_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                  false, true, WITH_GEMM_SWIZZLED_SCALES,
                                                  THREADS_PER_CHUNK_COLWISE>;
                  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

                  kernel<<<grid, block_size, shmem_size, stream>>>(
                      tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                      tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                      tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                      scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                      scale_stride_colwise, p);
                  break;
                }
                case ScalingType::BIDIMENSIONAL: {
                  auto kernel =
                      quantize_gated_mxfp8_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType,
                                                  true, true, WITH_GEMM_SWIZZLED_SCALES,
                                                  THREADS_PER_CHUNK_NON_COLWISE>;
                  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

                  kernel<<<grid, block_size, shmem_size, stream>>>(
                      tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                      tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                      tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                      scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                      scale_stride_colwise, p);
                  break;
                }
              } NVTE_CHECK_CUDA(cudaGetLastError());
#endif //#ifdef __HIP_PLATFORM_AMD__
          );  // NOLINT(*)
      );                                                // NOLINT(*)
  );                                                    // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GATED_MXFP8_CUH_
