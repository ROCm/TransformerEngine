/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <cfloat>
#include <hip/hip_runtime.h>

#include "../common.h"
#include "math.h"
#include "ptx.cuh"
#include "rocm_vectorized_2d.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/cast.h"
#include "vectorized_pointwise.h"
#include "../utils.cuh"

namespace transformer_engine {

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

namespace gated_kernels {

constexpr size_t ALIGNMENT_SIZE = 128;
// TODO: Identify optimal chunk/thread size for MI350+
constexpr size_t ROCM_CHUNK_DIM_Y = 64;
constexpr size_t ROCM_CHUNK_DIM_X = 64;
constexpr size_t ROCM_THREADS_PER_CHUNK = 256;
constexpr size_t ROCM_THREADS_PER_CHUNK_X = 64;
constexpr size_t ROCM_THREADS_PER_CHUNK_Y = ROCM_THREADS_PER_CHUNK / ROCM_THREADS_PER_CHUNK_X;  // 4 = 256 / 64
constexpr size_t ROCM_BUFFERS_NUM = 1; // No async load for HIP
constexpr size_t ROCM_BUFFER_DIM_Y = 32;
constexpr size_t ROCM_BUFFER_DIM_X = ROCM_CHUNK_DIM_X;  // 64
constexpr size_t ROCM_SHMEM_DIM_Y = ROCM_BUFFER_DIM_Y;  // 32
constexpr size_t ROCM_SHMEM_DIM_X = ROCM_BUFFER_DIM_X;  // 64

constexpr size_t ROCM_BUFFER_STAGES_NUM = ROCM_BUFFER_DIM_Y / ROCM_THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ROCM_ITERATIONS = ROCM_CHUNK_DIM_Y / ROCM_BUFFER_DIM_Y;                 //   2 = 64 / 32
static_assert(ROCM_ITERATIONS >= 1);

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(ROCM_THREADS_PER_CHUNK)
    cast_mxfp8_gated_kernel(const IType *grad_ptr,
                            const IType *input_act,
                            const IType *input_gate,
                            OType *output_act_rowwise,
                            OType *output_gate_rowwise,
                            OType *output_act_colwise,
                            OType *output_gate_colwise,
                            e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                            const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
                            const size_t scale_stride_colwise) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = ROCM_CHUNK_DIM_Y;                //  64
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = ROCM_CHUNK_DIM_X / SCALE_DIM_X;  //   2 = 64 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = ROCM_CHUNK_DIM_Y / SCALE_DIM_Y;  //   2 = 64 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = ROCM_CHUNK_DIM_X;                //  64

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int chunk_offset_Y = blockIdx.y * ROCM_CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * ROCM_CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / ROCM_THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % ROCM_THREADS_PER_CHUNK_X;

  constexpr size_t VECTOR_WIDTH = (IS_ALIGNED ?: 2) * 8 / sizeof(OType);

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const bool col_out_of_bounds = (chunk_offset_X + thread_offset_X >= cols);

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  const size_t buff_elems = ROCM_SHMEM_DIM_Y * ROCM_SHMEM_DIM_X;
  const size_t buff_elems_total = ROCM_BUFFERS_NUM * buff_elems;
  const size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  const size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = buff_size_aligned_out;
  const size_t out_mem = out_act_mem + out_gate_mem;

  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols; 

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  OType *out_act_colwise_sh = out_act_rowwise_sh;
  OType *out_gate_colwise_sh = out_gate_rowwise_sh;

  if constexpr (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
    out_act_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem);
    out_gate_colwise_sh =
        reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem + out_act_mem);
  }

  __shared__ float stage_amax_sh[ROCM_THREADS_PER_CHUNK_Y][ROCM_CHUNK_DIM_X];

  __syncthreads();

  for (int it = 0; it < ROCM_ITERATIONS; it++) {
    const int chunk_it_offset_y = chunk_offset_Y + it * ROCM_BUFFER_DIM_Y;
    const int chunk_it_offset_x = chunk_offset_X;
    const size_t row_base = chunk_it_offset_y; 

    // Initiate bulk tensor copy
    if constexpr (IS_DGATED) {
      copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_grad_sh[0], grad_ptr, chunk_it_offset_x, chunk_it_offset_y,
                        cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);
    }

    // Act
    copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_act_sh[0], input_act, chunk_it_offset_x, chunk_it_offset_y,
                      2*cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);

    // Gate
    copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_gate_sh[0], input_gate, chunk_it_offset_x, chunk_it_offset_y,
                      2*cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);

    __syncthreads();

    const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + it;
    const int iteration_scale_rowwise_offset_Y = scales_rowwise_chunk_offset_Y + it * ROCM_BUFFER_DIM_Y;

    float after_dact_reg[ROCM_BUFFER_STAGES_NUM];
    float after_dgate_reg[ROCM_BUFFER_STAGES_NUM];
    float thread_Y_mx_block_amax = 0.0f;
    float thread_Y_mx_block_amax_gate = 0.0f;

    for (int stage = 0; stage < ROCM_BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * ROCM_THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * ROCM_SHMEM_DIM_X + shmem_offset_x;

      const size_t row = row_base + shmem_offset_y;
      const bool row_out_of_bounds = (row >= rows);
      const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

      float act_elt = static_cast<float>(in_act_sh[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh[shmem_idx]);

      if constexpr (IS_DGATED) {
        float grad_elt = static_cast<float>(in_grad_sh[shmem_idx]);
        const float x = act_elt;
        float act_x;
        float dact_x;

        if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
          const float s = sigmoidf(x);
          act_x = x * s;
          dact_x = x * s * (1 - s) + s;
        } else {
          act_x = ActOP(x, {});
          dact_x = DActOP(x, {});
        }
        after_dact_reg[stage] = dact_x * grad_elt * gate_elt;
        after_dgate_reg[stage] = act_x * grad_elt;
      } else {
        after_dact_reg[stage] = ActOP(act_elt, {}) * gate_elt;
      }

      // Numerical truncation: downcast to IType (BF16/FP16) and upcast back to FP32
      if constexpr (!std::is_same_v<IType, float>) {
        after_dact_reg[stage] = static_cast<float>(static_cast<IType>(after_dact_reg[stage]));
        if constexpr (IS_DGATED) {
          after_dgate_reg[stage] = static_cast<float>(static_cast<IType>(after_dgate_reg[stage]));
        }
      }

      if constexpr (USE_ROWWISE_SCALING) {
        if constexpr (IS_DGATED) {
          // dgate
          float amax = fabsf(after_dgate_reg[stage]);
          const float mx_block_X_amax = warp_reduce_max_broadcast(amax);
          const e8m0_t biased_exponent_X =
              ptx::float_to_e8m0(mx_block_X_amax * Quantized_Limits<OType>::max_norm_rcp);
          const float scale_reciprocal_X = ptx::exp2f_rcp(biased_exponent_X);

          out_gate_rowwise_sh[shmem_idx] =
              static_cast<OType>(scale_reciprocal_X * after_dgate_reg[stage]);

          // Only single thread writes the computed scaling factor
          if ((tid_X % SCALE_DIM_X == 0) && !out_of_bounds) {
            const int global_scales_offset_Y =
                iteration_scale_rowwise_offset_Y + stage_offset_Y + thread_offset_Y;
            const int global_scales_offset_X =
                scales_rowwise_chunk_offset_X + (tid_X + cols) / SCALE_DIM_X;
            const int scale_idx =
                global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
            scales_rowwise[scale_idx] = biased_exponent_X;
          }
        }
        float amax = fabsf(after_dact_reg[stage]);
        const float mx_block_X_amax = warp_reduce_max_broadcast(amax);
        const e8m0_t biased_exponent_X =
            ptx::float_to_e8m0(mx_block_X_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal_X = ptx::exp2f_rcp(biased_exponent_X);

        out_act_rowwise_sh[shmem_idx] =
            static_cast<OType>(scale_reciprocal_X * after_dact_reg[stage]);

        // Only single thread writes the computed scaling factor
        if ((tid_X % SCALE_DIM_X == 0) && !out_of_bounds) {
          const int global_scales_offset_Y =
              iteration_scale_rowwise_offset_Y + stage_offset_Y + thread_offset_Y;
          const int global_scales_offset_X = scales_rowwise_chunk_offset_X + tid_X / SCALE_DIM_X;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
          scales_rowwise[scale_idx] = biased_exponent_X;
        }
      }

      if constexpr (USE_COLWISE_SCALING) {
        __builtin_assume(thread_Y_mx_block_amax >= 0);
        __builtin_assume(thread_Y_mx_block_amax_gate >= 0);
        thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, fabsf(after_dact_reg[stage]));
        if constexpr (IS_DGATED) {
          thread_Y_mx_block_amax_gate =
              fmaxf(thread_Y_mx_block_amax_gate, fabsf(after_dgate_reg[stage]));
        }
      }
    }

    if constexpr (USE_COLWISE_SCALING) {
      const bool row_out_of_bounds = (row_base >= rows);
      const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

      if constexpr (IS_DGATED) {
        // Colwise max reduction of the amax element
        if (tid_Y > 0) {
          stage_amax_sh[tid_Y][tid_X] = thread_Y_mx_block_amax_gate;
        }
        __syncthreads();
        if (tid_Y == 0) {
#pragma unroll
          for (int y = 1; y < ROCM_THREADS_PER_CHUNK_Y; ++y) {
            thread_Y_mx_block_amax_gate =
                fmaxf(thread_Y_mx_block_amax_gate, stage_amax_sh[y][tid_X]);
          }
          stage_amax_sh[0][tid_X] = thread_Y_mx_block_amax_gate;  // write mx column-block amax
        }
        __syncthreads();

        const float mx_block_Y_amax = stage_amax_sh[0][tid_X];  // read the mx column-block amax

        // For the scaling along both dimensions, the thread amax is already computed in ROWWISE section
        if constexpr (!USE_ROWWISE_SCALING) {
          __builtin_assume(mx_block_Y_amax >= 0);
        }

        const e8m0_t biased_exponent =
            ptx::float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal = ptx::exp2f_rcp(biased_exponent);

        // Only single thread writes the computed scaling factor
        // Also assuming one iteration covers exactly 32 rows
        if ((tid_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_X + cols;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < ROCM_BUFFER_STAGES_NUM; ++stage) {
          const int stage_offset_Y = stage * ROCM_THREADS_PER_CHUNK_Y;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X;
          const int shmem_idx = shmem_offset_y * ROCM_SHMEM_DIM_X + shmem_offset_x;

          out_gate_colwise_sh[shmem_idx] =
              static_cast<OType>(scale_reciprocal * after_dgate_reg[stage]);
        }
      }
      // Colwise max reduction of the amax element
      if (tid_Y > 0) {
        stage_amax_sh[tid_Y][tid_X] = thread_Y_mx_block_amax;
      }
      __syncthreads();
      if (tid_Y == 0) {
#pragma unroll
        for (int y = 1; y < ROCM_THREADS_PER_CHUNK_Y; ++y) {
          thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, stage_amax_sh[y][tid_X]);
        }
        stage_amax_sh[0][tid_X] = thread_Y_mx_block_amax;  // write mx column-block amax
      }
      __syncthreads();

      const float mx_block_Y_amax = stage_amax_sh[0][tid_X];  // read the mx column-block amax

      // For the scaling along both dimensions, the thread amax is already computed in ROWWISE section
      if constexpr (!USE_ROWWISE_SCALING) {
        __builtin_assume(mx_block_Y_amax >= 0);
      }

      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
      const float scale_reciprocal = ptx::exp2f_rcp(biased_exponent);

      // Only single thread writes the computed scaling factor
      // Also assuming one iteration covers exactly 32 rows
      if ((tid_Y == 0) && !out_of_bounds) {
        const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;
      }

#pragma unroll
      for (int stage = 0; stage < ROCM_BUFFER_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage * ROCM_THREADS_PER_CHUNK_Y;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;
        const int shmem_idx = shmem_offset_y * ROCM_SHMEM_DIM_X + shmem_offset_x;

        out_act_colwise_sh[shmem_idx] =
            static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
      }
    }

    __syncthreads();

    if constexpr (USE_ROWWISE_SCALING) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_act_rowwise_sh[0], output_act_rowwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);
      if constexpr (IS_DGATED) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_gate_rowwise_sh[0], output_gate_rowwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);
      }
    }
    
    if constexpr (USE_COLWISE_SCALING) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_act_colwise_sh[0], output_act_colwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);
      if constexpr (IS_DGATED) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_gate_colwise_sh[0], output_gate_colwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, ROCM_SHMEM_DIM_Y, ROCM_SHMEM_DIM_X, rows, cols);
      }
    }
    __syncthreads();
  }
}
} // namespace gated_kernels

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void rocm_cast_mxfp8_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                            cudaStream_t stream) {
  using namespace gated_kernels;

  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, ROCM_CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, ROCM_CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const dim3 block_size(ROCM_THREADS_PER_CHUNK);

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

          const IType *grad_ptr = IS_DGATED ? reinterpret_cast<const IType *>(grad.data.dptr) : nullptr;
          const IType *input_act_ptr = reinterpret_cast<const IType *>(gated_input.data.dptr);
          const IType *input_gate_ptr = reinterpret_cast<const IType *>(gated_input.data.dptr) + cols;
          OType *output_act_rowwise_ptr = USE_ROWWISE_SCALING ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
          OType *output_gate_rowwise_ptr = USE_ROWWISE_SCALING ? reinterpret_cast<OType *>(output->data.dptr) + cols : nullptr;
          OType *output_act_colwise_ptr = USE_COLWISE_SCALING ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;
          OType *output_gate_colwise_ptr = USE_COLWISE_SCALING ? reinterpret_cast<OType *>(output->columnwise_data.dptr) + cols : nullptr;

          constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
          constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

          const size_t buff_elems_total = ROCM_BUFFERS_NUM * ROCM_BUFFER_DIM_Y * ROCM_BUFFER_DIM_X;
          const size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
          const size_t output_buff_size = (buff_elems_total * output_type_bit_size) / 8;
          const size_t buff_size_aligned_in =
              DIVUP_TO_MULTIPLE(input_buff_size, ALIGNMENT_SIZE);
          const size_t buff_size_aligned_out =
              DIVUP_TO_MULTIPLE(output_buff_size, ALIGNMENT_SIZE);

          const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);
          const size_t in_mem = grad_mem + buff_size_aligned_in + buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;
          size_t out_mem = out_act_mem + out_gate_mem;
          if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) { out_mem *= 2; }
          const size_t shmem_size = in_mem + out_mem + ALIGNMENT_SIZE;

          TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
            (USE_COLWISE_SCALING ? 32 : 1), SCALE_DIM_Y,
            TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
              (USE_ROWWISE_SCALING ? 32 : 1), SCALE_DIM_X,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(!(cols % (32 * sizeof(IType))), IS_ALIGNED, {
                NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                    cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType,
                                            SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

                cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType,
                                        SCALE_DIM_Y, SCALE_DIM_X, IS_ALIGNED>
                    <<<grid, block_size, shmem_size, stream>>>(
                        grad_ptr, input_act_ptr, input_gate_ptr,
                        output_act_rowwise_ptr, output_gate_rowwise_ptr,
                        output_act_colwise_ptr, output_gate_colwise_ptr,
                        scales_rowwise_ptr, scales_colwise_ptr, rows, cols,
                        scale_stride_rowwise, scale_stride_colwise);
                NVTE_CHECK_CUDA(cudaGetLastError());
          })));  // NOLINT(*)
      );       // NOLINT(*)
  );           // NOLINT(*)
}

} // namespace transformer_engine
