/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// drop-in rocm replacement for mxfp8 gated quantize kernel
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

constexpr size_t ALIGNMENT_SIZE = 128;
constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t THREADS_PER_CHUNK = 256;
constexpr size_t BUFFERS_NUM = 1; // No async load for HIP
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;  // 2
static_assert(ITERATIONS >= 1);

constexpr size_t ELEMS_PER_THREAD = 8;
constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  // 8
constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;  // 32
static_assert(THREADS_PER_CHUNK_Y_ROWWISE <= BUFFER_DIM_Y);
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;  // 64
constexpr size_t THREADS_PER_CHUNK_Y_COLWISE = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_COLWISE;  // 4
constexpr size_t BUFFER_STAGES_NUM_COLWISE = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_COLWISE;  // 8
constexpr size_t THREADS_PER_CHUNK_X = THREADS_PER_CHUNK_X_COLWISE;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK_Y_COLWISE;
constexpr size_t BUFFER_STAGES_NUM = BUFFER_STAGES_NUM_COLWISE;

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType>
__device__ inline void compute_gated_activation(
    float act_elt, float gate_elt, float grad_elt,
    const ParamOP &p, float &result_act, float &result_gate) {

  bool dgate_elt_valid = true;
  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    dgate_elt_valid = gate_elt <= p.limit && gate_elt >= -p.limit;
    gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
  }

  if constexpr (IS_DGATED) {
    const float x = act_elt;
    float act_x, dact_x;
    if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
      const float cx = min(act_elt, p.limit);
      const float s  = sigmoidf(p.alpha * cx);
      act_x  = cx * s;
      dact_x = act_elt <= p.limit ? s + s * (1 - s) * p.alpha * cx : 0.0f;
    } else {
      if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
        const float s = sigmoidf(x);
        act_x  = x * s;
        dact_x = x * s * (1 - s) + s;
      } else {
        act_x  = ActOP(x, p);
        dact_x = DActOP(x, p);
      }
    }
    result_act  = dact_x * grad_elt * gate_elt;
    result_gate = dgate_elt_valid ? act_x * grad_elt : 0.0f;
  } else {
    result_act  = ActOP(act_elt, p) * gate_elt;
    result_gate = 0.0f;
  }

  if constexpr (!std::is_same_v<IType, float>) {
    result_act = static_cast<float>(static_cast<IType>(result_act));
    if constexpr (IS_DGATED) {
      result_gate = static_cast<float>(static_cast<IType>(result_gate));
    }
  }
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_gated_mxfp8_kernel(
      const IType *grad_ptr, const IType *input_act, const IType *input_gate, OType *output_act_rowwise, 
      OType *output_gate_rowwise, OType *output_act_colwise, OType *output_gate_colwise,
      e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise, const size_t rows, const size_t cols, 
      const size_t scale_stride_rowwise, const size_t scale_stride_colwise, const ParamOP p) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE = DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);  // 4
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;

  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;

  const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  constexpr size_t VECTOR_WIDTH_IN = IS_ALIGNED ? 8 : 16;
  constexpr size_t VECTOR_WIDTH_OUT = 16;

  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  constexpr size_t ROWS_PER_THREAD = CHUNK_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;

  // ROWWISE-ONLY PATH: Direct global memory, no shared memory
  if constexpr (USE_ROWWISE_SCALING && !USE_COLWISE_SCALING) {
    const size_t col_start = chunk_offset_X + thread_offset_X_rowwise;
    const bool col_valid = (col_start < cols);

#pragma unroll
    for (size_t r = 0; r < ROWS_PER_THREAD; r++) {
      const size_t row = chunk_offset_Y + tid_rowwise_Y + r * THREADS_PER_CHUNK_Y_ROWWISE;

      const bool row_valid = (row < rows);

      Vec<IType, ELEMS_PER_THREAD> act_vec, gate_vec, grad_vec;

      if (row_valid && col_valid) {
        if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
          act_vec.load_from(&input_act[row * 2*cols + col_start]);
          gate_vec.load_from(&input_gate[row * 2*cols + col_start]);
          if constexpr (IS_DGATED) {
            grad_vec.load_from(&grad_ptr[row * cols + col_start]);
          }
        } else {
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            act_vec.data.elt[j]  = (col_start + j < cols) ? input_act[row * 2*cols + col_start + j] : static_cast<IType>(0);
            gate_vec.data.elt[j] = (col_start + j < cols) ? input_gate[row * 2*cols + col_start + j] : static_cast<IType>(0);
            if constexpr (IS_DGATED) {
              grad_vec.data.elt[j] = (col_start + j < cols) ? grad_ptr[row * cols + col_start + j] : static_cast<IType>(0);
            }
          }
        }
      }

      // Compute activations
      float computed_act[ELEMS_PER_THREAD];
      float computed_gate[ELEMS_PER_THREAD];
      float act_amax = 0;
      float gate_amax = 0;

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        const bool out_of_bounds = (!row_valid || !col_valid || col_start + j >= cols);
        float act_elt  = static_cast<float>(act_vec.data.elt[j]);
        float gate_elt = static_cast<float>(gate_vec.data.elt[j]);
        float grad_elt = IS_DGATED ? static_cast<float>(grad_vec.data.elt[j]) : 0.0f;

        compute_gated_activation<IS_DGATED, ParamOP, ActOP, DActOP, IType>(
            act_elt, gate_elt, grad_elt, p, computed_act[j], computed_gate[j]);

        if (!out_of_bounds) {
          act_amax = fmaxf(act_amax, fabsf(computed_act[j]));
          if constexpr (IS_DGATED) {
            gate_amax = fmaxf(gate_amax, fabsf(computed_gate[j]));
          }
        }
      }

      // --- Act rowwise quantization ---
      {
        __builtin_assume(act_amax >= 0);
        const float scale_amax = rocm_subwarp_allreduce<SUBWARP_WIDTH>(act_amax, rocm_op::max{});
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp<float>(biased_exp);

        Vec<OType, ELEMS_PER_THREAD> out_vec;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          out_vec.data.elt[j] = static_cast<OType>(computed_act[j] * scale_inv);
        }

        if (row_valid && col_valid) {
          if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
            reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_vec)->nt_store(
                &output_act_rowwise[row * output_cols + col_start]);
          } else {
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              if (col_start + j < cols) {
                output_act_rowwise[row * output_cols + col_start + j] = out_vec.data.elt[j];
              }
            }
          }
        }

        if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
          const int scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
          scales_rowwise[scale_idx] = biased_exp;
        }
      }

      // --- Gate rowwise quantization (BWD only) ---
      if constexpr (IS_DGATED) {
        __builtin_assume(gate_amax >= 0);
        const float scale_amax = rocm_subwarp_allreduce<SUBWARP_WIDTH>(gate_amax, rocm_op::max{});
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp<float>(biased_exp);

        Vec<OType, ELEMS_PER_THREAD> out_vec;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          out_vec.data.elt[j] = static_cast<OType>(computed_gate[j] * scale_inv);
        }

        if (row_valid && col_valid) {
          if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
            reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_vec)->nt_store(
                &output_gate_rowwise[row * output_cols + col_start]);
          } else {
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              if (col_start + j < cols) {
                output_gate_rowwise[row * output_cols + col_start + j] = out_vec.data.elt[j];
              }
            }
          }
        }

        if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
          const int scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE +
              DIVUP(cols, SCALE_DIM_X);
          scales_rowwise[scale_idx] = biased_exp;
        }
      }
    }
  }

  // COLWISE PATH: Shared memory for input + colwise output
  if constexpr (USE_COLWISE_SCALING) {
    extern __shared__ char dshmem_unaligned[];
    const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
    const uint64_t dshmem_aligned_as_uint =
        DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
    char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

    const size_t buff_elems           = SHMEM_DIM_Y * SHMEM_DIM_X;
    const size_t buff_elems_total     = BUFFERS_NUM * buff_elems;
    const size_t buff_size_aligned_in =
        DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
    const size_t buff_size_aligned_out =
        DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

    const size_t grad_mem   = (IS_DGATED ? buff_size_aligned_in : 0);
    const size_t in_act_mem = buff_size_aligned_in;
    const size_t in_mem     = in_act_mem + buff_size_aligned_in;  // act + gate

    IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
    IType *in_act_sh  = reinterpret_cast<IType *>(dshmem + grad_mem);
    IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

    OType *out_act_colwise_sh  = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
    OType *out_gate_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + buff_size_aligned_out);

    // For colwise cross-thread Y reduction
    __shared__ float stage_amax_sh[THREADS_PER_CHUNK_Y_COLWISE][CHUNK_DIM_X];

    __syncthreads();

  for (int it = 0; it < ITERATIONS; it++) {
    const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
    const int chunk_it_offset_x = chunk_offset_X;

    // === Load input to shmem ===
    if constexpr (IS_DGATED) {
      copy_2d_to_shared<IType, VECTOR_WIDTH_IN, IS_ALIGNED>(
          &in_grad_sh[0], grad_ptr, chunk_it_offset_x, chunk_it_offset_y,
          cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    }
    copy_2d_to_shared<IType, VECTOR_WIDTH_IN, IS_ALIGNED>(
        &in_act_sh[0], input_act, chunk_it_offset_x, chunk_it_offset_y,
        2*cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    copy_2d_to_shared<IType, VECTOR_WIDTH_IN, IS_ALIGNED>(
        &in_gate_sh[0], input_gate, chunk_it_offset_x, chunk_it_offset_y,
        2*cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);

    __syncthreads();

    if constexpr (USE_ROWWISE_SCALING) {
      const size_t row       = chunk_it_offset_y + tid_rowwise_Y;
      const size_t col_start = chunk_offset_X + thread_offset_X_rowwise;
  
      const bool row_valid = (row < rows);
      const bool col_valid = (col_start < cols);

      const int shmem_base = tid_rowwise_Y * SHMEM_DIM_X + thread_offset_X_rowwise;
      Vec<IType, ELEMS_PER_THREAD> act_vec, gate_vec;
      act_vec.load_from(&in_act_sh[shmem_base]);
      gate_vec.load_from(&in_gate_sh[shmem_base]);
      Vec<IType, ELEMS_PER_THREAD> grad_vec;
      if constexpr (IS_DGATED) {
        grad_vec.load_from(&in_grad_sh[shmem_base]);
      }

      float computed_act[ELEMS_PER_THREAD];
      float computed_gate[ELEMS_PER_THREAD];
      float act_amax  = 0;
      float gate_amax = 0;

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        float act_elt  = static_cast<float>(act_vec.data.elt[j]);
        float gate_elt = static_cast<float>(gate_vec.data.elt[j]);
        float grad_elt = IS_DGATED ? static_cast<float>(grad_vec.data.elt[j]) : 0.0f;

        compute_gated_activation<IS_DGATED, ParamOP, ActOP, DActOP, IType>(
            act_elt, gate_elt, grad_elt, p, computed_act[j], computed_gate[j]);

        act_amax = fmaxf(act_amax, fabsf(computed_act[j]));
        if constexpr (IS_DGATED) {
          gate_amax = fmaxf(gate_amax, fabsf(computed_gate[j]));
        }
      }

      // --- Act rowwise quantization ---
      {
        __builtin_assume(act_amax >= 0);
        const float scale_amax = rocm_subwarp_allreduce<SUBWARP_WIDTH>(act_amax, rocm_op::max{});
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp<float>(biased_exp);

        Vec<OType, ELEMS_PER_THREAD> out_vec;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          out_vec.data.elt[j] = static_cast<OType>(computed_act[j] * scale_inv);
        }

        if (row_valid && col_valid) {
          if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
            reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_vec)->nt_store(
                &output_act_rowwise[row * output_cols + col_start]);
          } else {
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              if (col_start + j < cols) {
                output_act_rowwise[row * output_cols + col_start + j] = out_vec.data.elt[j];
              }
            }
          }
        }

        if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
          const int scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
          scales_rowwise[scale_idx] = biased_exp;
        }
      }

      // --- Gate rowwise quantization (BWD only) ---
      if constexpr (IS_DGATED) {
        __builtin_assume(gate_amax >= 0);
        const float scale_amax = rocm_subwarp_allreduce<SUBWARP_WIDTH>(gate_amax, rocm_op::max{});
        const e8m0_t biased_exp =
            ptx::float_to_e8m0(scale_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_inv = ptx::exp2f_rcp<float>(biased_exp);

        Vec<OType, ELEMS_PER_THREAD> out_vec;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          out_vec.data.elt[j] = static_cast<OType>(computed_gate[j] * scale_inv);
        }

        if (row_valid && col_valid) {
          if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
            reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_vec)->nt_store(
                &output_gate_rowwise[row * output_cols + col_start]);
          } else {
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              if (col_start + j < cols) {
                output_gate_rowwise[row * output_cols + col_start + j] = out_vec.data.elt[j];
              }
            }
          }
        }

        if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
          const int scale_idx = row * scale_stride_rowwise +
              scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE +
              DIVUP(cols, SCALE_DIM_X);
          scales_rowwise[scale_idx] = biased_exp;
        }
      }

      {
        Vec<IType, ELEMS_PER_THREAD> cached_act, cached_gate;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          cached_act.data.elt[j] = static_cast<IType>(computed_act[j]);
        }
        cached_act.store_to(&in_act_sh[shmem_base]);
        if constexpr (IS_DGATED) {
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            cached_gate.data.elt[j] = static_cast<IType>(computed_gate[j]);
          }
          cached_gate.store_to(&in_gate_sh[shmem_base]);
        }
      }
    }

    if constexpr (USE_COLWISE_SCALING) {
      if constexpr (USE_ROWWISE_SCALING) {
        __syncthreads();
      }

      const bool col_out_of_bounds = (chunk_offset_X + tid_colwise_X >= cols);
      const size_t row_base = chunk_it_offset_y;
      const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + it;

      float after_dact_reg[BUFFER_STAGES_NUM_COLWISE];
      float after_dgate_reg[BUFFER_STAGES_NUM_COLWISE];

      float thread_Y_mx_block_amax      = 0.0f;
      float thread_Y_mx_block_amax_gate = 0.0f;

      for (int stage = 0; stage < BUFFER_STAGES_NUM_COLWISE; ++stage) {
        const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_COLWISE;
        const int shmem_offset_y = tid_colwise_Y + stage_offset_Y;
        const int shmem_idx      = shmem_offset_y * SHMEM_DIM_X + tid_colwise_X;

        if constexpr (USE_ROWWISE_SCALING) {
          after_dact_reg[stage] = static_cast<float>(in_act_sh[shmem_idx]);
          if constexpr (IS_DGATED) {
            after_dgate_reg[stage] = static_cast<float>(in_gate_sh[shmem_idx]);
          }
        } else {
          float act_elt  = static_cast<float>(in_act_sh[shmem_idx]);
          float gate_elt = static_cast<float>(in_gate_sh[shmem_idx]);
          float grad_elt = 0.0f;
          if constexpr (IS_DGATED) {
            grad_elt = static_cast<float>(in_grad_sh[shmem_idx]);
          }
          compute_gated_activation<IS_DGATED, ParamOP, ActOP, DActOP, IType>(
              act_elt, gate_elt, grad_elt, p, after_dact_reg[stage], after_dgate_reg[stage]);
        }

        __builtin_assume(thread_Y_mx_block_amax >= 0);
        thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, fabsf(after_dact_reg[stage]));
        if constexpr (IS_DGATED) {
          __builtin_assume(thread_Y_mx_block_amax_gate >= 0);
          thread_Y_mx_block_amax_gate =
              fmaxf(thread_Y_mx_block_amax_gate, fabsf(after_dgate_reg[stage]));
        }
      }

      const bool row_out_of_bounds = (row_base >= rows);
      const bool out_of_bounds     = (col_out_of_bounds || row_out_of_bounds);

      if constexpr (IS_DGATED) {
        if (tid_colwise_Y > 0) {
          stage_amax_sh[tid_colwise_Y][tid_colwise_X] = thread_Y_mx_block_amax_gate;
        }
        __syncthreads();
        if (tid_colwise_Y == 0) {
#pragma unroll
          for (int y = 1; y < THREADS_PER_CHUNK_Y_COLWISE; ++y) {
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
        const float scale_reciprocal = ptx::exp2f_rcp<float>(biased_exponent);

        if ((tid_colwise_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X + cols;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < BUFFER_STAGES_NUM_COLWISE; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_COLWISE;
          const int shmem_idx = (tid_colwise_Y + stage_offset_Y) * SHMEM_DIM_X + tid_colwise_X;
          out_gate_colwise_sh[shmem_idx] =
              static_cast<OType>(scale_reciprocal * after_dgate_reg[stage]);
        }
      }

      {
        if (tid_colwise_Y > 0) {
          stage_amax_sh[tid_colwise_Y][tid_colwise_X] = thread_Y_mx_block_amax;
        }
        __syncthreads();
        if (tid_colwise_Y == 0) {
#pragma unroll
          for (int y = 1; y < THREADS_PER_CHUNK_Y_COLWISE; ++y) {
            thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, stage_amax_sh[y][tid_colwise_X]);
          }
          stage_amax_sh[0][tid_colwise_X] = thread_Y_mx_block_amax;
        }
        __syncthreads();

        const float mx_block_Y_amax = stage_amax_sh[0][tid_colwise_X];
        __builtin_assume(mx_block_Y_amax >= 0);

        const e8m0_t biased_exponent =
            ptx::float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal = ptx::exp2f_rcp<float>(biased_exponent);

        if ((tid_colwise_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X;

          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < BUFFER_STAGES_NUM_COLWISE; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_COLWISE;
          const int shmem_idx = (tid_colwise_Y + stage_offset_Y) * SHMEM_DIM_X + tid_colwise_X;
          out_act_colwise_sh[shmem_idx] =
              static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
        }
      }
    }

    __syncthreads();

    bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH_OUT, IS_ALIGNED>(
        &out_act_colwise_sh[0], output_act_colwise, chunk_it_offset_x,
        chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    if constexpr (IS_DGATED) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH_OUT, IS_ALIGNED>(
          &out_gate_colwise_sh[0], output_gate_colwise, chunk_it_offset_x,
          chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    }
    __syncthreads();
  }
  }
}
