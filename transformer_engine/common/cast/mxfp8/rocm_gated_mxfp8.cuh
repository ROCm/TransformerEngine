/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once
// drop-in rocm replacement for mxfp8 gated quantize kernel

constexpr size_t ALIGNMENT_SIZE = 128;
// TODO: Identify optimal chunk/thread size for MI350+
constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t THREADS_PER_CHUNK = 256;
constexpr size_t THREADS_PER_CHUNK_X = 64;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 256 / 64
constexpr size_t BUFFERS_NUM = 1; // No async load for HIP
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //   4 = 128 / 32
static_assert(ITERATIONS >= 1);

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X, bool IS_ALIGNED>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_gated_mxfp8_kernel(
      const IType *grad_ptr,
      const IType *input_act,
      const IType *input_gate,
      OType *output_act_rowwise,
      OType *output_gate_rowwise,
      OType *output_act_colwise,
      OType *output_gate_colwise,
      e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
      const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
      const size_t scale_stride_colwise, const ParamOP p) {
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  constexpr size_t VECTOR_WIDTH = (IS_ALIGNED ?: 2) * 8 / sizeof(OType);

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const bool col_out_of_bounds = (chunk_offset_X + thread_offset_X >= cols);

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  const size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  const size_t buff_elems_total = BUFFERS_NUM * buff_elems;
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

  __shared__ float stage_amax_sh[THREADS_PER_CHUNK_Y][CHUNK_DIM_X];

  __syncthreads();

  for (int it = 0; it < ITERATIONS; it++) {
    const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
    const int chunk_it_offset_x = chunk_offset_X;
    const size_t row_base = chunk_it_offset_y; 

    // Initiate bulk tensor copy
    if constexpr (IS_DGATED) {
      copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_grad_sh[0], grad_ptr, chunk_it_offset_x, chunk_it_offset_y,
                        cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    }

    // Act
    copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_act_sh[0], input_act, chunk_it_offset_x, chunk_it_offset_y,
                      2*cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
    
    // Gate
    copy_2d_to_shared<IType, VECTOR_WIDTH, IS_ALIGNED>(&in_gate_sh[0], input_gate, chunk_it_offset_x, chunk_it_offset_y,
                      2*cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);

    __syncthreads();

    const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + it;
    const int iteration_scale_rowwise_offset_Y = scales_rowwise_chunk_offset_Y + it * BUFFER_DIM_Y;

    float after_dact_reg[BUFFER_STAGES_NUM];
    float after_dgate_reg[BUFFER_STAGES_NUM];
    float thread_Y_mx_block_amax = 0.0f;
    float thread_Y_mx_block_amax_gate = 0.0f;

    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      const size_t row = row_base + shmem_offset_y;
      const bool row_out_of_bounds = (row >= rows);
      const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

      float act_elt = static_cast<float>(in_act_sh[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh[shmem_idx]);

      bool dgate_elt = true;  // gating is ideally an identity function
      if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
        // In case of GPT OSS, clamp the activation and gate values
        dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;  // Derivative of clamp
        gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
      }

      if constexpr (IS_DGATED) {
        float grad_elt = static_cast<float>(in_grad_sh[shmem_idx]);
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
        
        after_dact_reg[stage] = dact_x * grad_elt * gate_elt;
        after_dgate_reg[stage] = dgate_elt ? act_x * grad_elt : 0.0f;
      } else {
        after_dact_reg[stage] = ActOP(act_elt, p) * gate_elt;
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
          for (int y = 1; y < THREADS_PER_CHUNK_Y; ++y) {
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
        for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X;
          const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

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
        for (int y = 1; y < THREADS_PER_CHUNK_Y; ++y) {
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
      for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;
        const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

        out_act_colwise_sh[shmem_idx] =
            static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
      }
    }

    __syncthreads();

    if constexpr (USE_ROWWISE_SCALING) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_act_rowwise_sh[0], output_act_rowwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      if constexpr (IS_DGATED) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_gate_rowwise_sh[0], output_gate_rowwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      }
    }
    
    if constexpr (USE_COLWISE_SCALING) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_act_colwise_sh[0], output_act_colwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      if constexpr (IS_DGATED) {
      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH, IS_ALIGNED>(&out_gate_colwise_sh[0], output_gate_colwise, chunk_it_offset_x,
                                      chunk_it_offset_y, output_cols, SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      }
    }
    __syncthreads();
  }
}
