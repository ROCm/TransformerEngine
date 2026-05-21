/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;

  constexpr bool COMPUTE_DBIAS_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;
  constexpr size_t SHMEM_DIM_Y  = MXFP8_BUFFER_DIM_Y;
  constexpr size_t SHMEM_DIM_X  = BUFFER_DIM_X;

  constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;
  constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;
  constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;

  constexpr size_t BUFF_STAGES_NUM = MXFP8_BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;
  constexpr size_t ITERATIONS      = CHUNK_DIM_Y / MXFP8_BUFFER_DIM_Y;

  constexpr size_t SCALES_ROWWISE_PER_BLOCK_X = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t SCALES_COLWISE_PER_BLOCK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      //   2 = 32 / 16
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE;  //   2
  constexpr size_t VECTOR_WIDTH_IN  = ROCM_VEC_BYTES / sizeof(IType);   // BF16/FP16: 8, FP32: 4
  constexpr size_t VECTOR_WIDTH_OUT = ROCM_VEC_BYTES / sizeof(OType);   // FP8: 16

  const int block_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const int block_offset_X = block_id_X * CHUNK_DIM_X;
  const int scales_rowwise_block_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const int scales_rowwise_block_offset_X = block_id_X * SCALES_ROWWISE_PER_BLOCK_X;
  const int scales_colwise_block_offset_Y = block_id_Y * SCALES_COLWISE_PER_BLOCK_Y;
  const int scales_colwise_block_offset_X = block_id_X * CHUNK_DIM_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y         = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;

  const int dbias_rowwise_offset_Y       = dbias_y_offset + tid_rowwise_Y;
  const int dbias_rowwise_block_offset_X = block_offset_X + thread_offset_X_rowwise;
  const int dbias_colwise_offset_Y       = dbias_y_offset;
  const int dbias_colwise_block_offset_X = block_offset_X + tid_colwise_X;
  const int dbias_stride                 = cols;

  Vec<float, ELEMS_PER_THREAD> partial_dbias_rowwise;
  float partial_dbias_colwise = 0;
  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
      partial_dbias_rowwise.clear();
    }
  }

  float block_amax = 0;

  constexpr size_t ROWS_PER_THREAD = CHUNK_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;

  if constexpr (USE_ROWWISE_SCALING && !USE_COLWISE_SCALING) {
    const size_t col_start = block_offset_X + thread_offset_X_rowwise;
    const bool col_valid   = (col_start < cols);
#pragma unroll
    for (size_t r = 0; r < ROWS_PER_THREAD; r++) {
      const size_t row     = block_offset_Y + tid_rowwise_Y + r * THREADS_PER_CHUNK_Y_ROWWISE;
      const bool row_valid = (row < rows);

      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<IType, ELEMS_PER_THREAD> act_in;

      if (row_valid && col_valid) {
        if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
          in.load_from(&input_ptr[row * cols + col_start]);
          if constexpr (IS_DACT) {
            act_in.load_from(&act_input_ptr[row * cols + col_start]);
          }
        } else {
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            in.data.elt[j] = (col_start + j < cols) ? input_ptr[row * cols + col_start + j]
                                                    : static_cast<IType>(0);
          }
          if constexpr (IS_DACT) {
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              act_in.data.elt[j] = (col_start + j < cols) ? act_input_ptr[row * cols + col_start + j]
                                                          : static_cast<IType>(0);
            }
          }
        }
      }

      float thread_amax = 0;
      float in_compute[ELEMS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        const bool out_of_bounds = (!row_valid || !col_valid || col_start + j >= cols);
        float elt = static_cast<float>(in.data.elt[j]);
        if constexpr (IS_ACT) {
          elt = OP(elt, {});
        }
        if constexpr (IS_DACT) {
          float act_in_elt = static_cast<float>(act_in.data.elt[j]);
          elt *= OP(act_in_elt, {});
        }
        if constexpr (IS_DBIAS && COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
          if (!out_of_bounds) {
            partial_dbias_rowwise.data.elt[j] += elt;
          }
        }
        if constexpr (!std::is_same_v<IType, float>) {
          elt = static_cast<float>(static_cast<IType>(elt));
        }
        in_compute[j] = elt;
        if (!out_of_bounds) {
          thread_amax = fmaxf(thread_amax, fabsf(elt));
        }
      }

      __builtin_assume(block_amax >= 0);
      __builtin_assume(thread_amax >= 0);
      block_amax = fmaxf(block_amax, thread_amax);

      const float subwarp_amax = subwarp_reduce_max_broadcast<SUBWARP_WIDTH>(thread_amax);
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(subwarp_amax * Quantized_Limits<OType>::max_norm_rcp);

      {
        constexpr size_t SCALES_PER_GROUP = THREADS_PER_CHUNK_X_ROWWISE / THREADS_PER_SCALE_X_ROWWISE;
        static_assert(SCALES_PER_GROUP < 4 || SCALES_PER_GROUP % 4 == 0,
            "SCALES_PER_GROUP must be < 4 or a multiple of 4");
        uint32_t my_scale = static_cast<uint32_t>(biased_exponent);
        if constexpr (SCALES_PER_GROUP >= 4) {
#pragma unroll
          for (int g = 0; g < SCALES_PER_GROUP / 4; g++) {
            uint32_t s0 = __shfl_down(my_scale, (g*4+0) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
            uint32_t s1 = __shfl_down(my_scale, (g*4+1) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
            uint32_t s2 = __shfl_down(my_scale, (g*4+2) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
            uint32_t s3 = __shfl_down(my_scale, (g*4+3) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
            uint32_t packed = (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) | ((s3 & 0xFF) << 24);
            if (tid_rowwise_X == 0 && row_valid && col_valid) {
              const int scale_idx = row * scale_stride_rowwise + scales_rowwise_block_offset_X;
              reinterpret_cast<uint32_t*>(&scales_rowwise[scale_idx])[g] = packed;
            }
          }
        } else {
          if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
            const int scale_idx =
                row * scale_stride_rowwise +
                scales_rowwise_block_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
            scales_rowwise[scale_idx] = biased_exponent;
          }
        }
      }

      Vec<OType, ELEMS_PER_THREAD> out_c;
#ifdef HAS_CVT_4xFLOAT8
      {
        const float cvt_scale = (biased_exponent == 0) ? 1.0f : ptx::exp2f(biased_exponent);
        union {
          uint32_t packed[ELEMS_PER_THREAD / 4];
          mxfp8_v2i16_t v2i16[ELEMS_PER_THREAD / 4];
        } cvt_out{};
#pragma unroll
        for (int p = 0; p < ELEMS_PER_THREAD / 4; p++) {
          cvt_out.packed[p] = rocm_cvt_4xfloat8<OType>(
              in_compute[p*4+0], in_compute[p*4+1],
              in_compute[p*4+2], in_compute[p*4+3], cvt_scale);
        }
        memcpy(out_c.data.elt, cvt_out.packed, ELEMS_PER_THREAD * sizeof(OType));
      }
#else
      {
        const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          out_c.data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
        }
      }
#endif  // #ifdef HAS_CVT_4xFLOAT8

      if (row_valid && col_valid) {
        if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
          reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_c)->nt_store(
              &output_rowwise[row * cols + col_start]);
        } else {
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            if (col_start + j < cols) {
              output_rowwise[row * cols + col_start + j] = out_c.data.elt[j];
            }
          }
        }
      }
    }
  }

  if constexpr (USE_COLWISE_SCALING) {
    alignas(128) __shared__ IType in_sh[SHMEM_DIM_Y][SHMEM_DIM_X];
    alignas(128) __shared__ IType act_in_sh[IS_DACT ? SHMEM_DIM_Y : 1][IS_DACT ? SHMEM_DIM_X : 1];
    alignas(128) __shared__ OType out_colwise_sh[SHMEM_DIM_Y][SHMEM_DIM_X];

    const size_t col = block_offset_X + tid_colwise_X;
    const bool col_valid_colwise = (col < cols);

#pragma unroll
    for (int iter = 0; iter < ITERATIONS; iter++) {
      const size_t row_base = block_offset_Y + iter * MXFP8_BUFFER_DIM_Y;

      if constexpr (IS_DACT) {
        copy_2d_to_shared<IType, VECTOR_WIDTH_IN, IS_ALIGNED>(
            &act_in_sh[0][0], act_input_ptr,
            block_offset_X, row_base, cols,
            SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      }
      copy_2d_to_shared<IType, VECTOR_WIDTH_IN, IS_ALIGNED>(
          &in_sh[0][0], input_ptr,
          block_offset_X, row_base, cols,
          SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);
      __syncthreads();

      if constexpr (USE_ROWWISE_SCALING) {
        const size_t col_start = block_offset_X + thread_offset_X_rowwise;
        const bool col_valid   = (col_start < cols);

#pragma unroll
        for (int stage = 0; stage < BUFF_STAGES_NUM; stage++) {
          const int shmem_y    = thread_offset_Y + stage * THREADS_PER_CHUNK_Y_ROWWISE;
          const size_t row     = row_base + shmem_y;
          const bool row_valid = (row < rows);

          Vec<IType, ELEMS_PER_THREAD> in;
          Vec<IType, ELEMS_PER_THREAD> act_in;
          in.load_from(&in_sh[shmem_y][thread_offset_X_rowwise]);
          if constexpr (IS_DACT) {
            act_in.load_from(&act_in_sh[shmem_y][thread_offset_X_rowwise]);
          }

          float thread_amax = 0;
          float in_compute[ELEMS_PER_THREAD];

#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            const bool out_of_bounds = (!row_valid || !col_valid || col_start + j >= cols);
            float elt = static_cast<float>(in.data.elt[j]);
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            if constexpr (IS_DACT) {
              float act_in_elt = static_cast<float>(act_in.data.elt[j]);
              elt *= OP(act_in_elt, {});
            }
            if constexpr (IS_DBIAS && COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
              if (!out_of_bounds) {
                partial_dbias_rowwise.data.elt[j] += elt;
              }
            }
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            in_compute[j] = elt;
            if (!out_of_bounds) {
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
          }

          __builtin_assume(block_amax >= 0);
          __builtin_assume(thread_amax >= 0);
          block_amax = fmaxf(block_amax, thread_amax);

          const float subwarp_amax = subwarp_reduce_max_broadcast<SUBWARP_WIDTH>(thread_amax);
          const e8m0_t biased_exponent =
              ptx::float_to_e8m0(subwarp_amax * Quantized_Limits<OType>::max_norm_rcp);

          {
            constexpr size_t SCALES_PER_GROUP = THREADS_PER_CHUNK_X_ROWWISE / THREADS_PER_SCALE_X_ROWWISE;
            static_assert(SCALES_PER_GROUP < 4 || SCALES_PER_GROUP % 4 == 0,
                "SCALES_PER_GROUP must be < 4 or a multiple of 4");
            uint32_t my_scale = static_cast<uint32_t>(biased_exponent);
            if constexpr (SCALES_PER_GROUP >= 4) {
#pragma unroll
              for (int g = 0; g < SCALES_PER_GROUP / 4; g++) {
                uint32_t s0 = __shfl_down(my_scale, (g*4+0) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
                uint32_t s1 = __shfl_down(my_scale, (g*4+1) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
                uint32_t s2 = __shfl_down(my_scale, (g*4+2) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
                uint32_t s3 = __shfl_down(my_scale, (g*4+3) * THREADS_PER_SCALE_X_ROWWISE, THREADS_PER_CHUNK_X_ROWWISE);
                uint32_t packed = (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) | ((s3 & 0xFF) << 24);
                if (tid_rowwise_X == 0 && row_valid && col_valid) {
                  const int scale_idx = row * scale_stride_rowwise + scales_rowwise_block_offset_X;
                  reinterpret_cast<uint32_t*>(&scales_rowwise[scale_idx])[g] = packed;
                }
              }
            } else {
              if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0 && row_valid && col_valid) {
                const int scale_idx = row * scale_stride_rowwise +
                    scales_rowwise_block_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
                scales_rowwise[scale_idx] = biased_exponent;
              }
            }
          }

          Vec<OType, ELEMS_PER_THREAD> out_c;
#ifdef HAS_CVT_4xFLOAT8
          {
            const float cvt_scale = (biased_exponent == 0) ? 1.0f : ptx::exp2f(biased_exponent);
            union {
              uint32_t packed[ELEMS_PER_THREAD / 4];
              mxfp8_v2i16_t v2i16[ELEMS_PER_THREAD / 4];
            } cvt_out{};
#pragma unroll
            for (int p = 0; p < ELEMS_PER_THREAD / 4; p++) {
              cvt_out.packed[p] = rocm_cvt_4xfloat8<OType>(
                  in_compute[p*4+0], in_compute[p*4+1],
                  in_compute[p*4+2], in_compute[p*4+3], cvt_scale);
            }
            memcpy(out_c.data.elt, cvt_out.packed, ELEMS_PER_THREAD * sizeof(OType));
          }
#else
          {
            const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              out_c.data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
            }
          }
#endif  // #ifdef HAS_CVT_4xFLOAT8

          if (row_valid && col_valid) {
            if (IS_ALIGNED || col_start + ELEMS_PER_THREAD <= cols) {
              reinterpret_cast<const NTVec<OType, ELEMS_PER_THREAD>*>(&out_c)->nt_store(
                  &output_rowwise[row * cols + col_start]);
            } else {
#pragma unroll
              for (int j = 0; j < ELEMS_PER_THREAD; j++) {
                if (col_start + j < cols) {
                  output_rowwise[row * cols + col_start + j] = out_c.data.elt[j];
                }
              }
            }
          }

          if constexpr (IS_ACT || IS_DACT) {
            Vec<IType, ELEMS_PER_THREAD> cached;
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {
              cached.data.elt[j] = static_cast<IType>(in_compute[j]);
            }
            cached.store_to(&in_sh[shmem_y][thread_offset_X_rowwise]);
          }
        }
      }

      if constexpr (IS_ACT || IS_DACT) {
        __syncthreads();
      }

      if (threadIdx.x < CHUNK_DIM_X) {
        float in_compute[SCALE_DIM_Y];
        float amax = 0;

#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; i++) {
          const size_t row = row_base + i;
          const bool out_of_bounds = (!col_valid_colwise || row >= rows);

          float elt = static_cast<float>(in_sh[i][tid_colwise_X]);
          constexpr bool ACT_CACHED = USE_ROWWISE_SCALING && (IS_ACT || IS_DACT);
          if constexpr (!ACT_CACHED) {
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            if constexpr (IS_DACT) {
              float act_in_elt = static_cast<float>(act_in_sh[i][tid_colwise_X]);
              elt *= OP(act_in_elt, {});
            }
          }
          if constexpr (IS_DBIAS) {
            if (!out_of_bounds) {
              partial_dbias_colwise += elt;
            }
          }
          if constexpr (!ACT_CACHED) {
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
          }
          in_compute[i] = elt;
          if (!out_of_bounds) {
            amax = fmaxf(amax, fabsf(elt));
          }
        }

        __builtin_assume(block_amax >= 0);
        __builtin_assume(amax >= 0);
        block_amax = fmaxf(block_amax, amax);

        const e8m0_t biased_exponent = ptx::float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);

        if (col_valid_colwise && row_base < rows) {
          const int scale_idx =
              (scales_colwise_block_offset_Y + iter) * scale_stride_colwise + col;
          scales_colwise[scale_idx] = biased_exponent;
        }

#ifdef HAS_CVT_4xFLOAT8
        {
          const float cvt_scale = (biased_exponent == 0) ? 1.0f : ptx::exp2f(biased_exponent);
#pragma unroll
          for (int i = 0; i < SCALE_DIM_Y; i += 2) {
            uint32_t packed = rocm_cvt_4xfloat8<OType>(
                in_compute[i], in_compute[i+1], 0.0f, 0.0f, cvt_scale);
            OType val0, val1;
            memcpy(&val0, &packed, sizeof(OType));
            memcpy(&val1, reinterpret_cast<uint8_t *>(&packed) + 1, sizeof(OType));
            out_colwise_sh[i][tid_colwise_X] = val0;
            if (i + 1 < SCALE_DIM_Y) {
              out_colwise_sh[i+1][tid_colwise_X] = val1;
            }
          }
        }
#else
        {
          const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
#pragma unroll
          for (int i = 0; i < SCALE_DIM_Y; i++) {
            out_colwise_sh[i][tid_colwise_X] =
                static_cast<OType>(in_compute[i] * block_scale_inverse);
          }
        }
#endif  // #ifdef HAS_CVT_4xFLOAT8
      }

      __syncthreads();

      bulk_tensor_2d_shared_to_global<OType, VECTOR_WIDTH_OUT, IS_ALIGNED>(
          &out_colwise_sh[0][0], output_colwise,
          block_offset_X, row_base, cols,
          SHMEM_DIM_Y, SHMEM_DIM_X, rows, cols);

      __syncthreads();
    }
  }

  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
      constexpr size_t Y = THREADS_PER_CHUNK_Y_ROWWISE - 1;
      constexpr size_t X = THREADS_PER_CHUNK_X_ROWWISE;
      __shared__ float shmem_partial_dbias_rowwise[Y][X][ELEMS_PER_THREAD];

      if (tid_rowwise_Y > 0) {
        partial_dbias_rowwise.store_to(
            &shmem_partial_dbias_rowwise[tid_rowwise_Y - 1][tid_rowwise_X]);
      }
      __syncthreads();

      if (tid_rowwise_Y == 0) {
        Vec<float, ELEMS_PER_THREAD> other_row_dbias;
        const int dbias_offset = dbias_rowwise_offset_Y * dbias_stride + dbias_rowwise_block_offset_X;
        const int left_bound   = dbias_rowwise_block_offset_X;
        const int right_bound  = dbias_rowwise_block_offset_X + ELEMS_PER_THREAD - 1;

#pragma unroll
        for (int i = 0; i < Y; i++) {
          other_row_dbias.load_from(&shmem_partial_dbias_rowwise[i][tid_rowwise_X]);
#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            partial_dbias_rowwise.data.elt[j] += other_row_dbias.data.elt[j];
          }
        }

        if (right_bound < cols) {
          partial_dbias_rowwise.store_to(&dbias_workspace[dbias_offset]);
        } else if (left_bound < cols && right_bound >= cols) {
          const int in_bound_elts_count = cols - left_bound;
          partial_dbias_rowwise.store_to_elts(&dbias_workspace[dbias_offset], 0,
                                               in_bound_elts_count);
        }
      }
    } else {
      if (threadIdx.x < CHUNK_DIM_X) {
        const int dbias_offset = dbias_colwise_offset_Y * dbias_stride + dbias_colwise_block_offset_X;
        const bool col_out_of_bounds = (dbias_colwise_block_offset_X >= cols);
        if (!col_out_of_bounds) {
          dbias_workspace[dbias_offset] = partial_dbias_colwise;
        }
      }
    }
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    block_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (threadIdx.x == 0 && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }
