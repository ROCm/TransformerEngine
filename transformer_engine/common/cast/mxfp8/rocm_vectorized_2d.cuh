/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "../../util/vectorized_pointwise.h"
#include "../../util/rocm_device_utils.cuh"

namespace transformer_engine {
// These 2d copy functions replace TMA tensormap async copies for AMD GPUs.
template <typename T, int N_VEC, bool ALIGNED_ACCESS>
__device__ inline void copy_2d_to_shared(T *sh_ptr_base, const T *g_ptr, size_t g_start_col,
                                          size_t g_start_row, size_t g_stride, size_t chunk_dim_y,
                                          size_t chunk_dim_x, size_t total_rows,
                                          size_t total_cols) {
    size_t chunk_dim_x_vec_elements = (chunk_dim_x + N_VEC - 1) / N_VEC;
    const size_t l_idx = threadIdx.x;

    for (size_t i_vec = l_idx; i_vec < chunk_dim_y * chunk_dim_x_vec_elements; i_vec += blockDim.x) {
        size_t l_y = (i_vec / chunk_dim_x_vec_elements);
        size_t l_x_vec = (i_vec % chunk_dim_x_vec_elements);

        size_t g_row = g_start_row + l_y;
        size_t g_col_primitive_start = g_start_col + l_x_vec * N_VEC;

        if (g_row < total_rows && g_col_primitive_start < total_cols) {
            const T* current_g_row_base_ptr = g_ptr + g_row * g_stride;
            VectorizedLoader<T, N_VEC, ALIGNED_ACCESS>global_loader(current_g_row_base_ptr, total_cols);

            T* current_sh_row_base_ptr = sh_ptr_base + l_y * chunk_dim_x;
            VectorizedStorer<T, N_VEC, ALIGNED_ACCESS>shared_storer(current_sh_row_base_ptr, chunk_dim_x);

            global_loader.load(g_col_primitive_start / N_VEC, total_cols);
            shared_storer.storage_.scratch_ = global_loader.storage_.scratch_;
            shared_storer.store(l_x_vec, chunk_dim_x);

        } else {
            T* current_sh_row_base_ptr = sh_ptr_base + l_y * chunk_dim_x;
            VectorizedStorer<T, N_VEC, ALIGNED_ACCESS> shared_storer(current_sh_row_base_ptr, chunk_dim_x);

#pragma unroll
            for (int i = 0; i < N_VEC; ++i) {
                shared_storer.separate()[i] = static_cast<T>(0);
            }
            shared_storer.store(l_x_vec, chunk_dim_x);
        }
    }
}

template <typename T, int N_VEC, bool ALIGNED_ACCESS>
__device__ inline void bulk_tensor_2d_shared_to_global(const T *sh_ptr_base, T *g_ptr, size_t g_start_col,
                                                         size_t g_start_row, size_t g_stride, size_t chunk_dim_y,
                                                         size_t chunk_dim_x, size_t total_rows,
                                                         size_t total_cols) {
  const size_t chunk_dim_x_vec_elements = (chunk_dim_x + N_VEC - 1) / N_VEC;
  const size_t l_idx = threadIdx.x;

  for (size_t i_vec = l_idx; i_vec < chunk_dim_y * chunk_dim_x_vec_elements; i_vec += blockDim.x) {
    size_t l_y = (i_vec / chunk_dim_x_vec_elements);
    size_t l_x_vec = (i_vec % chunk_dim_x_vec_elements);

    size_t g_row = g_start_row + l_y;
    size_t g_col_primitive_start = g_start_col + l_x_vec * N_VEC;

    if (g_row < total_rows) {
      const T *sh_row = sh_ptr_base + l_y * chunk_dim_x;
      T *g_row_ptr = g_ptr + g_row * g_stride;

      if (ALIGNED_ACCESS || g_col_primitive_start + N_VEC <= total_cols) {
        NTVec<T, N_VEC> v;
        v.load(sh_row + l_x_vec * N_VEC);
        v.nt_store(g_row_ptr + g_col_primitive_start);
      } else {
        for (int i = 0; i < N_VEC; i++) {
          if (g_col_primitive_start + i < total_cols) {
            g_row_ptr[g_col_primitive_start + i] = sh_row[l_x_vec * N_VEC + i];
          }
        }
      }
    }
  }
}
} // namespace transformer_engine
