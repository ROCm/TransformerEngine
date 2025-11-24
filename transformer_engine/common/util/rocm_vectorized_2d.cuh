/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include "../util/vectorized_pointwise.h"

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

        if (g_row < total_rows) {
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

    const T* current_sh_row_base_ptr = sh_ptr_base + l_y * chunk_dim_x;
    VectorizedLoader<T, N_VEC, ALIGNED_ACCESS> shared_loader(current_sh_row_base_ptr, chunk_dim_x);

    T* current_g_row_base_ptr = g_ptr + g_row * g_stride;
    VectorizedStorer<T, N_VEC, ALIGNED_ACCESS> global_storer(current_g_row_base_ptr, total_cols);

    shared_loader.load(l_x_vec, chunk_dim_x);

    if (g_row < total_rows) {
      global_storer.storage_.scratch_ = shared_loader.storage_.scratch_;
      global_storer.store(g_col_primitive_start / N_VEC, total_cols);
    }
  }
}
} // namespace transformer_engine
