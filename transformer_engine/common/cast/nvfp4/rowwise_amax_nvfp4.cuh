/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_ROWWISE_AMAX_NVFP4_CUH_
#define TRANSFORMER_ENGINE_ROWWISE_AMAX_NVFP4_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace rowwise_amax {

constexpr int ROWWISE_AMAX_BLOCK_SIZE = 256;
constexpr int ROWWISE_AMAX_SF_VEC_SIZE = 16;

// One block per row; the block reduces the abs-max over the row and thread 0
// writes it to output_rowwise_amax[row].
template <typename IType>
__global__ void __launch_bounds__(ROWWISE_AMAX_BLOCK_SIZE)
    compute_rowwise_amax_kernel(const int num_rows, const int num_cols,
                                const IType *__restrict__ input,
                                float *__restrict__ output_rowwise_amax,
                                const float *__restrict__ noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const int row_idx = blockIdx.x;
  if (row_idx >= num_rows) {
    return;
  }

  const IType *input_row = input + static_cast<size_t>(row_idx) * num_cols;
  float thread_max = 0.0f;
  for (int i = threadIdx.x; i < num_cols; i += ROWWISE_AMAX_BLOCK_SIZE) {
    thread_max = fmaxf(thread_max, fabsf(static_cast<float>(input_row[i])));
  }

  const float row_amax = reduce_max<ROWWISE_AMAX_BLOCK_SIZE / THREADS_PER_WARP>(
      thread_max, threadIdx.x / THREADS_PER_WARP);

  if (threadIdx.x == 0) {
    output_rowwise_amax[row_idx] = row_amax;
  }
}

template <typename IType>
void launch_compute_rowwise_amax(const int num_rows, const int num_cols, const IType *input,
                                 float *output_rowwise_amax, cudaStream_t stream,
                                 const float *noop = nullptr) {
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  const dim3 grid(num_rows);
  const dim3 block(ROWWISE_AMAX_BLOCK_SIZE);
  compute_rowwise_amax_kernel<IType>
      <<<grid, block, 0, stream>>>(num_rows, num_cols, input, output_rowwise_amax, noop);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace rowwise_amax

inline void compute_rowwise_amax(const Tensor &input, const Tensor *noop, Tensor *output,
                                 cudaStream_t stream) {
  using namespace rowwise_amax;

  const auto [rows, cols] = input.flat_2d_dims();
  NVTE_CHECK(cols % ROWWISE_AMAX_SF_VEC_SIZE == 0,
             "Row-scaled NVFP4 quantization requires last dim divisible by ",
             ROWWISE_AMAX_SF_VEC_SIZE, ".");

  auto *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  NVTE_CHECK(amax_ptr != nullptr, "Row-scaled rowwise amax tensor must be allocated.");
  NVTE_CHECK(output->amax.numel() == rows, "Row-scaled rowwise amax must have ", rows,
             " entries, got ", output->amax.shape, ".");

  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  if (input.dtype() == DType::kBFloat16) {
    const auto *input_ptr = reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr);
    launch_compute_rowwise_amax<__nv_bfloat16>(static_cast<int>(rows), static_cast<int>(cols),
                                               input_ptr, amax_ptr, stream, noop_ptr);
  } else if (input.dtype() == DType::kFloat16) {
    const auto *input_ptr = reinterpret_cast<const half *>(input.data.dptr);
    launch_compute_rowwise_amax<half>(static_cast<int>(rows), static_cast<int>(cols), input_ptr,
                                      amax_ptr, stream, noop_ptr);
  } else if (input.dtype() == DType::kFloat32) {
    const auto *input_ptr = reinterpret_cast<const float *>(input.data.dptr);
    launch_compute_rowwise_amax<float>(static_cast<int>(rows), static_cast<int>(cols), input_ptr,
                                       amax_ptr, stream, noop_ptr);
  } else {
    NVTE_ERROR(
        "Unsupported input dtype for row-scaled NVFP4 quantization. "
        "Expected BFloat16, Float16, or Float32.");
  }
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ROWWISE_AMAX_NVFP4_CUH_
