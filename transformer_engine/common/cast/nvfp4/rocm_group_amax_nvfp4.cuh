/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/*! \file rocm_group_amax_nvfp4.cuh
 *  \brief Grouped per-row / per-column amax for row-scaled NVFP4 on ROCm.
 *
 *  Portable counterpart of the TMA-based group_compute_fused_amax: the input is one packed
 *  (sum_M, K) BF16 buffer and each 128x128 chunk lies inside a single expert, so one launch covers
 *  all experts and only the per-expert amax output pointers are selected per block.
 */

#ifndef TRANSFORMER_ENGINE_ROCM_GROUP_AMAX_NVFP4_CUH_
#define TRANSFORMER_ENGINE_ROCM_GROUP_AMAX_NVFP4_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <vector>

#include "../../common.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace rocm_group_amax {

constexpr int kMaxTensors = 64;
constexpr int kChunkDimY = 128;
constexpr int kChunkDimX = 128;
constexpr int kThreads = 256;

struct GroupAmaxArgs {
  float *row_amax_list[kMaxTensors];           // [M_i], null if the expert has no rowwise amax
  float *col_amax_list[kMaxTensors];           // [K], null if the expert has no columnwise amax
  int split_sections_range[kMaxTensors + 1];  // prefix sum of M_i
  int num_tensors;
};

__device__ __forceinline__ int get_tensor_id(const GroupAmaxArgs &args, const int row_offset) {
  int tensor_id = 0;
  while (args.split_sections_range[tensor_id + 1] <= row_offset) {
    ++tensor_id;
  }
  return tensor_id;
}

// Zero the per-expert amax vectors so the main kernel can atomicMax into them.
static __global__ void group_amax_zero_kernel(GroupAmaxArgs args, const int cols) {
  const int tensor_id = blockIdx.x;
  const int rows = args.split_sections_range[tensor_id + 1] - args.split_sections_range[tensor_id];
  float *row_amax = args.row_amax_list[tensor_id];
  float *col_amax = args.col_amax_list[tensor_id];
  const int stride = blockDim.x * gridDim.y;
  const int base = threadIdx.x + blockIdx.y * blockDim.x;
  if (row_amax != nullptr) {
    for (int i = base; i < rows; i += stride) {
      row_amax[i] = 0.0f;
    }
  }
  if (col_amax != nullptr) {
    for (int i = base; i < cols; i += stride) {
      col_amax[i] = 0.0f;
    }
  }
}

// One block per 128x128 chunk. Columns: thread t scans 64 rows of column t % 128. Rows: thread t
// scans 64 columns of row t / 2. The two halves are combined through shared memory and folded into
// the per-expert outputs with atomicMax (all values are non-negative).
template <bool DO_ROW, bool DO_COL>
__global__ void __launch_bounds__(kThreads)
    group_amax_kernel(const __nv_bfloat16 *__restrict__ input, const GroupAmaxArgs args,
                      const size_t cols) {
  __shared__ float row_partial[kThreads];
  __shared__ float col_partial[kThreads];

  const int tid = threadIdx.x;
  const int block_offset_Y = blockIdx.y * kChunkDimY;
  const int block_offset_X = blockIdx.x * kChunkDimX;
  const int tensor_id = get_tensor_id(args, block_offset_Y);
  const __nv_bfloat16 *chunk = input + static_cast<size_t>(block_offset_Y) * cols + block_offset_X;

  if constexpr (DO_COL) {
    const int col = tid % kChunkDimX;
    const int row_begin = (tid / kChunkDimX) * (kChunkDimY / 2);
    float amax = 0.0f;
    for (int r = row_begin; r < row_begin + kChunkDimY / 2; ++r) {
      amax = fmaxf(amax, fabsf(static_cast<float>(chunk[static_cast<size_t>(r) * cols + col])));
    }
    col_partial[tid] = amax;
  }
  if constexpr (DO_ROW) {
    const int row = tid / 2;
    const int col_begin = (tid % 2) * (kChunkDimX / 2);
    const __nv_bfloat16 *row_ptr = chunk + static_cast<size_t>(row) * cols;
    float amax = 0.0f;
    for (int c = col_begin; c < col_begin + kChunkDimX / 2; ++c) {
      amax = fmaxf(amax, fabsf(static_cast<float>(row_ptr[c])));
    }
    row_partial[tid] = amax;
  }
  __syncthreads();

  if (tid < kChunkDimX) {
    if constexpr (DO_COL) {
      float *col_amax = args.col_amax_list[tensor_id];
      if (col_amax != nullptr) {
        atomicMaxFloat(&col_amax[block_offset_X + tid],
                       fmaxf(col_partial[tid], col_partial[tid + kChunkDimX]));
      }
    }
    if constexpr (DO_ROW) {
      float *row_amax = args.row_amax_list[tensor_id];
      if (row_amax != nullptr) {
        const int local_row = block_offset_Y - args.split_sections_range[tensor_id] + tid;
        atomicMaxFloat(&row_amax[local_row], fmaxf(row_partial[2 * tid], row_partial[2 * tid + 1]));
      }
    }
  }
}

}  // namespace rocm_group_amax

inline void group_compute_fused_amax(const Tensor &input, std::vector<Tensor *> &output_list,
                                     const size_t *split_sections, const size_t num_tensors,
                                     cudaStream_t stream) {
  using namespace rocm_group_amax;

  const auto [rows, cols] = input.flat_2d_dims();
  NVTE_CHECK(input.dtype() == DType::kBFloat16, "Grouped fused amax requires BF16 input.");
  NVTE_CHECK(num_tensors <= static_cast<size_t>(kMaxTensors), "Number of tensors should be <= ",
             kMaxTensors);
  NVTE_CHECK(cols % kChunkDimX == 0, "Grouped fused amax requires cols to be a multiple of ",
             kChunkDimX, " (got ", cols, ").");

  // A direction runs if any non-empty expert allocates it; per-expert null buffers are skipped by
  // the kernel, so experts may opt in independently.
  bool do_row = false;
  bool do_col = false;
  for (size_t i = 0; i < num_tensors; ++i) {
    if (split_sections[i] == 0) {
      continue;
    }
    NVTE_CHECK(split_sections[i] % kChunkDimY == 0,
               "Grouped fused amax requires each split to be a multiple of ", kChunkDimY,
               " (split ", i, " = ", split_sections[i], ").");
    if (output_list[i]->amax.dptr != nullptr) {
      do_row = true;
    }
    if (output_list[i]->columnwise_amax.dptr != nullptr) {
      do_col = true;
    }
  }
  if (!do_row && !do_col) {
    return;
  }

  GroupAmaxArgs args;
  args.num_tensors = 0;
  args.split_sections_range[0] = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    if (split_sections[i] == 0) {
      continue;
    }
    args.row_amax_list[args.num_tensors] = reinterpret_cast<float *>(output_list[i]->amax.dptr);
    args.col_amax_list[args.num_tensors] =
        reinterpret_cast<float *>(output_list[i]->columnwise_amax.dptr);
    args.split_sections_range[args.num_tensors + 1] =
        args.split_sections_range[args.num_tensors] + static_cast<int>(split_sections[i]);
    args.num_tensors++;
  }
  NVTE_CHECK(args.split_sections_range[args.num_tensors] == static_cast<int>(rows),
             "Sum of split sections must equal the number of input rows.");

  const dim3 zero_grid(static_cast<unsigned>(args.num_tensors),
                       DIVUP(std::max(rows, cols), static_cast<size_t>(1024)));
  group_amax_zero_kernel<<<zero_grid, kThreads, 0, stream>>>(args, static_cast<int>(cols));
  NVTE_CHECK_CUDA(cudaGetLastError());

  const dim3 grid(static_cast<unsigned>(cols / kChunkDimX), static_cast<unsigned>(rows / kChunkDimY));
  const auto *input_ptr = reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr);
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      do_row, DO_ROW, TRANSFORMER_ENGINE_SWITCH_CONDITION(do_col, DO_COL, {
        group_amax_kernel<DO_ROW, DO_COL><<<grid, kThreads, 0, stream>>>(input_ptr, args, cols);
      }));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ROCM_GROUP_AMAX_NVFP4_CUH_
