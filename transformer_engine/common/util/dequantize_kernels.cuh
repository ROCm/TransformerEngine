/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_kernels.cuh
 *  \brief CUDA kernels to cast from MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_

#include <cuda.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cudaTypedefs.h>
#endif //#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <limits>

#include "../common.h"
#include "cuda_runtime.h"
#include "../transpose/cast_transpose.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/transpose.h"
#ifdef __HIP_PLATFORM_AMD__
#include "rocm_dequantize_kernels.cuh"
#include "tdm.cuh"
#endif

namespace transformer_engine {

namespace dequantization {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t BUFFERS_NUM = 2;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 16;
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = CHUNK_DIM_X;   // 128

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //   8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     // 128
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //   8 = 128 / 16

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_kernel(
#ifdef __HIP_PLATFORM_AMD__
                            const IType *__restrict__ input_ptr,
                            OType *__restrict__ output_ptr,
#else
                            const __grid_constant__ CUtensorMap tensor_map_input,
                            const __grid_constant__ CUtensorMap tensor_map_output,
#endif
                            const e8m0_t *const scales_ptr, const size_t rows, const size_t cols,
                            const size_t scales_stride) {
#if defined(__gfx1250__) || ((defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);  // 2 = 32 / 16

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType out_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
#ifndef __HIP_PLATFORM_AMD__
  // TMA mbarriers require the expected byte count to know when the async copy is done.
  // TDM does not need this — it uses s_wait_tensorcnt which counts outstanding ops, not bytes.
  constexpr int transaction_size = shmem_buff_size;
#endif

  const bool is_master_thread = (threadIdx.x == 0);

#ifndef __HIP_PLATFORM_AMD__
// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
  constexpr int iteration_zero = 0;
  constexpr int buffer_zero = 0;
  if (is_master_thread) {
    const int chunk_stage_offset_Y = chunk_offset_Y;
    const int chunk_stage_offset_X = chunk_offset_X;
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(&in_sh[buffer_zero]),
        reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
        chunk_stage_offset_Y, &mbar[iteration_zero]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[iteration_zero], transaction_size);

  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[iteration_zero]);
  }
#else  // __HIP_PLATFORM_AMD__ — TDM prefetch
  constexpr uint32_t deq_in_data_sz = tdm::get_data_size_from_bits(sizeof(IType) * 8);
  constexpr uint32_t deq_out_data_sz = tdm::get_data_size_from_bits(sizeof(OType) * 8);

  // Prefetch first iteration
  tdm::copy_2d_to_shared(&in_sh[0][0][0], input_ptr,
                         chunk_offset_X, chunk_offset_Y,
                         SHMEM_DIM_X, SHMEM_DIM_Y,
                         cols, rows, cols, deq_in_data_sz);
#endif  // __HIP_PLATFORM_AMD__

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const int buff = iter % BUFFERS_NUM;
    const int next_iter = iter + 1;
    if (next_iter < ITERATIONS) {
#ifndef __HIP_PLATFORM_AMD__
      if (is_master_thread) {
        const int next_buff = next_iter % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
            chunk_it_offset_y, &mbar[next_iter]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
#else  // __HIP_PLATFORM_AMD__ — TDM prefetch next iteration
      {
        const int next_buff = next_iter % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        tdm::copy_2d_to_shared(&in_sh[next_buff][0][0], input_ptr,
                               chunk_it_offset_x, chunk_it_offset_y,
                               SHMEM_DIM_X, SHMEM_DIM_Y,
                               cols, rows, cols, deq_in_data_sz);
      }
#endif  // __HIP_PLATFORM_AMD__
    }

#ifndef __HIP_PLATFORM_AMD__
    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);
#else
    // Wait for current buffer's loads (and any prior stores) to complete,
    // but keep the just-issued prefetch for the next buffer alive.
    if (next_iter < ITERATIONS) {
      tdm::wait_tensorcnt_1();  // 1 prefetch load in flight
    } else {
      tdm::wait_tensorcnt_0();  // Last iteration — drain all
    }
    __syncthreads();
#endif

    const int scale_offset_Y =
        USE_ROWWISE_SCALING ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
                            : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

    const int scale_offset_X =
        USE_ROWWISE_SCALING
            ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
            : (scales_colwise_chunk_offset_X + tid_colwise_X);

    const int scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
    const e8m0_t biased_exponent = scales_ptr[scale_idx];
    const float block_scale = ptx::exp2f(biased_exponent);

    if constexpr (USE_ROWWISE_SCALING) {
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;

      const int shmem_offset_y = thread_offset_Y;
      const int shmem_offset_x = thread_offset_X_rowwise;
      in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
        out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[buff][shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y; ++i) {
        const float elt = static_cast<float>(in_sh[buff][i][tid_colwise_X]);
        out_sh[buff][i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
      }
    }

#ifndef __HIP_PLATFORM_AMD__
    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();
    }
#else  // __HIP_PLATFORM_AMD__ — TDM store
    __syncthreads();
    {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      tdm::store_2d_to_global(&out_sh[buff][0][0], output_ptr,
                              chunk_it_offset_x, chunk_it_offset_y,
                              SHMEM_DIM_X, SHMEM_DIM_Y,
                              cols, rows, cols, deq_out_data_sz);
      tdm::wait_tensorcnt_0();
      __syncthreads();
    }
#endif  // __HIP_PLATFORM_AMD__
  }

#ifndef __HIP_PLATFORM_AMD__
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  parity ^= 1;

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  // TDM does not use mbarriers — it uses s_wait_tensorcnt, so no barrier destroy is needed.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#else
  tdm::wait_tensorcnt_0();
#endif
#endif  // #if defined(__gfx1250__) || ((defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
}

static void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(OType);
          detail::DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
              reinterpret_cast<const IType *>(input.data.dptr), nullptr,
              reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

static void mxfp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  bool use_rowwise_scaling = input.has_data();
  bool use_colwise_scaling = input.has_columnwise_data();
#ifndef __HIP_PLATFORM_AMD__
  checkCuDriverContext(stream);
#endif

  const auto &input_shape = input.data.shape;
  NVTE_CHECK(input_shape.size() >= 2, "Input must have at least 2 dimensions.");

  if (use_rowwise_scaling) {
    NVTE_CHECK(input.has_data(), "Cannot dequantize tensor without rowwise data.");
    NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  }

  if (use_colwise_scaling) {
    NVTE_CHECK(input.has_columnwise_data(), "Cannot dequantize tensor without columnwise data.");
    NVTE_CHECK(is_fp8_dtype(input.columnwise_data.dtype), "Input must have FP8 type.");
  }

  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? 32 : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? 32 : 1;

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);

  const size_t unpadded_scales_Y_rowwise = rows;
  const size_t unpadded_scales_X_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t unpadded_scales_Y_colwise = DIVUP(rows, scale_dim_Y_colwise);
  const size_t unpadded_scales_X_colwise = cols;

  const size_t scales_Y_rowwise =
      DIVUP_TO_MULTIPLE(unpadded_scales_Y_rowwise, scale_tensor_alignment_Y_rowwise);
  const size_t scales_X_rowwise =
      DIVUP_TO_MULTIPLE(unpadded_scales_X_rowwise, scale_tensor_alignment_X_rowwise);
  const size_t scales_Y_colwise =
      DIVUP_TO_MULTIPLE(unpadded_scales_Y_colwise, scale_tensor_alignment_Y_colwise);
  const size_t scales_X_colwise =
      DIVUP_TO_MULTIPLE(unpadded_scales_X_colwise, scale_tensor_alignment_X_colwise);

  const e8m0_t *const scales_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(input.scale_inv.dptr)
                          : reinterpret_cast<e8m0_t *>(input.columnwise_scale_inv.dptr);

  const size_t scales_stride = use_rowwise_scaling ? scales_X_rowwise : scales_X_colwise;

  const SimpleTensor &input_data = use_rowwise_scaling ? input.data : input.columnwise_data;

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(chunks_X, chunks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
                  output->dtype(), OType,
#ifdef __HIP_PLATFORM_AMD__
                  // TDM flow — uses dequantization::dequantize_mxfp8_kernel
                  dequantization::dequantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X>
                      <<<grid, block, 0, stream>>>(
                          reinterpret_cast<const IType *>(input_data.dptr),
                          reinterpret_cast<OType *>(output->data.dptr),
                          scales_ptr, rows, cols, scales_stride);
#else // #ifdef __HIP_PLATFORM_AMD__
                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_output{};

                  create_2D_tensor_map(tensor_map_input, input_data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(input.dtype()));
                  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(output->dtype()));

                  dequantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X>
                  <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_output, scales_ptr,
                                               rows, cols, scales_stride);
#endif // #ifdef __HIP_PLATFORM_AMD__
          );                                                                  // NOLINT(*)
      );                                                                      // NOLINT(*)
  );                                                                          // NOLINT(*)
  );                                                                          // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}
}  // namespace dequantization

namespace detail {

void dequantize_helper(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if (is_tensor_scaling(input.scaling_mode)) {
    dequantization::fp8_dequantize(input, output, stream);
  } else if (is_mxfp_scaling(input.scaling_mode)) {
#if defined(__HIP_PLATFORM_AMD__) && defined(NVTE_ARCH_HAS_TDM)
    static const bool use_tdm_flow = [] {
      const char *env = std::getenv("NVTE_USE_TDM_FLOW");
      return env != nullptr && env[0] == '1' && env[1] == '\0' &&
             cuda::sm_arch_name().find("gfx1250") != std::string::npos;
    }();
    if (use_tdm_flow) {
      dequantization::mxfp8_dequantize(input, output, stream);
    } else {
      rocm_mxfp8_dequantize(input, output, stream);
    }
#elif defined(__HIP_PLATFORM_AMD__)
    rocm_mxfp8_dequantize(input, output, stream);
#else
    if (is_supported_by_CC_100()) {
      dequantization::mxfp8_dequantize(input, output, stream);
    } else {
      NVTE_ERROR("MXFP8 Dequantization is NOT supported by architectures < 10.0");
    }
#endif
  } else {
    // TODO(kwyss): Move dequantization code from torch to C++ for NVTE_BLOCK_SCALING
    NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace detail

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
