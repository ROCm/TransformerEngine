/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <transformer_engine/hadamard_transform.h>

#include <cuda/barrier>

#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#ifndef __HIP_PLATFORM_AMD__
#include "hadamard_transform_utils.cuh"
#endif

namespace transformer_engine {
namespace {

#ifndef __HIP_PLATFORM_AMD__
constexpr int kThreadsPerWarp = 32;
template <typename IType, int kHadamardDimension, int BUFF_DIM_Y, int BUFF_DIM_X,
          bool kReturnPreRhtAmax, bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__device__ __forceinline__ void ComputeKernel(uint32_t b_frag_i[4], uint32_t b_frag_t[4],
                                              IType* in_sh_ptr, uint32_t& local_pre_rht_amax_reg,
                                              uint32_t& local_amax_reg,
                                              uint32_t& local_amax_t_reg) {
  uint32_t a_frag[4];  // A matrix fragment
  uint32_t c_frag[4];  // Result fragment

  int warp_id = threadIdx.x / kThreadsPerWarp;
  int local_rank = (threadIdx.x % kThreadsPerWarp);

  int ld_row_idx = local_rank % kHadamardDimension;
  int ld_col_idx = local_rank / kHadamardDimension + warp_id * 2;
  int swizzle_idx = swizzle_128B_atom_32B(ld_row_idx, ld_col_idx);

  uint32_t temp_amax_reg;
  uint32_t temp_amax_t_reg;

  if (kReturnIdentityAmax) {
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);

    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag_i[0], b_frag_i[1], b_frag_i[2],
        b_frag_i[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax_reg);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_reg)
                 : "r"(local_amax_reg), "r"(temp_amax_reg));
  }

  if (kReturnTransposedAmax) {
    // TODO(Frank): This is not efficient, since we could directly load the
    // matrix in transposed layout.
    if (!kReturnIdentityAmax) {
      ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                         reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    }

    matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[3]);

    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnTransposedAmax>(
        a_frag[0], a_frag[2], a_frag[1], a_frag[3], b_frag_t[0], b_frag_t[1], b_frag_t[2],
        b_frag_t[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax_t_reg);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_t_reg)
                 : "r"(local_amax_t_reg), "r"(temp_amax_t_reg));
  }

  if (kReturnPreRhtAmax) {
    if (!kReturnIdentityAmax && !kReturnTransposedAmax) {
      ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                         reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    }

    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[0])
                 : "r"(a_frag[0]), "r"(a_frag[1]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[2])
                 : "r"(a_frag[2]), "r"(a_frag[3]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[0])
                 : "r"(a_frag[0]), "r"(a_frag[2]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_pre_rht_amax_reg)
                 : "r"(a_frag[0]), "r"(local_pre_rht_amax_reg));
  }
}

template <int kN>
__device__ __host__ constexpr int NextPowerOf2() {
  static_assert(kN > 0, "kN must be > 0");
  // Round up to the next power of 2 by counting leading zeros.
  return 1 << (32 - __builtin_clz(kN - 1));
}

template <int kNumWarps, bool kReturnPreRhtAmax, bool kReturnIdentityAmax,
          bool kReturnTransposedAmax>
__device__ __forceinline__ void ReduceMax(const float pre_rht_amax, const float identity_amax,
                                          const float transpose_amax, float* staging_for_pre_rht,
                                          float* staging_for_identity, float* staging_for_transpose,
                                          float* output_pre_rht_amax_ptr,
                                          float* output_identity_amax_ptr,
                                          float* output_transpose_amax_ptr, const int warpid) {
  // intra-warp reduction
  constexpr int kWarpSize = 32;
  int local_rank = threadIdx.x % 32;
  float warp_pre_rht_amax = kReturnPreRhtAmax ? warp_reduce_max<kWarpSize>(pre_rht_amax) : 0.0f;
  float warp_identity_amax = kReturnIdentityAmax ? warp_reduce_max<kWarpSize>(identity_amax) : 0.0f;
  float warp_transpose_amax =
      kReturnTransposedAmax ? warp_reduce_max<kWarpSize>(transpose_amax) : 0.0f;

  // inter-warp reduction
  if (threadIdx.x % 32 == 0) {
    if (kReturnPreRhtAmax) {
      staging_for_pre_rht[warpid] = warp_pre_rht_amax;
    }
    if (kReturnIdentityAmax) {
      staging_for_identity[warpid] = warp_identity_amax;
    }
    if (kReturnTransposedAmax) {
      staging_for_transpose[warpid] = warp_transpose_amax;
    }
  }
  __syncthreads();
  constexpr int kNumWarpsPow2 = NextPowerOf2<kNumWarps>();
  if (warpid == 0) {
    if (kReturnIdentityAmax) {
      float identity_accum = local_rank < kNumWarps ? staging_for_identity[local_rank] : 0.0f;
      identity_accum = warp_reduce_max<kNumWarpsPow2>(identity_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_identity_amax_ptr, identity_accum);
      }
    }
  }
  if (warpid == 1) {
    if (kReturnTransposedAmax) {
      float transpose_accum = local_rank < kNumWarps ? staging_for_transpose[local_rank] : 0.0f;
      transpose_accum = warp_reduce_max<kNumWarpsPow2>(transpose_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_transpose_amax_ptr, transpose_accum);
      }
    }
  }
  if (warpid == 2) {
    if (kReturnPreRhtAmax) {
      float pre_rht_accum = local_rank < kNumWarps ? staging_for_pre_rht[local_rank] : 0.0f;
      pre_rht_accum = warp_reduce_max<kNumWarpsPow2>(pre_rht_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_pre_rht_amax_ptr, pre_rht_accum);
      }
    }
  }
}

__launch_bounds__(1) __global__ void ZeroAmaxKernel(float* __restrict__ output_pre_rht_amax_ptr,
                                                    float* __restrict__ output_identity_amax_ptr,
                                                    float* __restrict__ output_transpose_amax_ptr) {
  if (output_pre_rht_amax_ptr != nullptr) {
    *output_pre_rht_amax_ptr = 0;
  }
  if (output_identity_amax_ptr != nullptr) {
    *output_identity_amax_ptr = 0;
  }
  if (output_transpose_amax_ptr != nullptr) {
    *output_transpose_amax_ptr = 0;
  }
}

template <typename IType, int kHadamardDimension, int CHUNK_DIM_Y, int CHUNK_DIM_X, int BUFF_DIM_Y,
          int BUFF_DIM_X, int THREADS_PER_CHUNK, int THREADS_PER_Y, bool kReturnPreRhtAmax,
          bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__global__ void HadamardAmaxTmaKernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                      float* __restrict__ output_pre_rht_amax_ptr,
                                      float* __restrict__ output_identity_amax_ptr,
                                      float* __restrict__ output_transpose_amax_ptr,
                                      uint16_t random_sign_mask, uint16_t random_sign_mask_t,
                                      uint64_t num_rows, uint64_t row_length) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y && CHUNK_DIM_Y % BUFF_DIM_Y == 0);
  static_assert(CHUNK_DIM_X >= BUFF_DIM_X && CHUNK_DIM_X % BUFF_DIM_X == 0);

  constexpr size_t STAGES_Y = CHUNK_DIM_Y / BUFF_DIM_Y;
  constexpr size_t STAGES_X = CHUNK_DIM_X / BUFF_DIM_X;

  constexpr int kNumWarps = (THREADS_PER_CHUNK * THREADS_PER_Y) / kThreadsPerWarp;

  const int input_block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int input_block_offset_X = blockIdx.x * CHUNK_DIM_X;

  extern __shared__ __align__(128) char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uint8_t* dshmem = reinterpret_cast<uint8_t*>((base_shmem_ptr + 127) & ~127ULL);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  constexpr size_t in_buff_size = BUFF_DIM_X * BUFF_DIM_Y * sizeof(IType);
  IType* in_sh_0 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;
  IType* in_sh_1 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;

  IType* in_shs[2] = {in_sh_0, in_sh_1};

  constexpr int shmem_buff_size = BUFF_DIM_X * BUFF_DIM_Y * sizeof(IType);

  const bool is_master_thread = (threadIdx.x == 0 && threadIdx.y == 0);

  // Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  uint64_t* mbar = reinterpret_cast<uint64_t*>(dshmem);
  dshmem += sizeof(uint64_t) * (STAGES_X * STAGES_Y);

  float* max_staging_identity = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_transpose = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_pre_rht = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;

  initialize_barriers<STAGES_X * STAGES_Y, THREADS_PER_CHUNK * THREADS_PER_Y>(mbar,
                                                                              is_master_thread);

  copy_2d_to_shared(in_shs[0], reinterpret_cast<const void*>(&tensor_map_input),
                    input_block_offset_X, input_block_offset_Y, shmem_buff_size, &mbar[0],
                    is_master_thread);

  uint32_t had_frag_i[4];
  uint32_t had_frag_t[4];
  get_hadamard_matrix_fragment<kReturnIdentityAmax, kReturnTransposedAmax, false, false>(
      had_frag_i, random_sign_mask, had_frag_t, random_sign_mask_t);

  float local_pre_rht_amax = 0.0;
  float local_amax = 0.0;
  float local_amax_t = 0.0;
  uint32_t local_pre_rht_amax_reg = *reinterpret_cast<uint32_t*>(&local_pre_rht_amax);
  uint32_t local_amax_reg = *reinterpret_cast<uint32_t*>(&local_amax);
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);

  for (int stage_y = 0; stage_y < STAGES_Y; ++stage_y) {
    for (int stage_x = 0; stage_x < STAGES_X; ++stage_x) {
      int stage = STAGES_X * stage_y + stage_x;

      const int next_stage = stage + 1;
      const int next_stage_x = stage_x + 1 == STAGES_X ? 0 : stage_x + 1;
      const int next_stage_y = stage_x + 1 == STAGES_X ? stage_y + 1 : stage_y;

      if (next_stage < STAGES_X * STAGES_Y) {
        const int input_global_offset_Y = input_block_offset_Y + next_stage_y * BUFF_DIM_Y;
        const int input_global_offset_X = input_block_offset_X + next_stage_x * BUFF_DIM_X;

        copy_2d_to_shared(in_shs[next_stage % 2],  // ping-pong
                          reinterpret_cast<const void*>(&tensor_map_input), input_global_offset_X,
                          input_global_offset_Y, shmem_buff_size, &mbar[next_stage],
                          is_master_thread);
      }

      ptx::fence_proxy_async_shared_cta();

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity(&mbar[stage], 0);

      const size_t compute_stage_x_num =
          BUFF_DIM_X / (kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp));
      const size_t compute_stage_y_num = BUFF_DIM_Y / (kHadamardDimension * THREADS_PER_Y);

      const size_t in_row_stride = BUFF_DIM_X;

      IType* in_sh_ptr = in_shs[stage % 2];

#pragma unroll
      for (size_t compute_stage_y = 0; compute_stage_y < compute_stage_y_num; compute_stage_y++) {
        const int row_idx_offset = (compute_stage_y * kHadamardDimension * THREADS_PER_Y +
                                    threadIdx.y * kHadamardDimension);
        const int in_row_offset = row_idx_offset * in_row_stride;

#pragma unroll
        for (size_t compute_stage_x = 0; compute_stage_x < compute_stage_x_num; compute_stage_x++) {
          ComputeKernel<IType, kHadamardDimension, BUFF_DIM_Y, BUFF_DIM_X, kReturnPreRhtAmax,
                        kReturnIdentityAmax, kReturnTransposedAmax>(
              had_frag_i, had_frag_t,
              in_sh_ptr + in_row_offset +
                  (compute_stage_x * kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp)),
              local_pre_rht_amax_reg, local_amax_reg, local_amax_t_reg);
        }

        // Ensure all threads have finished their computation before new data over-writes the shared
        // memory.
        __syncthreads();
      }
    }
  }

  const int warpid = (threadIdx.x + threadIdx.y * blockDim.x) / kThreadsPerWarp;

  if constexpr (kReturnPreRhtAmax) {
    unpack_max_of_packed_bf16(local_pre_rht_amax_reg, local_pre_rht_amax);
  }
  if constexpr (kReturnIdentityAmax) {
    unpack_max_of_packed_bf16(local_amax_reg, local_amax);
  }
  if constexpr (kReturnTransposedAmax) {
    unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
  }

  ReduceMax<kNumWarps, kReturnPreRhtAmax, kReturnIdentityAmax, kReturnTransposedAmax>(
      local_pre_rht_amax, local_amax, local_amax_t, max_staging_pre_rht, max_staging_identity,
      max_staging_transpose, output_pre_rht_amax_ptr, output_identity_amax_ptr,
      output_transpose_amax_ptr, warpid);

  destroy_barriers<STAGES_X * STAGES_Y>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("Kernel is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename T, int kHadamardDimension, bool kComputeIdentity, bool kComputeTransposed,
          bool kReturnIdentity, bool kReturnTransposed, bool kUpdateIdentityAmax,
          bool kUpdateTransposeAmax, bool kOutputTrueTransposed>
__global__ void HadamardTransformKernel(const T* __restrict__ input, T* __restrict__ output,
                                        T* __restrict__ output_t, uint16_t random_sign_mask,
                                        uint16_t random_sign_mask_t, uint64_t num_input_rows,
                                        uint64_t num_input_cols, float* __restrict__ amax,
                                        float* __restrict__ amax_t, bool inverse_hadamard) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(kHadamardDimension == 16, "Currently only hadamard dimension 16 is supported.");

  // The whole threadblock will share the same smem.
  extern __shared__ __align__(16) T smem[];

  // Each 32 threads process a 16x16 matrix. There is a (y, z) grid of 16x16.
  // If y = 4, z = 4, then each threadblock is processing a 4x4 grid of 16x16 matrices.
  int32_t tid = threadIdx.x;
  int32_t warp_id = threadIdx.y * blockDim.z + threadIdx.z;
  int32_t local_bx = threadIdx.y;
  int32_t local_by = threadIdx.z;

  // Define the register fragments
  uint32_t a_frag[4];    // A matrix fragment
  uint32_t b_frag_i[4];  // Transposed Hadamard matrix fragment, used for A @ B(col major)
  uint32_t b_frag_t[4];  // Hadamard matrix fragment, used for A.T @ B.T(col major)
  uint32_t c_frag[4];    // Result fragment

  // row and col for each thread. 32 threads will work together in 128 chunk to
  // load the data from global memory to shared memory.
  uint32_t row = tid / (kHadamardDimension * sizeof(T) / sizeof(uint4));
  uint32_t col = tid % (kHadamardDimension * sizeof(T) / sizeof(uint4));

  uint32_t smem_index = tid;

  uint32_t input_start_col = (blockIdx.x * blockDim.y + local_bx) * kHadamardDimension;
  uint32_t input_start_row = (blockIdx.y * blockDim.z + local_by) * kHadamardDimension;

  bool load = (input_start_col < num_input_cols) && (input_start_row < num_input_rows);
  if (!load) {
    // Out of bound, we are returning early. No thread divergence since the whole warp
    // will return early.
    return;
  }

  uint64_t global_offset = input_start_col + input_start_row * num_input_cols;
  uint64_t global_offset_t =
      kOutputTrueTransposed ? (input_start_row + input_start_col * num_input_rows) : global_offset;

  T* base_smem = smem + kHadamardDimension * kHadamardDimension * warp_id;

  uint32_t* smem_b32 = reinterpret_cast<uint32_t*>(base_smem);
  uint4* smem_b128 = reinterpret_cast<uint4*>(base_smem);

  // Asynchronously load the data from global memory to shared memory.
  const uint4* input_b128 = reinterpret_cast<const uint4*>(input + global_offset);
  // Each 16x16 chunk is divided into 4 8x8 matrices, we are trying to load each
  // 8x8 chunks consecutively into the smem, so we could leverage ldmatrix m8n8x4
  // to load the data in the tensor core swizzled format.
  __pipeline_memcpy_async(&smem_b128[smem_index],
                          &input_b128[row * num_input_cols / (sizeof(uint4) / sizeof(T)) + col],
                          sizeof(uint4));
  __pipeline_commit();  // Commit the memcpy. Wait when we are in the computation.

  if (inverse_hadamard) {
    get_hadamard_matrix_fragment<kComputeIdentity, kComputeTransposed,
                                 /*kInverseHadamard=*/true,
                                 /*kInverseHadamardTransposed=*/true>(b_frag_i, random_sign_mask,
                                                                      b_frag_t, random_sign_mask_t);
  } else {
    get_hadamard_matrix_fragment<kComputeIdentity, kComputeTransposed,
                                 /*kInverseHadamard=*/false,
                                 /*kInverseHadamardTransposed=*/false>(
        b_frag_i, random_sign_mask, b_frag_t, random_sign_mask_t);
  }

  float local_amax = 0.0;
  float local_amax_t = 0.0;
  uint32_t local_amax_reg = *reinterpret_cast<uint32_t*>(&local_amax);
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);
  __pipeline_wait_prior(0);

  __syncwarp();  // ensure all lanes finished their cp.async before reading smem

  // Load the A to a_frag.
  if constexpr (kComputeIdentity) {
    load_matrix_16x16_from_shared<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3], smem_b32,
                                         kHadamardDimension);

    // 16x16 @ 16x16 leveraging all threads in the warp.
    mma_m16_n16_k16_b16_b16_b16_noacc<kUpdateIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag_i[0], b_frag_i[1], b_frag_i[2],
        b_frag_i[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], local_amax_reg);

    // Store the result to the shared memory in non-transposed order.
    if constexpr (kReturnIdentity) {
      uint4* output_b128 = reinterpret_cast<uint4*>(output + global_offset);
      store_matrix_16x16_to_global<false>(c_frag[0], c_frag[1], c_frag[2], c_frag[3], output_b128,
                                          num_input_cols);
    }
  }

  if constexpr (kComputeTransposed) {
    if (kComputeIdentity) {
      matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[3]);
    } else {
      load_matrix_16x16_from_shared<true>(a_frag[0],
                                          a_frag[2],  // NOTE: intentional index swapping
                                          a_frag[1],  // NOTE: intentional index swapping
                                          a_frag[3], smem_b32, kHadamardDimension);
    }

    mma_m16_n16_k16_b16_b16_b16_noacc<kUpdateTransposeAmax>(
        a_frag[0],
        // 2,1 is used if we are using movmatrix instruction.
        // Thus loading the matrix in 2,1 order will just be normal.
        // This is to be compatible with the movmatrix instruction.
        a_frag[2],  // NOTE: intentional index swapping for transpose purpose.
        a_frag[1],  // NOTE: intentional index swapping for transpose purpose.
        a_frag[3], b_frag_t[0], b_frag_t[1], b_frag_t[2], b_frag_t[3], c_frag[0], c_frag[1],
        c_frag[2], c_frag[3], local_amax_t_reg);

    // Store the result to the shared memory in non-transposed order.
    if constexpr (kReturnTransposed) {
      uint4* output_t_b128 = reinterpret_cast<uint4*>(output_t + global_offset_t);
      store_matrix_16x16_to_global<!kOutputTrueTransposed>(
          c_frag[0], c_frag[1], c_frag[2], c_frag[3], output_t_b128,
          kOutputTrueTransposed ? num_input_rows : num_input_cols);
    }
  }

  if constexpr (kUpdateIdentityAmax) {
    unpack_max_of_packed_bf16(local_amax_reg, local_amax);
    local_amax = warp_reduce_max<kThreadsPerWarp>(local_amax);
    // broadcast the amax to all threads in a warp from the lane 0
    constexpr int lane_zero = 0;
    local_amax = __shfl_sync(0xFFFFFFFF, local_amax, lane_zero);
    // atomic CAS to output memory.
    if (tid % kThreadsPerWarp == 0) {
      atomicMaxFloat(amax, local_amax);
    }
  }
  if constexpr (kUpdateTransposeAmax) {
    unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
    local_amax_t = warp_reduce_max<kThreadsPerWarp>(local_amax_t);
    // broadcast the amax to all threads in a warp from the lane 0
    constexpr int lane_zero = 0;
    local_amax_t = __shfl_sync(0xFFFFFFFF, local_amax_t, lane_zero);
    // atomic CAS to output memory.
    if (tid % kThreadsPerWarp == 0) {
      atomicMaxFloat(amax_t, local_amax_t);
    }
  }
#else
  NVTE_DEVICE_ERROR("Kernel is only supported on SM 9.0+.");
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

#endif  // __HIP_PLATFORM_AMD__

#ifdef __HIP_PLATFORM_AMD__

// Tiling / layout constants
//
// A 16-point WHT operates on tiles of kHadamardDim (16) elements.
// Each tile is processed by kThreadsPerWHT (4) threads, each holding
// kElemsPerThread (4) values, so one wavefront of kWarpSize (64) lanes
// handles kRowsPerWarp (16) independent tiles (= rows) simultaneously.
// kWarpsPerBlock wavefronts are combined into a thread-block that covers
// kRowsPerBlock (64) consecutive rows.
static constexpr int kHadamardDim     = 16;   // WHT dimension (H16)
static constexpr int kWarpSize        = 64;   // Wavefront width
static constexpr int kThreadsPerWHT   = 4;    // threads per 16-pt WHT
static constexpr int kElemsPerThread  = 4;    // elements each thread owns
static constexpr int kRowsPerWarp     = kWarpSize / kThreadsPerWHT;   // 16
static constexpr int kWarpsPerBlock   = 4;
static constexpr int kRowsPerBlock    = kRowsPerWarp * kWarpsPerBlock; // 64
static constexpr int kThreadsPerBlock = kWarpSize   * kWarpsPerBlock;  // 256
static constexpr float kHadamardScale = 0.25f; // 1/sqrt(16)

// Reduce per-warp amax values in warp 0 and atomically update a global amax.
__device__ __forceinline__ void reduce_block_amax(
        const float* __restrict__ warp_amax, int lane_id,
        float* __restrict__ global_amax) {
    float val = (lane_id < kWarpsPerBlock) ? warp_amax[lane_id] : 0.f;
    for (int off = kWarpSize / 2; off >= 1; off >>= 1)
        val = fmaxf(val, __shfl_xor(val, off));
    if (lane_id == 0)
        atomicMaxFloat(global_amax, val);
}

// ds_swizzle: sub-wavefront exchange without LDS.
// Same instructions as cast_transpose_mxfp4_kernel_shuffled.cu.
__device__ __forceinline__ float ds_swizzle_xor1(float v) {
    float r;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
                 "s_waitcnt lgkmcnt(0)" : "=v"(r) : "v"(v));
    return r;
}

__device__ __forceinline__ float ds_swizzle_xor2(float v) {
    float r;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
                 "s_waitcnt lgkmcnt(0)" : "=v"(r) : "v"(v));
    return r;
}

// BF16 helpers
__device__ __forceinline__ float          to_f32 (__hip_bfloat16 v) { return static_cast<float>(v); }
__device__ __forceinline__ __hip_bfloat16 to_bf16(float v)          { return static_cast<__hip_bfloat16>(v); }

// Bit-cast __hip_bfloat16->uint16_t without address-of-temporary.
__device__ __forceinline__ uint16_t bf16_to_bits(__hip_bfloat16 v) {
    return __builtin_bit_cast(uint16_t, v);
}

// Unpack/pack 4 BF16 values as uint64_t (vectorised global load/store).
// Same trick as cast_transpose_mxfp4_kernel_shuffled.cu::bf16x4_to_float4.
__device__ __forceinline__ void unpack_bf16x4(uint64_t p,
                                              float& v0, float& v1, float& v2, float& v3) {
    v0 = __uint_as_float(((uint32_t)( p        & 0xFFFF)) << 16);
    v1 = __uint_as_float(((uint32_t)((p >> 16) & 0xFFFF)) << 16);
    v2 = __uint_as_float(((uint32_t)((p >> 32) & 0xFFFF)) << 16);
    v3 = __uint_as_float(((uint32_t)((p >> 48) & 0xFFFF)) << 16);
}

__device__ __forceinline__ uint64_t pack_bf16x4(float v0, float v1, float v2, float v3) {
    return  (uint64_t)bf16_to_bits(to_bf16(v0))
         | ((uint64_t)bf16_to_bits(to_bf16(v1)) << 16)
         | ((uint64_t)bf16_to_bits(to_bf16(v2)) << 32)
         | ((uint64_t)bf16_to_bits(to_bf16(v3)) << 48);
}

// -----------------------------------------------------------------------
// 16-point WHT via the Kronecker trick  (no shared memory)
// -----------------------------------------------------------------------
//
// 1. The vec operator
//    vec() flattens a matrix into a column vector by stacking its
//    columns one on top of the other:
//
//        X = |a  c|         vec(X) = |a|
//            |b  d|                  |b|
//                                    |c|
//                                    |d|
//
// 2. The "Kronecker trick" for 1D -> 2D
//    The fundamental identity that connects these concepts is:
//
//        vec(B . X . A^T) = (A (x) B) . vec(X)
//
//    For a 16-point Hadamard transform (H16 = H4 (x) H4),
//    set A = H4 and B = H4.  The formula becomes:
//
//        H16 . x = vec(H4 . X . H4^T)
//
// 3. Data layout  (column-major, one column per thread)
//    Reshape the 16-element 1D vector x into a 4x4 matrix X
//    by filling columns first:
//
//        X = | x0   x4   x8   x12 |    thread 0 holds col 0: v0..v3 = x0 ..x3
//            | x1   x5   x9   x13 |    thread 1 holds col 1: v0..v3 = x4 ..x7
//            | x2   x6   x10  x14 |    thread 2 holds col 2: v0..v3 = x8 ..x11
//            | x3   x7   x11  x15 |    thread 3 holds col 3: v0..v3 = x12..x15
//
// 4. Three-stage computation
//    Stage 1 (local H4)   : left-multiply   H4 . X    (within each thread)
//    Stage 2 (xor-1 swap) : \                         (across 4 threads)
//    Stage 3 (xor-2 swap) : / right-multiply . H4^T   together these two butterfly stages = H4^T
//
//    Result: vec(H4 . X . H4^T) = H16 . x
//
// 5. Randomised Hadamard Transform (RHT)
//    A diagonal sign matrix D (from sign_mask) is applied either
//    before the WHT (apply_pre=true, forward) or after (inverse).
//
// Adapted from cast_transpose_mxfp4_kernel_shuffled.cu::hadamard16_inplace,
// extended with NV random_sign_mask (uint16_t bitmask).
// thread_in_group [0,3]: drives ds_swizzle polarity (identical to MLPerf tid & 3).
// apply_pre=true -> D before WHT (forward); false -> D after WHT (inverse).
__device__ __forceinline__ void wht16(
        float& v0, float& v1, float& v2, float& v3,
        int thread_in_group, uint16_t sign_mask, bool apply_pre) {
    auto sgn = [&](int k) -> float {
        return ((sign_mask >> (thread_in_group * kElemsPerThread + k)) & 1u) ? -1.f : 1.f;
    };

    if (apply_pre) {
      v0*=sgn(0); v1*=sgn(1); v2*=sgn(2); v3*=sgn(3);
    }

    // Stage 1: local H4
    float a0=v0+v1, a1=v0-v1, a2=v2+v3, a3=v2-v3;
    v0=a0+a2; v2=a0-a2; v1=a1+a3; v3=a1-a3;

    // Stage 2: cross-thread XOR-1
    { float p0=ds_swizzle_xor1(v0), p1=ds_swizzle_xor1(v1),
            p2=ds_swizzle_xor1(v2), p3=ds_swizzle_xor1(v3);
      bool up=(thread_in_group&1);
      v0=up?(p0-v0):(p0+v0); v1=up?(p1-v1):(p1+v1);
      v2=up?(p2-v2):(p2+v2); v3=up?(p3-v3):(p3+v3); }

    // Stage 3: cross-thread XOR-2
    { float p0=ds_swizzle_xor2(v0), p1=ds_swizzle_xor2(v1),
            p2=ds_swizzle_xor2(v2), p3=ds_swizzle_xor2(v3);
      bool up=(thread_in_group>>1)&1;
      v0=up?(p0-v0):(p0+v0); v1=up?(p1-v1):(p1+v1);
      v2=up?(p2-v2):(p2+v2); v3=up?(p3-v3):(p3+v3); }

    v0*=kHadamardScale; v1*=kHadamardScale; v2*=kHadamardScale; v3*=kHadamardScale;

    if (!apply_pre) {
      v0*=sgn(0); v1*=sgn(1); v2*=sgn(2); v3*=sgn(3);
    }
}

// Grid:  blockIdx.x = col tile [0, row_length/16)
//        blockIdx.y = row batch [0, ceil(num_rows/64))
// Block: 256 threads = 4 wavefronts of 64 lanes.
//   lane/4 = row_in_warp (0..15), lane%4 = thread_in_grp (0..3)
template <bool kComputeIdentity, bool kComputeTransposed,
          bool kUpdateAmax,      bool kUpdateAmaxT>
__global__ __launch_bounds__(kThreadsPerBlock, 4)
void HadamardTransformKernel(
        const __hip_bfloat16* __restrict__ input,
        __hip_bfloat16*       __restrict__ output,
        __hip_bfloat16*       __restrict__ output_t,
        uint16_t random_sign_mask, uint16_t random_sign_mask_t,
        uint64_t num_rows, uint64_t row_length,
        float* __restrict__ amax, float* __restrict__ amax_t,
        bool inverse_hadamard) {
    const int tid           = threadIdx.x;
    const int warp_id       = tid / kWarpSize;
    const int lane_id       = tid % kWarpSize;
    const int row_in_warp   = lane_id / kThreadsPerWHT;
    const int thread_in_grp = lane_id % kThreadsPerWHT;
    const uint64_t col_tile_base = (uint64_t)blockIdx.x * kHadamardDim;
    const uint64_t row_batch     = (uint64_t)blockIdx.y * kRowsPerBlock;
    const uint64_t global_row    = row_batch + (uint64_t)warp_id*kRowsPerWarp + row_in_warp;
    const uint64_t col_base      = col_tile_base + (uint64_t)thread_in_grp * kElemsPerThread;

    const bool apply_pre = !inverse_hadamard;
    const bool in_bounds = (global_row < num_rows) && (col_base + kElemsPerThread - 1 < row_length);

    // Smem for transposed path: 64*(16+1) BF16; +1 avoids LDS bank conflict.
    __shared__ __hip_bfloat16 smem[kRowsPerBlock][kHadamardDim + 1];

    __shared__ float block_amax[kWarpsPerBlock];
    __shared__ float block_amax_t[kWarpsPerBlock];

    float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
    if (in_bounds) {
        unpack_bf16x4(*reinterpret_cast<const uint64_t*>(
            &input[global_row * row_length + col_base]), v0, v1, v2, v3);
    }

    // Identity path: WHT along row dimension
    if constexpr (kComputeIdentity || kUpdateAmax) {
        float r0=v0, r1=v1, r2=v2, r3=v3;
        float local_amax = 0.f;
        if (global_row < num_rows) {
            wht16(r0, r1, r2, r3, thread_in_grp, random_sign_mask, apply_pre);
            if constexpr (kUpdateAmax) {
                // Down-cast to BF16 and back so the amax matches the
                // stored/output precision (matches upstream NV behaviour).
                const float r0_bf16 = to_f32(to_bf16(r0));
                const float r1_bf16 = to_f32(to_bf16(r1));
                const float r2_bf16 = to_f32(to_bf16(r2));
                const float r3_bf16 = to_f32(to_bf16(r3));
                local_amax = fmaxf(fmaxf(fabsf(r0_bf16), fabsf(r1_bf16)),
                                   fmaxf(fabsf(r2_bf16), fabsf(r3_bf16)));
                for (int off=kWarpSize/2; off>=1; off>>=1)
                  local_amax=fmaxf(local_amax,__shfl_xor(local_amax,off));
            }
            if constexpr (kComputeIdentity)
                if (output && in_bounds)
                    *reinterpret_cast<uint64_t*>(&output[global_row*row_length+col_base]) =
                        pack_bf16x4(r0,r1,r2,r3);
        }
        if constexpr (kUpdateAmax) {
            if (lane_id == 0)
              block_amax[warp_id] = local_amax;
        }
    }

    // Transposed path: WHT along column dimension via smem transpose
    if constexpr (kComputeTransposed || kUpdateAmaxT) {
        const int local_row  = warp_id * kRowsPerWarp + row_in_warp;
        const int col_offset = thread_in_grp * kElemsPerThread;
        float local_amax = 0.f;
        smem[local_row][col_offset+0] = to_bf16(global_row < num_rows ? v0 : 0.f);
        smem[local_row][col_offset+1] = to_bf16(global_row < num_rows ? v1 : 0.f);
        smem[local_row][col_offset+2] = to_bf16(global_row < num_rows ? v2 : 0.f);
        smem[local_row][col_offset+3] = to_bf16(global_row < num_rows ? v3 : 0.f);
        __syncthreads();

        // Re-read: row_in_warp -> column index, thread_in_grp -> 4 rows
        const int t_col      = row_in_warp;
        const int smem_rbase = warp_id*kRowsPerWarp + thread_in_grp*kElemsPerThread;

        float c0=to_f32(smem[smem_rbase+0][t_col]), c1=to_f32(smem[smem_rbase+1][t_col]);
        float c2=to_f32(smem[smem_rbase+2][t_col]), c3=to_f32(smem[smem_rbase+3][t_col]);

        wht16(c0, c1, c2, c3, thread_in_grp, random_sign_mask_t, apply_pre);

        if constexpr (kUpdateAmaxT) {
            // Down-cast to BF16 and back so the amax matches the
            // stored/output precision (matches upstream NV behaviour).
            const float c0_bf16 = to_f32(to_bf16(c0));
            const float c1_bf16 = to_f32(to_bf16(c1));
            const float c2_bf16 = to_f32(to_bf16(c2));
            const float c3_bf16 = to_f32(to_bf16(c3));
            local_amax = fmaxf(fmaxf(fabsf(c0_bf16), fabsf(c1_bf16)),
                               fmaxf(fabsf(c2_bf16), fabsf(c3_bf16)));

            for (int off=kWarpSize/2; off>=1; off>>=1)
              local_amax=fmaxf(local_amax,__shfl_xor(local_amax,off));
        }

        if constexpr (kComputeTransposed) {
            if (output_t) {
                const uint64_t global_col   = col_tile_base + t_col;
                const uint64_t out_row_base = row_batch + (uint64_t)warp_id*kRowsPerWarp
                                              + (uint64_t)thread_in_grp*kElemsPerThread;
                if (global_col < row_length && out_row_base+kElemsPerThread-1 < num_rows)
                    *reinterpret_cast<uint64_t*>(
                        &output_t[global_col*num_rows+out_row_base]) =
                            pack_bf16x4(c0,c1,c2,c3);
            }
        }

        if constexpr (kUpdateAmaxT) {
            if (lane_id == 0)
              block_amax_t[warp_id] = local_amax;
        }
    }

    if constexpr (kUpdateAmax || kUpdateAmaxT) {
        __syncthreads();

        if (warp_id == 0) {
            if constexpr (kUpdateAmax)
                reduce_block_amax(block_amax, lane_id, amax);
            if constexpr (kUpdateAmaxT)
                reduce_block_amax(block_amax_t, lane_id, amax_t);
        }
    }
}

// Pre-RHT amax: max|input| before any transform.
__global__ void PreRhtAmaxKernel(const __hip_bfloat16* __restrict__ input,
                                 float* __restrict__ amax_out, uint64_t num_elems) {
    __shared__ float block_amax[kWarpsPerBlock];
    float local_amax = 0.f;
    for (uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
         i < num_elems; i += (uint64_t)gridDim.x*blockDim.x)
      local_amax = fmaxf(local_amax, fabsf(to_f32(input[i])));

    for (int off=kWarpSize/2; off>=1; off>>=1)
      local_amax=fmaxf(local_amax,__shfl_xor(local_amax,off));

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (lane_id == 0)
      block_amax[warp_id] = local_amax;

    __syncthreads();

    if (warp_id == 0)
      reduce_block_amax(block_amax, lane_id, amax_out);
}

static inline dim3 transform_grid(uint64_t num_rows, uint64_t row_length) {
    return dim3((uint32_t)(row_length / kHadamardDim),
                (uint32_t)DIVUP(num_rows, (uint64_t)kRowsPerBlock));
}

#endif  // __HIP_PLATFORM_AMD__

}  // namespace

void hadamard_transform(const Tensor& input_, Tensor& output_, uint16_t random_sign_mask,
                        uint16_t random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform);

  // Check tensors
  // NOTE (frsun): This is non-intuitive, we are writing the result of
  // transposed RHT to the output of rowwise.
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  NVTE_CHECK(output_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor must be simple tensor, but scaling mode is ",
             to_string(output_.scaling_mode), ".");
  const SimpleTensor& input = input_.data;
  SimpleTensor output;
  SimpleTensor& output_t = output_.data;

  // Check requested outputs
  const bool return_identity = output.dptr != nullptr;
  const bool return_transposed = output_t.dptr != nullptr;
  if (!return_identity && !return_transposed) {  // Nothing to do/ill-defined behavior.
    return;
  }

#ifndef __HIP_PLATFORM_AMD__
  checkCuDriverContext(stream);
#endif

  const size_t ndim = input.shape.size();
  const size_t row_length = input.shape[ndim - 1];
  size_t num_rows = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    num_rows *= input.shape[i];
  }

#ifdef __HIP_PLATFORM_AMD__
  using IType = __hip_bfloat16;
  constexpr int kHadamardDimension = kHadamardDim;
#else
  using IType = bf16;

  constexpr int kHadamardDimension = 16;
#endif
  NVTE_CHECK(row_length % kHadamardDimension == 0,
             "row_length must be divisible by hadamard_dimension.");
  NVTE_CHECK(num_rows % kHadamardDimension == 0,
             "num_rows must be divisible by hadamard_dimension");

#ifdef __HIP_PLATFORM_AMD__
  dim3 block(kThreadsPerBlock);
  dim3 grid = transform_grid(num_rows, row_length);
#else
  constexpr uint64_t kThreadBlockX = 4;
  // Configure 4 is used for Hopper, 8 is used for Blackwell for extra memory bandwidth.
  constexpr uint64_t kThreadBlockY = 4;

  uint64_t kNumWarpsPerSM = kThreadBlockX * kThreadBlockY;

  // The shared memory number of bytes required for **the whole threadblock**.
  size_t shmem_bytes = kHadamardDimension * kHadamardDimension * sizeof(IType) * kNumWarpsPerSM;

  dim3 block(kThreadsPerWarp, kThreadBlockX, kThreadBlockY);

  dim3 grid(DIVUP(row_length / kHadamardDimension, kThreadBlockX),
            DIVUP(num_rows / kHadamardDimension, kThreadBlockY));
#endif

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      return_transposed, kReturnTransposed,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          return_identity, kReturnIdentity,

#ifdef __HIP_PLATFORM_AMD__
          HadamardTransformKernel<kReturnIdentity, kReturnTransposed, false, false>
              <<<grid, block, 0, stream>>>(
                  reinterpret_cast<const IType*>(input.dptr),
                  reinterpret_cast<IType*>(output.dptr),
                  reinterpret_cast<IType*>(output_t.dptr), random_sign_mask,
                  random_sign_mask_t, static_cast<uint64_t>(num_rows),
                  static_cast<uint64_t>(row_length), nullptr, nullptr, false);););
#else
          auto kernel =
              HadamardTransformKernel<IType, kHadamardDimension, kReturnIdentity, kReturnTransposed,
                                      kReturnIdentity, kReturnTransposed, false, false, true>;

          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

          kernel<<<grid, block, shmem_bytes, stream>>>(
              reinterpret_cast<const IType*>(input.dptr), reinterpret_cast<IType*>(output.dptr),
              reinterpret_cast<IType*>(output_t.dptr), random_sign_mask, random_sign_mask_t,
              num_rows, row_length, nullptr, nullptr, false);););
#endif

  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Kernel that will apply the 16x16 hadamard transform the input and input.T, and then
// get the absolute max value of the result.
void hadamard_transform_amax(const Tensor& input_, Tensor& output_, uint16_t random_sign_mask,
                             uint16_t random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform_amax);
#if CUDA_VERSION >= 12080 || defined(__HIP_PLATFORM_AMD__)

  // Check input tensor
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor& input = input_.data;

  // Check amax tensors
  SimpleTensor& output_pre_rht_amax = output_.amax;
  SimpleTensor output_identity_amax;
  SimpleTensor& output_transpose_amax = output_.columnwise_amax;

  // Check requested outputs
  const bool return_pre_rht_amax = output_pre_rht_amax.dptr != nullptr;
  const bool return_identity_amax = output_identity_amax.dptr != nullptr;
  const bool return_transposed_amax = output_transpose_amax.dptr != nullptr;
  if (!return_identity_amax && !return_transposed_amax &&
      !return_pre_rht_amax) {  // Nothing to do/ill-defined behavior.
    return;
  }

#ifndef __HIP_PLATFORM_AMD__
  // Zero out amaxes if needed
  ZeroAmaxKernel<<<1, 1, 0, stream>>>(reinterpret_cast<float*>(output_pre_rht_amax.dptr),
                                      reinterpret_cast<float*>(output_identity_amax.dptr),
                                      reinterpret_cast<float*>(output_transpose_amax.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());

  checkCuDriverContext(stream);

  using IType = bf16;
#endif

  const size_t ndim = input.shape.size();
  const size_t row_length = input.shape[ndim - 1];
  size_t num_rows = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    num_rows *= input.shape[i];
  }

#ifdef __HIP_PLATFORM_AMD__
  auto* pre_amax_ptr = reinterpret_cast<float*>(output_pre_rht_amax.dptr);
  auto* id_amax_ptr = reinterpret_cast<float*>(output_identity_amax.dptr);
  auto* tr_amax_ptr = reinterpret_cast<float*>(output_transpose_amax.dptr);

  NVTE_CHECK(row_length % kHadamardDim == 0, "row_length must be divisible by 16.");
  NVTE_CHECK(num_rows % kHadamardDim == 0, "num_rows must be divisible by 16.");

  auto* in_ptr = reinterpret_cast<const __hip_bfloat16*>(input.dptr);

  if (pre_amax_ptr) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(pre_amax_ptr, 0, sizeof(float), stream));
  }
  if (id_amax_ptr) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(id_amax_ptr, 0, sizeof(float), stream));
  }
  if (tr_amax_ptr) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(tr_amax_ptr, 0, sizeof(float), stream));
  }

  if (return_pre_rht_amax) {
    const uint64_t num_elems = static_cast<uint64_t>(num_rows) * row_length;
    dim3 grid(DIVUP(num_elems, static_cast<uint64_t>(kThreadsPerBlock)));
    PreRhtAmaxKernel<<<grid, kThreadsPerBlock, 0, stream>>>(in_ptr, pre_amax_ptr, num_elems);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
#else
  constexpr int kHadamardDimension = 16;
  NVTE_CHECK(row_length % kHadamardDimension == 0,
             "row_length must be divisible by hadamard_dimension.");
  NVTE_CHECK(num_rows % kHadamardDimension == 0,
             "num_rows must be divisible by hadamard_dimension");

  constexpr uint64_t kChunkBlockXSmall = 128;
  constexpr uint64_t kChunkBlockYSmall = 128;
  constexpr uint64_t kBuffDimX = 64;
  constexpr uint64_t kBuffDimY = 64;

  alignas(64) CUtensorMap tensor_map_input{};

  create_2D_tensor_map(
      /*tensorMap=*/tensor_map_input,
      /*tensor=*/input,
      /*globalY=*/num_rows,
      /*globalX=*/row_length,
      /*shmemY=*/kBuffDimY,
      /*shmemX=*/kBuffDimX,
      /*stride_elems=*/row_length,
      /*offset_elems=*/0,
      /*type_num_bits=*/sizeof(IType) * 8,
      /*swizzle=*/CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B);

  constexpr uint64_t kThreadBlockX = 4;
  constexpr uint64_t kThreadBlockY = 1;
  constexpr uint64_t kNumWarps = kThreadBlockX * kThreadBlockY;

  dim3 block(kThreadBlockX * kThreadsPerWarp, kThreadBlockY);

  dim3 grid(DIVUP(row_length, kChunkBlockXSmall), DIVUP(num_rows, kChunkBlockYSmall));
#endif

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      return_transposed_amax, kReturnTransposedAmax,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          return_identity_amax, kReturnIdentityAmax,

#ifdef __HIP_PLATFORM_AMD__
          if (kReturnIdentityAmax || kReturnTransposedAmax) {
            dim3 grid = transform_grid(num_rows, row_length), block(kThreadsPerBlock);
            HadamardTransformKernel<kReturnIdentityAmax, kReturnTransposedAmax,
                                    kReturnIdentityAmax, kReturnTransposedAmax>
                <<<grid, block, 0, stream>>>(in_ptr, nullptr, nullptr, random_sign_mask,
                                             random_sign_mask_t, static_cast<uint64_t>(num_rows),
                                             static_cast<uint64_t>(row_length), id_amax_ptr,
                                             tr_amax_ptr, false);
          }
        ));
#else
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_pre_rht_amax, kReturnPreRhtAmax,

              // *2 for ping-pong
              size_t in_sh_size = kBuffDimX * kBuffDimY * 2 * sizeof(IType);
              size_t mbar_size = sizeof(uint64_t) * (kChunkBlockXSmall / kBuffDimX) *
                                 (kChunkBlockYSmall / kBuffDimY);
              size_t shmem_bytes = in_sh_size + mbar_size + kNumWarps * sizeof(float) * 3;
              // Add padding in case shmem ptr is not aligned to 128 bytes.
              shmem_bytes = (shmem_bytes + 128);

              auto kernel = HadamardAmaxTmaKernel<
                  IType, kHadamardDimension, kChunkBlockYSmall, kChunkBlockXSmall, kBuffDimY,
                  kBuffDimX, kThreadBlockX * kThreadsPerWarp, kThreadBlockY, kReturnPreRhtAmax,
                  kReturnIdentityAmax, kReturnTransposedAmax>;
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_bytes);

              kernel<<<grid, block, shmem_bytes, stream>>>(
                  tensor_map_input, reinterpret_cast<float*>(output_pre_rht_amax.dptr),
                  reinterpret_cast<float*>(output_identity_amax.dptr),
                  reinterpret_cast<float*>(output_transpose_amax.dptr), random_sign_mask,
                  random_sign_mask_t, num_rows, row_length);)));
#endif

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("Hadamard transform requires CUDA 12.8+, but compile-time CUDA version is ",
             CUDA_VERSION);
#endif  // CUDA_VERSION >= 12080 || __HIP_PLATFORM_AMD__
}

}  // namespace transformer_engine

void nvte_hadamard_transform(const NVTETensor input, NVTETensor output, int random_sign_mask,
                             int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform);
  using namespace transformer_engine;
  hadamard_transform(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                     static_cast<uint16_t>(random_sign_mask),
                     static_cast<uint16_t>(random_sign_mask_t), stream);
}

void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output, int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform_amax);
  using namespace transformer_engine;
  hadamard_transform_amax(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                          static_cast<uint16_t>(random_sign_mask),
                          static_cast<uint16_t>(random_sign_mask_t), stream);
}
