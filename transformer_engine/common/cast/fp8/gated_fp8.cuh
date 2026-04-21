/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated_fp8.cuh
 *  \brief CUDA kernels to cast to FP8 with gated activations.
 */

#ifndef TRANSFORMER_ENGINE_GATED_FP8_CUH_
#define TRANSFORMER_ENGINE_GATED_FP8_CUH_

#include <cuda.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cudaTypedefs.h>
#endif //#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../util/vectorized_pointwise.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {

// ---------------------------------------------------------------------------
// TDM (Tensor Data Mover) gated FP8 kernel for gfx1250
// ---------------------------------------------------------------------------
#if defined(__gfx1250__)
}  // temporarily close namespace fp8
}  // temporarily close namespace dispatch
#include "../../util/tdm.cuh"
namespace dispatch {
namespace fp8 {

namespace cast_gated_tdm_kernel {

constexpr size_t TDM_GATED_CHUNK_DIM_Y   = 128;
constexpr size_t TDM_GATED_CHUNK_DIM_X   = 128;
constexpr size_t TDM_GATED_THREADS       = 128;
constexpr size_t TDM_GATED_BUFFERS_NUM   = 2;
constexpr size_t TDM_GATED_BUFFER_DIM_Y  = 16;
constexpr size_t TDM_GATED_BUFFER_DIM_X  = TDM_GATED_CHUNK_DIM_X;   // 128
constexpr size_t TDM_GATED_SHMEM_DIM_Y   = TDM_GATED_BUFFER_DIM_Y;  // 16
constexpr size_t TDM_GATED_SHMEM_DIM_X   = TDM_GATED_BUFFER_DIM_X;  // 128
constexpr size_t TDM_GATED_BUFF_STAGES   = TDM_GATED_BUFFER_DIM_Y;  // 16
constexpr size_t TDM_GATED_ITERATIONS    = TDM_GATED_CHUNK_DIM_Y / TDM_GATED_BUFFER_DIM_Y;  // 8

// IS_BWD=false: loads act + gate (2 loads/iter)      → wait(2) while prefetch live, wait(0) last
// IS_BWD=true : loads grad + act + gate (3 loads/iter) → wait(3) while prefetch live, wait(0) last
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(TDM_GATED_THREADS)
    cast_fp8_gated_tdm_kernel(
        // Backward only: pointer to gradient tensor (rows × cols, stride = cols)
        const IType *__restrict__ grad_ptr,
        // Forward activation half: interleaved in gated_input at column offset 0
        //   gated_input layout: [rows][cols*2], act at col 0, gate at col cols
        const IType *__restrict__ act_ptr,
        // Gate half: same base pointer as act, column offset = cols
        const IType *__restrict__ gate_ptr,
        // Output act:  [rows][output_cols], act  at col 0
        OType *__restrict__ out_act_ptr,
        // Output gate: [rows][output_cols], gate at col cols (BWD only; ignored for FWD)
        OType *__restrict__ out_gate_ptr,
        float *const amax_ptr,
        float *const scale_inv_ptr,
        const float *const scale_ptr,
        const size_t rows,
        const size_t cols,
        // Row stride of the gated_input tensor (= cols*2 for interleaved act/gate)
        const size_t input_stride,
        // Row stride of the grad tensor (= cols, BWD only)
        const size_t grad_stride,
        // Row stride of the output tensor (= output_cols = IS_BWD ? 2*cols : cols)
        const size_t output_stride,
        const ParamOP p) {
  using namespace transformer_engine::tdm;

  const size_t chunk_offset_Y = blockIdx.y * TDM_GATED_CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * TDM_GATED_CHUNK_DIM_X;

  // Within-block thread coordinates
  // All threads cover one column each (tid_X) and one row within the shmem tile (tid_Y=0 for all).
  // Since TDM_GATED_THREADS=128 = TDM_GATED_SHMEM_DIM_X, each thread owns one column.
  const size_t tid_X = threadIdx.x % TDM_GATED_SHMEM_DIM_X;
  const size_t tid_Y = threadIdx.x / TDM_GATED_SHMEM_DIM_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // Shared memory buffers: double-buffered for loads, single for output
  __shared__ alignas(128) IType in_act_sh [TDM_GATED_BUFFERS_NUM][TDM_GATED_SHMEM_DIM_Y][TDM_GATED_SHMEM_DIM_X];
  __shared__ alignas(128) IType in_gate_sh[TDM_GATED_BUFFERS_NUM][TDM_GATED_SHMEM_DIM_Y][TDM_GATED_SHMEM_DIM_X];
  __shared__ alignas(128) IType in_grad_sh[IS_BWD ? TDM_GATED_BUFFERS_NUM : 1][TDM_GATED_SHMEM_DIM_Y][TDM_GATED_SHMEM_DIM_X];
  __shared__ alignas(128) OType out_act_sh [TDM_GATED_BUFFERS_NUM][TDM_GATED_SHMEM_DIM_Y][TDM_GATED_SHMEM_DIM_X];
  __shared__ alignas(128) OType out_gate_sh[IS_BWD ? TDM_GATED_BUFFERS_NUM : 1][TDM_GATED_SHMEM_DIM_Y][TDM_GATED_SHMEM_DIM_X];

  constexpr uint32_t input_data_size  = get_data_size_from_bits(sizeof(IType) * 8);
  constexpr uint32_t output_data_size = get_data_size_from_bits(sizeof(OType) * 8);

  const uint32_t tensor_w_in  = static_cast<uint32_t>(cols);
  const uint32_t tensor_h     = static_cast<uint32_t>(rows);
  const uint32_t stride_in    = static_cast<uint32_t>(input_stride);
  const uint32_t stride_grad  = static_cast<uint32_t>(grad_stride);
  const uint32_t stride_out   = static_cast<uint32_t>(output_stride);
  // Each output sub-tensor (act or gate) is cols elements wide; stride_out is the full row stride.
  const uint32_t tensor_w_out = static_cast<uint32_t>(cols);

  // --- Prologue: prefetch iteration 0 into buffer 0 ---
  // NOTE: grad has row stride = cols, while act/gate have row stride = cols*2 (interleaved layout).
  // copy_2d_to_shared_x3 uses a single stride for all three tensors, so we cannot mix strides.
  // Instead, for BWD we issue: copy_2d_to_shared (grad, stride_grad) +
  //                            copy_2d_to_shared_x2 (act+gate, stride_in).
  // Total TDM ops per prefetch: BWD=3, FWD=2. Wait counts below match accordingly.
  {
    const uint32_t cy = static_cast<uint32_t>(chunk_offset_Y);
    const uint32_t cx = static_cast<uint32_t>(chunk_offset_X);
    if constexpr (IS_BWD) {
      copy_2d_to_shared(
          &in_grad_sh[0][0][0], grad_ptr, cx, cy,
          TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_grad, input_data_size);
      copy_2d_to_shared_x2(
          &in_act_sh [0][0][0], act_ptr,  cx, cy,
          &in_gate_sh[0][0][0], gate_ptr, cx, cy,
          TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_in, input_data_size);
    } else {
      copy_2d_to_shared_x2(
          &in_act_sh [0][0][0], act_ptr,  cx, cy,
          &in_gate_sh[0][0][0], gate_ptr, cx, cy,
          TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
          tensor_w_in, tensor_h, stride_in, input_data_size);
    }
  }

  // --- Main loop ---
#pragma unroll
  for (int iter = 0; iter < (int)TDM_GATED_ITERATIONS; ++iter) {
    const int buff      = iter % TDM_GATED_BUFFERS_NUM;
    const int next_iter = iter + 1;
    const int next_buff = next_iter % TDM_GATED_BUFFERS_NUM;

    // Prefetch next iteration if available
    if (next_iter < (int)TDM_GATED_ITERATIONS) {
      const uint32_t ncy = static_cast<uint32_t>(chunk_offset_Y + next_iter * TDM_GATED_BUFFER_DIM_Y);
      const uint32_t ncx = static_cast<uint32_t>(chunk_offset_X);
      if constexpr (IS_BWD) {
        copy_2d_to_shared(
            &in_grad_sh[next_buff][0][0], grad_ptr, ncx, ncy,
            TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_grad, input_data_size);
        copy_2d_to_shared_x2(
            &in_act_sh [next_buff][0][0], act_ptr,  ncx, ncy,
            &in_gate_sh[next_buff][0][0], gate_ptr, ncx, ncy,
            TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_in, input_data_size);
      } else {
        copy_2d_to_shared_x2(
            &in_act_sh [next_buff][0][0], act_ptr,  ncx, ncy,
            &in_gate_sh[next_buff][0][0], gate_ptr, ncx, ncy,
            TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
            tensor_w_in, tensor_h, stride_in, input_data_size);
      }
    }

    // Wait for current buffer's loads.
    // TENSORcnt counts outstanding ops oldest-first. After issuing the next
    // prefetch, we wait for ops older than the prefetch to finish.
    //   IS_BWD: 3 loads/iter issued before this iter's compute.
    //   !IS_BWD: 2 loads/iter.
    // When next_iter < ITERATIONS we keep prefetch ops in-flight (wait N),
    // otherwise drain everything (wait 0).
    if (is_tdm_wave()) {
      if (next_iter < (int)TDM_GATED_ITERATIONS) {
        if constexpr (IS_BWD) {
          wait_tensorcnt_3();
        } else {
          wait_tensorcnt_2();
        }
      } else {
        wait_tensorcnt_0();
      }
    }
    __syncthreads();

    // --- Compute ---
#pragma unroll
    for (int stage = 0; stage < (int)TDM_GATED_BUFF_STAGES; ++stage) {
      const size_t shmem_y = tid_Y + stage;
      const size_t shmem_x = tid_X;

      float act_elt  = static_cast<float>(in_act_sh [buff][shmem_y][shmem_x]);
      float gate_elt = static_cast<float>(in_gate_sh[buff][shmem_y][shmem_x]);

      bool dgate_elt = true;
      if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
        dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;
        gate_elt  = min(max(-p.limit, gate_elt), p.limit) + 1;
      }

      if constexpr (IS_BWD) {
        float grad_elt = static_cast<float>(in_grad_sh[buff][shmem_y][shmem_x]);

        const float x = act_elt;
        float act_x;
        float dact_x;
        if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
          const float xc = min(act_elt, p.limit);
          const float s  = sigmoidf(p.alpha * xc);
          act_x = xc * s;
          if (act_elt <= p.limit) {
            dact_x = s + s * (1 - s) * p.alpha * xc;
          } else {
            dact_x = 0.0f;
          }
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
        float after_dact  = dact_x * grad_elt * gate_elt;
        float after_dgate = dgate_elt ? act_x * grad_elt : 0.0f;

        out_act_sh [buff][shmem_y][shmem_x] = static_cast<OType>(scale * after_dact);
        out_gate_sh[buff][shmem_y][shmem_x] = static_cast<OType>(scale * after_dgate);

        amax = fmaxf(amax, fabsf(after_dact));
        amax = fmaxf(amax, fabsf(after_dgate));
      } else {
        const float after_act = ActOP(act_elt, p) * gate_elt;
        out_act_sh[buff][shmem_y][shmem_x] = static_cast<OType>(scale * after_act);
        amax = fmaxf(amax, fabsf(after_act));
      }
    }

    // Ensure all threads finished writing shmem before TDM reads it
    __syncthreads();

    // TDM store: output shmem → global
    {
      const uint32_t scy = static_cast<uint32_t>(chunk_offset_Y + iter * TDM_GATED_BUFFER_DIM_Y);
      const uint32_t scx = static_cast<uint32_t>(chunk_offset_X);

      store_2d_to_global(
          &out_act_sh[buff][0][0], out_act_ptr,
          scx, scy,
          TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
          tensor_w_out, tensor_h, stride_out, output_data_size);

      if constexpr (IS_BWD) {
        store_2d_to_global(
            &out_gate_sh[buff][0][0], out_gate_ptr,
            scx, scy,
            TDM_GATED_SHMEM_DIM_X, TDM_GATED_SHMEM_DIM_Y,
            tensor_w_out, tensor_h, stride_out, output_data_size);
      }
    }
  }

  // Drain all remaining TDM ops (stores)
  if (is_tdm_wave()) {
    wait_tensorcnt_0();
  }
  __syncthreads();

  // --- Reduce amax ---
  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    amax = reduce_max<TDM_GATED_THREADS / THREADS_PER_WARP>(amax, warp_id);
    if (threadIdx.x == 0) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse (only one block writes it)
  if (threadIdx.x == 0 && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }
}

}  // namespace cast_gated_tdm_kernel

// ---------------------------------------------------------------------------
// TDM launcher for gated FP8 cast
// ---------------------------------------------------------------------------

// Runtime check for gfx1250 TDM support.
// Guard against re-definition when quantize_fp8.cuh is included in the same TU.
#ifndef TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_
#define TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_
inline bool is_gfx1250() {
  static int result = -1;
  if (result < 0) {
    int device;
    (void)hipGetDevice(&device);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, device);
    result = (strncmp(prop.gcnArchName, "gfx1250", 7) == 0) ? 1 : 0;
  }
  return result == 1;
}
#endif  // TRANSFORMER_ENGINE_IS_GFX1250_DEFINED_

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_gated_tma_tdm(const Tensor &gated_input, const Tensor &grad, Tensor *output, ParamOP &p,
                        cudaStream_t stream) {
  using namespace cast_gated_tdm_kernel;

  NVTE_CHECK(!output->has_columnwise_data(), "Only rowwise cast supported in this function.");

  const size_t rows        = gated_input.flat_first_dim();
  const size_t cols        = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, TDM_GATED_CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, TDM_GATED_CHUNK_DIM_X);

  float *const amax_ptr     = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr    = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(TDM_GATED_THREADS);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          const IType *act_ptr  = reinterpret_cast<const IType *>(gated_input.data.dptr);
          const IType *gate_ptr = act_ptr + cols;  // gate follows act in interleaved layout
          const IType *grad_ptr = IS_BWD ? reinterpret_cast<const IType *>(grad.data.dptr)
                                         : nullptr;

          OType *out_act_ptr  = reinterpret_cast<OType *>(output->data.dptr);
          OType *out_gate_ptr = IS_BWD ? out_act_ptr + cols : nullptr;

          const size_t input_stride  = cols * 2;   // gated_input: [rows][cols*2]
          const size_t grad_stride   = cols;        // grad:        [rows][cols]
          const size_t output_stride = output_cols; // output:      [rows][output_cols]

          cast_fp8_gated_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType>
              <<<grid_dim, block_dim, 0, stream>>>(
                  grad_ptr, act_ptr, gate_ptr,
                  out_act_ptr, out_gate_ptr,
                  amax_ptr, scale_inv_ptr, scale_ptr,
                  rows, cols,
                  input_stride, grad_stride, output_stride,
                  p);
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                              // NOLINT(*)
}

#endif  // defined(__gfx1250__)

#ifndef __HIP_PLATFORM_AMD__
namespace kernel {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 512;
constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 512 / 128
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //  4 = 128 / 32
static_assert(ITERATIONS >= 1);

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                          const __grid_constant__ CUtensorMap tensor_map_input_act,
                          const __grid_constant__ CUtensorMap tensor_map_input_gate,
                          const __grid_constant__ CUtensorMap tensor_map_output_act,
                          const __grid_constant__ CUtensorMap tensor_map_output_gate,
                          float *const amax_ptr, float *const scale_inv_ptr,
                          const float *const scale_ptr, const size_t rows, const size_t cols,
                          const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const size_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const size_t tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const size_t thread_offset_Y = tid_Y;
  const size_t thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  constexpr size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  constexpr size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t grad_mem = IS_BWD ? buff_size_aligned_in : 0;

  constexpr size_t in_act_mem = buff_size_aligned_in;
  constexpr size_t in_gate_mem = buff_size_aligned_in;
  constexpr size_t in_mem = in_act_mem + in_gate_mem;

  constexpr size_t out_act_mem = buff_size_aligned_out;
  constexpr size_t in_transaction_size = buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);
  OType *out_act_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_in_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
  const uint64_t *TMAP_in_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);
  const uint64_t *TMAP_output_act = reinterpret_cast<const uint64_t *>(&tensor_map_output_act);
  const uint64_t *TMAP_output_gate = reinterpret_cast<const uint64_t *>(&tensor_map_output_gate);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  // Prefetch data of the first stage

  if constexpr (IS_BWD) {
    copy_2d_to_sharedx3(in_grad_sh, TMAP_grad_in, chunk_offset_X, chunk_offset_Y, in_act_sh,
                        TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh, TMAP_in_gate,
                        chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  } else {
    copy_2d_to_sharedx2(in_act_sh, TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh,
                        TMAP_in_gate, chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const size_t buff = it % BUFFERS_NUM;
    const size_t next_it = it + 1;
    if (next_it < ITERATIONS) {
      const size_t next_buff = next_it % BUFFERS_NUM;
      const size_t chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_BWD) {
        copy_2d_to_sharedx3(
            &in_grad_sh[next_buff * buff_elems], TMAP_grad_in, chunk_it_offset_x, chunk_it_offset_y,
            &in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x, chunk_it_offset_y,
            &in_gate_sh[next_buff * buff_elems], TMAP_in_gate, chunk_it_offset_x, chunk_it_offset_y,
            in_transaction_size, &mbar[next_it], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x,
                            chunk_it_offset_y, &in_gate_sh[next_buff * buff_elems], TMAP_in_gate,
                            chunk_it_offset_x, chunk_it_offset_y, in_transaction_size,
                            &mbar[next_it], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_sh_curr = out_act_sh + buff * buff_elems;
    OType *out_gate_sh_curr = out_gate_sh + buff * buff_elems;
#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const size_t stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const size_t shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const size_t shmem_offset_x = thread_offset_X;
      const size_t shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);
      bool dgate_elt = true;  // gating is ideally an identity function
      if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
        // In case of GPT OSS, clamp the activation and gate values
        dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;  // Derivative of clamp
        gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1;
      }

      if constexpr (IS_BWD) {
        float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);

        const float x = act_elt;
        float act_x;
        float dact_x;
        if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
          const float x = min(act_elt, p.limit);
          const float s = sigmoidf(p.alpha * x);
          act_x = x * s;
          if (act_elt <= p.limit) {
            dact_x = s + s * (1 - s) * p.alpha * x;
          } else {
            dact_x = 0.0f;
          }
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
        float after_dact = dact_x * grad_elt * gate_elt;
        float after_dgate = dgate_elt ? act_x * grad_elt : 0.0f;

        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dact);
        out_gate_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dgate);

        amax = fmaxf(amax, fabsf(after_dact));
        amax = fmaxf(amax, fabsf(after_dgate));
      } else {
        const float after_act = ActOP(act_elt, p) * gate_elt;
        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_act);
        amax = fmaxf(amax, fabsf(after_act));
      }
    }

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output_act, chunk_it_offset_x,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_act_sh_curr));

      if constexpr (IS_BWD) {
        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_gate, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_sh_curr));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace kernel

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_gated_tma(const Tensor &gated_input, const Tensor &grad, Tensor *output, ParamOP &p,
                    cudaStream_t stream) {
  using namespace kernel;
  checkCuDriverContext(stream);

  NVTE_CHECK(!output->has_columnwise_data(), "Only rowwise cast supported in this function.");
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act{};
          alignas(64) CUtensorMap tensor_map_output_gate{};

          if constexpr (IS_BWD) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X,
                                 cols, 0, typeToNumBits(gated_input.dtype()));
          }

          const uint32_t tensor_stride_elems = output_cols;

          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, 0, typeToNumBits(gated_input.dtype()));
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, cols, typeToNumBits(gated_input.dtype()));
          create_2D_tensor_map(tensor_map_output_act, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, 0, typeToNumBits(output->dtype()));
          create_2D_tensor_map(tensor_map_output_gate, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, cols,
                               typeToNumBits(output->dtype()));

          const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
          const size_t buff_size_aligned_in =
              DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
          const size_t buff_size_aligned_out =
              DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);
          const size_t grad_mem = (IS_BWD ? buff_size_aligned_in : 0);
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;

          const size_t shmem_size = grad_mem + (in_act_mem + in_gate_mem) +
                                    (out_act_mem + out_gate_mem) + TMA_SHMEM_ALIGNMENT;

          auto kernel = cast_fp8_gated_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType>;
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               shmem_size));

          kernel<<<grid_dim, block_dim, shmem_size, stream>>>(
              tensor_map_grad, tensor_map_input_act, tensor_map_input_gate, tensor_map_output_act,
              tensor_map_output_gate, amax_ptr, scale_inv_ptr, scale_ptr, rows, cols, p);
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                              // NOLINT(*)
}
#endif //#ifndef __HIP_PLATFORM_AMD__

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void cast_gated_fwd(const Tensor &input, Tensor *output, ParamOP &p, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          constexpr int nvec = 32 / sizeof(IType);
          GatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), input.flat_first_dim(),
              output->flat_last_dim(), p, stream););  // NOLINT(*)
  );                                                  // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_gated_bwd(const Tensor &input, const Tensor &grad, Tensor *output, ParamOP &p,
                    cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          constexpr int nvec = 32 / sizeof(IType);
          DGatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, DActOP>(
              reinterpret_cast<const IType *>(grad.data.dptr),
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), grad.flat_first_dim(),
              grad.flat_last_dim(), p, stream););  // NOLINT(*)
  );                                               // NOLINT(*)
}
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GATED_FP8_CUH_
