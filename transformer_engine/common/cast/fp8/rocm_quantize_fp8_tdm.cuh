/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 FP8 tensor-scaling quantize with the Tensor Data Mover: each wave moves its own
// BAND_Y-row band of a 128x128 chunk global->LDS->global, and for the transposed output the waves
// assemble the chunk's transpose in LDS. The launchers return false where TDM does not apply,
// leaving the work to the generic kernels. Like the other rocm_ headers, this expects its includer
// to have included the common headers, util/system.h and util/cuda_runtime.h.

#include <vector>

#include "../../util/math.h"
#include "../../util/rocm_device_utils.cuh"
#include "../../util/rocm_tdm.cuh"
#include "../core/rocm_tdm_common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace tdm_kernel {

using common::use_tdm;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t WAVE_SIZE = 32;
constexpr size_t WAVES = THREADS_PER_CHUNK / WAVE_SIZE;    // 4
constexpr size_t BAND_Y = CHUNK_DIM_Y / WAVES;             // 32
constexpr size_t COLS_PER_LANE = CHUNK_DIM_X / WAVE_SIZE;  // 4
constexpr size_t QUADS = BAND_Y / 4;                       // 8
constexpr size_t MULTI_MAX_TENSORS = 256;
static_assert(CHUNK_DIM_Y == common::tdm_dbias_kernel::CHUNK_DIM_Y,
              "A dbias partial row covers a chunk");

template <typename T>
using band_t = T[BAND_Y][CHUNK_DIM_X];

#ifdef __gfx1250__
// Like the generic kernels, plain casts saturate as rocm_pack_4xfloat8 does, Inf included, and
// fused ones as static_cast<OType>, which keeps Inf and NaN.
template <typename OType, bool SATURATE_INF>
__device__ __forceinline__ uint32_t cvt_4x_to_fp8(const float x0, const float x1, const float x2,
                                                  const float x3) {
  if constexpr (SATURATE_INF) {
    return rocm_pack_4xfloat8<OType>(x0, x1, x2, x3);
  } else {
    constexpr float max = transformer_engine::detail::TypeExtrema<OType>::max;
    const auto saturate = [](const float x) {
      return __builtin_isfinite(x) ? __builtin_amdgcn_fmed3f(x, max, -max) : x;
    };
    int packed = 0;
    if constexpr (std::is_same_v<OType, fp8e4m3>) {
      packed = __builtin_amdgcn_cvt_pk_fp8_f32(saturate(x0), saturate(x1), packed, false);
      packed = __builtin_amdgcn_cvt_pk_fp8_f32(saturate(x2), saturate(x3), packed, true);
    } else {
      packed = __builtin_amdgcn_cvt_pk_bf8_f32(saturate(x0), saturate(x1), packed, false);
      packed = __builtin_amdgcn_cvt_pk_bf8_f32(saturate(x2), saturate(x3), packed, true);
    }
    return static_cast<uint32_t>(packed);
  }
}
#endif  // __gfx1250__

// Quantizes chunk (chunk_Y, chunk_X) of a rows x cols tensor into output and, with TRANSPOSE, into
// its cols x rows transpose output_t. The input holds the first in_rows rows; the rest are zero.
// Writes the chunk's dbias partials to row chunk_Y of dbias_workspace.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), bool TRANSPOSE, typename IType, typename OType>
__device__ __forceinline__ void quantize_chunk(const IType *input, const IType *act_input,
                                               OType *output, OType *output_t, const float scale,
                                               float *amax_ptr, float *dbias_workspace,
                                               const size_t in_rows, const size_t rows,
                                               const size_t cols, const size_t chunk_Y,
                                               const size_t chunk_X) {
#ifdef __gfx1250__
  constexpr bool FUSED = IS_DBIAS || IS_DACT || IS_ACT;
  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  __shared__ alignas(128)
      IType act_sh[IS_DACT ? WAVES : 1][IS_DACT ? BAND_Y : 1][IS_DACT ? CHUNK_DIM_X : 1];
  __shared__ alignas(128) band_t<OType> out_sh[WAVES];
  // Wave w stores rows [w * BAND_Y, (w + 1) * BAND_Y) of the transposed chunk. TDM stores read LDS
  // densely.
  __shared__ alignas(128) OType t_sh[TRANSPOSE ? WAVES : 1][BAND_Y][CHUNK_DIM_Y];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_offset_X = lane * COLS_PER_LANE;
  const size_t chunk_offset_Y = chunk_Y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = chunk_X * CHUNK_DIM_X;
  const size_t band_offset_Y = chunk_offset_Y + wave * BAND_Y;
  auto &in = in_sh[wave];
  auto &out = out_sh[wave];
  // Out-of-bounds elements are zero in LDS and every activation (or derivative times a zero
  // gradient) keeps them zero, so they leave the amax and dbias unchanged.
  float amax = 0;
  float partial_dbias[COLS_PER_LANE] = {};

  if (band_offset_Y < rows) {
    tdm::load_2d(in, input, in_rows, cols, cols, band_offset_Y, chunk_offset_X);
    if constexpr (IS_DACT) {
      tdm::load_2d(act_sh[wave], act_input, in_rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
    tdm::wait<0>();

    // A lane converts its COLS_PER_LANE columns a quad of rows at a time and transposes each 4x4
    // block of bytes. A transposed dword's bank depends only on its quad, so the lanes start at
    // different quads to spread their transposed writes over QUADS banks.
    const int first_quad = TRANSPOSE ? lane % QUADS : 0;
#pragma unroll
    for (int step = 0; step < QUADS; step++) {
      const int quad = (first_quad + step) % QUADS;
      uint32_t packed[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int y = quad * 4 + i;
        Vec<IType, COLS_PER_LANE> x;
        Vec<IType, COLS_PER_LANE> a;
        x.load_from(&in[y][lane_offset_X]);
        if constexpr (IS_DACT) {
          a.load_from(&act_sh[wave][y][lane_offset_X]);
        }
        float elt[COLS_PER_LANE];
#pragma unroll
        for (int j = 0; j < COLS_PER_LANE; j++) {
          elt[j] = static_cast<float>(x.data.elt[j]);
          if constexpr (IS_ACT) {
            elt[j] = OP(elt[j], {});
          }
          if constexpr (IS_DACT) {
            elt[j] *= OP(static_cast<float>(a.data.elt[j]), {});
          }
          if constexpr (IS_DBIAS) {
            partial_dbias[j] += elt[j];
          }
          __builtin_assume(amax >= 0);
          amax = fmaxf(fabsf(elt[j]), amax);
        }
        packed[i] = cvt_4x_to_fp8<OType, !FUSED>(elt[0] * scale, elt[1] * scale, elt[2] * scale,
                                                 elt[3] * scale);
        *reinterpret_cast<uint32_t *>(&out[y][lane_offset_X]) = packed[i];
      }
      if constexpr (TRANSPOSE) {
        // Column j of the quad: byte j of each row
        const uint32_t lo01 = __builtin_amdgcn_perm(packed[1], packed[0], 0x05010400);
        const uint32_t hi01 = __builtin_amdgcn_perm(packed[1], packed[0], 0x07030602);
        const uint32_t lo23 = __builtin_amdgcn_perm(packed[3], packed[2], 0x05010400);
        const uint32_t hi23 = __builtin_amdgcn_perm(packed[3], packed[2], 0x07030602);
        const uint32_t column[COLS_PER_LANE] = {__builtin_amdgcn_perm(lo23, lo01, 0x05040100),
                                                __builtin_amdgcn_perm(lo23, lo01, 0x07060302),
                                                __builtin_amdgcn_perm(hi23, hi01, 0x05040100),
                                                __builtin_amdgcn_perm(hi23, hi01, 0x07060302)};
#pragma unroll
        for (int j = 0; j < COLS_PER_LANE; j++) {
          const int x = lane_offset_X + j;
          *reinterpret_cast<uint32_t *>(&t_sh[x / BAND_Y][x % BAND_Y][wave * BAND_Y + quad * 4]) =
              column[j];
        }
      }
    }
    tdm::store_2d(output, out, rows, cols, cols, band_offset_Y, chunk_offset_X);
  }

  // Every wave leaves its column sums and amax in its consumed in_sh band.
  asm volatile("" ::: "memory");  // in_sh is reused as float
  float *const wave_scratch = reinterpret_cast<float *>(in_sh[wave]);
  if constexpr (IS_DBIAS) {
    Vec<float, COLS_PER_LANE> partial;
#pragma unroll
    for (int i = 0; i < COLS_PER_LANE; i++) {
      partial.data.elt[i] = partial_dbias[i];
    }
    partial.store_to(wave_scratch + lane_offset_X);
  }
  if (amax_ptr != nullptr) {
    const float wave_amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
    if (lane == 0) {
      wave_scratch[CHUNK_DIM_X] = wave_amax;
    }
  }
  __syncthreads();
  if constexpr (TRANSPOSE) {
    // Rows past the tensor's are garbage in t_sh but outside the transpose's columns.
    const size_t t_offset_Y = chunk_offset_X + wave * BAND_Y;
    if (t_offset_Y < cols) {
      tdm::store_2d(output_t, t_sh[wave], cols, rows, rows, t_offset_Y, chunk_offset_Y);
    }
  }
  if constexpr (IS_DBIAS) {
    const size_t col = chunk_offset_X + threadIdx.x;
    if (col < cols) {
      float dbias = 0;
#pragma unroll
      for (int w = 0; w < WAVES; w++) {
        dbias += reinterpret_cast<const float *>(in_sh[w])[threadIdx.x];
      }
      dbias_workspace[chunk_Y * cols + col] = dbias;
    }
  }

  if (amax_ptr != nullptr && threadIdx.x == 0) {
    float block_amax = 0;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      block_amax = fmaxf(block_amax, reinterpret_cast<const float *>(in_sh[w])[CHUNK_DIM_X]);
    }
    // Atomics on one address serialize, so only those that may raise the amax are issued. It only
    // grows during the kernel, so a stale read can only cause a redundant atomic.
    if (block_amax > *amax_ptr) {
      rocm_atomicMaxFloat(amax_ptr, block_amax);
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), bool TRANSPOSE, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_fp8_tdm_kernel(const IType *input, const IType *act_input, OType *output,
                            OType *output_t, const float *noop, const float *scale_ptr,
                            float *amax_ptr, float *scale_inv_ptr, float *dbias_workspace,
                            const size_t rows, const size_t cols) {
  // The launcher leaves fused quantizations with a noop flag to the generic kernels.
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const float scale = scale_ptr != nullptr ? *scale_ptr : 1.0f;
  quantize_chunk<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, TRANSPOSE, IType, OType>(
      input, act_input, output, output_t, scale, amax_ptr, dbias_workspace, rows, rows, cols,
      blockIdx.y, blockIdx.x);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && scale_inv_ptr != nullptr) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
}

struct MultiCastTransposeArgs {
  const void *input[MULTI_MAX_TENSORS];
  void *output[MULTI_MAX_TENSORS];
  void *output_t[MULTI_MAX_TENSORS];
  const float *scale[MULTI_MAX_TENSORS];
  float *amax[MULTI_MAX_TENSORS];
  float *scale_inv[MULTI_MAX_TENSORS];
  uint32_t in_rows[MULTI_MAX_TENSORS];
  uint32_t rows[MULTI_MAX_TENSORS];
  uint32_t cols[MULTI_MAX_TENSORS];
  // Tensor t owns chunk rows [first_chunk_Y[t], first_chunk_Y[t + 1]).
  uint32_t first_chunk_Y[MULTI_MAX_TENSORS + 1];
  int num_tensors;
};

template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    multi_cast_transpose_fp8_tdm_kernel(const MultiCastTransposeArgs args) {
  int t = 0;
  int hi = args.num_tensors - 1;
  while (t < hi) {
    const int mid = t + (hi - t) / 2;
    if (args.first_chunk_Y[mid + 1] <= blockIdx.y) {
      t = mid + 1;
    } else {
      hi = mid;
    }
  }
  const size_t cols = args.cols[t];
  const size_t chunk_Y = blockIdx.y - args.first_chunk_Y[t];
  if (blockIdx.x * CHUNK_DIM_X >= cols) {
    return;
  }
  const float scale = args.scale[t] != nullptr ? *args.scale[t] : 1.0f;
  quantize_chunk<false, false, false, Empty, nullptr, true, IType, OType>(
      static_cast<const IType *>(args.input[t]), nullptr, static_cast<OType *>(args.output[t]),
      static_cast<OType *>(args.output_t[t]), scale, args.amax[t], nullptr, args.in_rows[t],
      args.rows[t], cols, chunk_Y, blockIdx.x);
  if (chunk_Y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && args.scale_inv[t] != nullptr) {
    *args.scale_inv[t] = __frcp_rn(scale);
  }
}

}  // namespace tdm_kernel

// FP8 tensor scaling of input, after an activation or its derivative, into output's rowwise data
// and, if allocated, its transposed columnwise data.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
bool quantize_tdm(const Tensor &input, const Tensor *act_input, const Tensor *noop, Tensor *output,
                  Tensor *dbias, Tensor *workspace, hipStream_t stream) {
  using namespace tdm_kernel;
  constexpr bool FUSED = IS_DBIAS || IS_DACT || IS_ACT;
  const bool transpose = output->has_columnwise_data();
  const auto [rows, cols] = input.flat_2d_dims();
  const bool has_noop = noop != nullptr && noop->data.dptr != nullptr;
  // The transpose's rows are rows long.
  if (!output->has_data() || !is_tensor_scaling(output->scaling_mode) ||
      is_fp8_dtype(input.dtype()) || !is_fp8_dtype(output->dtype()) || rows == 0 ||
      (transpose && rows % 16 != 0) || (FUSED && has_noop) ||
      (IS_DACT && (act_input == nullptr || !act_input->has_data())) ||
      (IS_DBIAS && (dbias == nullptr || workspace == nullptr)) ||
      !use_tdm(rows, cols,
               {input.data.dptr, IS_DACT ? act_input->data.dptr : nullptr, output->data.dptr,
                output->columnwise_data.dptr})) {
    return false;
  }
  // One row of dbias partials per chunk row
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  if constexpr (IS_DBIAS) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {chunks_Y, cols};
      workspace->data.dtype = DType::kFloat32;
      return true;
    }
    NVTE_CHECK(workspace->data.dtype == DType::kFloat32 &&
                   workspace->data.numel() >= chunks_Y * cols &&
                   reinterpret_cast<uintptr_t>(workspace->data.dptr) % 16 == 0,
               "Invalid dbias workspace (expected an aligned FP32 buffer of ", chunks_Y * cols,
               " elements)");
    CheckOutputTensor(*dbias, "quantize_fp8_tdm_dbias");
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
  }
  if constexpr (IS_DACT) {
    CheckInputTensor(*act_input, "quantize_fp8_tdm_act_input");
    NVTE_CHECK(input.dtype() == act_input->dtype(), "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input->data.shape, "Shapes of both inputs must match.");
  }
  if (has_noop) {
    CheckNoopTensor(*noop, "quantize_fp8_tdm_noop");
  }
  CheckInputTensor(input, "quantize_fp8_tdm_input");
  CheckOutputTensor(*output, "quantize_fp8_tdm_output");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes do not match (input=",
             input.data.shape, ", output=", output->data.shape, ")");
  if (transpose) {
    NVTE_CHECK(output->data.dtype == output->columnwise_data.dtype,
               "Cast and transposed output types must match.");
    NVTE_CHECK(output->scale_inv.dptr == output->columnwise_scale_inv.dptr,
               "Cast and transposed outputs need to share scale-inverse tensor.");
  }

  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), chunks_Y);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              transpose, TRANSPOSE,
              quantize_fp8_tdm_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, TRANSPOSE, IType,
                                      OType><<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                  reinterpret_cast<const IType *>(input.data.dptr),
                  IS_DACT ? reinterpret_cast<const IType *>(act_input->data.dptr) : nullptr,
                  reinterpret_cast<OType *>(output->data.dptr),
                  reinterpret_cast<OType *>(output->columnwise_data.dptr),
                  has_noop ? reinterpret_cast<const float *>(noop->data.dptr) : nullptr,
                  reinterpret_cast<const float *>(output->scale.dptr),
                  reinterpret_cast<float *>(output->amax.dptr),
                  reinterpret_cast<float *>(output->scale_inv.dptr), workspace_ptr, rows,
                  cols);)));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  if constexpr (IS_DBIAS) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input.dtype(), IType,
        common::reduce_dbias_tdm<IType>(reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr,
                                        cols, 1, chunks_Y, nullptr, nullptr, stream););
  }
  return true;
}

// FP8 cast and transpose of each input into its output, whose rows past the input's (with
// valid_num_rows, valid_num_rows[i] of them) are zero padding. TDM only if every tensor qualifies.
inline bool multi_cast_transpose_tdm(const std::vector<Tensor *> &input_list,
                                     const std::vector<Tensor *> &output_list,
                                     const int *valid_num_rows, hipStream_t stream) {
  using namespace tdm_kernel;
  const size_t num_tensors = input_list.size();
  for (size_t i = 0; i < num_tensors; i++) {
    const Tensor &output = *output_list[i];
    const size_t rows = output.data.shape[0];
    if (!is_tensor_scaling(output.scaling_mode) || is_fp8_dtype(input_list[i]->dtype()) ||
        !is_fp8_dtype(output.dtype()) || rows % 16 != 0 ||
        !use_tdm(rows, output.data.shape[1],
                 {input_list[i]->data.dptr, output.data.dptr, output.columnwise_data.dptr})) {
      return false;
    }
  }
  for (size_t first = 0; first < num_tensors; first += MULTI_MAX_TENSORS) {
    const size_t count = std::min(num_tensors - first, MULTI_MAX_TENSORS);
    MultiCastTransposeArgs args;
    args.num_tensors = count;
    args.first_chunk_Y[0] = 0;
    size_t chunks_X = 0;
    for (size_t j = 0; j < count; j++) {
      const Tensor &input = *input_list[first + j];
      const Tensor &output = *output_list[first + j];
      const size_t rows = output.data.shape[0];
      const size_t cols = output.data.shape[1];
      args.input[j] = input.data.dptr;
      args.output[j] = output.data.dptr;
      args.output_t[j] = output.columnwise_data.dptr;
      args.scale[j] = reinterpret_cast<const float *>(output.scale.dptr);
      args.amax[j] = reinterpret_cast<float *>(output.amax.dptr);
      args.scale_inv[j] = reinterpret_cast<float *>(output.scale_inv.dptr);
      args.in_rows[j] = valid_num_rows != nullptr ? valid_num_rows[first + j] : rows;
      args.rows[j] = rows;
      args.cols[j] = cols;
      args.first_chunk_Y[j + 1] = args.first_chunk_Y[j] + DIVUP(rows, CHUNK_DIM_Y);
      chunks_X = std::max(chunks_X, DIVUP(cols, CHUNK_DIM_X));
    }
    if (args.first_chunk_Y[count] == 0 || chunks_X == 0) {
      continue;
    }
    const dim3 grid(chunks_X, args.first_chunk_Y[count]);
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input_list[0]->dtype(), IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output_list[0]->dtype(), OType,
            multi_cast_transpose_fp8_tdm_kernel<IType, OType>
            <<<grid, THREADS_PER_CHUNK, 0, stream>>>(args);));  // NOLINT(*)
    NVTE_CHECK_CUDA(hipGetLastError());
  }
  return true;
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine
