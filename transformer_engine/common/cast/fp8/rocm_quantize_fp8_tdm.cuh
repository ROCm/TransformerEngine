/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// gfx1250 FP8 tensor-scaling quantize with the Tensor Data Mover: each wave moves its own
// BAND_Y-row band of a 128x128 chunk global->LDS->global, and for the transposed output the waves
// assemble the chunk's transpose in LDS. 2D block scaling quantizes each chunk with its own scale,
// and dequantize moves bands the same way. The launchers return false where TDM does not apply,
// leaving the work to the generic kernels. Like the other rocm_ headers, this expects its includer
// to have included the common headers, util/system.h and util/cuda_runtime.h.

#include <vector>

#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../util/rocm_amax_slots.cuh"
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
using common::cvt_from_float;

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

// Writes the lane's COLS_PER_LANE columns of a quad of packed rows of wave's band, transposed, into
// the chunk's transpose t.
template <typename OType, size_t N>
__device__ __forceinline__ void store_transposed_quad(OType (&t)[N][BAND_Y][CHUNK_DIM_Y],
                                                      const uint32_t (&packed)[4],
                                                      const int lane_offset_X, const int wave,
                                                      const int quad) {
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
    *reinterpret_cast<uint32_t *>(&t[x / BAND_Y][x % BAND_Y][wave * BAND_Y + quad * 4]) =
        column[j];
  }
}
#endif  // __gfx1250__

// Quantizes chunk (chunk_Y, chunk_X) of a rows x cols tensor into output and, with TRANSPOSE, into
// its cols x rows transpose output_t. The input holds the first in_rows rows; the rest are zero.
// Writes the chunk's dbias partials to row chunk_Y of dbias_workspace and raises the amax at its
// rocm_amax_target.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), bool TRANSPOSE, typename IType, typename OType>
__device__ __forceinline__ void quantize_chunk(const IType *input, const IType *act_input,
                                               OType *output, OType *output_t, const float scale,
                                               float *amax_ptr, float *amax_slots,
                                               float *dbias_workspace, const size_t in_rows,
                                               const size_t rows, const size_t cols,
                                               const size_t chunk_Y, const size_t chunk_X) {
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
        store_transposed_quad(t_sh, packed, lane_offset_X, wave, quad);
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
    float *const amax_dst =
        rocm_amax_target(amax_ptr, amax_slots, blockIdx.y * gridDim.x + blockIdx.x);
    // Atomics on one address serialize, so only those that may raise the amax are issued. It only
    // grows during the kernel, so a stale read can only cause a redundant atomic.
    if (block_amax > *amax_dst) {
      rocm_atomicMaxFloat(amax_dst, block_amax);
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
                            float *amax_ptr, float *amax_slots, float *scale_inv_ptr,
                            float *dbias_workspace, const size_t rows, const size_t cols) {
  // The launcher leaves fused quantizations with a noop flag to the generic kernels.
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const float scale = scale_ptr != nullptr ? *scale_ptr : 1.0f;
  quantize_chunk<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, TRANSPOSE, IType, OType>(
      input, act_input, output, output_t, scale, amax_ptr, amax_slots, dbias_workspace, rows, rows,
      cols, blockIdx.y, blockIdx.x);
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
      static_cast<OType *>(args.output_t[t]), scale, args.amax[t], nullptr, nullptr,
      args.in_rows[t], args.rows[t], cols, chunk_Y, blockIdx.x);
  if (chunk_Y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && args.scale_inv[t] != nullptr) {
    *args.scale_inv[t] = __frcp_rn(scale);
  }
}

// FP8 tensor scaling of chunk (blockIdx.y, blockIdx.x) of a gated activation of input, whose rows
// hold the cols act inputs and then the cols gate inputs, or with IS_DGATED of its act and gate
// gradients for grad. The element math is that of gated_act_kernel and dgated_act_kernel.
template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), bool TRANSPOSE, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_gated_fp8_tdm_kernel(const IType *grad, const IType *input, OType *output,
                                  OType *output_t, const float *scale_ptr, float *amax_ptr,
                                  float *amax_slots, float *scale_inv_ptr, const size_t rows,
                                  const size_t cols, const ParamOP p) {
#ifdef __gfx1250__
  // The act and gate gradient halves of each output row, or the activation alone
  constexpr int HALVES = IS_DGATED ? 2 : 1;
  // With TRANSPOSE, the packed outputs stay in registers until the consumed input bands can take
  // them: a wave's output bands go into its act band, and the transposed chunk into the gate bands.
  // Without it, a wave writes its rows into its own output bands as it computes them.
  static_assert(HALVES * sizeof(band_t<OType>) <= sizeof(band_t<IType>));
  static_assert(HALVES * WAVES * BAND_Y * CHUNK_DIM_Y * sizeof(OType) <=
                WAVES * sizeof(band_t<IType>));
  // The act, gate and, with IS_DGATED, grad bands of the waves
  __shared__ alignas(128) band_t<IType> in_sh[IS_DGATED ? 3 : 2][WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[TRANSPOSE ? 1 : HALVES][TRANSPOSE ? 1 : WAVES];
  auto &act_sh = in_sh[0];
  auto &gate_sh = in_sh[1];
  auto &grad_sh = in_sh[IS_DGATED ? 2 : 0];
  // A wave's amax goes after its output bands, or into its grad band if they fill the act band.
  const auto wave_amax = [&](const int w) -> float * {
    if constexpr (HALVES * sizeof(band_t<OType>) < sizeof(band_t<IType>)) {
      return reinterpret_cast<float *>(reinterpret_cast<char *>(act_sh[w]) +
                                       HALVES * sizeof(band_t<OType>));
    } else {
      return reinterpret_cast<float *>(grad_sh[w]);
    }
  };

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_offset_X = lane * COLS_PER_LANE;
  const size_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t band_offset_Y = chunk_offset_Y + wave * BAND_Y;
  const bool has_band = band_offset_Y < rows;
  const float scale = scale_ptr != nullptr ? *scale_ptr : 1.0f;
  const auto out_band = [&](const int h) -> band_t<OType> & {
    if constexpr (TRANSPOSE) {
      return reinterpret_cast<band_t<OType> *>(act_sh[wave])[h];
    } else {
      return out_sh[h][wave];
    }
  };
  // Out-of-bounds elements are zero and every gated activation and gradient keeps them zero.
  float amax = 0;
  // As in quantize_chunk, the lanes start at different quads to spread the transposed writes. The
  // packed rows of step s are those of quad (first_quad + s) % QUADS.
  const int first_quad = TRANSPOSE ? lane % QUADS : 0;
  uint32_t packed[QUADS][HALVES][4];

  if (has_band) {
    tdm::load_2d(act_sh[wave], input, rows, cols, 2 * cols, band_offset_Y, chunk_offset_X);
    tdm::load_2d(gate_sh[wave], input + cols, rows, cols, 2 * cols, band_offset_Y, chunk_offset_X);
    if constexpr (IS_DGATED) {
      tdm::load_2d(grad_sh[wave], grad, rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
    tdm::wait<0>();

#pragma unroll
    for (int step = 0; step < QUADS; step++) {
      const int quad = (first_quad + step) % QUADS;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int y = quad * 4 + i;
        Vec<IType, COLS_PER_LANE> a, g, d;
        a.load_from(&act_sh[wave][y][lane_offset_X]);
        g.load_from(&gate_sh[wave][y][lane_offset_X]);
        if constexpr (IS_DGATED) {
          d.load_from(&grad_sh[wave][y][lane_offset_X]);
        }
        float elt[HALVES][COLS_PER_LANE];
#pragma unroll
        for (int j = 0; j < COLS_PER_LANE; j++) {
          const float act_in = static_cast<float>(a.data.elt[j]);
          float gate_in = static_cast<float>(g.data.elt[j]);
          float dgate_in = 1.0f;
          if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
            const float limit = p.limit;
            dgate_in = gate_in <= limit && gate_in >= -limit;
            gate_in = std::min(std::max(-limit, gate_in), limit) + p.glu_linear_offset;
          } else if constexpr (std::is_same_v<ParamOP, SiTUGLUParam>) {
            if constexpr (IS_DGATED) {
              dgate_in = dsitu_up<float, float>(gate_in, p);
            }
            gate_in = situ_up<float, float>(gate_in, p);
          }
          if constexpr (IS_DGATED) {
            const float grad_val = static_cast<float>(d.data.elt[j]);
            elt[0][j] = DActOP(act_in, p) * grad_val * gate_in;
            elt[1][j] = dgate_in * grad_val * ActOP(act_in, p);
          } else {
            elt[0][j] = ActOP(act_in, p) * gate_in;
          }
#pragma unroll
          for (int h = 0; h < HALVES; h++) {
            __builtin_assume(amax >= 0);
            amax = fmaxf(fabsf(elt[h][j]), amax);
          }
        }
#pragma unroll
        for (int h = 0; h < HALVES; h++) {
          packed[step][h][i] = cvt_4x_to_fp8<OType, false>(elt[h][0] * scale, elt[h][1] * scale,
                                                           elt[h][2] * scale, elt[h][3] * scale);
          if constexpr (!TRANSPOSE) {
            *reinterpret_cast<uint32_t *>(&out_band(h)[y][lane_offset_X]) = packed[step][h][i];
          }
        }
      }
    }

    if constexpr (TRANSPOSE) {
      asm volatile("" ::: "memory");  // act_sh is reused for the output bands
#pragma unroll
      for (int step = 0; step < QUADS; step++) {
        const int quad = (first_quad + step) % QUADS;
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
          for (int h = 0; h < HALVES; h++) {
            *reinterpret_cast<uint32_t *>(&out_band(h)[quad * 4 + i][lane_offset_X]) =
                packed[step][h][i];
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < HALVES; h++) {
      tdm::store_2d(output + h * cols, out_band(h), rows, cols, HALVES * cols, band_offset_Y,
                    chunk_offset_X);
    }
  }

  asm volatile("" ::: "memory");  // the consumed bands are reused as float
  if (amax_ptr != nullptr) {
    const float reduced = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
    if (lane == 0) {
      *wave_amax(wave) = reduced;
    }
  }
  if constexpr (TRANSPOSE) {
    __syncthreads();  // every wave is done with the gate bands
    asm volatile("" ::: "memory");
    auto &t_sh = *reinterpret_cast<OType(*)[HALVES][WAVES][BAND_Y][CHUNK_DIM_Y]>(gate_sh);
    if (has_band) {
#pragma unroll
      for (int step = 0; step < QUADS; step++) {
#pragma unroll
        for (int h = 0; h < HALVES; h++) {
          store_transposed_quad(t_sh[h], packed[step][h], lane_offset_X, wave,
                                (first_quad + step) % QUADS);
        }
      }
    }
    __syncthreads();
    // Rows past the tensor's are garbage in t_sh but outside the transpose's columns.
    const size_t t_offset_Y = chunk_offset_X + wave * BAND_Y;
    if (t_offset_Y < cols) {
#pragma unroll
      for (int h = 0; h < HALVES; h++) {
        tdm::store_2d(output_t + h * cols * rows, t_sh[h][wave], cols, rows, rows, t_offset_Y,
                      chunk_offset_Y);
      }
    }
  } else {
    __syncthreads();
  }
  if (amax_ptr != nullptr && threadIdx.x == 0) {
    float block_amax = 0;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      block_amax = fmaxf(block_amax, *wave_amax(w));
    }
    float *const amax_dst =
        rocm_amax_target(amax_ptr, amax_slots, blockIdx.y * gridDim.x + blockIdx.x);
    if (block_amax > *amax_dst) {
      rocm_atomicMaxFloat(amax_dst, block_amax);
    }
  }
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && scale_inv_ptr != nullptr) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

// FP8 tensor-scaling dequantize of chunk (blockIdx.y, blockIdx.x): per step, 8 lanes cover a row of
// the wave's band with 16 elements each.
template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_fp8_tdm_kernel(const IType *input, OType *output, const float *scale_inv_ptr,
                              const size_t rows, const size_t cols) {
#ifdef __gfx1250__
  constexpr int ELEMS = 16;
  constexpr int LANES_PER_ROW = CHUNK_DIM_X / ELEMS;        // 8
  constexpr int ROWS_PER_STEP = WAVE_SIZE / LANES_PER_ROW;  // 4
  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[WAVES];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const size_t band_offset_Y = blockIdx.y * CHUNK_DIM_Y + wave * BAND_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  if (band_offset_Y >= rows) {
    return;
  }
  tdm::load_2d(in_sh[wave], input, rows, cols, cols, band_offset_Y, chunk_offset_X);
  const float scale_inv = *scale_inv_ptr;
  tdm::wait<0>();
  const int lane_offset_X = (lane % LANES_PER_ROW) * ELEMS;
#pragma unroll
  for (int step = 0; step < BAND_Y / ROWS_PER_STEP; step++) {
    const int y = step * ROWS_PER_STEP + lane / LANES_PER_ROW;
    Vec<IType, ELEMS> in;
    Vec<OType, ELEMS> out;
    in.load_from(&in_sh[wave][y][lane_offset_X]);
#pragma unroll
    for (int i = 0; i < ELEMS; i++) {
      out.data.elt[i] = cvt_from_float<OType>(static_cast<float>(in.data.elt[i]) * scale_inv);
    }
    out.store_to(&out_sh[wave][y][lane_offset_X]);
  }
  tdm::store_2d(output, out_sh[wave], rows, cols, cols, band_offset_Y, chunk_offset_X);
  tdm::wait<0>();
#endif  // __gfx1250__
}

// FP8 2D block scaling of chunk (blockIdx.y, blockIdx.x) with the scale from the chunk's amax into
// output, if given, and, with TRANSPOSE, into output_t. The scale inverse goes where the generic
// block_scaled_cast_transpose_kernel_notaligned puts it.
template <bool TRANSPOSE, typename IType, typename OType>
__device__ __forceinline__ void quantize_block2d_chunk(
    const IType *input, OType *output, OType *output_t, float *scale_inv, float *scale_inv_t,
    const size_t rows, const size_t cols, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    const bool pow_2_scaling) {
#ifdef __gfx1250__
  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[WAVES];
  __shared__ alignas(128) OType t_sh[TRANSPOSE ? WAVES : 1][BAND_Y][CHUNK_DIM_Y];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_offset_X = lane * COLS_PER_LANE;
  const size_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t band_offset_Y = chunk_offset_Y + wave * BAND_Y;
  const bool has_band = band_offset_Y < rows;
  auto &in = in_sh[wave];
  auto &out = out_sh[wave];

  // Out-of-bounds elements are zero in LDS and leave the amax unchanged.
  float amax = 0;
  if (has_band) {
    tdm::load_2d(in, input, rows, cols, cols, band_offset_Y, chunk_offset_X);
    tdm::wait<0>();
#pragma unroll
    for (int y = 0; y < BAND_Y; y++) {
      Vec<IType, COLS_PER_LANE> x;
      x.load_from(&in[y][lane_offset_X]);
#pragma unroll
      for (int j = 0; j < COLS_PER_LANE; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(fabsf(static_cast<float>(x.data.elt[j])), amax);
      }
    }
  }
  // The waves exchange their amaxes through their not yet written out_sh bands.
  amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
  asm volatile("" ::: "memory");  // out_sh is used as float
  if (lane == 0) {
    *reinterpret_cast<float *>(out) = amax;
  }
  __syncthreads();
  float block_amax = 0;
#pragma unroll
  for (int w = 0; w < WAVES; w++) {
    block_amax = fmaxf(block_amax, *reinterpret_cast<const float *>(out_sh[w]));
  }
  __syncthreads();
  asm volatile("" ::: "memory");
  const float scale = compute_scale_from_types<IType, OType>(block_amax, epsilon, pow_2_scaling);
  if (threadIdx.x == 0) {
    const float s_inv = 1.0f / scale;
    if (scale_inv != nullptr) {
      scale_inv[blockIdx.y * scale_stride_y + blockIdx.x * scale_stride_x] = s_inv;
    }
    if constexpr (TRANSPOSE) {
      scale_inv_t[blockIdx.x * scale_t_stride_y + blockIdx.y * scale_t_stride_x] = s_inv;
    }
  }

  if (has_band) {
    // As in quantize_chunk, the lanes start at different quads to spread the transposed writes.
    const int first_quad = TRANSPOSE ? lane % QUADS : 0;
#pragma unroll
    for (int step = 0; step < QUADS; step++) {
      const int quad = (first_quad + step) % QUADS;
      uint32_t packed[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int y = quad * 4 + i;
        Vec<IType, COLS_PER_LANE> x;
        x.load_from(&in[y][lane_offset_X]);
        packed[i] = cvt_4x_to_fp8<OType, false>(
            static_cast<float>(x.data.elt[0]) * scale, static_cast<float>(x.data.elt[1]) * scale,
            static_cast<float>(x.data.elt[2]) * scale, static_cast<float>(x.data.elt[3]) * scale);
        if (output != nullptr) {
          *reinterpret_cast<uint32_t *>(&out[y][lane_offset_X]) = packed[i];
        }
      }
      if constexpr (TRANSPOSE) {
        store_transposed_quad(t_sh, packed, lane_offset_X, wave, quad);
      }
    }
    if (output != nullptr) {
      tdm::store_2d(output, out, rows, cols, cols, band_offset_Y, chunk_offset_X);
    }
  }
  if constexpr (TRANSPOSE) {
    __syncthreads();
    // Rows past the tensor's are garbage in t_sh but outside the transpose's columns.
    const size_t t_offset_Y = chunk_offset_X + wave * BAND_Y;
    if (t_offset_Y < cols) {
      tdm::store_2d(output_t, t_sh[wave], cols, rows, rows, t_offset_Y, chunk_offset_Y);
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

// FP8 1D block scaling of chunk (blockIdx.y, blockIdx.x): each chunk row with its own scale into
// output, if given, and, with TRANSPOSE, each chunk column with its own scale into output_t. The
// scale inverses go where the generic block_scaled_1d_cast_transpose_kernel puts them.
template <bool TRANSPOSE, typename IType, typename OType>
__device__ __forceinline__ void quantize_block1d_chunk(
    const IType *input, OType *output, OType *output_t, float *scale_inv, float *scale_inv_t,
    const size_t rows, const size_t cols, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    const bool pow_2_scaling) {
#ifdef __gfx1250__
  constexpr int ELEMS = 16;
  constexpr int LANES_PER_ROW = CHUNK_DIM_X / ELEMS;        // 8
  constexpr int ROWS_PER_STEP = WAVE_SIZE / LANES_PER_ROW;  // 4
  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[WAVES];
  __shared__ alignas(128) OType t_sh[TRANSPOSE ? WAVES : 1][BAND_Y][CHUNK_DIM_Y];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const size_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t band_offset_Y = chunk_offset_Y + wave * BAND_Y;
  const bool has_band = band_offset_Y < rows;
  auto &in = in_sh[wave];
  auto &out = out_sh[wave];

  if (has_band) {
    tdm::load_2d(in, input, rows, cols, cols, band_offset_Y, chunk_offset_X);
    tdm::wait<0>();
  }

  // For the transpose, a lane owns COLS_PER_LANE columns. Out-of-bounds elements are zero in LDS
  // and leave the amaxes unchanged.
  const int col_offset_X = lane * COLS_PER_LANE;
  float col_scale[COLS_PER_LANE];
  if constexpr (TRANSPOSE) {
    Vec<float, COLS_PER_LANE> col_amax;
    col_amax.clear();
    if (has_band) {
#pragma unroll
      for (int y = 0; y < BAND_Y; y++) {
        Vec<IType, COLS_PER_LANE> x;
        x.load_from(&in[y][col_offset_X]);
#pragma unroll
        for (int j = 0; j < COLS_PER_LANE; j++) {
          __builtin_assume(col_amax.data.elt[j] >= 0);
          col_amax.data.elt[j] =
              fmaxf(col_amax.data.elt[j], fabsf(static_cast<float>(x.data.elt[j])));
        }
      }
    }
    // The waves exchange their band's column amaxes through their not yet written out_sh bands.
    asm volatile("" ::: "memory");  // out_sh is used as float
    col_amax.store_to(reinterpret_cast<float *>(out) + col_offset_X);
    __syncthreads();
#pragma unroll
    for (int w = 1; w < WAVES; w++) {
      Vec<float, COLS_PER_LANE> other;
      other.load_from(reinterpret_cast<const float *>(out_sh[(wave + w) % WAVES]) + col_offset_X);
#pragma unroll
      for (int j = 0; j < COLS_PER_LANE; j++) {
        col_amax.data.elt[j] = fmaxf(col_amax.data.elt[j], other.data.elt[j]);
      }
    }
    __syncthreads();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int j = 0; j < COLS_PER_LANE; j++) {
      col_scale[j] = compute_scale_from_types<IType, OType>(col_amax.data.elt[j], epsilon,
                                                            pow_2_scaling);
      const size_t col = chunk_offset_X + col_offset_X + j;
      if (wave == 0 && col < cols) {
        scale_inv_t[col * scale_t_stride_y + blockIdx.y * scale_t_stride_x] = 1.0 / col_scale[j];
      }
    }
  }

  if (has_band && output != nullptr) {
    // Per step, 8 lanes cover a row with 16 elements each.
    const int lane_offset_X = (lane % LANES_PER_ROW) * ELEMS;
#pragma unroll
    for (int step = 0; step < BAND_Y / ROWS_PER_STEP; step++) {
      const int y = step * ROWS_PER_STEP + lane / LANES_PER_ROW;
      Vec<IType, ELEMS> x;
      x.load_from(&in[y][lane_offset_X]);
      float amax = 0;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<float>(x.data.elt[i])));
      }
      amax = rocm_subwarp_allreduce<LANES_PER_ROW>(amax, rocm_op::max{});
      const float scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
      const size_t row = band_offset_Y + y;
      if (lane % LANES_PER_ROW == 0 && row < rows) {
        scale_inv[row * scale_stride_y + blockIdx.x * scale_stride_x] = 1.0 / scale;
      }
      uint32_t packed[ELEMS / 4];
#pragma unroll
      for (int i = 0; i < ELEMS / 4; i++) {
        packed[i] = cvt_4x_to_fp8<OType, false>(static_cast<float>(x.data.elt[4 * i]) * scale,
                                                static_cast<float>(x.data.elt[4 * i + 1]) * scale,
                                                static_cast<float>(x.data.elt[4 * i + 2]) * scale,
                                                static_cast<float>(x.data.elt[4 * i + 3]) * scale);
      }
      *reinterpret_cast<uint4 *>(&out[y][lane_offset_X]) =
          make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
    tdm::store_2d(output, out, rows, cols, cols, band_offset_Y, chunk_offset_X);
  }

  if constexpr (TRANSPOSE) {
    if (has_band) {
      // As in quantize_chunk, the lanes start at different quads to spread the transposed writes.
      const int first_quad = lane % QUADS;
#pragma unroll
      for (int step = 0; step < QUADS; step++) {
        const int quad = (first_quad + step) % QUADS;
        uint32_t packed[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          Vec<IType, COLS_PER_LANE> x;
          x.load_from(&in[quad * 4 + i][col_offset_X]);
          packed[i] = cvt_4x_to_fp8<OType, false>(static_cast<float>(x.data.elt[0]) * col_scale[0],
                                                  static_cast<float>(x.data.elt[1]) * col_scale[1],
                                                  static_cast<float>(x.data.elt[2]) * col_scale[2],
                                                  static_cast<float>(x.data.elt[3]) * col_scale[3]);
        }
        store_transposed_quad(t_sh, packed, col_offset_X, wave, quad);
      }
    }
    __syncthreads();
    // Rows past the tensor's are garbage in t_sh but outside the transpose's columns.
    const size_t t_offset_Y = chunk_offset_X + wave * BAND_Y;
    if (t_offset_Y < cols) {
      tdm::store_2d(output_t, t_sh[wave], cols, rows, rows, t_offset_Y, chunk_offset_Y);
    }
  }
  tdm::wait<0>();
#endif  // __gfx1250__
}

// FP8 block scaling of chunk (blockIdx.y, blockIdx.x) with BLOCK_DIM 1 (1x128 and 128x1 tiles) or 2
// (128x128 tiles).
template <int BLOCK_DIM, bool TRANSPOSE, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_block_fp8_tdm_kernel(const IType *input, OType *output, OType *output_t,
                                  float *scale_inv, float *scale_inv_t, const size_t rows,
                                  const size_t cols, const size_t scale_stride_x,
                                  const size_t scale_stride_y, const size_t scale_t_stride_x,
                                  const size_t scale_t_stride_y, const float epsilon,
                                  const bool pow_2_scaling, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  if constexpr (BLOCK_DIM == 1) {
    quantize_block1d_chunk<TRANSPOSE>(input, output, output_t, scale_inv, scale_inv_t, rows, cols,
                                      scale_stride_x, scale_stride_y, scale_t_stride_x,
                                      scale_t_stride_y, epsilon, pow_2_scaling);
  } else {
    quantize_block2d_chunk<TRANSPOSE>(input, output, output_t, scale_inv, scale_inv_t, rows, cols,
                                      scale_stride_x, scale_stride_y, scale_t_stride_x,
                                      scale_t_stride_y, epsilon, pow_2_scaling);
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
  // Up to this size, the amax finalize launch costs a plain cast-transpose more than TDM saves.
  constexpr size_t MAX_GENERIC_AMAX_CT_ELEMS = size_t{1} << 25;
  const bool small_amax_ct = !FUSED && transpose && output->amax.dptr != nullptr &&
                             rows * cols <= MAX_GENERIC_AMAX_CT_ELEMS;
  // The transpose's rows are rows long.
  if (!output->has_data() || !is_tensor_scaling(output->scaling_mode) ||
      is_fp8_dtype(input.dtype()) || !is_fp8_dtype(output->dtype()) || rows == 0 ||
      (transpose && rows % 16 != 0) || (FUSED && has_noop) || small_amax_ct ||
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
  const float *const noop_ptr =
      has_noop ? reinterpret_cast<const float *>(noop->data.dptr) : nullptr;
  float *const amax = reinterpret_cast<float *>(output->amax.dptr);
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), chunks_Y);
  float *const amax_slots = amax != nullptr ? rocm_amax_slots(grid.x * grid.y, stream) : nullptr;
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
                  reinterpret_cast<OType *>(output->columnwise_data.dptr), noop_ptr,
                  reinterpret_cast<const float *>(output->scale.dptr), amax, amax_slots,
                  reinterpret_cast<float *>(output->scale_inv.dptr), workspace_ptr, rows,
                  cols);)));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  if (amax_slots != nullptr) {
    rocm_amax_finalize(amax_slots, amax, noop_ptr, stream);
  }
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
// A template, like dequantize_tdm, so that only the files calling it compile its kernels.
template <typename = void>
bool multi_cast_transpose_tdm(const std::vector<Tensor *> &input_list,
                              const std::vector<Tensor *> &output_list, const int *valid_num_rows,
                              hipStream_t stream) {
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

// FP8 tensor scaling of a gated activation of gated_input or, with IS_BWD, of its gradient for grad
// into output's rowwise data and, if allocated, its transposed columnwise data.
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
bool quantize_gated_tdm(const Tensor &gated_input, const Tensor &grad, Tensor *output,
                        const ParamOP &p, hipStream_t stream) {
  using namespace tdm_kernel;
  const bool transpose = output->has_columnwise_data();
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  // The transpose's rows are rows long.
  if (!output->has_data() || !is_fp8_dtype(output->dtype()) ||
      is_fp8_dtype(gated_input.dtype()) || rows == 0 || (transpose && rows % 16 != 0) ||
      !use_tdm(rows, cols,
               {gated_input.data.dptr, IS_BWD ? grad.data.dptr : nullptr, output->data.dptr,
                output->columnwise_data.dptr})) {
    return false;
  }
  if (transpose) {
    NVTE_CHECK(output->data.dtype == output->columnwise_data.dtype,
               "Cast and transposed output types must match.");
  }
  float *const amax = reinterpret_cast<float *>(output->amax.dptr);
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  float *const amax_slots = amax != nullptr ? rocm_amax_slots(grid.x * grid.y, stream) : nullptr;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              transpose, TRANSPOSE,
              quantize_gated_fp8_tdm_kernel<IS_BWD, ParamOP, ActOP, DActOP, TRANSPOSE, IType, OType>
              <<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                  IS_BWD ? reinterpret_cast<const IType *>(grad.data.dptr) : nullptr,
                  reinterpret_cast<const IType *>(gated_input.data.dptr),
                  reinterpret_cast<OType *>(output->data.dptr),
                  reinterpret_cast<OType *>(output->columnwise_data.dptr),
                  reinterpret_cast<const float *>(output->scale.dptr), amax, amax_slots,
                  reinterpret_cast<float *>(output->scale_inv.dptr), rows, cols, p);)));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  if (amax_slots != nullptr) {
    rocm_amax_finalize(amax_slots, amax, nullptr, stream);
  }
  return true;
}

// FP8 tensor-scaling dequantize of input into output.
template <typename = void>
bool dequantize_tdm(const Tensor &input, Tensor *output, hipStream_t stream) {
  using namespace tdm_kernel;
  const auto [rows, cols] = input.flat_2d_dims();
  if (!use_tdm(rows, cols, {input.data.dptr, output->data.dptr})) {
    return false;
  }
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,
          dequantize_fp8_tdm_kernel<IType, OType><<<grid, THREADS_PER_CHUNK, 0, stream>>>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const float *>(input.scale_inv.dptr), rows, cols);));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

// FP8 block scaling of the rows x cols input with BLOCK_DIM 1 (1x128 and 128x1 tiles) or 2 (128x128
// tiles) into output, with rowwise, and into its transpose output_t, with return_transpose. Takes
// the scale-inverse strides of quantize_transpose_{vector,square}_blockwise.
template <int BLOCK_DIM>
bool quantize_blockwise_tdm(const SimpleTensor &input, const SimpleTensor &output,
                            const SimpleTensor &output_t, const SimpleTensor &scale_inv,
                            const SimpleTensor &scale_inv_t, const size_t rows, const size_t cols,
                            const bool rowwise, const bool return_transpose,
                            const size_t scale_stride_x, const size_t scale_stride_y,
                            const size_t scale_t_stride_x, const size_t scale_t_stride_y,
                            const float epsilon, const bool pow_2_scale, const float *noop,
                            hipStream_t stream) {
  using namespace tdm_kernel;
  // The transpose's rows are rows long.
  if (is_fp8_dtype(input.dtype) || rows == 0 || cols == 0 ||
      (return_transpose && rows % 16 != 0) ||
      !use_tdm(rows, cols,
               {input.dptr, rowwise ? output.dptr : nullptr,
                return_transpose ? output_t.dptr : nullptr})) {
    return false;
  }
  const DType out_dtype = rowwise ? output.dtype : output_t.dtype;
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, CHUNK_DIM_Y));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          out_dtype, OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, TRANSPOSE,
              quantize_block_fp8_tdm_kernel<BLOCK_DIM, TRANSPOSE, IType, OType>
              <<<grid, THREADS_PER_CHUNK, 0, stream>>>(
                  reinterpret_cast<const IType *>(input.dptr),
                  rowwise ? reinterpret_cast<OType *>(output.dptr) : nullptr,
                  reinterpret_cast<OType *>(output_t.dptr),
                  reinterpret_cast<float *>(scale_inv.dptr),
                  reinterpret_cast<float *>(scale_inv_t.dptr), rows, cols, scale_stride_x,
                  scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon, pow_2_scale,
                  noop);)));  // NOLINT(*)
  NVTE_CHECK_CUDA(hipGetLastError());
  return true;
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine
