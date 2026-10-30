/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// LayerNorm and RMSNorm backward with rows split across the waves of a block, as
// rocm_norm_common.cuh describes. Parts are at most BWD_VECS_PER_LANE vectors per lane, which
// leaves registers for the lane's dgamma and dbeta sums over the rows the wave processes; each
// block writes the sums of its waves to its row of the partials, which a finalize step reduces.

#include "rocm_norm_common.cuh"

namespace transformer_engine {
namespace normalization {
namespace rocm {

constexpr int BWD_VECS_PER_LANE = 4;
constexpr int BWD_SLICE_BYTES = WAVE_SIZE * BWD_VECS_PER_LANE * VEC_BYTES;  // 2 KB

// The dynamic LDS of the backward kernel's blocks: the slices of x, dz and, with add, add.
inline size_t bwd_block_slice_bytes(const RowSplit &split, const bool add) {
  return static_cast<size_t>(block_waves(split)) * STAGES * (add ? 3 : 2) * BWD_SLICE_BYTES;
}

// The gradients of LayerNorm, or RMSNorm with RMS, for the rows of params.x with the statistics
// params.mu (LayerNorm) and params.rs: params.dx, plus params.add with ADD, and the block's dgamma
// and dbeta (LayerNorm) partials in row blockIdx.x of params.dgamma_part and params.dbeta_part.
// dz, gamma, add and dx are of the input type. Persistent blocks step through the rows as
// norm_fwd_kernel does, staging x, dz and add.
template <bool RMS, typename IType, int MAX_WAVES_PER_BLOCK, bool ADD>
__global__ void __launch_bounds__(MAX_WAVES_PER_BLOCK * WAVE_SIZE)
    norm_bwd_kernel(const BackwardKernelParams params, const RowSplit split) {
#ifdef __gfx1250__
  constexpr int VEC = VEC_BYTES / sizeof(IType);
  constexpr int K = BWD_VECS_PER_LANE;
  constexpr int TENSORS = ADD ? 3 : 2;
  using bwd_slice_t = slice_t<IType, BWD_SLICE_BYTES>;
  extern __shared__ __attribute__((aligned(128))) char slices_smem[];
  __shared__ float2 sums_sh[2][MAX_WAVES_PER_BLOCK];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const RowPart<IType> part(split, params.cols, wave, lane);
  const uint64_t rows = params.rows;
  const uint32_t cols = params.cols;
  // slices[s][t]: stage s of tensor t, which is x, dz or add
  bwd_slice_t(*const slices)[TENSORS] =
      reinterpret_cast<bwd_slice_t(*)[TENSORS]>(slices_smem) + wave * STAGES;
  const IType *const inputs[3] = {static_cast<const IType *>(params.x),
                                  static_cast<const IType *>(params.dz),
                                  static_cast<const IType *>(params.add)};
  const float *const mu = static_cast<const float *>(params.mu);
  const float *const rs = static_cast<const float *>(params.rs);

  Vec<IType, VEC> gamma[K];
  float dgamma[K][VEC] = {};
  float dbeta[K][VEC] = {};
#pragma unroll
  for (int k = 0; k < K; k++) {
    gamma[k].clear();
    if (part.valid & (1u << k)) {
      gamma[k].load_from(static_cast<const IType *>(params.gamma) + part.col(k, lane));
    }
  }
  const float shift = params.zero_centered_gamma ? 1.0f : 0.0f;

  const uint64_t step = static_cast<uint64_t>(gridDim.x) * split.rows_per_step;
  const uint64_t row_offset = wave / split.waves_per_row;
  const auto stage_row = [&](bwd_slice_t(&stage)[TENSORS], const uint64_t row) {
    const uint32_t staged_row = static_cast<uint32_t>(row < rows ? row : rows);
#pragma unroll
    for (int t = 0; t < TENSORS; t++) {
      stage_slice(stage[t], inputs[t], rows, part.col_end, cols, staged_row, part.col_begin);
    }
  };
  // Rows past the end have zero statistics as well as zero inputs, so they add nothing.
  const auto row_stats = [&](const uint64_t row) -> float2 {
    if (row >= rows) {
      return {0.0f, 0.0f};
    }
    return {RMS ? 0.0f : mu[row], rs[row]};
  };
  uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * split.rows_per_step;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    stage_row(slices[s], row0 + s * step + row_offset);
  }
  // A step ahead, like the slices
  float2 next_stats = row_stats(row0 + row_offset);
  for (int it = 0; row0 < rows; row0 += step, it++) {
    const uint64_t row = row0 + row_offset;
    const float2 stats = next_stats;
    next_stats = row_stats(row + step);
    stage_row(slices[(it + STAGES - 1) % STAGES], row + (STAGES - 1) * step);
    wait_staged<(STAGES - 1) * TENSORS>();
    Vec<IType, VEC> xv[K];
    Vec<IType, VEC> dzv[K];
    load_part(xv, slices[it % STAGES][0], lane);
    load_part(dzv, slices[it % STAGES][1], lane);

    // y and dy, recomputed for dx below, as ln_bwd_tuned_kernel defines them
    const auto y_dy = [&](const int k, const int j) -> float2 {
      const float x = static_cast<float>(xv[k].data.elt[j]);
      const float y = RMS ? stats.y * x : stats.y * (x - stats.x);
      return {y, (static_cast<float>(gamma[k].data.elt[j]) + shift) *
                     static_cast<float>(dzv[k].data.elt[j])};
    };
    float2 sums = {0.0f, 0.0f};  // of dy and dy * y
#pragma unroll
    for (int k = 0; k < K; k++) {
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        const float2 v = y_dy(k, j);
        const float dz = static_cast<float>(dzv[k].data.elt[j]);
        if constexpr (!RMS) {
          sums.x += v.y;
          dbeta[k][j] += dz;
        }
        sums.y += v.y * v.x;
        dgamma[k][j] += dz * v.x;
      }
    }
    sums = row_sum({RMS ? 0.0f : wave_sum(sums.x), wave_sum(sums.y)}, sums_sh, it % 2, wave,
                   split.waves_per_row);
    if (row >= rows) {
      continue;
    }
    const float mdy = sums.x * part.rn;
    const float mdyy = sums.y * part.rn;
    Vec<IType, VEC> addv[K];
    if constexpr (ADD) {
      load_part(addv, slices[it % STAGES][2], lane);
    }
    IType *const dx_row = static_cast<IType *>(params.dx) + row * cols;
#pragma unroll
    for (int k = 0; k < K; k++) {
      if (k < part.vecs && (part.valid & (1u << k))) {
        float dx[VEC];
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          const float2 v = y_dy(k, j);
          dx[j] = stats.y * (v.y - (mdyy * v.x + mdy));
          if constexpr (ADD) {
            dx[j] += static_cast<float>(addv[k].data.elt[j]);
          }
        }
        float unused_amax = 0.0f;
        store_out(dx_row + part.col(k, lane), dx, 1.0f, false, unused_amax);
      }
    }
  }
  // The block's LDS must outlive the slices staged past the end.
  wait_staged<0>();

  // The waves of a part sum their partials in the consumed slices, and the one in the first row
  // slot writes them.
  float *const sums_lds = reinterpret_cast<float *>(slices_smem);
  const auto store_partials = [&](const float(&sum)[K][VEC], float *const partial) {
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; k++) {
      Vec<float, VEC> v;
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        v.data.elt[j] = sum[k][j];
      }
      v.store_to(sums_lds + ((wave * K + k) * WAVE_SIZE + lane) * VEC);
    }
    __syncthreads();
    if (row_offset != 0) {
      return;
    }
#pragma unroll
    for (int k = 0; k < K; k++) {
      if (k < part.vecs && (part.valid & (1u << k))) {
        Vec<float, VEC> total;
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          total.data.elt[j] = sum[k][j];
        }
        for (int r = 1; r < split.rows_per_step; r++) {
          Vec<float, VEC> v;
          v.load_from(sums_lds +
                      (((r * split.waves_per_row + wave) * K + k) * WAVE_SIZE + lane) * VEC);
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            total.data.elt[j] += v.data.elt[j];
          }
        }
        total.store_to(partial + static_cast<size_t>(blockIdx.x) * cols + part.col(k, lane));
      }
    }
  };
  store_partials(dgamma, static_cast<float *>(params.dgamma_part));
  if constexpr (!RMS) {
    store_partials(dbeta, static_cast<float *>(params.dbeta_part));
  }
#endif  // __gfx1250__
}

}  // namespace rocm
}  // namespace normalization
}  // namespace transformer_engine
