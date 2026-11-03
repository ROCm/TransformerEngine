/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// LayerNorm and RMSNorm forward with rows split across the waves of a block, as
// rocm_norm_common.cuh describes: norm_fwd_kernel for 16- and 32-bit and FP8 tensor-scaled outputs
// and, after its statistics, norm_mxfp8_kernel for MXFP8 and norm_fp8_transpose_kernel for FP8
// tensor-scaled outputs with their transpose. This expects its includer to have included
// normalization/common.h, which brings the MXFP8 TDM quantize.

#include "../cast/fp8/rocm_quantize_fp8_tdm.cuh"
#include "rocm_norm_common.cuh"

namespace transformer_engine {
namespace normalization {
namespace rocm {

namespace mxfp8_tdm = dispatch::mxfp8::tdm_kernel;
namespace fp8_tdm = dispatch::fp8::tdm_kernel;

#ifdef __gfx1250__
// The mean, 0 for RMSNorm, and the rsigma of a row whose part xv the wave holds, with all waves of
// the block taking part as in row_sum. Zeros stand in for the elements outside the row. Explicit
// FMAs keep the results independent of the compiler's contractions, so all kernels agree.
template <bool RMS, typename IType, uint32_t VEC, int MAX_WAVES_PER_BLOCK>
__device__ __forceinline__ float2 row_stats(const Vec<IType, VEC> (&xv)[MAX_VECS_PER_LANE],
                                            const RowPart<IType> &part, const float epsilon,
                                            float2 (&stats_sh)[2][MAX_WAVES_PER_BLOCK],
                                            const int parity, const int wave,
                                            const int waves_per_row) {
  if constexpr (RMS) {
    // Separate products and sums, which schedule better than a chain of FMAs
#pragma clang fp contract(off)
    float sumsq = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        const float xj = static_cast<float>(xv[k].data.elt[j]);
        sumsq += xj * xj;
      }
    }
    sumsq = wave_sum(sumsq);
    sumsq = row_sum({sumsq, 0.0f}, stats_sh, parity, wave, waves_per_row).x;
    return {0.0f, rsqrtf(fmaf(part.rn, sumsq, epsilon))};
  } else {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        sum += static_cast<float>(xv[k].data.elt[j]);
      }
    }
    const float part_mean = wave_sum(sum) * part.part_rn;
    float m2 = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
      if (k < part.vecs && (part.valid & (1u << k))) {
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          const float d = static_cast<float>(xv[k].data.elt[j]) - part_mean;
          m2 = fmaf(d, d, m2);
        }
      }
    }
    m2 = wave_sum(m2);
    const float2 stats = row_mean_m2({part_mean, m2}, stats_sh, parity, wave, waves_per_row,
                                     part.part_cols, part.cols, part.rn);
    return {stats.x, rsqrtf(fmaf(part.rn, stats.y, epsilon))};
  }
}
#endif  // __gfx1250__

// LayerNorm, or RMSNorm with RMS, of the rows of params.x into params.z, with the statistics of row
// r in params.mu[r] (LayerNorm) and params.rs[r]. Like ln_fwd_tuned_kernel, it raises the amax,
// here at its rocm_amax_target, and sets the scale inverse of FP8 outputs. Persistent blocks step
// through rows_per_step rows at a time, each wave staging its slices of the next rows while it
// normalizes the current one. STATS_ONLY only writes the statistics.
template <bool RMS, typename IType, typename WType, typename OType, int MAX_WAVES_PER_BLOCK,
          bool STATS_ONLY = false>
__global__ void __launch_bounds__(MAX_WAVES_PER_BLOCK * WAVE_SIZE)
    norm_fwd_kernel(const ForwardKernelParams params, const RowSplit split,
                    float *const amax_slots) {
#ifdef __gfx1250__
  constexpr int VEC = VEC_BYTES / sizeof(IType);
  extern __shared__ __attribute__((aligned(128))) char slices_smem[];
  __shared__ float2 stats_sh[2][MAX_WAVES_PER_BLOCK];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const RowPart<IType> part(split, params.cols, wave, lane);
  const uint64_t rows = params.rows;
  const uint32_t cols = params.cols;
  slice_t<IType> *const slices = wave_slices<IType>(slices_smem, wave);
  const IType *const x = static_cast<const IType *>(params.x);

  Vec<WType, VEC> gamma[MAX_VECS_PER_LANE];
  Vec<WType, VEC> beta[MAX_VECS_PER_LANE];
#pragma unroll
  for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
    gamma[k].clear();
    beta[k].clear();
    if (!STATS_ONLY && (part.valid & (1u << k))) {
      gamma[k].load_from(static_cast<const WType *>(params.gamma) + part.col(k, lane));
      if constexpr (!RMS) {
        beta[k].load_from(static_cast<const WType *>(params.beta) + part.col(k, lane));
      }
    }
  }

  const float scale = is_fp8<OType>::value ? *static_cast<const float *>(params.scale) : 1.0f;
  const bool requires_amax = !STATS_ONLY && params.amax != nullptr;
  float amax = 0.0f;

  const uint64_t step = static_cast<uint64_t>(gridDim.x) * split.rows_per_step;
  const uint64_t row_offset = wave / split.waves_per_row;
  const auto stage_row = [&](slice_t<IType> &slice, const uint64_t row) {
    const uint32_t staged_row = static_cast<uint32_t>(row < rows ? row : rows);
    stage_slice(slice, x, rows, part.col_end, cols, staged_row, part.col_begin);
  };
  uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * split.rows_per_step;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    stage_row(slices[s], row0 + s * step + row_offset);
  }
  for (int it = 0; row0 < rows; row0 += step, it++) {
    const uint64_t row = row0 + row_offset;
    // Rows past the end stage as zeros, keeping the count of slices in flight fixed.
    stage_row(slices[(it + STAGES - 1) % STAGES], row + (STAGES - 1) * step);
    wait_staged<STAGES - 1>();
    Vec<IType, VEC> xv[MAX_VECS_PER_LANE];
    load_part(xv, slices[it % STAGES], lane);
    const float2 stats =
        row_stats<RMS>(xv, part, params.epsilon, stats_sh, it % 2, wave, split.waves_per_row);
    if (row >= rows) {
      continue;
    }
    if (part.index == 0 && lane == 0) {
      if constexpr (!RMS) {
        static_cast<float *>(params.mu)[row] = stats.x;
      }
      static_cast<float *>(params.rs)[row] = stats.y;
    }
    if constexpr (STATS_ONLY) {
      continue;
    }

    OType *const z_row = static_cast<OType *>(params.z) + row * cols;
#pragma unroll
    for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
      opaque(gamma[k]);
      if constexpr (!RMS) {
        opaque(beta[k]);
      }
      if (k < part.vecs && (part.valid & (1u << k))) {
        float out[VEC];
        normalize<RMS>(xv[k], gamma[k], beta[k], stats.x, stats.y, params.zero_centered_gamma,
                       out);
        store_out(z_row + part.col(k, lane), out, scale, requires_amax, amax);
      }
    }
  }
  // The block's LDS must outlive the slices staged past the end.
  wait_staged<0>();

  if (requires_amax) {
    block_raise_amax<MAX_WAVES_PER_BLOCK>(out_amax<OType>(amax), static_cast<float *>(params.amax),
                                          amax_slots);
  }
  if constexpr (is_fp8<OType>::value && !STATS_ONLY) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && params.scale_inv != nullptr) {
      reciprocal<float>(static_cast<float *>(params.scale_inv), scale);
    }
  }
#endif  // __gfx1250__
}

// The MXFP8 output of a normalization, as Tensor lays it out.
struct Mxfp8Out {
  void *rowwise;
  void *colwise;
  e8m0_t *scales_rowwise;
  e8m0_t *scales_colwise;
  size_t scale_stride_rowwise;
  size_t scale_stride_colwise;
};

// LayerNorm, or RMSNorm with RMS, of the 128x128 chunk (blockIdx.y, blockIdx.x) of params.x into
// the MXFP8 out, with the statistics norm_fwd_kernel left in params.mu and params.rs. Each wave
// normalizes its band of the chunk in LDS, rounded to IType, and quantizes it with the band
// functions of the MXFP8 TDM quantize, so the output is that quantize's of the IType output of
// norm_fwd_kernel.
template <bool RMS, typename IType, typename WType, typename OType, bool COLWISE>
__global__ void __launch_bounds__(mxfp8_tdm::THREADS_PER_CHUNK)
    norm_mxfp8_kernel(const ForwardKernelParams params, const Mxfp8Out out) {
#ifdef __gfx1250__
  using mxfp8_tdm::band_t;
  constexpr int BAND_Y = mxfp8_tdm::BAND_Y;
  constexpr int LANE_COLS = mxfp8_tdm::COLS_PER_LANE;
  constexpr int SCALE_DIM = 32;
  __shared__ alignas(128) band_t<IType> in_sh[mxfp8_tdm::WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[mxfp8_tdm::WAVES][1 + COLWISE];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const uint64_t rows = params.rows;
  const uint32_t cols = params.cols;
  const uint64_t band_row0 = static_cast<uint64_t>(blockIdx.y) * mxfp8_tdm::CHUNK_DIM_Y +
                             wave * BAND_Y;
  const uint32_t tile_col = blockIdx.x * mxfp8_tdm::CHUNK_DIM_X;
  if (band_row0 >= rows) {
    return;
  }
  auto &tile = in_sh[wave];
  tdm::load_2d(tile, static_cast<const IType *>(params.x), rows, cols, cols, band_row0, tile_col);

  // Lane y holds the statistics of row y, the lane's columns their gamma and beta.
  const uint64_t valid_rows = rows - band_row0 < BAND_Y ? rows - band_row0 : BAND_Y;
  float mu = 0.0f;
  float rs = 0.0f;
  if (lane < valid_rows) {
    if constexpr (!RMS) {
      mu = static_cast<const float *>(params.mu)[band_row0 + lane];
    }
    rs = static_cast<const float *>(params.rs)[band_row0 + lane];
  }
  const uint32_t col = tile_col + lane * LANE_COLS;
  const bool col_valid = col < cols;
  Vec<WType, LANE_COLS> g;
  Vec<WType, LANE_COLS> b;
  g.clear();
  b.clear();
  if (col_valid) {
    g.load_from(static_cast<const WType *>(params.gamma) + col);
    if constexpr (!RMS) {
      b.load_from(static_cast<const WType *>(params.beta) + col);
    }
  }
  tdm::wait<0>();

  // Elements outside the tensor stay zero, as the quantize expects.
#pragma unroll
  for (int y = 0; y < BAND_Y; y++) {
    Vec<IType, LANE_COLS> v;
    v.load_from(&tile[y][lane * LANE_COLS]);
    float o[LANE_COLS];
    normalize<RMS>(v, g, b, __int_as_float(__builtin_amdgcn_readlane(__float_as_int(mu), y)),
                   __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rs), y)),
                   params.zero_centered_gamma, o);
    const bool valid = col_valid && y < valid_rows;
#pragma unroll
    for (int j = 0; j < LANE_COLS; j++) {
      v.data.elt[j] = cvt_from_float<IType>(valid ? o[j] : 0.0f);
    }
    v.store_to(&tile[y][lane * LANE_COLS]);
  }
  const size_t scale_col = tile_col / SCALE_DIM;
  mxfp8_tdm::quantize_band_rowwise<IType, OType, SCALE_DIM>(
      tile, out_sh[wave][0], out.scales_rowwise + band_row0 * out.scale_stride_rowwise + scale_col,
      out.scale_stride_rowwise, valid_rows, out.scale_stride_rowwise - scale_col);
  tdm::store_2d(static_cast<OType *>(out.rowwise), out_sh[wave][0], rows, cols, cols, band_row0,
                tile_col);
  if constexpr (COLWISE) {
    mxfp8_tdm::quantize_band_colwise<IType, OType>(
        tile, out_sh[wave][COLWISE],
        out.scales_colwise + band_row0 / SCALE_DIM * out.scale_stride_colwise + tile_col,
        cols - tile_col);
    tdm::store_2d(static_cast<OType *>(out.colwise), out_sh[wave][COLWISE], rows, cols, cols,
                  band_row0, tile_col);
  }
  // The block's LDS must outlive the stores.
  tdm::wait<0>();
#endif  // __gfx1250__
}

// LayerNorm, or RMSNorm with RMS, of the 128x128 chunk (blockIdx.y, blockIdx.x) of params.x into
// the FP8 params.z and its cols x rows transpose z_t, with the statistics norm_fwd_kernel left in
// params.mu and params.rs. The outputs and the amax are those of norm_fwd_kernel. As in the FP8
// TDM quantize, each wave normalizes its band of the chunk and the waves assemble the chunk's
// transpose in LDS, which stays at the quantize's 64 KB for 16-bit inputs.
template <bool RMS, typename IType, typename WType, typename OType>
__global__ void __launch_bounds__(fp8_tdm::THREADS_PER_CHUNK)
    norm_fp8_transpose_kernel(const ForwardKernelParams params, void *const z_t,
                              float *const amax_slots) {
#ifdef __gfx1250__
  using fp8_tdm::band_t;
  constexpr int WAVES = fp8_tdm::WAVES;
  constexpr int BAND_Y = fp8_tdm::BAND_Y;
  constexpr int LANE_COLS = fp8_tdm::COLS_PER_LANE;
  constexpr int QUADS = fp8_tdm::QUADS;
  // DPP8 selector: lane k of each group of 8 reads lane (k + 1) % 8.
  constexpr unsigned NEXT_LANE = 1 | 2 << 3 | 3 << 6 | 4 << 9 | 5 << 12 | 6 << 15 | 7 << 18;
  static_assert(QUADS == 8, "A group of DPP8 lanes holds the statistics of a band");
  __shared__ alignas(128) band_t<IType> in_sh[WAVES];
  __shared__ alignas(128) band_t<OType> out_sh[WAVES];
  // Wave w stores rows [w * BAND_Y, (w + 1) * BAND_Y) of the transposed chunk.
  __shared__ alignas(128) OType t_sh[WAVES][BAND_Y][fp8_tdm::CHUNK_DIM_Y];

  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
  const int lane = threadIdx.x % WAVE_SIZE;
  const int lane_col = lane * LANE_COLS;
  const uint64_t rows = params.rows;
  const uint32_t cols = params.cols;
  const uint64_t chunk_row = static_cast<uint64_t>(blockIdx.y) * fp8_tdm::CHUNK_DIM_Y;
  const uint32_t chunk_col = blockIdx.x * fp8_tdm::CHUNK_DIM_X;
  const uint64_t band_row0 = chunk_row + wave * BAND_Y;
  const float scale = *static_cast<const float *>(params.scale);
  float amax = 0.0f;

  if (band_row0 < rows) {
    auto &tile = in_sh[wave];
    tdm::load_2d(tile, static_cast<const IType *>(params.x), rows, cols, cols, band_row0,
                 chunk_col);

    // As in the quantize, the lanes start at different quads of rows to spread their transposed
    // writes over the banks. A lane holds the statistics of its quad's rows and takes those of its
    // next quad from the next lane of its group of QUADS lanes. Rows past the tensor's take the
    // statistics of its last row; their outputs are masked.
    const uint64_t valid_rows = rows - band_row0 < BAND_Y ? rows - band_row0 : BAND_Y;
    const int first_quad = lane % QUADS;
    float mu[4] = {};
    float rs[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint64_t y = first_quad * 4 + i < valid_rows ? first_quad * 4 + i : valid_rows - 1;
      if constexpr (!RMS) {
        mu[i] = static_cast<const float *>(params.mu)[band_row0 + y];
      }
      rs[i] = static_cast<const float *>(params.rs)[band_row0 + y];
    }
    // Gamma, plus 1 when zero-centered, and beta as normalize reads them, in FP32 to keep the
    // zero-centered branch out of the loop over rows.
    const uint32_t col = chunk_col + lane_col;
    const bool col_valid = col < cols;
    Vec<float, LANE_COLS> g;
    Vec<float, LANE_COLS> b;
    g.clear();
    b.clear();
    if (col_valid) {
      Vec<WType, LANE_COLS> w;
      w.load_from(static_cast<const WType *>(params.gamma) + col);
#pragma unroll
      for (int j = 0; j < LANE_COLS; j++) {
        const float gj = static_cast<float>(w.data.elt[j]);
        g.data.elt[j] = params.zero_centered_gamma ? gj + 1.0f : gj;
      }
      if constexpr (!RMS) {
        w.load_from(static_cast<const WType *>(params.beta) + col);
#pragma unroll
        for (int j = 0; j < LANE_COLS; j++) {
          b.data.elt[j] = static_cast<float>(w.data.elt[j]);
        }
      }
    }
    tdm::wait<0>();

#pragma unroll
    for (int step = 0; step < QUADS; step++) {
      const int quad = (first_quad + step) % QUADS;
      uint32_t packed[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int y = quad * 4 + i;
        Vec<IType, LANE_COLS> v;
        v.load_from(&tile[y][lane_col]);
        float o[LANE_COLS];
        normalize<RMS>(v, g, b, mu[i], rs[i], false, o);
        // Elements outside the tensor leave the amax unchanged.
        const bool valid = col_valid && y < valid_rows;
#pragma unroll
        for (int j = 0; j < LANE_COLS; j++) {
          o[j] = valid ? o[j] : 0.0f;
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(o[j]));
        }
        packed[i] = cvt_4x_to_fp8<OType, false>(o[0] * scale, o[1] * scale, o[2] * scale,
                                                o[3] * scale);
        *reinterpret_cast<uint32_t *>(&out_sh[wave][y][lane_col]) = packed[i];
      }
      fp8_tdm::store_transposed_quad(t_sh, packed, lane_col, wave, quad);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if constexpr (!RMS) {
          mu[i] = __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(mu[i]), NEXT_LANE));
        }
        rs[i] = __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(rs[i]), NEXT_LANE));
      }
    }
    tdm::store_2d(static_cast<OType *>(params.z), out_sh[wave], rows, cols, cols, band_row0,
                  chunk_col);
  }

  // Every wave leaves its amax in its consumed in_sh band.
  asm volatile("" ::: "memory");  // in_sh is reused as float
  float *const wave_scratch = reinterpret_cast<float *>(in_sh[wave]);
  if (params.amax != nullptr) {
    amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
    if (lane == 0) {
      *wave_scratch = amax;
    }
  }
  __syncthreads();
  // Bands past the tensor's rows leave garbage in t_sh, but outside the transpose's columns.
  const uint32_t t_row = chunk_col + wave * BAND_Y;
  if (t_row < cols) {
    tdm::store_2d(static_cast<OType *>(z_t), t_sh[wave], cols, rows, rows, t_row, chunk_row);
  }
  if (params.amax != nullptr && threadIdx.x == 0) {
    float block_amax = 0.0f;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      block_amax = fmaxf(block_amax, *reinterpret_cast<const float *>(in_sh[w]));
    }
    float *const amax_dst = rocm_amax_target(static_cast<float *>(params.amax), amax_slots,
                                             blockIdx.y * gridDim.x + blockIdx.x);
    // As in the quantize, only atomics that may raise the amax are issued.
    if (block_amax > *amax_dst) {
      rocm_atomicMaxFloat(amax_dst, block_amax);
    }
  }
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && params.scale_inv != nullptr) {
    reciprocal<float>(static_cast<float *>(params.scale_inv), scale);
  }
  // The block's LDS must outlive the stores.
  tdm::wait<0>();
#endif  // __gfx1250__
}

}  // namespace rocm
}  // namespace normalization
}  // namespace transformer_engine
