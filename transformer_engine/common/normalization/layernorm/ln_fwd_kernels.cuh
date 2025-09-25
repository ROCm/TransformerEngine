/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_
#define TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_

#include <cfloat>
#include <cstdio>

#include "../../utils.cuh"
#include "../common.h"

namespace transformer_engine {
namespace normalization {
using namespace transformer_engine;

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_fwd_tuned_kernel(
    ForwardKernelParams params) {
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::NUM_ELTS };
  enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

  using output_t = typename Ktraits::output_t;
  using index_t = typename Ktraits::index_t;
  using compute_t = typename Ktraits::compute_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  #ifdef __HIP_PLATFORM_AMD__
  using Cvec = typename Ktraits::Cvec;
  #endif

  using Stats = typename Ktraits::Stats;
  using stats_t = typename Stats::stats_t;

  extern __shared__ char smem_[];

  const index_t tidx = threadIdx.x;
  const index_t bidn = blockIdx.x % CTAS_PER_ROW;
  const index_t bidm = blockIdx.x / CTAS_PER_ROW;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t r = bidm * ROWS_PER_CTA + warp_m;
  const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

  Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

  compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
  compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

  Wvec gamma[LDGS];
  Wvec beta[LDGS];
  index_t idx = c;
#pragma unroll
  for (int it = 0; it < LDGS; ++it) {
    gamma[it].load_from(params.gamma, idx);
    beta[it].load_from(params.beta, idx);
    idx += VEC_COLS_PER_LDG;
  }

  constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

  compute_t scale = 1.f;
  if (params.fp8_out) {
    scale = *reinterpret_cast<compute_t *>(params.scale);
  }
  compute_t amax = 0;

  for (int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA) {
    Ivec x[LDGS];
    index_t idx = row * Ktraits::VEC_COLS + c;
    compute_t xf[LDGS * NUM_ELTS];
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      x[it].load_from(params.x, idx);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_ij = compute_t(x[it].data.elt[jt]);
        xf[it * NUM_ELTS + jt] = x_ij;
      }
      idx += VEC_COLS_PER_LDG;
    }

    stats_t s = stats.compute(xf, rn);

    compute_t mu = Get<0>::of<stats_t, compute_t>(s);
    compute_t m2 = Get<1>::of<stats_t, compute_t>(s);

    if (bidn == 0 && warp_n == 0 && lane == 0) {
      mu_ptr[row] = mu;
    }

    compute_t rs = rsqrtf(rn * m2 + params.epsilon);

    if (bidn == 0 && warp_n == 0 && lane == 0) {
      rs_ptr[row] = rs;
    }

#ifdef __HIP_PLATFORM_AMD__
  if (params.mxfp8_out) {
    Cvec z[LDGS];
    idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t y_ij = rs * (xf[it * NUM_ELTS + jt] - mu);
        compute_t g_ij = gamma[it].data.elt[jt];
        if (params.zero_centered_gamma) {
          g_ij += 1;
        }
        compute_t b_ij = beta[it].data.elt[jt];
        compute_t temp_output = g_ij * y_ij + b_ij;

        z[it].data.elt[jt] = output_t(temp_output);
      }
      z[it].store_to(params.z, idx);
      idx += VEC_COLS_PER_LDG;
    }
  } else {
#endif
    Ovec z[LDGS];
    idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t y_ij = rs * (xf[it * NUM_ELTS + jt] - mu);
        compute_t g_ij = gamma[it].data.elt[jt];
        if (params.zero_centered_gamma) {
          g_ij += 1;
        }
        compute_t b_ij = beta[it].data.elt[jt];
        compute_t temp_output = g_ij * y_ij + b_ij;

        if (params.fp8_out) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(temp_output));
          temp_output = temp_output * scale;
        }

        z[it].data.elt[jt] = output_t(temp_output);
      }
      z[it].store_to(params.z, idx);
      idx += VEC_COLS_PER_LDG;
    }
#ifdef __HIP_PLATFORM_AMD__
  }
#endif
  }
  if (params.fp8_out) {
    // Reduce amax over block
    if (params.amax != nullptr) {
      amax = reduce_max<WARPS_M * WARPS_N>(amax, warp);
      if (threadIdx.x == 0) {
        static_assert(std::is_same<compute_t, float>::value);
        atomicMaxFloat(reinterpret_cast<compute_t *>(params.amax), amax);
      }
    }

    // Update scale-inverse
    if (blockIdx.x == 0 && threadIdx.x == 0 && params.scale_inv != nullptr) {
      reciprocal<compute_t>(reinterpret_cast<compute_t *>(params.scale_inv), scale);
    }
  }
}

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_fwd_general_kernel(
    ForwardKernelParams params) {
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::NUM_ELTS };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { WARPS_N = Ktraits::WARPS_N };

  using input_t = typename Ktraits::input_t;
  using weight_t = typename Ktraits::weight_t;
  using output_t = typename Ktraits::output_t;
  using index_t = typename Ktraits::index_t;
  using compute_t = typename Ktraits::compute_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;

  const index_t tidx = threadIdx.x;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t bdimm = WARPS_M;
  const index_t bdimn = WARPS_N * THREADS_PER_WARP;
  const index_t bidm = blockIdx.x / params.ctas_per_row;
  const index_t bidn = blockIdx.x % params.ctas_per_row;

  const index_t gdimm = bdimm * params.ctas_per_col;
  const index_t gdimn = bdimn * params.ctas_per_row;
  const index_t gidm = bidm * bdimm + warp_m;
  const index_t gidn = (bidn * THREADS_PER_WARP + warp_n * params.ctas_per_row * THREADS_PER_WARP +
                        lane);  // Order threads by warp x cta x lane

  using Stats = typename Ktraits::Stats;
  using stats_t = typename Stats::stats_t;
  extern __shared__ char smem[];
  Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem);

  compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
  compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

  compute_t scale = params.fp8_out ? *reinterpret_cast<compute_t *>(params.scale) : 1.f;
  compute_t amax = 0;

  for (int cta_row = bidm * bdimm; cta_row < params.rows; cta_row += gdimm) {
    int row = cta_row + warp_m;
    if (row >= params.rows) continue;

    compute_t mu = 0.f, m2 = 0.f;
    int count = 0;

    // Step 1: mean and m2
#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS;
         ++it, col += gdimn * NUM_ELTS) {
      Ivec x_vec;
      x_vec.load_from_elts(params.x, row * params.cols + col, params.cols - col);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; ++jt) {
        if (col + jt < params.cols) {
          compute_t x = compute_t(x_vec.data.elt[jt]);
          count += 1;
          compute_t delta = x - mu;
          mu += delta / count;
          m2 += delta * (x - mu);
        }
      }
    }

    Vec3<compute_t,int> stat = stats.reduce(Vec3<compute_t,int>(mu, m2, count));
    mu = stat.x;
    m2 = stat.y;
    compute_t rs = rsqrtf((m2 / stat.z) + params.epsilon);

    if (gidn == 0) {
      mu_ptr[row] = mu;
      rs_ptr[row] = rs;
    }

    // Step 2: store output (no need to store xf[])
#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && col < params.cols;
         ++it, col += gdimn * NUM_ELTS) {
      Ivec x_vec;
      x_vec.load_from_elts(params.x, row * params.cols + col, params.cols - col);
      Wvec g_raw, b_raw;
      g_raw.load_from_elts(params.gamma, col, params.cols - col);
      b_raw.load_from_elts(params.beta,  col, params.cols - col);

      Cvec z;
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; ++jt) {
        if (col + jt < params.cols) {
          compute_t x = compute_t(x_vec.data.elt[jt]);
          compute_t norm = rs * (x - mu);
          compute_t g = compute_t(g_raw.data.elt[jt]) + (params.zero_centered_gamma ? 1.f : 0.f);
          compute_t b = compute_t(b_raw.data.elt[jt]);
          compute_t val = g * norm + b;
          if (params.fp8_out) {
            amax = fmaxf(amax, fabsf(val));
            val *= scale;
          }
          z.data.elt[jt] = output_t(val);
        }
      }

      // Store output
#ifdef __HIP_PLATFORM_AMD__
      if (params.mxfp8_out) {
        z.store_to_elts(params.z, row * params.cols + col, params.cols - col);
      } else {
#endif
      Ovec z_out;
      z.to(z_out);
      z_out.store_to_elts(params.z, row * params.cols + col, params.cols - col);
#ifdef __HIP_PLATFORM_AMD__
      }
#endif
    }
  }

  // Finalize fp8 factors
  if (params.fp8_out) {
    // Reduce amax over block
    if (params.amax != nullptr) {
      amax = reduce_max<WARPS_M * WARPS_N>(amax, warp);
      if (threadIdx.x == 0) {
        static_assert(std::is_same<compute_t, float>::value);
        atomicMaxFloat(reinterpret_cast<compute_t *>(params.amax), amax);
      }
    }

    // Update scale-inverse
    if (blockIdx.x == 0 && threadIdx.x == 0 && params.scale_inv != nullptr) {
      reciprocal<compute_t>(reinterpret_cast<compute_t *>(params.scale_inv), scale);
    }
  }
}

}  // namespace normalization
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_
