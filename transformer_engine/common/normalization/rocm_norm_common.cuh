/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Pieces shared by the ROCm normalization kernels. A block splits each row into slices, one per
// wave, so that no row spans blocks: a wave stages its slice in LDS, holds it in registers and
// exchanges only per-row statistics with the other waves of its row. The arch-specific parts are
// the staging, done by the Tensor Data Mover on gfx1250, and the wave32 reductions and BF16 mixed
// FMAs of the device helpers. This expects its includer to have included the common headers,
// util/system.h and util/cuda_runtime.h.

#include <algorithm>

#include "../cast/core/rocm_tdm_common.cuh"
#include "../util/rocm_amax_slots.cuh"
#include "../util/rocm_device_utils.cuh"
#include "../util/rocm_tdm.cuh"

namespace transformer_engine {
namespace normalization {
namespace rocm {

constexpr int WAVE_SIZE = 32;
constexpr int VEC_BYTES = 16;
constexpr int MAX_VECS_PER_LANE = 8;
constexpr int SLICE_BYTES = WAVE_SIZE * MAX_VECS_PER_LANE * VEC_BYTES;  // 4 KB
constexpr int STAGES = 3;  // slices per wave: the one in use and the next ones in flight
// Blocks of 32 waves would leave 128 VGPRs per lane, which 16-bit LayerNorm exceeds.
constexpr int MAX_WAVES = 16;
constexpr int SMALL_BLOCK_WAVES = 8;

// How blocks split cols-wide rows: waves_per_row waves share a row, each holding vecs_per_lane
// 16-byte vectors per lane, and a block processes rows_per_step rows at a time.
struct RowSplit {
  int waves_per_row;
  int vecs_per_lane;
  int rows_per_step;
};

// The split of rows of cols elements of elt_bytes each into slices of at most max_vecs_per_lane
// vectors per lane, with as few waves per row as that allows, but at least min_waves_per_row while
// each wave keeps a vector per lane, and up to SMALL_BLOCK_WAVES waves per block unless a row needs
// more, or a zero split for rows too wide for a block.
inline RowSplit make_row_split(const size_t cols, const size_t elt_bytes,
                               const int min_waves_per_row = 1,
                               const int max_vecs_per_lane = MAX_VECS_PER_LANE) {
  const size_t vecs = cols * elt_bytes / VEC_BYTES;
  size_t waves_per_row = DIVUP(vecs, static_cast<size_t>(WAVE_SIZE * max_vecs_per_lane));
  while (waves_per_row < static_cast<size_t>(min_waves_per_row) &&
         2 * waves_per_row * WAVE_SIZE <= vecs && 2 * waves_per_row <= MAX_WAVES) {
    waves_per_row *= 2;
  }
  if (waves_per_row == 0 || waves_per_row > MAX_WAVES) {
    return {0, 0, 0};
  }
  // As many waves as the vectors per lane need, so that every part has columns
  const size_t vecs_per_lane = DIVUP(vecs, WAVE_SIZE * waves_per_row);
  const int wpr = static_cast<int>(DIVUP(vecs, WAVE_SIZE * vecs_per_lane));
  return {wpr, static_cast<int>(vecs_per_lane), std::max(1, SMALL_BLOCK_WAVES / wpr)};
}

// The blocks of a split's kernels: the waves, their threads and the LDS for their slices.
inline int block_waves(const RowSplit &split) {
  return split.waves_per_row * split.rows_per_step;
}

inline size_t block_slice_bytes(const RowSplit &split) {
  return static_cast<size_t>(block_waves(split)) * STAGES * SLICE_BYTES;
}

#ifdef __gfx1250__
using dispatch::common::cvt_4x_to_fp8;
using dispatch::common::cvt_from_float;

template <typename T, int BYTES = SLICE_BYTES>
using slice_t = T[1][BYTES / sizeof(T)];

// The STAGES slices of wave in a block's dynamic LDS.
template <typename T>
__device__ __forceinline__ slice_t<T> *wave_slices(char *smem, const int wave) {
  return reinterpret_cast<slice_t<T> *>(smem) + wave * STAGES;
}

// Starts staging the elements [col, col_end) of row `row` of the rows x ld row-major x into lds,
// where the rest reads as zero, as does all of a row from `rows` on.
template <typename T, int N>
__device__ __forceinline__ void stage_slice(T (&lds)[1][N], const T *x, const uint32_t rows,
                                            const uint32_t col_end, const uint64_t ld,
                                            const uint32_t row, const uint32_t col) {
  tdm::load_2d(lds, x, rows, col_end, ld, row, col);
}

// Waits until at most N of this wave's slices are still being staged.
template <int N>
__device__ __forceinline__ void wait_staged() {
  tdm::wait<N>();
}

// A wave's part of the rows it processes: its columns and the lane's vectors among them.
template <typename IType>
struct RowPart {
  static constexpr int VEC = VEC_BYTES / sizeof(IType);
  int index;  // among the waves of a row
  int vecs;   // per lane
  uint32_t col_begin;
  uint32_t col_end;
  uint32_t part_cols;
  uint32_t cols;
  uint32_t valid;  // bit k: the lane's vector k lies within the row
  float rn;        // 1 / cols
  float part_rn;   // 1 / the part's columns

  __device__ __forceinline__ RowPart(const RowSplit &split, const uint32_t cols, const int wave,
                                     const int lane)
      : index(wave % split.waves_per_row),
        vecs(split.vecs_per_lane),
        col_begin(index * split.vecs_per_lane * WAVE_SIZE * VEC),
        col_end(min(cols, col_begin + split.vecs_per_lane * WAVE_SIZE * VEC)),
        part_cols(split.vecs_per_lane * WAVE_SIZE * VEC),
        cols(cols),
        valid(0),
        rn(1.0f / cols),
        part_rn(1.0f / (col_end - col_begin)) {
#pragma unroll
    for (int k = 0; k < MAX_VECS_PER_LANE; k++) {
      if (k < vecs && col(k, lane) < col_end) {
        valid |= 1u << k;
      }
    }
  }

  // The first column of the lane's vector k
  __device__ __forceinline__ uint32_t col(const int k, const int lane) const {
    return col_begin + (k * WAVE_SIZE + lane) * VEC;
  }
};

// Loads the lane's K vectors of a staged part. Staging zero-fills the slice past the part, so all
// loads can issue before one wait.
template <typename IType, uint32_t VEC, int K, int N>
__device__ __forceinline__ void load_part(Vec<IType, VEC> (&xv)[K], const IType (&slice)[1][N],
                                          const int lane) {
  static_assert(K * WAVE_SIZE * VEC <= N, "The vectors lie within the slice");
#pragma unroll
  for (int k = 0; k < K; k++) {
    xv[k].load_from(&slice[0][(k * WAVE_SIZE + lane) * VEC]);
  }
}

// Hides v from the optimizer, so that a loop converts loop-invariant values where it uses them
// rather than holding converted copies in registers.
template <typename T>
__device__ __forceinline__ void opaque(T &v) {
  static_assert(sizeof(T) % 4 == 0, "v fills whole registers");
  uint32_t words[sizeof(T) / 4];
  __builtin_memcpy(words, &v, sizeof(T));
#pragma unroll
  for (int i = 0; i < static_cast<int>(sizeof(T) / 4); i++) {
    asm volatile("" : "+v"(words[i]));
  }
  __builtin_memcpy(&v, words, sizeof(T));
}

// v of the lane whose index differs from this lane's in the bits of MASK, within rows of 16 lanes.
template <int MASK>
__device__ __forceinline__ float row_xmask(const float v) {
  constexpr int ROW_XMASK = 0x160;
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                                               ROW_XMASK | MASK, 0xF, 0xF, false));
}

// The sum of v over the wave, wave-uniform: DPP sums within each row of 16 lanes, which are then
// read out, without the LDS round trips of shuffles.
__device__ __forceinline__ float wave_sum(float v) {
  v += row_xmask<1>(v);
  v += row_xmask<2>(v);
  v += row_xmask<4>(v);
  v += row_xmask<8>(v);
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 0)) +
         __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 16));
}

// g * y + b for the BF16 g and b in the low or, with hi, high halves of g_pair and b_pair, rounded
// once like the FP32 FMA of the converted values, which the instruction reads without converting.
__device__ __forceinline__ float fma_bf16(const uint32_t g_pair, const float y,
                                          const uint32_t b_pair, const bool hi) {
  float d;
  if (hi) {
    asm("v_fma_mix_f32_bf16 %0, %1, %2, %3 op_sel:[1,0,1] op_sel_hi:[1,0,1]"
        : "=v"(d)
        : "v"(g_pair), "v"(y), "v"(b_pair));
  } else {
    asm("v_fma_mix_f32_bf16 %0, %1, %2, %3 op_sel_hi:[1,0,1]"
        : "=v"(d)
        : "v"(g_pair), "v"(y), "v"(b_pair));
  }
  return d;
}

// g * y for the BF16 g in the low or, with hi, high half of g_pair, as fma_bf16 reads it. Adding -0
// keeps the product, its sign included.
__device__ __forceinline__ float mul_bf16(const uint32_t g_pair, const float y, const bool hi) {
  float d;
  if (hi) {
    asm("v_fma_mix_f32_bf16 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,0]"
        : "=v"(d)
        : "v"(g_pair), "v"(y), "v"(-0.0f));
  } else {
    asm("v_fma_mix_f32_bf16 %0, %1, %2, %3 op_sel_hi:[1,0,0]"
        : "=v"(d)
        : "v"(g_pair), "v"(y), "v"(-0.0f));
  }
  return d;
}

// The outputs of the N inputs x before conversion, as ln_fwd_tuned_kernel computes them:
// g * (rs * (x - mu)) + b for LayerNorm and g * (rs * x) for RMSNorm, g being gamma, plus 1 when
// zero-centered.
template <bool RMS, typename IType, typename WType, uint32_t N>
__device__ __forceinline__ void normalize(const Vec<IType, N> &x, const Vec<WType, N> &gamma,
                                          const Vec<WType, N> &beta, const float mu,
                                          const float rs, const bool zero_centered,
                                          float (&out)[N]) {
  float y[N];
#pragma unroll
  for (int j = 0; j < N; j++) {
    const float xj = static_cast<float>(x.data.elt[j]);
    y[j] = RMS ? rs * xj : rs * (xj - mu);
  }
  if constexpr (std::is_same_v<WType, bf16>) {
    if (!zero_centered) {
      uint32_t g[N / 2];
      uint32_t b[N / 2];
      __builtin_memcpy(g, &gamma.data, sizeof(g));
      __builtin_memcpy(b, &beta.data, sizeof(b));
#pragma unroll
      for (int j = 0; j < N; j++) {
        out[j] = RMS ? mul_bf16(g[j / 2], y[j], j % 2) : fma_bf16(g[j / 2], y[j], b[j / 2], j % 2);
      }
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
    float g = static_cast<float>(gamma.data.elt[j]);
    if (zero_centered) {
      g += 1.0f;
    }
    out[j] = RMS ? g * y[j] : fmaf(g, y[j], static_cast<float>(beta.data.elt[j]));
  }
}

// a and b as a register of two BF16 or FP16 values, a in the low half, rounded to nearest even.
template <typename OType>
__device__ __forceinline__ uint32_t cvt_2x_to_16bit(const float a, const float b) {
  if constexpr (std::is_same_v<OType, bf16>) {
    typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
    return __builtin_bit_cast(uint32_t, bf16x2{static_cast<__bf16>(a), static_cast<__bf16>(b)});
  } else {
    static_assert(std::is_same_v<OType, fp16>, "16-bit outputs are BF16 or FP16");
    typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
    return __builtin_bit_cast(uint32_t,
                              f16x2{static_cast<_Float16>(a), static_cast<_Float16>(b)});
  }
}

// The elements of a row's part `part` when parts of part_cols elements split cols.
__device__ __forceinline__ uint32_t part_elts(const uint32_t part, const uint32_t part_cols,
                                              const uint32_t cols) {
  const uint32_t begin = min(cols, part * part_cols);
  return min(part_cols, cols - begin);
}

// For the wave-uniform statistics v of the wave's part of a row, the sums over the waves of the
// row. Every wave of the block takes part, with alternating parities, so that sh needs no second
// barrier.
template <int MAX_WAVES_PER_BLOCK>
__device__ __forceinline__ float2 row_sum(const float2 v, float2 (&sh)[2][MAX_WAVES_PER_BLOCK],
                                          const int parity, const int wave,
                                          const int waves_per_row) {
  if (waves_per_row == 1) {
    return v;
  }
  if (threadIdx.x % WAVE_SIZE == 0) {
    sh[parity][wave] = v;
  }
  __syncthreads();
  const int first = wave - wave % waves_per_row;
  float2 sum = {0.0f, 0.0f};
  for (int w = first; w < first + waves_per_row; w++) {
    const float2 p = sh[parity][w];
    sum.x += p.x;
    sum.y += p.y;
  }
  return sum;
}

// The mean and the sum of squared deviations of a row from (mean, m2) of the wave's part, combined
// over the parts as row_sum does; rn is 1 / cols.
template <int MAX_WAVES_PER_BLOCK>
__device__ __forceinline__ float2 row_mean_m2(const float2 v, float2 (&sh)[2][MAX_WAVES_PER_BLOCK],
                                              const int parity, const int wave,
                                              const int waves_per_row, const uint32_t part_cols,
                                              const uint32_t cols, const float rn) {
  if (waves_per_row == 1) {
    return v;
  }
  if (threadIdx.x % WAVE_SIZE == 0) {
    sh[parity][wave] = v;
  }
  __syncthreads();
  const int first = wave - wave % waves_per_row;
  float sum = 0.0f;
  for (int p = 0; p < waves_per_row; p++) {
    sum = fmaf(part_elts(p, part_cols, cols), sh[parity][first + p].x, sum);
  }
  const float mean = sum * rn;
  float m2 = 0.0f;
  for (int p = 0; p < waves_per_row; p++) {
    const float2 s = sh[parity][first + p];
    const float d = s.x - mean;
    m2 += fmaf(part_elts(p, part_cols, cols) * d, d, s.y);
  }
  return {mean, m2};
}

// Stores the N values v to dst as OType, scaled by scale for FP8, and with requires_amax, which is
// wave-uniform, raises amax to their max magnitude before scaling and conversion. Rounding is
// monotonic, so the TE norm kernels' amax over converted 16-bit outputs is that of out_amax.
template <typename OType, int N>
__device__ __forceinline__ void store_out(OType *dst, const float (&v)[N], const float scale,
                                          const bool requires_amax, float &amax) {
  if (requires_amax) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(v[j]));
    }
  }
  Vec<OType, N> out;
  if constexpr (is_fp8<OType>::value) {
    static_assert(N % 4 == 0, "FP8 values convert four at a time");
    uint32_t packed[N / 4];
#pragma unroll
    for (int i = 0; i < N / 4; i++) {
      packed[i] = cvt_4x_to_fp8<OType, false>(v[4 * i] * scale, v[4 * i + 1] * scale,
                                              v[4 * i + 2] * scale, v[4 * i + 3] * scale);
    }
    static_assert(sizeof(out.data) == sizeof(packed), "N FP8 values pack into N / 4 words");
    __builtin_memcpy(&out.data, packed, sizeof(packed));
  } else if constexpr (sizeof(OType) == 2) {
    // Pairs convert into whole registers, which elementwise stores would split.
    uint32_t packed[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; i++) {
      packed[i] = cvt_2x_to_16bit<OType>(v[2 * i], v[2 * i + 1]);
    }
    static_assert(sizeof(out.data) == sizeof(packed), "N 16-bit values pack into N / 2 words");
    __builtin_memcpy(&out.data, packed, sizeof(packed));
  } else {
#pragma unroll
    for (int j = 0; j < N; j++) {
      out.data.elt[j] = cvt_from_float<OType>(v[j]);
    }
  }
  out.store_to(dst);
}

// The amax of outputs whose values before conversion have the amax amax, as the TE norm kernels
// define it: FP8 outputs take it before conversion, others after.
template <typename OType>
__device__ __forceinline__ float out_amax(const float amax) {
  if constexpr (is_fp8<OType>::value) {
    return amax;
  } else {
    return static_cast<float>(cvt_from_float<OType>(amax));
  }
}

// Raises *amax_ptr, through its rocm_amax_target, to the max of the block's per-thread amax.
template <int MAX_WAVES_PER_BLOCK>
__device__ __forceinline__ void block_raise_amax(float amax, float *amax_ptr, float *amax_slots) {
  __shared__ float amax_sh[MAX_WAVES_PER_BLOCK];
  amax = rocm_subwarp_allreduce<WAVE_SIZE>(amax, rocm_op::max{});
  if (threadIdx.x % WAVE_SIZE == 0) {
    amax_sh[threadIdx.x / WAVE_SIZE] = amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_amax = 0.0f;
    for (int w = 0; w < static_cast<int>(blockDim.x / WAVE_SIZE); w++) {
      block_amax = fmaxf(block_amax, amax_sh[w]);
    }
    rocm_atomicMaxFloat(rocm_amax_target(amax_ptr, amax_slots, blockIdx.x), block_amax);
  }
}
#endif  // __gfx1250__

}  // namespace rocm
}  // namespace normalization
}  // namespace transformer_engine
