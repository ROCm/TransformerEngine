/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

#include "../common.h"
#include "../recipe/recipe_common.cuh"
#include "../util/logging.h"
#include "../utils.cuh"
#include "fp8_blockwise_direct_transpose.cuh"

namespace transformer_engine {
namespace fp8_blockwise_restripe {
namespace {

constexpr int kBlockLen = 128;                     // 1D blockwise block length
constexpr int kThreads = 256;
constexpr int kTileDwords = kBlockLen / 4;         // dwords in one 128-byte tile row
constexpr int kChunksPerRow = kBlockLen / 16;      // 16-byte loads in one tile row
constexpr int kLoadsPerThread = kBlockLen * kChunksPerRow / kThreads;
constexpr int kMaxGroups = 256;                    // group ends travel in the kernel arguments

enum class Mode {
  kScalesOnly,  // regroup the rowwise scale_inv only
  kDirect,      // columnwise by exponent arithmetic (power-of-two scales)
  kRequant,     // columnwise by dequantize -> column amax -> requantize
};

// The tile is staged as [row][dword]. The transposed read takes one dword column over 16 rows
// per thread, so column dwords are XOR-swizzled by the row's 16-row band; with the one-dword pad
// that spreads a wavefront's reads and its row-major writes over all banks.
__device__ __forceinline__ int swizzle(int row, int dword) { return dword ^ (((row >> 4) & 3) << 3); }

template <typename OType>
__device__ __forceinline__ float fp8_to_float(uint32_t byte) {
  OType v;
  *reinterpret_cast<uint8_t *>(&v) = static_cast<uint8_t>(byte);
  return static_cast<float>(v);
}

template <typename OType>
__device__ __forceinline__ uint32_t float_to_fp8(float f) {
  const OType v = static_cast<OType>(f);
  return *reinterpret_cast<const uint8_t *>(&v);
}

// Exclusive end row of each group, passed by value so the kernel needs no device copy of the
// split lengths.
struct GroupEnds {
  uint32_t end[kMaxGroups];
  int num_groups;
};

// Every group length is a multiple of kBlockLen, so a 128-row tile lies in exactly one group:
// the one whose [off, end) holds row0. The loop is uniform across the block.
__device__ __forceinline__ void find_group(const GroupEnds &groups, size_t row0, size_t *off,
                                           size_t *end) {
  size_t prev = 0;
  for (int g = 0; g < groups.num_groups; ++g) {
    const size_t e = groups.end[g];
    if (e > row0) {
      *off = prev;
      *end = e;
      return;
    }
    prev = e;
  }
  *off = prev;
  *end = prev;
}

/* Grouped 1x128 rowwise -> 128x1 columnwise restripe of a packed [M, K] FP8 tensor.
 *
 * One block per 128x128 tile. Outputs are group-contiguous, so each group's slices are dense:
 *   rs_out : group g at offset KB*off_g, shape [KB, m_g]     (rowwise scale_inv)
 *   y      : group g at offset K*off_g,  shape [K, m_g]      (columnwise payload, transposed)
 *   cs     : [M/128, K]; group g owns rows off_g/128 .. (off_g+m_g)/128
 */
template <Mode kMode, typename OType, bool kFnuz, bool kPow2>
__global__ void __launch_bounds__(kThreads)
    restripe_kernel(const uint8_t *__restrict__ x, size_t n_rows, size_t n_cols,
                    const float *__restrict__ rs_in, const GroupEnds groups,
                    float *__restrict__ rs_out, uint8_t *__restrict__ y,
                    float *__restrict__ cs, float max_fp8, float epsilon) {
  using F = fp8_blockwise_direct_transpose::Fmt<std::is_same<OType, fp8e5m2>::value ? 1 : 0, kFnuz>;

  const size_t num_k_blocks = n_cols / kBlockLen;
  const size_t tile_m = blockIdx.x / num_k_blocks;
  const size_t tile_k = blockIdx.x % num_k_blocks;
  const size_t row0 = tile_m * kBlockLen;
  const size_t col0 = tile_k * kBlockLen;
  const int tid = threadIdx.x;

  size_t g_off, g_end;
  find_group(groups, row0, &g_off, &g_end);
  const size_t g_len = g_end - g_off;
  const size_t local_row0 = row0 - g_off;

  // Issue the tile's loads first so they overlap the scale traffic below.
  uint4 raw[kLoadsPerThread];
  if constexpr (kMode != Mode::kScalesOnly) {
#pragma unroll
    for (int i = 0; i < kLoadsPerThread; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx / kChunksPerRow;
      const int c = idx % kChunksPerRow;
      raw[i] = *reinterpret_cast<const uint4 *>(x + (row0 + r) * n_cols + col0 + c * 16);
    }
  }

  __shared__ __align__(16) float s_row[kBlockLen];
  if (tid < kBlockLen) {
    const float s = rs_in[tile_k * n_rows + row0 + tid];
    s_row[tid] = s;
    rs_out[num_k_blocks * g_off + tile_k * g_len + local_row0 + tid] = s;
  }
  if constexpr (kMode == Mode::kScalesOnly) {
    return;
  }

  __shared__ uint32_t tile[kBlockLen][kTileDwords + 1];
  __syncthreads();

  if constexpr (kMode == Mode::kDirect) {
    // Columnwise scale of the whole tile: the max of its row scales. Positive floats order like
    // their bit patterns, and every thread reads the same words (LDS broadcast).
    uint32_t t_bits = 0;
#pragma unroll
    for (int i = 0; i < kBlockLen / 4; ++i) {
      const uint4 s4 = reinterpret_cast<const uint4 *>(s_row)[i];
      t_bits = max(max(t_bits, s4.x), max(s4.y, max(s4.z, s4.w)));
    }
    if (tid < kBlockLen) {
      cs[tile_m * n_cols + col0 + tid] = __uint_as_float(t_bits);
    }
#pragma unroll
    for (int i = 0; i < kLoadsPerThread; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx / kChunksPerRow;
      const int c = idx % kChunksPerRow;
      const int k = fp8_blockwise_direct_transpose::pow2_scale_shift(
          t_bits, __float_as_uint(s_row[r]));
      const uint32_t w[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        tile[r][swizzle(r, c * 4 + j)] = fp8_blockwise_direct_transpose::fp8_rescale_dword<F>(w[j], k);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kLoadsPerThread; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx / kChunksPerRow;
      const int c = idx % kChunksPerRow;
      const uint32_t w[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        tile[r][swizzle(r, c * 4 + j)] = w[j];
      }
    }
  }
  __syncthreads();

  // Transposed phase: this thread owns tile rows [m0, m0 + 16) x columns [n0, n0 + 4), and
  // writes 16 contiguous bytes of each of its 4 output rows. The 8 threads sharing a column
  // block cover one 128-byte output segment.
  const int mb = tid % (kBlockLen / 16);
  const int nb = tid / (kBlockLen / 16);
  const int m0 = mb * 16;
  const int n0 = nb * 4;
  uint32_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    w[i] = tile[m0 + i][swizzle(m0 + i, nb)];
  }
  uint8_t *y_tile = y + n_cols * g_off + local_row0 + m0;

  if constexpr (kMode == Mode::kDirect) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        o[q] = ((w[4 * q] >> (8 * b)) & 0xFF) | (((w[4 * q + 1] >> (8 * b)) & 0xFF) << 8) |
               (((w[4 * q + 2] >> (8 * b)) & 0xFF) << 16) |
               (((w[4 * q + 3] >> (8 * b)) & 0xFF) << 24);
      }
      *reinterpret_cast<uint4 *>(y_tile + (col0 + n0 + b) * g_len) =
          make_uint4(o[0], o[1], o[2], o[3]);
    }
  } else {
    __shared__ float col_amax[kBlockLen / 16][kBlockLen];
    float amax[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float s = s_row[m0 + i];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        amax[b] = fmaxf(amax[b], fabsf(fp8_to_float<OType>((w[i] >> (8 * b)) & 0xFF) * s));
      }
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      col_amax[mb][n0 + b] = amax[b];
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < 4; ++b) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < kBlockLen / 16; ++j) {
        a = fmaxf(a, col_amax[j][n0 + b]);
      }
      const float scale = compute_scale_from_amax(a, max_fp8, kPow2, epsilon, FLT_MAX);
      if (mb == 0) {
        cs[tile_m * n_cols + col0 + n0 + b] = __frcp_rn(scale);
      }

      uint32_t o[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float v = fp8_to_float<OType>((w[i] >> (8 * b)) & 0xFF) * s_row[m0 + i] * scale;
        o[i / 4] |= float_to_fp8<OType>(fminf(fmaxf(v, -max_fp8), max_fp8)) << (8 * (i % 4));
      }
      *reinterpret_cast<uint4 *>(y_tile + (col0 + n0 + b) * g_len) =
          make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

template <Mode kMode, typename OType, bool kFnuz, bool kPow2>
void launch(const Tensor &data, const Tensor &scale_inv, const GroupEnds &groups,
            Tensor *rs_out, Tensor *col_data, Tensor *col_scale_inv, size_t n_rows,
            size_t n_cols, float epsilon, cudaStream_t stream) {
  const size_t num_blocks = (n_rows / kBlockLen) * (n_cols / kBlockLen);
  const bool colwise = kMode != Mode::kScalesOnly;
  restripe_kernel<kMode, OType, kFnuz, kPow2><<<num_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(data.data.dptr), n_rows, n_cols,
      reinterpret_cast<const float *>(scale_inv.data.dptr), groups,
      reinterpret_cast<float *>(rs_out->data.dptr),
      colwise ? reinterpret_cast<uint8_t *>(col_data->data.dptr) : nullptr,
      colwise ? reinterpret_cast<float *>(col_scale_inv->data.dptr) : nullptr,
      std::is_same<OType, fp8e5m2>::value ? 57344.f : (kFnuz ? 240.f : 448.f), epsilon);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename OType, bool kFnuz>
void dispatch_mode(bool colwise, bool direct, bool pow2, const Tensor &data,
                   const Tensor &scale_inv, const GroupEnds &groups, Tensor *rs_out,
                   Tensor *col_data, Tensor *col_scale_inv, size_t n_rows, size_t n_cols,
                   float epsilon, cudaStream_t stream) {
#define TE_RESTRIPE_LAUNCH(MODE, POW2)                                            \
  launch<MODE, OType, kFnuz, POW2>(data, scale_inv, groups, rs_out, col_data,     \
                                   col_scale_inv, n_rows, n_cols, epsilon, stream)
  if (!colwise) {
    TE_RESTRIPE_LAUNCH(Mode::kScalesOnly, false);
  } else if (direct) {
    TE_RESTRIPE_LAUNCH(Mode::kDirect, true);
  } else if (pow2) {
    TE_RESTRIPE_LAUNCH(Mode::kRequant, true);
  } else {
    TE_RESTRIPE_LAUNCH(Mode::kRequant, false);
  }
#undef TE_RESTRIPE_LAUNCH
}

}  // namespace

void fp8_blockwise_1d_rowwise_to_columnwise_grouped(const Tensor &data, const Tensor &scale_inv,
                                                    const size_t *group_lens, size_t num_groups,
                                                    Tensor *rs_out,
                                                    Tensor *col_data, Tensor *col_scale_inv,
                                                    float epsilon, bool force_pow_2_scales,
                                                    bool direct, cudaStream_t stream) {
  NVTE_CHECK(data.data.shape.size() == 2, "Rowwise data must be 2D [M, K].");
  const size_t n_rows = data.data.shape[0];
  const size_t n_cols = data.data.shape[1];
  NVTE_CHECK(n_rows % kBlockLen == 0, "Rows (", n_rows, ") must be a multiple of ", kBlockLen,
             ".");
  NVTE_CHECK(n_cols % kBlockLen == 0, "Columns (", n_cols, ") must be a multiple of ",
             kBlockLen, ".");
  NVTE_CHECK(is_fp8_dtype(data.data.dtype), "Rowwise data must be FP8.");
  NVTE_CHECK(scale_inv.data.dtype == DType::kFloat32, "Rowwise scale_inv must be FP32.");
  NVTE_CHECK(scale_inv.data.numel() == (n_cols / kBlockLen) * n_rows,
             "Rowwise scale_inv must be a contiguous [K/128, M] buffer.");
  NVTE_CHECK(rs_out->data.dtype == DType::kFloat32 && rs_out->data.numel() == scale_inv.data.numel(),
             "Grouped rowwise scale_inv output must be FP32 with K/128 * M elements.");
  NVTE_CHECK(num_groups <= static_cast<size_t>(kMaxGroups), "At most ", kMaxGroups,
             " groups are supported, got ", num_groups, ".");
  NVTE_CHECK(n_rows <= UINT32_MAX, "Rows (", n_rows, ") must fit in 32 bits.");
  GroupEnds groups{};
  groups.num_groups = static_cast<int>(num_groups);
  size_t acc = 0;
  for (size_t g = 0; g < num_groups; ++g) {
    NVTE_CHECK(group_lens[g] % kBlockLen == 0, "Group lengths must be multiples of ", kBlockLen,
               ", got ", group_lens[g], ".");
    acc += group_lens[g];
    groups.end[g] = static_cast<uint32_t>(acc);
  }
  NVTE_CHECK(acc == n_rows, "Group lengths sum to ", acc, " but the tensor has ", n_rows,
             " rows.");
  const bool colwise = col_data->data.dptr != nullptr;
  if (colwise) {
    NVTE_CHECK(col_data->data.numel() == n_rows * n_cols,
               "Columnwise data must hold M * K elements.");
    NVTE_CHECK(col_scale_inv->data.dtype == DType::kFloat32 &&
                   col_scale_inv->data.numel() == (n_rows / kBlockLen) * n_cols,
               "Columnwise scale_inv must be FP32 [M/128, K].");
    NVTE_CHECK(!direct || force_pow_2_scales,
               "The direct transpose needs power-of-two scales.");
  }
  NVTE_CHECK(reinterpret_cast<uintptr_t>(data.data.dptr) % 16 == 0 &&
                 (!colwise || reinterpret_cast<uintptr_t>(col_data->data.dptr) % 16 == 0),
             "FP8 buffers must be 16-byte aligned.");
  if (n_rows == 0) {
    return;
  }

  const bool fnuz = te_fp8_fnuz();
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      data.data.dtype, OType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          fnuz, kFnuz,
          dispatch_mode<OType, kFnuz>(colwise, direct, force_pow_2_scales, data, scale_inv,
                                      groups, rs_out, col_data, col_scale_inv, n_rows, n_cols,
                                      epsilon, stream);););
}

}  // namespace fp8_blockwise_restripe
}  // namespace transformer_engine

void nvte_fp8_blockwise_1d_rowwise_to_columnwise_grouped(
    const NVTETensor rowwise_data, const NVTETensor rowwise_scale_inv,
    const size_t *group_lens, size_t num_groups, NVTETensor grouped_rowwise_scale_inv,
    NVTETensor columnwise_data, NVTETensor columnwise_scale_inv, float epsilon,
    int force_pow_2_scales, int direct, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_blockwise_1d_rowwise_to_columnwise_grouped);
  using namespace transformer_engine;
  fp8_blockwise_restripe::fp8_blockwise_1d_rowwise_to_columnwise_grouped(
      *convertNVTETensorCheck(rowwise_data), *convertNVTETensorCheck(rowwise_scale_inv),
      group_lens, num_groups, convertNVTETensorCheck(grouped_rowwise_scale_inv),
      convertNVTETensorCheck(columnwise_data), convertNVTETensorCheck(columnwise_scale_inv),
      epsilon, force_pow_2_scales != 0, direct != 0, stream);
}
