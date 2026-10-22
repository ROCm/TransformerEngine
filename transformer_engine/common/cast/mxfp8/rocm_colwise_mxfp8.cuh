/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
// Tile helpers shared by the ROCm kernels that emit a columnwise-only MXFP8 tensor
// (fused grouped requantize, grouped scaled SwiGLU). The includer provides e8m0_t, ptx::
// and Quantized_Limits.
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

#include <cstring>

#include "../../util/rocm_device_utils.cuh"

namespace transformer_engine {
namespace rocm_colwise_mxfp8 {

// One MXFP8 columnwise scale block of rows, by the 128 columns one thread each owns.
constexpr int kTileRows = 32;
constexpr int kTileCols = 128;
constexpr int kThreads = kTileCols;

// Copies a kTileRows x kTileCols tile of a row-major [rows, cols] view with row stride
// `stride` into shared memory. Out-of-range elements read as zero. kAligned promises that
// cols is a multiple of the 16-byte vector and that src is 16-byte aligned.
template <typename T, bool kAligned>
__device__ __forceinline__ void load_tile(T (&dst)[kTileRows][kTileCols], const T *src,
                                          const size_t row_base, const size_t col_base,
                                          const size_t stride, const size_t rows,
                                          const size_t cols) {
  constexpr int kVec = ROCM_VEC_BYTES / sizeof(T);
  constexpr int kVecsPerRow = kTileCols / kVec;
  for (int i = threadIdx.x; i < kTileRows * kVecsPerRow; i += blockDim.x) {
    const int r = i / kVecsPerRow;
    const int c = (i % kVecsPerRow) * kVec;
    const size_t row = row_base + r;
    const size_t col = col_base + c;
    if (kAligned && row < rows && col < cols) {
      *reinterpret_cast<uint4 *>(&dst[r][c]) =
          *reinterpret_cast<const uint4 *>(src + row * stride + col);
    } else {
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        dst[r][c + k] = (row < rows && col + k < cols) ? src[row * stride + col + k]
                                                       : static_cast<T>(0.0f);
      }
    }
  }
}

// Writes the in-range part of a shared kTileRows x kTileCols tile to a row-major [rows, cols]
// tensor. kAligned as in load_tile.
template <typename T, bool kAligned>
__device__ __forceinline__ void store_tile(const T (&src)[kTileRows][kTileCols], T *dst,
                                           const size_t row_base, const size_t col_base,
                                           const size_t rows, const size_t cols) {
  constexpr int kVec = ROCM_VEC_BYTES / sizeof(T);
  constexpr int kVecsPerRow = kTileCols / kVec;
  for (int i = threadIdx.x; i < kTileRows * kVecsPerRow; i += blockDim.x) {
    const int r = i / kVecsPerRow;
    const int c = (i % kVecsPerRow) * kVec;
    const size_t row = row_base + r;
    const size_t col = col_base + c;
    if (row >= rows || col >= cols) {
      continue;
    }
    if (kAligned) {
      *reinterpret_cast<uint4 *>(dst + row * cols + col) =
          *reinterpret_cast<const uint4 *>(&src[r][c]);
    } else {
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        if (col + k < cols) {
          dst[row * cols + col + k] = src[r][c + k];
        }
      }
    }
  }
}

// Quantizes one column of kTileRows values as a single MXFP8 block, storing the codes with
// stride `out_stride` and returning the e8m0 scale. The arithmetic is the columnwise path of
// rocm_quantize_mxfp8_body.inc, so the result matches nvte_quantize bit for bit.
template <typename OType>
__device__ __forceinline__ e8m0_t quantize_column(const float (&values)[kTileRows], OType *out,
                                                  const int out_stride) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    amax = fmaxf(amax, fabsf(values[i]));
  }
  const e8m0_t biased_exponent =
      ptx::float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);

#ifdef HAS_CVT_4xFLOAT8
  const float cvt_scale = (biased_exponent == 0) ? 1.0f : ptx::exp2f(biased_exponent);
#pragma unroll
  for (int i = 0; i < kTileRows; i += 4) {
    const uint32_t packed = rocm_cvt_4xfloat8<OType>(values[i], values[i + 1], values[i + 2],
                                                     values[i + 3], cvt_scale);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      memcpy(&out[(i + k) * out_stride], reinterpret_cast<const uint8_t *>(&packed) + k,
             sizeof(OType));
    }
  }
#else
  const float block_scale_inverse = ptx::exp2f_rcp<float>(biased_exponent);
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    out[i * out_stride] = static_cast<OType>(values[i] * block_scale_inverse);
  }
#endif  // #ifdef HAS_CVT_4xFLOAT8
  return biased_exponent;
}

}  // namespace rocm_colwise_mxfp8
}  // namespace transformer_engine
