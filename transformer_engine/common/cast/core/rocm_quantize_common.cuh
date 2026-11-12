/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" //dummy include to prevent hipification adding this header

// Device helpers shared by the ROCm quantize and dequantize kernels: conversions, and the 16-point
// Hadamard transform of the FP4 paths. This expects its includer to have included the common
// headers.

#include <type_traits>

namespace transformer_engine {
namespace dispatch {
namespace common {

// hip_bfloat16 rounds with per-element branches; __bf16 is branch-free (v_cvt_pk_bf16_f32 on
// gfx950+).
template <typename T>
__device__ __forceinline__ T cvt_from_float(const float x) {
  if constexpr (std::is_same_v<T, bf16>) {
    return __builtin_bit_cast(T, static_cast<__bf16>(x));
  } else {
    return static_cast<T>(x);
  }
}

#ifdef __gfx1250__
typedef float f32x8 __attribute__((ext_vector_type(8)));

// x / scale for 8 values as FP4, rounded to nearest even, the first in the low nibble.
__device__ __forceinline__ uint32_t cvt_8x_to_fp4(const float *x, const float scale) {
  const f32x8 v = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]};
  return __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(v, scale);
}

// The 16-point Walsh-Hadamard transform, scaled by 1/4, with the operations of the four-lane
// butterfly of hadamard16_inplace and wht16, lane q's values being x[4q..4q+3].
__device__ __forceinline__ void hadamard16(float *x) {
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const float a0 = x[4 * q] + x[4 * q + 1];
    const float a1 = x[4 * q] - x[4 * q + 1];
    const float a2 = x[4 * q + 2] + x[4 * q + 3];
    const float a3 = x[4 * q + 2] - x[4 * q + 3];
    x[4 * q] = a0 + a2;
    x[4 * q + 2] = a0 - a2;
    x[4 * q + 1] = a1 + a3;
    x[4 * q + 3] = a1 - a3;
  }
#pragma unroll
  for (int d = 1; d <= 2; d *= 2) {
    float y[16];
#pragma unroll
    for (int q = 0; q < 4; q++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float p = x[4 * (q ^ d) + i];
        y[4 * q + i] = (q & d) ? p - x[4 * q + i] : p + x[4 * q + i];
      }
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
      x[i] = y[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 16; i++) {
    x[i] *= 0.25f;
  }
}
#endif  // __gfx1250__

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine
