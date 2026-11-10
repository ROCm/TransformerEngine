/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Header-only helpers shared by the ck_fused_attn wrapper and the TE CK glue (fused_attn_rocm).
// Depends only on HIP, so it can be included from either side.

#ifndef CK_FUSED_ATTN_DEVICE_H
#define CK_FUSED_ATTN_DEVICE_H

#include <cstdint>
#include <stdexcept>
#include <hip/hip_runtime.h>

#include "ck_fused_attn/ck_fused_attn.hpp"

// Dispatch a ck_fused_attn::DType to a 16-bit element type. The concrete types are caller-chosen
// (e.g. ck_tile::half_t/bf16_t or __half/hip_bfloat16). Throws on any other dtype.
#define CK_FUSED_ATTN_DTYPE_SWITCH_16BIT(dtype, type, fp16_type, bf16_type, ...) \
  switch (dtype) {                                                               \
    case ::ck_fused_attn::DType::kFloat16: {                                     \
      using type = fp16_type;                                                    \
      __VA_ARGS__;                                                               \
      break;                                                                     \
    }                                                                            \
    case ::ck_fused_attn::DType::kBFloat16: {                                    \
      using type = bf16_type;                                                    \
      __VA_ARGS__;                                                               \
      break;                                                                     \
    }                                                                            \
    default:                                                                     \
      throw std::runtime_error("Invalid type for 16 bit..");                     \
  }

namespace ck_fused_attn {

// Device replacement for std::upper_bound over a cu_seqlens array.
// For an increasing array of size len, returns the index such that array[index] <= target < array[index+1].
// Requires 0 <= target <= array[len-1].
__forceinline__ __device__ int binary_search(int32_t target, const int32_t* array, uint64_t len) {
  int left = 1, right = static_cast<int>(len) - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

}  // namespace ck_fused_attn
#endif  // CK_FUSED_ATTN_DEVICE_H
