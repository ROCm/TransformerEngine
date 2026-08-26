/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_UTILS_H
#define CK_FUSED_ATTN_UTILS_H

#include<fstream>
#include<utility>
#include<type_traits>
#include<hip/hip_runtime.h>
#if ENABLE_CK
#include "ck_tile/host.hpp"
#else
// CK-free (gfx1250) build: the real ck_tile template headers do not compile for
// this arch. Pull the lightweight shim (ck_tile::index_t / stream_config, plus
// mask_enum / bias_enum) via aiter_hip_common.h, and provide the few ck_tile
// numeric symbols the reduction kernels use. half_t is the _Float16 scalar
// (unambiguous float arithmetic, unlike __half which is ambiguous in the
// kernels' implicit-conversion else-branch); bf16_t is __hip_bfloat16. Both are
// bit-compatible with fp16/bf16 so the kernel sources stay unchanged.
#include "aiter_hip_common.h"
#include <hip/hip_bf16.h>
namespace ck_tile {
using half_t = _Float16;
using bf16_t = __hip_bfloat16;
__host__ __device__ inline float bf16_to_float(bf16_t x) { return __bfloat162float(x); }
__host__ __device__ inline bf16_t float_to_bf16(float x) { return __float2bfloat16(x); }
}  // namespace ck_tile
#endif
#include "ck_fused_attn/ck_fused_attn.hpp"

//forward declaration for ck_tile enum
enum class mask_enum;
//forward declaration for ck_tile enum
enum class bias_enum;

namespace ck_fused_attn{

#define CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, type, ...)   \
switch (dtype) {                                            \
  case DType::kFloat16: {                                   \
    using type = ck_tile::half_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  case DType::kBFloat16: {                                  \
    using type = ck_tile::bf16_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  default:                                                  \
    throw std::runtime_error("Invalid type for 16 bit..");  \
}

// Device-side dtype conversion helpers for the reduction kernels.
// ck_tile::bf16_t lacks implicit float conversions and needs explicit
// bf16_to_float / float_to_bf16; fp16/fp32 convert implicitly.
template<typename T>
__device__ __forceinline__ float to_f32(T x) {
  if constexpr (std::is_same_v<T, ck_tile::bf16_t>) {
    return ck_tile::bf16_to_float(x);
  } else {
    return static_cast<float>(x);
  }
}

template<typename T>
__device__ __forceinline__ T from_f32(float x) {
  if constexpr (std::is_same_v<T, ck_tile::bf16_t>) {
    return ck_tile::float_to_bf16(x);
  } else {
    return static_cast<T>(x);
  }
}

// element-wise bias shape
enum class BiasShape{
  k11SS = 0,
  k1HSS = 1,
  kB1SS = 2,
  kBHSS = 3,
  kNumBiasShapes  /*!< Number of supported bias shapes */
};

std::string get_data_type_str(DType dtype);
BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h);

//get ck_tile bias_type and CK_FUSED_ATTN bias_shape from a fwd/bwd args struct
std::pair<bias_enum, BiasShape> get_ck_bias_type_shape(const CKAttnCommonArgs* args);

uint64_t get_runtime_max_seqlen(uint64_t b, const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr, void* workspace, hipStream_t stream);

// This helper merely standardizes the logging to make it a bit easier to parse
// through it at a glance while guaranteeing uniformity.
template<typename T>
void log_value(std::ostream* log_file, const char* label, const T& value) {
    (*log_file) << label << ": " << value << "\n";
}

ck_tile::index_t get_batch_stride_bias(
  ck_tile::index_t bias_h,
  BiasShape bias_shape,
  ck_tile::index_t max_seqlen_q,
  ck_tile::index_t max_seqlen_k,
  bool is_group_mode,
  bool is_fwd
);
ck_tile::index_t get_nhead_stride_bias(
  BiasShape bias_shape,
  ck_tile::index_t max_seqlen_q,
  ck_tile::index_t max_seqlen_k,
  bool is_group_mode
);
std::ostream* get_ck_log_stream();

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_UTILS_H
