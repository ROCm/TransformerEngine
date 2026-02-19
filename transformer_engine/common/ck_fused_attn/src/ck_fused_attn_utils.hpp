/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_UTILS_H
#define CK_FUSED_ATTN_UTILS_H

#include<iostream>
#include<cstdint>
#include<hip/hip_runtime.h>
#include "ck_tile/host.hpp"

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

// element-wise bias shape
enum class BiasShape{
  k11SS = 0,
  k1HSS = 1,
  kB1SS = 2,
  kBHSS = 3,
  kNumBiasShapes  /*!< Number of supported bias shapes */
};

//forward declaration of ck_fused_attn::DType
enum class DType ;
//forward declaration of ck_fused_attn::MaskType
enum class MaskType;
//forward declaration of ck_fused_attn::BiasType
enum class BiasType;

std::string get_data_type_str(DType dtype);
BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h);
std::pair<bias_enum, BiasShape> get_ck_bias_type_shape(BiasType attn_bias_type, uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h);

uint64_t get_runtime_max_seqlen(uint64_t b, const void* cu_seqlen_ptr, const void* cu_seqlen_padded_ptr, void* workspace, hipStream_t stream);

// This helper merely standardizes the logging to make it a bit easier to parse
// through it at a glance while guaranteeing uniformity.
template<typename T>
void log_value(const char* label, const T& value) {
    std::cout << label << ": " << value << "\n";
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

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_UTILS_H
