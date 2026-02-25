/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_UTILS_H
#define CK_FUSED_ATTN_UTILS_H

#include<cstdlib>
#include<fstream>
#include<string>
#include<hip/hip_runtime.h>

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

std::ostream* get_ck_log_stream();

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_UTILS_H
