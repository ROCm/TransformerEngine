/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_UTILS_H
#define CK_FUSED_ATTN_UTILS_H

#include <iostream>
#include<cstdint>

namespace ck_fused_attn{

// element-wise bias shape
enum BiasShape{
  k11SS = 0,
  k1HSS = 1,
  kB1SS = 2,
  kBHSS = 3,
  kNumBiasShapes  /*!< Number of supported bias shapes */
};

BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h);

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_UTILS_H
