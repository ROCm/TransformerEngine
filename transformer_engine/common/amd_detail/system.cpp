/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include "hip_float8.h"

extern "C" bool nvte_is_rocm_build() {
#ifdef USE_ROCM
  return true;
#else
  return false;
#endif
}

#ifdef USE_ROCM
extern "C" bool nvte_uses_fp8_fnuz() 
{
  return te_fp8_fnuz();
}
#endif
