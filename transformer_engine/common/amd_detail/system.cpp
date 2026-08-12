/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
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

extern "C" int nvte_is_hipkittens_gemm_available() {
#ifdef USE_HIPKITTENS_GEMM
  return 1;
#else
  return 0;
#endif
}
#endif
