/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

 #include <iostream>
#include "hip_float8.h"
#include "../util/string.h"

extern "C" bool nvte_is_rocm_build() {
#ifdef USE_ROCM
  return true;
#else
  return false;
#endif
}

#ifdef USE_ROCM
bool te_check_fp8_fnuz() {
  hipDeviceProp_t prop;
  hipError_t res= hipGetDeviceProperties(&prop, 0);
  if (res != hipSuccess) {
    if (res == hipErrorNoDevice) {
      // no device, default to OCP
      std::cerr << "No HIP device found, defaulting to OCP format for FP8\n";
      return false;
    }
    //TODO: better error out system
    throw std::runtime_error(transformer_engine::concat_strings(
      "hipGetDeviceProperties failed with error: ", hipGetErrorString(res)));
  }
  return prop.major == 9 && prop.minor == 4;
}


extern "C" bool nvte_uses_fp8_fnuz() 
{
#if HIP_VERSION >= 60300000
  return te_fp8_fnuz();
#endif
  return true; // default to true for older versions compatibility
}
#endif
