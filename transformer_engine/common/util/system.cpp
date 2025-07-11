/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/system.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "../common.h"

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
#if HIP_VERSION >= 60300000
  return te_fp8_fnuz();
#endif
  return true; // default to true for older versions compatibility
}
#endif