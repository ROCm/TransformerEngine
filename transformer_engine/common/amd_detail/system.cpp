/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include "hip_float8.h"

#include <transformer_engine/nvte_rocm.h>

// TE_ROCM_CORE_ABI (plugin plan S3.3): the load-time contract check. Nothing links against this
// library by SONAME - Python and the framework extension bind symbols after an RTLD_GLOBAL
// preload - so this version, compared at load, is the only enforcement of core-ABI identity.
extern "C" int64_t nvte_rocm_core_abi_version() { return NVTE_ROCM_CORE_ABI_VERSION; }

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
