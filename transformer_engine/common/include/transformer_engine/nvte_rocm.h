/* Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information. */

/*! \file nvte_rocm.h
 *  \brief ROCm core introspection ABI (TE_ROCM_CORE_ABI, plugin plan S3.3 / manifest ABI-001).
 *
 *  The deliberately tiny, versioned C surface that PYTHON is allowed to call on the core
 *  library via ctypes. Nothing links against libtransformer_engine by SONAME - the framework
 *  extension resolves its symbols through the RTLD_GLOBAL preload - so core-ABI compatibility
 *  is enforced at load time by comparing nvte_rocm_core_abi_version() against the value the
 *  caller was built/assembled for. Bump NVTE_ROCM_CORE_ABI_VERSION on ANY change to a symbol
 *  the framework extension or these ctypes callers use.
 *
 *  Every function here must also appear in the seam inventory's ctypes-demand section
 *  (tools/seam_inventory.py); adding one without inventory coverage is a governance failure.
 */

#ifndef TRANSFORMER_ENGINE_NVTE_ROCM_H_
#define TRANSFORMER_ENGINE_NVTE_ROCM_H_

#include <stdbool.h>
#include <stdint.h>

/*! Version of this introspection ABI plus the extension<->core symbol contract. */
#define NVTE_ROCM_CORE_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief The TE_ROCM_CORE_ABI version this library was built with. */
int64_t nvte_rocm_core_abi_version(void);

/*! \brief True when this core library was built for ROCm. */
bool nvte_is_rocm_build(void);

/*! \brief True when FP8 on the active architecture uses the FNUZ formats (gfx942)
 *         rather than OCP (gfx950+). Drives Format.E4M3/E5M2 max values in Python. */
bool nvte_uses_fp8_fnuz(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_NVTE_ROCM_H_
