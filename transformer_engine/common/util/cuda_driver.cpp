/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <filesystem>

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace cuda_driver {

void *get_symbol(const char *symbol) {
  void *entry_point;
#ifdef __HIP_PLATFORM_AMD__
  hipDriverProcAddressQueryResult driver_result;
  NVTE_CHECK_CUDA(hipGetProcAddress(symbol, &entry_point, HIP_VERSION_MAJOR*100+HIP_VERSION_MINOR, 0, &driver_result));
  NVTE_CHECK(driver_result == HIP_GET_PROC_ADDRESS_SUCCESS,
             "Could not find CUDA driver entry point for ", symbol);
#else
  cudaDriverEntryPointQueryResult driver_result;
  NVTE_CHECK_CUDA(cudaGetDriverEntryPoint(symbol, &entry_point, cudaEnableDefault, &driver_result));
  NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
             "Could not find CUDA driver entry point for ", symbol);
#endif
  return entry_point;
}

}  // namespace cuda_driver

}  // namespace transformer_engine
