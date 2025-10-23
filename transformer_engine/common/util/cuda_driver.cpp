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

<<<<<<< HEAD
void *get_symbol(const char *symbol) {
  void *entry_point;
#ifdef __HIP_PLATFORM_AMD__
  hipDriverProcAddressQueryResult driver_result;
  NVTE_CHECK_CUDA(hipGetProcAddress(symbol, &entry_point, HIP_VERSION_MAJOR*100+HIP_VERSION_MINOR, 0, &driver_result));
  NVTE_CHECK(driver_result == HIP_GET_PROC_ADDRESS_SUCCESS,
             "Could not find CUDA driver entry point for ", symbol);
#else
=======
typedef cudaError_t (*VersionedGetEntryPoint)(const char *, void **, unsigned int,
                                              unsigned long long,  // NOLINT(*)
                                              cudaDriverEntryPointQueryResult *);
typedef cudaError_t (*GetEntryPoint)(const char *, void **, unsigned long long,  // NOLINT(*)
                                     cudaDriverEntryPointQueryResult *);

void *get_symbol(const char *symbol, int cuda_version) {
  constexpr char driver_entrypoint[] = "cudaGetDriverEntryPoint";
  constexpr char driver_entrypoint_versioned[] = "cudaGetDriverEntryPointByVersion";
  // We link to the libcudart.so already, so can search for it in the current context
  static GetEntryPoint driver_entrypoint_fun =
      reinterpret_cast<GetEntryPoint>(dlsym(RTLD_DEFAULT, driver_entrypoint));
  static VersionedGetEntryPoint driver_entrypoint_versioned_fun =
      reinterpret_cast<VersionedGetEntryPoint>(dlsym(RTLD_DEFAULT, driver_entrypoint_versioned));

>>>>>>> ca7407e
  cudaDriverEntryPointQueryResult driver_result;
  void *entry_point = nullptr;
  if (driver_entrypoint_versioned_fun != nullptr) {
    // Found versioned entrypoint function
    NVTE_CHECK_CUDA(driver_entrypoint_versioned_fun(symbol, &entry_point, cuda_version,
                                                    cudaEnableDefault, &driver_result));
  } else {
    NVTE_CHECK(driver_entrypoint_fun != nullptr, "Error finding the CUDA Runtime-Driver interop.");
    // Versioned entrypoint function not found
    NVTE_CHECK_CUDA(driver_entrypoint_fun(symbol, &entry_point, cudaEnableDefault, &driver_result));
  }
  NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
             "Could not find CUDA driver entry point for ", symbol);
#endif
  return entry_point;
}

}  // namespace cuda_driver

}  // namespace transformer_engine
