/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved. 
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "utils.h"

#include <cuda_runtime_api.h>
#ifndef USE_ROCM  // Disabled on ROCm
#include <cudnn_frontend_version.h>
#endif

#include <cassert>

#include "common/util/cuda_runtime.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion() {
  int ver = 0;
  NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&ver));
  return ver;
}

#ifndef USE_ROCM
size_t GetCudnnRuntimeVersion() { return cudnnGetVersion(); }
#endif

#ifndef USE_ROCM
size_t GetCudnnFrontendVersion() { return CUDNN_FRONTEND_VERSION; }
#endif

int GetDeviceComputeCapability(int gpu_id) { return transformer_engine::cuda::sm_arch(gpu_id); }

}  // namespace jax
}  // namespace transformer_engine
