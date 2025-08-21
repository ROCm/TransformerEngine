/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved. 
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>
#include <transformer_engine/fused_attn.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/util/logging.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion();
size_t GetCudnnRuntimeVersion();
int GetDeviceComputeCapability(int gpu_id);

<<<<<<< HEAD:transformer_engine/jax/csrc/utils.h
#ifndef USE_ROCM
void PopulateRngStateAsync(void *rng_state_dst, const void *const seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream);
#else
void PopulateRngStateAsync(void *rng_state_dst, 
                           const void *const seed,
                           size_t batch_size, 
                           size_t num_heads, 
                           size_t q_max_seqlen, 
                           size_t kv_max_seqlen,
                           cudaStream_t stream);
#endif

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t len, cudaStream_t stream);

=======
>>>>>>> 42b51c40c4e39adce9640cf98f8a3f5869f5f270:transformer_engine/jax/csrc/extensions/utils.h
class cudaDevicePropertiesManager {
 public:
  static cudaDevicePropertiesManager &Instance() {
    static thread_local cudaDevicePropertiesManager instance;
    return instance;
  }

  int GetMultiProcessorCount() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      (void)cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.multiProcessorCount;
  }

  int GetMajor() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      (void)cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.major;
  }

 private:
  bool prop_queried_ = false;
  cudaDeviceProp prop_;
};

}  // namespace jax
}  // namespace transformer_engine
