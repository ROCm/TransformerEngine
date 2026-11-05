/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <map>
#include <mutex>
#include <utility>

#include "../common.h"
#include "cuda_runtime.h"
#include "logging.h"
#include "rocm_amax_slots.cuh"
#include "rocm_device_utils.cuh"
#include "system.h"

namespace transformer_engine {
namespace {

constexpr int FINALIZE_THREADS = 32;

__global__ void __launch_bounds__(FINALIZE_THREADS)
    rocm_amax_finalize_kernel(float *slots, float *amax, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  // The maximum of the bits as integers, like the atomics take it
  int amax_bits = 0;
  for (int i = threadIdx.x; i < ROCM_AMAX_SLOTS; i += FINALIZE_THREADS) {
    float *const slot = slots + i * ROCM_AMAX_SLOT_STRIDE;
    amax_bits = max(amax_bits, __float_as_int(*slot));
    *slot = 0.0f;
  }
  amax_bits = rocm_subwarp_allreduce<FINALIZE_THREADS>(
      amax_bits, [](const int a, const int b) { return max(a, b); });
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<int *>(amax), amax_bits);
  }
}

}  // namespace

float *rocm_amax_slots(const size_t num_blocks, hipStream_t stream) {
  static const bool enabled = !getenv<bool>("NVTE_USE_ATOMIC_AMAX");
  if (!enabled || num_blocks <= ROCM_AMAX_SLOTS) {
    return nullptr;
  }
  // One buffer per stream, so that concurrent kernels never share slots, kept for the graphs
  // captured with it. Leaked deliberately: device memory must not be freed at static teardown.
  static std::mutex &mutex = *new std::mutex;
  static auto &buffers = *new std::map<std::pair<int, hipStream_t>, float *>;
  std::lock_guard<std::mutex> lock(mutex);
  float *&slots = buffers[{cuda::current_device(), stream}];
  if (slots == nullptr) {
    hipStreamCaptureStatus capture_status;
    NVTE_CHECK_CUDA(hipStreamIsCapturing(stream, &capture_status));
    if (capture_status != hipStreamCaptureStatusNone) {
      return nullptr;
    }
    constexpr size_t bytes = ROCM_AMAX_SLOTS * ROCM_AMAX_SLOT_STRIDE * sizeof(float);
    NVTE_CHECK_CUDA(hipMalloc(&slots, bytes));
    NVTE_CHECK_CUDA(hipMemsetAsync(slots, 0, bytes, stream));
  }
  return slots;
}

void rocm_amax_finalize(float *slots, float *amax, const float *noop, hipStream_t stream) {
  rocm_amax_finalize_kernel<<<1, FINALIZE_THREADS, 0, stream>>>(slots, amax, noop);
  NVTE_CHECK_CUDA(hipGetLastError());
}

}  // namespace transformer_engine
