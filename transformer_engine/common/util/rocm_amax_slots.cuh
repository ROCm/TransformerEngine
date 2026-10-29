/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
//#include "hip/hip_runtime.h" // prevent hipification of this rocm_ file

// Atomics on one address serialize, so a kernel with many blocks can reduce its amax in two stages:
// each block raises one of ROCM_AMAX_SLOTS slots, and rocm_amax_finalize then folds the slots into
// the amax and zeroes them for the next kernel.

#include <cstddef>

constexpr int ROCM_AMAX_SLOTS = 64;
constexpr int ROCM_AMAX_SLOT_STRIDE = 32;  // floats: a 128-byte line per slot

// The address a block raises the amax at: its slot, or the amax itself without slots.
__device__ __forceinline__ float *rocm_amax_target(float *amax, float *slots, unsigned int block) {
  return slots != nullptr ? slots + (block % ROCM_AMAX_SLOTS) * ROCM_AMAX_SLOT_STRIDE : amax;
}

namespace transformer_engine {

// The zeroed slots for a kernel of num_blocks blocks on stream, or nullptr to raise the amax
// directly: for few blocks, with NVTE_USE_ATOMIC_AMAX=1, or while capturing a stream that has no
// slots yet.
float *rocm_amax_slots(size_t num_blocks, hipStream_t stream);

// Folds the slots into *amax and zeroes them, unless noop is set.
void rocm_amax_finalize(float *slots, float *amax, const float *noop, hipStream_t stream);

}  // namespace transformer_engine
