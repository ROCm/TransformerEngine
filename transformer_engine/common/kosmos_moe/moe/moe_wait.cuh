/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The kernels' waits on other ranks (spin loops and idle passes), bounded in time.
#pragma once

#include "hip/hip_runtime.h"

namespace hk_moe {

// Waits on other ranks are bounded: a wait past MOE_WAIT_LIMIT_S (a rank that never launched its call, or died in it)
// traps: the kernel fails (a GPU exception; the process aborts, or the next HIP call returns the error) instead of
// hanging every rank. No call of the kernels runs close to it. Time: s_memrealtime (100 MHz) >> 20, ~10.5 ms units.
#ifndef MOE_WAIT_LIMIT_S
#define MOE_WAIT_LIMIT_S 600
#endif
constexpr unsigned MOE_WAIT_LIMIT_UNITS = (unsigned)(((unsigned long long)MOE_WAIT_LIMIT_S * 100000000ull) >> 20);

// A wait's clock in its counter n: n >= 0 not started; n < 0 started at the low 31 bits of n.
__device__ __forceinline__ void moe_wait_check(int &n) {
  const unsigned now = (unsigned)(__builtin_amdgcn_s_memrealtime() >> 20) & 0x7fffffffu;
  if (n >= 0) {
    n = (int)(0x80000000u | now);
  } else if (((now - ((unsigned)n & 0x7fffffffu)) & 0x7fffffffu) > MOE_WAIT_LIMIT_UNITS) {
    // s_trap 2 (what __builtin_trap emits) as inline asm: the builtin also ends the block as unreachable, which
    // trips clang 24 ("illegal VGPR to SGPR copy") in the MXFP8 shared-expert backward kernels.
    asm volatile("s_trap 2");
  }
}

__device__ __forceinline__ void spin_backoff(int &n) {
  if (n >= 0 && n < 6) {
    __builtin_amdgcn_s_sleep(1);
    n++;
  } else {
    moe_wait_check(n);
    __builtin_amdgcn_s_sleep(8);
  }
}

// An idle pass of a work loop that waits on other ranks: sleeps S, bounded as spin_backoff (n: the loop's clock).
template <int S>
__device__ __forceinline__ void idle_sleep(int &n) {
  moe_wait_check(n);
  __builtin_amdgcn_s_sleep(S);
}

}  // namespace hk_moe
