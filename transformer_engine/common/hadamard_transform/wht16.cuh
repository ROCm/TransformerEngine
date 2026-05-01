/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Shared 16-point Walsh-Hadamard transform primitives for AMDGPU.

#ifndef TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_WHT16_CUH_
#define TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_WHT16_CUH_

#include "hip/hip_runtime.h"

#ifdef __HIP_PLATFORM_AMD__

static constexpr int kHadamardDim     = 16;
static constexpr int kWarpSize        = 64;
static constexpr int kThreadsPerWHT   = 4;
static constexpr int kElemsPerThread  = 4;
static constexpr int kRowsPerWarp     = kWarpSize / kThreadsPerWHT;   // 16
static constexpr int kWarpsPerBlock   = 4;
static constexpr int kRowsPerBlock    = kRowsPerWarp * kWarpsPerBlock; // 64
static constexpr int kThreadsPerBlock = kWarpSize   * kWarpsPerBlock;  // 256
static constexpr float kHadamardScale = 0.25f;

// ds_swizzle: sub-wavefront exchange without LDS.
__device__ __forceinline__ float ds_swizzle_xor1(float v) {
    return __shfl_xor(v, 1);
}

__device__ __forceinline__ float ds_swizzle_xor2(float v) {
    return __shfl_xor(v, 2);
}

// -----------------------------------------------------------------------
// 16-point WHT via the Kronecker trick  (no shared memory)
// -----------------------------------------------------------------------
//
// 1. The vec operator
//    vec() flattens a matrix into a column vector by stacking its
//    columns one on top of the other:
//
//        X = |a  c|         vec(X) = |a|
//            |b  d|                  |b|
//                                    |c|
//                                    |d|
//
// 2. The "Kronecker trick" for 1D -> 2D
//    The fundamental identity that connects these concepts is:
//
//        vec(B . X . A^T) = (A (x) B) . vec(X)
//
//    For a 16-point Hadamard transform (H16 = H4 (x) H4),
//    set A = H4 and B = H4.  The formula becomes:
//
//        H16 . x = vec(H4 . X . H4^T)
//
// 3. Data layout  (column-major, one column per thread)
//    Reshape the 16-element 1D vector x into a 4x4 matrix X
//    by filling columns first:
//
//        X = | x0   x4   x8   x12 |    thread 0 holds col 0: v0..v3 = x0 ..x3
//            | x1   x5   x9   x13 |    thread 1 holds col 1: v0..v3 = x4 ..x7
//            | x2   x6   x10  x14 |    thread 2 holds col 2: v0..v3 = x8 ..x11
//            | x3   x7   x11  x15 |    thread 3 holds col 3: v0..v3 = x12..x15
//
// 4. Three-stage computation
//    Stage 1 (local H4)   : left-multiply   H4 . X    (within each thread)
//    Stage 2 (xor-1 swap) : \                         (across 4 threads)
//    Stage 3 (xor-2 swap) : / right-multiply . H4^T   together these two butterfly stages = H4^T
//
//    Result: vec(H4 . X . H4^T) = H16 . x
//
// 5. Randomised Hadamard Transform (RHT)
//    A diagonal sign matrix D (from sign_mask) is applied either
//    before the WHT (apply_pre=true, forward) or after (inverse).
//
// Adapted from cast_transpose_mxfp4_kernel_shuffled.cu::hadamard16_inplace,
// extended with NV random_sign_mask (uint16_t bitmask).
// thread_in_group [0,3]: drives ds_swizzle polarity (identical to MLPerf tid & 3).
// apply_pre=true -> D before WHT (forward); false -> D after WHT (inverse).
__device__ __forceinline__ void wht16(
        float& v0, float& v1, float& v2, float& v3,
        int thread_in_group, uint16_t sign_mask, bool apply_pre) {
    auto sgn = [&](int k) -> float {
        return ((sign_mask >> (thread_in_group * kElemsPerThread + k)) & 1u) ? -1.f : 1.f;
    };

    if (apply_pre) {
      v0*=sgn(0); v1*=sgn(1); v2*=sgn(2); v3*=sgn(3);
    }

    // Stage 1: local H4
    float a0=v0+v1, a1=v0-v1, a2=v2+v3, a3=v2-v3;
    v0=a0+a2; v2=a0-a2; v1=a1+a3; v3=a1-a3;

    // Stage 2: cross-thread XOR-1
    { float p0=ds_swizzle_xor1(v0), p1=ds_swizzle_xor1(v1),
            p2=ds_swizzle_xor1(v2), p3=ds_swizzle_xor1(v3);
      bool up=(thread_in_group&1);
      v0=up?(p0-v0):(p0+v0); v1=up?(p1-v1):(p1+v1);
      v2=up?(p2-v2):(p2+v2); v3=up?(p3-v3):(p3+v3); }

    // Stage 3: cross-thread XOR-2
    { float p0=ds_swizzle_xor2(v0), p1=ds_swizzle_xor2(v1),
            p2=ds_swizzle_xor2(v2), p3=ds_swizzle_xor2(v3);
      bool up=(thread_in_group>>1)&1;
      v0=up?(p0-v0):(p0+v0); v1=up?(p1-v1):(p1+v1);
      v2=up?(p2-v2):(p2+v2); v3=up?(p3-v3):(p3+v3); }

    v0*=kHadamardScale; v1*=kHadamardScale; v2*=kHadamardScale; v3*=kHadamardScale;

    if (!apply_pre) {
      v0*=sgn(0); v1*=sgn(1); v2*=sgn(2); v3*=sgn(3);
    }
}

#endif  // __HIP_PLATFORM_AMD__

#endif  // TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_WHT16_CUH_
