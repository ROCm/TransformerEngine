/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Shared 16-point Walsh-Hadamard transform primitives for AMDGPU.

#ifndef TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_WHT16_CUH_
#define TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_WHT16_CUH_

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
    float r;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
                 "s_waitcnt lgkmcnt(0)" : "=v"(r) : "v"(v));
    return r;
}

__device__ __forceinline__ float ds_swizzle_xor2(float v) {
    float r;
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
                 "s_waitcnt lgkmcnt(0)" : "=v"(r) : "v"(v));
    return r;
}

// 16-point WHT: in-register, no shared memory.
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
