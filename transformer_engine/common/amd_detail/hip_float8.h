/*************************************************************************
 * Copyright (c) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once
#include <hip/hip_runtime.h>

#if !defined(__HIP_DEVICE_COMPILE__)
/* Platforms that have both MI300 family and other families GPUs are unknown and not supported.
* Thus, FP8 format is selected once by the current (any) GPU architecture.
*/
#include <iostream>
#include <optional>
#include "../util/string.h"
static bool _te_check_fp8_fnuz() {
  hipDeviceProp_t prop;
  hipError_t res= hipGetDeviceProperties(&prop, 0);
  if (res != hipSuccess) {
    if (res == hipErrorNoDevice) {
      // no device, default to OCP
      std::cerr << "No HIP device found, defaulting to OCP format for FP8\n";
      return false;
    }
    //TODO: better error out system
    throw std::runtime_error(transformer_engine::concat_strings(
      "hipGetDeviceProperties failed with error: ", hipGetErrorString(res)));
  }
  return prop.major == 9 && prop.minor == 4;
}

static inline bool te_fp8_fnuz() {
  static std::optional<bool> use_fnuz;
  if (!use_fnuz.has_value()) {
    use_fnuz = _te_check_fp8_fnuz();
  }
  return use_fnuz.value();
}
#endif //__HIP_DEVICE_COMPILE__

#ifdef __HIPCC__

#include <hip/hip_version.h> //For RTC it should be included explicitly
#include <hip/hip_fp8.h>

#if !defined(__HIP_DEVICE_COMPILE__)

/* Device methods in _te_hip_fp8 are dummy and are needed for compilation
* because HIPCC compiles __device__ and __global__ functions for host.
* The results are discarded so those methods are declared but not defined
*/
template<typename FNUZ, typename OCP>
union _te_hip_fp8 {
  FNUZ fnuz;
  OCP ocp;
  __host__ __device__ _te_hip_fp8<FNUZ, OCP>() = default;

  __host__ operator float() const {
    return te_fp8_fnuz() ? fnuz.operator float() : ocp.operator float();
  }
  __device__ operator float() const;

  __host__ _te_hip_fp8<FNUZ, OCP>(const float& v) {
    if (te_fp8_fnuz()) fnuz=v; else ocp=v;
  }
  __device__ _te_hip_fp8<FNUZ, OCP>(const float& v);
};

typedef _te_hip_fp8<__hip_fp8_e4m3_fnuz, __hip_fp8_e4m3> _te_hip_fp8_e4m3;
typedef _te_hip_fp8<__hip_fp8_e5m2_fnuz, __hip_fp8_e5m2> _te_hip_fp8_e5m2;

#elif HIP_FP8_TYPE_FNUZ
typedef __hip_fp8_e4m3_fnuz _te_hip_fp8_e4m3;
typedef __hip_fp8_e5m2_fnuz _te_hip_fp8_e5m2;
static constexpr inline bool te_fp8_fnuz() { return true; }
#elif HIP_FP8_TYPE_OCP
typedef __hip_fp8_e4m3 _te_hip_fp8_e4m3;
typedef __hip_fp8_e5m2 _te_hip_fp8_e5m2;
static constexpr inline bool te_fp8_fnuz() { return false; }
#else
#error "Unsupported HIP_FP8_TYPE"
#endif //__HIP_DEVICE_COMPILE__

struct te_hip_fp8_e4m3 {  
  _te_hip_fp8_e4m3 data;

  __host__ __device__ te_hip_fp8_e4m3() = default;

  __host__ __device__ operator float() const { return data.operator float(); }

  __host__ __device__ te_hip_fp8_e4m3(const float& v): data(v) {}
};
static_assert(sizeof(te_hip_fp8_e4m3) == 1, "Size mismatch");

union te_hip_fp8_e5m2 {
  _te_hip_fp8_e5m2 data;

  __host__ __device__ te_hip_fp8_e5m2() = default;

  __host__ __device__ operator float() const { return data.operator float(); }

  __host__ __device__ te_hip_fp8_e5m2(const float& v) { data = v; }
};
static_assert(sizeof(te_hip_fp8_e5m2) == 1, "Size mismatch");

// Unsigned E5M3 (bias 15, max 114688, 0xFF = NaN, no Inf), the encoding of CUDA's __nv_fp8_ue5m3.
// Converted in software: gfx1250 A0 and gfx1250-strict have no E5M3 conversion instructions.
struct te_hip_fp8_ue5m3 {
  unsigned char data;

  __host__ __device__ te_hip_fp8_ue5m3() = default;

  // Round to nearest even and saturate to the max finite value; negative values encode as zero.
  __host__ __device__ te_hip_fp8_ue5m3(const float& v) {
    const unsigned int bits = __builtin_bit_cast(unsigned int, v);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
      data = 0xFF;
    } else if (!(v > 0.0f)) {
      data = 0;
    } else if (v >= 114688.0f) {
      data = 0xFE;
    } else if (v < 0x1p-14f) {
      data = static_cast<unsigned char>(__builtin_rintf(v * 0x1p17f));
    } else {
      // Rounding the FP32 mantissa to 3 bits may carry into the exponent, which is the right code.
      data = static_cast<unsigned char>(((bits + 0x7FFFFu + ((bits >> 20) & 1u)) >> 20) - (112u << 3));
    }
  }

  __host__ __device__ operator float() const {
    const unsigned int e = data >> 3;
    const unsigned int m = data & 7u;
    if (data == 0xFF) {
      return __builtin_nanf("");
    }
    return e == 0 ? static_cast<float>(m) * 0x1p-17f
                  : __builtin_bit_cast(float, ((e + 112u) << 23) | (m << 20));
  }
};
static_assert(sizeof(te_hip_fp8_ue5m3) == 1, "Size mismatch");

#else //__HIPCC__
typedef struct {char storage;} te_hip_fp8_e4m3;
typedef struct {char storage;} te_hip_fp8_e5m2;
typedef struct {char storage;} te_hip_fp8_ue5m3;
#endif //__HIPCC__
