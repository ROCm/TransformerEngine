/*************************************************************************
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#pragma once

#include <hip/hip_runtime.h>

#if !defined(__HIP_DEVICE_COMPILE__)
/* Platforms that have both MI300 family and other families GPUs are unknown and not supported.
* Thus, FP8 format is selected once by the current (any) GPU architecture.
*/
#include <optional>
#include "../util/string.h"
static bool _te_check_fp8_fnuz() {
  hipDeviceProp_t prop;
  hipError_t res= hipGetDeviceProperties(&prop, 0);
  if (res != hipSuccess) {
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

#if defined(__HIP_DEVICE_COMPILE__) && HIP_VERSION < 60300000
static constexpr inline bool te_fp8_fnuz() { return true; }
#endif

#if HIP_VERSION >= 60200000
#include <hip/hip_fp8.h>

#if HIP_VERSION >= 60300000
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

#else //HIP_VERSION >= 60300000
typedef __hip_fp8_e4m3_fnuz _te_hip_fp8_e4m3;
typedef __hip_fp8_e5m2_fnuz _te_hip_fp8_e5m2;
#endif //HIP_VERSION >= 60300000

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

#else //HIP_VERSION >= 60200000

// FP8 header version 0.3, 2021/05/11

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_DEVICE  __device__
#define HIP_HOST __host__ 

#define E5M2_AMAX 57344.0
#define E4M3_AMAX 240.0

namespace hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
HIP_HOST_DEVICE
uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE
T cast_from_f8(uint8_t x);

} // namespace hip_f8_impl

#include "hip_f8_impl.h"

enum class hip_f8_type {
  bf8 = 0, // 1:5:2
  fp8 = 1  // 1:4:3
};


enum class hip_f8_rounding_mode {
  standard,
  stochastic
};


// bias mode bit implementation
//
// For MI100 simulation purpose, we keep a copy of it on the host and device
// (MI300 HW implementation will be different)
//
// The bias mode should only be accessed via its get/set routines.
// The set routine sets both copies to the same value, keeping them in sync
// The get routine will return the device copy for device functions and
// the host copy for host functions
//
// "bias mode optimial"
//    => "bias mode bit" = 1
//    => bias = 16 for 152, 8 for 143
//    => NAN/INF are represented as negative_zero
//
// "bias mode ieee"
//    => "bias mode bit" = 0
//    => bias = 15 for 152, 7 for 143
//    => NAN/INF are represented as per IEEE conventions

#ifndef __HIPCC_RTC__
static bool hip_f8_bias_mode_bit_host = true;

static inline __host__ bool get_hip_f8_bias_mode() {
  return hip_f8_bias_mode_bit_host;
}
#endif // __HIPCC_RTC__

static __device__ bool hip_f8_bias_mode_bit_device = true;

static inline __device__ bool get_hip_f8_bias_mode() {
  return hip_f8_bias_mode_bit_device;
}

#ifndef __HIPCC_RTC__
static __global__ void set_hip_f8_bias_mode_bit(bool v) {
  hip_f8_bias_mode_bit_device = v;
}

static void set_hip_f8_bias_mode_ieee() {
  hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, false);
  hip_f8_bias_mode_bit_host = false;
}

static void set_hip_f8_bias_mode_optimal() {
  hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, true);
  hip_f8_bias_mode_bit_host = true;
}
#endif // __HIPCC_RTC__


template<hip_f8_type T>
struct hip_f8 {
  uint8_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8() = default;

  // constructor from bits
  explicit HIP_HOST_DEVICE hip_f8(uint8_t v) {
    data = v;
  }

  // constructor from float
#ifdef __gfx942__
  explicit HIP_DEVICE hip_f8(float v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    union {
      float fval;
      uint32_t i32val;
      uint8_t i8val[4];
    } val;
    uint32_t ival = 0;
    val.fval = v;

    if (T == hip_f8_type::bf8) { // bf8
      if ((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
	val.fval = __builtin_amdgcn_fmed3f(val.fval, E5M2_AMAX, -E5M2_AMAX);
      if (rm == hip_f8_rounding_mode::standard) { // RNE rounding
	ival = __builtin_amdgcn_cvt_pk_bf8_f32(
	    val.fval, val.fval, ival, false); // false -> WORD0
	val.i32val = ival;
	data     = val.i8val[0];
      }
      else { //stochastic rounding
	ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
	val.i32val = ival;
	data     = val.i8val[0]; // little endian
      }
    } 
    else { // fp8
      if ((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
	val.fval = __builtin_amdgcn_fmed3f(val.fval, E4M3_AMAX, -E4M3_AMAX);
      if (rm == hip_f8_rounding_mode::standard) { // RNE rounding
	ival = __builtin_amdgcn_cvt_pk_fp8_f32(
	    val.fval, val.fval, ival, false); // false -> WORD0
	val.i32val = ival;
	data     = val.i8val[0];
      }
      else { //stochastic rounding
	ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
	val.i32val = ival;
	data     = val.i8val[0]; // little endian
      }
    }
  }

#ifndef __HIPCC_RTC__
  explicit HIP_HOST //Code host still uses SW simulated conversion on gfx942
  hip_f8(float v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }
#endif
#else // #ifndef __gfx942__
  explicit HIP_HOST_DEVICE // On architectures other than gfx942, both host and device still use SW simulated conversion
  hip_f8(float v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }
#endif // #ifdef __gfx942__
			 
  // constructor from half
#ifdef __gfx942__
  explicit HIP_DEVICE hip_f8(half v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) 
	  : hip_f8((float)v, rm, rng)
  {
  }
#ifndef __HIPCC_RTC__
  explicit HIP_HOST //Code host still uses SW simulated conversion on gfx942
  hip_f8(half v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }
#endif
#else // #ifndef __gfx942__
  explicit HIP_HOST_DEVICE // On architectures other than gfx942, both host and device still use SW simulated conversion
  hip_f8(half v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }
#endif // #ifdef __gfx942__

  // constructor from hip_bfloat16
  explicit HIP_HOST_DEVICE hip_f8(hip_bfloat16 v, hip_f8_rounding_mode r=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float
#ifdef __gfx942__
  HIP_DEVICE operator float() const {
    union
    {
      float    fval;
      uint32_t i32val;
      uint8_t  i8val[4]; // dependent of endian
    } val;

   // assign 8bit data in position [7:0]
   val.i32val   = 0;
   val.i8val[3] = data; // little endian

   // upcast
   if(T == hip_f8_type::bf8)
     val.fval = __builtin_amdgcn_cvt_f32_bf8(val.i32val, 3); // 0 pos
   else // fp8
     val.fval = __builtin_amdgcn_cvt_f32_fp8(val.i32val, 3); // 0 pos

   return val.fval;
  }
#ifndef __HIPCC_RTC__
  explicit inline HIP_HOST //Code host still uses SW simulated conversion on gfx942
  operator float() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, float, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, float, false/*negative_zero_nan*/>(data);
      }
    }
  }
#endif
#else // #ifdef __gfx942__
  explicit inline HIP_HOST_DEVICE // On architectures other than gfx942, both host and device still use SW simulated conversion
  operator float() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, float, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, float, false/*negative_zero_nan*/>(data);
      }
    }
  }
#endif // #ifdef __gfx942__

  // convert to half
#ifdef __gfx942__
  explicit HIP_DEVICE inline operator half() const {
    return __half(float(*this));
  }
#ifndef __HIPCC_RTC__
  explicit inline HIP_HOST //Code host still uses SW simulated conversion on gfx942
  operator half() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, half, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, half, false/*negative_zero_nan*/>(data);
      }
    }
  }
#endif
#else // #ifndef __gfx942__
  explicit inline HIP_HOST_DEVICE // On architectures other than gfx942, both host and device still use SW simulated conversion
  operator half() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, half, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, half, false/*negative_zero_nan*/>(data);
      }
    }
  }
#endif // #ifdef __gfx942__

  // convert to hip_bfloat16
  explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const;

  // check for zero
  inline HIP_HOST_DEVICE bool is_zero() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x00;
    } else {
      return (data == 0x00) || (data == 0x80);
    }
  }
  
  // check for nan
  inline HIP_HOST_DEVICE bool is_nan() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x80;
    } else {
      if (T == hip_f8_type::bf8) {
	return
	  (data == 0x7d) || (data == 0x7e) || (data == 0x7f) ||
	  (data == 0xfd) || (data == 0xfe) || (data == 0xff);
      } else {
	return
	  (data == 0x79) || (data == 0x7a) || (data == 0x7b) || (data == 0x7c) || (data == 0x7d) || (data == 0x7e) || (data == 0x7f) ||
	  (data == 0xf9) || (data == 0xfa) || (data == 0xfb) || (data == 0xfc) || (data == 0xfd) || (data == 0xfe) || (data == 0xff);
      }
    }
  }
  
  // check for inf
  inline HIP_HOST_DEVICE bool is_inf() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x80;
    } else {
      if (T == hip_f8_type::bf8) {
	return (data == 0x7c) || (data == 0xfc);
      } else {
	return (data == 0x78) || (data == 0xf8);
      }
    }
  }
};


template<hip_f8_type T>
struct hip_f8x4 {
  // define some convenience types
  typedef float float32x2 __attribute__((ext_vector_type(2)));
  typedef float float32x4 __attribute__((ext_vector_type(4)));

  typedef _Float16 halfx2 __attribute__((ext_vector_type(2)));
  typedef _Float16 halfx4 __attribute__((ext_vector_type(4)));

  typedef uint16_t hip_bfloat16x2 __attribute__((ext_vector_type(2)));
  typedef uint16_t hip_bfloat16x4 __attribute__((ext_vector_type(4)));

  uint32_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x4() = default;

  // constructor from bits
  HIP_HOST_DEVICE hip_f8x4(uint32_t v);

  // constructor from float
  HIP_HOST_DEVICE hip_f8x4(float v0, float v1=0, float v2=0, float v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from half
  HIP_HOST_DEVICE hip_f8x4(half v0, half v1=0, half v2=0, half v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from hip_bfloat16
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16 v0, hip_bfloat16 v1=hip_bfloat16(0.0f), hip_bfloat16 v2=hip_bfloat16(0.0f), hip_bfloat16 v3=hip_bfloat16(0.0f), hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float32x4
  inline HIP_HOST_DEVICE operator float32x4() const;

  // convert to halfx4
  inline HIP_HOST_DEVICE operator halfx4() const;

  // convert to hip_bfloat16x4
  inline HIP_HOST_DEVICE operator hip_bfloat16x4() const;
};



template<hip_f8_type T>
struct hip_f8x8 {
  // define some convenience types
  typedef hip_f8x4<T>  f8x8 __attribute__((ext_vector_type(2)));

  f8x8 data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x8() = default;

  // do we need to define other constructors or any conversion routines here?
};

// If we do not end up needing either any constructors or conversion routines for the above type, then
// we can simplify the above type to the following
#if USE_SIMPLER_HIP_F8x8
template <hip_f8_type T>
using hip_f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));
#endif

typedef float hip_float32x4  __attribute__((ext_vector_type(4)));
typedef float hip_float32x16 __attribute__((ext_vector_type(16)));

// these are device-specific and we don't expect them to exist unless we're compiling with hip-clang for gfx942.
template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x4 mfma_f32_16x16x32(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x4 c);

template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x16 mfma_f32_32x32x16(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x16 c);


typedef hip_f8<hip_f8_type::fp8> te_hip_fp8_e4m3;
typedef hip_f8<hip_f8_type::bf8> te_hip_fp8_e5m2;

#endif //HIP_VERSION >= 60200000

#else //__HIPCC__
typedef struct {char storage;} te_hip_fp8_e4m3;
typedef struct {char storage;} te_hip_fp8_e5m2;
#endif //__HIPCC__
