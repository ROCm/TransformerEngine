/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"

#include <cstddef>

#define KITTENS_BOOL_SWITCH(val, NAME, ...) \
    if (val) { constexpr bool NAME = true; __VA_ARGS__ } \
    else { constexpr bool NAME = false; __VA_ARGS__ }

static inline size_t kittens_align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

namespace te_kittens::blockwise {

// dtype codes are NVTEDType: 6 = bfloat16, 5 = float16, anything else = float32
__device__ inline float read_elem(const void *p, int dtype, int idx) {
    if (dtype == 6) return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(p)[idx]);
    if (dtype == 5) return __half2float(reinterpret_cast<const __half *>(p)[idx]);
    return reinterpret_cast<const float *>(p)[idx];
}

enum struct GemmEpilogue {
    DEFAULT,
    BIAS,
    GELU_AUX,
    BETA,
    BIAS_BETA,
    GELU_AUX_BETA,
};

__host__ __device__ inline constexpr bool epilogue_has_bias(GemmEpilogue e) {
    return e == GemmEpilogue::BIAS || e == GemmEpilogue::BIAS_BETA;
}
__host__ __device__ inline constexpr bool epilogue_has_gelu(GemmEpilogue e) {
    return e == GemmEpilogue::GELU_AUX || e == GemmEpilogue::GELU_AUX_BETA;
}
__host__ __device__ inline constexpr bool epilogue_has_beta(GemmEpilogue e) {
    return e == GemmEpilogue::BETA || e == GemmEpilogue::BIAS_BETA
        || e == GemmEpilogue::GELU_AUX_BETA;
}

inline GemmEpilogue select_epilogue(bool has_bias, bool has_gelu, bool has_beta) {
    if (has_gelu) return has_beta ? GemmEpilogue::GELU_AUX_BETA : GemmEpilogue::GELU_AUX;
    if (has_bias) return has_beta ? GemmEpilogue::BIAS_BETA     : GemmEpilogue::BIAS;
    return has_beta ? GemmEpilogue::BETA : GemmEpilogue::DEFAULT;
}

}  // namespace te_kittens::blockwise
