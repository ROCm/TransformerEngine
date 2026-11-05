/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "hip/hip_runtime.h"
#include <stdexcept>
#include <string>
#include "../kittens_common.h"

namespace te_kittens::cdna5 {

enum Epi { NONE, BIAS, GELU_AUX, GELU_AUX_BIAS, DGELU, BGRADB, SPLIT };

constexpr int BLOCK_M       = 256;
constexpr int BLOCK_N       = 256;
constexpr int NUM_THREADS   = 256;
constexpr int BF16_BLOCK_K  = 128;
constexpr int MXFP8_BLOCK_K = 256;
constexpr int LDS_MAX_BYTES = 320 * 1024;

typedef int v4i __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

#ifdef NVTE_HK_GFX1250_MCAST
inline bool use_mcast() { return NVTE_HK_GFX1250_MCAST != 0; }
#else
inline bool use_mcast() {
    return transformer_engine::cuda::sm_arch_name().find("-strict") == std::string::npos;
}
#endif

#ifdef __gfx1250_strict__
constexpr bool ARCH_MCAST = false;
#else
constexpr bool ARCH_MCAST = true;
#endif

template <bool TRANSA, bool TRANSB>
bool launch_mxfp8(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);
template <bool TRANSA, bool TRANSB>
bool launch_bf16(const GemmArgs &args, void *workspace, int splits, Epi epi, bool mcast);

void splitk_reduce(const GemmArgs &args, const void *workspace, int splits);

inline void kittens_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

inline int side_dtype(const GemmArgs &args) {
    if (args.bias != nullptr) {
        return args.bias_dtype;
    }
    return args.aux_gelu != nullptr ? args.aux_dtype : args.out_dtype;
}

}  // namespace te_kittens::cdna5
