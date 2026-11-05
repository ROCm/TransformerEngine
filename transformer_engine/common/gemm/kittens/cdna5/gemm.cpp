/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include "gemm.h"

namespace te_kittens::cdna5 {

constexpr int REDUCE_THREADS    = 256;
constexpr int REDUCE_MAX_BLOCKS = 65536;
constexpr int WS_ALIGN          = 256;

template <typename OutT, bool ACC>
__global__ void __launch_bounds__(REDUCE_THREADS)
splitk_reduce_kernel(const float *__restrict__ ws, OutT *__restrict__ D, float alpha, float beta,
                     size_t mn, int splits) {
    for (size_t i = (size_t(blockIdx.x) * REDUCE_THREADS + threadIdx.x) * 8; i < mn;
         i += size_t(gridDim.x) * REDUCE_THREADS * 8) {
        v8f v = *reinterpret_cast<const v8f *>(ws + i);
        for (int z = 1; z < splits; z++) {
            v += *reinterpret_cast<const v8f *>(ws + z * mn + i);
        }
        v *= alpha;
        if constexpr (ACC) {
            v += beta * __builtin_convertvector(*reinterpret_cast<const vec8<OutT> *>(D + i), v8f);
        }
        *reinterpret_cast<vec8<OutT> *>(D + i) = __builtin_convertvector(v, vec8<OutT>);
    }
}

template <typename OutT>
static void launch_reduce(const GemmArgs &args, const void *workspace, int splits) {
    const size_t mn     = size_t(args.M) * args.N;
    const size_t blocks = std::min<size_t>((mn / 8 + REDUCE_THREADS - 1) / REDUCE_THREADS, REDUCE_MAX_BLOCKS);
    const auto kernel = args.beta != 0.f ? splitk_reduce_kernel<OutT, true> : splitk_reduce_kernel<OutT, false>;
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(REDUCE_THREADS), 0, args.stream,
                       static_cast<const float *>(workspace), static_cast<OutT *>(args.C), args.alpha, args.beta,
                       mn, splits);
    kittens_check(hipGetLastError(), "gfx1250 splitk_reduce_kernel");
}

void splitk_reduce(const GemmArgs &args, const void *workspace, int splits) {
    if (args.out_dtype == KITTENS_FLOAT32) {
        launch_reduce<float>(args, workspace, splits);
    } else if (args.out_dtype == KITTENS_FLOAT16) {
        launch_reduce<_Float16>(args, workspace, splits);
    } else {
        launch_reduce<__bf16>(args, workspace, splits);
    }
}

static bool aligned16(const void *p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

class GemmCdna5 final : public GemmBackend {
 public:
    bool gemm(const GemmArgs &args) override {
        const bool mxfp8 = args.mxfp8_compact;
        const bool bf16  = args.a_dtype == args.b_dtype &&
                           (args.a_dtype == KITTENS_BFLOAT16 || args.a_dtype == KITTENS_FLOAT16);
        if (!(mxfp8 || bf16) || (args.transa && args.transb)) {
            return false;
        }
        if (args.M <= 0 || args.N <= 0 || args.K <= 0 ||
            (mxfp8 && (args.M % 4 != 0 || args.N % 4 != 0 || args.K % 32 != 0))) {
            return false;
        }
        if (!aligned16(args.A) || !aligned16(args.B) || !aligned16(args.C) || !aligned16(args.scale_A) ||
            !aligned16(args.scale_B) || !aligned16(args.bias) || !aligned16(args.aux_gelu)) {
            return false;
        }

        const bool bias = args.bias != nullptr;
        const bool aux  = args.aux_gelu != nullptr;
        if ((bias && aux && args.bias_dtype != args.aux_dtype) || (args.grad && bias && aux)) {
            return false;
        }
        Epi epi = NONE;
        if (args.grad) {
            epi = bias ? BGRADB : aux ? DGELU : NONE;
        } else if (bias || aux) {
            epi = !aux ? BIAS : bias ? GELU_AUX_BIAS : GELU_AUX;
        }
        if (args.transa && epi == NONE && args.beta == 0.f && args.out_dtype != KITTENS_FLOAT32 &&
            args.alpha != 1.f) {
            return false;
        }

        const uintptr_t ws_raw = reinterpret_cast<uintptr_t>(args.workspace);
        const uintptr_t ws     = (ws_raw + WS_ALIGN - 1) & ~uintptr_t(WS_ALIGN - 1);
        const size_t pad     = ws - ws_raw;
        const size_t ws_size = args.workspace_size > pad ? args.workspace_size - pad : 0;
        const int k_block = mxfp8 ? MXFP8_BLOCK_K : BF16_BLOCK_K;
        const int64_t tiles = int64_t(ceil_div(args.M, BLOCK_M)) * ceil_div(args.N, BLOCK_N);
        const size_t split_bytes = size_t(args.M) * args.N * sizeof(float);
        int splits = 1;
        if (epi == NONE && args.M % 8 == 0) {
            if (args.splitk > 1 && args.splitk <= args.K / k_block && split_bytes * args.splitk <= ws_size) {
                splits = args.splitk;
            }
            double best_t = args.K;
            for (int s = 2; args.splitk == 0 && s <= 16 && tiles * s <= args.cu_count; s *= 2) {
                const double t = double(args.K) / s + 512 + 2048 * double(tiles * s) / args.cu_count;
                if (s <= args.K / k_block && split_bytes * s <= ws_size && t < best_t) {
                    splits = s;
                    best_t = t;
                }
            }
        }

        void *workspace = reinterpret_cast<void *>(ws);
        const bool mcast = use_mcast();
        if (args.transa && !args.transb) {
            return mxfp8 ? launch_mxfp8<true, false>(args, workspace, splits, epi, mcast)
                         : launch_bf16<true, false>(args, workspace, splits, epi, mcast);
        }
        if (!args.transa && args.transb) {
            return mxfp8 ? launch_mxfp8<false, true>(args, workspace, splits, epi, mcast)
                         : launch_bf16<false, true>(args, workspace, splits, epi, mcast);
        }
        return mxfp8 ? launch_mxfp8<false, false>(args, workspace, splits, epi, mcast)
                     : launch_bf16<false, false>(args, workspace, splits, epi, mcast);
    }

    bool grouped_gemm(const MXFP8GroupedGemmArgs &) override { return false; }
    bool grouped_wgrad(const MXFP8WgradArgs &) override { return false; }
};

}  // namespace te_kittens::cdna5

GemmBackend *GemmBackend::get_cdna5() {
    static te_kittens::cdna5::GemmCdna5 impl;
    return &impl;
}
