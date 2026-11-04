/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstdint>
#include "gemm.h"

namespace te_kittens::gfx1250 {

constexpr int TILE           = 256;
constexpr int K_BLOCK        = 128;
constexpr int REDUCE_THREADS = 256;

typedef float v8f __attribute__((ext_vector_type(8)));
template <typename T> using vec8 = T __attribute__((ext_vector_type(8)));

// D = alpha * (sum of the split partials in split order) (+ beta * D), eight rows per thread.
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

template <typename OutT, bool ACC>
void launch_reduce(const Launch &l) {
    const size_t mn     = size_t(l.M) * l.N;
    const size_t blocks = std::min<size_t>((mn / 8 + REDUCE_THREADS - 1) / REDUCE_THREADS, 65536);
    const auto kernel = splitk_reduce_kernel<OutT, ACC>;
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(REDUCE_THREADS), 0, l.stream,
                       static_cast<const float *>(l.workspace), static_cast<OutT *>(l.D), l.alpha,
                       l.beta, mn, l.splits);
    kittens_check(hipGetLastError(), "gfx1250 splitk_reduce_kernel");
}

void splitk_reduce(const Launch &l) {
    if (l.out_dtype == KITTENS_FLOAT32) {
        l.acc ? launch_reduce<float, true>(l) : launch_reduce<float, false>(l);
    } else if (l.out_dtype == KITTENS_FLOAT16) {
        l.acc ? launch_reduce<_Float16, true>(l) : launch_reduce<_Float16, false>(l);
    } else {
        l.acc ? launch_reduce<__bf16, true>(l) : launch_reduce<__bf16, false>(l);
    }
}

/* Split-K factor S for fp32 partials of S whole k_block slices, which take M % 8 == 0.
 * NVTE_HK_SPLITK forces S when legal; auto picks S in {1,2,4,8,16} minimising the one-wave cost
 * K/S + a + b*tiles*S/cus. */
static int splitk_pick(const GemmArgs &args, size_t workspace_size, int k_block) {
    constexpr double a = 512;
    constexpr double b = 2048;
    const int64_t tiles = int64_t((args.M + TILE - 1) / TILE) * ((args.N + TILE - 1) / TILE);
    auto fits = [&](int S) {
        return S <= args.K / k_block && args.M % 8 == 0 &&
               size_t(args.M) * args.N * S * sizeof(float) <= workspace_size;
    };
    if (args.splitk != 0) {
        return args.splitk > 1 && fits(args.splitk) ? args.splitk : 1;
    }
    int best = 1;
    double best_t = args.K;
    for (int S = 2; S <= 16 && tiles * S <= args.cu_count; S *= 2) {
        const double t = double(args.K) / S + a + b * double(tiles * S) / args.cu_count;
        if (fits(S) && t < best_t) {
            best   = S;
            best_t = t;
        }
    }
    return best;
}

class GemmGfx1250 final : public GemmBackend {
 public:
    bool gemm(const GemmArgs &args) override {
        const bool mxfp8 = args.mxfp8_compact && kittens_is_fp8(args.a_dtype) &&
                           kittens_is_fp8(args.b_dtype);
        const bool bf16  = args.a_dtype == args.b_dtype &&
                           (args.a_dtype == KITTENS_BFLOAT16 || args.a_dtype == KITTENS_FLOAT16);
        if (!(mxfp8 || bf16) || (args.transa && args.transb)) {
            return false;
        }
        // MXFP8: whole scale blocks of K, and DWord rows of an M- or N-contiguous operand.
        if (args.M <= 0 || args.N <= 0 || args.K <= 0 ||
            (mxfp8 && (args.M % 4 != 0 || args.N % 4 != 0 || args.K % 32 != 0))) {
            return false;
        }
        auto aligned = [](const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
        if (!aligned(args.A) || !aligned(args.B) || !aligned(args.C) || !aligned(args.scale_A) ||
            !aligned(args.scale_B) || !aligned(args.bias) || !aligned(args.aux_gelu)) {
            return false;
        }

        // Bias and aux share one dtype, which the instance lists match.
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
        const bool acc = args.beta != 0.f;
        const int side_dtype = bias ? args.bias_dtype : aux ? args.aux_dtype : args.out_dtype;
        // TN stores a 16-bit D without epilogue as accumulated, with no alpha (no training path).
        if (args.transa && epi == NONE && !acc && args.out_dtype != KITTENS_FLOAT32 &&
            args.alpha != 1.f) {
            return false;
        }

        // Split partials start 256 B aligned. A slice holds whole K-blocks, 256 for MXFP8 or 128;
        // a K % k_block head joins slice 0.
        const uintptr_t ws_raw = reinterpret_cast<uintptr_t>(args.workspace);
        const uintptr_t ws     = (ws_raw + 255) & ~uintptr_t(255);
        const size_t pad     = ws - ws_raw;
        const size_t ws_size = args.workspace_size > pad ? args.workspace_size - pad : 0;
        const int k_block = mxfp8 ? 2 * K_BLOCK : K_BLOCK;
        const int splits  = epi == NONE ? splitk_pick(args, ws_size, k_block) : 1;

        const Launch l{args.A, args.B, args.scale_A, args.scale_B, args.C, args.bias, args.aux_gelu,
                       reinterpret_cast<void *>(ws), args.alpha, args.beta, args.M, args.N, args.K,
                       splits, args.a_dtype, args.b_dtype, args.out_dtype, side_dtype, epi, acc,
                       use_mcast(), args.stream};
        using Launcher = bool (*)(const Launch &);
        static constexpr Launcher launchers[2][3] = {
            {launch_bf16<TN>, launch_bf16<NN>, launch_bf16<NT>},
            {launch_mxfp8<TN>, launch_mxfp8<NN>, launch_mxfp8<NT>},
        };
        const Layout layout = args.transa ? TN : args.transb ? NT : NN;
        return launchers[mxfp8][layout](l);
    }

    // No gfx1250 grouped kernels: grouped GEMMs take CK or the per-expert path.
    bool grouped_gemm(const MXFP8GroupedGemmArgs &) override { return false; }
    bool grouped_wgrad(const MXFP8WgradArgs &) override { return false; }
    bool grouped16_gemm(const Grouped16GemmArgs &, const char **why) override {
        *why = "no gfx1250 BF16/FP16 grouped kernel";
        return false;
    }
};

}  // namespace te_kittens::gfx1250

GemmBackend *GemmBackend::get_gfx1250() {
    static te_kittens::gfx1250::GemmGfx1250 impl;
    return &impl;
}
