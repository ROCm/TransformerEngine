/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Unified host launcher for HipKittens blockwise FP8 GEMM. Pure host code
// (no HipKittens headers) so it can see both the cdna3 (gfx942) and cdna4
// (gfx950) implementations, which are built against different HipKittens
// submodule branches. Dispatch is by runtime GPU arch.
//
// When the cdna3/cdna4 HipKittens branches are eventually merged upstream,
// the two impls can collapse into one and this launcher becomes a thin
// pass-through (or disappears) without touching the rocm_gemm.cu call site.

#include <hip/hip_runtime.h>
#include <cstring>
#include "blockwise_fp8_gemm.h"
#include "cdna3/blockwise_fp8_gemm.h"
#include "cdna4/blockwise_fp8_gemm.h"

namespace {
bool current_device_is_gfx950() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return std::strstr(prop.gcnArchName, "gfx950") != nullptr;
}
}  // namespace

void kittens_blockwise_fp8_gemm(
    const void *A, const void *B, void *C,
    const void *scale_A, const void *scale_B,
    int M, int N, int K,
    bool transa, bool transb,
    int a_dtype, int b_dtype,
    int a_scaling_mode, int b_scaling_mode,
    int out_dtype,
    const void *bias, int bias_dtype,
    const void *gelu_aux, int gelu_aux_dtype,
    const void *c_in, float beta,
    hipStream_t stream) {
    if (current_device_is_gfx950()) {
        // CDNA4 impl supports a limited path (e4m3 x e4m3, bf16 out, TN, no
        // bias/gelu/beta, fixed square shapes). Returns false when unsupported;
        // fall through to the CDNA3 impl as a last resort.
        bool handled = kittens_blockwise_fp8_gemm_impl_cdna4(
            A, B, C, scale_A, scale_B, M, N, K, stream);
        if (handled) return;
    }

    kittens_blockwise_fp8_gemm_impl_cdna3(
        A, B, C, scale_A, scale_B, M, N, K, transa, transb,
        a_dtype, b_dtype, a_scaling_mode, b_scaling_mode, out_dtype,
        bias, bias_dtype, gelu_aux, gelu_aux_dtype, c_in, beta, stream);
}
