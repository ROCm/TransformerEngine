/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MXFP8 kernels' includes and forward launch, shared by the bf16-combine and fp8-combine translation units.
#pragma once

#include "moe_fwd_dev.cuh"
#include "moe_router.cuh"
#include "moe_comm_kernels.cuh"
#include "moe_gemm_nn.cuh"
#include "moe_gemm_nt.cuh"
#include "moe_leg_a.cuh"
#include "moe_mx_train.cuh"
#include "moe_bwd_single.cuh"
#include "moe_plan.cuh"

namespace hk_moe {

// MoeMxV: the translation unit's kernel family, 0: k in {6, 8, 10} (moe_mx.cu, moe_mx_q8.cu), 1: the small-k kernels,
// k in {1, 2, 4} (moe_mx_ks.cu, moe_mx_q8_ks.cu), 2: probs on input, k = 1 (moe_mx_ks_pin.cu, moe_mx_q8_ks_pin.cu).
#ifndef MOE_MX_V
#define MOE_MX_V 0
#endif
constexpr int MoeMxV = MOE_MX_V;

template <typename ACT, bool KS, typename... MC>
void mx_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
                   bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args,
                   const MC &...mc) {
  switch (burst) {
    case 8:
      moe_fwd_single<8, KS, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, mc...);
      break;
    case 7:
      moe_fwd_single<7, KS, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, mc...);
      break;
    case 6:
      moe_fwd_single<6, KS, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, mc...);
      break;
    case 0:
      moe_fwd_single<0, KS, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, mc...);
      break;
    default:
      moe_fwd_single<4, KS, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, mc...);
      break;
  }
}

// Kernel family V's forward with the call records mc (and, probs on input, the PinCall after them).
template <typename ACT, int V, typename... MC>
void mx_launch_fwd_v(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
                     bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args,
                     const MC &...mc) {
  if constexpr (V == 2) {
    mx_launch_fwd<ACT, true>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, mc..., PinCall());
  } else {
    mx_launch_fwd<ACT, V != 0>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, mc...);
  }
}

}
