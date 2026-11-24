/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the probs-on-input (k = 1) kernels without the router or the shared expert, and their launches,
// built once per MoeAct.
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

// The clamped build's kernel in a namespace of its own (an anonymous one changes the kernel's code).
#if MOE_ACT_CLAMP10
namespace c10 {
#endif

__global__ __launch_bounds__(NUM_THREADS, 2)
void moe_bwd_single_ks_pin(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                           const SingleArgs *__restrict__ args, const SingleCall call, const bf16 *__restrict__ pin_x,
                           const float *__restrict__ pin_w) {
  constexpr bool ROUTER = false;
  constexpr bool SH     = false;
  constexpr bool MX     = false;
  constexpr bool Q8     = false;
  using ACT = MoeAct;
  const unsigned long long *rcw = nullptr;
  const MxBwdCall mxc = {};
#define MOE_BWD_KS 1
#define MOE_BWD_PIN 1
#include "moe_bwd_single.inc"
#undef MOE_BWD_PIN
#undef MOE_BWD_KS
}

#if MOE_ACT_CLAMP10
}
using c10::moe_bwd_single_ks_pin;
#endif

template <typename ACT>
void moe_ks_pin_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                           const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                           const MoeFwdArgs *args) {
  const PinCall p;
  switch (burst) {
    case 8:
      moe_fwd_single<8, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, p);
      break;
    case 7:
      moe_fwd_single<7, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, p);
      break;
    case 6:
      moe_fwd_single<6, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, p);
      break;
    case 0:
      moe_fwd_single<0, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, p);
      break;
    default:
      moe_fwd_single<4, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, p);
      break;
  }
}

template <typename ACT>
void moe_ks_pin_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                           const SingleCall &call, const bf16 *x, const float *w) {
  moe_bwd_single_ks_pin<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, x, w);
}

template void moe_ks_pin_launch_fwd<MoeAct>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                            const bf16 *, bf16 *, bf16 *, const int *, const float *, int *,
                                            const MoeFwdArgs *);
template void moe_ks_pin_launch_bwd<MoeAct>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *, const SingleArgs *,
                                            const SingleCall &, const bf16 *, const float *);

}
