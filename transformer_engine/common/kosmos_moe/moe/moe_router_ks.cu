/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the fused router's small-k (k in {1, 2, 4}) forward and backward kernels and their launches, once
// per (MoeAct, MoeScore).
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

namespace {

__global__ __launch_bounds__(NUM_THREADS, 2)
void moe_bwd_single_router_ks(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                              const SingleArgs *__restrict__ args, const SingleCall call, const SingleRouterCall rc) {
  constexpr bool ROUTER = true;
  constexpr bool SH     = false;
  constexpr bool MX     = false;
  constexpr bool Q8     = false;
  using ACT = MoeAct;
  const unsigned long long *rcw = (const unsigned long long *)&rc;
  const MxBwdCall mxc   = {};
#define MOE_BWD_KS 1
#include "moe_bwd_single.inc"
#undef MOE_BWD_KS
}

}

template <typename ACT, int SC>
void moe_router_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const RouterCall &rcall) {
  RouterCallS<SC> rc;
  static_cast<RouterCall &>(rc) = rcall;
  switch (burst) {
    case 8:
      moe_fwd_single<8, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, rc);
      break;
    case 7:
      moe_fwd_single<7, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, rc);
      break;
    case 6:
      moe_fwd_single<6, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, rc);
      break;
    case 0:
      moe_fwd_single<0, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, rc);
      break;
    default:
      moe_fwd_single<4, true, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, G, idx, w, tc, args, rc);
      break;
  }
}

template <typename ACT, int SC>
void moe_router_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCall &call, const SingleRouterCall &rc) {
  moe_bwd_single_router_ks<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, rc);
}

template void moe_router_launch_fwd_ks<MoeAct, MoeScore>(int, int, hipStream_t, const FsGl &, const bf16 *,
                                                         const bf16 *, const bf16 *, bf16 *, bf16 *, const int *,
                                                         const float *, int *, const MoeFwdArgs *, const RouterCall &);
template void moe_router_launch_bwd_ks<MoeAct, MoeScore>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *,
                                                         const SingleArgs *, const SingleCall &,
                                                         const SingleRouterCall &);

}
