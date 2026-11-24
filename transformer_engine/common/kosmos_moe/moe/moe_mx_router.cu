/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the MXFP8 training kernels with the fused router and their launches, once per (MoeAct, MoeScore),
// kernel family MoeMxV (*_ks.cu), combine (MOE_MX_Q8: *_q8*.cu, the fp8 combines) and shared expert (MOE_MX_RSH:
// moe_mx_shared_router*.cu; MoeShGate: *_bwdg.cu, the gated backward only).
#include "moe_mx_kern.cuh"

#ifndef MOE_MX_Q8
#define MOE_MX_Q8 0
#endif
#ifndef MOE_MX_RSH
#define MOE_MX_RSH 0
#endif

namespace hk_moe {

namespace {

using MxrCall = std::conditional_t<MOE_MX_RSH, SingleCallSh, SingleCall>;
using MxrBwd  = std::conditional_t<MOE_MX_RSH, MxBwdCallSh, MxBwdCall>;

__global__ __launch_bounds__(NUM_THREADS, 2)
void moe_bwd_single_mx_router(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                              const SingleArgs *__restrict__ args, const MxrCall call, const SingleRouterCall rc,
                              const MxrBwd mxc) {
  constexpr bool ROUTER = true;
  constexpr bool SH     = MOE_MX_RSH;
  constexpr bool MX     = true;
  constexpr bool Q8     = MOE_MX_Q8;
  const unsigned long long *rcw = (const unsigned long long *)&rc;
  using ACT = MoeAct;
#define MOE_BWD_SH MOE_MX_RSH
#define MOE_BWD_KS (MOE_MX_V != 0)
#include "moe_bwd_single.inc"
#undef MOE_BWD_KS
#undef MOE_BWD_SH
}

template <int SC, int V, bool S, typename ACT, typename MT>
void mxr_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
             bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args,
             const RouterCall &rcall, const MT &mt, const ShCallMx *sc) {
  RouterCallS<SC> rc;
  static_cast<RouterCall &>(rc) = rcall;
  if constexpr (S) {
    mx_launch_fwd<ACT, V != 0>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, rc, mt, *sc);
  } else {
    mx_launch_fwd<ACT, V != 0>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, rc, mt);
  }
}

void mxr_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
             const SingleCall &call, const SingleShCall *sc, const SingleRouterCall &rc, const MxBwdCall &mc) {
#if MOE_MX_RSH
  moe_bwd_single_mx_router<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, SingleCallSh{call, *sc}, rc,
                                                        static_cast<const MxBwdCallSh &>(mc));
#else
  moe_bwd_single_mx_router<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, rc, mc);
#endif
}

}

#if MOE_MX_Q8
#if !MOE_SHARED_GATE
template <typename ACT, int SC, int V, bool S>
void moe_mx_router_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const RouterCall &rc, const MxTrainCall &mt,
                                 const ShCallMx *sc) {
  MxTrainCall8 m8;
  static_cast<MxTrainCall &>(m8) = mt;
  mxr_fwd<SC, V, S, ACT>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, rc, m8, sc);
}

template void moe_mx_router_launch_fwd_q8<MoeAct, MoeScore, MoeMxV, MOE_MX_RSH>(
    int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *, const bf16 *, bf16 *, bf16 *, const int *,
    const float *, int *, const MoeFwdArgs *, const RouterCall &, const MxTrainCall &, const ShCallMx *);
#endif

template <typename ACT, int SC, int V, bool S, bool G>
void moe_mx_router_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCall &call, const SingleShCall *sc,
                                 const SingleRouterCall &rc, const MxBwdCall &mc) {
  mxr_bwd(grid, s, A, tc, args, call, sc, rc, mc);
}

template void moe_mx_router_launch_bwd_q8<MoeAct, MoeScore, MoeMxV, MOE_MX_RSH, MoeShGate>(
    int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *, const SingleArgs *, const SingleCall &,
    const SingleShCall *, const SingleRouterCall &, const MxBwdCall &);
#else
#if !MOE_SHARED_GATE
template <typename ACT, int SC, int V, bool S>
void moe_mx_router_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const RouterCall &rc, const MxTrainCall &mt, const ShCallMx *sc,
                              bool q8) {
  if (q8) {
    moe_mx_router_launch_fwd_q8<ACT, SC, V, S>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, rc, mt, sc);
  } else {
    mxr_fwd<SC, V, S, ACT>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, rc, mt, sc);
  }
}

template void moe_mx_router_launch_fwd<MoeAct, MoeScore, MoeMxV, MOE_MX_RSH>(
    int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *, const bf16 *, bf16 *, bf16 *, const int *,
    const float *, int *, const MoeFwdArgs *, const RouterCall &, const MxTrainCall &, const ShCallMx *, bool);
#endif

template <typename ACT, int SC, int V, bool S, bool G>
void moe_mx_router_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCall &call, const SingleShCall *sc, const SingleRouterCall &rc,
                              const MxBwdCall &mc, bool q8) {
  if (q8) {
    moe_mx_router_launch_bwd_q8<ACT, SC, V, S, G>(grid, s, A, tc, args, call, sc, rc, mc);
  } else {
    mxr_bwd(grid, s, A, tc, args, call, sc, rc, mc);
  }
}

template void moe_mx_router_launch_bwd<MoeAct, MoeScore, MoeMxV, MOE_MX_RSH, MoeShGate>(
    int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *, const SingleArgs *, const SingleCall &,
    const SingleShCall *, const SingleRouterCall &, const MxBwdCall &, bool);
#endif

}
