/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the MXFP8 training kernels with the shared expert and their launches, once per MoeAct, kernel family
// MoeMxV (moe_mx_shared_ks.cu, moe_mx_shared_ks_pin.cu), combine (MOE_MX_Q8: moe_mx_shared_q8*.cu, the fp8 combines)
// and MoeShGate (moe_mx_shared_bwdg.cu, moe_mx_shared_q8_bwdg.cu: the gated backward only).
#include "moe_mx_kern.cuh"

#ifndef MOE_MX_Q8
#define MOE_MX_Q8 0
#endif

namespace hk_moe {

namespace {

__global__ __launch_bounds__(NUM_THREADS, 2)
#if MOE_MX_V == 2
void moe_bwd_single_mx_sh(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                          const SingleArgs *__restrict__ args, const SingleCallSh call, const MxBwdCallSh mxc,
                          const bf16 *__restrict__ pin_x, const float *__restrict__ pin_w) {
#else
void moe_bwd_single_mx_sh(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                          const SingleArgs *__restrict__ args, const SingleCallSh call, const MxBwdCallSh mxc) {
#endif
  constexpr bool ROUTER = false;
  constexpr bool SH     = true;
  constexpr bool MX     = true;
  constexpr bool Q8     = MOE_MX_Q8;
  const unsigned long long *rcw = nullptr;
  using ACT = MoeAct;
#define MOE_BWD_SH 1
#define MOE_BWD_KS (MOE_MX_V != 0)
#define MOE_BWD_PIN (MOE_MX_V == 2)
#include "moe_bwd_single.inc"
#undef MOE_BWD_PIN
#undef MOE_BWD_KS
#undef MOE_BWD_SH
}

}

#if MOE_MX_Q8
#if !MOE_SHARED_GATE
template <typename ACT, int V>
void moe_mx_shared_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const MxTrainCall &mt, const ShCallMx &sc) {
  MxTrainCall8 m8;
  static_cast<MxTrainCall &>(m8) = mt;
  mx_launch_fwd_v<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, m8, sc);
}

template void moe_mx_shared_launch_fwd_q8<MoeAct, MoeMxV>(int, int, hipStream_t, const FsGl &, const bf16 *,
                                                          const bf16 *, const bf16 *, bf16 *, bf16 *, const int *,
                                                          const float *, int *, const MoeFwdArgs *, const MxTrainCall &,
                                                          const ShCallMx &);
#endif

template <typename ACT, int V, bool G>
void moe_mx_shared_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCallSh &call, const MxBwdCallSh &mc,
                                 const float *pw) {
#if MOE_MX_V == 2
  moe_bwd_single_mx_sh<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc, mc.x, pw);
#else
  moe_bwd_single_mx_sh<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc);
#endif
}

template void moe_mx_shared_launch_bwd_q8<MoeAct, MoeMxV, MoeShGate>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &,
                                                                     int *, const SingleArgs *, const SingleCallSh &,
                                                                     const MxBwdCallSh &, const float *);
#else
#if !MOE_SHARED_GATE
template <typename ACT, int V>
void moe_mx_shared_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const MxTrainCall &mt, const ShCallMx &sc, bool q8) {
  if (q8) {
    moe_mx_shared_launch_fwd_q8<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, mt, sc);
  } else {
    mx_launch_fwd_v<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, mt, sc);
  }
}

template void moe_mx_shared_launch_fwd<MoeAct, MoeMxV>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                                       const bf16 *, bf16 *, bf16 *, const int *, const float *, int *,
                                                       const MoeFwdArgs *, const MxTrainCall &, const ShCallMx &, bool);
#endif

// pw: probs on input's slot weights (V 2).
template <typename ACT, int V, bool G>
void moe_mx_shared_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                              const SingleCallSh &call, const MxBwdCallSh &mc, bool q8, const float *pw) {
  if (q8) {
    moe_mx_shared_launch_bwd_q8<ACT, V, G>(grid, s, A, tc, args, call, mc, pw);
  } else {
#if MOE_MX_V == 2
    moe_bwd_single_mx_sh<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc, mc.x, pw);
#else
    moe_bwd_single_mx_sh<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc);
#endif
  }
}

template void moe_mx_shared_launch_bwd<MoeAct, MoeMxV, MoeShGate>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &,
                                                                  int *, const SingleArgs *, const SingleCallSh &,
                                                                  const MxBwdCallSh &, bool, const float *);
#endif

}
