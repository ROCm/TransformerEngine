/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the MXFP8 training kernels with the fp8 combines (fp8_combine) and launches, once per MoeAct and
// per kernel family MoeMxV (moe_mx_q8_ks.cu, moe_mx_q8_ks_pin.cu).
#include "moe_mx_kern.cuh"

namespace hk_moe {

// The fp8 combines: dtopkw reads the forward's MXFP8 Y rows, and dXp is pushed and folded as MXFP8 rows.
namespace {

__global__ __launch_bounds__(NUM_THREADS, 2)
#if MOE_MX_V == 2
void moe_bwd_single_mx_q8(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                          const SingleArgs *__restrict__ args, const SingleCall call, const MxBwdCall mxc,
                          const bf16 *__restrict__ pin_x, const float *__restrict__ pin_w) {
#else
void moe_bwd_single_mx_q8(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                          const SingleArgs *__restrict__ args, const SingleCall call, const MxBwdCall mxc) {
#endif
  constexpr bool ROUTER = false;
  constexpr bool SH     = false;
  constexpr bool MX     = true;
  constexpr bool Q8     = true;
  const unsigned long long *rcw = nullptr;
  using ACT = MoeAct;
#define MOE_BWD_KS (MOE_MX_V != 0)
#define MOE_BWD_PIN (MOE_MX_V == 2)
#include "moe_bwd_single.inc"
#undef MOE_BWD_PIN
#undef MOE_BWD_KS
}

}

template <typename ACT, int V>
void moe_mx_launch_fwd_q8(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                          const bf16 *W2, bf16 *out, bf16 *G, const int *idx, const float *w, int *tc,
                          const MoeFwdArgs *args, const MxTrainCall &mt) {
  MxTrainCall8 m8;
  static_cast<MxTrainCall &>(m8) = mt;
  mx_launch_fwd_v<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, m8);
}

template <typename ACT, int V>
void moe_mx_launch_bwd_q8(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                          const SingleCall &call, const MxBwdCall &mc, const float *pw) {
#if MOE_MX_V == 2
  moe_bwd_single_mx_q8<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc, mc.x, pw);
#else
  moe_bwd_single_mx_q8<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc);
#endif
}

template void moe_mx_launch_fwd_q8<MoeAct, MoeMxV>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                                   const bf16 *, bf16 *, bf16 *, const int *, const float *, int *,
                                                   const MoeFwdArgs *, const MxTrainCall &);
template void moe_mx_launch_bwd_q8<MoeAct, MoeMxV>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *,
                                                   const SingleArgs *, const SingleCall &, const MxBwdCall &,
                                                   const float *);

}
