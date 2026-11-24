/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the training API's kernels (kosmos_moe_train_forward / _backward; bf16, router-free, unshared, k in
// {6, 8, 10}; moe_train_api_ks.cu: k in {1, 2, 4}) and their launches, built once per MoeAct.
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
void moe_bwd_single_train_api(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                              const SingleArgs *__restrict__ args, const SingleCall call, const TrainApiBwdCall tbc) {
  constexpr bool ROUTER = false;
  constexpr bool SH     = false;
  constexpr bool MX     = false;
  constexpr bool Q8     = false;
  const unsigned long long *rcw = nullptr;
  const MxBwdCall mxc   = {};
  using ACT = MoeAct;
#define MOE_BWD_TRAIN_API 1
#include "moe_bwd_single.inc"
#undef MOE_BWD_TRAIN_API
}

}

template <typename ACT>
void moe_train_api_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                              const bf16 *W2, bf16 *out, const int *idx, const float *w, int *tc,
                              const MoeFwdArgs *args, const TrainApiCall &v) {
  switch (burst) {
    case 8:
      moe_fwd_single<8, false, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 7:
      moe_fwd_single<7, false, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 6:
      moe_fwd_single<6, false, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 0:
      moe_fwd_single<0, false, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    default:
      moe_fwd_single<4, false, 0, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
  }
}

template <typename ACT>
void moe_train_api_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                              const SingleArgs *args, const SingleCall &call, const TrainApiBwdCall &v) {
  moe_bwd_single_train_api<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, v);
}

template void moe_train_api_launch_fwd<MoeAct>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                               const bf16 *, bf16 *, const int *, const float *, int *,
                                               const MoeFwdArgs *, const TrainApiCall &);
template void moe_train_api_launch_bwd<MoeAct>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *,
                                               const SingleArgs *, const SingleCall &, const TrainApiBwdCall &);

}
