/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the training API's small-k kernels (k in {1, 2, 4}; bf16, router-free, unshared) and their launches,
// built once per MoeAct.
#include "moe_fwd_dev.cuh"
#include "moe_comm_kernels.cuh"
#include "moe_gemm_nn.cuh"
#include "moe_gemm_nt.cuh"
#include "moe_leg_a.cuh"
#include "moe_bwd_single.cuh"
#include "moe_plan.cuh"

namespace hk_moe {

namespace {

__global__ __launch_bounds__(NUM_THREADS, 2)
void moe_bwd_single_ks_train_api(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                                 const SingleArgs *__restrict__ args, const SingleCall call,
                                 const TrainApiBwdCall tbc) {
  using ACT = MoeAct;
#define MOE_BWD_KS 1
#define MOE_BWD_TRAIN_API 1
#include "moe_bwd_single.inc"
#undef MOE_BWD_TRAIN_API
#undef MOE_BWD_KS
}

}

template <typename ACT>
void moe_train_api_launch_fwd_ks(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1,
                                 const bf16 *W2, bf16 *out, const int *idx, const float *w, int *tc,
                                 const MoeFwdArgs *args, const TrainApiCall &v) {
  switch (burst) {
    case 8:
      moe_fwd_single<8, true, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 7:
      moe_fwd_single<7, true, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 6:
      moe_fwd_single<6, true, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    case 0:
      moe_fwd_single<0, true, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
    default:
      moe_fwd_single<4, true, ACT><<<grid, NUM_THREADS, 0, s>>>(A, X, W1, W2, out, nullptr, idx, w, tc, args, v);
      break;
  }
}

template <typename ACT>
void moe_train_api_launch_bwd_ks(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc,
                                 const SingleArgs *args, const SingleCall &call, const TrainApiBwdCall &v) {
  moe_bwd_single_ks_train_api<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, v);
}

template void moe_train_api_launch_fwd_ks<MoeAct>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                                  const bf16 *, bf16 *, const int *, const float *, int *,
                                                  const MoeFwdArgs *, const TrainApiCall &);
template void moe_train_api_launch_bwd_ks<MoeAct>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *,
                                                  const SingleArgs *, const SingleCall &, const TrainApiBwdCall &);

}
