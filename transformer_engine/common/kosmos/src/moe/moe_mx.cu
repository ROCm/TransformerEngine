/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// libkosmos_moe.a: the MXFP8 forward (inference and training) and backward kernels and their launches, once per
// MoeAct and per kernel family MoeMxV (moe_mx_ks.cu, moe_mx_ks_pin.cu); the weight quantization in the SwiGLU build of
// k in {6, 8, 10}.
#include "moe_mx_kern.cuh"

namespace hk_moe {

namespace {

__global__ __launch_bounds__(NUM_THREADS, 2)
#if MOE_MX_V == 2
void moe_bwd_single_mx(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                       const SingleArgs *__restrict__ args, const SingleCall call, const MxBwdCall mxc,
                       const bf16 *__restrict__ pin_x, const float *__restrict__ pin_w) {
#else
void moe_bwd_single_mx(const gl<bf16, 1, 1, -1, -1> A0, int *__restrict__ tile_counter,
                       const SingleArgs *__restrict__ args, const SingleCall call, const MxBwdCall mxc) {
#endif
  constexpr bool ROUTER = false;
  constexpr bool SH     = false;
  constexpr bool MX     = true;
  constexpr bool Q8     = false;
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
void moe_mx_launch_fwd(int burst, int grid, hipStream_t s, const FsGl &A, const bf16 *X, const bf16 *W1, const bf16 *W2,
                       bf16 *out, bf16 *G, const int *idx, const float *w, int *tc, const MoeFwdArgs *args,
                       const MxCall *mc, const MxTrainCall *mt, bool q8) {
  if (mt && q8) {
    moe_mx_launch_fwd_q8<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, *mt);
  } else if (mt) {
    mx_launch_fwd_v<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, *mt);
  } else {
    mx_launch_fwd_v<ACT, V>(burst, grid, s, A, X, W1, W2, out, G, idx, w, tc, args, *mc);
  }
}

#if !MOE_ACT_CLAMP10 && !MOE_MX_V
// One thread per 32-element block (32 contiguous elements, or with COL 32 rows of one column of a [rows][cols] matrix).
template <bool COL>
__global__ void moe_mx_quant_weights(const uint16_t *__restrict__ w, int rows, int cols, long long nblk,
                                     uint8_t *__restrict__ q, uint8_t *__restrict__ s) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nblk) {
    return;
  }
  long long o = 32 * i, st = 1;
  if constexpr (COL) {
    const long long per = (long long)rows / 32 * cols;
    o = i / per * rows * cols + i % per / cols * 32 * cols + i % cols, st = cols;
  }
  uint16_t v[32];
  uint32_t amax = 0;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    v[j] = w[o + j * st];
    amax = max(amax, (uint32_t)(v[j] & 0x7fff));
  }
  const uint32_t sc = mx_sc2(amax) & 0xffff;
  const float f = mx_scf(sc);
#pragma unroll
  for (int j = 0; j < 32; j += 4) {
    const uint32_t r = mx_cvt4(v[j] | (uint32_t)v[j + 1] << 16, v[j + 2] | (uint32_t)v[j + 3] << 16, f, f);
    if constexpr (COL) {
#pragma unroll
      for (int b = 0; b < 4; b++) {
        q[o + (j + b) * st] = (uint8_t)(r >> 8 * b);
      }
    } else {
      *(uint32_t *)(q + o + j) = r;
    }
  }
  s[i] = (uint8_t)sc;
}

void moe_mx_launch_quant(const void *w, int nmat, int rows, int cols, bool col, void *q, void *s, hipStream_t st) {
  const long long nblk = (long long)nmat * rows * cols / 32;
  const unsigned grid = (unsigned)((nblk + 255) / 256);
  if (col) {
    moe_mx_quant_weights<true><<<grid, 256, 0, st>>>((const uint16_t *)w, rows, cols, nblk, (uint8_t *)q, (uint8_t *)s);
  } else {
    moe_mx_quant_weights<false><<<grid, 256, 0, st>>>((const uint16_t *)w, rows, cols, nblk, (uint8_t *)q,
                                                      (uint8_t *)s);
  }
}

#endif

// pw: probs on input's slot weights (V 2).
template <typename ACT, int V>
void moe_mx_launch_bwd(int grid, hipStream_t s, const gl<bf16, 1, 1, -1, -1> &A, int *tc, const SingleArgs *args,
                       const SingleCall &call, const MxBwdCall &mc, bool q8, const float *pw) {
  if (q8) {
    moe_mx_launch_bwd_q8<ACT, V>(grid, s, A, tc, args, call, mc, pw);
  } else {
#if MOE_MX_V == 2
    moe_bwd_single_mx<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc, mc.x, pw);
#else
    moe_bwd_single_mx<<<grid, NUM_THREADS, 0, s>>>(A, tc, args, call, mc);
#endif
  }
}

#if !MOE_MX_V
template <typename ACT>
const void *moe_mx_bwd_kernel() {
  return (const void *)moe_bwd_single_mx;
}

#endif

template void moe_mx_launch_fwd<MoeAct, MoeMxV>(int, int, hipStream_t, const FsGl &, const bf16 *, const bf16 *,
                                                const bf16 *, bf16 *, bf16 *, const int *, const float *, int *,
                                                const MoeFwdArgs *, const MxCall *, const MxTrainCall *, bool);
template void moe_mx_launch_bwd<MoeAct, MoeMxV>(int, hipStream_t, const gl<bf16, 1, 1, -1, -1> &, int *,
                                                const SingleArgs *, const SingleCall &, const MxBwdCall &, bool,
                                                const float *);
#if !MOE_MX_V
template const void *moe_mx_bwd_kernel<MoeAct>();
#endif

}
