/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/
// MXFP8 tile K-loop: operand layouts a_col, b_col and tiles A8, B8 from the includer, which also defines MX_DMA_A,
// MX_DMA_B, MX_SCALE and MX_MMA (undefined here).
#define MX_LD_A(P, H) mx_load<(2 * (P) + (H)) * SZ8, MxStT<a_col>>(a, al)
#define MX_LD_B(R, P, H) mx_load<(2 * (P) + (H)) * SZ8, MxStT<b_col>>(R, bl)
#define MX_LDT_A(P, H) mx_load_now<a_col, (2 * (P) + (H)) * SZ8, MxStT<a_col>>(a, a_t0, warp_m * REG_M)
#define MX_LDT_B(R, P, H) mx_load_now<b_col, (2 * (P) + (H)) * SZ8, MxStT<b_col>>(R, b_t0, warp_n * REG_N)
#define MX_SA(P, H) mx_scales<4, a_col>(scl[P], (H) * HALF_ROW + warp_m * REG_M)
#define MX_SB(P, H) mx_scales<2, b_col>(scl[P] + BLOCK_ROW, (H) * HALF_COL + warp_n * REG_N)
#define MX_PHASE(C, B, SA, SB)         \
  do {                                 \
    __builtin_amdgcn_s_setprio(2);     \
    MX_MMA(C, B, SA, SB);              \
    __builtin_amdgcn_s_setprio(0);     \
    __builtin_amdgcn_s_barrier();      \
    __builtin_amdgcn_sched_barrier(0); \
  } while (0)
#define MX_STEP(T, O, KK)                 \
  do {                                    \
    MX_SCALE((KK) + 1, O);                \
    MX_LD_B(b0, T, 0);                    \
    MX_LD_A(T, 0);                        \
    MX_DMA_A(O, 1, (KK) + 1);             \
    asm volatile("s_waitcnt lgkmcnt(8)"); \
    __builtin_amdgcn_s_barrier();         \
                                          \
    const fp8e8m0_4 sa_h0 = MX_SA(T, 0);  \
    const fp8e8m0_4 sb_h0 = MX_SB(T, 0);  \
    MX_PHASE(cA, b0, sa_h0, sb_h0);       \
                                          \
    const fp8e8m0_4 sb_h1 = MX_SB(T, 1);  \
    MX_LD_B(b1, T, 1);                    \
    MX_DMA_A(T, 0, (KK) + 2);             \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __builtin_amdgcn_s_barrier();         \
                                          \
    MX_PHASE(cB, b1, sa_h0, sb_h1);       \
                                          \
    const fp8e8m0_4 sa_h1 = MX_SA(T, 1);  \
    MX_LD_A(T, 1);                        \
    MX_DMA_B(T, 0, (KK) + 2);             \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __builtin_amdgcn_s_barrier();         \
                                          \
    MX_PHASE(cC, b0, sa_h1, sb_h0);       \
                                          \
    MX_DMA_B(T, 1, (KK) + 2);             \
    asm volatile("s_waitcnt vmcnt(6)");   \
    __builtin_amdgcn_s_barrier();         \
                                          \
    MX_PHASE(cD, b1, sa_h1, sb_h1);       \
  } while (0)
#define MX_TAIL(T, O, KK)                   \
  do {                                      \
    {                                       \
      MX_SCALE((KK) + 1, O);                \
      asm volatile("s_waitcnt vmcnt(0)");   \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      __builtin_amdgcn_s_barrier();         \
      const fp8e8m0_4 sa_h0 = MX_SA(T, 0);  \
      const fp8e8m0_4 sb_h0 = MX_SB(T, 0);  \
                                            \
      MX_LDT_B(b0, T, 0);                   \
      MX_LDT_A(T, 0);                       \
      MX_DMA_A(O, 1, (KK) + 1);             \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cA, b0, sa_h0, sb_h0);       \
                                            \
      const fp8e8m0_4 sb_h1 = MX_SB(T, 1);  \
      MX_LDT_B(b1, T, 1);                   \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cB, b1, sa_h0, sb_h1);       \
                                            \
      const fp8e8m0_4 sa_h1 = MX_SA(T, 1);  \
      MX_LDT_A(T, 1);                       \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cC, b0, sa_h1, sb_h0);       \
                                            \
      MX_LDT_B(b0, O, 0);                   \
      asm volatile("s_waitcnt vmcnt(4)");   \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cD, b1, sa_h1, sb_h1);       \
    }                                       \
    {                                       \
      asm volatile("s_waitcnt vmcnt(0)");   \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      __builtin_amdgcn_s_barrier();         \
      const fp8e8m0_4 sa_h0 = MX_SA(O, 0);  \
      const fp8e8m0_4 sb_h0 = MX_SB(O, 0);  \
                                            \
      MX_LDT_A(O, 0);                       \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cA, b0, sa_h0, sb_h0);       \
                                            \
      const fp8e8m0_4 sb_h1 = MX_SB(O, 1);  \
      MX_LDT_B(b1, O, 1);                   \
      asm volatile("s_waitcnt vmcnt(0)");   \
      __builtin_amdgcn_s_barrier();         \
      __builtin_amdgcn_sched_barrier(0);    \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      MX_PHASE(cB, b1, sa_h0, sb_h1);       \
                                            \
      const fp8e8m0_4 sa_h1 = MX_SA(O, 1);  \
      MX_LDT_A(O, 1);                       \
      __builtin_amdgcn_s_barrier();         \
                                            \
      asm volatile("s_waitcnt lgkmcnt(0)"); \
      __builtin_amdgcn_s_setprio(2);        \
      MX_MMA(cC, b0, sa_h1, sb_h0);         \
      MX_MMA(cD, b1, sa_h1, sb_h1);         \
      __builtin_amdgcn_s_setprio(0);        \
      __builtin_amdgcn_s_barrier();         \
      __builtin_amdgcn_sched_barrier(0);    \
    }                                       \
  } while (0)

      MxRtT<a_col, REG_M> a;
      MxRtT<b_col, REG_N> b0, b1;
      const uint32_t a_t0 = (uint32_t)(uintptr_t)&A8[0][0].data[0], b_t0 = (uint32_t)(uintptr_t)&B8[0][0].data[0];
      const auto al = mx_lanes<a_col, MxStT<a_col>, decltype(a)>(a_t0, warp_m * REG_M);
      const auto bl = mx_lanes<b_col, MxStT<b_col>, decltype(b0)>(b_t0, warp_n * REG_N);
      // The prologue's buffer is the step count's parity, so the tail always starts in buffer 0.
      const int n_steps = k_iters - 2, p0 = n_steps & 1;
      // Step 0, its scales and step 1 are issued back to back: the first step's vmcnt(6) drains step 1 as in the loop.
      MX_DMA_B(p0, 0, 0);
      MX_DMA_A(p0, 0, 0);
      MX_DMA_B(p0, 1, 0);
      MX_DMA_A(p0, 1, 0);
      MX_SCALE(0, p0);
      MX_DMA_A(p0 ^ 1, 0, 1);
      MX_DMA_B(p0 ^ 1, 0, 1);
      MX_DMA_B(p0 ^ 1, 1, 1);
      asm volatile("s_waitcnt vmcnt(6)");
      asm volatile("s_waitcnt lgkmcnt(0)");
      if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
      }
      __builtin_amdgcn_s_barrier();

      int k = 0;
      if (p0) {
        MX_STEP(1, 0, 0);
        k = 1;
      }
#pragma unroll 1
      for (; k < n_steps; k += 2) {
        MX_STEP(0, 1, k);
        MX_STEP(1, 0, k + 1);
      }
      MX_TAIL(0, 1, n_steps);
#undef MX_STEP
#undef MX_TAIL
#undef MX_PHASE

      if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
      }
#undef MX_DMA_A
#undef MX_DMA_B
#undef MX_LD_A
#undef MX_LD_B
#undef MX_LDT_A
#undef MX_LDT_B
#undef MX_SCALE
#undef MX_SA
#undef MX_SB
#undef MX_MMA
