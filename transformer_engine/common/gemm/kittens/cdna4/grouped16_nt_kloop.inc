/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// NT tile k-loop of the grouped wgrad, textually included by nt_kernel.
#define NT_MMA(C, A, B)                \
  do {                                 \
    __builtin_amdgcn_sched_barrier(0); \
    mma_AtB(C, A, B);                  \
    __builtin_amdgcn_sched_barrier(0); \
  } while (0)
#define NT_MMA2(C0, C1, A, B0, B1)     \
  do {                                 \
    __builtin_amdgcn_sched_barrier(0); \
    mma_AtB(C0, A, B0);                \
    mma_AtB(C1, A, B1);                \
    __builtin_amdgcn_sched_barrier(0); \
  } while (0)
#define NT_PH(READS, LD, P, J, KT) \
  do {                             \
    LD.load_h(P, J, KT, 0);        \
    READS;                         \
    LD.load_h(P, J, KT, 1);        \
  } while (0)
zero(c00); zero(c01); zero(c10); zero(c11);
if (k_tiles > 0) {
  asm volatile("s_waitcnt vmcnt(4)");
  __builtin_amdgcn_s_barrier();
  ldB.load(1, 0, 1); ldA.load(1, 0, 1); ldB.load(1, 1, 1);
  asm volatile("s_waitcnt vmcnt(6)");
  __builtin_amdgcn_s_barrier();
  NT_READ_B(B_tile_0, 0, 0);
  for (int kt = 0; kt < k_tiles - 2; kt += 2) {
    NT_PH(NT_READ_A(A_tile, 0, 0), ldA, 1, 1, kt + 1);
    asm volatile("s_waitcnt lgkmcnt(8)"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c00, A_tile, B_tile_0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    NT_PH(NT_READ_B(B_tile_1, 0, 1), ldB, 0, 0, kt + 2); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c01, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
    // LDS-DMA is not ordered against a later ds_read of its destination: vmcnt(10) retires the Bs[1][0] DMAs two
    // barriers before the P7 read (warp rows run one barrier apart). Do not remove or move.
    NT_PH(NT_READ_A(A_tile, 0, 1), ldA, 0, 0, kt + 2);
    asm volatile("s_waitcnt vmcnt(10)" ::: "memory"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c10, A_tile, B_tile_0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    NT_PH(NT_READ_B(B_tile_0, 1, 0), ldB, 0, 1, kt + 2);
    asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
    NT_MMA(c11, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
    NT_PH(NT_READ_A(A_tile, 1, 0), ldA, 0, 1, kt + 2);
    asm volatile("s_waitcnt lgkmcnt(8)"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c00, A_tile, B_tile_0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    NT_PH(NT_READ_B(B_tile_1, 1, 1), ldB, 1, 0, kt + 3); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c01, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
    NT_PH(NT_READ_A(A_tile, 1, 1), ldA, 1, 0, kt + 3);
    asm volatile("s_waitcnt vmcnt(10)" ::: "memory");  // P3's Bs[0][0] DMA, read in P15
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c10, A_tile, B_tile_0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    NT_PH(NT_READ_B(B_tile_0, 0, 0), ldB, 1, 1, kt + 3);
    asm volatile("s_waitcnt vmcnt(6)"); __builtin_amdgcn_s_barrier();
    NT_MMA(c11, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
  }
  {
    const int kt = k_tiles - 2;
    NT_READ_B(B_tile_0, 0, 0); NT_PH(NT_READ_A(A_tile, 0, 0), ldA, 1, 1, kt + 1);
    __builtin_amdgcn_s_barrier(); asm volatile("s_waitcnt lgkmcnt(0)");
    NT_MMA(c00, A_tile, B_tile_0); __builtin_amdgcn_s_barrier();
    NT_READ_B(B_tile_1, 0, 1); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c01, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
    NT_READ_A(A_tile, 0, 1); asm volatile("s_waitcnt vmcnt(4)"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA2(c10, c11, A_tile, B_tile_0, B_tile_1);
    __builtin_amdgcn_s_barrier();
  }
  if (k_real == k_tiles) {
    NT_READ_B(B_tile_0, 1, 0); NT_READ_A(A_tile, 1, 0);
    asm volatile("s_waitcnt vmcnt(2)"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c00, A_tile, B_tile_0); __builtin_amdgcn_s_barrier();
    NT_READ_B(B_tile_1, 1, 1); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA(c01, A_tile, B_tile_1); __builtin_amdgcn_s_barrier();
    NT_READ_A(A_tile, 1, 1); __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)"); NT_MMA2(c10, c11, A_tile, B_tile_0, B_tile_1);
    __builtin_amdgcn_s_barrier();
  } else {
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_s_barrier();
  }
}
if (warp_m == 0) {
  __builtin_amdgcn_s_barrier();
}
#undef NT_MMA
#undef NT_MMA2
#undef NT_PH
