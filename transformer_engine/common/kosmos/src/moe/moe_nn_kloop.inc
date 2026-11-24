/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// NN tile body for dgrad2 and dgrad1, textually included by moe_bwd_single.
#define MOE_NN_LOAD_B_ONE(P, J, I, SOFF)                                                               \
  llvm_amdgcn_raw_buffer_load_lds(                                                                     \
      b_srd,                                                                                           \
      (as3_uint32_ptr)(uintptr_t)((uint32_t)(uintptr_t)(&Bs[P][J][(I) * NUM_WARPS * (16 * 64 / (int)sizeof(bf16))]) + \
                                  b_lds_wo),                                                           \
      16, (int)b_lane_off[I], (int)(SOFF), 0, static_cast<int>(coherency::cache_all))
// Tile base in soffset (stride field 0 keeps it inside the range check). Hand-unrolled: a loop lets IndVarSimplify
// sink the previous phase's MFMAs past setprio/barriers.
#define MOE_NN_LOAD_B(P, J, KT)                                                                        \
  do {                                                                                                 \
    const uint32_t soff_ = (                                                                           \
        (uint32_t)(((size_t)(KT)*K_STEP_64 * N + (size_t)block_col * BLOCK_COL + (J)*HALF_COL) *          \
                   sizeof(bf16)));                                                                     \
    static_assert(B_INSTS == 2, "MOE_NN_LOAD_B is hand-unrolled for B_INSTS == 2");                   \
    MOE_NN_LOAD_B_ONE(P, J, 0, soff_);                                                                 \
    MOE_NN_LOAD_B_ONE(P, J, 1, soff_);                                                                 \
  } while (0)
#define MOE_NN_LOAD_B_H(P, J, KT, H)                                                                   \
  MOE_NN_LOAD_B_ONE(P, J, H,                                                                           \
                    (uint32_t)(((size_t)(KT)*K_STEP_64 * N + (size_t)block_col * BLOCK_COL + (J)*HALF_COL) * \
                               sizeof(bf16)))
#define NN_A_H(H, DST, KT, SW, LDS) dma_load_h<H>(DST, A, {0, 0, 0, KT}, SW, a_srd, a_base, LDS)
#define MOE_NN_READ_B(BT, P, J)                                                                        \
  do {                                                                                                 \
    const uint32_t addr_ = (uint32_t)(uintptr_t)(&Bs[0][0][0]) + (uint32_t)(warp_n * 4096 + lane_b * 8); \
    constexpr int bo_ = ((P) * 2 + (J)) * B_HALF * (int)sizeof(bf16);                                \
    _MOE_RDB(BT, 0, 0, bo_ + 0, bo_ + 512);                                                            \
    _MOE_RDB(BT, 0, 1, bo_ + 1024, bo_ + 1536);                                                        \
    _MOE_RDB(BT, 1, 0, bo_ + 2048, bo_ + 2560);                                                        \
    _MOE_RDB(BT, 1, 1, bo_ + 3072, bo_ + 3584);                                                        \
  } while (0)
// First output earlyclobber: see nt_load_frag (src/utils/gemm_common.cuh).
#define _MOE_RDB(BT, I, J, O0, O1)                                                                     \
  asm volatile("ds_read_b64_tr_b16 %0, %2 offset:%3\n"                                                  \
               "ds_read_b64_tr_b16 %1, %2 offset:%4\n"                                                  \
               : "=&v"(*reinterpret_cast<float2 *>(&(BT).tiles[I][J].data[0])),                         \
                 "=v"(*reinterpret_cast<float2 *>(&(BT).tiles[I][J].data[2]))                           \
               : "v"(addr_), "i"(O0), "i"(O1)                                                          \
               : "memory")

#define NN_VM(N) asm volatile("s_waitcnt vmcnt(" #N ")")
#define NN_VM_LDSDMA(N) asm volatile("s_waitcnt vmcnt(" #N ")" ::: "memory")
#define NN_READ_A(DST, P, J) \
  moe_nn_read_A<((P) * 2 + (J)) * (int)sizeof(ST_A)>(DST, subtile_inplace<REG_M, K_STEP_64>(As[0][0], {warp_m, 0}))
#define NN_LGKM(N) asm volatile("s_waitcnt lgkmcnt(" #N ")")
RT_A a_tile;
RT_B b_tile_0, b_tile_1;
RT_C c00, c01, c10, c11;
zero(c00);
zero(c01);
zero(c10);
zero(c11);

constexpr int tic = 0, toc = 1;

MOE_NN_LOAD_B(tic, 0, 0);
moe_nn_load_A(As[tic][0], A, {0, 0, 0, 0}, sw_A, a_srd, a_base, a_lds_00);
MOE_NN_LOAD_B(tic, 1, 0);
moe_nn_load_A(As[tic][1], A, {0, 0, 0, 0}, sw_A1, a_srd, a_base, a_lds_01);

if (warp_m == 1) {
  __builtin_amdgcn_s_barrier();
}

NN_VM(4);
__builtin_amdgcn_s_barrier();

MOE_NN_LOAD_B(toc, 0, 1);
moe_nn_load_A(As[toc][0], A, {0, 0, 0, 1}, sw_A, a_srd, a_base, a_lds_10);
MOE_NN_LOAD_B(toc, 1, 1);

NN_VM(4);
__builtin_amdgcn_s_barrier();
MOE_NN_READ_B(b_tile_0, 0, 0);

#pragma unroll 1
for (int tile = 0; tile < k_tiles - 2; tile += 2) {
  NN_A_H(0, As[1][1], tile + 1, sw_A1, a_lds_11);
  NN_READ_A(a_tile, 0, 0);
  NN_A_H(1, As[1][1], tile + 1, sw_A1, a_lds_11);
  NN_LGKM(8);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c00, a_tile, b_tile_0, c00);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  MOE_NN_LOAD_B_H(0, 0, tile + 2, 0);
  MOE_NN_READ_B(b_tile_1, 0, 1);
  MOE_NN_LOAD_B_H(0, 0, tile + 2, 1);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c01, a_tile, b_tile_1, c01);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  NN_A_H(0, As[0][0], tile + 2, sw_A, a_lds_00);
  NN_READ_A(a_tile, 0, 1);
  NN_A_H(1, As[0][0], tile + 2, sw_A, a_lds_00);
  // LDS-DMA is not ordered against a later ds_read of its destination: vmcnt(10) retires the Bs[1][0] DMAs two
  // barriers before the P7 read (warp rows run one barrier apart). Do not remove or move.
  NN_VM_LDSDMA(10);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c10, a_tile, b_tile_0, c10);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  MOE_NN_LOAD_B_H(0, 1, tile + 2, 0);
  MOE_NN_READ_B(b_tile_0, 1, 0);
  MOE_NN_LOAD_B_H(0, 1, tile + 2, 1);
  NN_VM(6);
  __builtin_amdgcn_s_barrier();

  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c11, a_tile, b_tile_1, c11);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  NN_A_H(0, As[0][1], tile + 2, sw_A1, a_lds_01);
  NN_READ_A(a_tile, 1, 0);
  NN_A_H(1, As[0][1], tile + 2, sw_A1, a_lds_01);
  NN_LGKM(8);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c00, a_tile, b_tile_0, c00);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  MOE_NN_LOAD_B_H(1, 0, tile + 3, 0);
  MOE_NN_READ_B(b_tile_1, 1, 1);
  MOE_NN_LOAD_B_H(1, 0, tile + 3, 1);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c01, a_tile, b_tile_1, c01);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  NN_A_H(0, As[1][0], tile + 3, sw_A, a_lds_10);
  NN_READ_A(a_tile, 1, 1);
  NN_A_H(1, As[1][0], tile + 3, sw_A, a_lds_10);
  NN_VM_LDSDMA(10);  // P3's Bs[0][0] DMA, read in P15: two barriers ahead, as P5's
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c10, a_tile, b_tile_0, c10);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  MOE_NN_LOAD_B_H(1, 1, tile + 3, 0);
  MOE_NN_READ_B(b_tile_0, 0, 0);
  MOE_NN_LOAD_B_H(1, 1, tile + 3, 1);
  NN_VM(4);
  __builtin_amdgcn_s_barrier();

  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c11, a_tile, b_tile_1, c11);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
}

{
  int tile = k_tiles - 2;
  MOE_NN_READ_B(b_tile_0, 0, 0);
  NN_A_H(0, As[1][1], tile + 1, sw_A1, a_lds_11);
  NN_READ_A(a_tile, 0, 0);
  NN_A_H(1, As[1][1], tile + 1, sw_A1, a_lds_11);
  __builtin_amdgcn_s_barrier();
  asm volatile("s_waitcnt lgkmcnt(0)");

  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c00, a_tile, b_tile_0, c00);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  MOE_NN_READ_B(b_tile_1, 0, 1);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c01, a_tile, b_tile_1, c01);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  NN_READ_A(a_tile, 0, 1);
  NN_VM(4);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c10, a_tile, b_tile_0, c10);
  mma_ABt(c11, a_tile, b_tile_1, c11);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
}

{
  MOE_NN_READ_B(b_tile_0, 1, 0);
  NN_READ_A(a_tile, 1, 0);
  NN_VM(2);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c00, a_tile, b_tile_0, c00);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  MOE_NN_READ_B(b_tile_1, 1, 1);
  asm volatile("s_waitcnt vmcnt(0)");
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c01, a_tile, b_tile_1, c01);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();

  NN_READ_A(a_tile, 1, 1);
  __builtin_amdgcn_s_barrier();

  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_setprio(1);
  mma_ABt(c10, a_tile, b_tile_0, c10);
  mma_ABt(c11, a_tile, b_tile_1, c11);
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_barrier();
}

if (warp_m == 0) {
  __builtin_amdgcn_s_barrier();
}
#undef NN_VM
#undef NN_READ_A
#undef NN_VM_LDSDMA
#undef NN_LGKM
#undef MOE_NN_LOAD_B
#undef MOE_NN_LOAD_B_ONE
#undef MOE_NN_LOAD_B_H
#undef NN_A_H
#undef MOE_NN_READ_B
#undef _MOE_RDB
