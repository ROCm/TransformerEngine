/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "grouped_f16_gemm_helper.cuh"

namespace te_kittens::cdna4::grouped_f16 {

template <typename T, typename O, bool ACCUMULATE>
__global__ __launch_bounds__(NUM_THREADS, 2)
void grouped_f16_nn_kernel(const GroupedTab tab, int num_experts, int N, int K, int *ctr) {
    using ST_A = kittens::st<T, HALF_ROW, BLOCK_K, kittens::st_16x32_s>;
    using RT_A = kittens::rt<T, REG_M, BLOCK_K, kittens::row_l, kittens::rt_16x32_s>;
    using RT_B = kittens::rt<T, REG_N, BLOCK_K, kittens::col_l, kittens::rt_16x32_s>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using R    = raw16_t<T>;
    constexpr int B_HALF = HALF_COL * BLOCK_K;
    __shared__ alignas(16) unsigned char s_tiles[LDS_TILES];
    __shared__ int s_unit;
    static_assert(4 * sizeof(ST_A) + 4 * B_HALF * sizeof(T) == LDS_TILES, "NN tiles must fill the region");
    ST_A(&As)[2][2] = *reinterpret_cast<ST_A(*)[2][2]>(s_tiles);
    T(&Bs)[2][2][B_HALF] = *reinterpret_cast<T(*)[2][2][B_HALF]>(s_tiles + 4 * sizeof(ST_A));
    constexpr int B_INSTS = (HALF_COL * BLOCK_K * (int)sizeof(T)) / (16 * NUM_THREADS);
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;

    const int wid0 = kittens::warpid() % NUM_WARPS;
    const uintptr_t lds_warp_off = wid0 * elem_per_warp * sizeof(T);
    const uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    const uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    const uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0][0]) + lds_warp_off));
    const uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1][0]) + lds_warp_off));
    const uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0][0]) + lds_warp_off));
    const uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1][0]) + lds_warp_off));
    uint32_t sw_A[2], sw_A1[2];

    const int units   = tab.tile_off[num_experts];
    const int w_tiles = K / BLOCK_COL;
#pragma unroll 1
    for (;;) {
        const int u = claim(ctr, units, &s_unit);
        if (u >= units) {
            break;
        }
        const int e    = hk_upper_bound(tab.tile_off, num_experts, u);
        const int rows = tab.rows[e];
        int block_col, block_row;
        unit_tile(u - tab.tile_off[e], w_tiles, (rows + BLOCK_ROW - 1) / BLOCK_ROW, block_col, block_row);
        const int rows_left = min(rows - block_row * BLOCK_ROW, BLOCK_ROW);
        const int k_tiles   = N / BLOCK_K;
        const uint32_t a_row_bytes = (uint32_t)N * sizeof(T);
        const R *a_base = uniform_ptr((const R *)tab.b[e] + (size_t)block_row * BLOCK_ROW * N);
        const R *b_base = uniform_ptr((const R *)tab.a[e]);
        const kittens::i32x4 a_srd = uniform_srd(a_base, (uint32_t)rows_left * a_row_bytes);
        const kittens::i32x4 b_srd = uniform_srd(b_base, (uint32_t)N * (uint32_t)K * sizeof(T));

        int lane_o = (int)threadIdx.x & 63;
        asm volatile("" : "+v"(lane_o));
        int tid_o = (int)threadIdx.x;
        asm volatile("" : "+v"(tid_o));
        const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
        const int warp_m = wv_o / WARPS_COL;
        const int warp_n = wv_o % WARPS_COL;
        const int wid    = wv_o % NUM_WARPS;
        tile_lane_offsets<ST_A>(lane_o, wv_o, a_row_bytes, sw_A);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            sw_A1[i] = sw_A[i] + HALF_ROW * a_row_bytes;
        }
        static_assert(B_INSTS == 2, "the B DMAs are written out for B_INSTS == 2");
        uint32_t b_lane_off[B_INSTS];
#pragma unroll
        for (int i = 0; i < B_INSTS; i++) {
            const int q  = (wid + i * NUM_WARPS) * 64 + lane_o;
            const int W  = q / 32, within = q % 32;
            const int nb = W / 4, w = W % 4;
            const int r  = within / 2, c0 = (within % 2) * 8;
            const int n  = nb * 16 + c0;
            const int kk = 32 * (w / 2) + 8 * (r / 4) + (r % 4) + 4 * (w % 2);
            b_lane_off[i] = (uint32_t)((kk * K + n) * (int)sizeof(T));
        }
        constexpr uint32_t K_BYTES = BLOCK_K * sizeof(T);
        const uint32_t b_k_bytes = (uint32_t)BLOCK_K * (uint32_t)K * sizeof(T);
        const uint32_t b_off0    = (uint32_t)(block_col * BLOCK_COL) * sizeof(T);
        const uint32_t b_off1    = (uint32_t)(block_col * BLOCK_COL + HALF_COL) * sizeof(T);
#define NN_RDB(BT, I, J, O0, O1)                                                    \
    asm volatile("ds_read_b64_tr_b16 %0, %2 offset:%3\n"                            \
                 "ds_read_b64_tr_b16 %1, %2 offset:%4\n"                            \
                 : "=&v"(*reinterpret_cast<float2 *>(&(BT).tiles[I][J].data[0])),   \
                   "=v"(*reinterpret_cast<float2 *>(&(BT).tiles[I][J].data[2]))     \
                 : "v"(addr_), "i"(O0), "i"(O1)                                     \
                 : "memory")
#define NN_READ_B(BT, P, J)                                                                                  \
    do {                                                                                                     \
        const uint32_t addr_ = (uint32_t)(uintptr_t)(&Bs[0][0][0]) + (uint32_t)(warp_n * 4096 + lane_o * 8); \
        constexpr int bo_    = ((P) * 2 + (J)) * B_HALF * (int)sizeof(T);                                    \
        NN_RDB(BT, 0, 0, bo_ + 0, bo_ + 512);                                                                \
        NN_RDB(BT, 0, 1, bo_ + 1024, bo_ + 1536);                                                            \
        NN_RDB(BT, 1, 0, bo_ + 2048, bo_ + 2560);                                                            \
        NN_RDB(BT, 1, 1, bo_ + 3072, bo_ + 3584);                                                            \
    } while (0)
#define NN_READ_A(DST, P, J)                                                                                   \
    load_frag_rows<((P) * 2 + (J)) * (int)sizeof(ST_A)>(DST,                                                   \
        kittens::subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0}), kittens::laneid())

        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        kittens::zero(c00);
        kittens::zero(c01);
        kittens::zero(c10);
        kittens::zero(c11);

        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_00, 16, b_lane_off[0], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_00 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_00, 16, sw_A[0], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_00 + 16 * NUM_THREADS), 16, sw_A[1], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_01, 16, b_lane_off[0], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_01 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_01, 16, sw_A1[0], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_01 + 16 * NUM_THREADS), 16, sw_A1[1], 0, 0, 0);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_10, 16, b_lane_off[0], b_off0 + b_k_bytes, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_10 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off0 + b_k_bytes, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_10, 16, sw_A[0], K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_10 + 16 * NUM_THREADS), 16, sw_A[1], K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_11, 16, b_lane_off[0], b_off1 + b_k_bytes, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_11 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off1 + b_k_bytes, 0, 0);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        NN_READ_B(b_tile_0, 0, 0);

#pragma unroll 1
        for (int tile = 0; tile < k_tiles - 2; tile += 2) {
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_11, 16, sw_A1[0], (tile + 1) * K_BYTES, 0, 0);
            NN_READ_A(a_tile, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_11 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 1) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_00, 16, b_lane_off[0], b_off0 + (tile + 2) * b_k_bytes, 0, 0);
            NN_READ_B(b_tile_1, 0, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_00 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off0 + (tile + 2) * b_k_bytes, 0, 0);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_00, 16, sw_A[0], (tile + 2) * K_BYTES, 0, 0);
            NN_READ_A(a_tile, 0, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_00 + 16 * NUM_THREADS), 16, sw_A[1], (tile + 2) * K_BYTES, 0, 0);
            // LDS-DMA is not ordered against a later ds_read of its destination: retire the Bs[1][0] DMAs two barriers
            // before their read, since the two warp rows run one barrier apart
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_01, 16, b_lane_off[0], b_off1 + (tile + 2) * b_k_bytes, 0, 0);
            NN_READ_B(b_tile_0, 1, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_01 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off1 + (tile + 2) * b_k_bytes, 0, 0);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_01, 16, sw_A1[0], (tile + 2) * K_BYTES, 0, 0);
            NN_READ_A(a_tile, 1, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_01 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 2) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_10, 16, b_lane_off[0], b_off0 + (tile + 3) * b_k_bytes, 0, 0);
            NN_READ_B(b_tile_1, 1, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_10 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off0 + (tile + 3) * b_k_bytes, 0, 0);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_10, 16, sw_A[0], (tile + 3) * K_BYTES, 0, 0);
            NN_READ_A(a_tile, 1, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_10 + 16 * NUM_THREADS), 16, sw_A[1], (tile + 3) * K_BYTES, 0, 0);
            // Same LDS-DMA hazard for the Bs[0][0] DMAs
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_11, 16, b_lane_off[0], b_off1 + (tile + 3) * b_k_bytes, 0, 0);
            NN_READ_B(b_tile_0, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_11 + 16 * NUM_THREADS), 16, b_lane_off[1], b_off1 + (tile + 3) * b_k_bytes, 0, 0);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            int tile = k_tiles - 2;
            NN_READ_B(b_tile_0, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_11, 16, sw_A1[0], (tile + 1) * K_BYTES, 0, 0);
            NN_READ_A(a_tile, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_11 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 1) * K_BYTES, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            NN_READ_B(b_tile_1, 0, 1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            NN_READ_A(a_tile, 0, 1);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, a_tile, b_tile_0, c10);
            kittens::mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            NN_READ_B(b_tile_0, 1, 0);
            NN_READ_A(a_tile, 1, 0);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            NN_READ_B(b_tile_1, 1, 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            NN_READ_A(a_tile, 1, 1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, a_tile, b_tile_0, c10);
            kittens::mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }
#undef NN_RDB
#undef NN_READ_B
#undef NN_READ_A
        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(warp_n);
        int lane_epi = kittens::laneid();
        asm volatile("" : "+v"(lane_epi));
        const O *d_base = uniform_ptr((const O *)tab.d[e] + (size_t)block_row * BLOCK_ROW * K + block_col * BLOCK_COL);
        const kittens::i32x4 d_srd = uniform_srd(d_base, (uint32_t)rows_left * (uint32_t)K * sizeof(O));
        epi_staged<O, ACCUMULATE>(s_tiles, c00, c01, c10, c11, rf0, cf0, lane_epi, d_srd, K);
    }
}

template <typename T, typename O>
static void launch_nn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr,
                      bool accumulate, hipStream_t stream) {
    KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
        grouped_f16_nn_kernel<T, O, ACCUMULATE><<<grid, NUM_THREADS, 0, stream>>>(tab, num_experts, N, K, ctr);
    ) // NOLINT(*)
}

void launch_grouped_f16_nn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool accumulate, hipStream_t stream) {
    if (in_dtype == KITTENS_BFLOAT16) {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_nn<kittens::bf16, float>(tab, num_experts, N, K, grid, ctr, accumulate, stream);
        } else {
            launch_nn<kittens::bf16, __bf16>(tab, num_experts, N, K, grid, ctr, accumulate, stream);
        }
    } else {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_nn<kittens::half, float>(tab, num_experts, N, K, grid, ctr, accumulate, stream);
        } else {
            launch_nn<kittens::half, _Float16>(tab, num_experts, N, K, grid, ctr, accumulate, stream);
        }
    }
}

}  // namespace te_kittens::cdna4::grouped_f16
