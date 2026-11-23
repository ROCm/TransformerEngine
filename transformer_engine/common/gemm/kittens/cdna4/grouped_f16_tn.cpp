/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "grouped_f16_gemm_helper.cuh"

namespace te_kittens::cdna4::grouped_f16 {

template <typename T, typename O, bool BIAS, bool ACCUMULATE>
__global__ __launch_bounds__(NUM_THREADS, 2)
void grouped_f16_tn_kernel(const GroupedTab tab, int num_experts, int N, int K, int *ctr) {
    using ST_A = kittens::st<T, HALF_ROW, BLOCK_K, kittens::st_16x32_s>;
    using ST_B = kittens::st<T, HALF_COL, BLOCK_K, kittens::st_16x32_s>;
    using RT_A = kittens::rt<T, REG_M, BLOCK_K, kittens::row_l, kittens::rt_16x32_s>;
    using RT_B = kittens::rt<T, REG_N, BLOCK_K, kittens::row_l, kittens::rt_16x32_s>;
    using RT_C = kittens::rt_fl<REG_N, REG_M, kittens::col_l, kittens::rt_16x16_s>;
    using R    = raw16_t<T>;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ int s_unit;
    constexpr int SZA = sizeof(ST_A);  // As[p][q] = As[0][0] + (2p + q) SZA: the fragment loads' immediate
    constexpr int SZB = sizeof(ST_B);
    static_assert(std::is_same_v<ST_A, ST_B>, "B's lane offsets are A's");

    const int wid = kittens::warpid() % NUM_WARPS;
    const uintptr_t lds_warp_off = wid * 16 * kittens::WARP_THREADS;
    const uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    const uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    const uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    const uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + lds_warp_off));
    const uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + lds_warp_off));
    const uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + lds_warp_off));
    const uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + lds_warp_off));
    uint32_t sw_A[2], sw_A1[2], sw_B[2];

    const int units   = tab.tile_off[num_experts];
    const int w_tiles = N / BLOCK_COL;
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
        const int k_tiles   = K / BLOCK_K;
        const uint32_t row_bytes = (uint32_t)K * sizeof(T);
        const R *a_base = uniform_ptr((const R *)tab.b[e] + (size_t)block_row * BLOCK_ROW * K);
        const R *b_base = uniform_ptr((const R *)tab.a[e]);
        const kittens::i32x4 a_srd = uniform_srd(a_base, (uint32_t)rows_left * row_bytes);
        const kittens::i32x4 b_srd = uniform_srd(b_base, (uint32_t)N * row_bytes);

        const int ft     = opaque_tid();
        const int warp_m = ft / (kittens::WARP_THREADS * WARPS_COL), warp_n = ft / kittens::WARP_THREADS % WARPS_COL;
        const int lane   = ft % kittens::WARP_THREADS;
        tile_lane_offsets<ST_A>(lane, ft / kittens::WARP_THREADS, row_bytes, sw_A);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            sw_B[i]  = sw_A[i];
            sw_A1[i] = sw_A[i] + HALF_ROW * row_bytes;
        }
        constexpr uint32_t K_BYTES = BLOCK_K * sizeof(T);
        const uint32_t b_off0 = (uint32_t)(block_col * 2 * HALF_COL) * row_bytes;
        const uint32_t b_off1 = (uint32_t)((block_col * 2 + 1) * HALF_COL) * row_bytes;

        const auto sa0 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0});
        const auto sb0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[0][0], {warp_n, 0});
        RT_A a_tile;
        RT_B b_tile_0, b_tile_1;
        RT_C c00, c01, c10, c11;
        kittens::zero(c00);
        kittens::zero(c01);
        kittens::zero(c10);
        kittens::zero(c11);

        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_00, 16, sw_B[0], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_00 + 16 * NUM_THREADS), 16, sw_B[1], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_00, 16, sw_A[0], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_00 + 16 * NUM_THREADS), 16, sw_A[1], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_01, 16, sw_B[0], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_01 + 16 * NUM_THREADS), 16, sw_B[1], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_01, 16, sw_A1[0], 0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_01 + 16 * NUM_THREADS), 16, sw_A1[1], 0, 0, 0);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_10, 16, sw_B[0], b_off0 + K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_10 + 16 * NUM_THREADS), 16, sw_B[1], b_off0 + K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_10, 16, sw_A[0], K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_10 + 16 * NUM_THREADS), 16, sw_A[1], K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_11, 16, sw_B[0], b_off1 + K_BYTES, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_11 + 16 * NUM_THREADS), 16, sw_B[1], b_off1 + K_BYTES, 0, 0);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        load_frag_rows<0>(b_tile_0, sb0, lane);

#pragma unroll 1
        for (int tile = 0; tile < k_tiles - 2; tile += 2) {
            load_frag_rows<0>(a_tile, sa0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_11, 16, sw_A1[0], (tile + 1) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_11 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 1) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, b_tile_0, a_tile, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<SZB>(b_tile_1, sb0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_00, 16, sw_B[0], b_off0 + (tile + 2) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_00 + 16 * NUM_THREADS), 16, sw_B[1], b_off0 + (tile + 2) * K_BYTES, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, b_tile_1, a_tile, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZA>(a_tile, sa0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_00, 16, sw_A[0], (tile + 2) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_00 + 16 * NUM_THREADS), 16, sw_A[1], (tile + 2) * K_BYTES, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, b_tile_0, a_tile, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<2 * SZB>(b_tile_0, sb0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_01, 16, sw_B[0], b_off1 + (tile + 2) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_01 + 16 * NUM_THREADS), 16, sw_B[1], b_off1 + (tile + 2) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c11, b_tile_1, a_tile, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<2 * SZA>(a_tile, sa0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_01, 16, sw_A1[0], (tile + 2) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_01 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 2) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, b_tile_0, a_tile, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<3 * SZB>(b_tile_1, sb0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_10, 16, sw_B[0], b_off0 + (tile + 3) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_10 + 16 * NUM_THREADS), 16, sw_B[1], b_off0 + (tile + 3) * K_BYTES, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, b_tile_1, a_tile, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZA>(a_tile, sa0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_10, 16, sw_A[0], (tile + 3) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_10 + 16 * NUM_THREADS), 16, sw_A[1], (tile + 3) * K_BYTES, 0, 0);
            // LDS-DMA is not ordered against a later ds_read of its destination: retire the Bs[0][0] DMAs two barriers
            // before their read, since the two warp rows run one barrier apart
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, b_tile_0, a_tile, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_frag_rows<0>(b_tile_0, sb0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds_11, 16, sw_B[0], b_off1 + (tile + 3) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds_11 + 16 * NUM_THREADS), 16, sw_B[1], b_off1 + (tile + 3) * K_BYTES, 0, 0);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c11, b_tile_1, a_tile, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            const int tile = k_tiles - 2;
            load_frag_rows<0>(b_tile_0, sb0, lane);
            load_frag_rows<0>(a_tile, sa0, lane);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds_11, 16, sw_A1[0], (tile + 1) * K_BYTES, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds_11 + 16 * NUM_THREADS), 16, sw_A1[1], (tile + 1) * K_BYTES, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, b_tile_0, a_tile, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZB>(b_tile_1, sb0, lane);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, b_tile_1, a_tile, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<SZA>(a_tile, sa0, lane);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, b_tile_0, a_tile, c10);
            kittens::mma_ABt(c11, b_tile_1, a_tile, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_frag_rows<2 * SZB>(b_tile_0, sb0, lane);
            load_frag_rows<2 * SZA>(a_tile, sa0, lane);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c00, b_tile_0, a_tile, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZB>(b_tile_1, sb0, lane);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c01, b_tile_1, a_tile, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_frag_rows<3 * SZA>(a_tile, sa0, lane);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            kittens::mma_ABt(c10, b_tile_0, a_tile, c10);
            kittens::mma_ABt(c11, b_tile_1, a_tile, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        const int fe = opaque_tid();
        const int we = fe / kittens::WARP_THREADS;
        if (we / WARPS_COL == 0) {
            __builtin_amdgcn_s_barrier();
        }
        const int rf0 = __builtin_amdgcn_readfirstlane(we / WARPS_COL);
        const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + we / WARPS_COL);
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + we % WARPS_COL);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + we % WARPS_COL);
        int lane_epi = fe % kittens::WARP_THREADS;
        asm volatile("" : "+v"(lane_epi));
        const O *d_base = uniform_ptr((const O *)tab.d[e] + (size_t)block_row * BLOCK_ROW * N);
        const kittens::i32x4 d_srd = uniform_srd(d_base, (uint32_t)rows_left * (uint32_t)N * sizeof(O));
        const R *bias = BIAS ? uniform_ptr((const R *)tab.bias[e]) : nullptr;
        epi_t16<O, BIAS, ACCUMULATE>(d_srd, c00, rf0 * REG_M, cf0 * REG_N, N, lane_epi, bias);
        epi_t16<O, BIAS, ACCUMULATE>(d_srd, c01, rf0 * REG_M, cf1 * REG_N, N, lane_epi, bias);
        epi_t16<O, BIAS, ACCUMULATE>(d_srd, c10, rf1 * REG_M, cf0 * REG_N, N, lane_epi, bias);
        epi_t16<O, BIAS, ACCUMULATE>(d_srd, c11, rf1 * REG_M, cf1 * REG_N, N, lane_epi, bias);
    }
}

template <typename T, typename O>
static void launch_tn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr,
                      bool bias, bool accumulate, hipStream_t stream) {
    KITTENS_BOOL_SWITCH(bias, BIAS,
        KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
            grouped_f16_tn_kernel<T, O, BIAS, ACCUMULATE>
                <<<grid, NUM_THREADS, 0, stream>>>(tab, num_experts, N, K, ctr);
    )) // NOLINT(*)
}

void launch_grouped_f16_tn(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool bias, bool accumulate, hipStream_t stream) {
    if (in_dtype == KITTENS_BFLOAT16) {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_tn<kittens::bf16, float>(tab, num_experts, N, K, grid, ctr, bias, accumulate, stream);
        } else {
            launch_tn<kittens::bf16, __bf16>(tab, num_experts, N, K, grid, ctr, bias, accumulate, stream);
        }
    } else {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_tn<kittens::half, float>(tab, num_experts, N, K, grid, ctr, bias, accumulate, stream);
        } else {
            launch_tn<kittens::half, _Float16>(tab, num_experts, N, K, grid, ctr, bias, accumulate, stream);
        }
    }
}

}  // namespace te_kittens::cdna4::grouped_f16
