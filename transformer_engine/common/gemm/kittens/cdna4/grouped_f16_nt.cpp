/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "grouped_f16_gemm_helper.cuh"

namespace te_kittens::cdna4::grouped_f16 {

template <typename T, typename O, bool BGRAD, bool ACCUMULATE>
__global__ __launch_bounds__(NUM_THREADS, 2)
void grouped_f16_nt_kernel(const GroupedTab tab, int num_experts, int N, int K, int *ctr) {
    using ST_AB = kittens::st<T, BLOCK_K, HALF_COL, kittens::st_16x32_s>;
    using RT_A  = kittens::rt<T, BLOCK_K, REG_M, kittens::col_l, kittens::rt_32x16_s>;
    using RT_B  = kittens::rt<T, BLOCK_K, REG_N, kittens::col_l, kittens::rt_32x16_s>;
    using RT_C  = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;
    using R     = raw16_t<T>;
    __shared__ alignas(16) unsigned char s_tiles[LDS_TILES];
    __shared__ int s_unit;
    static_assert(8 * sizeof(ST_AB) == LDS_TILES, "NT tiles must fill the region");
    ST_AB(&As)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(s_tiles);
    ST_AB(&Bs)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(s_tiles + 4 * sizeof(ST_AB));

    const int tiles   = tab.tile_off[num_experts];
    const int nb      = N / BLOCK_ROW;
    const int units   = tiles + (BGRAD ? num_experts * nb : 0);
    const int w_tiles = K / BLOCK_COL;
#pragma unroll 1
    for (;;) {
        const int u = claim(ctr, units, &s_unit);
        if (u >= units) {
            break;
        }
        if (BGRAD && u >= tiles) {
            typedef __attribute__((address_space(1))) const u32x4_t gu4;
            const int e    = (u - tiles) / nb;
            const int cb   = u - tiles - e * nb;
            const int rows = tab.rows[e];
            const int tid  = opaque_tid();
            const int c = tid & 31, g = tid >> 5;
            const __attribute__((address_space(1))) T *p =
                (const __attribute__((address_space(1))) T *)uniform_ptr((const T *)tab.b[e]) + cb * BLOCK_COL + c * 8;
            float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
            for (int r = g; r < rows; r += 16) {
                const u32x4_t w = *(gu4 *)(p + (size_t)r * N);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    s[2 * i] += lo16<R>(w[i]);
                    s[2 * i + 1] += hi16<R>(w[i]);
                }
            }
            auto *sl = (__attribute__((address_space(3))) float *)(lds_u8 *)s_tiles;
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 8; i++) {
                sl[g * BLOCK_COL + c * 8 + i] = s[i];
            }
            __syncthreads();
            if (tid < BLOCK_COL) {
                float v = 0.f;
#pragma unroll
                for (int q = 0; q < 16; q++) {
                    v += sl[q * BLOCK_COL + tid];
                }
                __attribute__((address_space(1))) O *o =
                    (__attribute__((address_space(1))) O *)uniform_ptr((O *)tab.bias[e]);
                o[cb * BLOCK_COL + tid] = (O)v;
            }
            continue;
        }
        const int e    = hk_upper_bound(tab.tile_off, num_experts, u);
        const int rows = tab.rows[e];
        int mblk, nblk;
        unit_tile(u - tab.tile_off[e], w_tiles, nb, mblk, nblk);
        const int k_real  = (rows + BLOCK_K - 1) / BLOCK_K;
        // The pipeline retires k-steps in pairs; for an odd k_real the extra step reloads k-step k_real - 1 (never
        // read) and is drained, not computed: epi_staged's first barrier orders its DMAs before the LDS reuse
        const int k_tiles = k_real + (k_real & 1);
        const uint32_t a_lead = (uint32_t)N * sizeof(T), b_lead = (uint32_t)K * sizeof(T);
        const R *a_base = uniform_ptr((const R *)tab.b[e]);
        const R *b_base = uniform_ptr((const R *)tab.a[e]);
        const kittens::i32x4 a_srd = uniform_srd(a_base, (uint32_t)rows * a_lead);
        const kittens::i32x4 b_srd = uniform_srd(b_base, (uint32_t)rows * b_lead);

        int tid_o = (int)threadIdx.x;
        asm volatile("" : "+v"(tid_o));
        const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
        const int warp_m = wv_o / WARPS_COL;
        const int warp_n = wv_o % WARPS_COL;
        const int wid    = wv_o % NUM_WARPS;
        const uintptr_t lds_off = wid * 16 * kittens::WARP_THREADS;
        uint32_t a_lds[2][2], b_lds[2][2];
#pragma unroll
        for (int p = 0; p < 2; p++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                a_lds[p][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[p][j].data[0]) + lds_off));
                b_lds[p][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[p][j].data[0]) + lds_off));
            }
        }
        uint32_t sw_A[2], sw_B[2];
        int lane_t = (int)threadIdx.x & 63;
        asm volatile("" : "+v"(lane_t));
        tile_lane_offsets<ST_AB>(lane_t, wv_o, (int)a_lead, sw_A);
        tile_lane_offsets<ST_AB>(lane_t, wv_o, (int)b_lead, sw_B);
        const int a_col0 = nblk * BLOCK_ROW;
        const int b_col0 = mblk * BLOCK_COL;
        const uint32_t a_k_bytes = (uint32_t)BLOCK_K * a_lead, b_k_bytes = (uint32_t)BLOCK_K * b_lead;
        const uint32_t a_off0    = (uint32_t)a_col0 * sizeof(T), a_off1 = (uint32_t)(a_col0 + HALF_COL) * sizeof(T);
        const uint32_t b_off0    = (uint32_t)b_col0 * sizeof(T), b_off1 = (uint32_t)(b_col0 + HALF_COL) * sizeof(T);
#define NT_READ_A(TL, P, J) \
    nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(TL, As[0][0], warp_m * (REG_M), lane_t)
#define NT_READ_B(TL, P, J) \
    nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(TL, Bs[0][0], warp_n * (REG_N), lane_t)

        RT_C c00, c01, c10, c11;
        RT_A A_tile;
        RT_B B_tile_0, B_tile_1;

        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[0][0], 16, sw_B[0], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[0][0] + 16 * NUM_THREADS), 16, sw_B[1], b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[0][0], 16, sw_A[0], a_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[0][0] + 16 * NUM_THREADS), 16, sw_A[1], a_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[0][1], 16, sw_B[0], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[0][1] + 16 * NUM_THREADS), 16, sw_B[1], b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[0][1], 16, sw_A[0], a_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[0][1] + 16 * NUM_THREADS), 16, sw_A[1], a_off1, 0, 0);
        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        kittens::zero(c00);
        kittens::zero(c01);
        kittens::zero(c10);
        kittens::zero(c11);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[1][0], 16, sw_B[0], min(1, k_real - 1) * b_k_bytes + b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[1][0] + 16 * NUM_THREADS), 16, sw_B[1], min(1, k_real - 1) * b_k_bytes + b_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[1][0], 16, sw_A[0], min(1, k_real - 1) * a_k_bytes + a_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[1][0] + 16 * NUM_THREADS), 16, sw_A[1], min(1, k_real - 1) * a_k_bytes + a_off0, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[1][1], 16, sw_B[0], min(1, k_real - 1) * b_k_bytes + b_off1, 0, 0);
        kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[1][1] + 16 * NUM_THREADS), 16, sw_B[1], min(1, k_real - 1) * b_k_bytes + b_off1, 0, 0);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        NT_READ_B(B_tile_0, 0, 0);
        for (int kt = 0; kt < k_tiles - 2; kt += 2) {
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[1][1], 16, sw_A[0], min(kt + 1, k_real - 1) * a_k_bytes + a_off1, 0, 0);
            NT_READ_A(A_tile, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[1][1] + 16 * NUM_THREADS), 16, sw_A[1], min(kt + 1, k_real - 1) * a_k_bytes + a_off1, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c00, A_tile, B_tile_0, c00);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[0][0], 16, sw_B[0], min(kt + 2, k_real - 1) * b_k_bytes + b_off0, 0, 0);
            NT_READ_B(B_tile_1, 0, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[0][0] + 16 * NUM_THREADS), 16, sw_B[1], min(kt + 2, k_real - 1) * b_k_bytes + b_off0, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c01, A_tile, B_tile_1, c01);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[0][0], 16, sw_A[0], min(kt + 2, k_real - 1) * a_k_bytes + a_off0, 0, 0);
            NT_READ_A(A_tile, 0, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[0][0] + 16 * NUM_THREADS), 16, sw_A[1], min(kt + 2, k_real - 1) * a_k_bytes + a_off0, 0, 0);
            // LDS-DMA is not ordered against a later ds_read of its destination: retire the Bs[1][0] DMAs two barriers
            // before their read, since the two warp rows run one barrier apart
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c10, A_tile, B_tile_0, c10);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[0][1], 16, sw_B[0], min(kt + 2, k_real - 1) * b_k_bytes + b_off1, 0, 0);
            NT_READ_B(B_tile_0, 1, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[0][1] + 16 * NUM_THREADS), 16, sw_B[1], min(kt + 2, k_real - 1) * b_k_bytes + b_off1, 0, 0);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c11, A_tile, B_tile_1, c11);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[0][1], 16, sw_A[0], min(kt + 2, k_real - 1) * a_k_bytes + a_off1, 0, 0);
            NT_READ_A(A_tile, 1, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[0][1] + 16 * NUM_THREADS), 16, sw_A[1], min(kt + 2, k_real - 1) * a_k_bytes + a_off1, 0, 0);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c00, A_tile, B_tile_0, c00);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[1][0], 16, sw_B[0], min(kt + 3, k_real - 1) * b_k_bytes + b_off0, 0, 0);
            NT_READ_B(B_tile_1, 1, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[1][0] + 16 * NUM_THREADS), 16, sw_B[1], min(kt + 3, k_real - 1) * b_k_bytes + b_off0, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c01, A_tile, B_tile_1, c01);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[1][0], 16, sw_A[0], min(kt + 3, k_real - 1) * a_k_bytes + a_off0, 0, 0);
            NT_READ_A(A_tile, 1, 1);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[1][0] + 16 * NUM_THREADS), 16, sw_A[1], min(kt + 3, k_real - 1) * a_k_bytes + a_off0, 0, 0);
            // Same LDS-DMA hazard for the Bs[0][0] DMAs
            asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c10, A_tile, B_tile_0, c10);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)b_lds[1][1], 16, sw_B[0], min(kt + 3, k_real - 1) * b_k_bytes + b_off1, 0, 0);
            NT_READ_B(B_tile_0, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(b_srd, (kittens::as3_uint32_ptr)(b_lds[1][1] + 16 * NUM_THREADS), 16, sw_B[1], min(kt + 3, k_real - 1) * b_k_bytes + b_off1, 0, 0);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c11, A_tile, B_tile_1, c11);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
        }
        {
            NT_READ_B(B_tile_0, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)a_lds[1][1], 16, sw_A[0], (k_real - 1) * a_k_bytes + a_off1, 0, 0);
            NT_READ_A(A_tile, 0, 0);
            kittens::llvm_amdgcn_raw_buffer_load_lds(a_srd, (kittens::as3_uint32_ptr)(a_lds[1][1] + 16 * NUM_THREADS), 16, sw_A[1], (k_real - 1) * a_k_bytes + a_off1, 0, 0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c00, A_tile, B_tile_0, c00);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            NT_READ_B(B_tile_1, 0, 1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c01, A_tile, B_tile_1, c01);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            NT_READ_A(A_tile, 0, 1);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c10, A_tile, B_tile_0, c10);
            kittens::mma_AtB(c11, A_tile, B_tile_1, c11);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
        }
        if (k_real == k_tiles) {
            NT_READ_B(B_tile_0, 1, 0);
            NT_READ_A(A_tile, 1, 0);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c00, A_tile, B_tile_0, c00);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            NT_READ_B(B_tile_1, 1, 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c01, A_tile, B_tile_1, c01);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            NT_READ_A(A_tile, 1, 1);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_AtB(c10, A_tile, B_tile_0, c10);
            kittens::mma_AtB(c11, A_tile, B_tile_1, c11);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
        } else {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }
#undef NT_READ_A
#undef NT_READ_B
        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(warp_n);
        int lane_epi = lane_t;
        asm volatile("" : "+v"(lane_epi));
        const O *d_base = uniform_ptr((const O *)tab.d[e] + (size_t)nblk * BLOCK_ROW * K + mblk * BLOCK_COL);
        const kittens::i32x4 d_srd = uniform_srd(d_base, (uint32_t)BLOCK_ROW * (uint32_t)K * sizeof(O));
        epi_staged<O, ACCUMULATE>(s_tiles, c00, c01, c10, c11, rf0, cf0, lane_epi, d_srd, K);
    }
}

template <typename T, typename O>
static void launch_nt(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr,
                      bool bgrad, bool accumulate, hipStream_t stream) {
    KITTENS_BOOL_SWITCH(bgrad, BGRAD,
        KITTENS_BOOL_SWITCH(accumulate, ACCUMULATE,
            grouped_f16_nt_kernel<T, O, BGRAD, ACCUMULATE>
                <<<grid, NUM_THREADS, 0, stream>>>(tab, num_experts, N, K, ctr);
    )) // NOLINT(*)
}

void launch_grouped_f16_nt(const GroupedTab &tab, int num_experts, int N, int K, int grid, int *ctr, int in_dtype,
                           int out_dtype, bool bgrad, bool accumulate, hipStream_t stream) {
    if (in_dtype == KITTENS_BFLOAT16) {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_nt<kittens::bf16, float>(tab, num_experts, N, K, grid, ctr, bgrad, accumulate, stream);
        } else {
            launch_nt<kittens::bf16, __bf16>(tab, num_experts, N, K, grid, ctr, bgrad, accumulate, stream);
        }
    } else {
        if (out_dtype == KITTENS_FLOAT32) {
            launch_nt<kittens::half, float>(tab, num_experts, N, K, grid, ctr, bgrad, accumulate, stream);
        } else {
            launch_nt<kittens::half, _Float16>(tab, num_experts, N, K, grid, ctr, bgrad, accumulate, stream);
        }
    }
}

}  // namespace te_kittens::cdna4::grouped_f16
