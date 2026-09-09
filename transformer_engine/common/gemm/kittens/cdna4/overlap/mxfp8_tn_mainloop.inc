/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Main K-loop for the TN fused AG+GEMM. Included inside the persistent tile loop with the
//! stage/scale ping-pong state (tic/toc, tic_scales/toc_scales), the LDS handles and the swizzle
//! tables in scope. Not a header: no include guard, and it is a statement, not a declaration.
//! The K-tile epilogues deliberately stay in the caller.
#pragma unroll 2
        for (int k = 0; k < k_tiles - 2; k++, tic ^= 1, toc ^= 1, tic_scales ^= 1, toc_scales ^= 1) {
            if (k + 1 < k_tiles) {
                G_group::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * tiles_M + block_row, 0, 0, 0});
                G_group::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col),     0, 0, 0});
                G_group::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * tiles_N + block_col) + 1, 0, 0, 0});
            }
            auto bs0 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            kittens::load(b0, bs0);
            auto as0 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            kittens::load(a, as0);
            G_group::load(As[toc][1], A, {0, 0, a_half1, k + 1}, sw_A);
            asm volatile("s_waitcnt lgkmcnt(8)"); // wait for scale + data LDS stores; As[toc][1] can overlap
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            auto bs1 = kittens::subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            G_group::load(As[tic][0], A, {0, 0, a_half0, k + 2}, sw_A);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need bs1 in registers for mma_B
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto as1 = kittens::subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            kittens::load(a, as1);
            G_group::load(Bs[tic][0], B, {0, 0, b_half0, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need as1 in registers for mma_C
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G_group::load(Bs[tic][1], B, {0, 0, b_half1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data loads; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
