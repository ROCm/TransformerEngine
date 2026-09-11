/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Main K-loop for the MXFP8 NN GEMM, shared by mxfp8_gemm_nn_kernel (mxfp8_gemm.cpp) and the
//! fused AG+GEMM (overlap/fused_ag_mxfp8_gemm_nn.cuh): the steady-state loop plus both K-tile
//! epilogues and the skew-closing barrier, i.e. everything up to the C writeback, matching the
//! boundary tn_mainloop.inc draws. Included inside the persistent tile loop with the stage/scale
//! ping-pong state (tic/toc, tic_scales/toc_scales), the LDS handles and the swizzle tables in
//! scope. Not a header: no include guard, and it is a statement, not a declaration.
//!
//! Mirrors the K-loop of mxfp8_gemm_nn_kernel in ../mxfp8_gemm.cpp, which is the ground truth for
//! this schedule; the AG path adopts its spelling (k_iters, BLOCK_K, G, A_local/B_local,
//! sa_stride/sa_batch, a_row_tile/n_tile) so the two can be diffed line-for-line -- and because
//! mxfp8_gemm.cpp's device ISA is the acceptance gate, its spelling is the one that survives here.
//! Both use the BLAS convention: A is the weight, K-major [K,M], so it takes the ds_read_*_tr path
//! (a_col_off) and indexes {0,0,k,a_row_tile*2+h}; B is the K-contiguous [N,K] operand, read with
//! subtile_inplace and indexed {0,0,n_tile*2+h,k}. The AG caller declares a_row_tile/n_tile as
//! aliases of its own block_row/block_col. What is left in the caller is only the C writeback.
#pragma unroll 2
        for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, tic_scales ^= 1, toc_scales ^= 1) {
            if (k + 1 < k_iters) {
                G::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
                G::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
                G::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});
            }
            auto bs0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
            kittens::load(b0, bs0);
            kittens::load(a, As[tic][0], a_col_off);
            G::load(As[toc][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: the A tr-reads must land before mma_A
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);
            auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            G::load(As[tic][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[tic][0]);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need bs1 in registers for mma_B
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            kittens::load(a, As[tic][1], a_col_off);
            G::load(Bs[tic][0], B_local, {0, 0, n_tile * 2,    k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need As[tic][1] in registers for mma_C
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], B_local, {0, 0, n_tile * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data loads; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        { // Epilogue k = k_iters - 2
            int k = k_iters - 2;
            if (k + 1 < k_iters) {
                G::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
                G::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
                G::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});
            }
            asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM: last prefetch iteration
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);

            auto bs0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
            kittens::load(b0, bs0);
            kittens::load(a, As[tic][0], a_col_off);
            G::load(As[toc][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need As[tic][0]/bs0 in registers for mma_A
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need bs1 in registers for mma_B
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::load(a, As[tic][1], a_col_off);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need As[tic][1] in registers for mma_C
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto bs0_next = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
            kittens::load(b0, bs0_next);
            asm volatile("s_waitcnt vmcnt(4)"); // wait for toc data; As[toc][1] still in flight
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            tic ^= 1; toc ^= 1;
            tic_scales ^= 1; toc_scales ^= 1;
        }

        { // Final epilogue k = k_iters - 1
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);

            kittens::load(a, As[tic][0], a_col_off);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            auto bs1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
            kittens::load(b1, bs1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::load(a, As[tic][1], a_col_off);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

#ifdef MXFP8_AG_SKEW_CLOSE
        // Closes the warp_m skew the AG prologue opens, so the next queue tile starts even. Only
        // the fused AG kernels open that skew; the standalone GEMM has no persistent tile loop.
        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }
#endif
