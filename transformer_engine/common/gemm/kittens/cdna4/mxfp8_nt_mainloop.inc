/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Main K-loop for the MXFP8 NT GEMM, shared by mxfp8_gemm_nt_kernel (mxfp8_gemm.cpp) and the
//! bulk RS+GEMM overlap; only the C writeback is left to the caller.
#pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, tic_scales ^= 1, toc_scales ^= 1) {
        if (k + 1 < k_iters) {
            G::load(scale_A_smem[toc_scales], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
            G::load(scale_B_lo[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
            G::load(scale_B_hi[toc_scales],   scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});
        }

        kittens::load(b0, Bs[tic][0], b_col_off);
        kittens::load(a, As[tic][0], a_col_off);
        G::load(As[toc][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
        kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);
        kittens::load(b1, Bs[tic][1], b_col_off);
        G::load(As[tic][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[tic][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
        kittens::load(a, As[tic][1], a_col_off);
        G::load(Bs[tic][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[tic][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data; next-iter prefetches in flight
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
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_A_smem[tic_scales], warp_m);
        kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_A_smem[tic_scales], 2 + warp_m);
        kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_B_lo[tic_scales], warp_n);
        kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_B_hi[tic_scales], warp_n);

        kittens::load(b0, Bs[tic][0], b_col_off);
        kittens::load(a, As[tic][0], a_col_off);
        G::load(As[toc][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::load(b1, Bs[tic][1], b_col_off);
        __builtin_amdgcn_s_barrier();

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
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, Bs[toc][0], b_col_off);
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

        kittens::load(b1, Bs[tic][1], b_col_off);
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
