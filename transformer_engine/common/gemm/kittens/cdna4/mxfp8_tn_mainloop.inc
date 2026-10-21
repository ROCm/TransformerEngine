/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Main K-loop for the MXFP8 TN GEMM, shared by mxfp8_gemm_tn_kernel (mxfp8_gemm.cpp) and the
//! fused AG+GEMM (overlap/fused_ag_gemm_tn.cuh); only the C writeback is left to the caller.
//! Both operands are K-contiguous, so unlike the NN pair there is no layout mismatch here.
        asm volatile("s_mov_b32 m0, %0" :: "s"((s0 ^ 1) ? a_dma.m0[1][1] : a_dma.m0[0][1]) : "memory", "m0");

        int k = 0;
        if (s0) {
            constexpr int TIC = 1;
            constexpr int TOC = TIC ^ 1;
            load_rows<ST_B, 2 * TIC>(b0, b_frag0, b_frag1);
            load_rows<ST_A, 2 * TIC>(a, a_frag0, a_frag1);
            tile_dma_issue(a_dma, 1, k + 1);
            asm volatile("s_waitcnt lgkmcnt(8)"); // wait for scale + data LDS stores; As[toc][1] can overlap
            __builtin_amdgcn_s_barrier();

            asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.m0[TOC]) : "memory", "m0");
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            scale_dma_issue(s_dma, k + 1);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][0]) : "memory", "m0");
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_rows<ST_B, 2 * TIC + 1>(b1, b_frag0, b_frag1);
            tile_dma_issue(a_dma, 0, k + 2);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need bs1 in registers for mma_B
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][0]) : "memory", "m0");
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_rows<ST_A, 2 * TIC + 1>(a, a_frag0, a_frag1);
            tile_dma_issue(b_dma, 0, k + 2);
            asm volatile("s_waitcnt lgkmcnt(0)"); // drain LDS: need as1 in registers for mma_C
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][1]) : "memory", "m0");
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            tile_dma_issue(b_dma, 1, k + 2);
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data loads; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][1]) : "memory", "m0");
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            k++;
        }
        while (k < n_iters) {
            {
                constexpr int TIC = 0;
                constexpr int TOC = TIC ^ 1;
                load_rows<ST_B, 2 * TIC>(b0, b_frag0, b_frag1);
                load_rows<ST_A, 2 * TIC>(a, a_frag0, a_frag1);
                tile_dma_issue(a_dma, 1, k + 1);
                asm volatile("s_waitcnt lgkmcnt(8)");
                __builtin_amdgcn_s_barrier();

                asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.m0[TOC]) : "memory", "m0");
                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                scale_dma_issue(s_dma, k + 1);
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][0]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                load_rows<ST_B, 2 * TIC + 1>(b1, b_frag0, b_frag1);
                tile_dma_issue(a_dma, 0, k + 2);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][0]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_rows<ST_A, 2 * TIC + 1>(a, a_frag0, a_frag1);
                tile_dma_issue(b_dma, 0, k + 2);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][1]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                tile_dma_issue(b_dma, 1, k + 2);
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][1]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
            {
                constexpr int TIC = 1;
                constexpr int TOC = TIC ^ 1;
                load_rows<ST_B, 2 * TIC>(b0, b_frag0, b_frag1);
                load_rows<ST_A, 2 * TIC>(a, a_frag0, a_frag1);
                tile_dma_issue(a_dma, 1, k + 1);
                asm volatile("s_waitcnt lgkmcnt(8)");
                __builtin_amdgcn_s_barrier();

                asm volatile("s_mov_b32 m0, %0" :: "s"(s_dma.m0[TOC]) : "memory", "m0");
                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                scale_dma_issue(s_dma, k + 1);
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][0]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                load_rows<ST_B, 2 * TIC + 1>(b1, b_frag0, b_frag1);
                tile_dma_issue(a_dma, 0, k + 2);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][0]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_rows<ST_A, 2 * TIC + 1>(a, a_frag0, a_frag1);
                tile_dma_issue(b_dma, 0, k + 2);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(b_dma.m0[TIC][1]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                tile_dma_issue(b_dma, 1, k + 2);
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_mov_b32 m0, %0" :: "s"(a_dma.m0[TIC][1]) : "memory", "m0");
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
        }

        { // Epilogue k = k_iters - 2
            constexpr int TIC = 0;
            constexpr int TOC = TIC ^ 1;
            int k = k_iters - 2;
            scale_dma_load(s_dma, TOC, k + 1);
            asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM: last prefetch iteration
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_rows<ST_B, 2 * TIC>(b0, b_frag0, b_frag1);
            load_rows<ST_A, 2 * TIC>(a, a_frag0, a_frag1);
            tile_dma_load(a_dma, TOC, 1, k + 1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need as0/bs0 in registers for mma_A
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_rows<ST_B, 2 * TIC + 1>(b1, b_frag0, b_frag1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need bs1 in registers for mma_B
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_rows<ST_A, 2 * TIC + 1>(a, a_frag0, a_frag1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)"); // need as1 in registers for mma_C
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_rows<ST_B, 2 * TOC>(b0, b_frag0, b_frag1);
            asm volatile("s_waitcnt vmcnt(4)"); // wait for toc data; As[toc][1] still in flight
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        { // Final epilogue k = k_iters - 1
            constexpr int TIC = 1;
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_rows<ST_A, 2 * TIC>(a, a_frag0, a_frag1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_rows<ST_B, 2 * TIC + 1>(b1, b_frag0, b_frag1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_rows<ST_A, 2 * TIC + 1>(a, a_frag0, a_frag1);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_sched_barrier(0);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

#ifdef MXFP8_AG_SKEW_CLOSE
        // Closes the warp_m skew the AG prologue opens; the standalone GEMM never opens one.
        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }
#endif
