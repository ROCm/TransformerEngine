/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Main K-loop for the MXFP8 NT GEMM, shared by mxfp8_gemm_nt_kernel (mxfp8_gemm.cpp) and the
//! bulk RS+GEMM overlap (overlap/bulk_rs_gemm_nt.cuh); only the C writeback is left to the caller.
//! Split-K is the caller's business: it offsets A_local/B_local and sa_batch/sb_batch per slice.
        int k = 0;
        if (s0) {
            constexpr int TIC = 1;
            constexpr int TOC = TIC ^ 1;
            G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
            G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
            G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
            asm volatile("s_waitcnt vmcnt(6)"); // wait for toc data; next-iter prefetches in flight
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            k++;
        }
        while (k < n_iters) {
            {
                constexpr int TIC = 0;
                constexpr int TOC = TIC ^ 1;
                G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
                G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
                G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

                load_cols<ST_B, 2 * TIC>(b0, b_frag);
                load_cols<ST_A, 2 * TIC>(a, a_frag);
                load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
                load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
                load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
            {
                constexpr int TIC = 1;
                constexpr int TOC = TIC ^ 1;
                G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
                G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
                G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});

                load_cols<ST_B, 2 * TIC>(b0, b_frag);
                load_cols<ST_A, 2 * TIC>(a, a_frag);
                load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
                kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);
                load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
                load_srd(As[TIC][0], A_local, {0, 0, k + 2, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[TIC][0]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
                load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
                load_srd(Bs[TIC][0], B_local, {0, 0, k + 2, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[TIC][0]);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                load_srd(Bs[TIC][1], B_local, {0, 0, k + 2, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[TIC][1]);
                asm volatile("s_waitcnt vmcnt(6)");
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(2);
                kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            }
            k++;
        }

        // Opaque single-trip loops make the peeled stages' MFMAs accumulate in place; without them a
        // persistent caller copies the accumulators and spills. Plain blocks for the standalone GEMM.
#ifdef MXFP8_NT_OPAQUE_PEEL
        int once = 1;
        asm volatile("" : "+s"(once));
#define MXFP8_NT_PEEL for (int o = once; o > 0; o--)
#else
#define MXFP8_NT_PEEL
#endif
        MXFP8_NT_PEEL { // Epilogue k = k_iters - 2
            constexpr int TIC = 0;
            constexpr int TOC = TIC ^ 1;
            int k = k_iters - 2;
            G::load(scale_smem[TOC][0], scale_A_gl, {(k + 1) * sa_stride + sa_batch, 0, 0, 0});
            G::load(scale_smem[TOC][1], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch),     0, 0, 0});
            G::load(scale_smem[TOC][2], scale_B_gl, {2 * ((k + 1) * sb_stride + sb_batch) + 1, 0, 0, 0});
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_cols<ST_B, 2 * TIC>(b0, b_frag);
            load_cols<ST_A, 2 * TIC>(a, a_frag);
            load_srd(As[TOC][1], A_local, {0, 0, k + 1, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[TOC][1]);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_B, 2 * TOC>(b0, b_frag);
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

        MXFP8_NT_PEEL { // Final epilogue k = k_iters - 1
            constexpr int TIC = 1;
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            kittens::fp8e8m0_4 sa_h0 = lane_rd(scale_smem[TIC][0], warp_m);
            kittens::fp8e8m0_4 sa_h1 = lane_rd(scale_smem[TIC][0], 2 + warp_m);
            kittens::fp8e8m0_4 sb_h0 = lane_rd(scale_smem[TIC][1], warp_n);
            kittens::fp8e8m0_4 sb_h1 = lane_rd(scale_smem[TIC][2], warp_n);

            load_cols<ST_A, 2 * TIC>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cA, a, b0, cA, &sa_h0, &sb_h0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_B, 2 * TIC + 1>(b1, b_frag);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cB, a, b1, cB, &sa_h0, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_cols<ST_A, 2 * TIC + 1>(a, a_frag);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cC, a, b0, cC, &sa_h1, &sb_h0);
            kittens::mma_ABt_scaled<CBSZ, BLGP>(cD, a, b1, cD, &sa_h1, &sb_h1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
#undef MXFP8_NT_PEEL
