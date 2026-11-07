/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
        G::load(Bs[tic][0], B_local, {0, 0, b_half0, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
        G::load(As[tic][0], A_local, {0, 0, a_half0, 0}, sw_A);
        G::load(Bs[tic][1], B_local, {0, 0, b_half1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
        G::load(As[tic][1], A_local, {0, 0, a_half1, 0}, sw_A);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        asm volatile("s_waitcnt vmcnt(4)"); // wait for tic[0] halves; tic[1] halves still in flight
        __builtin_amdgcn_s_barrier();

        G::load(As[toc][0], A_local, {0, 0, a_half0, 1}, sw_A);
        G::load(Bs[toc][0], B_local, {0, 0, b_half0, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
        G::load(Bs[toc][1], B_local, {0, 0, b_half1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for tic[1] halves; 3 toc loads + scales in flight
        __builtin_amdgcn_s_barrier();

        G::load(scale_A_smem[0], scale_A_gl, {sa_batch, 0, 0, 0});
        G::load(scale_B_lo[0],   scale_B_gl, {2 * sb_batch,     0, 0, 0});
        G::load(scale_B_hi[0],   scale_B_gl, {2 * sb_batch + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM before first MMA
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
