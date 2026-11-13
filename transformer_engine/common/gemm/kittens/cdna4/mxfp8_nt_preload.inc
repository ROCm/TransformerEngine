/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
        load_srd(Bs[s0][0], B_local, {0, 0, 0, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[s0][0]);
        load_srd(As[s0][0], A_local, {0, 0, 0, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[s0][0]);
        load_srd(Bs[s0][1], B_local, {0, 0, 0, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[s0][1]);
        load_srd(As[s0][1], A_local, {0, 0, 0, a_row_tile * 2 + 1}, sw_A, a_srd, a_base, a_lds[s0][1]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        load_srd(As[!s0][0], A_local, {0, 0, 1, a_row_tile * 2    }, sw_A, a_srd, a_base, a_lds[!s0][0]);
        load_srd(Bs[!s0][0], B_local, {0, 0, 1, n_tile * 2    }, sw_B, b_srd, b_base, b_lds[!s0][0]);
        load_srd(Bs[!s0][1], B_local, {0, 0, 1, n_tile * 2 + 1}, sw_B, b_srd, b_base, b_lds[!s0][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        G::load(scale_smem[s0][0], scale_A_gl, {sa_batch, 0, 0, 0});
        G::load(scale_smem[s0][1], scale_B_gl, {2 * sb_batch,     0, 0, 0});
        G::load(scale_smem[s0][2], scale_B_gl, {2 * sb_batch + 1, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
