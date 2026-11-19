/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
        tile_load_m0(b_dma, s0, 0, 0);
        tile_load_m0(a_dma, s0, 0, 0);
        tile_load_m0(b_dma, s0, 1, 0);
        tile_load_m0(a_dma, s0, 1, 0);

        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
        asm volatile("s_waitcnt vmcnt(4)"); // wait for tic[0] halves; tic[1] halves still in flight
        __builtin_amdgcn_s_barrier();

        tile_load_m0(a_dma, !s0, 0, 1);
        tile_load_m0(b_dma, !s0, 0, 1);
        tile_load_m0(b_dma, !s0, 1, 1);
        asm volatile("s_waitcnt vmcnt(6)"); // wait for tic[1] halves; 3 toc loads + scales in flight
        __builtin_amdgcn_s_barrier();

        scale_load_m0(s_dma, s0, 0);
        asm volatile("s_waitcnt vmcnt(0)"); // drain all VMEM before first MMA
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
