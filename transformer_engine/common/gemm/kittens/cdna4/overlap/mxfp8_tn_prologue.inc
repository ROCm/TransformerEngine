/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Kernel setup for the TN fused AG+GEMM, mirroring upstream tn_prologue.inc. It DECLARES the
//! names the rest of the kernel body uses (As, Bs, a_frag0, warp_m, ...), so the include site
//! is where those come into scope.
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
    using RT_C_T   = rt_fl<REG_N, REG_M, row_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_smem[2][3];

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;

    using T = kittens::fp8e4m3;

    // Aliases: the AG path has no grouped variant, but the shared main loop spells its operands
    // A_local/B_local, which in mxfp8_gemm.cpp are the per-expert views.
    const auto &A_local = A;
    const auto &B_local = B;

    const uint32_t a_row_stride = A_local.template stride<2>() * sizeof(T);
    const uint32_t b_row_stride = B_local.template stride<2>() * sizeof(T);

    const uint32_t a_frag0 = row_frag_addr<RT_A>(As[0][0], warp_m, 0);
    const uint32_t a_frag1 = row_frag_addr<RT_A>(As[0][0], warp_m, 1);
    const uint32_t b_frag0 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 0);
    const uint32_t b_frag1 = row_frag_addr<RT_B>(Bs[0][0], warp_n, 1);

    const int n_iters = k_iters - 2;
    const int s0      = n_iters & 1;
