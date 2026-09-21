/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Kernel setup for the NT bulk RS+GEMM, mirroring mxfp8_tn_prologue.inc. It DECLARES the names
//! the shared NT main loop uses (As, Bs, sw_A, a_srd, b_srd, warp_m, ...), so the include site is
//! where those come into scope. Unlike TN, neither operand is gathered, so both carry an SRD.
    using ST_A     = st_fp8e4m3<BLOCK_K, HALF_ROW, st_16x128_s>;
    using ST_B     = st_fp8e4m3<BLOCK_K, HALF_COL, st_16x128_s>;
    using RT_A     = rt<fp8e4m3, REG_M, BLOCK_K, col_l, rt_16x128_s>;
    using RT_B     = rt<fp8e4m3, REG_N, BLOCK_K, col_l, rt_16x128_s>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
    using RT_C_T   = rt_fl<REG_N, REG_M, row_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_A_smem[2], scale_B_lo[2], scale_B_hi[2];

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int wid    = kittens::warpid() % NUM_WARPS;

    using T = kittens::fp8e4m3;

    const auto &A_local = A;
    const auto &B_local = B;

    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = BLOCK_K * HALF_ROW * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_K * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];

    G::prefill_swizzled_offsets(As[0][0], A_local, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B_local, sw_B);

    const T *a_base = (const T *)&A_local[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B_local[{0, 0, 0, 0}];
    const int a_row_stride = A_local.template stride<2>() * sizeof(T);
    const int b_row_stride = B_local.template stride<2>() * sizeof(T);
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, A_local.rows() * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, B_local.rows() * b_row_stride, b_row_stride);

    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
    }

    const int a_col_off = warp_m * REG_M;
    const int b_col_off = warp_n * REG_N;
