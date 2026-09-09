/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
//! Kernel setup for the NN fused AG+GEMM: tile types, shared buffers, warp ids, swizzle
//! tables, buffer SRDs and LDS handles. Included after the operand dims are computed and before
//! the gather. Not a header: no include guard, and it is a statement sequence, not a declaration.
//! It DECLARES names the rest of the kernel body uses (As, Bs, sw_A, a_srd, warp_m, ...), so the
//! include site is where those come into scope. This mirrors upstream tn_prologue.inc.
    const int tiles_M = M / BLOCK_ROW;
    const int tiles_N = N_TOTAL / BLOCK_COL;

    // A is K-contiguous [M,K] as in TN; B is K-major [K,N] so its shared tile is transposed and
    // its registers are filled with the ds_read_*_tr path (col_l), matching mxfp8_gemm_nn_kernel.
    using ST_A     = st_fp8e4m3<HALF_ROW, K_STEP, st_16x128_s>;
    using ST_B     = st_fp8e4m3<K_STEP, HALF_COL, st_16x128_s>;
    using RT_A     = rt_fp8e4m3<REG_M, K_STEP>;
    using RT_B     = rt<fp8e4m3, REG_N, K_STEP, col_l, rt_16x128_s>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    // B needs 8 scale groups = 2 tiles
    __shared__ ST_Scale scale_A_smem[2], scale_B_lo[2], scale_B_hi[2];

    const int warp_m = kittens::warpid() / WARPS_COL;
    const int warp_n = kittens::warpid() % WARPS_COL;
    const int wid    = kittens::warpid() % NUM_WARPS;

    using T = kittens::fp8e4m3;

    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * K_STEP * sizeof(T) / bpm;
    constexpr int copies_B = K_STEP * HALF_COL * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G_group::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G_group::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    kittens::i32x4 a_srd = kittens::make_srsrc(a_base, (size_t)M * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = kittens::make_srsrc(b_base, (size_t)K * b_row_stride, b_row_stride);
    bf16 *c_base = (bf16 *)&C[{0, 0, 0, 0}];

    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
    }

    const int b_col_off = warp_n * REG_N;
