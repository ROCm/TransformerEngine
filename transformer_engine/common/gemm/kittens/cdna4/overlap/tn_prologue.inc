    using ST_A = st_bf<HALF_ROW, K_STEP, st_16x32_s>;
    using ST_B = st_bf<HALF_COL, K_STEP, st_16x32_s>;
    using RT_A = rt_bf<REG_M, K_STEP, row_l, rt_16x32_s>;
    using RT_B = rt_bf<REG_N, K_STEP, row_l, rt_16x32_s>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
    const int wid    = warpid() % NUM_WARPS;

    using T = bf16;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * K_STEP * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * K_STEP * sizeof(T) / bpm;

    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    const uintptr_t lds_warp_off = wid * elem_per_warp * sizeof(T);

    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + lds_warp_off));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + lds_warp_off));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + lds_warp_off));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + lds_warp_off));

    uint32_t sw_A[copies_A], sw_B[copies_B];
    G_group::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G_group::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base        = (const T *)&A[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd            = make_srsrc(a_base, (size_t)M * a_row_stride, a_row_stride);
    const T *b_base        = (const T *)&B[{0, 0, 0, 0}];
    i32x4 b_srd            = make_srsrc(b_base, (size_t)N_TOTAL * b_row_stride, b_row_stride);
