        for (int tile = 0; tile < k_tiles - 2; tile += 2) {

            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[0][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[0][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[1][1], A, {0, 0, block_row * 2 + 1, tile + 1}, sw_A, a_srd, a_base, a_lds_11);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[0][1], {warp_n, 0});
            load(b_tile_1, st_b);
            G_group::load(Bs[0][0], B, {0, 0, block_col * 2, tile + 2}, sw_B, b_srd, b_base, b_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[0][1], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[0][0], A, {0, 0, block_row * 2, tile + 2}, sw_A, a_srd, a_base, a_lds_00);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[1][0], {warp_n, 0});
            load(b_tile_0, st_b);
            G_group::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, tile + 2}, sw_B, b_srd, b_base, b_lds_01);
            asm volatile("s_waitcnt vmcnt(6)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[1][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[0][1], A, {0, 0, block_row * 2 + 1, tile + 2}, sw_A, a_srd, a_base, a_lds_01);
            asm volatile("s_waitcnt lgkmcnt(8)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[1][1], {warp_n, 0});
            load(b_tile_1, st_b);
            G_group::load(Bs[1][0], B, {0, 0, block_col * 2, tile + 3}, sw_B, b_srd, b_base, b_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[1][1], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[1][0], A, {0, 0, block_row * 2, tile + 3}, sw_A, a_srd, a_base, a_lds_10);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G_group::load(Bs[1][1], B, {0, 0, block_col * 2 + 1, tile + 3}, sw_B, b_srd, b_base, b_lds_11);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilogue: K-tile k_tiles-2 (stage tic)
        {
            int tile = k_tiles - 2;

            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            load(a_tile, st_a);
            G_group::load(As[toc][1], A, {0, 0, block_row * 2 + 1, tile + 1}, sw_A, a_srd, a_base, a_lds_11);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            load(b_tile_1, st_b);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            load(a_tile, st_a);
            asm volatile("s_waitcnt vmcnt(4)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        // Epilogue: K-tile k_tiles-1 (stage tic after flip)
        {
            auto st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][0], {warp_n, 0});
            load(b_tile_0, st_b);
            auto st_a = subtile_inplace<REG_M, K_STEP>(As[tic][0], {warp_m, 0});
            load(a_tile, st_a);
            asm volatile("s_waitcnt vmcnt(2)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c00, a_tile, b_tile_0, c00);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_b = subtile_inplace<REG_N, K_STEP>(Bs[tic][1], {warp_n, 0});
            load(b_tile_1, st_b);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c01, a_tile, b_tile_1, c01);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            st_a = subtile_inplace<REG_M, K_STEP>(As[tic][1], {warp_m, 0});
            load(a_tile, st_a);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(c10, a_tile, b_tile_0, c10);
            mma_ABt(c11, a_tile, b_tile_1, c11);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (warp_m == 0) {
            __builtin_amdgcn_s_barrier();
        }
