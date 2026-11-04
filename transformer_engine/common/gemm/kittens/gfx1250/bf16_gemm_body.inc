/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Body of bf16_gemm_kernel and bf16_gemm_kernel_nomc (bf16_gemm.cuh). In scope: the kernel
// template parameters and arguments, and MC (cluster with multicast fills).
    using Tl     = Tiles<T, L>;
    using A_deep = typename Tl::A_deep;
    using B_deep = typename Tl::B_deep;
    using A_reg  = typename Tl::A_reg;
    using B_reg  = typename Tl::B_reg;
    constexpr bool BGRAD = E == BGRADB;
    static_assert(!BGRAD || Tl::TRANS_B, "BGRADB is the NT (wgrad) epilogue");

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al(reinterpret_cast<int *>(&__shm[0]));
    A_deep &a0 = al.allocate_in<kittens::segment<0>, A_deep>();
    B_deep &b0 = al.allocate_in<kittens::segment<0>, B_deep>();
    A_deep &a1 = al.allocate_in<kittens::segment<0>, A_deep>();
    B_deep &b1 = al.allocate_in<kittens::segment<0>, B_deep>();
    A_deep *A_st[S] = {&a0, &a1};
    B_deep *B_st[S] = {&b0, &b1};

    kittens::sched::lock_simd();

    C_acc C;
    kittens::zero(C);
    float bsum[BGRAD ? B_FRAGS : 1] = {};

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    const int tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    const int wid     = kittens::warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int nb      = (K + BLOCK_K - 1) / BLOCK_K;
    int kb0 = 0, k_blocks = nb;      // this workgroup's K-blocks [kb0, kb0 + k_blocks)
    if constexpr (E == SPLIT) {
        const int splits = gridDim.z, z = blockIdx.z;
        kb0      = nb * z / splits;
        k_blocks = nb * (z + 1) / splits - kb0;
    }
    if (k_blocks <= 0) {
        return;
    }
    // Past the matrix. A workgroup that ends leaves the cluster barrier's member count (ISA 5.6.6).
    if (tile_m >= tiles_m || tile_n >= tiles_n) {
        return;
    }

    // The wave's slice of each operand tile: an element index (K-contiguous) or lane byte offset.
    const uint32_t warp_off_a = Tl::TRANS_A ? tr_lane_offset<A_deep>(warp_r * WARP_M)
                                            : uint32_t(warp_r * WARP_M * BLOCK_K);
    const uint32_t warp_off_b = Tl::TRANS_B ? tr_lane_offset<B_deep>(warp_c * WARP_N)
                                            : uint32_t(warp_c * WARP_N * BLOCK_K);

    // Multicast sets: WGs sharing tile_m receive A, those sharing tile_n receive B (ISA 10.7).
    uint32_t a_mask = 0, b_mask = 0;
    if constexpr (MC) {
        const uint32_t cx   = blockIdx.x & 3u;
        const uint32_t cy   = blockIdx.y & 3u;
        const uint32_t in_x = min(tiles_m - int(blockIdx.x & ~3u), CLUSTER_DIM);
        const uint32_t in_y = min(tiles_n - int(blockIdx.y & ~3u), CLUSTER_DIM);
        a_mask = (0x1111u << cx) & ((1u << (4u * in_y)) - 1u);
        b_mask = ((1u << in_x) - 1u) << (4u * cy);
    }

    /* One whole A tile (wave 0) and B tile (wave 1) per block; a K-major tile is indexed (k, row).
     * The extents count from the tile origin: kx K of the head, rows to M and N. Blocks after the
     * head start back elements early, through operands moved back by that much. */
    const int m_rem = M - tile_m * BLOCK_M;
    const int n_rem = N - tile_n * BLOCK_N;
    const int kh    = K % BLOCK_K;
    const int back  = kh ? BLOCK_K - kh : 0;
    gl_e<T> ga_k = ga, gb_k = gb;
    ga_k.raw_ptr -= size_t(back) * (Tl::TRANS_A ? M : 1);
    gb_k.raw_ptr -= size_t(back) * (Tl::TRANS_B ? N : 1);
    auto issue_fill = [&](const gl_e<T> &a, const gl_e<T> &b, int kx, int slot, int kblock,
                          uint32_t count) {
        if (wid == 0) {
            if constexpr (Tl::TRANS_A) {
                kittens::tdm::load_async(*A_st[slot], a, {0, 0, kblock, tile_m}, kx, m_rem, M,
                                         a_mask, count);
            } else {
                kittens::tdm::load_async(*A_st[slot], a, {0, 0, tile_m, kblock}, m_rem, kx, K,
                                         a_mask, count);
            }
        }
        if (wid == 1) {
            if constexpr (Tl::TRANS_B) {
                kittens::tdm::load_async(*B_st[slot], b, {0, 0, kblock, tile_n}, kx, n_rem, N,
                                         b_mask, count);
            } else {
                kittens::tdm::load_async(*B_st[slot], b, {0, 0, tile_n, kblock}, n_rem, kx, K,
                                         b_mask, count);
            }
        }
    };

    if (kb0 == 0) {
        issue_fill(ga, gb, kh ? kh : K, 0, 0, 1);
    } else {
        issue_fill(ga_k, gb_k, K, 0, kb0, 1);
    }
    kittens::sync::wait_tdm<0>();
    kittens::sched::compiler_fence();
    if constexpr (MC) {
        kittens::cluster::sync();
    } else {
        // The empty branches keep the cluster kernel's block structure, so the two K loops differ
        // only in the cluster barrier.
        kittens::sync::arrive();
        if (wid == 0) {
            asm volatile("");
        }
        kittens::sync::wait();
    }
    kittens::sched::compiler_fence();

    // Two register slots, named rather than indexed so they stay in registers; K steps alternate.
    struct operand_slot {
        A_reg a;
        B_reg b;
    };
    operand_slot slot0;
    operand_slot slot1;
    auto load_slot = [&]<int KS>(operand_slot &slot, int stage) {
        load_step<Tl::TRANS_A, KS>(slot.a, *A_st[stage], warp_off_a);
        load_step<Tl::TRANS_B, KS>(slot.b, *B_st[stage], warp_off_b);
    };
    // Column sums of K step KS, after its matrix ops so the per-step interleave is kept.
    auto bgrad = [&](int ks, const B_reg &b) {
        if constexpr (BGRAD) {
            if (tile_m == 0 && warp_r == ks) {
                colsum(bsum, b);
            }
        }
    };

    load_slot.template operator()<0>(slot0, 0);

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S;
        const int nxt = (kb + 1) % S;

        // A count of zero skips the tail transfer without a branch (TDM cannot be EXEC-masked).
        const int fill_k = kb0 + ((kb + 1 < k_blocks) ? kb + 1 : 0);
        issue_fill(ga_k, gb_k, K, nxt, fill_k, (kb + 1 < k_blocks) ? 1u : 0u);
        kittens::sched::compiler_fence();

        load_slot.template operator()<1>(slot1, cur);
        kittens::mma_ABt(C, slot0.a, slot0.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(0, slot0.b);

        load_slot.template operator()<2>(slot0, cur);
        kittens::mma_ABt(C, slot1.a, slot1.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(1, slot1.b);

        load_slot.template operator()<3>(slot1, cur);
        kittens::mma_ABt(C, slot0.a, slot0.b, C);
        pin_interleave();
        kittens::sched::compiler_fence();
        bgrad(2, slot0.b);

        // Last K step: PRE_BAR matrix ops overlap the wait and barrier, the rest the next reads.
        mma_zigzag<0, PRE_BAR>(C, slot1.a, slot1.b);
        kittens::sched::compiler_fence();

        kittens::sync::wait_ds<0>();
        kittens::sync::wait_tdm<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();
        if (wid == 0) {
            if constexpr (MC) {
                kittens::cluster::arrive();
            } else {
                asm volatile("");
            }
        }
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::sched::compiler_fence();

        load_slot.template operator()<0>(slot0, nxt);
        mma_zigzag<PRE_BAR, WMMA_STEP>(C, slot1.a, slot1.b);

        pin_interleave<(WMMA_STEP - PRE_BAR) / PIN_REP, DS_SUBSTEP / PIN_REP>();
        kittens::sched::compiler_fence();
        if constexpr (MC) {
            kittens::cluster::wait();
        }
        kittens::sched::compiler_fence();
        bgrad(3, slot1.b);
    }

    kittens::sync::wait_tdm<0>();
    kittens::sync::wait_ds<0>();

    // Accumulator lane L holds column L%16 and rows 8(L/16) + 0..7 of each 16x16 tile.
    using StT = staged_t<T, E, ACC, OutT>;
    int lane = kittens::laneid();
    if constexpr (L == NT && E == NONE && !std::is_same_v<StT, float>) {
        // Else the 16-bit staging reuses the transposed reads' lane terms past the K loop (+1 VGPR).
        asm volatile("" : "+v"(lane));
    }
    const int row0 = warp_r * WARP_M + 8 * (lane / 16);
    const int col0 = warp_c * WARP_N + lane % 16;
    auto runs = [&](auto &&f) {
        static_for<A_FRAGS>([&](auto i) {
            static_for<B_FRAGS>([&](auto j) {
                const auto &d = C.tiles[i][j].data;
                const v8f v{d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y};
                f(v, row0 + 16 * i, col0 + 16 * j);
            });
        });
    };
    // An edge tile stores only D's rows and columns.
    const bool edge = (tile_m + 1) * BLOCK_M > M || (tile_n + 1) * BLOCK_N > N || M % 8 != 0;
    if constexpr (E == SPLIT) {
        float *ws = reinterpret_cast<float *>(D) + size_t(blockIdx.z) * M * N +
                    size_t(tile_n) * BLOCK_N * M + size_t(tile_m) * BLOCK_M;
        auto store = [&](auto EDGE) {
            runs([&](v8f v, int row, int col) {
                if (!EDGE || (row < m_rem && col < n_rem)) {
                    *reinterpret_cast<v8f *>(ws + size_t(col) * M + row) = v;
                }
            });
        };
        edge ? store(std::true_type{}) : store(std::false_type{});
        return;
    }

    const uint32_t lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&__shm[0]));
    if constexpr (BGRAD) {
        // Partials [warp_r][lane half][column] in LDS; thread n adds column n's 8 in that order.
        if (tile_m == 0) {
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
            static_for<B_FRAGS>([&](auto t) {
                *lds_ptr<float>(lds0 + ((warp_r * 2 + lane / 16) * BLOCK_N + col0 + t * 16) * 4) =
                    bsum[t];
            });
            kittens::sync::wait_ds<0>();
            kittens::sync::sync();
            float sum = 0.f;
#pragma unroll
            for (int p = 0; p < WARPS_M * 2; p++) {
                sum += *lds_ptr<float>(lds0 + (p * BLOCK_N + threadIdx.x) * 4);
            }
            if (int(threadIdx.x) < n_rem) {
                bias[tile_n * BLOCK_N + threadIdx.x] = static_cast<SideT>(sum);
            }
        }
    }

    // Stage C column-major in LDS (D itself when PLAIN, else fp32), then copy out.
    constexpr int SSTR = BLOCK_M * sizeof(StT) + 16;
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    runs([&](v8f v, int row, int col) {
        *lds_ptr<vec8<StT>>(lds0 + col * SSTR + row * sizeof(StT)) =
            __builtin_convertvector(v, vec8<StT>);
    });
    kittens::sync::wait_ds<0>();
    kittens::sync::sync();
    auto out = [&](auto EDGE) {
        copy_out<StT, E, ACC, OutT, SideT, L != TN, decltype(EDGE)::value>(
            lds0, D, bias, aux, alpha, beta, M, N, tile_m, tile_n);
    };
    edge ? out(std::true_type{}) : out(std::false_type{});
