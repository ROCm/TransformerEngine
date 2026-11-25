/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The training API's backward kernel body from its round state on (moe_bwd_single.inc) to its last workgroup's reset;
// textually included.
  // Over capacity (nr > 1): each round from its dispatch to its drained tile loop, the fold in the last; single_round_*
  // derive its tables, rows and wgrad modes.
  if (fs_tid() == 0) {
    s_rr = 0, s_nrb = max(1, s_rhdr[ROUTE_HDR_NR]), s_nrow = s_rhdr[ROUTE_HDR_NROW], s_ngr = (int)gridDim.x;
  }
  for (int x = fs_tid(); x < MOE_BWD_MAX_EXPERTS; x += NUM_THREADS) {
    s_wm[x] = 1;
  }
  __syncthreads();
  if (*fs_lds(&s_nrb) > 1) {
    // Every peer's RFREE from this rank: started (its forward of this call is done), then after each round.
    if (wg < MOE_EP && fs_tid() == 0) {
      SArgsC *ka = fs_args(args);
      __hip_atomic_store((int *)ka->a.g.push.peers.sig[wg] +
                             hk_comm::bfree_index(MOE_EP + ka->a.g.push.my_rank, ka->a.g.push.sigb),
                         (int)((unsigned)bcall << 16 | 1u), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    single_round_open(args, s_tiles, (int)gridDim.x);
    single_round_info(args, &s_nrow, s_wm);
  }
  // In the round loop the thread id and the grid size are opaque (BTID, BGRID): masks and offsets derived from them
  // would be hoisted out of it and held across the tile loop.
#define BTID ((unsigned)fs_tid())
#define BGRID (*fs_lds(&s_ngr))
  // Never taken (s_rr is 0 here): a second entry into the round loop makes it irreducible, so the optimizer hoists no
  // loop-invariant value out of it (one held across the tile loop spills).
  if (s_rr < 0) {
    goto BWD_RMID;
  }
  for (;;) {
  if (wg < fs_args(args)->a.g.push.ncomm) {
    const int rr = *fs_lds(&s_rr);
    if (*fs_lds(&s_nrb) > 1 && fs_tid() == 0) {
      // A round's rows (and slots) go where the destination's previous round, or its forward, read them: its RFREE.
      // bfree cannot tell: on the workspace's first backward it is 0 before the peer starts.
      SArgsC *ka = fs_args(args);
      plan_wait(fs_g(ka->a.g.push.peers.sig[ka->a.g.push.my_rank]) + hk_comm::bfree_index(MOE_EP + wg % MOE_EP,
                                                                                           ka->a.g.push.sigb),
                (int)((unsigned)__builtin_amdgcn_readfirstlane(fs_g(ka->wep)[0]) << 16 | (unsigned)(rr + 1)), true);
    }
    hk_comm::moe_bwd_dispatch_push_role(
        &args->a.g.push, wg, SCALL(const unsigned short, SC_DOUT), SCALL(float, SC_DTK),
        fs_g(fs_args(args)->wep)[0], !rr && !fs_g(fs_args(args)->wep)[1],
        fs_uniform_ptr((const unsigned short *)*fs_lds(&s_tbcp[TRAIN_BWD_Y])));
    if (*fs_lds(&s_rg) == 2) {
      moe_bwd_xpush<true>(fs_args(args), wg, fs_uniform_ptr((const bf16 *)*fs_lds(&s_tbcp[TRAIN_BWD_X])),
                               *fs_lds(&s_nrb));
    }
    // The role's publishes are FLAT atomics (LGKM_CNT, out of order with LDS reads): drain before the tile loop.
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    if (BTID == 0) {
      (void)__hip_atomic_fetch_add(TCX + SINGLE_DISP_CTR, 1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    const int dn = fs_args(args)->a.g.push.ncomm;
    single_zero_pads<true>(single_a0<true>(A0, s_a0), fs_args(args), wg - dn, BGRID - dn,
                           TCX + SINGLE_ZERO_CTR,
                           *fs_lds(&s_rg) == 2 ? fs_args(args)->xrp[fs_args(args)->a.g.push.my_rank] : nullptr,
                           *fs_lds(&s_nrow), BTID);
  }
BWD_RMID:
  const int ne = fs_args(args)->E;

  using ST_A = st_bf<HALF_ROW, K_STEP_64, st_16x32_s>;
  using RT_A = rt_bf<REG_M, K_STEP_64, row_l, rt_16x32_s>;
  using RT_B = rt<bf16, REG_N, K_STEP_64, col_l, rt_16x32_s>;
  using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
  constexpr int B_HALF = HALF_COL * K_STEP_64;
  static_assert(4 * sizeof(ST_A) + 4 * B_HALF * sizeof(bf16) == LEGA_LDS_TILES, "NN tiles must fill the region");
  ST_A(&As)[2][2] = *reinterpret_cast<ST_A(*)[2][2]>(s_tiles);
  bf16(&Bs)[2][2][B_HALF] = *reinterpret_cast<bf16(*)[2][2][B_HALF]>(s_tiles + 4 * sizeof(ST_A));
  constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
  constexpr int bpm      = bpt * NUM_THREADS;
  constexpr int copies_A = HALF_ROW * K_STEP_64 * sizeof(bf16) / bpm;
  constexpr int B_INSTS  = (HALF_COL * K_STEP_64 * (int)sizeof(bf16)) / (16 * NUM_THREADS);
  static_assert(B_INSTS == copies_A, "B must issue the same number of memory ops as A");
  // bf16, the forward kept Xp only (s_rg): every workgroup first recomputes GEMM1 tiles, claimed from the count after
  // this call's row blocks, so each one precedes the dgrad2 and wgrad2 that wait on it, before deriving the queue's
  // long-lived state. A tile: Xp (gathered rows) x W1's gate and up halves, G into dG's buffer and A into ascr, written
  // through, then one add to its row block's count.
  if (*fs_lds(&s_rg)) {
    using RT_G1 = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
    constexpr uint32_t BO = 4 * sizeof(ST_A);
    ST_A(&Bt)[2][2] = *reinterpret_cast<ST_A(*)[2][2]>(s_tiles + BO);
    while (true) {
      SArgsC *ka = fs_args(args);
      const int nrow = *fs_lds(&s_nrow);
      const int H = ka->H, I = ka->I, N_TOTAL = 2 * I, tN1 = N_TOTAL / BLOCK_COL, nrb = (nrow - 1) / BLOCK_ROW;
      if (fs_tid() == 0) {
        const int t = __hip_atomic_fetch_add(fs_g(ka->rg_done) + nrb, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        int e = 0;
        while (e + 1 < ka->E && fs_g(ka->row_off)[e + 1] / BLOCK_ROW <= t / tN1) {
          e++;
        }
        s_rgt[0] = t, s_rgt[1] = e;
      }
      __syncthreads();
      const int t = __builtin_amdgcn_readfirstlane(s_rgt[0]), expert = __builtin_amdgcn_readfirstlane(s_rgt[1]);
      __syncthreads();
      if (t >= nrb * tN1) {
        break;
      }
      const int block_row = t / tN1, block_col = t - block_row * tN1;
      if (*fs_lds(&s_rg) == 2) {
        // The forward kept nothing: the block's X rows, re-sent by every source, and its zeroed pad rows.
        const int lane = fs_tid() % kittens::WARP_THREADS;
        if (fs_tid() < kittens::WARP_THREADS) {
          if (lane == 0) {
            const int nz = BGRID - ka->a.g.push.ncomm;
            int n = 0;
            while (__hip_atomic_load(TCX + SINGLE_ZERO_CTR, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) <
                   nz) {
              spin_backoff(n);
            }
          }
          if (lane < MOE_EP) {
            gcint *xs = fs_g((const int *)ka->xsg[ka->a.g.push.my_rank]);
            const int b0 = block_row * BLOCK_ROW - fs_g(ka->row_off)[expert], b1 = b0 + BLOCK_ROW;
            const int si = sig_index(lane, expert), c = fs_g(ka->a.g.cnt_by_src)[si];
            const int st = fs_g(ka->a.g.start_by_src)[si];
            if (c > 0 && st < b1 && st + c > b0) {
              const int need = min(b1, st + c) - max(b0, st);
              const int sb   = hk_comm::dsig_index(lane, block_row, ka->a.g.push.sigb);
              int n = 0;
              while (__hip_atomic_load(xs + sb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
                spin_backoff(n);
              }
            }
          }
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        }
        __syncthreads();
      }
      const FsGl A = fs_view(A0S, SAVED_XP, nrow, H);
      const bf16 *b_base = SCALL(const bf16, SC_W1) + (size_t)expert * N_TOTAL * H;
      const FsGl B_local = fs_view(A0S, b_base, N_TOTAL, H);
      const bf16 *a_base = fs_uniform_ptr(A.raw_ptr);
      const i32x4 a_srd = single_srd(a_base, (uint32_t)((size_t)nrow * H * 2), 0);
      const i32x4 b_srd = single_srd(b_base, (uint32_t)((size_t)N_TOTAL * H * 2), H * 2);
      const int ft = fs_tid(), lane = ft % kittens::WARP_THREADS, wv = ft / kittens::WARP_THREADS;
      const int warp_m = wv / WARPS_COL, warp_n = wv % WARPS_COL;
      const uint32_t lo = wv * (16 / sizeof(bf16)) * kittens::WARP_THREADS * sizeof(bf16);
      uint32_t al[2][2];
#pragma unroll
      for (int p = 0; p < 2; p++) {
#pragma unroll
        for (int q = 0; q < 2; q++) {
          al[p][q] = __builtin_amdgcn_readfirstlane((uint32_t)reinterpret_cast<uintptr_t>(&As[p][q].data[0]) + lo);
        }
      }
      int ga_row[copies_A], ga_col[copies_A];
      uint32_t sw_A[copies_A], sw_A1[copies_A], sw_B[copies_A];
      gather_parts<ST_A>(lane, wv, ga_row, ga_col);
      const auto *a_gather = fs_g(ka->a_gather);
#pragma unroll
      for (int i = 0; i < copies_A; i++) {
        sw_A[i] = MOE_SW_OFF(a_gather[block_row * BLOCK_ROW + ga_row[i]], H * 2, ga_col[i]);
        sw_A1[i] = MOE_SW_OFF(a_gather[block_row * BLOCK_ROW + HALF_ROW + ga_row[i]], H * 2, ga_col[i]);
        sw_B[i] = MOE_SW_OFF(ga_row[i], H * 2, ga_col[i]);
      }
      RT_G1 c00, c01, c10, c11;
      g1_kloop<true, BO>(As, Bt, A, B_local, a_srd, b_srd, a_base, b_base, sw_A, sw_A1, sw_B, al[0][0], al[0][1],
                         al[1][0], al[1][1], 0, 0, 0, 0, 2 * block_col, 2 * block_col + 1, H / K_STEP_64, lane,
                         warp_m, warp_n, c00, c01, c10, c11);
      if (wv / WARPS_COL == 0) {
        __builtin_amdgcn_s_barrier();
      }
      const int rf0 = __builtin_amdgcn_readfirstlane(wv / WARPS_COL);
      const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + wv / WARPS_COL);
      int lane_epi = lane;
      asm volatile("" : "+v"(lane_epi));
      bf16 *gb = fs_uniform_ptr(ka->dg) + (size_t)block_row * BLOCK_ROW * N_TOTAL;
      const int cg = __builtin_amdgcn_readfirstlane(block_col * 2 * WARPS_COL + wv % WARPS_COL);
      store_T16_bf16<true>(gb, c00, rf0 * REG_M, cg * REG_N, N_TOTAL, lane_epi);
      store_T16_bf16<true>(gb, c01, rf0 * REG_M, (cg + WARPS_COL) * REG_N, N_TOTAL, lane_epi);
      store_T16_bf16<true>(gb, c10, rf1 * REG_M, cg * REG_N, N_TOTAL, lane_epi);
      store_T16_bf16<true>(gb, c11, rf1 * REG_M, (cg + WARPS_COL) * REG_N, N_TOTAL, lane_epi);
      swiglu_inplace<ACT>(c00, c01);
      swiglu_inplace<ACT>(c10, c11);
      bf16 *ab = fs_uniform_ptr(ka->ascr) + (size_t)block_row * BLOCK_ROW * I;
      const int cf = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL + wv % WARPS_COL);
      store_T16_bf16<true>(ab, c00, rf0 * REG_M, cf * REG_N, I, lane_epi);
      store_T16_bf16<true>(ab, c10, rf1 * REG_M, cf * REG_N, I, lane_epi);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (fs_tid() == 0) {
        (void)__hip_atomic_fetch_add(fs_g(ka->rg_done) + block_row, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
  const int wid0 = warpid() % NUM_WARPS;
  const uintptr_t lds_warp_off = wid0 * elem_per_warp * sizeof(bf16);
  uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
  uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
  uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
  uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
  uint32_t b_lds_wo_k = 0;

  static_assert(MOE_WALK_GM == 4, "a wgrad run of 1 or 2 tiles divides the MOE_WALK_GM = 4 group");
  for (int x = BTID; x <= ne; x += NUM_THREADS) {
    s_rb0[x] = (x < ne ? fs_g(fs_args(args)->row_off)[x] : (*fs_lds(&s_nrow) - 1)) /
               BLOCK_ROW;
  }
  for (int x = BTID; x < ne; x += NUM_THREADS) {
    s_wkt[x] = (fs_g(fs_args(args)->cnt)[x] + K_STEP_64 - 1) / K_STEP_64;
  }
  if (BTID < MOE_BWD_MAX_EXPERTS / 32) {
    s_dgok[BTID] = 0u;
    s_rgok[BTID] = 0u;
  }
  if (BTID == 0) {
    s_acq_upto = -1;
    s_zok = 0;
    s_arr = 0;
    s_nwg = BGRID;
  }
  bool c_left  = wg < fs_args(args)->b.c.ncomm;
  bool f_early = c_left && (*fs_lds(&s_rr) + 1 >= *fs_lds(&s_nrb));
  if (c_left) {
    hk_comm::moe_bwd_combine_init(&args->b.c, wg, s_cdone);
  }
  __syncthreads();
  for (int x = BTID; x < ne; x += NUM_THREADS) {
    const int kx = s_wkt[x];
    int r = 0;
    for (int y = single_opaque(0); y < ne; y++) {
      const int ky = s_wkt[y];
      r += (ky > kx) || (ky == kx && y < x);
    }
    s_ord[r] = x;
  }
  __syncthreads();
  if (BTID == 0) {
    SArgsC *ka = fs_args(args);
    const int run2 = ka->a.w.run, run1 = ka->b.w.run;
    const int tNi = ka->I / BLOCK_COL, tNh = ka->H / BLOCK_COL, nnb2 = ka->H / BLOCK_ROW, nnb1 = 2 * ka->I / BLOCK_ROW;
    const int tpp2 = ((nnb2 + MOE_WALK_GM - 1) / MOE_WALK_GM) * tNi * (MOE_WALK_GM >> (run2 - 1));
    const int tpp1 = ((nnb1 + MOE_WALK_GM - 1) / MOE_WALK_GM) * tNh * (MOE_WALK_GM >> (run1 - 1));
    int acc = 0, q = 0;
    auto put = [&](int kind, int e) {
      s_seg[q] = acc;
      // Opaque under rounds: the kinds' constants would be hoisted out of the round loop and held across it.
      s_segk[q] = (short)((single_opaque(kind) << 8) | e);
      q++;
      const int mt = s_rb0[e + 1] - s_rb0[e];
      acc += kind == SK_DG2 ? mt * tNi : kind == SK_DG1 ? mt * tNh : kind == SK_WG2 ? tpp2 : tpp1;
    };
    // Each producer ahead of its consumers. Over capacity: an expert's wgrads in the rounds with its rows (s_wm).
    const int rr = *fs_lds(&s_rr);
    const bool rlast = rr + 1 >= *fs_lds(&s_nrb);
    int qn = 0;
    for (int e = 0; e < ne; e++) {
      put(SK_DG2, e);
      if (e > 0 && s_wm[e - 1]) {
        put(SK_WG2, e - 1);
      }
    }
    if (s_wm[ne - 1]) {
      put(SK_WG2, ne - 1);
    }
    // I > 1024: wgrad1 of experts < ne / 2 follows dgrad1 of the next expert, the rest longest K first. Shorter I puts
    // all wgrad1 after dgrad1, so the combine push starts as early as possible.
    const int m = ka->I > 1024 ? ne / 2 : 0;
    for (int e = 0; e < ne; e++) {
      put(SK_DG1, e);
      if (e > 0 && e <= m && s_wm[e - 1]) {
        put(SK_WG1, e - 1);
      }
    }
    for (int p = 0; p < ne; p++) {
      if (s_ord[p] >= m && (s_wm[s_ord[p]])) {
        put(SK_WG1, s_ord[p]);
      }
    }
    s_seg[q] = acc;
    s_nseg = q;
    s_ntiles = acc;
  }
  __syncthreads();
  if (fs_tid() < NUM_XCDS_AFF) {
    const int num_tiles = s_ntiles;
    const int bb = BTID, nu = (num_tiles + MOE_WALK_U - 1) / MOE_WALK_U;
    int c = bb < nu ? ((nu - bb + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF) * MOE_WALK_U : 0;
    if (bb < nu && (nu - 1) % NUM_XCDS_AFF == bb) {
      c -= nu * MOE_WALK_U - num_tiles;
    }
    s_cap[bb] = c;
  }
  __syncthreads();
  bool drained = false;
  int w_sub = 0, w_n = 0, w_kind = SK_WG2;
  int f_pend = -1;

  while (true) {
    if (c_left) {
      SArgsC *ka = fs_args(args);
      const int sel = hk_comm::moe_bwd_combine_probe(&args->b.c, wg, ka->H / BLOCK_COL, s_cdone, &s_csel);
      if (sel >= 0) {
        hk_comm::moe_bwd_combine_push_unit(&args->b.c, wg + sel * ka->b.c.ncomm);
        continue;
      }
      c_left = sel == -1;
    }
    if (w_sub < w_n) {
      // Ragged K: live rows rounded up to K_STEP_64; the <= 63 pad rows read add exact zeros.
      const int e    = __builtin_amdgcn_readfirstlane(s_desc[1]);
      const int nblk = __builtin_amdgcn_readfirstlane(s_desc[2]) + w_sub;
      const int mblk = __builtin_amdgcn_readfirstlane(s_desc[3]);
      const bool w1k = w_kind == SK_WG1;
      SArgsC *ka = fs_args(args);
      // The B operand from the call (its rows the record's); acc adds into dW1 and dW2.
      const FsGl aw = w1k ? fs_view(A0S, ka->dg, SAVED_NROW, 2 * ka->I) : A0S;
      lega_wgrad2_tile<true, true, true>(aw, nullptr, s_tiles, e, nblk, mblk,
                                                s_rb0[e] * (BLOCK_ROW / K_STEP_64), s_wkt[e],
                                                w1k ? SCALL(bf16, SC_DW1) : SCALL(bf16, SC_DW2),
                                                w1k ? SAVED_XP : SAVED_A, SAVED_NROW, w1k ? ka->H : ka->I,
                                                BLOCK_ROW + (s_wm[e] == 2),
                                                __builtin_amdgcn_readfirstlane(*fs_lds(&s_rhdr[TRAIN_BWD_ACC])));
      w_sub++;
      continue;
    }
    if (f_pend >= 0) {
      __syncthreads();
      if (BTID < 64 && !s_arr) {
        const bool a = single_fold_ready(args, TCX);
        if (BTID == 0) {
          s_arr = a;
        }
      }
      __syncthreads();
      if (s_arr) {
        {
          SArgsC *ka = fs_args(args);
          hk_comm::cw_fold_item<KS>(&args->b.s, ka->b.c.peers.comb[ka->b.c.my_rank], ka->H, f_pend,
                                    SCALL(bf16, SC_DX), SCALL(float, SC_DPR));
        }
        f_pend = -1;
      } else {
        __builtin_amdgcn_s_sleep(16);
      }
      continue;
    }
    if (f_early && !c_left && !drained) {
      __syncthreads();
      if (BTID < 64) {
        bool a = s_arr ||
                 (single_fold_ready(args, TCX));
        if (BTID == 0) {
          int i = -1;
          if (a) {
            s_arr = 1;
            i = atomicAdd(&TCX[SINGLE_FOLD_CTR], 1);
            i = i < hk_comm::cside_fold_items(fs_args(args)->b.s.T) ? i : -2;
          }
          s_desc[1] = i;
        }
      }
      __syncthreads();
      const int i = __builtin_amdgcn_readfirstlane(s_desc[1]);
      if (i >= 0) {
        f_pend = i;
        continue;
      }
      f_early = i == -1;
    }
    if (drained) {
      if (c_left) {
        __builtin_amdgcn_s_sleep(32);
        continue;
      }
      if (*fs_lds(&s_rr) + 1 < *fs_lds(&s_nrb)) {
        break;
      }
      __syncthreads();
      if (BTID == 0) {
        const int i = atomicAdd(&TCX[SINGLE_FOLD_CTR], 1);
        s_desc[1] = i < hk_comm::cside_fold_items(fs_args(args)->b.s.T) ? i : -1;
      }
      __syncthreads();
      f_pend = __builtin_amdgcn_readfirstlane(s_desc[1]);
      if (f_pend >= 0) {
        continue;
      }
      break;
    }
    __syncthreads();
    // Deadlock-free at any grid >= 9, all WGs resident: pushes wait only for peers to start, an unclaimed producer
    // sits ahead of its waiter in a bucket with a resident home WG, combine WGs never block on folds. Wave 0 claims.
    if (BTID < 64) {
      const int lane = (int)BTID;
      SArgsC *ka = fs_args(args);
      int *const tc = single_opaque(TCX);
      const int num_tiles = s_ntiles, nseg = s_nseg, home = single_opaque((int)blockIdx.x) % NUM_XCDS_AFF;
      const int tNi = ka->I / BLOCK_COL, tNh = ka->H / BLOCK_COL;
      int t = num_tiles, e = 0, rb = 0, col = 0, kind = 0, ntl = 0;
      if (lane == 0) {
        t = moe_walk_claim(tc, s_cap, home, num_tiles);
        if (t < num_tiles) {
          int lo = 0, hi = nseg - 1;
          while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (s_seg[mid] <= t) {
              lo = mid;
            } else {
              hi = mid - 1;
            }
          }
          kind = s_segk[lo] >> 8;
          e = s_segk[lo] & 0xff;
          const int li = t - s_seg[lo];
          // Keep the divisor loop-variant to the optimiser (else LICM hoists the udiv reciprocal into a VGPR).
          int tNv = (kind == SK_DG2 || kind == SK_WG2) ? tNi : tNh;
          asm volatile("" : "+v"(tNv));
          if (kind == SK_DG2 || kind == SK_DG1) {
            const int rb0 = s_rb0[e];
            const int mt  = s_rb0[e + 1] - rb0;
            const int band  = li / (MOE_WALK_GM * tNv);
            const int first = band * MOE_WALK_GM;
            const int gsz   = min(mt - first, MOE_WALK_GM);
            const int q     = li - band * (MOE_WALK_GM * tNv);
            rb = rb0 + first + q % gsz;
            col = q / gsz;
          } else {
            const int wrun = kind == SK_WG2 ? ka->a.w.run : ka->b.w.run, wsh = 3 - wrun;
            const int nnb = kind == SK_WG2 ? ka->H / BLOCK_ROW : 2 * ka->I / BLOCK_ROW;
            const int grp = li / (tNv << wsh);
            const int r   = li - grp * (tNv << wsh);
            col = r >> wsh;
            rb = grp * MOE_WALK_GM + (r & ((1 << wsh) - 1)) * wrun;
            ntl = max(0, min(nnb - rb, wrun));
          }
        }
      }
      t = __builtin_amdgcn_readfirstlane(t);
      e = __builtin_amdgcn_readfirstlane(e);
      rb = __builtin_amdgcn_readfirstlane(rb);
      kind = __builtin_amdgcn_readfirstlane(kind);
      if (t < num_tiles) {
        gcint *dg_done = fs_g(ka->dg_done);
        if (kind == SK_DG1) {
          if (lane == 0) {
            int n = 0;
            while (__hip_atomic_load(dg_done + rb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tNi) {
              spin_backoff(n);
            }
          }
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        } else if (kind == SK_WG1) {
          // dG of the whole expert (one row block per lane), acquired once per (workgroup, expert); the bit is set
          // only after the fence.
          if (!((s_dgok[e >> 5] >> (e & 31)) & 1u)) {
            for (int r = s_rb0[e] + lane; r < s_rb0[e + 1]; r += 64) {
              int n = 0;
              while (__hip_atomic_load(dg_done + r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tNi) {
                spin_backoff(n);
              }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if (lane == 0) {
              s_dgok[e >> 5] |= 1u << (e & 31);
            }
          }
        } else if (e > s_acq_upto) {
          gcint *g_sig          = fs_g(ka->a.g.sig);
          gcint *g_cnt_by_src   = fs_g(ka->a.g.cnt_by_src);
          gcint *g_start_by_src = fs_g(ka->a.g.start_by_src);
          gcint *g_row_off_e    = fs_g(ka->row_off);
          const int g_sigb = ka->a.g.push.sigb;
          const int s      = lane;
          if (!s_zok) {
            // Every single_zero_pads has published; the acquire fence below also orders this observation.
            if (lane == 0) {
              const int nz = s_nwg - ka->a.g.push.ncomm;
              int n = 0;
              while (__hip_atomic_load(tc + SINGLE_ZERO_CTR, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < nz) {
                spin_backoff(n);
              }
              s_zok = 1;
            }
          }
          if (kind == SK_WG2) {
            int x = s_acq_upto + 1;
            for (; x <= e; x++) {
              single_rows_landed(x, true, s, g_cnt_by_src, g_row_off_e, g_start_by_src, g_sig, g_sigb);
            }
            for (; x < ka->E && !__ballot(!single_rows_landed(x, false, s, g_cnt_by_src, g_row_off_e, g_start_by_src,
                                                              g_sig, g_sigb));
                 x++) {
            }
            // Every counter read above holds the value of the last release add into it: one system-scope acquire
            // orders every row of experts <= x - 1 for this workgroup.
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
            if (lane == 0) {
              s_acq_upto = x - 1;
            }
          } else if (s < MOE_EP) {
            const int b0 = rb * BLOCK_ROW - g_row_off_e[e];
            const int b1 = b0 + BLOCK_ROW;
            const int si = sig_index(s, e);
            const int c  = g_cnt_by_src[si];
            const int st = g_start_by_src[si];
            if (c > 0 && st < b1 && st + c > b0) {
              const int need = ((b1 < st + c) ? b1 : (st + c)) - ((b0 > st) ? b0 : st);
              const int sb   = hk_comm::dsig_index(s, rb, g_sigb);
              int n = 0;
              while (__hip_atomic_load(g_sig + sb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
                spin_backoff(n);
              }
            }
          }
          if (kind != SK_WG2) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
          }
        }
        // dgrad2 reads its row block's recomputed G, wgrad2 the recomputed A of its whole expert.
        if (*fs_lds(&s_rg) && (kind == SK_DG2 || (kind == SK_WG2 && !((s_rgok[e >> 5] >> (e & 31)) & 1u)))) {
          gcint *rd = fs_g(ka->rg_done);
          const int r0 = kind == SK_DG2 ? rb : s_rb0[e], r1 = kind == SK_DG2 ? rb + 1 : s_rb0[e + 1];
          for (int r = r0 + lane; r < r1; r += 64) {
            int n = 0;
            while (__hip_atomic_load(rd + r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < 2 * tNi) {
              spin_backoff(n);
            }
          }
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
          if (kind == SK_WG2 && lane == 0) {
            s_rgok[e >> 5] |= 1u << (e & 31);
          }
        }
      }
      if (lane == 0) {
        s_desc[0] = t;
        s_desc[1] = e;
        s_desc[2] = rb;
        s_desc[3] = col;
        s_desc[4] = kind;
        s_desc[5] = ntl;
      }
    }
    __syncthreads();
    const int tile_idx = __builtin_amdgcn_readfirstlane(s_desc[0]);
    if (tile_idx >= __builtin_amdgcn_readfirstlane(s_ntiles)) {
      drained = true;
      continue;
    }
    const int kd = __builtin_amdgcn_readfirstlane(s_desc[4]);
    if (kd == SK_WG2 || kd == SK_WG1) {
      w_kind = kd;
      w_sub = 0;
      w_n = __builtin_amdgcn_readfirstlane(s_desc[5]);
      continue;
    }
    const int expert    = __builtin_amdgcn_readfirstlane(s_desc[1]);
    const int block_row = __builtin_amdgcn_readfirstlane(s_desc[2]);
    const int block_col = __builtin_amdgcn_readfirstlane(s_desc[3]);
    const bool d2 = kd == SK_DG2;
    SArgsC *ka = fs_args(args);
    const FsGl A = d2 ? A0S : fs_view(A0S, ka->dg, SAVED_NROW, 2 * ka->I);
    const int N = d2 ? ka->I : ka->H;
    const int K = d2 ? ka->H : 2 * ka->I;
    const int KB = K;
    const bf16 *b_base_all = d2 ? SCALL(const bf16, SC_W2) : SCALL(const bf16, SC_W1);
    const long long w_stride_elems = (long long)K * N;
    const int *a_gather = fs_uniform_ptr(d2 ? ka->a_gather : ka->gid);
    const int M       = A.rows();
    const int k_tiles = K / K_STEP_64;
    // The B-load LDS addresses are derived per tile (else 8 of them are hoisted out of the loop and SGPR-spilled), from
    // the opaque thread id (a kernel-scope copy would hold an SGPR across the loop).
    const uint32_t b_lds_wo = single_opaque(__builtin_amdgcn_readfirstlane((uint32_t)((fs_tid() >> 6) %
                                                         NUM_WARPS * elem_per_warp * sizeof(bf16))));
    int lane_o = (int)BTID & 63;
    asm volatile("" : "+v"(lane_o));
    int tid_o = (int)BTID;
    asm volatile("" : "+v"(tid_o));
    const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
    const int warp_m = wv_o / WARPS_COL;
    const int warp_n = wv_o % WARPS_COL;
    const int wid    = wv_o % NUM_WARPS;
    int ga_row[copies_A], ga_col[copies_A];
    gather_parts<ST_A>(lane_o, kittens::warpid(), ga_row, ga_col);
    const int lane_b = lane_o;
    uint32_t b_lane_off[B_INSTS];
    {
      const int lane = lane_o;
#pragma unroll
      for (int i = 0; i < B_INSTS; i++) {
        const int q = (wid + i * NUM_WARPS) * 64 + lane;
        const int W = q / 32, within = q % 32;
        const int nb = W / 4, w = W % 4;
        const int r = within / 2, c0 = (within % 2) * 8;
        const int n  = nb * 16 + c0;
        const int kk = 32 * (w / 2) + 8 * (r / 4) + (r % 4) + 4 * (w % 2);
        b_lane_off[i] = (uint32_t)(((size_t)kk * N + n) * sizeof(bf16));
      }
    }
    const bf16 *a_base = (const bf16 *)&A[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(bf16);
    i32x4 a_srd = make_srsrc(a_base, (uint32_t)((size_t)M * a_row_stride), 0);
    const bf16 *b_base = b_base_all + (long long)expert * w_stride_elems;
    i32x4 b_srd = make_srsrc(b_base, (uint32_t)((size_t)KB * N * sizeof(bf16)));
    uint32_t sw_A[copies_A], sw_A1[copies_A];
    const int gbase = block_row * BLOCK_ROW;
#pragma unroll
    for (int i = 0; i < copies_A; i++) {
      const int r0 = a_gather[gbase + ga_row[i]];
      const int r1 = a_gather[gbase + HALF_ROW + ga_row[i]];
      sw_A[i] = MOE_SW_OFF(r0, a_row_stride, ga_col[i]);
      sw_A1[i] = MOE_SW_OFF(r1, a_row_stride, ga_col[i]);
    }
#include "moe_nn_kloop.inc"
    const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
    const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + warp_m);
    int lane_epi = kittens::laneid();
    asm volatile("" : "+v"(lane_epi));
    const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + warp_n);
    const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + warp_n);
    gint *pub = fs_g(fs_args(args)->b.c.tile_done);
    if (d2) {
      pub = fs_g(fs_args(args)->dg_done);
      const size_t roff = (size_t)block_row * BLOCK_ROW * (2 * N);
      const uint32_t tile_bytes = (uint32_t)(BLOCK_ROW * 2 * N * sizeof(bf16));
      const i32x4 g_srd  = make_srsrc((SCALL(const bf16, SC_G)) + roff, tile_bytes);
      const i32x4 dg_srd = make_srsrc(fs_uniform_ptr(fs_args(args)->dg) + roff,
                                      tile_bytes);
      const uint32_t lane_voff = (uint32_t)((RT_C::base_tile_stride * (lane_epi / RT_C::base_tile_cols) * 2 * N +
                                            lane_epi % RT_C::base_tile_cols) * 2);
      swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c00, rf0, cf0, 2 * N, lane_voff);
      swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c01, rf0, cf1, 2 * N, lane_voff);
      swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c10, rf1, cf0, 2 * N, lane_voff);
      swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c11, rf1, cf1, 2 * N, lane_voff);
    } else {
      // dgrad1: C staged through LDS as bf16 [256][256], stored 16 B per lane write-through (sc0 sc1), so the
      // tile_done add can be RELAXED (no L2 writeback per tile).
      const int c0 = __builtin_amdgcn_readfirstlane(warp_n), c1 = __builtin_amdgcn_readfirstlane(WARPS_COL + warp_n);
      {
        bf16 *tb = fs_uniform_ptr(fs_args(args)->dxp) + (size_t)block_row * BLOCK_ROW * N + block_col * BLOCK_COL;
        store_c_staged<MOE_AUX_WT>(s_tiles, c00, c01, c10, c11, rf0, rf1, c0, c1, lane_epi, tb, N);
      }
    }
    // Every wave's stores drained before the barrier, then one add for the whole tile: RELEASE after dG's write-back
    // stores, RELAXED after dXp's write-through ones.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (BTID == 0) {
      if (d2) {
        (void)__hip_atomic_fetch_add(pub + block_row, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      } else {
        (void)__hip_atomic_fetch_add(pub + block_row, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  // A middle launch's queue is its own: its last workgroup zeroes the claim counters and the exit counter only.
  if (*fs_lds(&s_rr) + 1 < *fs_lds(&s_nrb)) {
    single_round_end(args, TCX, s_tiles, *fs_lds(&s_rr), *fs_lds(&s_rb0[fs_args(args)->E]), *fs_lds(&s_nwg),
                     (unsigned)__builtin_amdgcn_readfirstlane(fs_g(fs_args(args)->wep)[0]));
    single_round_info(args, &s_nrow, s_wm);
    if (fs_tid() == 0) {
      s_rr++;
    }
    __syncthreads();
    continue;
  }
  if (single_last_wg(single_opaque(TCX) + NUM_XCDS_AFF, s_nwg)) {
    for (int i = BTID; i < SINGLE_NCTR; i += NUM_THREADS) {
      __hip_atomic_store(TCX + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    gint *tile_done = fs_g(fs_args(args)->b.c.tile_done);
    gint *dg_done   = fs_g(fs_args(args)->dg_done);
    int t_rs = (int)BTID;
    for (int i = t_rs; i < s_rb0[ne]; i += NUM_THREADS) {
      __hip_atomic_store(tile_done + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(dg_done + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    SArgsC *ka = fs_args(args);
    if (*fs_lds(&s_rg)) {
      for (int i = fs_tid(); i <= s_rb0[ne]; i += NUM_THREADS) {
        __hip_atomic_store(fs_g(ka->rg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    if (*fs_lds(&s_rg) == 2) {
      hk_comm::moe_bwd_dispatch_credit(ka->xsg[ka->a.g.push.my_rank], ka->a.g.cnt_by_src, ka->a.g.start_by_src,
                                       ka->row_off, ne, ka->a.g.push.sigb);
    }
    hk_comm::moe_bwd_dispatch_credit(ka->a.g.push.peers.sig[ka->a.g.push.my_rank], ka->a.g.cnt_by_src,
                                     ka->a.g.start_by_src, ka->row_off, ne, ka->a.g.push.sigb);
    hk_comm::moe_bwd_combine_wait(&args->b.c);
    if (fs_tid() == 0 && *fs_lds(&s_nrb) > 1) {
      __hip_atomic_store(fs_g(ka->rinfo) + RINFO_GO, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (ka->pool) {
      pool_release(args, (int *)s_tiles);
    }
    if (BTID == 0) {
      __hip_atomic_store(fs_uniform_ptr((int *)*fs_lds(&s_tbcp[TRAIN_BWD_CTR])) + INST_CTR_BWD, 0, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    // Every workgroup has read the words: the next backward's count, no forward since, this layer's arena released.
    if (BTID == 0) {
      gint *w = (gint *)fs_g(ka->wep);
      (void)__hip_atomic_fetch_add(w, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(w + 1, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      (void)__hip_atomic_fetch_add((gint *)fs_g(ka->ep) + 1, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  break;
  }
#undef BTID
#undef BGRID
