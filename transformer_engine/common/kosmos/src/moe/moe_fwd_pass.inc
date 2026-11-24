/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// moe_fwd_single from its first pass on: each pass (RND: of a call in rounds) to its drained tile loop, then the
// last workgroup's reset; textually included.
  // Each pass (RND: the overflow layout's, then one per round) to its drained tile loop. Values the two copies could
  // share come opaquely (fs_tid, s_ngr, fs_args, fs_opaque_s): held across the other copy, they spill.
  for (;;) {
  {
    char *scr = reinterpret_cast<char *>(&As[0][0]);
    MoeLayout *L = reinterpret_cast<MoeLayout *>(scr + PLAN_LAYOUT_OFF);
    FsArgsC *ka = fs_args(args);
    const int ng = *fs_lds(&s_ngr);
    const int dn = ka->dncomm, d0 = ng > dn ? dn : 0, nw = ng - d0;
    const int rnd = RND ? *fs_lds(&s_rnd) : 0;
    if (rnd > 0) {
      moe_plan_layout<false, true, true>(ka, L, (unsigned)fk_epoch<true>(epoch, s_kp, 0), rnd - 1,
                                             RND ? *fs_lds(&s_fz) : 0);
    }
    if (tid == 0) {
      const int nrb = L->mpad / BLOCK_ROW;
      s_mpad = L->mpad;
      s_nt[0] = nrb * (2 * ka->I / BLOCK_COL);
      s_nt[1] = nrb * (ka->H / BLOCK_COL);
      s_cur = L->cur, s_gr = L->gr, s_ar = L->ar;
      if constexpr (RND) {
        s_nr = L->nr;
      }
      // The kept tier, alike on every rank (Mpad_max): nothing over capacity or past the caller's rows.
      if (!rnd) {
        int *vi = train_ints();
        unsigned long long *vp = train_ptrs();
        const int rq = vp[TRAIN_PTR_ROUTE] ? vi[TRAIN_INT_TIER] : -1;
        const int kept = rq < 0 ? -1 : L->nr > 1 || L->mpad_max > vi[TRAIN_INT_ROWS] ? 0 : rq;
        vi[TRAIN_INT_KEPT] = kept;
        vi[TRAIN_INT_MPAD] = L->mpad, vi[TRAIN_INT_MPADX] = L->mpad_max, vi[TRAIN_INT_NR] = L->nr,
            vi[TRAIN_INT_CROWS] = L->crows;
        vp[TRAIN_PTR_G_OUT] = kept == 2 ? vp[TRAIN_PTR_G] : 0ull;
        vp[TRAIN_PTR_A_OUT] = kept == 2 ? vp[TRAIN_PTR_A] : (unsigned long long)ka->act;
        vp[TRAIN_PTR_Y_OUT] = kept >= 0 ? vp[TRAIN_PTR_Y] : 0ull;
      }
    }
    if (wg >= d0) {
      moe_plan_queues<RND>(ka, L);
      moe_plan_tables<true, true>(ka, L, wg - d0, nw);
      fs_index_phase<RND>(ka, L, (RND ? fs_opaque_s(wg) : wg) - d0, nw, !rnd);
    }
    if (RND && rnd && wg < dn && tid == 0) {
      // A round's rows go where the destination's previous pass read them: wait for its RFREE.
      plan_wait(fs_g(ka->my_sig) + RFREE_OFF + wg % MOE_EP,
                (int)((unsigned)fk_epoch<true>(epoch, s_kp, 0) << 16 | (unsigned)rnd), true);
    }
    if (wg < dn) {
      fs_dispatch_role<BURST == 0, true>(
          ka, L, RND ? fs_uniform_ptr((const bf16 *)*fs_lds(&s_kp[FK_X])) : FKP(const bf16, X, FK_X), wg, ka->H,
          fk_epoch<true>(xepoch, s_kp, 32));
      asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        __hip_atomic_fetch_add(fs_g(ka->disp_done), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    if (wg >= d0 && !rnd) {
      moe_plan_late<true, true>(fs_args(args), L, wg - d0, nw);
      // The record's copies wait for every plan workgroup's tables.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        (void)__hip_atomic_fetch_add(train_ptr<int>(TRAIN_PTR_CTR) + INST_CTR_LATE, 1, __ATOMIC_RELEASE,
                                     __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
FWD_RMID:

  if (tid == 0) {
    FsArgsC *jp = fs_args(args);
    auto *ix = fs_g(jp->idx_done);
    int n = 0;
    const int ng = *fs_lds(&s_ngr);
    const int nw = ng > jp->dncomm ? ng - jp->dncomm : ng;
    while (__hip_atomic_load(ix, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < nw) {
      spin_backoff(n);
    }
    (void)__hip_atomic_load(ix, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();

  constexpr int SZA = sizeof(ST_A);  // As[p][q] = As[0][0] + (2p + q) * SZA: the fragment loads' immediate
  constexpr int SZB = sizeof(ST_B);
  const int wid = (warpid()) % NUM_WARPS;
  using T = bf16;
  constexpr int bpt           = ST_A::underlying_subtile_bytes_per_thread;
  constexpr int bpm           = bpt * NUM_THREADS;
  constexpr int copies_A      = HALF_ROW * K_STEP_64 * sizeof(T) / bpm;
  constexpr int copies_B      = HALF_COL * K_STEP_64 * sizeof(T) / bpm;
  constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
  const uintptr_t lds_warp_off = wid * elem_per_warp * sizeof(T);
  uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + lds_warp_off));
  uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + lds_warp_off));
  uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + lds_warp_off));
  uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + lds_warp_off));
  uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + lds_warp_off));
  uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + lds_warp_off));
  uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + lds_warp_off));
  uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + lds_warp_off));
  uint32_t sw_A[copies_A], sw_A1[copies_A];

  __shared__ int s_cap[NUM_XCDS_AFF];
  __shared__ int s_desc[5];
  __shared__ int s_g1done;
  __shared__ unsigned s_dmask[2];
  __shared__ int s_dsel[1];
  __shared__ int s_dall;
  __shared__ int s_arr;
  __shared__ int s_frd;
  __shared__ int s_fsel;
  bool c_left = wg < fs_args(args)->cncomm;
  const int ts = fs_tid();
  if (ts < 2) {
    s_dmask[ts] = 0u;
  }
  if (ts == 0) {
    s_arr = 0;
    s_g1done = false;
    s_dall = 0;
  }
  moe_walk_caps(s_cap, s_nt[0], ts);
  __syncthreads();

  bool drained = false;
  bool f_early = c_left && (!RND || *fs_lds(&s_rnd) + 1 == *fs_lds(&s_nr));
  int f_pend = -1;
  while (true) {
    if (c_left) {
      FsArgsC *cp = fs_args(args);
      const int sel = fs_combine_claim(cp, wg, fs_lds(s_dmask), fs_lds(s_dsel), fs_lds(&s_dall));
      if (sel >= 0) {
        cp = fs_args(args);
        fs_combine_push<BURST, !RND>(cp, sel, cp->H, s_kp);
        continue;
      }
      c_left = sel == -1;
      if (!c_left && tid == 0) {
        __hip_atomic_fetch_add(fs_g(cp->fold_ctr) + 1, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      if (sel == -1 && drained) {
        __builtin_amdgcn_s_sleep(32);
        continue;
      }
    }
    if (f_pend >= 0) {
      __syncthreads();
      if (tid < 64 && !*fs_lds(&s_arr)) {
        FsArgsC *fp = fs_args(args);
        const bool ok = moe_fold_arrived((const int *)fs_my_acc(fp), fp->owed, fp->fold_ctr + 1, fp->cncomm,
                                         nullptr, fp->T / BLOCK_ROW * (fp->H / BLOCK_COL));
        bool rd = ok;
        if constexpr (!RND) {
          // One pass: an item may also start once its own rows have landed (fs_fold_block_ready), before every
          // source's count; rounds wait for the counts (a unit may land over several rounds).
          rd = ok || fs_fold_block_ready(fp, s_kp, f_pend);
        }
        if (tid == 0) {
          *fs_lds(&s_arr) = ok;
          if constexpr (!RND) {
            *fs_lds(&s_frd) = rd;
          }
        }
      }
      __syncthreads();
      if (*fs_lds(&s_arr) || (!RND && *fs_lds(&s_frd))) {
        FsArgsC *fp = fs_args(args);
        __builtin_amdgcn_s_waitcnt(0);
        moe_fold_item<KS, true, true>(fp->lpeers.comb[fp->my_rank], fp->wts,
                                                      FKP(bf16, fold_out, FK_OUT), fp->T, fp->H, fp->topk, f_pend,
                                                      nullptr, train_ptr<bf16>(TRAIN_PTR_Y_OUT));
        f_pend = -1;
      } else {
        __builtin_amdgcn_s_sleep(16);
      }
      continue;
    }
    if (f_early && !c_left && !drained) {
      __syncthreads();
      if (tid < 64) {
        FsArgsC *fp = fs_args(args);
        const bool ok =
            *fs_lds(&s_arr) || moe_fold_arrived((const int *)fs_my_acc(fp), fp->owed, fp->fold_ctr + 1, fp->cncomm,
                                                nullptr, fp->T / BLOCK_ROW * (fp->H / BLOCK_COL));
        if (tid == 0) {
          int i = -1;
          if (ok) {
            *fs_lds(&s_arr) = fs_imm<1>();
            i = __hip_atomic_fetch_add(fs_g(fp->fold_ctr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            i = i < (fp->T + FOLD_TOK - 1) / FOLD_TOK ? i : -2;
          }
          *fs_lds(&s_fsel) = i;
        }
      }
      __syncthreads();
      const int i = __builtin_amdgcn_readfirstlane(*fs_lds(&s_fsel));
      if (i >= 0) {
        f_pend = i;
        continue;
      }
      f_early = i == -1;
    }
    if (drained) {
      if (RND && *fs_lds(&s_rnd) + 1 < *fs_lds(&s_nr)) {
        break;
      }
      {
        // The call's last pass: the record's and Xp's copy items run in the combine's tail, before any fold item.
        const int ng = *fs_lds(&s_ngr), dn = fs_args(args)->dncomm;
        if (route_fwd_copy(args, train_ptr<uint8_t>(TRAIN_PTR_ROUTE), train_ptr<int>(TRAIN_PTR_CTR),
                           train_ptr<const RouteSeg>(TRAIN_PTR_SEG), train_ptr<bf16>(TRAIN_PTR_XP),
                           train_int(TRAIN_INT_NSEG), train_int(TRAIN_INT_OFFSLOT), train_int(TRAIN_INT_KEPT),
                           train_int(TRAIN_INT_MPAD), train_int(TRAIN_INT_NR), train_int(TRAIN_INT_CROWS),
                           (unsigned)fk_epoch<true>(epoch, s_kp, 0), ng > dn ? ng - dn : ng,
                           &train_ints()[TRAIN_INT_ITEM])) {
          continue;
        }
      }
      __syncthreads();
      if (tid == 0) {
        FsArgsC *fp = fs_args(args);
        const int i = __hip_atomic_fetch_add(fs_g(fp->fold_ctr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        *fs_lds(&s_fsel) = i < (fp->T + FOLD_TOK - 1) / FOLD_TOK ? i : -1;
      }
      __syncthreads();
      f_pend = __builtin_amdgcn_readfirstlane(*fs_lds(&s_fsel));
      if (f_pend >= 0) {
        continue;
      }
      break;
    }
    __syncthreads();
    if (tid == 0) {
      FsArgsC *cp = fs_args(args);
      const int nt1 = *fs_lds(&s_nt[0]), tN1 = 2 * cp->I / BLOCK_COL;
      auto *a_done = fs_g(cp->a_done);
      int t = nt1, kind = FS_G1, dr = 0;
      int e = 0, rb = 0, col = 0;
      if (!*fs_lds(&s_g1done)) {
        t = moe_walk_claim(tile_counter, fs_lds(s_cap), (RND ? fs_opaque_s(wg) : wg) % NUM_XCDS_AFF, nt1);
      }
      if (t < nt1) {
        const auto *qd = fs_g(cp->q1) + t;
        e = qd->expert, rb = qd->tile_m, col = qd->tile_n;
      } else {
        *fs_lds(&s_g1done) = fs_imm<1>();
        kind = FS_G2;
        t = __hip_atomic_fetch_add(fs_g(cp->g2_ctr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (t < *fs_lds(&s_nt[1])) {
          const auto *qd = fs_g(cp->q2) + t;
          e = qd->expert, rb = qd->tile_m, col = qd->tile_n;
          int n = 0;
          while (__hip_atomic_load(a_done + rb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tN1) {
            spin_backoff(n);
          }
          (void)__hip_atomic_load(a_done + rb, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
          dr = 1;
        }
      }
      int *sd = fs_lds(s_desc);
      sd[0] = dr;
      sd[1] = e;
      sd[2] = rb;
      sd[3] = col;
      sd[4] = kind;
    }
    __syncthreads();
    const int *sd = fs_lds(s_desc);
    if (__builtin_amdgcn_readfirstlane(sd[0])) {
      drained = true;
      continue;
    }
    const int el        = __builtin_amdgcn_readfirstlane(sd[1]);
    const int block_row = __builtin_amdgcn_readfirstlane(sd[2]);
    const int block_col = __builtin_amdgcn_readfirstlane(sd[3]);
    const int kd        = __builtin_amdgcn_readfirstlane(sd[4]);
    if (kd == FS_G1) {
      if (fs_tid() < MOE_EP) {
        FsArgsC *gp = fs_args(args);
        const int s  = fs_tid();
        const int b0 = block_row * BLOCK_ROW - fs_g(gp->row_off)[el];
        const int b1 = b0 + BLOCK_ROW;
        const int c  = fs_g(gp->cnt_by_src)[sig_index(s, el)];
        const int st = fs_g(gp->start_by_src)[sig_index(s, el)];
        const auto *ms = fs_g(gp->my_sig);
        if (c > 0 && st < b1 && st + c > b0) {
          const unsigned want = ((unsigned)fk_epoch<true>(epoch, s_kp, 0) + 1u) * (unsigned)(gp->dncomm / MOE_EP);
          int n = 0;
          while ((int)((unsigned)__hip_atomic_load(ms + sig_index(s, el), __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_SYSTEM) - want) < 0) {
            spin_backoff(n);
          }
          (void)__hip_atomic_load(ms + sig_index(s, el), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      __syncthreads();
      using RT_C = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
      FsArgsC *ap = fs_args(args);
      const int H = ap->H, I = ap->I;
      // The pass's rows start at row s_cur (0 in one pass), in A's view (dma_load's offsets are A's addresses less
      // a_base); the epilogue reads it again: kept live across the K loop, it raises the SGPR peak.
      FsGl A(A0);
      {
        const int cur = !RND ? 0 : __builtin_amdgcn_readfirstlane(*fs_lds(&s_cur));
        A = fs_view(A0, A0.raw_ptr + (size_t)cur * H, A0.rows(), H);
      }
      const bf16 *w1p = FKP(const bf16, W1, FK_W1);
      FsGl B_local(fs_view(A0, w1p, 2 * I, H));
      const T *w_base = fs_uniform_ptr(w1p);
      const long long w_stride_elems = (long long)2 * I * H;
      bf16 *out = fs_uniform_ptr(ap->act);
      const int K            = H;
      const int N_TOTAL      = 2 * I;
      const int k_tiles      = K / K_STEP_64;
      const int M            = A.rows();
      const int a_row_stride = K * sizeof(T);
      const int b_row_stride = K * sizeof(T);
      const T *a_base = fs_uniform_ptr((const T *)&A[{0, 0, 0, 0}]);
      i32x4 a_srd = fs_srd(a_base, (uint32_t)((size_t)M * a_row_stride), 0);
      const T *b_base = w_base;
      i32x4 b_srd;
      uint32_t sw_B[copies_B];
      const auto *a_gather = fs_g(ap->gather);
      b_base = w_base + (long long)el * w_stride_elems;
      B_local.raw_ptr = (bf16 *)b_base;
      b_srd = fs_srd(b_base, (uint32_t)((size_t)N_TOTAL * b_row_stride), b_row_stride);

      const int ft = fs_tid();
      const int warp_m = ft / (kittens::WARP_THREADS * WARPS_COL), warp_n = ft / kittens::WARP_THREADS % WARPS_COL;
      const int lane = ft % kittens::WARP_THREADS;
      int ga_row[copies_A], ga_col[copies_A];
      gather_parts<ST_A>(lane, ft / kittens::WARP_THREADS, ga_row, ga_col);
      const int gbase = block_row * BLOCK_ROW;
#pragma unroll
      for (int i = 0; i < copies_A; i++) {
        const int r0 = a_gather[gbase + ga_row[i]];
        const int r1 = a_gather[gbase + HALF_ROW + ga_row[i]];
        sw_A[i] = MOE_SW_OFF(r0, a_row_stride, ga_col[i]);
        sw_A1[i] = MOE_SW_OFF(r1, a_row_stride, ga_col[i]);
      }
      static_assert(std::is_same_v<ST_A, ST_B>, "B's swizzled offsets are A's gather parts");
#pragma unroll
      for (int i = 0; i < copies_B; i++) {
        sw_B[i] = MOE_SW_OFF(ga_row[i], b_row_stride, ga_col[i]);
      }

      RT_C c00, c01, c10, c11;
      if constexpr (!RND) {
        // A kept Xp: each column tile of a row block writes its share of the k-steps, in pairs.
        G1Xp xq = {nullptr, nullptr, nullptr, nullptr, 0u, 0, 0};
        static_assert(copies_A == 2, "two parts per lane and stage");
        if (train_int(TRAIN_INT_KEPT) >= 1 && block_row % XP_GEMM1_PERIOD < XP_GEMM1_BLOCKS) {
          const int tn = N_TOTAL / BLOCK_COL, upc = (k_tiles / 2 + tn - 1) / tn * 2;
          char *xb = (char *)train_ptr<bf16>(TRAIN_PTR_XP) + (size_t)gbase * a_row_stride;
          xq.a0 = xb + ga_row[0] * a_row_stride + ga_col[0];
          xq.a1 = xb + ga_row[1] * a_row_stride + ga_col[1];
          xq.b0 = xq.a0 + HALF_ROW * a_row_stride;
          xq.b1 = xq.a1 + HALF_ROW * a_row_stride;
          xq.xl = (uint32_t)reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + ft * 16;
          xq.k0 = block_col * upc;
          xq.kn = max(0, min(upc, k_tiles - xq.k0));
        }
        g1_kloop<false, 0, true>(As, Bs, A, B_local, a_srd, b_srd, a_base, b_base, sw_A, sw_A1, sw_B, a_lds_00,
                                 a_lds_01, a_lds_10, a_lds_11, b_lds_00, b_lds_01, b_lds_10, b_lds_11, block_col * 2,
                                 block_col * 2 + 1, k_tiles, lane, warp_m, warp_n, c00, c01, c10, c11, xq);
      } else {
        g1_kloop<false, 0>(As, Bs, A, B_local, a_srd, b_srd, a_base, b_base, sw_A, sw_A1, sw_B, a_lds_00, a_lds_01,
                           a_lds_10, a_lds_11, b_lds_00, b_lds_01, b_lds_10, b_lds_11, block_col * 2, block_col * 2 + 1,
                           k_tiles, lane, warp_m, warp_n, c00, c01, c10, c11);
      }
      const int fe = fs_tid();
      const int we = fe / kittens::WARP_THREADS;
      if (we / WARPS_COL == 0) {
        __builtin_amdgcn_s_barrier();
      }

      const int rf0 = __builtin_amdgcn_readfirstlane(we / WARPS_COL);
      const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + we / WARPS_COL);
      int lane_epi = fe % kittens::WARP_THREADS;
      asm volatile("" : "+v"(lane_epi));
      // G is read only by a later launch: write-back stores (act and c2 write through for this kernel's readers).
      bf16 *gout = nullptr;
      gout = train_ptr<bf16>(TRAIN_PTR_G_OUT);
      if (gout != nullptr) {
        const size_t gr =
            (size_t)(!RND ? 0 : __builtin_amdgcn_readfirstlane(*fs_lds(&s_gr))) * H;
        bf16 *gbase = gout + gr + (size_t)block_row * BLOCK_ROW * N_TOTAL;
        const int cg = __builtin_amdgcn_readfirstlane(block_col * 2 * WARPS_COL + we % WARPS_COL);
#define cu ((cg + WARPS_COL) * REG_N)
        store_T16_bf16<false>(gbase, c00, rf0 * REG_M, cg * REG_N, N_TOTAL, lane_epi);
        store_T16_bf16<false>(gbase, c01, rf0 * REG_M, cu, N_TOTAL, lane_epi);
        store_T16_bf16<false>(gbase, c10, rf1 * REG_M, cg * REG_N, N_TOTAL, lane_epi);
        store_T16_bf16<false>(gbase, c11, rf1 * REG_M, cu, N_TOTAL, lane_epi);
#undef cu
      }
      swiglu_inplace<ACT>(c00, c01);
      swiglu_inplace<ACT>(c10, c11);
      const int N_OUT = N_TOTAL / 2;
      const size_t ar =
          (size_t)(!RND ? 0 : __builtin_amdgcn_readfirstlane(*fs_lds(&s_ar))) * H;
      auto *sbase = train_ptr<bf16>(TRAIN_PTR_A_OUT) + (size_t)block_row * BLOCK_ROW * N_OUT;
      const int cf = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL + we % WARPS_COL);
      store_T16_bf16<true>(sbase, c00, rf0 * REG_M, cf * REG_N, N_OUT, lane_epi);
      store_T16_bf16<true>(sbase, c10, rf1 * REG_M, cf * REG_N, N_OUT, lane_epi);
      // One publish per tile: the G2 gate of this row block. Relaxed suffices (the sc0 sc1 stores write through,
      // as GEMM2's tile_done publish), but the vmcnt(0) drain of every wave before the add is REQUIRED.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        FsArgsC *pp = fs_args(args);
        (void)__hip_atomic_fetch_add(fs_g(pp->a_done) + block_row, 1, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      using RT_C = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
      FsArgsC *ap = fs_args(args);
      const int H = ap->H, I = ap->I;
      const size_t ar =
          (size_t)(!RND ? 0 : __builtin_amdgcn_readfirstlane(*fs_lds(&s_ar))) * H;
      const FsGl A(fs_view(A0, train_ptr<bf16>(TRAIN_PTR_A_OUT),
                           A0.rows(), I));
      const bf16 *w2p = FKP(const bf16, W2, FK_W2);
      FsGl B_local(fs_view(A0, w2p, H, I));
      const T *w_base = fs_uniform_ptr(w2p);
      const long long w_stride_elems = (long long)H * I;
      bf16 *out = fs_uniform_ptr(ap->c2);
      const int K            = I;
      const int N_TOTAL      = H;
      const int k_tiles      = K / K_STEP_64;
      const int M            = A.rows();
      const int a_row_stride = K * sizeof(T);
      const int b_row_stride = K * sizeof(T);
      const T *a_base = fs_uniform_ptr((const T *)&A[{0, 0, 0, 0}]);
      i32x4 a_srd = fs_srd(a_base, (uint32_t)((size_t)M * a_row_stride), 0);
      const T *b_base = w_base;
      i32x4 b_srd;
      uint32_t sw_B[copies_B];
      const auto *a_gather = fs_g(ap->gid);
      b_base = w_base + (long long)el * w_stride_elems;
      B_local.raw_ptr = (bf16 *)b_base;
      b_srd = fs_srd(b_base, (uint32_t)((size_t)N_TOTAL * b_row_stride), b_row_stride);

      const int ft = fs_tid();
      const int warp_m = ft / (kittens::WARP_THREADS * WARPS_COL), warp_n = ft / kittens::WARP_THREADS % WARPS_COL;
      const int lane = ft % kittens::WARP_THREADS;
      int ga_row[copies_A], ga_col[copies_A];
      gather_parts<ST_A>(lane, ft / kittens::WARP_THREADS, ga_row, ga_col);
      const int gbase = block_row * BLOCK_ROW;
#pragma unroll
      for (int i = 0; i < copies_A; i++) {
        const int r0 = a_gather[gbase + ga_row[i]];
        const int r1 = a_gather[gbase + HALF_ROW + ga_row[i]];
        sw_A[i] = MOE_SW_OFF(r0, a_row_stride, ga_col[i]);
        sw_A1[i] = MOE_SW_OFF(r1, a_row_stride, ga_col[i]);
      }
      static_assert(std::is_same_v<ST_A, ST_B>, "B's swizzled offsets are A's gather parts");
#pragma unroll
      for (int i = 0; i < copies_B; i++) {
        sw_B[i] = MOE_SW_OFF(ga_row[i], b_row_stride, ga_col[i]);
      }

      const auto sa0 = subtile_inplace<REG_M, K_STEP_64>(As[0][0], {warp_m, 0});
      const auto sb0 = subtile_inplace<REG_N, K_STEP_64>(Bs[0][0], {warp_n, 0});
      RT_A a_tile;
      RT_B b_tile_0, b_tile_1;
      RT_C c00, c01, c10, c11;
      zero(c00);
      zero(c01);
      zero(c10);
      zero(c11);

      int tic = 0, toc = 1;

      dma_load(Bs[tic][0], B_local, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, b_lds_00);
      dma_load(As[tic][0], A, {0, 0, 0, 0}, sw_A, a_srd, a_base, a_lds_00);
      dma_load(Bs[tic][1], B_local, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds_01);
      dma_load(As[tic][1], A, {0, 0, 0, 0}, sw_A1, a_srd, a_base, a_lds_01);

      if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
      }
      asm volatile("s_waitcnt vmcnt(4)");
      __builtin_amdgcn_s_barrier();

      dma_load(Bs[toc][0], B_local, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, b_lds_10);
      dma_load(As[toc][0], A, {0, 0, 0, 1}, sw_A, a_srd, a_base, a_lds_10);
      dma_load(Bs[toc][1], B_local, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds_11);

      asm volatile("s_waitcnt vmcnt(4)");
      __builtin_amdgcn_s_barrier();
      fs_load_frag<0>(b_tile_0, sb0, lane);

#pragma unroll 1
      for (int tile = 0; tile < k_tiles - 2; tile += 2) {
        fs_load_frag<0>(a_tile, sa0, lane);
        dma_load(As[1][1], A, {0, 0, 0, tile + 1}, sw_A1, a_srd, a_base, a_lds_11);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c00, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fs_load_frag<SZB>(b_tile_1, sb0, lane);
        dma_load(Bs[0][0], B_local, {0, 0, block_col * 2, tile + 2}, sw_B, b_srd, b_base, b_lds_00);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c01, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<SZA>(a_tile, sa0, lane);
        dma_load(As[0][0], A, {0, 0, 0, tile + 2}, sw_A, a_srd, a_base, a_lds_00);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c10, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fs_load_frag<2 * SZB>(b_tile_0, sb0, lane);
        dma_load(Bs[0][1], B_local, {0, 0, block_col * 2 + 1, tile + 2}, sw_B, b_srd, b_base, b_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c11, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<2 * SZA>(a_tile, sa0, lane);
        dma_load(As[0][1], A, {0, 0, 0, tile + 2}, sw_A1, a_srd, a_base, a_lds_01);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c00, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fs_load_frag<3 * SZB>(b_tile_1, sb0, lane);
        dma_load(Bs[1][0], B_local, {0, 0, block_col * 2, tile + 3}, sw_B, b_srd, b_base, b_lds_10);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c01, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<3 * SZA>(a_tile, sa0, lane);
        dma_load(As[1][0], A, {0, 0, 0, tile + 3}, sw_A, a_srd, a_base, a_lds_10);
        asm volatile("s_waitcnt vmcnt(10)" ::: "memory");  // P3's Bs[0][0] DMA, read in P15
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c10, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fs_load_frag<0>(b_tile_0, sb0, lane);
        dma_load(Bs[1][1], B_local, {0, 0, block_col * 2 + 1, tile + 3}, sw_B, b_srd, b_base, b_lds_11);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c11, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
      }

      {
        int tile = k_tiles - 2;
        fs_load_frag<0>(b_tile_0, sb0, lane);
        fs_load_frag<0>(a_tile, sa0, lane);
        dma_load(As[toc][1], A, {0, 0, 0, tile + 1}, sw_A1, a_srd, a_base, a_lds_11);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c00, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<SZB>(b_tile_1, sb0, lane);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c01, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<SZA>(a_tile, sa0, lane);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c10, a_tile, b_tile_0);
        mma_ABt_T(c11, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
      }

      {
        fs_load_frag<2 * SZB>(b_tile_0, sb0, lane);
        fs_load_frag<2 * SZA>(a_tile, sa0, lane);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c00, a_tile, b_tile_0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<3 * SZB>(b_tile_1, sb0, lane);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c01, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fs_load_frag<3 * SZA>(a_tile, sa0, lane);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_T(c10, a_tile, b_tile_0);
        mma_ABt_T(c11, a_tile, b_tile_1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
      }

      const int fe = fs_tid();
      const int we = fe / kittens::WARP_THREADS;
      if (we / WARPS_COL == 0) {
        __builtin_amdgcn_s_barrier();
      }

      const int rf0 = __builtin_amdgcn_readfirstlane(we / WARPS_COL);
      const int rf1 = __builtin_amdgcn_readfirstlane(WARPS_ROW + we / WARPS_COL);
      int lane_epi = fe % kittens::WARP_THREADS;
      asm volatile("" : "+v"(lane_epi));
      {
        bf16 *sbase = out + (size_t)block_row * BLOCK_ROW * N_TOTAL;
        const int cf0 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + we % WARPS_COL);
        const int cf1 = __builtin_amdgcn_readfirstlane(block_col * WARPS_COL * 2 + WARPS_COL + we % WARPS_COL);
        store_T16_bf16<true>(sbase, c00, rf0 * REG_M, cf0 * REG_N, N_TOTAL, lane_epi);
        store_T16_bf16<true>(sbase, c01, rf0 * REG_M, cf1 * REG_N, N_TOTAL, lane_epi);
        store_T16_bf16<true>(sbase, c10, rf1 * REG_M, cf0 * REG_N, N_TOTAL, lane_epi);
        store_T16_bf16<true>(sbase, c11, rf1 * REG_M, cf1 * REG_N, N_TOTAL, lane_epi);
      }
      // Every G2 tile MUST add to tile_done, or the comm role waits forever. Relaxed suffices (the sc0 sc1 stores
      // write through), but the vmcnt(0) drain is REQUIRED: asm hides the stores.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        FsArgsC *pp = fs_args(args);
        __hip_atomic_fetch_add(fs_g(pp->tile_done) + block_row, 1, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  if constexpr (!RND) {
    break;
  } else {
    if (*fs_lds(&s_rnd) + 1 == *fs_lds(&s_nr)) {
      break;
    }
    fs_round_end(args, tile_counter, *fs_lds(&s_rnd), *fs_lds(&s_mpad) / BLOCK_ROW,
                 (unsigned)fk_epoch<true>(epoch, s_kp, 0), *fs_lds(&s_ngr));
    if (tid == 0) {
      s_rnd++;
    }
    __syncthreads();
  }
  }
  if (moe_last_wg_n(tile_counter + NUM_XCDS_AFF, *fs_lds(&s_ngr))) {
    FsArgsC *rp = fs_args(args);
    const int nrb = *fs_lds(&s_mpad) / BLOCK_ROW;
    route_fwd_finish(args, train_ptr<uint8_t>(TRAIN_PTR_ROUTE), train_ptr<int>(TRAIN_PTR_CTR), train_int(TRAIN_INT_KEPT),
                     train_int(TRAIN_INT_TIER), train_int(TRAIN_INT_MPAD), train_int(TRAIN_INT_MPADX),
                     train_int(TRAIN_INT_NR), train_int(TRAIN_INT_CROWS),
                     (unsigned)fk_epoch<true>(epoch, s_kp, 0));
    fs_reset_words<true>(fs_g(tile_counter), NUM_XCDS_AFF + 1);
    fs_reset_words<true>(fs_g(rp->p.ctr), PC_N);
    fs_reset_words<true>(fs_g(rp->g2_ctr), 1);
    fs_reset_words<true>(fs_g(rp->disp_done), 1);
    fs_reset_words<true>(fs_g(rp->idx_done), 1);
    fs_reset_words<true>(fs_g(rp->tile_done), nrb);
    fs_reset_words<true>(fs_g(rp->a_done), nrb);
    // Every fold item is done. The re-check orders the subtraction after this call's adds on each acc word.
    const int te = fs_tid();
    if (RND && te < MOE_EP) {
      const int ow = fs_g(rp->owed)[te];
      int n = 0;
      gint *acc = fs_my_acc(rp) + te;
      while (__hip_atomic_load(acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < ow) {
        spin_backoff(n);
      }
      __hip_atomic_fetch_add(acc, -ow, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    fs_reset_words<true>(fs_g(rp->fold_ctr), 2 + rp->nrbs);
    if (rp->pool && FKP(bf16, G, FK_G) != nullptr) {
      pool_commit(args, (int *)&As[0][0], *fs_lds(&s_gr) - *fs_lds(&s_cur), RND && *fs_lds(&s_nr) > 1);
    }
    if constexpr (RND) {
      fs_reset_words<true>(fs_g(rp->rgo), 1);
    }
    // Every workgroup has read the epochs: the layer's next epoch, and the workspace's latest call is a forward.
    if (te == 0) {
      (void)__hip_atomic_fetch_add((gint *)fs_g(rp->ep), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store((gint *)fs_g(rp->wep) + 1, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
