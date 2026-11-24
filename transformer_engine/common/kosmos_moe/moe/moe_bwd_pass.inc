/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE backward kernel body from its round state on (moe_bwd_single.inc; RNDB: of a call in rounds), to its last
// workgroup's reset; textually included.
  // MXR: MXFP8 training's rounds copy, which also re-sends X, quantizes it once on arrival (SK_QX) and recomputes GEMM1
  // (SK_RG).
  constexpr bool SA0 = SA00 || RNDB;
  constexpr bool MXR = MX && RNDB;
  // Over capacity (nr > 1): each round from its dispatch to its drained tile loop, the fold in the last; single_round_*
  // derive its tables, rows and wgrad modes.
  if constexpr (RNDB) {
    if constexpr (MXR) {
      if (fs_tid() == 0) {
        s_a0[0] = (unsigned long long)A0.raw_ptr;
        s_a0[1] = (unsigned long long)(unsigned)A0.rows() | (unsigned long long)(unsigned)A0.cols() << 32;
        s_a0[2] = (unsigned long long)tile_counter;
#if MOE_BWD_PIN
        s_a0[3] = (unsigned long long)pin_x, s_a0[4] = (unsigned long long)pin_w;
#endif
        mx_kp()[4] = (unsigned long long)mxc.w1f, mx_kp()[5] = (unsigned long long)mxc.w1fs;
      }
    }
    if (fs_tid() == 0) {
      s_rr = 0, s_nrb = max(1, fs_args(args)->nr), s_nrow = fs_args(args)->nrow, s_ngr = (int)gridDim.x;
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
      single_round_open<MX>(args, s_tiles, (int)gridDim.x);
      single_round_info(args, &s_nrow, s_wm);
    }
  }
  // The router's: round 0's dispatch and the prep (once, every slot's dots first under rounds), before the round loop,
  // where the prep's constants would be hoisted out of it and held across it.
  if constexpr (ROUTER && RNDB) {
    if (wg < fs_args(args)->a.g.push.ncomm) {
      SArgsC *ka = fs_args(args);
      if (*fs_lds(&s_nrb) > 1 && fs_tid() == 0) {
        plan_wait(fs_g(ka->a.g.push.peers.sig[ka->a.g.push.my_rank]) +
                      hk_comm::bfree_index(MOE_EP + wg % MOE_EP, ka->a.g.push.sigb),
                  (int)((unsigned)bcall << 16 | 1u), true);
      }
      if constexpr (MX) {
        __syncthreads();
        mx_bwd_xpush<PIN>(fs_args(args), (const bf16 *)mx_ptr(0), wg);
      }
      hk_comm::moe_bwd_dispatch_push_role<MX, Q8, PIN>(&args->a.g.push, wg, SCALL(const unsigned short, SC_DOUT),
                                                       SCALL(float, SC_DTK), bcall, bwait,
                                                       MX ? fs_args(args)->xc_fwd : 0);
      if (!MX && *fs_lds(&s_rg) == 2) {
        moe_bwd_xpush<PIN>(fs_args(args), wg);
      }
      asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
      if (*fs_lds(&s_nrb) > 1) {
        single_round_dots<Q8>(fs_args(args), SCALL(const unsigned short, SC_DOUT), SCALL(float, SC_DTK), wg);
      }
      single_router_prep<MoeScore, MX>(args, fs_args(args)->r.aux_c * *fs_g(SCALL(const float, SC_AUX)),
                                       SCALL(float, SC_DPR),
                                       __builtin_amdgcn_readfirstlane((int)*fs_lds(&s_call[SC_ACC])));
    }
  }
  // In the round loop the thread id and the grid size are opaque (BTID, BGRID): masks and offsets derived from them
  // would be hoisted out of it and held across the tile loop.
#define BTID (RNDB ? (unsigned)fs_tid() : threadIdx.x)
#define BGRID (RNDB ? *fs_lds(&s_ngr) : (int)gridDim.x)
  // Never taken (s_rr is 0 here): a second entry into the round loop makes it irreducible, so the optimizer hoists no
  // loop-invariant value out of it (one held across the tile loop spills).
  if (RNDB && s_rr < 0) {
    goto BWD_RMID;
  }
  for (;;) {
  if (BPART <= 1 && wg < fs_args(args)->a.g.push.ncomm) {
    const int rr = RNDB ? *fs_lds(&s_rr) : 0;
    if (!(ROUTER && RNDB) || rr) {
      if (RNDB && *fs_lds(&s_nrb) > 1 && fs_tid() == 0) {
        // A round's rows (and slots) go where the destination's previous round, or its forward, read them: its RFREE.
        // bfree cannot tell: on the workspace's first backward it is 0 before the peer starts.
        SArgsC *ka = fs_args(args);
        plan_wait(fs_g(ka->a.g.push.peers.sig[ka->a.g.push.my_rank]) + hk_comm::bfree_index(MOE_EP + wg % MOE_EP,
                                                                                             ka->a.g.push.sigb),
                  (int)((unsigned)__builtin_amdgcn_readfirstlane(fs_g(ka->wep)[0]) << 16 | (unsigned)(rr + 1)), true);
      }
      if constexpr (MXR) {
        // X ahead of dY: the GEMM1 recompute precedes dgrad2.
        __syncthreads();
        mx_bwd_xpush<PIN>(fs_args(args), (const bf16 *)mx_ptr(0), wg);
      }
      hk_comm::moe_bwd_dispatch_push_role<MX, Q8, PIN>(&args->a.g.push, wg, SCALL(const unsigned short, SC_DOUT),
                                                       SCALL(float, SC_DTK), RNDB ? fs_g(fs_args(args)->wep)[0] : bcall,
                                                       RNDB ? !rr && !fs_g(fs_args(args)->wep)[1] : bwait,
                                                       MX ? fs_args(args)->xc_fwd : 0);
      if constexpr (!MX && BPART == 0) {
        if (*fs_lds(&s_rg) == 2) {
          moe_bwd_xpush<PIN>(fs_args(args), wg);
        }
      }
      // The role's publishes are FLAT atomics (LGKM_CNT, out of order with LDS reads): drain before the tile loop.
      asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
      if constexpr (ROUTER) {
        if constexpr (!RNDB) {
          single_router_prep<MoeScore, MX>(args, fs_args(args)->r.aux_c * *fs_g(SCALL(const float, SC_AUX)),
                                           SCALL(float, SC_DPR),
                                           __builtin_amdgcn_readfirstlane((int)*fs_lds(&s_call[SC_ACC])));
        }
      } else if (BTID == 0) {
        (void)__hip_atomic_fetch_add((RNDB ? TCX : tile_counter) + SINGLE_DISP_CTR, 1, __ATOMIC_RELEASE,
                                     __HIP_MEMORY_SCOPE_AGENT);
      }
      if constexpr (MX && !RNDB) {
        if (!fs_args(args)->xc_fwd) {
#if MOE_BWD_PIN
          mx_bwd_xcol<true>(fs_args(args), (const bf16 *)mx_ptr(0), wg, pin_w);
#else
          mx_bwd_xcol(fs_args(args), (const bf16 *)mx_ptr(0), wg);
#endif
        }
      }
    }
  } else if (BPART <= 1) {
    const int dn = fs_args(args)->a.g.push.ncomm;
    single_zero_pads<RNDB>(single_a0<RNDB>(A0, s_a0), fs_args(args), wg - dn, BGRID - dn,
                           (RNDB ? TCX : tile_counter) + SINGLE_ZERO_CTR,
                           !MX && *fs_lds(&s_rg) == 2 ? fs_args(args)->xrp[fs_args(args)->a.g.push.my_rank] : nullptr,
                           RNDB ? *fs_lds(&s_nrow) : fs_args(args)->nrow, BTID);
#if MOE_BWD_SH
    // The shared expert's MXFP8 operands of this rank's tokens (rounds: once): dOut row- and column-wise, X
    // column-wise.
    if constexpr (MX) {
      SArgsC *ka = fs_args(args);
      const int H = ka->H, nc = H / MX_QCOLS, nq = ka->sh.T / BLOCK_ROW * nc;
      int mine = 0;
      for (int q = RNDB && *fs_lds(&s_rr) ? 2 * nq : wg - dn; q < 2 * nq; q += BGRID - dn, mine++) {
        const int b = q % nq, r0 = b / nc * BLOCK_ROW, c0 = b % nc * MX_QCOLS;
        if (q < nq) {
          mx_quant_block<true>(SCL(const bf16, SC_DOUT), H, r0, BLOCK_ROW, c0, H, ka->sdoq, 2 * H, ka->sdoc, ka->sdocs);
        } else {
          mx_quant_block<true, false, false>(SCL(const bf16, SC_SX), H, r0, BLOCK_ROW, c0, H, nullptr, 0, ka->sxc,
                                             ka->sxcs);
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (fs_tid() == 0 && mine) {
        (void)__hip_atomic_fetch_add(fs_g(ka->sq_done), mine, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
#endif
  }
BWD_RMID:
  if constexpr (BPART == 1 && !MX) {
    return;
  }
  const int ne = fs_args(args)->E;
  if constexpr (SA0 && !RNDB) {
    if (fs_tid() == 0) {
      s_a0[0] = (unsigned long long)A0.raw_ptr;
      s_a0[1] = (unsigned long long)(unsigned)A0.rows() | (unsigned long long)(unsigned)A0.cols() << 32;
      s_a0[2] = (unsigned long long)tile_counter;
#if MOE_BWD_PIN
      s_a0[3] = (unsigned long long)pin_x, s_a0[4] = (unsigned long long)pin_w;
#endif
    }
    __syncthreads();
  }

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
  if constexpr (!MX && BPART == 0) {
    if (*fs_lds(&s_rg)) {
      using RT_G1 = rt_fl<REG_N, REG_M, col_l, rt_16x16_s>;
      constexpr uint32_t BO = 4 * sizeof(ST_A);
      ST_A(&Bt)[2][2] = *reinterpret_cast<ST_A(*)[2][2]>(s_tiles + BO);
      while (true) {
        SArgsC *ka = fs_args(args);
        const int nrow = RNDB ? *fs_lds(&s_nrow) : ka->nrow;
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
              while (__hip_atomic_load((RNDB ? TCX : tile_counter) + SINGLE_ZERO_CTR, __ATOMIC_RELAXED,
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
        const FsGl A = fs_view(A0S, ka->b.w.act, nrow, H);
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
  if constexpr (!ROUTER && !SH && MX) {
    b_lds_wo_k = __builtin_amdgcn_readfirstlane((uint32_t)lds_warp_off);
    asm volatile("" : "+s"(b_lds_wo_k));
  }

  static_assert(MOE_WALK_GM == 4, "a wgrad run of 1, 2 (or 4 with MX) tiles divides the MOE_WALK_GM = 4 group");
  for (int x = BTID; x <= ne; x += NUM_THREADS) {
    s_rb0[x] = (x < ne ? fs_g(fs_args(args)->row_off)[x] : ((RNDB ? *fs_lds(&s_nrow) : fs_args(args)->nrow) - 1)) /
               BLOCK_ROW;
  }
  for (int x = BTID; x < ne; x += NUM_THREADS) {
    if constexpr (MX) {
      const int cx = fs_g(fs_args(args)->cnt)[x];
      s_wkt[x] = cx ? max(2, (cx + K_STEP_128 - 1) / K_STEP_128) : 0;
    } else {
      s_wkt[x] = (fs_g(fs_args(args)->cnt)[x] + K_STEP_64 - 1) / K_STEP_64;
    }
  }
  if (BTID < MOE_BWD_MAX_EXPERTS / 32) {
    s_dgok[BTID] = 0u;
    if constexpr (!MX || RNDB) {
      s_rgok[BTID] = 0u;
    }
  }
  if (BTID == 0) {
    s_acq_upto = -1;
    s_zok = 0;
#if MOE_BWD_SH
    s_sdok = 0;
#endif
    s_arr = 0;
    s_nwg = BGRID;
  }
  bool c_left  = (BPART == 0 || BPART == 5) && wg < fs_args(args)->b.c.ncomm;
  bool f_early = c_left && (!RNDB || *fs_lds(&s_rr) + 1 >= *fs_lds(&s_nrb));
  if (c_left) {
    hk_comm::moe_bwd_combine_init(&args->b.c, wg, s_cdone);
  }
  __syncthreads();
  for (int x = BTID; x < ne; x += NUM_THREADS) {
    const int kx = s_wkt[x];
    int r = 0;
    for (int y = RNDB ? single_opaque(0) : 0; y < ne; y++) {
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
    const int tpp2 = ((nnb2 + MOE_WALK_GM - 1) / MOE_WALK_GM) * tNi * (MOE_WALK_GM >> (MX ? run2 >> 1 : run2 - 1));
    const int tpp1 = ((nnb1 + MOE_WALK_GM - 1) / MOE_WALK_GM) * tNh * (MOE_WALK_GM >> (MX ? run1 >> 1 : run1 - 1));
    int acc = 0, q = 0;
    auto put = [&](int kind, int e) {
      s_seg[q] = acc;
      // Opaque under rounds: the kinds' constants would be hoisted out of the round loop and held across it.
      s_segk[q] = (short)(((RNDB ? single_opaque(kind) : kind) << 8) | e);
      q++;
      if constexpr (SH) {
        if (kind >= SK_SDG2 && (!MX || kind <= SK_SWG1)) {
          const int ms = ka->sh.T / BLOCK_ROW;
          acc += kind == SK_SDG2 ? ms * tNi : kind == SK_SWG2 ? nnb2 * tNi : kind == SK_SDG1 ? ms * tNh
               : SH_KSPLIT * nnb1 * tNh;
          return;
        }
      }
      const int mt = s_rb0[e + 1] - s_rb0[e];
      if constexpr (ROUTER || MX) {
        acc += kind == SK_DG2 ? mt * tNi : kind == SK_DG1 ? mt * tNh : kind == SK_WG2 ? tpp2 : kind == SK_WG1 ? tpp1
             : ROUTER && kind == SK_RDG ? ka->r.T / BLOCK_ROW * tNh
             : ROUTER && (!MX || kind == SK_RWG) ? ka->r.Ed / BLOCK_ROW * tNh
             : MXR && kind == SK_RG ? mt * (2 * ka->I / BLOCK_COL) : mt * (ka->H / MX_QCOLS);
      } else {
        acc += kind == SK_DG2 ? mt * tNi : kind == SK_DG1 ? mt * tNh : kind == SK_WG2 ? tpp2 : tpp1;
      }
    };
    // The shared expert's tiles first (they need no dispatched row), each producer ahead of its consumers.
    // Over capacity: the shared expert's tiles in round 0, the router's in the last, an expert's wgrads in the rounds
    // with its rows (s_wm).
    const int rr = RNDB ? *fs_lds(&s_rr) : 0;
    const bool rlast = !RNDB || rr + 1 >= *fs_lds(&s_nrb);
    if (SH && !rr) {
      put(SK_SDG2, 0);
      put(SK_SWG2, 0);
      put(SK_SDG1, 0);
      put(SK_SWG1, 0);
    }
    // MX: dY quantized 2 experts ahead for I <= 2048 (dY arrival paces the start; items further ahead park on rows not
    // landed), 12 above (the GEMMs pace it); wgrad2 waits only for the items of its own columns.
    int qn = 0;
    if constexpr (MX && BPART == 1) {
      for (; qn < ne; qn++) {
        put(SK_QDY, qn);
      }
    }
    for (int e = 0; e < ne && (BPART == 0 || BPART == 2); e++) {
      if constexpr (MX && BPART == 0) {
        for (; qn < min(ne, min(2 * e + 2, e + (ka->I > 2048 ? 13 : 3))); qn++) {
          put(SK_QDY, qn);
        }
      }
      if constexpr (MXR) {
        // Rounds: expert e's X quantized and its GEMM1 recomputed (G for dgrad2, column-wise A for wgrad2).
        put(SK_QX, e);
        put(SK_RG, e);
      }
      put(SK_DG2, e);
      if (e > 0 && (!RNDB || s_wm[e - 1])) {
        put(SK_WG2, e - 1);
      }
    }
    if ((BPART == 0 || BPART == 2) && (!RNDB || s_wm[ne - 1])) {
      put(SK_WG2, ne - 1);
    }
    if (ROUTER && rlast) {
      put(SK_RDG, 0);
      put(SK_RWG, 0);
    }
    // bf16, I > 1024: wgrad1 of experts < ne / 2 follows dgrad1 of the next expert, the rest longest K first. MXFP8
    // and shorter I put all wgrad1 after dgrad1, so the combine push starts as early as possible.
    const int m = !MX && ka->I > 1024 ? ne / 2 : 0;
    for (int e = 0; e < ne; e++) {
      if (BPART == 0 || BPART == 3) {
        put(SK_DG1, e);
      }
      if ((BPART == 0 || BPART == 4) && e > 0 && e <= m && (!RNDB || s_wm[e - 1])) {
        put(SK_WG1, e - 1);
      }
    }
    for (int p = 0; p < ne; p++) {
      if ((BPART == 0 || BPART == 4) && s_ord[p] >= m && (!RNDB || s_wm[s_ord[p]])) {
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
  bool drained = BPART == 5;
  int w_sub = 0, w_n = 0, w_kind = SK_WG2;
  int f_pend = -1;
  int w_idle = 0;  // the idle passes' clock (idle_sleep)

  while (true) {
    if (c_left) {
      SArgsC *ka = fs_args(args);
      const int sel = hk_comm::moe_bwd_combine_probe(&args->b.c, wg, ka->H / BLOCK_COL, s_cdone, &s_csel);
      if (sel >= 0) {
        hk_comm::moe_bwd_combine_push_unit<MX, Q8>(&args->b.c, wg + sel * ka->b.c.ncomm);
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
      // main_grad accumulation (SingleCall::acc): the routed and router tiles are added into dW1, dW2 and dWr.
      const int wacc = __builtin_amdgcn_readfirstlane((int)*fs_lds(&s_call[SC_ACC]));
      if constexpr (SH && !MX) {
        // Shared fc2 wgrad dOut^T As and fc1 wgrad slices dGs^T X over K = T park fp32 partial tiles (rlim < 0) in the
        // workspace; finalize items reduce them later. One call site: a second inlined NT loop does not compile.
        const bool sw = w_kind >= SK_SDG2, s1 = w_kind == SK_SWG1, rw = ROUTER && w_kind == SK_RWG;
        const int st = ka->sh.T, kq = st / K_STEP_64 / SH_KSPLIT;
        const __attribute__((address_space(1))) LegAWgrad2 *wa =
            (const __attribute__((address_space(1))) LegAWgrad2 *)(w1k ? &args->b.w : &args->a.w);
        FsGl aw = A0S;
        if (sw) {
          aw = s1 ? fs_view(aw, ka->sh.dg, st, 2 * ka->I) : fs_view(aw, SCL(const bf16, SC_DOUT), st, ka->H);
        } else if (rw) {
          aw = fs_view(aw, ka->r.dl, ka->r.T, ka->r.Ed);
        } else if (w1k) {
          aw = fs_view(aw, ka->dg, ka->nrow, 2 * ka->I);
        }
        float *pw = fs_uniform_ptr(ka->sh.part);
        lega_wgrad2_tile<true, true, RNDB>(aw, nullptr, s_tiles, sw ? 0 : e, nblk, mblk,
                                     sw ? (s1 ? e * kq : 0) : rw ? 0 : s_rb0[e] * (BLOCK_ROW / K_STEP_64),
                                     sw ? (s1 ? kq : st / K_STEP_64) : rw ? ka->r.T / K_STEP_64 : s_wkt[e],
                                     sw ? (bf16 *)(void *)pw : rw ? SCALL(bf16, SC_DWR)
                                        : w1k ? SCALL(bf16, SC_DW1) : SCALL(bf16, SC_DW2),
                                     sw ? (s1 ? SCL(const bf16, SC_SX) : (const bf16 *)ka->sh.act)
                                        : rw ? SCALL(const bf16, SC_X) : wa->act,
                                     sw ? st : rw ? ka->r.T : wa->act_rows,
                                     sw ? (s1 ? ka->H : ka->I) : rw ? ka->H : wa->act_cols,
                                     sw   ? (s1 ? -2 - e : -1)
                                     : rw ? ka->r.E - nblk * BLOCK_ROW
                                          : BLOCK_ROW + (RNDB && s_wm[e] == 2),
                                     sw ? 0 : wacc);
        w_sub++;
        continue;
      }
      if constexpr (MX) {
        // The router's dX and dWr tiles run the bf16 NT body below.
        if (!ROUTER || (w_kind != SK_RDG && w_kind != SK_RWG)) {
#if MOE_BWD_SH
          if (w_kind == SK_SWG2 || w_kind == SK_SWG1) {
            // As the bf16 path: fc2 dOut^T As and the fc1 slices dGs^T X (rows e T / 2..) park fp32 tiles for finalize.
            const bool s1 = w_kind == SK_SWG1;
            const int T = ka->sh.T, H = ka->H, I = ka->I, Ca = s1 ? 2 * I : H, Cb = s1 ? H : I;
            MxRtC cA, cB, cC, cD;
            mx_nt_tile(s_tiles, s1 ? ka->sdgc : ka->sdoc, s1 ? ka->sdgcs : ka->sdocs, Ca, s1 ? ka->sxc : ka->sactc,
                       s1 ? ka->sxcs : ka->sactcs, Cb, s1 ? e * (T / 2) : 0, (s1 ? T / 2 : T) / K_STEP_128, nblk, mblk,
                       cA, cB, cC, cD);
            const int lane_w = mx_lane_opaque();
            const int rf0 = __builtin_amdgcn_readfirstlane(fs_tid() / (kittens::WARP_THREADS * WARPS_COL));
            const int cf0 = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS % WARPS_COL);
            const int p = s1 ? 1 + e : 0;
            const size_t hi = (size_t)H * I;
            float *base = fs_uniform_ptr(ka->sh.part);
            sh_tile_park(s_tiles, cA, cB, cC, cD, rf0, cf0, cf0 + WARPS_COL, lane_w,
                         base + hi * (p ? 2 * p - 1 : 0) + (size_t)nblk * BLOCK_ROW * Cb + mblk * BLOCK_COL, Cb);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
            if (fs_tid() == 0) {
              gint *wd = fs_g((int *)(base + 5 * hi));
              (void)__hip_atomic_fetch_add(wd + (p ? (int)(hi / (BLOCK_ROW * BLOCK_COL)) : 0) +
                                               nblk * (Cb / BLOCK_COL) + mblk, 1, __ATOMIC_RELEASE,
                                           __HIP_MEMORY_SCOPE_AGENT);
            }
            __syncthreads();
            w_sub++;
            continue;
          }
#endif
          const int Ca = w1k ? 2 * ka->I : ka->H, Cb = w1k ? ka->H : ka->I;
          MxRtC cA, cB, cC, cD;
          const uint8_t *bq = MXR ? mx_head(ka, SCALL(bf16, SC_G), w1k ? 2 : 0) : w1k ? ka->xpc : ka->actc;
          const uint8_t *bs = MXR ? mx_head(ka, SCALL(bf16, SC_G), w1k ? 3 : 1) : w1k ? ka->xpcs : ka->actcs;
          mx_nt_tile(s_tiles, w1k ? ka->dgc : ka->dyc, w1k ? ka->dgcs : ka->dycs, Ca, bq, bs, Cb, s_rb0[e] * BLOCK_ROW,
                     s_wkt[e], nblk, mblk, cA, cB, cC, cD);
          const int lane_w = mx_lane_opaque();
          const int rf0 = __builtin_amdgcn_readfirstlane(fs_tid() / (kittens::WARP_THREADS * WARPS_COL));
          const int cf0 = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS % WARPS_COL);
          bf16 *tb = (w1k ? SCALL(bf16, SC_DW1) : SCALL(bf16, SC_DW2)) + ((size_t)e * Ca + nblk * BLOCK_ROW) * Cb +
                     mblk * BLOCK_COL;
          if (wacc) {
            wg_tile_acc(s_tiles, cA, cB, cC, cD, rf0, cf0, cf0 + WARPS_COL, lane_w,
                        w1k ? SCALL(void, SC_DW1) : SCALL(void, SC_DW2),
                        ((size_t)e * Ca + nblk * BLOCK_ROW) * Cb + mblk * BLOCK_COL, Cb, wacc, BLOCK_ROW);
          } else if (MXR && s_wm[e] == 2) {
            mx_tile_add(s_tiles, cA, cB, cC, cD, rf0, cf0, cf0 + WARPS_COL, lane_w, tb, Cb);
          } else {
            store_c_staged<2>(s_tiles, cA, cB, cC, cD, rf0, rf0 + WARPS_ROW, cf0, cf0 + WARPS_COL, lane_w, tb, Cb);
          }
          __syncthreads();
          w_sub++;
          continue;
        }
      }
      if constexpr (ROUTER) {
        // dWr tiles (rw): dlogits [T][Ed] against X [T][H] over K = T; rows of dWr past E are not stored. MXFP8 also
        // runs the router's dX tiles (rx) here: dlogits^T [Ed][T] against Wr [E][H] over K = Ed (rows past E read 0).
        const bool rx = MX && w_kind == SK_RDG, rw = rx || w_kind == SK_RWG;
        const __attribute__((address_space(1))) LegAWgrad2 *wa =
            (const __attribute__((address_space(1))) LegAWgrad2 *)(w1k ? &args->b.w : &args->a.w);
        const FsGl aw = rx   ? fs_view(A0S, ka->rdlt, ka->r.Ed, ka->r.T)
                        : rw ? fs_view(A0S, ka->r.dl, ka->r.T, ka->r.Ed)
                             : w1k ? fs_view(A0S, ka->dg, ka->nrow, 2 * ka->I) : A0S;
        lega_wgrad2_tile<true, false, RNDB && !MX>(
            aw, nullptr, s_tiles, e, nblk, mblk, rw ? 0 : s_rb0[e] * (BLOCK_ROW / K_STEP_64),
            rx ? ka->r.Ed / K_STEP_64 : rw ? ka->r.T / K_STEP_64 : s_wkt[e],
            rx   ? (bf16 *)ka->r.dxr
            : rw ? SCALL(bf16, SC_DWR) : w1k ? SCALL(bf16, SC_DW1) : SCALL(bf16, SC_DW2),
            rx ? SCALL(const bf16, SC_WR) : rw ? SCALL(const bf16, SC_X) : wa->act,
            rx ? ka->r.E : rw ? ka->r.T : wa->act_rows, rw ? ka->H : wa->act_cols,
            rx ? BLOCK_ROW : rw ? ka->r.E - nblk * BLOCK_ROW : BLOCK_ROW + (RNDB && s_wm[e] == 2), rx ? 0 : wacc);
        if (rx) {
          fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->r.rdgdone) + nblk);
          if (fs_tid() == 0) {
            (void)__hip_atomic_fetch_add(fs_g(ka->r.bctr) + RC_RDG, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          }
        }
      } else {
        const FsGl aw = w1k ? fs_view(A0S, ka->dg, ka->nrow, 2 * ka->I) : A0S;
        lega_wgrad2_tile<false, false, RNDB>(aw, w1k ? &args->b.w : &args->a.w, s_tiles, e, nblk, mblk,
                                             s_rb0[e] * (BLOCK_ROW / K_STEP_64), s_wkt[e],
                                             w1k ? SCALL(bf16, SC_DW1) : SCALL(bf16, SC_DW2), nullptr, 0, 0,
                                             BLOCK_ROW + (RNDB && s_wm[e] == 2), wacc);
      }
      w_sub++;
      continue;
    }
    if (f_pend >= 0) {
      __syncthreads();
      if (BTID < 64 && !s_arr) {
        const bool a = ROUTER ? hk_comm::moe_bwd_combine_arrived(&args->b.c) : single_fold_ready(args, TCX);
        if (BTID == 0) {
          s_arr = a;
        }
      }
      __syncthreads();
      if (s_arr) {
        if constexpr (SH) {
          // The item's tokens' shared dX rows: every column tile of their row block.
          single_wait_rows(fs_g(fs_args(args)->sh.dx_done) + f_pend * hk_comm::CSIDE_FOLD_TOK / BLOCK_ROW,
                           fs_args(args)->H / BLOCK_COL);
          if constexpr (SGATE) {
            if (fs_tid() < hk_comm::CSIDE_FOLD_TOK) {
              single_sgate_grad(fs_args(args), f_pend * hk_comm::CSIDE_FOLD_TOK + fs_tid(), SCL(float, SC_SDSG));
            }
          }
        }
        if constexpr (ROUTER) {
          // The item's tokens' router dX rows: every column tile of their row block.
          single_wait_rows(fs_g(fs_args(args)->r.rdgdone) + f_pend * hk_comm::CSIDE_FOLD_TOK / BLOCK_ROW,
                           fs_args(args)->H / BLOCK_COL);
          SArgsC *ka = fs_args(args);
          hk_comm::cw_fold_item_r<SH, KS, Q8>(&args->b.s, ka->b.c.peers.comb[ka->b.c.my_rank], ka->H, f_pend,
                                              SCALL(bf16, SC_DX), ka->r.dxr, SH ? ka->sh.dx : nullptr);
        } else {
          SArgsC *ka = fs_args(args);
#if MOE_BWD_PIN
          hk_comm::cw_fold_item_pin<SH, Q8>(&args->b.s, ka->b.c.peers.comb[ka->b.c.my_rank], ka->H, f_pend,
                                        SCALL(bf16, SC_DX), SCALL(float, SC_DPR), SH ? ka->sh.dx : nullptr,
                                        fs_uniform_ptr((const bf16 *)*fs_lds(&s_a0[3])),
                                        fs_uniform_ptr((const float *)*fs_lds(&s_a0[4])));
#else
          hk_comm::cw_fold_item<SH, KS, Q8>(&args->b.s, ka->b.c.peers.comb[ka->b.c.my_rank], ka->H, f_pend,
                                            SCALL(bf16, SC_DX), SCALL(float, SC_DPR), SH ? ka->sh.dx : nullptr);
#endif
        }
        f_pend = -1;
      } else {
        idle_sleep<16>(w_idle);
      }
      continue;
    }
    if (f_early && !c_left && !drained) {
      __syncthreads();
      if (BTID < 64) {
        bool a = s_arr ||
                 (ROUTER ? hk_comm::moe_bwd_combine_arrived(&args->b.c) : single_fold_ready(args, TCX));
        if (BTID == 0) {
          int i = -1;
          if constexpr (ROUTER) {
            // An early fold may not wait for a router dX tile (it may be unclaimed): only once all are done.
            s_arr = a;
            a = a && single_router_dx_done(fs_args(args));
          }
          if constexpr (SH) {
            // An early fold may not wait for a shared dX tile (it may be unclaimed): only once all are done.
            if constexpr (!ROUTER) {
              s_arr = a;
            }
            SArgsC *ka = fs_args(args);
            a = a && __hip_atomic_load(fs_g(ka->sh.dx_all), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >=
                         ka->sh.T / BLOCK_ROW * (ka->H / BLOCK_COL);
          }
          if (a) {
            if constexpr (!ROUTER && !SH) {
              s_arr = 1;
            }
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
      if constexpr (BPART >= 1 && BPART <= 4) {
        break;
      }
      if (c_left) {
        idle_sleep<32>(w_idle);
        continue;
      }
      if (SH && (!RNDB || !*fs_lds(&s_rr))) {
        // Every tile is claimed: finalize the shared weight gradients (waits only for running wgrad tiles).
        __syncthreads();
        if (fs_tid() == 0) {
          s_desc[1] = __hip_atomic_fetch_add(fs_g(fs_args(args)->sh.fctr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        const int fi = __builtin_amdgcn_readfirstlane(s_desc[1]);
        if (fi < sh_fin_items(fs_args(args))) {
          sh_finalize_item(fs_args(args), fi, SCL(void, SC_SDW1), SCL(void, SC_SDW2),
                           __builtin_amdgcn_readfirstlane((int)*fs_lds(&s_call[SC_SACC])));
          continue;
        }
      }
      if (RNDB && *fs_lds(&s_rr) + 1 < *fs_lds(&s_nrb)) {
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
      const int lane = MX ? fs_tid() : (int)BTID;
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
          int tNv = (kind == SK_DG2 || kind == SK_WG2 || (SH && (kind == SK_SDG2 || kind == SK_SWG2))) ? tNi : tNh;
          asm volatile("" : "+v"(tNv));
          if (SH && kind == SK_SWG2) {
            rb = li / tNv;
            col = li - rb * tNv;
            ntl = 1;
          } else if (SH && kind == SK_SWG1) {
            // The two K slices of a tile are adjacent: they finish together and the second adds the first.
            const int tl = li >> 1;
            e = li & 1;
            rb = tl / tNv;
            col = tl - rb * tNv;
            ntl = 1;
          } else if (SH && (kind == SK_SDG2 || kind == SK_SDG1)) {
            const int mt    = ka->sh.T / BLOCK_ROW;
            const int band  = li / (MOE_WALK_GM * tNv);
            const int first = band * MOE_WALK_GM;
            const int gsz   = min(mt - first, MOE_WALK_GM);
            const int q     = li - band * (MOE_WALK_GM * tNv);
            rb = first + q % gsz;
            col = q / gsz;
          } else if (ROUTER && kind == SK_RWG) {
            rb = li / tNv;
            col = li - rb * tNv;
            ntl = 1;
          } else if (kind == SK_DG2 || kind == SK_DG1 || (ROUTER && kind == SK_RDG)) {
            const int rb0 = ROUTER && kind == SK_RDG ? 0 : s_rb0[e];
            const int mt  = ROUTER && kind == SK_RDG ? ka->r.T / BLOCK_ROW : s_rb0[e + 1] - rb0;
            const int band  = li / (MOE_WALK_GM * tNv);
            const int first = band * MOE_WALK_GM;
            const int gsz   = min(mt - first, MOE_WALK_GM);
            const int q     = li - band * (MOE_WALK_GM * tNv);
            rb = rb0 + first + q % gsz;
            col = q / gsz;
          } else {
            const int wrun = kind == SK_WG2 ? ka->a.w.run : ka->b.w.run, wsh = MX ? 2 - (wrun >> 1) : 3 - wrun;
            const int nnb = kind == SK_WG2 ? ka->H / BLOCK_ROW : 2 * ka->I / BLOCK_ROW;
            const int grp = li / (tNv << wsh);
            const int r   = li - grp * (tNv << wsh);
            col = r >> wsh;
            rb = grp * MOE_WALK_GM + (r & ((1 << wsh) - 1)) * wrun;
            ntl = max(0, min(nnb - rb, wrun));
          }
          if constexpr (MX) {
            if (kind >= SK_QDY) {
              const int nch = MXR && kind == SK_RG ? 2 * ka->I / BLOCK_COL : ka->H / MX_QCOLS;
              rb = s_rb0[e] + li / nch;
              col = li % nch;
              ntl = 0;
            }
          }
        }
      }
      t = __builtin_amdgcn_readfirstlane(t);
      e = __builtin_amdgcn_readfirstlane(e);
      rb = __builtin_amdgcn_readfirstlane(rb);
      kind = __builtin_amdgcn_readfirstlane(kind);
      if (t < num_tiles) {
        gcint *dg_done = fs_g(ka->dg_done);
        if (SH && kind >= SK_SDG2 && (!MX || kind <= SK_SWG1)) {
          // fc1 dgrad waits for its row block's dGs, an fc1 wgrad slice for the dGs of its half of the rows.
          gcint *sdd = fs_g(ka->sh.dg_done);
#if MOE_BWD_SH
          if constexpr (MX) {
            // Every other kind reads the quantized dOut or X: all their blocks, acquired once per workgroup.
            if (kind != SK_SDG1 && !((*fs_lds(&s_sdok) >> 2) & 1)) {
              if (lane == 0) {
                const int nq = 2 * (ka->sh.T / BLOCK_ROW) * (ka->H / MX_QCOLS);
                int n = 0;
                while (__hip_atomic_load(fs_g(ka->sq_done), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < nq) {
                  spin_backoff(n);
                }
              }
              __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
              if (lane == 0) {
                *fs_lds(&s_sdok) |= 4;
              }
            }
          }
#endif
          if (kind == SK_SDG1) {
            if (lane == 0) {
              int n = 0;
              while (__hip_atomic_load(sdd + rb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tNi) {
                spin_backoff(n);
              }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#if MOE_BWD_SH
          } else if (kind == SK_SWG1 && !((*fs_lds(&s_sdok) >> e) & 1)) {
            const int nh = ka->sh.T / BLOCK_ROW / SH_KSPLIT;
            for (int r = e * nh + lane; r < (e + 1) * nh; r += 64) {
              int n = 0;
              while (__hip_atomic_load(sdd + r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < tNi) {
                spin_backoff(n);
              }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if (lane == 0) {
              *fs_lds(&s_sdok) |= 1 << e;
            }
#endif
          }
        } else if (kind == SK_DG1) {
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
            if constexpr (MXR) {
              // Rounds: expert e's column-wise X, quantized here (every item of its row blocks).
              const int nch = ka->H / MX_QCOLS, nq = (s_rb0[e + 1] - s_rb0[e]) * nch;
              for (int i = lane; i < nq; i += 64) {
                int n = 0;
                while (__hip_atomic_load(fs_g(ka->qx_done) + s_rb0[e] * nch + i, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT) < 1) {
                  spin_backoff(n);
                }
              }
              __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            } else if constexpr (MX) {
              // Expert e's column-wise Xp from every source. Forward-written, it precedes its source's backward, so
              // every dY row dG waited for.
              if (lane < MOE_EP) {
                const int need = mx_xcol_need(ka, lane, e);
                gcint *xs = fs_g(ka->a.g.push.peers.sig[ka->a.g.push.my_rank]) +
                            hk_comm::xcol_index(lane, e, ka->a.g.push.sigb);
                int n = 0;
                while (__hip_atomic_load(xs, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < need) {
                  spin_backoff(n);
                }
              }
              __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
            } else {
              __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            }
            if (lane == 0) {
              s_dgok[e >> 5] |= 1u << (e & 31);
            }
          }
        } else if (ROUTER && kind >= SK_RDG && (!MX || kind <= SK_RWG)) {
          // Router dX tiles wait for their rows' dlogits, dWr tiles for every row's.
          if (lane == 0) {
            gcint *c = kind == SK_RDG ? fs_g(ka->r.rpdone) + rb : fs_g(ka->r.bctr) + RC_RP;
            const int need = kind == SK_RDG ? BLOCK_ROW / RG_ROWS : ka->r.T / RG_ROWS;
            int n = 0;
            while (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < need) {
              spin_backoff(n);
            }
          }
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        } else if constexpr (MX) {
          if (kind == SK_DG2 || kind == SK_WG2) {
            // dgrad2 needs every column chunk of its row block, a wgrad2 claim (dY columns [256 rb, 256 (rb + ntl)))
            // only its chunks, over every row block of the expert.
            gcint *qd = fs_g(ka->qdy_done);
            const int nch = ka->H / MX_QCOLS, cpb = MX_QCOLS / BLOCK_ROW;
            const int c0 = kind == SK_DG2 ? 0 : rb / cpb;
            const int nc = kind == SK_DG2 ? nch : (rb + __builtin_amdgcn_readfirstlane(ntl) - 1) / cpb + 1 - c0;
            const int r0 = kind == SK_DG2 ? rb : s_rb0[e], nq = ((kind == SK_DG2 ? rb + 1 : s_rb0[e + 1]) - r0) * nc;
            for (int i = lane; i < nq; i += 64) {
              const int r = r0 + i / nc;
              int n = 0;
              while (__hip_atomic_load(qd + r * nch + c0 + i % nc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < 1) {
                spin_backoff(n);
              }
            }
            if constexpr (MXR) {
              // Rounds: dgrad2 also reads its row block's recomputed G, wgrad2 the column-wise A of the whole expert.
              if (kind == SK_DG2 || !((s_rgok[e >> 5] >> (e & 31)) & 1u)) {
                gcint *rd = fs_g(ka->rg_done);
                const int q0 = kind == SK_DG2 ? rb : s_rb0[e], q1 = kind == SK_DG2 ? rb + 1 : s_rb0[e + 1];
                for (int r = q0 + lane; r < q1; r += 64) {
                  int n = 0;
                  while (__hip_atomic_load(rd + r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < 2 * tNi) {
                    spin_backoff(n);
                  }
                }
              }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if (MXR && kind == SK_WG2 && lane == 0) {
              s_rgok[e >> 5] |= 1u << (e & 31);
            }
          } else if (MXR && kind == SK_RG) {
            // A GEMM1 recompute tile reads every column chunk of its row block's quantized X.
            const int nch = ka->H / MX_QCOLS;
            for (int i = lane; i < nch; i += 64) {
              int n = 0;
              while (__hip_atomic_load(fs_g(ka->qx_done) + rb * nch + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
                     1) {
                spin_backoff(n);
              }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
          } else {
            mx_wait_rows(ka, lane, e, rb, MXR && kind == SK_QX ? ka->xsg[ka->a.g.push.my_rank] : nullptr);
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
        if constexpr (!MX && BPART == 0) {
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
    // MXFP8: the router's dX tiles (one each) run the bf16 NT body as wgrad runs do.
    if (kd == SK_WG2 || kd == SK_WG1 || (ROUTER && kd == SK_RWG) || (SH && (kd == SK_SWG2 || kd == SK_SWG1)) ||
        (MX && ROUTER && kd == SK_RDG)) {
      w_kind = kd;
      w_sub = 0;
      w_n = MX && ROUTER && kd == SK_RDG ? 1 : __builtin_amdgcn_readfirstlane(s_desc[5]);
      continue;
    }
    const int expert    = __builtin_amdgcn_readfirstlane(s_desc[1]);
    const int block_row = __builtin_amdgcn_readfirstlane(s_desc[2]);
    const int block_col = __builtin_amdgcn_readfirstlane(s_desc[3]);
    if constexpr (MX) {
      SArgsC *ka = fs_args(args);
      const int H = ka->H, I = ka->I, r0 = block_row * BLOCK_ROW;
#if MOE_BWD_SH
      if (kd == SK_SDG2 || kd == SK_SDG1) {
        // The shared expert's dgrads on this rank's rows: fc2 dOut Ws2 with the GLU backward into dGs (row- and
        // column-wise MXFP8, Megatron's [gate; up] order), fc1 dGs Ws1 into the bf16 dXs.
        const int T = ka->sh.T;
        const int wm = __builtin_amdgcn_readfirstlane(fs_tid() / (kittens::WARP_THREADS * WARPS_COL));
        const int wn = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS % WARPS_COL);
        if (kd == SK_SDG2) {
          auto *sgl = (__attribute__((address_space(3))) float *)(uintptr_t)(uint32_t)(uintptr_t)s_sg;
          if constexpr (SGATE) {
            if (fs_tid() < BLOCK_ROW) {
              s_sg[fs_tid()] = fs_g(SCL(const float, SC_SSG))[(size_t)r0 + fs_tid()];
            }
            __syncthreads();
          }
          MxRtCT cA, cB, cC, cD;
          mx_nn_tile(s_tiles, ka->sdoq, (uint32_t)T, 2 * H, H, block_row, (const uint8_t *)SCL(const bf16, SC_SW2),
                     fs_uniform_ptr((const uint8_t *)*fs_lds(&s_mxsh[1])), I, block_col, cA, cB, cC, cD);
          const int lane_e = mx_lane_opaque();
          mx_swiglu_bwd<ACT, true, SGATE>(s_tiles, SCL(const bf16, SC_SG), (uint8_t *)fs_uniform_ptr(ka->sh.dg),
                                          ka->sdgc, ka->sdgcs, cA, cB, cC, cD, r0, wm, block_col * BLOCK_COL, wn, I,
                                          lane_e, sgl);
          if constexpr (SGATE) {
            __syncthreads();
            if (fs_tid() < BLOCK_ROW) {
              float v = 0.0f;
#pragma unroll
              for (int c = 0; c < 2 * WARPS_COL; c++) {
                v += s_sg[BLOCK_ROW * (1 + c) + fs_tid()];
              }
              fs_g(ka->sh.dsp)[(size_t)block_col * T + r0 + fs_tid()] = v;
            }
          }
          fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->sh.dg_done) + block_row);
        } else {
          MxRtC cA, cB, cC, cD;
          mx_nn_tile(s_tiles, (const uint8_t *)fs_uniform_ptr(ka->sh.dg), (uint32_t)T, 4 * I, 2 * I, block_row,
                     (const uint8_t *)SCL(const bf16, SC_SW1), fs_uniform_ptr((const uint8_t *)*fs_lds(&s_mxsh[0])), H,
                     block_col, cA, cB, cC, cD);
          const int lane_e = mx_lane_opaque();
          bf16 *tb = fs_uniform_ptr(ka->sh.dx) + (size_t)r0 * H + block_col * BLOCK_COL;
          store_c_staged<MOE_AUX_WT>(s_tiles, cA, cB, cC, cD, wm, wm + WARPS_ROW, wn, wn + WARPS_COL, lane_e, tb, H);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          __syncthreads();
          if (fs_tid() == 0) {
            (void)__hip_atomic_fetch_add(fs_g(ka->sh.dx_done) + block_row, 1, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
            (void)__hip_atomic_fetch_add(fs_g(ka->sh.dx_all), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          }
        }
        continue;
      }
#endif
      if constexpr (MXR) {
        if (kd == SK_RG) {
          // The forward's GEMM1 tile again, from the quantized X rows: G (bf16, dgrad2's) and the bf16-rounded
          // activation quantized column-wise (wgrad2's), then one release add to the row block's count.
          MxRtCT cA, cB, cC, cD;
          mx_g1_tile(s_tiles, ka->xq8, (uint32_t)*fs_lds(&s_nrow), H, 2 * I, block_row,
                     mx_ptr(4) + (size_t)expert * 2 * I * H, mx_ptr(5) + (size_t)expert * 2 * I * (H / 32), block_col,
                     cA, cB, cC, cD);
          const int lane_e = mx_lane_opaque();
          const int wm = __builtin_amdgcn_readfirstlane(fs_tid() / (kittens::WARP_THREADS * WARPS_COL));
          const int wn = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS % WARPS_COL);
          bf16 *gbase = SCALL(bf16, SC_G) + (size_t)r0 * 2 * I;
          store_T16_bf16_4<false>(gbase, cA, cB, cC, cD, wm, 2 * block_col * WARPS_COL + wn, 2 * I, lane_e);
          swiglu_inplace<ACT, true>(cA, cB);
          swiglu_inplace<ACT, true>(cC, cD);
          const int n0 = __builtin_amdgcn_readfirstlane((block_col * WARPS_COL + wn) * REG_N);
          const int m0 = __builtin_amdgcn_readfirstlane(wm * REG_M);
          const int m1 = __builtin_amdgcn_readfirstlane((WARPS_ROW + wm) * REG_M);
          uint8_t *cq = (uint8_t *)mx_head(ka, SCALL(bf16, SC_G), 0) + (size_t)r0 * I;
          uint8_t *cs = (uint8_t *)mx_head(ka, SCALL(bf16, SC_G), 1) + (size_t)(r0 / 32) * I;
          store_T16_mx_col(cq, cs, cA, m0, n0, I, lane_e);
          store_T16_mx_col(cq, cs, cC, m1, n0, I, lane_e);
          fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->rg_done) + block_row);
          continue;
        }
        if (kd == SK_QX) {
          // The block's landed X rows (zero past the expert's count) quantized once, row-wise for the recompute and
          // column-wise for wgrad1.
          const int nv = min(max(fs_g(ka->row_off)[expert] + fs_g(ka->cnt)[expert] - r0, 0), BLOCK_ROW);
          mx_quant_block<true>(ka->xrp[ka->a.g.push.my_rank], H, r0, nv, block_col * MX_QCOLS, H, ka->xq8, 2 * H,
                               (uint8_t *)mx_head(ka, SCALL(bf16, SC_G), 2),
                               (uint8_t *)mx_head(ka, SCALL(bf16, SC_G), 3));
          fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->qx_done) + block_row * (H / MX_QCOLS) + block_col);
          continue;
        }
      }
      if (kd >= SK_QDY) {
        const int nv = min(max(fs_g(ka->row_off)[expert] + fs_g(ka->cnt)[expert] - r0, 0), BLOCK_ROW);
        mx_quant_block<true>((const bf16 *)mx_ptr(3), H, r0, nv, block_col * MX_QCOLS, H, ka->dyq, 2 * H, ka->dyc,
                       ka->dycs);
        fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->qdy_done) + block_row * (H / MX_QCOLS) + block_col);
        continue;
      }
      const bool d2 = kd == SK_DG2;
      const int K = d2 ? H : 2 * I, N = d2 ? I : H;
      const uint8_t *b_q = (const uint8_t *)(d2 ? SCALL(const bf16, SC_W2) : SCALL(const bf16, SC_W1)) +
                           (size_t)expert * K * N;
      const uint8_t *b_s = mx_ptr(d2 ? 2 : 1) + (size_t)expert * (K / 32) * N;
      const int wm = __builtin_amdgcn_readfirstlane(fs_tid() / (kittens::WARP_THREADS * WARPS_COL));
      const int wn = __builtin_amdgcn_readfirstlane(fs_tid() / kittens::WARP_THREADS % WARPS_COL);
      if (d2) {
        MxRtCT cA, cB, cC, cD;
        mx_nn_tile(s_tiles, ka->dyq, (uint32_t)ka->nrow, 2 * H, K, block_row, b_q, b_s, N, block_col, cA, cB, cC, cD);
        const int lane_e = mx_lane_opaque();
        const bf16 *g = SCALL(const bf16, SC_G);
        uint8_t *dq = (uint8_t *)fs_uniform_ptr(ka->dg);
        mx_swiglu_bwd<ACT>(s_tiles, g, dq, ka->dgc, ka->dgcs, cA, cB, cC, cD, r0, wm, block_col * 512, wn, I, lane_e);
        fs_tile_publish<__ATOMIC_RELEASE>(fs_g(ka->dg_done) + block_row);
      } else {
        std::conditional_t<Q8, MxRtCT, MxRtC> cA, cB, cC, cD;
        mx_nn_tile(s_tiles, (const uint8_t *)ka->dg, (uint32_t)ka->nrow, 4 * I, K, block_row, b_q, b_s, N, block_col,
                   cA, cB, cC, cD);
        const int lane_e = mx_lane_opaque();
        if constexpr (Q8) {
          // dXp rows as MXFP8 in their 2H-byte slots: H fp8 bytes, then H / 32 E8M0 bytes (the fp8 combine).
          uint8_t *qb = (uint8_t *)fs_uniform_ptr(ka->dxp) + (size_t)r0 * 2 * N;
          const int n0 = (block_col * WARPS_COL * 2 + wn) * REG_N, n1 = n0 + WARPS_COL * REG_N;
          const int m0 = wm * REG_M, m1 = (WARPS_ROW + wm) * REG_M;
          store_T16_mx(qb, cA, m0, n0, N + n0 / 32, 2 * N, lane_e);
          store_T16_mx(qb, cB, m0, n1, N + n1 / 32, 2 * N, lane_e);
          store_T16_mx(qb, cC, m1, n0, N + n0 / 32, 2 * N, lane_e);
          store_T16_mx(qb, cD, m1, n1, N + n1 / 32, 2 * N, lane_e);
        } else {
          bf16 *tb = fs_uniform_ptr(ka->dxp) + (size_t)r0 * N + block_col * BLOCK_COL;
          store_c_staged<MOE_AUX_WT>(s_tiles, cA, cB, cC, cD, wm, wm + WARPS_ROW, wn, wn + WARPS_COL, lane_e, tb, N);
        }
        fs_tile_publish(fs_g(ka->b.c.tile_done) + block_row);
      }
      continue;
    }
    // Router dX tiles (rd) run the dgrad1 body on dlogits [T][Ed] x Wr [E][H] through one operand record (nnr, else
    // nn1: selects per operand pin SGPRs); Wr rows past E read as zero.
    // Shared fc2 dgrad (sd2: dOut Ws2, SwiGLU-bwd into dGs) and fc1 dgrad (dGs Ws1 into dXs) run on the local rows.
    const bool sd2 = SH && kd == SK_SDG2, shn = SH && kd >= SK_SDG2;
    const bool d2 = kd == SK_DG2 || sd2, rd = ROUTER && kd == SK_RDG;
    SArgsC *ka = fs_args(args);
    const __attribute__((address_space(4))) NnOps *op = nullptr;
    if constexpr (ROUTER) {
      op = fs_args(rd ? &args->nnr : &args->nn1);
    }
    const FsGl A = shn ? (sd2 ? fs_view(A0S, SCL(const bf16, SC_DOUT), ka->sh.T, ka->H)
                              : fs_view(A0S, ka->sh.dg, ka->sh.T, 2 * ka->I))
                 : d2 ? A0S : ROUTER ? fs_view(A0S, op->a, rd ? op->rows : ka->nrow, op->cols)
                                     : fs_view(A0S, ka->dg, ka->nrow, 2 * ka->I);
    const int N = d2 ? ka->I : ka->H;
    const int K = d2 ? ka->H : ROUTER ? op->cols : 2 * ka->I;
    const int KB = d2 || !ROUTER ? K : op->kb;
    const bf16 *b_base_all = shn ? (sd2 ? SCL(const bf16, SC_SW2) : SCL(const bf16, SC_SW1))
                           : ROUTER ? fs_uniform_ptr((const bf16 *)s_call[d2 ? SC_W2 : rd ? SC_WR : SC_W1])
                                    : d2 ? SCALL(const bf16, SC_W2) : SCALL(const bf16, SC_W1);
    const long long w_stride_elems = (long long)K * N;
    const int *a_gather = fs_uniform_ptr(d2 ? ka->a_gather : ROUTER ? op->gather : ka->gid);
    const int M       = A.rows();
    const int k_tiles = K / K_STEP_64;
    // The B-load LDS addresses are derived per tile (else 8 of them are hoisted out of the loop and SGPR-spilled); the
    // router and bf16 kernels derive them from the opaque thread id (a kernel-scope copy would hold an SGPR across the
    // loop).
    const uint32_t b_lds_wo = single_opaque(ROUTER || SH || !MX ? __builtin_amdgcn_readfirstlane((uint32_t)((fs_tid() >> 6) %
                                                         NUM_WARPS * elem_per_warp * sizeof(bf16)))
                                                   : b_lds_wo_k);
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
      const int r0 = shn ? gbase + ga_row[i] : a_gather[gbase + ga_row[i]];
      const int r1 = shn ? gbase + HALF_ROW + ga_row[i] : a_gather[gbase + HALF_ROW + ga_row[i]];
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
      pub = fs_g(sd2 ? fs_args(args)->sh.dg_done : fs_args(args)->dg_done);
      const size_t roff = (size_t)block_row * BLOCK_ROW * (2 * N);
      const uint32_t tile_bytes = (uint32_t)(BLOCK_ROW * 2 * N * sizeof(bf16));
      const i32x4 g_srd  = make_srsrc((sd2 ? SCL(const bf16, SC_SG) : SCALL(const bf16, SC_G)) + roff, tile_bytes);
      const i32x4 dg_srd = make_srsrc(fs_uniform_ptr(sd2 ? fs_args(args)->sh.dg : fs_args(args)->dg) + roff,
                                      tile_bytes);
      const uint32_t lane_voff = (uint32_t)((RT_C::base_tile_stride * (lane_epi / RT_C::base_tile_cols) * 2 * N +
                                            lane_epi % RT_C::base_tile_cols) * 2);
      if (sd2) {
        // Gs and dGs in Megatron's [gate; up] column order: up is N columns after gate.
        const uint32_t up = (uint32_t)N * 2;
        if constexpr (SGATE) {
          // The row block's shared gates through LDS: a global load here would wait for the stores. The tile's gate
          // gradient partials come back in LDS, summed over its 8 column units into dsp[block_col][row].
          if (fs_tid() < BLOCK_ROW) {
            s_sg[fs_tid()] = fs_g(SCL(const float, SC_SSG))[(size_t)block_row * BLOCK_ROW + fs_tid()];
          }
          __syncthreads();
          auto *sgl = (__attribute__((address_space(3))) float *)(uintptr_t)(uint32_t)(uintptr_t)s_sg;
          swiglu_bwd_c_tile<ACT, true, true>(g_srd, dg_srd, c00, rf0, cf0, 2 * N, lane_voff, up, sgl, lane_epi);
          swiglu_bwd_c_tile<ACT, true, true>(g_srd, dg_srd, c01, rf0, cf1, 2 * N, lane_voff, up, sgl, lane_epi);
          swiglu_bwd_c_tile<ACT, true, true>(g_srd, dg_srd, c10, rf1, cf0, 2 * N, lane_voff, up, sgl, lane_epi);
          swiglu_bwd_c_tile<ACT, true, true>(g_srd, dg_srd, c11, rf1, cf1, 2 * N, lane_voff, up, sgl, lane_epi);
          __syncthreads();
          if (fs_tid() < BLOCK_ROW) {
            float v = 0.0f;
#pragma unroll
            for (int c = 0; c < BLOCK_COL / RT_C::cols; c++) {
              v += s_sg[BLOCK_ROW * (1 + c) + fs_tid()];
            }
            fs_g(fs_args(args)->sh.dsp)[(size_t)block_col * fs_args(args)->sh.T + block_row * BLOCK_ROW + fs_tid()] = v;
          }
        } else {
          swiglu_bwd_c_tile<ACT, true>(g_srd, dg_srd, c00, rf0, cf0, 2 * N, lane_voff, up);
          swiglu_bwd_c_tile<ACT, true>(g_srd, dg_srd, c01, rf0, cf1, 2 * N, lane_voff, up);
          swiglu_bwd_c_tile<ACT, true>(g_srd, dg_srd, c10, rf1, cf0, 2 * N, lane_voff, up);
          swiglu_bwd_c_tile<ACT, true>(g_srd, dg_srd, c11, rf1, cf1, 2 * N, lane_voff, up);
        }
      } else {
        swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c00, rf0, cf0, 2 * N, lane_voff);
        swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c01, rf0, cf1, 2 * N, lane_voff);
        swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c10, rf1, cf0, 2 * N, lane_voff);
        swiglu_bwd_c_tile<ACT>(g_srd, dg_srd, c11, rf1, cf1, 2 * N, lane_voff);
      }
    } else if (shn) {
      const int c0 = __builtin_amdgcn_readfirstlane(warp_n), c1 = __builtin_amdgcn_readfirstlane(WARPS_COL + warp_n);
      bf16 *tb = fs_uniform_ptr(fs_args(args)->sh.dx) + (size_t)block_row * BLOCK_ROW * N + block_col * BLOCK_COL;
      store_c_staged<MOE_AUX_WT>(s_tiles, c00, c01, c10, c11, rf0, rf1, c0, c1, lane_epi, tb, N);
      pub = fs_g(fs_args(args)->sh.dx_done);
    } else {
      // dgrad1: C staged through LDS as bf16 [256][256], stored 16 B per lane write-through (sc0 sc1), so the
      // tile_done add can be RELAXED (no L2 writeback per tile).
      const int c0 = __builtin_amdgcn_readfirstlane(warp_n), c1 = __builtin_amdgcn_readfirstlane(WARPS_COL + warp_n);
      if constexpr (ROUTER) {
        const __attribute__((address_space(4))) NnOps *eo = fs_args(rd ? &args->nnr : &args->nn1);
        bf16 *tb = fs_uniform_ptr(eo->out) + (size_t)block_row * BLOCK_ROW * N + block_col * BLOCK_COL;
        store_c_staged<MOE_AUX_WT>(s_tiles, c00, c01, c10, c11, rf0, rf1, c0, c1, lane_epi, tb, N);
        pub = fs_g(eo->pub);
      } else {
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
        if (shn) {
          (void)__hip_atomic_fetch_add(fs_g(fs_args(args)->sh.dx_all), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (rd) {
          (void)__hip_atomic_fetch_add(fs_g(fs_args(args)->r.bctr) + RC_RDG, 1, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
  // A middle launch's queue is its own: its last workgroup zeroes the claim counters and the exit counter only.
  if constexpr (RNDB) {
    if (*fs_lds(&s_rr) + 1 < *fs_lds(&s_nrb)) {
      single_round_end<MX>(args, TCX, s_tiles, *fs_lds(&s_rr), *fs_lds(&s_rb0[fs_args(args)->E]), *fs_lds(&s_nwg),
                       (unsigned)__builtin_amdgcn_readfirstlane(fs_g(fs_args(args)->wep)[0]));
      single_round_info(args, &s_nrow, s_wm);
      if (fs_tid() == 0) {
        s_rr++;
      }
      __syncthreads();
      continue;
    }
  }
  if ((BPART >= 1 && BPART <= 4) && single_last_wg(single_opaque(TCX) + NUM_XCDS_AFF, s_nwg)) {
    for (int i = BTID; i <= NUM_XCDS_AFF; i += NUM_THREADS) {
      __hip_atomic_store(TCX + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  if ((BPART == 0 || BPART == 5) && single_last_wg(single_opaque(TCX) + NUM_XCDS_AFF, s_nwg)) {
    if constexpr (MX) {
      for (int i = fs_tid(); i < SINGLE_NCTR; i += NUM_THREADS) {
        __hip_atomic_store(TCX + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      for (int i = BTID; i < SINGLE_NCTR; i += NUM_THREADS) {
        __hip_atomic_store(TCX + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    gint *tile_done = fs_g(fs_args(args)->b.c.tile_done);
    gint *dg_done   = fs_g(fs_args(args)->dg_done);
    // The shared kernels re-read E here: kept live across the loop it costs a register.
    const int ner = SH ? __builtin_amdgcn_readfirstlane(fs_args(args)->E) : ne;
    int t_rs = MX ? fs_tid() : (int)BTID;
    if constexpr (Q8) {
      asm volatile("" : "+v"(t_rs));  // opaque: else its offset is kept from the start phase and spilled
    }
    for (int i = t_rs; i < s_rb0[ner]; i += NUM_THREADS) {
      __hip_atomic_store(tile_done + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(dg_done + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if constexpr (MX) {
      for (int i = fs_tid(); i < s_rb0[ner] * (fs_args(args)->H / MX_QCOLS); i += NUM_THREADS) {
        __hip_atomic_store(fs_g(fs_args(args)->qdy_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if constexpr (MXR) {
          __hip_atomic_store(fs_g(fs_args(args)->qx_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
    SArgsC *ka = fs_args(args);
    if constexpr (MXR) {
      for (int i = fs_tid(); i <= s_rb0[ner]; i += NUM_THREADS) {
        __hip_atomic_store(fs_g(ka->rg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      hk_comm::moe_bwd_dispatch_credit(ka->xsg[ka->a.g.push.my_rank], ka->a.g.cnt_by_src, ka->a.g.start_by_src,
                                       ka->row_off, ner, ka->a.g.push.sigb);
    }
    if constexpr (!MX && BPART == 0) {
      if (*fs_lds(&s_rg)) {
        for (int i = fs_tid(); i <= s_rb0[ner]; i += NUM_THREADS) {
          __hip_atomic_store(fs_g(ka->rg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      if (*fs_lds(&s_rg) == 2) {
        hk_comm::moe_bwd_dispatch_credit(ka->xsg[ka->a.g.push.my_rank], ka->a.g.cnt_by_src, ka->a.g.start_by_src,
                                         ka->row_off, ner, ka->a.g.push.sigb);
      }
    }
    if constexpr (SH) {
      const int nrs = ka->sh.T / BLOCK_ROW;
      for (int i = MX ? fs_tid() : (int)BTID; i < 2 * nrs + 2 + MX; i += NUM_THREADS) {
        __hip_atomic_store(fs_g(ka->sh.dg_done) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      for (int i = MX ? fs_tid() : (int)BTID; i < 3 * (ka->H / BLOCK_COL) * (ka->I / BLOCK_COL);
           i += NUM_THREADS) {
        __hip_atomic_store(fs_g(ka->sh.wdone) + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    hk_comm::moe_bwd_dispatch_credit(ka->a.g.push.peers.sig[ka->a.g.push.my_rank], ka->a.g.cnt_by_src,
                                     ka->a.g.start_by_src, ka->row_off, ner, ka->a.g.push.sigb);
    if constexpr (MX && !RNDB) {
      // Subtracting this call's count, as the dispatch credit, keeps an early add of the next call.
      for (int i = BTID; i < ner * MOE_EP; i += NUM_THREADS) {
        (void)__hip_atomic_fetch_add(fs_g(ka->a.g.push.peers.sig[ka->a.g.push.my_rank]) +
                                         hk_comm::xcol_index(i % MOE_EP, i / MOE_EP, ka->a.g.push.sigb),
                                     -mx_xcol_need(ka, i % MOE_EP, i / MOE_EP), __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    hk_comm::moe_bwd_combine_wait(&args->b.c);
    if constexpr (RNDB) {
      if (fs_tid() == 0 && *fs_lds(&s_nrb) > 1) {
        __hip_atomic_store(fs_g(ka->rinfo) + RINFO_GO, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    if constexpr (ROUTER) {
      moe_router_bwd_reset<true>(&args->r);
    }
    if (ka->pool) {
      pool_release(args, (int *)s_tiles);
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
