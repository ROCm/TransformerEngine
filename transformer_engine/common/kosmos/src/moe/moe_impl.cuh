/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE library (moe.h): the training instance, its symmetric arenas and tables, the launches.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "moe_state.cuh"
#include "../utils/kosmos_comm_impl.h"

namespace kosmos_moe_impl {

using namespace hk_moe;
using namespace kosmos_host;
using kittens::bf16;
typedef kittens::gl<bf16, 1, 1, -1, -1> Gl;

#define KM_TRY(x)                                                                                          \
  do {                                                                                                     \
    const hipError_t e_ = (x);                                                                             \
    if (e_ != hipSuccess) {                                                                                \
      return kosmos_host::set_error(c->err, e_, "%s at %s:%d", hipGetErrorString(e_), __FILE__, __LINE__); \
    }                                                                                                      \
  } while (0)

static bool agree(KosmosMoe *c, bool ok) {
  const int r = coll_agree(c->cfg.coll, ok);
  if (r < 0) {
    c->err = "all-gather failed";
  } else if (r == 0 && ok) {
    c->err = "refused on another rank";
  }
  return r > 0;
}

template <typename T>
static bool gather_all(KosmosMoe *c, T v, std::vector<T> &out) {
  out.assign(MOE_EP, T());
  return c->cfg.coll.allgather(c->cfg.coll.ctx, &v, out.data(), sizeof(T)) == 0;
}

struct ArenaLayout {
  size_t recv, off_sig, off_comb, comb, off_csig, off_cnt, off_flag, off_slot, off_xrecv, off_xsig, total;
};

// The workspace's arena (bwd) and the context's forward arena, both with receive rows (the forward copies Xp out).
static ArenaLayout arena_layout(const KosmosMoeConfig &g, int rows, bool bwd) {
  ArenaLayout l;
  const int sigb = (rows + BLOCK_ROW - 1) / BLOCK_ROW;
  const int nblk = (g.T + COMB_BLOCK - 1) / COMB_BLOCK;
  // The backward's signals end with each source's RFREE (moe_plan.cuh, single_round_end), the forward's with RFREE.
  const size_t sig = (size_t)(bwd ? hk_comm::bfree_index(2 * MOE_EP, sigb) : RFREE_OFF + MOE_EP) * 4;
  l.recv = (size_t)(rows + 1) * g.H * 2;
  l.off_sig = align_up(l.recv, 4096);
  l.off_comb = align_up(l.off_sig + sig, 4096);
  l.comb = (size_t)g.T * g.k * g.H * 2;
  l.off_csig = align_up(l.off_comb + l.comb, 4096);
  l.total = l.off_csig + (size_t)MOE_EP * (nblk + (bwd ? 0 : g.E_loc)) * 4;
  l.off_cnt = l.off_flag = l.off_slot = l.off_xrecv = l.off_xsig = 0;
  if (!bwd) {
    l.off_cnt  = align_up(l.total, 4096);
    l.off_flag = align_up(l.off_cnt + (size_t)2 * MOE_EP * MOE_EP * g.E_loc * 4, 4096);
    l.off_slot = align_up(l.off_flag + (size_t)MOE_EP * 4, 4096);
    l.total    = l.off_slot + (size_t)rows * 4;
  } else {
    l.off_xrecv = align_up(l.total, 4096);
    l.total     = l.off_xrecv + (size_t)(rows + 1) * g.H * 2;
    l.off_xsig  = align_up(l.total, 4096);
    l.total     = l.off_xsig + (size_t)MOE_EP * sigb * 4;
  }
  return l;
}

// Whether an arena of rows rows has A-operand offsets beyond 32 bits.
static bool rows_a32(const KosmosMoeConfig &g, int rows) {
  return (unsigned long long)(rows + 1) * g.H * 2 >= (1ull << 32) ||
         (unsigned long long)(rows + 1) * g.I * 4 >= (1ull << 32);
}

static hipError_t refuse(std::string *why, const char *m) {
  if (why) {
    *why = m;
  }
  return hipErrorInvalidValue;
}

static hipError_t validate(const KosmosMoeConfig *g, std::string *why) {
  if (!g) {
    return refuse(why, "null config");
  }
  const int H = g->H, I = g->I, E = g->E_loc, k = g->k;
  if (g->coll.nranks != MOE_EP || g->coll.rank < 0 || g->coll.rank >= MOE_EP) {
    return refuse(why, "the EP group must be 8 ranks (coll.nranks) and coll.rank in [0, 8)");
  }
  if (!g->coll.allgather) {
    return refuse(why, "coll.allgather is null");
  }
  if (H <= 0 || I <= 0 || E <= 0 || g->T <= 0) {
    return refuse(why, "H, I, E_loc and T must be positive");
  }
  if (k != 1 && k != 2 && k != 4 && k != 6 && k != 8 && k != 10) {
    return refuse(why, "k must be 1, 2, 4, 6, 8 or 10");
  }
  if (H % BLOCK_COL || I % BLOCK_COL) {
    return refuse(why, "H and I must be multiples of 256");
  }
  if ((unsigned long long)I * H * 4 >= (1ull << 32) || (unsigned long long)g->T * I * 4 >= (1ull << 32) ||
      (unsigned long long)g->T * H * 2 >= (1ull << 31)) {
    return refuse(why, "32-bit buffer offsets: 4 I H and 4 T I must be < 2^32 bytes, 2 T H < 2^31");
  }
  if (E > SIG_STRIDE || E > MOE_BWD_MAX_EXPERTS) {
    return refuse(why, "E_loc must be <= 64 and within the backward kernel's expert tables");
  }
  if (g->wgcap < 9 || g->wgcap > GRID_CAP) {
    return refuse(why, "wgcap must be in [9, 256]");
  }
  if (g->capacity_rows <= 0) {
    return refuse(why, "capacity_rows must be > 0");
  }
  if (g->act != KOSMOS_MOE_ACT_SWIGLU && g->act != KOSMOS_MOE_ACT_SWIGLU_CLAMP10) {
    return refuse(why, "act: KOSMOS_MOE_ACT_SWIGLU or KOSMOS_MOE_ACT_SWIGLU_CLAMP10");
  }
  return hipSuccess;
}

// The small-k kernels take k in {1, 2, 4}, the others k in {6, 8, 10}; validate refuses any other k.
static bool small_k(const KosmosMoeConfig &g) {
  return g.k <= 4;
}

// The most rows one call can need on a rank: every rank's T k rows, each (source, expert) segment padded.
static int rows_max(const KosmosMoeConfig &g) {
  const long long n = (long long)MOE_EP * g.T * g.k + (long long)g.E_loc * (MOE_EP * 31 + EXPERT_PADDING);
  return (int)((n + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW);
}

static void free_tables(KosmosMoe *c) {
  KosmosMoeFwd &f = c->f;
  f.act = nullptr;
  dfree(f.cnt), dfree(f.roff), dfree(f.gather), dfree(f.gid), dfree(f.startby), dfree(f.cntby);
  dfree(f.sendslot), dfree(f.lbsrc), dfree(f.owed), dfree(f.rbhi), dfree(f.tiledone);
  dfree(f.t1), dfree(f.t2), dfree(f.blkctr), dfree(f.sbase);
  dfree(f.rbs), dfree(f.q1), dfree(f.q2);
  dfree(f.wt), dfree(f.adone), dfree(f.args), dfree(f.ep);
  KosmosMoeBwd &b = c->b;
  dfree(b.args), dfree(b.blks), dfree(b.cblks), dfree(b.cntbysrc), dfree(b.startbysrc), dfree(b.sendw);
  KosmosMoePlan &p = c->p;
  dfree(p.hist), dfree(p.pk), dfree(p.status), dfree(p.rec);
}

// The workspace's buffers in the context's device arguments.
static void set_shared(const KosmosMoe *c, MoeFwdArgs *fa, SingleArgs *ba) {
  const KosmosMoeWorkspace *w = c->ws;
  fa->c2 = w->y;
  fa->wep = w->wep;
  fa->r.dtk = w->dtk;
  ba->r.dtk = w->dtk;
  ba->ascr = w->ascr, ba->rg_done = w->rgdone, ba->fa = c->replan_args, ba->rinfo = w->rinfo;
  for (int i = 0; i < MOE_EP; i++) {
    ba->xrp[i] = w->xrecvp[i], ba->xsg[i] = w->xsigp[i];
  }
  ba->wep = w->wep;
  ba->nn1.a = w->dg, ba->nn1.out = w->y, ba->nn1.pub = w->tdone;
  NnArrivalGate &hg = ba->a.g;
  hg.sig = w->sig, hg.push.peers = w->dpeers, hg.push.sigb = w->sigb;
  ba->b.s.dtk = w->dtk;
  hk_comm::CCombine &hb = ba->b.c;
  hb.Xp = (const unsigned short *)w->y, hb.tile_done = w->tdone, hb.peers = w->cpeers;
  ba->dg = w->dg, ba->dxp = w->y, ba->dg_done = w->dgdone;
}

// Every table at its largest size for rows = arena_rows, allocated once; every forward fills them in place.
static hipError_t alloc_tables(KosmosMoe *c) {
  const KosmosMoeConfig &g = c->cfg;
  const int H = g.H, I = g.I, E = g.E_loc, TK = g.T * g.k, me = g.coll.rank, N1 = 2 * I;
  const int R = c->arena_rows, nrow = R + 1, nrbm = R / BLOCK_ROW, nu = E * MOE_EP;
  const int nch = (TK + PLAN_CH - 1) / PLAN_CH;
  KosmosMoeFwd &f = c->f;
  KosmosMoeBwd &b = c->b;
  KosmosMoePlan &p = c->p;
  KM_TRY(dzero(&f.cnt, E));
  KM_TRY(dzero(&f.roff, E));
  KM_TRY(dzero(&f.gather, nrow));
  KM_TRY(dzero(&f.gid, nrow));
  KM_TRY(dzero(&f.startby, (size_t)SIG_STRIDE * MOE_EP));
  KM_TRY(dzero(&f.cntby, (size_t)SIG_STRIDE * MOE_EP));
  KM_TRY(dzero(&f.sendslot, TK));
  KM_TRY(dzero(&f.lbsrc, R));
  KM_TRY(dzero(&f.owed, MOE_EP));
  KM_TRY(dzero(&f.sbase, nu));
  KM_TRY(dzero(&f.rbhi, nu));
  KM_TRY(dzero(&f.tiledone, nrbm));
  KM_TRY(dzero(&f.adone, nrbm));
  KM_TRY(dzero(&f.t1, NUM_XCDS_AFF + 1));
  KM_TRY(dzero(&f.t2, 4 + PC_N));
  KM_TRY(dzero(&f.blkctr, 2 + nu));
  KM_TRY(dzero(&f.rbs, nu));
  KM_TRY(dzero(&f.q1, (size_t)nrbm * (N1 / BLOCK_COL)));
  KM_TRY(dzero(&f.q2, (size_t)nrbm * (H / BLOCK_COL)));
  KM_TRY(dzero(&f.wt, TK));
  KM_TRY(dzero(&f.ep, 2));
  KM_TRY(dalloc(&f.args, 1));
  KM_TRY(dzero(&p.hist, (size_t)nch * nu));
  KM_TRY(dzero(&p.pk, TK));
  KM_TRY(dzero(&p.status, 1));
  KM_TRY(dzero(&p.rec, 1));
  SingleArgs hb;
  memset(&hb, 0, sizeof hb);
  KM_TRY(dzero(&b.blks, nu));
  KM_TRY(dzero(&b.cblks, nu));
  KM_TRY(dzero(&b.sendw, TK));
  KM_TRY(dzero(&b.cntbysrc, (size_t)MOE_EP * SIG_STRIDE));
  KM_TRY(dzero(&b.startbysrc, (size_t)MOE_EP * SIG_STRIDE));
  NnArrivalGate &hg = hb.a.g;
  hg.cnt_by_src = b.cntbysrc, hg.start_by_src = b.startbysrc;
  hg.push.send_w = b.sendw, hg.push.blks = b.blks, hg.push.send_slot = f.sendslot;
  hg.push.comby = (const unsigned short *)f.comb, hg.push.k = g.k;
  hg.push.nblks = nu, hg.push.my_rank = me, hg.push.H = H;
  LegAWgrad2 &hw = hb.a.w;
  hw.act = f.act, hw.act_cols = I;
  hk_comm::CCombine &hc = hb.b.c;
  hc.blks = b.cblks, hc.src = f.lbsrc, hc.slot = f.slot[me];
  hc.owed = f.owed, hc.nblks = nu, hc.my_rank = me, hc.H = H;
  LegAWgrad2 &hw1 = hb.b.w;
  hw1.act = f.recv, hw1.act_cols = H;
  hg.push.ncomm = b.dncomm, hc.ncomm = b.cncomm, hw.run = b.wg2run, hw1.run = b.wg1run;
  hb.b.s.pk = p.pk, hb.b.s.T = g.T, hb.b.s.k = g.k, hb.b.s.E = nu;
  hb.row_off = f.roff, hb.cnt = f.cnt, hb.a_gather = f.gather, hb.gid = f.gid;
  hb.H = H, hb.I = I, hb.E = E, hb.ep = f.ep;
  hb.nn1 = NnOps{nullptr, f.gid, nullptr, nullptr, 0, N1, N1};
  KM_TRY(dalloc(&b.args, 1));
  MoeFwdArgs ha;
  memset(&ha, 0, sizeof ha);
  ha.send_slot = f.sendslot, ha.dpeers = f.dpeers, ha.my_sig = f.sig;
  ha.cnt_by_src = f.cntby, ha.start_by_src = f.startby, ha.row_off = f.roff, ha.q1 = f.q1, ha.q2 = f.q2;
  ha.gather = f.gather, ha.gid = f.gid, ha.act = f.act, ha.a_done = f.adone;
  ha.tile_done = f.tiledone, ha.g2_ctr = f.t2, ha.disp_done = f.t2 + 1, ha.fold_ctr = f.blkctr;
  ha.idx_done = f.t2 + 2, ha.cnt = f.cnt, ha.E_loc = E;
  ha.rbs = f.rbs, ha.lb_src = f.lbsrc, ha.rb_hi = f.rbhi, ha.lpeers = f.lpeers, ha.wts = f.wt, ha.owed = f.owed;
  for (int i = 0; i < MOE_EP; i++) {
    ha.slot[i] = f.slot[i], ha.p.cnts[i] = f.cnts[i], ha.p.cflag[i] = f.cflag[i];
  }
  ha.nblks = nu, ha.nrbs = nu, ha.ep = f.ep;
  ha.my_rank = me, ha.dncomm = f.ncomm, ha.cncomm = f.fcomm;
  ha.H = H, ha.I = I, ha.T = g.T, ha.topk = g.k;
  MoePlanTabs &hp = ha.p;
  hp.ctr = f.t2 + 3, hp.hist = p.hist, hp.pk = p.pk;
  ha.rgo = f.t2 + 3 + PC_N, ha.sbase = f.sbase;
  hp.zf = f.recv, hp.rec = (int *)p.rec, hp.status = p.status;
  hp.cap = R, hp.nch = nch;
  hp.send_w = b.sendw, hp.bblks = (int *)b.blks, hp.cblks = (int *)b.cblks;
  hp.cnt_by_src = b.cntbysrc, hp.start_by_src = b.startbysrc, hp.ba = b.args;
  set_shared(c, &ha, &hb);
  KM_TRY(hipMemcpy(b.args, &hb, sizeof hb, hipMemcpyHostToDevice));
  KM_TRY(hipMemcpy(f.args, &ha, sizeof ha, hipMemcpyHostToDevice));
  return hipSuccess;
}

// Local; callers bracket it with agreements (no peer may be inside a call while the arena is zeroed).
static hipError_t reset_layer(KosmosMoe *c) {
  KosmosMoeFwd &f = c->f;
  KM_TRY(hipMemset(f.arena.base, 0, f.arena.bytes));
  KM_TRY(hipMemset(c->p.status, 0, sizeof(int)));
  KM_TRY(hipMemset(f.ep, 0, 2 * sizeof(unsigned)));
  KM_TRY(hipDeviceSynchronize());
  return hipSuccess;
}

static void ws_close(KosmosMoeWorkspace *w) {
  dfree(w->t), dfree(w->tdone), dfree(w->dgdone), dfree(w->y), dfree(w->dg), dfree(w->dtk);
  dfree(w->wep), dfree(w->ascr), dfree(w->rgdone), dfree(w->rinfo);
  w->arena = KosmosMoeArena{};
  w->recv = nullptr, w->sig = nullptr, w->xrecv = nullptr;
  for (int i = 0; i < MOE_EP; i++) {
    w->xrecvp[i] = nullptr, w->xsigp[i] = nullptr;
  }
  w->rows = 0;
}

static hipError_t ws_alloc(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const int nrbm = rows / BLOCK_ROW;
  KM_TRY(dzero(&w->wep, 2));
  KM_TRY(dalloc(&w->y, (size_t)(rows + 1) * g.H));
  KM_TRY(dzero(&w->t, 16));
  KM_TRY(dzero(&w->tdone, nrbm));
  KM_TRY(dzero(&w->dgdone, nrbm));
  KM_TRY(dalloc(&w->dg, (size_t)(rows + 1) * 2 * g.I));
  KM_TRY(dalloc(&w->dtk, (size_t)g.T * g.k));
  KM_TRY(dalloc(&w->ascr, (size_t)(rows + 1) * g.I));
  KM_TRY(dzero(&w->rgdone, nrbm + 1));
  KM_TRY(dzero(&w->rinfo, RINFO_WM + MOE_BWD_MAX_EXPERTS));
  KM_TRY(hipMemset(w->arena.base, 0, w->arena.bytes));
  KM_TRY(hipDeviceSynchronize());
  return hipSuccess;
}

// Collective: maps the workspace's arena (the comm from byte 0) and allocates its buffers at rows rows, all zeroed.
static hipError_t ws_open(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const int me = g.coll.rank;
  const ArenaLayout bl = arena_layout(g, rows, true);
  KosmosCommInfo ci;
  const hipError_t ie = kosmos_comm_info(w->comm, &ci);
  if (!agree(c, ie == hipSuccess && ci.bytes >= bl.total)) {
    if (ie != hipSuccess || ci.bytes < bl.total) {
      set_error(c->err, hipErrorInvalidValue, "the comm holds %zu bytes; the workspace needs %zu",
                ie == hipSuccess ? ci.bytes : 0, bl.total);
    }
    ws_close(w);
    return hipErrorInvalidValue;
  }
  w->arena = KosmosMoeArena{ci.local, bl.total};
  w->off_sig = bl.off_sig, w->off_comb = bl.off_comb, w->off_csig = bl.off_csig;
  w->sigb = (rows + BLOCK_ROW - 1) / BLOCK_ROW;
  for (int i = 0; i < MOE_EP; i++) {
    char *p = (char *)ci.peers[i];
    w->dpeers.recv[i] = (bf16 *)p;
    w->dpeers.sig[i] = (int *)(p + bl.off_sig);
    w->cpeers.comb[i] = (bf16 *)(p + bl.off_comb);
    w->cpeers.acc[i] = (int *)(p + bl.off_csig);
    w->xrecvp[i] = (bf16 *)(p + bl.off_xrecv);
    w->xsigp[i] = (int *)(p + bl.off_xsig);
  }
  w->recv = w->dpeers.recv[me], w->sig = w->dpeers.sig[me];
  w->xrecv = w->xrecvp[me];
  int bad = 0;
  for (int d = 0; d < MOE_EP && !bad; d++) {
    const int tok = 0x5A5A0000 | (me + 1);
    bad |= hipMemcpy(w->dpeers.sig[d] + me, &tok, 4, hipMemcpyHostToDevice) != hipSuccess;
  }
  bad |= hipDeviceSynchronize() != hipSuccess;
  if (!agree(c, !bad)) {
    ws_close(w);
    return set_error(c->err, hipErrorUnknown, "workspace arena: acceptance writes failed");
  }
  std::vector<int> got(MOE_EP, 0);
  if (hipMemcpy(got.data(), w->sig, (size_t)MOE_EP * 4, hipMemcpyDeviceToHost) != hipSuccess) {
    bad = MOE_EP;
  }
  for (int s = 0; s < MOE_EP && bad < MOE_EP; s++) {
    bad += got[(size_t)s] != (0x5A5A0000 | (s + 1));
  }
  if (!agree(c, !bad)) {
    if (bad) {
      set_error(c->err, hipErrorUnknown, "workspace arena ACCEPTANCE FAILED: %d/%d peer writes did not arrive", bad,
                MOE_EP);
    }
    ws_close(w);
    return hipErrorUnknown;
  }
  const hipError_t e = ws_alloc(c, rows);
  if (!agree(c, e == hipSuccess)) {
    ws_close(w);
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  w->rows = rows;
  return hipSuccess;
}

// Collective: opens the workspace, maps the context's forward arena (the comm from inst_foff) at rows rows, allocates
// its tables and zeroes its state.
static hipError_t open_arenas(KosmosMoe *c, int rows) {
  const KosmosMoeConfig &g = c->cfg;
  KosmosMoeFwd &f = c->f;
  const ArenaLayout fl = arena_layout(g, rows, false);
  f.ncomm = std::min(moe_prod_ncomm(g.I), g.wgcap / NUM_XCDS_AFF * NUM_XCDS_AFF);
  f.burst = moe_prod_burst(g.H);
  f.fcomm = std::min(128, g.wgcap / NUM_XCDS_AFF * NUM_XCDS_AFF);
  {
    // dncomm must be equal on every rank (the dispatch gate counts dncomm / 8 slices per source): min over ranks.
    std::vector<int> nc;
    if (!gather_all(c, f.ncomm, nc)) {
      return set_error(c->err, hipErrorUnknown, "ncomm all-gather failed");
    }
    f.ncomm = *std::min_element(nc.begin(), nc.end());
  }
  const bool a32 = rows_a32(g, rows);
  if (!agree(c, !a32)) {
    if (a32) {
      set_error(c->err, hipErrorInvalidValue, "arena of %d rows refused: A-operand offsets exceed 32 bits", rows);
    }
    return hipErrorInvalidValue;
  }
  hipError_t e = ws_open(c, rows);
  if (e != hipSuccess) {
    return e;
  }
  const size_t aoff = c->ws->inst_foff;
  KosmosCommInfo ci;
  e = kosmos_comm_info(c->ws->comm, &ci);
  if (e == hipSuccess && ci.bytes < aoff + fl.total) {
    e = set_error(c->err, hipErrorInvalidValue, "the comm holds %zu bytes; the instance needs %zu", ci.bytes,
                  aoff + fl.total);
  }
  if (!agree(c, e == hipSuccess)) {
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  f.arena = KosmosMoeArena{(char *)ci.local + aoff, fl.total};
  f.comb_bytes = fl.comb;
  for (int i = 0; i < MOE_EP; i++) {
    char *p = (char *)ci.peers[i] + aoff;
    f.dpeers.recv[i] = (bf16 *)p;
    f.dpeers.sig[i] = (int *)(p + fl.off_sig);
    f.lpeers.comb[i] = (bf16 *)(p + fl.off_comb);
    f.lpeers.acc[i] = (int *)(p + fl.off_csig);
    f.cnts[i] = (int *)(p + fl.off_cnt);
    f.cflag[i] = (int *)(p + fl.off_flag);
    f.slot[i] = (int *)(p + fl.off_slot);
  }
  const int me = g.coll.rank;
  f.recv = f.dpeers.recv[me], f.comb = f.lpeers.comb[me], f.sig = f.dpeers.sig[me];
  // Xp lands in the forward arena; A goes to the workspace's ascr unless the call keeps it (its caller's).
  f.act = c->ws->ascr;
  c->arena_rows = rows;
  c->info.arena_rows = rows;
  e = alloc_tables(c);
  if (e == hipSuccess) {
    e = reset_layer(c);
  }
  if (!agree(c, e == hipSuccess)) {
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  KosmosMoeInfo &in = c->info;
  in.fwd_ncomm = f.ncomm, in.fwd_fcomm = f.fcomm, in.fwd_burst = f.burst;
  in.bwd_grid = g.wgcap;
  return hipSuccess;
}

// The backward's crews and wgrad run lengths, once from the shape: the cost model fed with the balanced routing's
// T k / (EP E_loc) rows per (source, expert).
static void bwd_shape_plan(KosmosMoe *c) {
  const KosmosMoeConfig &g = c->cfg;
  const int E = g.E_loc, H = g.H, I = g.I;
  const double u     = (double)g.T * g.k / (MOE_EP * E);
  const double col   = MOE_EP * u;
  const double mpad  = E * ceil(col / EXPERT_PADDING) * EXPERT_PADDING;
  const double kk    = E * ceil(col / 64) * 64;
  const double off   = (double)g.T * g.k * (MOE_EP - 1) / MOE_EP;
  const double tflop = 2.0 * 2.0 * mpad * H * I / 1e12;
  const double cgb   = off * H * 2 / 1e9;
  const double tiles = (double)E * (H / BLOCK_COL) * (I / BLOCK_COL);
  KosmosMoeBwd &b = c->b;
  b.dncomm = hk_comm::c_dispatch_ncomm(cgb, tflop);
  if (b.dncomm + 1 > g.wgcap) {
    b.dncomm = ((g.wgcap - 1) / NUM_XCDS_AFF) * NUM_XCDS_AFF;
  }
  const int cn = hk_comm::c_combine_ncomm(cgb, tflop);
  b.cncomm = cn > g.wgcap ? (g.wgcap / NUM_XCDS_AFF) * NUM_XCDS_AFF : cn;
  b.wg2run = hk_comm::c_wgrad2_run(tiles, g.wgcap, kk / E);
  b.wg1run = hk_comm::c_wgrad2_run(2.0 * tiles, g.wgcap, kk / E);
  KosmosMoeInfo &in = c->info;
  in.bwd_dpush_gb = cgb, in.bwd_lega_tflop = tflop, in.bwd_cpush_gb = cgb, in.bwd_cncomm_clamped = cn > g.wgcap;
  in.bwd_wg2_tiles = tiles, in.bwd_wg2_meank = kk / E, in.bwd_wg2run = b.wg2run, in.bwd_wg1run = b.wg1run;
  for (int q = 0; q < MOE_EP; q++) {
    in.bwd_dncomm_all[q] = b.dncomm, in.bwd_cncomm_all[q] = b.cncomm;
  }
}

// The capacity rows of an instance (cfg.capacity_rows rounded up to 256, at most what one call can need).
static int inst_rows(const KosmosMoeConfig &g) {
  return std::min((g.capacity_rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW, rows_max(g));
}

// The instance's symmetric layout: the workspace's arena from 0, its context's forward arena from the returned offset.
static size_t inst_foff(const KosmosMoeConfig &g, int rows) {
  return align_up(arena_layout(g, rows, true).total, 4096);
}

static size_t inst_bytes(const KosmosMoeConfig &g, int rows) {
  return inst_foff(g, rows) + arena_layout(g, rows, false).total;
}

// One table the record keeps: its base, and its bytes at the capacity (rounded to 16; the allocations' tails are
// padding).
struct RouteTab {
  const void *p;
  size_t bytes;
};

// The tables a backward reads from its forward (single pass; a call in rounds re-derives all but the routing order's
// from the record's count matrix), in a fixed order.
static std::vector<RouteTab> route_tables(const KosmosMoe *c, const KosmosMoeConfig &g, int rows) {
  const size_t E = g.E_loc, TK = (size_t)g.T * g.k, nu = E * MOE_EP, R = rows, nrow = R + 1;
  const KosmosMoeFwd &f = c->f;
  const KosmosMoeBwd &b = c->b;
  const KosmosMoePlan &p = c->p;
  return {{f.cnt, E * 4},        {f.roff, E * 4},       {f.gather, nrow * 4},  {f.gid, nrow * 4},
          {f.sendslot, TK * 4},  {f.lbsrc, R * 4},      {f.owed, MOE_EP * 4},  {f.sbase, nu * 4},
          {p.pk, TK * 4},        {b.blks, nu * 20},     {b.cblks, nu * 20},    {b.sendw, TK * 4},
          {b.cntbysrc, (size_t)MOE_EP * SIG_STRIDE * 4}, {b.startbysrc, (size_t)MOE_EP * SIG_STRIDE * 4}};
}

// The record's layout at rows: the header and count matrix, the receive slots at *off_slot, then each table (its pieces
// of at most 64 KiB are the copy items) 256-byte aligned; returns its bytes.
static size_t route_layout(const KosmosMoeConfig &g, int rows, const std::vector<RouteTab> *tabs, size_t *off_slot,
                           std::vector<RouteSeg> *segs) {
  const size_t EG = (size_t)MOE_EP * g.E_loc, E = g.E_loc, TK = (size_t)g.T * g.k, nu = E * MOE_EP;
  const size_t R = rows, nrow = R + 1;
  const size_t sizes[] = {E * 4,  E * 4,  nrow * 4, nrow * 4, TK * 4, R * 4, MOE_EP * 4, nu * 4,
                          TK * 4, nu * 20, nu * 20, TK * 4,  (size_t)MOE_EP * SIG_STRIDE * 4,
                          (size_t)MOE_EP * SIG_STRIDE * 4};
  *off_slot = align_up(ROUTE_COUNTS_OFF + (size_t)MOE_EP * EG * 4, 256);
  size_t off = align_up(*off_slot + R * 4, 256);
  constexpr size_t PIECE = 65536;
  for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
    const size_t n = align_up(sizes[i], 16);
    for (size_t q = 0; tabs && segs && q < n; q += PIECE) {
      segs->push_back(RouteSeg{(char *)(*tabs)[i].p + q, (unsigned)(off + q), (unsigned)std::min(PIECE, n - q)});
    }
    off = align_up(off + n, 256);
  }
  return off;
}

static void inst_free(KosmosMoeInstance *in) {
  dfree(in->ctr), dfree(in->seg), dfree(in->scnt), dfree(in->sep);
  if (in->scratch) {
    dfree(in->scratch->replan_args);
  }
}

// After the context's arenas and tables: the backward's forward arguments (count slots and epochs staged), the counters
// and the record's segments.
static hipError_t inst_setup(KosmosMoeInstance *in) {
  KosmosMoe *c = in->scratch;
  const KosmosMoeConfig &g = in->cfg;
  const size_t EG = (size_t)MOE_EP * g.E_loc;
  MoeFwdArgs ha;
  KM_TRY(hipMemcpy(&ha, c->f.args, sizeof ha, hipMemcpyDeviceToHost));
  ha.p.cnts[g.coll.rank] = in->scnt, ha.ep = in->sep;
  KM_TRY(hipMemcpy(c->replan_args, &ha, sizeof ha, hipMemcpyHostToDevice));
  std::vector<RouteSeg> segs;
  const std::vector<RouteTab> tabs = route_tables(c, g, in->rows);
  in->route_bytes = route_layout(g, in->rows, &tabs, &in->off_slot, &segs);
  in->nseg = (int)segs.size();
  KM_TRY(dupload(&in->seg, segs));
  KM_TRY(hipMemset(in->scnt, 0, 2 * MOE_EP * EG * sizeof(int)));
  KM_TRY(hipDeviceSynchronize());
  return hipSuccess;
}

// The context's arenas and tables (if opened), then the workspace's buffers.
static hipError_t inst_close(KosmosMoeInstance *in) {
  hipError_t e = hipDeviceSynchronize();
  inst_free(in);
  if (in->scratch) {
    free_tables(in->scratch);
    delete in->scratch;
    in->scratch = nullptr;
  }
  if (in->ws) {
    ws_close(in->ws);
    delete in->ws;
    in->ws = nullptr;
  }
  if (in->own_comm && in->comm) {
    const hipError_t e2 = kosmos_comm_destroy(in->comm);
    e = e != hipSuccess ? e : e2;
  }
  return e;
}

}

hipError_t kosmos_moe_instance_bytes(const KosmosMoeConfig *cfg, size_t *bytes) {
  using namespace kosmos_moe_impl;
  if (!bytes || validate(cfg, nullptr) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  *bytes = inst_bytes(*cfg, inst_rows(*cfg));
  return hipSuccess;
}

hipError_t kosmos_moe_saved_bytes(const KosmosMoeConfig *cfg, int rows, KosmosMoeSavedBytes *bytes) {
  using namespace kosmos_moe_impl;
  if (!bytes || validate(cfg, nullptr) != hipSuccess || rows < 1 || rows > inst_rows(*cfg)) {
    return hipErrorInvalidValue;
  }
  size_t off_slot;
  const size_t H = cfg->H, I = cfg->I, r = (size_t)rows;
  bytes->xp = align_up((r + 1) * H * 2, 256);
  bytes->g = align_up(r * 2 * I * 2, 256);
  bytes->a = align_up(r * I * 2, 256);
  bytes->y = align_up((size_t)cfg->T * cfg->k * H * 2, 256);
  bytes->route = route_layout(*cfg, inst_rows(*cfg), nullptr, &off_slot, nullptr);
  return hipSuccess;
}

hipError_t kosmos_moe_instance_create(KosmosMoeInstance **inst, KosmosComm *comm, const KosmosMoeConfig *cfg) {
  using namespace kosmos_moe_impl;
  if (!inst) {
    return hipErrorInvalidValue;
  }
  *inst = nullptr;
  if (validate(cfg, nullptr) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  KosmosMoeInstance *in = new KosmosMoeInstance();
  in->cfg = *cfg;
  in->rows = inst_rows(in->cfg);
  in->cfg.capacity_rows = in->rows;
  const KosmosMoeConfig &g = in->cfg;
  const size_t need = inst_bytes(g, in->rows), EG = (size_t)MOE_EP * g.E_loc;
  hipError_t e = hipSuccess;
  if (comm) {
    KosmosCommInfo ci;
    e = kosmos_comm_info(comm, &ci);
    if (e == hipSuccess && (ci.rank != g.coll.rank || ci.nranks != g.coll.nranks || ci.bytes < need)) {
      e = set_error(in->err, hipErrorInvalidValue, "instance: the comm holds %zu bytes (rank %d of %d); it needs %zu"
                    " (kosmos_moe_instance_bytes)", ci.bytes, ci.rank, ci.nranks, need);
    }
    in->comm = comm;
  } else {
    e = kosmos_comm_create(&in->comm, &g.coll, need);
    in->own_comm = e == hipSuccess;
    if (e != hipSuccess) {
      in->comm = nullptr;
      set_error(in->err, e, "instance comm (%zu bytes): %s", need, hipGetErrorString(e));
    }
  }
  if (e == hipSuccess) {
    KosmosMoeWorkspace *w = new KosmosMoeWorkspace();
    w->cfg = g;
    w->comm = in->comm;
    w->inst_foff = inst_foff(g, in->rows);
    in->ws = w;
    KosmosMoe *c = new KosmosMoe();
    c->cfg = g;
    c->ws = w;
    bwd_shape_plan(c);
    in->scratch = c;
    e = dzero(&in->ctr, INST_CTR_N);
    e = e == hipSuccess ? dzero(&in->scnt, 2 * MOE_EP * EG) : e;
    e = e == hipSuccess ? dzero(&in->sep, 2) : e;
    // Before open_arenas: its set_shared points the backward's ba->fa here, and inst_setup fills it afterwards.
    e = e == hipSuccess ? dalloc(&c->replan_args, 1) : e;
    if (e != hipSuccess) {
      set_error(in->err, e, "instance state: %s", hipGetErrorString(e));
    }
  }
  // Collective from here: every rank takes the same steps (open_arenas agrees across ranks).
  const KosmosCollective &coll = g.coll;
  if (coll_agree(coll, e == hipSuccess) <= 0) {
    if (e == hipSuccess) {
      in->err = "instance: refused on another rank";
      e = hipErrorUnknown;
    }
  } else {
    e = open_arenas(in->scratch, in->rows);
    if (e != hipSuccess) {
      in->err = in->scratch->err;
    } else {
      e = inst_setup(in);
      if (e != hipSuccess) {
        in->err = in->scratch->err;
      }
    }
    if (coll_agree(coll, e == hipSuccess) <= 0 && e == hipSuccess) {
      in->err = "instance: setup refused on another rank";
      e = hipErrorUnknown;
    }
  }
  if (e != hipSuccess) {
    (void)inst_close(in);
    delete in;
    return e;
  }
  *inst = in;
  return hipSuccess;
}

hipError_t kosmos_moe_instance_destroy(KosmosMoeInstance *in) {
  using namespace kosmos_moe_impl;
  if (!in) {
    return hipSuccess;
  }
  const hipError_t e = inst_close(in);
  delete in;
  return e;
}

const char *kosmos_moe_instance_error(const KosmosMoeInstance *in) {
  return !in ? "null instance" : !in->err.empty() || !in->scratch ? in->err.c_str() : in->scratch->err.c_str();
}

const KosmosMoeInfo *kosmos_moe_instance_info(const KosmosMoeInstance *in) {
  return in && in->scratch ? &in->scratch->info : nullptr;
}

namespace kosmos_moe_impl {

// A saved state's pointers against its tier (every one 256-byte aligned) and rows.
static hipError_t saved_ok(KosmosMoeInstance *in, const KosmosMoeSaved *sv) {
  const auto al = [](const void *p) { return (uintptr_t)p % 256 == 0; };
  if (sv->tier < KOSMOS_MOE_SAVE_NONE || sv->tier > KOSMOS_MOE_SAVE_ALL || sv->rows < 1 || sv->rows > in->rows ||
      !sv->y || !sv->route || (sv->tier >= KOSMOS_MOE_SAVE_XP && !sv->xp) ||
      (sv->tier == KOSMOS_MOE_SAVE_ALL && (!sv->g || !sv->a)) || !al(sv->xp) || !al(sv->g) || !al(sv->a) ||
      !al(sv->y) || !al(sv->route)) {
    return set_error(in->err, hipErrorInvalidValue, "saved: tier in [0, 2], rows in [1, %d], y and route set, xp"
                     " (tier >= 1), g and a (tier 2) set, every pointer 256-byte aligned", in->rows);
  }
  return hipSuccess;
}

}

hipError_t kosmos_moe_train_forward(KosmosMoeInstance *in, const void *X, const void *W1, const void *W2,
                                    const int *topk_idx, const float *topk_w, void *out, const KosmosMoeSaved *saved,
                                    hipStream_t s) {
  using namespace kosmos_moe_impl;
  if (!in || !in->scratch) {
    return hipErrorInvalidValue;
  }
  in->err.clear();
  if (!X || !W1 || !W2 || !topk_idx || !topk_w || !out) {
    return set_error(in->err, hipErrorInvalidValue, "train_forward: X, W1, W2, topk_idx, topk_w and out must be set");
  }
  if (saved && saved_ok(in, saved) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  KosmosMoe *c = in->scratch;
  const KosmosMoeConfig &g = in->cfg;
  const KosmosMoeFwd &f = c->f;
  const Gl A_gl((bf16 *)f.recv, nullptr, nullptr, (size_t)c->arena_rows + 1, (size_t)g.H);
  const TrainApiCall v = {saved ? (bf16 *)saved->xp : nullptr, saved ? (bf16 *)saved->g : nullptr,
                          saved ? (bf16 *)saved->a : nullptr, saved ? (bf16 *)saved->y : nullptr,
                          saved ? (uint8_t *)saved->route : nullptr, in->ctr, in->seg, in->nseg,
                          saved ? saved->tier : -1, saved ? saved->rows : 0, (int)in->off_slot};
  const bf16 *x = (const bf16 *)X, *w1 = (const bf16 *)W1, *w2 = (const bf16 *)W2;
  bf16 *o = (bf16 *)out;
  if (small_k(g) && g.act) {
    moe_train_api_launch_fwd_ks<GluClamp10>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else if (small_k(g)) {
    moe_train_api_launch_fwd_ks<SwiGLU>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else if (g.act) {
    moe_train_api_launch_fwd<GluClamp10>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else {
    moe_train_api_launch_fwd<SwiGLU>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  }
  const hipError_t e = hipGetLastError();
  return e == hipSuccess ? e : set_error(in->err, e, "train_forward launch: %s", hipGetErrorString(e));
}

hipError_t kosmos_moe_train_backward(KosmosMoeInstance *in, const void *dout, const void *X, const void *W1,
                                     const void *W2, const int *topk_idx, const float *topk_w,
                                     const KosmosMoeSaved *saved, void *dX, float *dprobs, void *dW1, void *dW2,
                                     int acc, hipStream_t s) {
  using namespace kosmos_moe_impl;
  if (!in || !in->scratch) {
    return hipErrorInvalidValue;
  }
  in->err.clear();
  if (!dout || !X || !W1 || !W2 || !topk_idx || !topk_w || !saved || !dX || !dprobs || !dW1 || !dW2 || acc < 0 ||
      acc > 2) {
    return set_error(in->err, hipErrorInvalidValue, "train_backward: every operand and saved must be set, acc in"
                                                    " [0, 2]");
  }
  if (saved_ok(in, saved) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  KosmosMoe *c = in->scratch;
  KosmosMoeWorkspace *w = in->ws;
  const KosmosMoeConfig &g = in->cfg;
  const Gl dY_gl((bf16 *)w->recv, nullptr, nullptr, (size_t)w->rows + 1, (size_t)g.H);
  const SingleCall call = {(const unsigned short *)dout, (const bf16 *)saved->g, (const bf16 *)W1, (const bf16 *)W2,
                           (bf16 *)dX, w->dtk, (bf16 *)dW1, (bf16 *)dW2, dprobs};
  const TrainApiBwdCall v = {(const bf16 *)saved->xp, (const bf16 *)saved->g, (const bf16 *)saved->a,
                             (const bf16 *)saved->y, (const bf16 *)X, (const uint8_t *)saved->route, in->ctr, in->seg,
                             in->scnt, in->sep, in->nseg, (int)in->off_slot, acc};
  if (small_k(g) && g.act) {
    moe_train_api_launch_bwd_ks<GluClamp10>(g.wgcap, s, dY_gl, w->t, c->b.args, call, v);
  } else if (small_k(g)) {
    moe_train_api_launch_bwd_ks<SwiGLU>(g.wgcap, s, dY_gl, w->t, c->b.args, call, v);
  } else if (g.act) {
    moe_train_api_launch_bwd<GluClamp10>(g.wgcap, s, dY_gl, w->t, c->b.args, call, v);
  } else {
    moe_train_api_launch_bwd<SwiGLU>(g.wgcap, s, dY_gl, w->t, c->b.args, call, v);
  }
  const hipError_t e = hipGetLastError();
  return e == hipSuccess ? e : set_error(in->err, e, "train_backward launch: %s", hipGetErrorString(e));
}

hipError_t kosmos_moe_instance_status(KosmosMoeInstance *in, int *status) {
  using namespace kosmos_moe_impl;
  if (!in || !in->scratch || !status) {
    return hipErrorInvalidValue;
  }
  in->err.clear();
  KosmosMoe *c = in->scratch;
  c->err.clear();
  MoePlanRec r;
  int st = 0;
  KM_TRY(hipDeviceSynchronize());
  KM_TRY(hipMemcpy(&r, c->p.rec, sizeof r, hipMemcpyDeviceToHost));
  KosmosMoeInfo &info = c->info;
  info.Mpad = r.Mpad, info.Mpad_max = r.Mpad_max, info.nt1 = r.nt1, info.nt2 = r.nt2;
  KM_TRY(hipMemcpy(&st, c->p.status, sizeof(int), hipMemcpyDeviceToHost));
  KM_TRY(hipMemset(c->p.status, 0, sizeof(int)));
  *status = st;
  return hipSuccess;
}

#undef KM_TRY
