/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE library (moe.h): the shared workspace, per-layer arenas and tables, the host-array plan, the launches.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <utility>

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

static bool allgather(KosmosMoe *c, const void *send, void *recv, size_t bytes) {
  return c->cfg.coll.allgather(c->cfg.coll.ctx, send, recv, bytes) == 0;
}

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
  return allgather(c, &v, out.data(), sizeof(T));
}

static hipError_t record(const hipEvent_t *ev, int i, hipStream_t s) {
  return ev ? hipEventRecord(ev[i], s) : hipSuccess;
}

// Training keeps every context's saved state in the workspace's pool, rows of 2H bytes: bf16 Xp, G and A (pool_a_row),
// MXFP8 G, the column-wise A and with I < 1024 the column-wise Xp (pool_mx_rows).
static bool pool_on(const KosmosMoeConfig &g) {
  return g.backward;
}

// Column-wise Xp goes out in the backward's idle link time after dY, unless I < 1024 leaves it link-bound.
static bool mx_xcol_in_forward(const KosmosMoeConfig &g) {
  return g.I < 1024;
}

// A call's head (its whole block) at rows rows.
static int pool_head_rows(const KosmosMoeConfig &g, int rows) {
  if (g.mxfp8) {
    return pool_mx_rows(rows + BLOCK_ROW, g.H, g.I, mx_xcol_in_forward(g));
  }
  return pool_end(0, rows + BLOCK_ROW, pool_q(2 * g.I, g.H), pool_q(g.I, g.H));
}

// The pool rows a storing call must keep for the next one not to run in rounds: bf16 its full-capacity Xp (reserved
// for each layer still to store), MXFP8 its whole block (no reserve).
static int pool_xr(const KosmosMoeWorkspace *w) {
  return w->cfg.mxfp8 ? pool_head_rows(w->cfg, w->rows) : w->rows + BLOCK_ROW;
}

static int pool_head(const KosmosMoeWorkspace *w) {
  return pool_head_rows(w->cfg, w->rows);
}

struct ArenaLayout {
  size_t recv, off_sig, off_comb, comb, off_csig, off_cnt, off_flag, off_slot, off_xc, off_xsh, off_xrecv, off_xsig, total;
};

// Training keeps the forward's receive arena in the workspace (MXFP8: its backward reads the column-wise Xp instead) or
// its pool (bf16); MXFP8 the column-wise Xp in the workspace when the backward writes it, else in the pool. frecv: the
// forward arena has its receive rows anyway (an instance's, whose forward copies Xp out).
static ArenaLayout arena_layout(const KosmosMoeConfig &g, int rows, bool bwd, bool frecv = false) {
  ArenaLayout l;
  const int sigb = (rows + BLOCK_ROW - 1) / BLOCK_ROW;
  const int nblk = (g.T + COMB_BLOCK - 1) / COMB_BLOCK;
  const bool mxt = g.mxfp8 && g.backward;
  // The backward's signals end with each source's RFREE (moe_plan.cuh, single_round_end; MXFP8: then the column-wise
  // Xp words), the forward's with RFREE.
  const size_t sig = (size_t)(bwd ? (mxt ? hk_comm::xcol_index(0, g.E_loc, sigb)
                                         : hk_comm::bfree_index(2 * MOE_EP, sigb))
                                  : RFREE_OFF + MOE_EP) * 4;
  l.recv = bwd || !g.backward || frecv ? (size_t)(rows + 1) * g.H * 2 : 0;
  l.off_sig = align_up(l.recv, 4096);
  l.off_comb = align_up(l.off_sig + sig, 4096);
  l.comb = (size_t)g.T * g.k * g.H * 2;
  l.off_csig = align_up(l.off_comb + l.comb, 4096);
  l.total = l.off_csig + (size_t)MOE_EP * (nblk + (bwd ? 0 : g.E_loc)) * 4;
  l.off_cnt = l.off_flag = l.off_slot = l.off_xc = l.off_xsh = l.off_xrecv = l.off_xsig = 0;
  if (!bwd) {
    l.off_cnt  = align_up(l.total, 4096);
    l.off_flag = align_up(l.off_cnt + (size_t)2 * MOE_EP * MOE_EP * g.E_loc * 4, 4096);
    l.off_slot = align_up(l.off_flag + (size_t)MOE_EP * 4, 4096);
    l.total    = l.off_slot + (size_t)rows * 4;
  } else if (g.backward) {
    l.off_xrecv = align_up(l.total, 4096);
    l.total     = l.off_xrecv + (size_t)(rows + 1) * g.H * 2;
    l.off_xsig  = align_up(l.total, 4096);
    l.total     = l.off_xsig + (size_t)MOE_EP * sigb * 4;
  }
  if (mxt && bwd && !mx_xcol_in_forward(g)) {
    l.off_xc = align_up(l.total, 4096);
    l.off_xsh = align_up(l.off_xc + (size_t)rows * g.H + (size_t)(rows / 32 + 1) * g.H, 4096);
    l.total   = l.off_xsh + 4096 + (size_t)g.E_loc * MOE_EP * 2 * 32 * g.H * 2;
  }
  return l;
}

// MXFP8 training with the column-wise Xp written by the forward pads each (source, expert) segment to 32 rows.
static bool mx_pad32(const KosmosMoeConfig &g) {
  return g.mxfp8 && g.backward && mx_xcol_in_forward(g);
}

static size_t g_bytes(const KosmosMoeConfig &g, int rows) {
  return pool_on(g) ? 0 : (size_t)(rows + 1) * 2 * g.I * 2;
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

static hipError_t validate_router(const KosmosMoeConfig *g, std::string *why) {
  const KosmosMoeRouterConfig &r = g->router;
  const int E = MOE_EP * g->E_loc, k = g->k;
  if (!r.on) {
    return hipSuccess;
  }
  if (g->T % BLOCK_ROW || E > RG_EPL * 64 || k > RG_KMAX) {
    return refuse(why, "router: T must be a multiple of 256 and 8 * E_loc <= 512");
  }
  if (r.score < KOSMOS_ROUTER_SOFTMAX || r.score > KOSMOS_ROUTER_SOFTMAX_PRE) {
    return refuse(why, "router: score must be one of the KOSMOS_ROUTER_* score functions");
  }
  if (r.groups && (r.groups < 0 || r.groups > 64 || E % r.groups || r.group_topk < 1 || r.group_topk > r.groups ||
                   k % r.group_topk || k / r.group_topk > RG_TOPM || k / r.group_topk > E / r.groups ||
                   k > r.group_topk * (E / r.groups))) {
    return refuse(why, "router: groups must divide 8 * E_loc (<= 64), group_topk in [1, groups] divide k,"
                       " k / group_topk <= 4");
  }
  if (r.aux_groups < 0 || r.aux_groups > RG_NB || (r.aux_groups && g->T % r.aux_groups)) {
    return refuse(why, "router: aux_groups must be in [0, 4] and divide T");
  }
  return hipSuccess;
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
  if (E > SIG_STRIDE) {
    return refuse(why, "E_loc must be <= 64");
  }
  if (g->wgcap < 9 || g->wgcap > GRID_CAP) {
    return refuse(why, "wgcap must be in [9, 256]");
  }
  if (g->capacity_rows < 0) {
    return refuse(why, "capacity_rows must be >= 0");
  }
  if (g->comm_offset % 4096) {
    return refuse(why, "comm_offset must be a multiple of 4096");
  }
  if (pool_on(*g) && g->pool_bytes / (2 * (size_t)H) >= (1ull << 30)) {
    return refuse(why, "training: pool_bytes must be below 2^30 rows of 2H bytes");
  }
  if (g->backward && E > MOE_BWD_MAX_EXPERTS) {
    return refuse(why, "E_loc exceeds the backward kernel's expert tables");
  }
  if (g->shared && (g->T % (2 * BLOCK_ROW))) {
    return refuse(why, "shared expert: T must be a multiple of 512");
  }
  if (g->mxfp8 && ((g->router.on && !g->backward) || H % MX_QCOLS || (g->backward && g->T % BLOCK_ROW))) {
    return refuse(why, "mxfp8: needs H a multiple of 1024 and (training) T a multiple of 256; the fused router only"
                       " with training");
  }
  if (g->fp8_combine && !(g->mxfp8 && g->backward)) {
    return refuse(why, "fp8_combine: needs mxfp8 and backward (MXFP8 training)");
  }
  if (g->act != KOSMOS_MOE_ACT_SWIGLU && g->act != KOSMOS_MOE_ACT_SWIGLU_CLAMP10) {
    return refuse(why, "act: KOSMOS_MOE_ACT_SWIGLU or KOSMOS_MOE_ACT_SWIGLU_CLAMP10");
  }
  if (g->probs_in && (k != 1 || g->router.on)) {
    return refuse(why, "probs_in: needs k = 1 and no fused router");
  }
  return validate_router(g, why);
}

static void free_tables(KosmosMoe *c) {
  KosmosMoeFwd &f = c->f;
  if (c->cfg.backward) {
    f.act = nullptr;
  }
  dfree(f.cnt), dfree(f.roff), dfree(f.gather), dfree(f.gid), dfree(f.startby), dfree(f.cntby);
  dfree(f.sendslot), dfree(f.lbsrc), dfree(f.owed), dfree(f.rbhi), dfree(f.tiledone);
  dfree(f.t1), dfree(f.t2), dfree(f.blkctr), dfree(f.sbase);
  dfree(f.rbs), dfree(f.q1), dfree(f.q2);
  dfree(f.act), dfree(f.wt), dfree(f.adone), dfree(f.args);
  dfree(f.sact), dfree(f.sctr), dfree(f.sactc), dfree(f.sactcs), dfree(f.ep);
  KosmosMoeBwd &b = c->b;
  dfree(b.args), dfree(b.blks), dfree(b.cblks), dfree(b.cntbysrc), dfree(b.startbysrc), dfree(b.sendw);
  KosmosMoePlan &p = c->p;
  dfree(p.hist), dfree(p.pk), dfree(p.status), dfree(p.idx), dfree(p.w), dfree(p.rec);
  KosmosMoeRouter &r = c->r;
  dfree(r.logits), dfree(r.cnt), dfree(r.rbdone), dfree(r.seedv);
}

// The router's per-layer buffers (logits, aux counts and draw seed for the backward; the forward's counters).
static hipError_t alloc_router(KosmosMoe *c) {
  const KosmosMoeConfig &g = c->cfg;
  const KosmosMoeRouterConfig &rc = g.router;
  const int T = g.T, E = MOE_EP * g.E_loc, nb = rc.aux_groups, Tg = nb ? T / nb : 1;
  KosmosMoeRouter &r = c->r;
  KM_TRY(dalloc(&r.logits, (size_t)T * ((E + 63) / 64 * 64)));
  KM_TRY(dzero(&r.cnt, (size_t)std::max(nb, 1) * E));
  KM_TRY(dzero(&r.rbdone, (size_t)T / RG_ROWS + 1));
  KM_TRY(dzero(&r.seedv, 1));
  r.aux_c = nb ? (float)((double)E * rc.aux_coeff / ((double)g.k * Tg * Tg * nb)) : 0.0f;
  return hipSuccess;
}

// A context's router arguments: its own buffers and routing tables (the workspace's are added by set_shared).
static void router_args(const KosmosMoe *c, RouterArgs *ra) {
  const KosmosMoeConfig &g = c->cfg;
  const KosmosMoeRouterConfig &rc = g.router;
  const KosmosMoeRouter &r = c->r;
  const int T = g.T, E = MOE_EP * g.E_loc;
  memset(ra, 0, sizeof *ra);
  ra->logits = r.logits, ra->w = c->p.w, ra->idx = c->p.idx, ra->cnt = r.cnt, ra->seedv = r.seedv;
  ra->rbdone = r.rbdone, ra->routed = r.rbdone + T / RG_ROWS;
  ra->T = T, ra->E = E, ra->El = (E + 63) / 64 * 64, ra->Ec = (E + RG_COLS - 1) / RG_COLS * RG_COLS;
  ra->Ed = (E + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW;
  ra->k = g.k, ra->groups = rc.groups, ra->gtopk = rc.group_topk, ra->nb = rc.aux_groups;
  ra->force = rc.force, ra->rank = g.coll.rank;
  ra->scale = rc.scale != 0.0f ? rc.scale : 1.0f;
  ra->aux_c = r.aux_c;
}

// The router's per-call scratch, once per workspace: the forward's aux partials, the backward's dlogits, router dX and
// counters.
static hipError_t ws_alloc_router(KosmosMoe *c) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const int T = g.T, E = MOE_EP * g.E_loc, nb = std::max(g.router.aux_groups, 1);
  KosmosMoeWsRouter &r = w->r;
  KM_TRY(dalloc(&r.psum, (size_t)T / RG_ROWS * nb * E));
  KM_TRY(dalloc(&r.pcnt, (size_t)T / RG_ROWS * nb * E));
  if (g.backward) {
    std::vector<int> id((size_t)T);
    for (int t = 0; t < T; t++) {
      id[(size_t)t] = t;
    }
    KM_TRY(dupload(&r.rid, id));
    KM_TRY(dzero(&r.bctr, RC_N));
    KM_TRY(dzero(&r.rpdone, (size_t)T / BLOCK_ROW));
    KM_TRY(dzero(&r.rdgdone, (size_t)T / BLOCK_ROW));
    KM_TRY(dalloc(&r.dl, (size_t)T * ((E + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW)));
    KM_TRY(dalloc(&r.dlbp, (size_t)T / RG_ROWS * NUM_WARPS * E));
    KM_TRY(dalloc(&r.dxr, (size_t)T * g.H));
    if (g.mxfp8) {
      KM_TRY(dalloc(&r.dlt, (size_t)T * ((E + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW)));
    }
  }
  return hipSuccess;
}

static void ws_free_router(KosmosMoeWorkspace *w) {
  KosmosMoeWsRouter &r = w->r;
  dfree(r.psum), dfree(r.pcnt), dfree(r.rid), dfree(r.bctr), dfree(r.rpdone), dfree(r.rdgdone);
  dfree(r.dl), dfree(r.dxr), dfree(r.dlbp), dfree(r.dlt);
}

static void set_router_shared(const KosmosMoeWorkspace *w, RouterArgs *ra) {
  const KosmosMoeWsRouter &r = w->r;
  ra->psum = r.psum, ra->pcnt = r.pcnt, ra->dtk = w->dtk, ra->bctr = r.bctr, ra->rpdone = r.rpdone;
  ra->rdgdone = r.rdgdone, ra->rid = r.rid, ra->dl = r.dl, ra->dxr = r.dxr, ra->dlbp = r.dlbp;
}

static hipError_t close_arenas(KosmosMoe *c) {
  hipError_t e = hipSuccess;
  free_tables(c);
  if (c->comm) {
    e = kosmos_comm_destroy(c->comm);
    c->comm = nullptr;
  }
  c->f.arena = KosmosMoeArena{};
  c->arena_rows = 0;
  c->planned = 0;
  c->forwarded = 0;
  return e;
}

// The workspace's buffers in a context's device arguments (bind_ws has pointed the context's MXFP8 regions at them).
static void set_shared(const KosmosMoe *c, MoeFwdArgs *fa, SingleArgs *ba) {
  const KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeFwd &f = c->f;
  fa->c2 = w->y;
  fa->sy = w->sbuf;
  fa->wep = w->wep;
  fa->xq = w->mxq, fa->xq_done = w->qfx;
  if (ba && w->sctr) {
    const int nrs = w->cfg.T / BLOCK_ROW;
    ba->sh.dg = w->sdg, ba->sh.dx = w->sbuf, ba->sh.part = w->spart, ba->sh.dsp = w->sdsp;
    ba->sh.dg_done = w->sctr, ba->sh.dx_done = w->sctr + nrs, ba->sh.dx_all = w->sctr + 2 * nrs;
    ba->sh.fctr = w->sctr + 2 * nrs + 1, ba->sh.wdone = (int *)(w->spart + (size_t)w->cfg.H * w->cfg.I * 5);
    ba->sdoq = w->sdoq, ba->sdoc = w->sdoc, ba->sdocs = w->sdocs, ba->sxc = w->sxc, ba->sxcs = w->sxcs;
    ba->sdgc = w->sdgc, ba->sdgcs = w->sdgcs, ba->sq_done = w->sctr + 2 * nrs + 2;
  }
  set_router_shared(w, &fa->r);
  const bool mxt = w->cfg.mxfp8 && w->cfg.backward;
  fa->pool = w->pool, fa->pslot = c->pslot, fa->prows = w->prows;
  fa->phead = w->pool ? pool_head(w) : 0, fa->pres = w->rows ? pool_xr(w) : 0;
  fa->pgq = pool_q(2 * w->cfg.I, w->cfg.H), fa->paq = pool_q(w->cfg.I, w->cfg.H), fa->pascr = w->ascr, fa->pxre = w->xrecv;
  fa->pbase = mxt ? (uint8_t *)w->xpool : nullptr;
  if (mxt) {
    fa->sact = w->sact;
    for (int i = 0; i < MOE_EP; i++) {
      fa->xcp[i] = mx_xcol_in_forward(w->cfg) ? (uint8_t *)w->xpoolp[i] : nullptr;
    }
  }
  if (ba) {
    set_router_shared(w, &ba->r);
    ba->pool = w->pool;
    ba->pslot = c->pslot, ba->prows = w->prows, ba->phead = fa->phead, ba->pres = fa->pres;
    ba->pbase = fa->pbase, ba->phr = w->rows + BLOCK_ROW;
    ba->ascr = w->ascr, ba->rg_done = w->rgdone, ba->fa = c->inst_fa ? c->inst_fa : c->f.args, ba->rinfo = w->rinfo;
    ba->xq8 = w->xq8, ba->qx_done = w->qx;
    for (int i = 0; i < MOE_EP; i++) {
      ba->xrp[i] = w->xrecvp[i], ba->xsg[i] = w->xsigp[i];
    }
    ba->wep = w->wep;
    ba->nn1.a = w->dg, ba->nn1.out = w->y, ba->nn1.pub = w->tdone;
    ba->nnr = NnOps{w->r.dl, w->r.rid, w->r.dxr, w->r.rdgdone, ba->r.T, ba->r.Ed, ba->r.E};
    ba->rdlt = w->r.dlt;
    NnArrivalGate &hg = ba->a.g;
    hg.sig = w->sig, hg.push.peers = w->dpeers, hg.push.sigb = w->sigb;
    ba->b.s.dtk = w->dtk;
    hk_comm::CCombine &hb = ba->b.c;
    hb.Xp = (const unsigned short *)w->y, hb.tile_done = w->tdone, hb.peers = w->cpeers;
    ba->dg = w->dg, ba->dxp = w->y, ba->dg_done = w->dgdone;
    ba->dyq = w->mxq, ba->dyc = w->dyc, ba->dycs = w->dycs, ba->dgc = w->dgc, ba->dgcs = w->dgcs;
    ba->qdy_done = w->qdy;
  }
  if (w->pool || w->xrecv) {
    for (int i = 0; i < MOE_EP; i++) {
      fa->dpeers.recv[i] = f.dpeers.recv[i];
    }
    fa->act = f.act, fa->p.zf = f.recv;
    if (ba && (!w->pool || mxt)) {
      ba->a.w.act = f.act, ba->b.w.act = f.recv;
    }
  }
  if (ba && w->xc) {
    ba->xpc = f.xc, ba->xpcs = f.xc + f.xcs_off;
    for (int i = 0; i < MOE_EP; i++) {
      ba->xcp[i] = f.xcp[i], ba->xcsp[i] = f.xcp[i] + f.xcs_off, ba->xshp[i] = f.xshp[i];
    }
  }
}

// MXFP8 training: the context's forward receive arena, row-wise A, the shared expert's row-wise A and (I >= 1024)
// column-wise Xp are the workspace's (its G, column-wise A and, I < 1024, column-wise Xp the pool's); bf16 training:
// its receive arena and A are the pool's (each call at the pool's next base).
static void bind_ws(KosmosMoe *c) {
  const KosmosMoeWorkspace *w = c->ws;
  KosmosMoeFwd &f = c->f;
  if (w->inst) {
    // An instance: Xp lands in the context's own arena; A goes to ascr unless the call keeps it (its caller's).
    f.act = w->ascr;
    return;
  }
  if (w->pool && !w->cfg.mxfp8) {
    for (int i = 0; i < MOE_EP; i++) {
      f.dpeers.recv[i] = w->xpoolp[i];
    }
    f.recv = w->xpool, f.act = w->xpool;
  } else if (w->xrecv) {
    for (int i = 0; i < MOE_EP; i++) {
      f.dpeers.recv[i] = w->xrecvp[i];
    }
    f.recv = w->xrecv, f.act = w->act;
  }
  if (w->xc) {
    for (int i = 0; i < MOE_EP; i++) {
      f.xcp[i] = w->xcp[i], f.xshp[i] = w->xshp[i];
    }
    f.xc = w->xc, f.xcs_off = w->xcs_off;
  }
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
  const int nrbs = g.shared ? g.T / BLOCK_ROW : 0;
  KM_TRY(dzero(&f.q1, (size_t)(nrbm + nrbs) * (N1 / BLOCK_COL)));
  KM_TRY(dzero(&f.q2, (size_t)(nrbm + nrbs) * (H / BLOCK_COL) + (size_t)nrbs * (N1 / BLOCK_COL)));
  if (!g.backward) {
    KM_TRY(dalloc(&f.act, (size_t)nrow * I));
  }
  if (nrbs) {
    KM_TRY(dzero(&f.sctr, 1 + nrbs));
    if (g.mxfp8 && g.backward) {
      KM_TRY(dalloc(&f.sactc, (size_t)g.T * I));
      KM_TRY(dalloc(&f.sactcs, (size_t)g.T / 32 * I));
    } else {
      KM_TRY(dalloc(&f.sact, (size_t)g.T * I));
    }
  }
  KM_TRY(dzero(&f.wt, TK));
  KM_TRY(dzero(&f.ep, 2));
  KM_TRY(dalloc(&f.args, 1));
  KM_TRY(dzero(&p.hist, (size_t)nch * nu));
  KM_TRY(dzero(&p.pk, TK));
  KM_TRY(dzero(&p.status, 1));
  KM_TRY(dzero(&p.idx, TK));
  KM_TRY(dzero(&p.w, TK));
  KM_TRY(dzero(&p.rec, 1));
  if (g.router.on) {
    KM_TRY(alloc_router(c));
  }
  SingleArgs hb;
  memset(&hb, 0, sizeof hb);
  if (g.backward) {
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
    hb.sactc = f.sactc, hb.sactcs = f.sactcs;
    hb.xpc = f.xc, hb.xpcs = f.xc ? f.xc + f.xcs_off : nullptr, hb.xc_fwd = mx_xcol_in_forward(g);
    for (int i = 0; i < MOE_EP; i++) {
      hb.xcp[i] = f.xcp[i], hb.xcsp[i] = f.xcp[i] ? f.xcp[i] + f.xcs_off : nullptr, hb.xshp[i] = f.xshp[i];
    }
    hb.H = H, hb.I = I, hb.E = E, hb.ep = f.ep;
    hb.nn1 = NnOps{nullptr, f.gid, nullptr, nullptr, 0, N1, N1};
    if (g.router.on) {
      router_args(c, &hb.r);
    }
    KM_TRY(dalloc(&b.args, 1));
  }
  MoeFwdArgs ha;
  memset(&ha, 0, sizeof ha);
  ha.send_slot = f.sendslot, ha.dpeers = f.dpeers, ha.my_sig = f.sig;
  ha.cnt_by_src = f.cntby, ha.start_by_src = f.startby, ha.row_off = f.roff, ha.q1 = f.q1, ha.q2 = f.q2;
  ha.gather = f.gather, ha.gid = f.gid, ha.act = f.act, ha.a_done = f.adone;
  ha.tile_done = f.tiledone, ha.g2_ctr = f.t2, ha.disp_done = f.t2 + 1, ha.fold_ctr = f.blkctr;
  ha.idx_done = f.t2 + 2, ha.cnt = f.cnt, ha.E_loc = E;
  ha.sactc = f.sactc, ha.sactcs = f.sactcs;
  ha.mxt = g.mxfp8 && g.backward;
  // Only MXFP8 training pads segments (plan_windows and plan_clip read xc_fwd alone).
  ha.xc_fwd = mx_pad32(g);
  for (int i = 0; i < MOE_EP; i++) {
    ha.xcp[i] = f.xcp[i], ha.xcsp[i] = f.xcp[i] ? f.xcp[i] + f.xcs_off : nullptr, ha.xshp[i] = f.xshp[i];
  }
  ha.rbs = f.rbs, ha.lb_src = f.lbsrc, ha.rb_hi = f.rbhi, ha.lpeers = f.lpeers, ha.wts = f.wt, ha.owed = f.owed;
  for (int i = 0; i < MOE_EP; i++) {
    ha.slot[i] = f.slot[i], ha.p.cnts[i] = f.cnts[i], ha.p.cflag[i] = f.cflag[i];
  }
  ha.nblks = nu, ha.nrbs = nu, ha.ep = f.ep;
  ha.my_rank = me, ha.dncomm = f.ncomm, ha.cncomm = f.fcomm;
  ha.H = H, ha.I = I, ha.T = g.T, ha.topk = g.k;
  ha.sact = f.sact, ha.sy_done = f.sctr, ha.sa_done = f.sctr ? f.sctr + 1 : nullptr;
  hb.sh.act = f.sact, hb.sh.T = g.T;
  if (g.router.on) {
    router_args(c, &ha.r);
  }
  MoePlanTabs &hp = ha.p;
  hp.ctr = f.t2 + 3, hp.hist = p.hist, hp.pk = p.pk;
  ha.rgo = f.t2 + 3 + PC_N, ha.sbase = f.sbase;
  hp.zf = f.recv, hp.rec = (int *)p.rec, hp.status = p.status;
  hp.cap = R, hp.nch = nch;
  if (g.backward) {
    hp.send_w = b.sendw, hp.bblks = (int *)b.blks, hp.cblks = (int *)b.cblks;
    hp.cnt_by_src = b.cntbysrc, hp.start_by_src = b.startbysrc, hp.ba = b.args;
  }
  set_shared(c, &ha, g.backward ? &hb : nullptr);
  if (g.backward) {
    KM_TRY(hipMemcpy(b.args, &hb, sizeof hb, hipMemcpyHostToDevice));
  }
  KM_TRY(hipMemcpy(f.args, &ha, sizeof ha, hipMemcpyHostToDevice));
  return hipSuccess;
}

// Buffers that moved: [from[i], from[i] + bytes[i]) now at to[i] (to_bytes[i] bytes; an offset beyond: its start).
struct Moved {
  uintptr_t from[3];
  size_t bytes[3];
  char *to[3];
  size_t to_bytes[3];
};

static const bf16 *moved(const Moved &mv, const bf16 *p) {
  const uintptr_t a = (uintptr_t)p;
  for (int i = 0; i < 3; i++) {
    if (mv.to[i] && a >= mv.from[i] && a - mv.from[i] < mv.bytes[i]) {
      const size_t o = a - mv.from[i];
      return (const bf16 *)(mv.to[i] + (o < mv.to_bytes[i] ? o : 0));
    }
  }
  return p;
}

// After the workspace moved: re-points a context's device arguments, keeping what its latest forward wrote there (the
// backward's operands its pool_commit wrote, moved with their buffers).
static hipError_t rebind(KosmosMoe *c, const Moved &mv) {
  bind_ws(c);
  MoeFwdArgs ha;
  SingleArgs hb;
  KM_TRY(hipMemcpy(&ha, c->f.args, sizeof ha, hipMemcpyDeviceToHost));
  if (c->cfg.backward) {
    KM_TRY(hipMemcpy(&hb, c->b.args, sizeof hb, hipMemcpyDeviceToHost));
    hb.a.w.act = moved(mv, hb.a.w.act);
    hb.b.w.act = moved(mv, hb.b.w.act);
    hb.g = moved(mv, hb.g);
    hb.actc = (const uint8_t *)moved(mv, (const bf16 *)hb.actc);
    hb.actcs = (const uint8_t *)moved(mv, (const bf16 *)hb.actcs);
    if (c->cfg.mxfp8 && mx_xcol_in_forward(c->cfg)) {
      hb.xpc = (const uint8_t *)moved(mv, (const bf16 *)hb.xpc);
      hb.xpcs = (const uint8_t *)moved(mv, (const bf16 *)hb.xpcs);
    }
  }
  set_shared(c, &ha, c->cfg.backward ? &hb : nullptr);
  if (c->cfg.backward) {
    KM_TRY(hipMemcpy(c->b.args, &hb, sizeof hb, hipMemcpyHostToDevice));
  }
  KM_TRY(hipMemcpy(c->f.args, &ha, sizeof ha, hipMemcpyHostToDevice));
  return hipSuccess;
}

// Local; callers bracket it with agreements (no peer may be inside a call while the arena is zeroed).
static hipError_t reset_layer(KosmosMoe *c) {
  KosmosMoeFwd &f = c->f;
  KM_TRY(hipMemset(f.arena.base, 0, f.arena.bytes));
  KM_TRY(hipMemset(c->p.status, 0, sizeof(int)));
  KM_TRY(hipMemset(f.ep, 0, 2 * sizeof(unsigned)));
  if (c->cfg.router.on) {
    KM_TRY(hipMemset(c->r.rbdone, 0, ((size_t)c->cfg.T / RG_ROWS + 1) * sizeof(int)));
  }
  KM_TRY(hipDeviceSynchronize());
  c->forwarded = 0;
  return hipSuccess;
}

// As reset_layer, for the context and its workspace.
static hipError_t reset_state(KosmosMoe *c) {
  KosmosMoeWorkspace *w = c->ws;
  if (w->arena.base) {
    KM_TRY(hipMemset(w->arena.base, 0, w->arena.bytes));
  }
  if (w->r.bctr) {
    KM_TRY(hipMemset(w->r.bctr, 0, RC_N * sizeof(int)));
    KM_TRY(hipMemset(w->r.rpdone, 0, (size_t)w->cfg.T / BLOCK_ROW * sizeof(int)));
    KM_TRY(hipMemset(w->r.rdgdone, 0, (size_t)w->cfg.T / BLOCK_ROW * sizeof(int)));
  }
  if (w->qdy) {
    const size_t nrbm = (size_t)w->rows / BLOCK_ROW;
    KM_TRY(hipMemset(w->qfx, 0, sizeof(int)));
    KM_TRY(hipMemset(w->qdy, 0, nrbm * (w->cfg.H / MX_QCOLS) * sizeof(int)));
    KM_TRY(hipMemset(w->qx, 0, nrbm * (w->cfg.H / MX_QCOLS) * sizeof(int)));
    KM_TRY(hipMemset(w->rgdone, 0, (nrbm + 1) * sizeof(int)));
    KM_TRY(hipMemset(w->rinfo, 0, (RINFO_WM + MOE_BWD_MAX_EXPERTS) * sizeof(int)));
  }
  KM_TRY(hipMemset(w->wep, 0, 2 * sizeof(unsigned)));
  return reset_layer(c);
}

static hipError_t ws_close(KosmosMoeWorkspace *w) {
  hipError_t e = hipSuccess;
  dfree(w->t), dfree(w->tdone), dfree(w->dgdone), dfree(w->y), dfree(w->dg), dfree(w->dtk);
  dfree(w->sbuf), dfree(w->sdg), dfree(w->spart), dfree(w->sctr), dfree(w->sdsp);
  dfree(w->mxq), dfree(w->dyc), dfree(w->dycs), dfree(w->dgc), dfree(w->dgcs);
  dfree(w->qdy), dfree(w->qfx), dfree(w->act), dfree(w->wep), dfree(w->ascr), dfree(w->rgdone), dfree(w->rinfo);
  dfree(w->xq8), dfree(w->qx);
  dfree(w->sdoq), dfree(w->sdoc), dfree(w->sdocs), dfree(w->sxc), dfree(w->sxcs), dfree(w->sdgc), dfree(w->sdgcs);
  dfree(w->sact);
  ws_free_router(w);
  if (w->own_comm && w->comm) {
    e = kosmos_comm_destroy(w->comm);
    w->comm = nullptr;
    w->own_comm = 0;
  }
  w->arena = KosmosMoeArena{};
  w->recv = nullptr, w->sig = nullptr, w->xrecv = nullptr, w->xc = nullptr;
  for (int i = 0; i < MOE_EP; i++) {
    w->xrecvp[i] = nullptr, w->xsigp[i] = nullptr, w->xcp[i] = nullptr, w->xshp[i] = nullptr;
  }
  w->rows = 0;
  return e;
}

static void pool_close(KosmosMoeWorkspace *w) {
  dfree(w->pool);
  if (w->pcomm) {
    (void)kosmos_comm_destroy(w->pcomm);
    w->pcomm = nullptr;
  }
  w->xpool = nullptr;
  for (int i = 0; i < MOE_EP; i++) {
    w->xpoolp[i] = nullptr;
  }
  w->prows = 0;
}

// Collective: the pool at the ranks' least cfg.pool_bytes (every rank's pool decides alike), at least one call's block,
// empty; on any rank's failure none is open.
static hipError_t pool_open(KosmosMoe *c) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  std::vector<size_t> pb;
  if (!gather_all(c, g.pool_bytes, pb)) {
    return set_error(c->err, hipErrorUnknown, "pool size all-gather failed");
  }
  w->prows = std::max((int)(*std::min_element(pb.begin(), pb.end()) / (2 * (size_t)g.H)), pool_head(w));
  const size_t xb = (size_t)w->prows * g.H * 2;
  hipError_t e = kosmos_comm_create(&w->pcomm, &g.coll, xb);
  if (e != hipSuccess) {
    w->pcomm = nullptr;
    pool_close(w);
    return set_error(c->err, e, "pool comm (%zu bytes): %s", xb, hipGetErrorString(e));
  }
  KosmosCommInfo ci;
  e = kosmos_comm_info(w->pcomm, &ci);
  if (e == hipSuccess) {
    for (int i = 0; i < MOE_EP; i++) {
      w->xpoolp[i] = (bf16 *)ci.peers[i];
    }
    w->xpool = (bf16 *)ci.local;
    e = dzero(&w->pool, POOL_FIT + 3);
  }
  if (!agree(c, e == hipSuccess)) {
    if (e != hipSuccess) {
      set_error(c->err, e, "pool of %d rows: %s", w->prows, hipGetErrorString(e));
    }
    pool_close(w);
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  return hipSuccess;
}

// The pool's next base with head free rows, from its device state st (as pool_next), or -1.
static int pool_next_host(const std::vector<int> &st, int prows, int head) {
  const int n = st[1];
  int best = -1;
  for (int t = 0; t <= n; t++) {
    const int c = t == n ? 0 : st[4 + 3 * (size_t)t];
    bool ok = (t == n || c > 0) && c + head <= prows;
    for (int j = 0; j < n && ok; j++) {
      ok = st[4 + 3 * (size_t)j] == 0 || st[2 + 3 * (size_t)j] >= c + head || st[4 + 3 * (size_t)j] <= c;
    }
    best = ok && (best < 0 || c < best) ? c : best;
  }
  return best;
}

// With the device drained: the slots in use and the next base for the workspace's rows, into the pool's state.
static hipError_t pool_fit(KosmosMoe *c) {
  KosmosMoeWorkspace *w = c->ws;
  std::vector<int> st(POOL_FIT);
  KM_TRY(hipMemcpy(st.data(), w->pool, st.size() * sizeof(int), hipMemcpyDeviceToHost));
  st[1] = 0;
  for (const KosmosMoe *x : w->ctxs) {
    st[1] = std::max(st[1], x->pslot + 1);
  }
  const int head = pool_head(w), xr = pool_xr(w);
  st[0] = pool_next_host(st, w->prows, head);
  if (st[0] < 0) {
    return set_error(c->err, hipErrorOutOfMemory, "the pool's %d rows hold no call's block of %d rows besides the"
                     " stored ones", w->prows, head);
  }
  const int *rec = st.data() + 2, n = st[1], nb = st[0];
  int fit = 0;
  for (int t = 0; t <= n + 1; t++) {
    fit |= pool_keeps_xp(rec, n, -1, w->prows, head, nb, nb + xr, t, pool_cand(rec, n, -1, t, nb + xr, 0));
  }
  KM_TRY(hipMemcpy(w->pool, st.data(), 2 * sizeof(int), hipMemcpyHostToDevice));
  const int fw[3] = {fit, fit, -1};
  KM_TRY(hipMemcpy(w->pool + POOL_FIT, fw, sizeof fw, hipMemcpyHostToDevice));
  return hipSuccess;
}

static hipError_t ws_alloc(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const int nrbm = rows / BLOCK_ROW;
  KM_TRY(dzero(&w->wep, 2));
  KM_TRY(dalloc(&w->y, (size_t)(rows + 1) * g.H));
  if (g.backward) {
    KM_TRY(dzero(&w->t, 16));
    KM_TRY(dzero(&w->tdone, nrbm));
    KM_TRY(dzero(&w->dgdone, nrbm));
    KM_TRY(dalloc(&w->dg, (size_t)(rows + 1) * 2 * g.I));
    KM_TRY(dalloc(&w->dtk, (size_t)g.T * g.k));
    if (!g.mxfp8) {
      KM_TRY(dalloc(&w->ascr, (size_t)(rows + 1) * g.I));
    }
    KM_TRY(dzero(&w->rgdone, nrbm + 1));
    KM_TRY(dzero(&w->rinfo, RINFO_WM + MOE_BWD_MAX_EXPERTS));
    KM_TRY(hipMemset(w->arena.base, 0, w->arena.bytes));
    if (g.mxfp8) {
      const size_t r1 = (size_t)rows + 1, rs = (size_t)rows / 32 + 1;
      KM_TRY(dalloc(&w->mxq, std::max(r1, (size_t)g.T * (g.probs_in && g.shared ? 2 : 1)) * 2 * g.H));
      KM_TRY(dalloc(&w->dyc, r1 * g.H));
      KM_TRY(dalloc(&w->dycs, rs * g.H));
      KM_TRY(dalloc(&w->dgc, r1 * 2 * g.I));
      KM_TRY(dalloc(&w->dgcs, rs * 2 * g.I));
      KM_TRY(dzero(&w->qdy, (size_t)nrbm * (g.H / MX_QCOLS)));
      KM_TRY(dzero(&w->qfx, 1));
      KM_TRY(dalloc(&w->act, r1 * g.I));
      KM_TRY(dalloc(&w->xq8, r1 * 2 * g.H));
      KM_TRY(dzero(&w->qx, (size_t)nrbm * (g.H / MX_QCOLS)));
    }
  }
  if (g.router.on) {
    KM_TRY(ws_alloc_router(c));
  }
  if (g.shared) {
    // Ys (forward) and dXs (backward) share one buffer, as Y and dXp do.
    KM_TRY(dalloc(&w->sbuf, (size_t)g.T * g.H));
    if (g.backward) {
      const int nrs = g.T / BLOCK_ROW, nwt = (g.H / BLOCK_COL) * (g.I / BLOCK_COL) * 3;
      KM_TRY(dalloc(&w->sdg, (size_t)g.T * 2 * g.I));
      // The fp32 partials (fc2 [H][I], fc1 slices [2I][H]), then their tile counters.
      KM_TRY(dzero(&w->spart, (size_t)g.H * g.I * (1 + 2 * SH_KSPLIT) + nwt));
      KM_TRY(dzero(&w->sctr, 2 * nrs + 2 + (g.mxfp8 ? 1 : 0)));
      KM_TRY(dalloc(&w->sdsp, (size_t)g.T * (g.I / BLOCK_COL)));
      if (g.mxfp8) {
        const size_t T = (size_t)g.T, H = (size_t)g.H, N1 = (size_t)2 * g.I;
        KM_TRY(dalloc(&w->sact, T * g.I));
        KM_TRY(dalloc(&w->sdoq, T * 2 * H));
        KM_TRY(dalloc(&w->sdoc, T * H));
        KM_TRY(dalloc(&w->sdocs, T / 32 * H));
        KM_TRY(dalloc(&w->sxc, T * H));
        KM_TRY(dalloc(&w->sxcs, T / 32 * H));
        KM_TRY(dalloc(&w->sdgc, T * N1));
        KM_TRY(dalloc(&w->sdgcs, T / 32 * N1));
      }
    }
  }
  KM_TRY(hipDeviceSynchronize());
  return hipSuccess;
}

// Collective: maps the workspace's arena and allocates its buffers at rows rows, all zeroed.
static hipError_t ws_open(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const int me = g.coll.rank;
  if (g.backward) {
    const ArenaLayout bl = arena_layout(g, rows, true);
    const size_t off  = w->comm ? g.comm_offset : 0;
    const size_t need = off + bl.total;
    if (!w->comm) {
      const hipError_t e = kosmos_comm_create(&w->comm, &g.coll, need);
      if (e != hipSuccess) {
        w->comm = nullptr;
        return set_error(c->err, e, "workspace comm (%zu bytes): %s", need, hipGetErrorString(e));
      }
      w->own_comm = 1;
    }
    KosmosCommInfo ci;
    const hipError_t e = kosmos_comm_info(w->comm, &ci);
    if (!agree(c, e == hipSuccess && ci.bytes >= need)) {
      if (e != hipSuccess || ci.bytes < need) {
        set_error(c->err, hipErrorInvalidValue, "the comm holds %zu bytes; the workspace needs %zu at comm_offset %zu"
                  " (kosmos_moe_arena_bytes)", e == hipSuccess ? ci.bytes : 0, need, off);
      }
      ws_close(w);
      return hipErrorInvalidValue;
    }
    w->arena = KosmosMoeArena{(char *)ci.local + off, bl.total};
    w->off_sig = bl.off_sig, w->off_comb = bl.off_comb, w->off_csig = bl.off_csig;
    w->sigb = (rows + BLOCK_ROW - 1) / BLOCK_ROW;
    for (int i = 0; i < MOE_EP; i++) {
      char *p = (char *)ci.peers[i] + off;
      w->dpeers.recv[i] = (bf16 *)p;
      w->dpeers.sig[i] = (int *)(p + bl.off_sig);
      w->cpeers.comb[i] = (bf16 *)(p + bl.off_comb);
      w->cpeers.acc[i] = (int *)(p + bl.off_csig);
      w->xrecvp[i] = bl.off_xrecv ? (bf16 *)(p + bl.off_xrecv) : nullptr;
      w->xsigp[i] = bl.off_xsig ? (int *)(p + bl.off_xsig) : nullptr;
      w->xcp[i] = bl.off_xc ? (uint8_t *)(p + bl.off_xc) : nullptr;
      w->xshp[i] = bl.off_xsh ? (uint8_t *)(p + bl.off_xsh) : nullptr;
    }
    w->recv = w->dpeers.recv[me], w->sig = w->dpeers.sig[me];
    w->xrecv = w->xrecvp[me], w->xc = w->xcp[me], w->xcs_off = (size_t)rows * g.H;
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
  }
  hipError_t e = ws_alloc(c, rows);
  if (!agree(c, e == hipSuccess)) {
    ws_close(w);
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  w->rows = rows;
  if (pool_on(g) && !w->inst) {
    e = w->pool ? hipSuccess : pool_open(c);
    if (!agree(c, e == hipSuccess)) {
      ws_close(w);
      return e != hipSuccess ? e : hipErrorUnknown;
    }
    e = pool_fit(c);
    if (!agree(c, e == hipSuccess)) {
      ws_close(w);
      return e != hipSuccess ? e : hipErrorOutOfMemory;
    }
  }
  return hipSuccess;
}

// Collective: grows the workspace to at least rows rows. Every rank first drains its device (a peer's backward may
// still write into the old arena), and every other open context is then re-pointed.
static hipError_t ws_reserve(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  if (w->rows >= rows) {
    return hipSuccess;
  }
  if (!agree(c, hipDeviceSynchronize() == hipSuccess)) {
    return set_error(c->err, hipErrorUnknown, "device synchronize failed before the workspace grew");
  }
  Moved mv = {};
  const size_t r1 = (size_t)w->rows + 1;
  mv.from[0] = (uintptr_t)w->ascr, mv.bytes[0] = r1 * w->cfg.I * 2;
  mv.from[1] = (uintptr_t)w->xrecv, mv.bytes[1] = r1 * w->cfg.H * 2;
  hipError_t e = ws_close(w);
  if (e == hipSuccess) {
    e = ws_open(c, rows);
  }
  const size_t n1 = (size_t)w->rows + 1;
  mv.to[0] = (char *)w->ascr, mv.to_bytes[0] = n1 * w->cfg.I * 2;
  mv.to[1] = (char *)w->xrecv, mv.to_bytes[1] = n1 * w->cfg.H * 2;
  for (size_t i = 0; i < w->ctxs.size() && e == hipSuccess; i++) {
    KosmosMoe *x = w->ctxs[i];
    if (x != c && x->arena_rows) {
      e = rebind(x, mv);
      if (e != hipSuccess) {
        c->err = x->err;
      }
    }
  }
  if (!agree(c, e == hipSuccess)) {
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  return hipSuccess;
}

// Collective: drains every rank's device, or refuses alike on every rank inside a stream capture and (live) while a
// forward's stored block awaits its backward.
static hipError_t ws_gate(KosmosMoe *c, const char *op, bool live) {
  KosmosMoeWorkspace *w = c->ws;
  hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
  int why = hipStreamIsCapturing(nullptr, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone ? 1 : 0;
  if (!why) {
    const hipError_t e = hipDeviceSynchronize();
    why = e == hipSuccess ? 0 : e == hipErrorStreamCaptureUnsupported || e == hipErrorStreamCaptureImplicit ? 1 : 2;
  }
  if (!why && live && w->pool) {
    std::vector<int> st(POOL_FIT);
    why = hipMemcpy(st.data(), w->pool, st.size() * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess ? 2 : 0;
    for (int j = 0; !why && j < st[1]; j++) {
      why = st[3 + 3 * (size_t)j] || st[4 + 3 * (size_t)j] ? 3 : 0;
    }
  }
  std::vector<int> whys;
  if (!gather_all(c, why, whys)) {
    return set_error(c->err, hipErrorUnknown, "%s: all-gather failed", op);
  }
  for (int q = 1; q <= 3; q++) {
    const int r = (int)(std::find(whys.begin(), whys.end(), q) - whys.begin());
    if (r < MOE_EP && q == 1) {
      return set_error(c->err, hipErrorStreamCaptureUnsupported, "%s inside a stream capture (rank %d)", op, r);
    }
    if (r < MOE_EP && q == 2) {
      return set_error(c->err, hipErrorUnknown, "%s: device synchronize or pool read failed (rank %d)", op, r);
    }
    if (r < MOE_EP) {
      return set_error(c->err, hipErrorNotReady, "%s: a stored block awaits its backward (rank %d)", op, r);
    }
  }
  return hipSuccess;
}

// Collective: every rank unmaps the old pool before any rank frees it (kosmos_comm_destroy), and no rank returns before
// every rank's contexts point at the new one (the final agreement).
static hipError_t pool_resize(KosmosMoe *c, size_t bytes) {
  KosmosMoeWorkspace *w = c->ws;
  const size_t row = 2 * (size_t)w->cfg.H;
  c->err.clear();
  bytes = std::min(bytes, (((size_t)1 << 30) - 1) * row);
  const hipError_t ge = ws_gate(c, "pool resize", true);
  if (ge != hipSuccess) {
    return ge;
  }
  std::vector<size_t> pb;
  if (!gather_all(c, bytes, pb)) {
    return set_error(c->err, hipErrorUnknown, "pool resize: all-gather failed");
  }
  bytes = *std::min_element(pb.begin(), pb.end());
  for (KosmosMoe *x : w->ctxs) {
    x->forwarded = 0;
  }
  if (!w->pool || !w->rows) {
    pool_close(w);
    w->cfg.pool_bytes = bytes;
    return hipSuccess;
  }
  Moved mv = {};
  mv.from[0] = (uintptr_t)w->xpool, mv.bytes[0] = (size_t)w->prows * row;
  pool_close(w);
  hipError_t e = hipErrorOutOfMemory;
  for (size_t b = bytes; e != hipSuccess; b /= 2) {
    w->cfg.pool_bytes = b;
    e = pool_open(c);
    if (e != hipSuccess) {
      (void)hipGetLastError();
    }
    if (e != hipSuccess && b / row <= (size_t)pool_head(w)) {
      return e;
    }
  }
  mv.to[0] = (char *)w->xpool, mv.to_bytes[0] = (size_t)w->prows * row;
  e = pool_fit(c);
  for (size_t i = 0; i < w->ctxs.size() && e == hipSuccess; i++) {
    KosmosMoe *x = w->ctxs[i];
    if (x->arena_rows) {
      e = rebind(x, mv);
      if (e != hipSuccess) {
        c->err = x->err;
      }
    }
  }
  if (!agree(c, e == hipSuccess)) {
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  return hipSuccess;
}

// Collective: grows the workspace if needed, maps this context's arena at rows rows, allocates its tables and zeroes
// its state.
static hipError_t open_arenas(KosmosMoe *c, int rows) {
  const KosmosMoeConfig &g = c->cfg;
  const int me = g.coll.rank;
  KosmosMoeFwd &f = c->f;
  const bool inst = c->ws->inst;
  const ArenaLayout fl = arena_layout(g, rows, false, inst);
  f.ncomm = std::min(g.mxfp8 ? moe_mx_ncomm(g.I, g.backward && mx_xcol_in_forward(g)) : moe_prod_ncomm(g.I),
                     g.wgcap / NUM_XCDS_AFF * NUM_XCDS_AFF);
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
  hipError_t e = ws_reserve(c, rows);
  if (e != hipSuccess) {
    return e;
  }
  // An instance's arena: the workspace's comm at inst_foff (sized by kosmos_moe_instance_bytes); else its own comm.
  const size_t aoff = inst ? c->ws->inst_foff : 0;
  if (!inst) {
    e = kosmos_comm_create(&c->comm, &g.coll, fl.total);
    if (e != hipSuccess) {
      c->comm = nullptr;
      return set_error(c->err, e, "arena comm (%zu bytes): %s", fl.total, hipGetErrorString(e));
    }
  }
  KosmosCommInfo ci;
  e = kosmos_comm_info(inst ? c->ws->comm : c->comm, &ci);
  if (e == hipSuccess && inst && ci.bytes < aoff + fl.total) {
    e = set_error(c->err, hipErrorInvalidValue, "the comm holds %zu bytes; the instance needs %zu", ci.bytes,
                  aoff + fl.total);
  }
  if (!agree(c, e == hipSuccess)) {
    close_arenas(c);
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
    f.xcp[i] = fl.off_xc ? (uint8_t *)(p + fl.off_xc) : nullptr;
    f.xshp[i] = fl.off_xsh ? (uint8_t *)(p + fl.off_xsh) : nullptr;
  }
  f.xc = f.xcp[me], f.xcs_off = (size_t)rows * g.H;
  f.recv = f.dpeers.recv[me], f.comb = f.lpeers.comb[me], f.sig = f.dpeers.sig[me];
  bind_ws(c);
  c->arena_rows = rows;
  c->info.arena_rows = rows;
  e = alloc_tables(c);
  if (e == hipSuccess) {
    e = reset_layer(c);
  }
  if (!agree(c, e == hipSuccess)) {
    close_arenas(c);
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  KosmosMoeInfo &in = c->info;
  in.fwd_ncomm = f.ncomm, in.fwd_fcomm = f.fcomm, in.fwd_burst = f.burst;
  in.g_bytes = g_bytes(g, rows);
  in.bwd_grid = g.backward ? g.wgcap : 0;
  return hipSuccess;
}

template <typename U>
static void census(const std::vector<U> &units, int n, double *mean, int *mx) {
  double sum = 0.0;
  int nw = 0;
  *mx = 0;
  for (int w = 0; w < n; w++) {
    int msk = 0;
    for (size_t j = (size_t)w; j < units.size(); j += (size_t)n) {
      if (units[j].cnt > 0) {
        msk |= 1 << units[j].dst;
      }
    }
    if (!msk) {
      continue;
    }
    const int cc = __builtin_popcount(msk);
    sum += cc, *mx = std::max(*mx, cc), nw++;
  }
  *mean = nw ? sum / nw : 0.0;
}

template <typename T>
static hipError_t readback(std::vector<T> &h, const T *d, size_t n) {
  h.resize(n);
  return hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
}

static void set_mirrors(KosmosMoe *c, const MoePlanRec &r) {
  KosmosMoeFwd &f = c->f;
  f.Mpad = r.Mpad, f.nrow = r.Mpad + 1, f.nrb = r.Mpad / BLOCK_ROW, f.nt1 = r.nt1, f.nt2 = r.nt2;
  KosmosMoeInfo &in = c->info;
  in.Mpad = r.Mpad, in.Mpad_max = r.Mpad_max, in.nt1 = r.nt1, in.nt2 = r.nt2;
}

// After a synchronized forward: its plan's settings into info and the host mirrors.
static hipError_t read_plan(KosmosMoe *c) {
  MoePlanRec r;
  KM_TRY(hipMemcpy(&r, c->p.rec, sizeof r, hipMemcpyDeviceToHost));
  KM_TRY(readback(c->pool_cnt, c->f.cnt, (size_t)c->cfg.E_loc));
  set_mirrors(c, r);
  return hipSuccess;
}

// The backward's crews and wgrad run lengths, once per context from the shape: the cost model fed with the balanced
// routing's T k / (EP E_loc) rows per (source, expert).
static void bwd_shape_plan(KosmosMoe *c) {
  const KosmosMoeConfig &g = c->cfg;
  const int E = g.E_loc, H = g.H, I = g.I;
  const bool mx = g.mxfp8 && g.backward, p32 = mx_pad32(g);
  const double u     = (double)g.T * g.k / (MOE_EP * E);
  const double col   = MOE_EP * (p32 ? ceil(u / 32) * 32 : u);
  const double mpad  = E * ceil(col / EXPERT_PADDING) * EXPERT_PADDING;
  const double kk    = E * ceil(col / 64) * 64;
  const double off   = (double)g.T * g.k * (MOE_EP - 1) / MOE_EP;
  const double tflop = 2.0 * 2.0 * mpad * H * I / 1e12, gt = mx ? tflop * hk_comm::C_MX_GEMM_SCALE : tflop;
  const double cgb   = off * H * 2 / 1e9;
  const double dgb   = mx && !mx_xcol_in_forward(g) ? off * (3 * H + H / 32) / 1e9 : cgb;
  const double tiles = (double)E * (H / BLOCK_COL) * (I / BLOCK_COL);
  KosmosMoeBwd &b = c->b;
  b.dncomm = hk_comm::c_dispatch_ncomm(dgb, gt, mx);
  if (b.dncomm + 1 > g.wgcap) {
    b.dncomm = ((g.wgcap - 1) / NUM_XCDS_AFF) * NUM_XCDS_AFF;
  }
  const int cn = hk_comm::c_combine_ncomm(cgb, gt, mx);
  b.cncomm = cn > g.wgcap ? (g.wgcap / NUM_XCDS_AFF) * NUM_XCDS_AFF : cn;
  b.wg2run = hk_comm::c_wgrad2_run(tiles, g.wgcap, kk / E, mx);
  b.wg1run = hk_comm::c_wgrad2_run(2.0 * tiles, g.wgcap, kk / E, mx);
  KosmosMoeInfo &in = c->info;
  in.bwd_dpush_gb = cgb, in.bwd_lega_tflop = tflop, in.bwd_cpush_gb = cgb, in.bwd_cncomm_clamped = cn > g.wgcap;
  in.bwd_wg2_tiles = tiles, in.bwd_wg2_meank = kk / E, in.bwd_wg2run = b.wg2run, in.bwd_wg1run = b.wg1run;
  for (int q = 0; q < MOE_EP; q++) {
    in.bwd_dncomm_all[q] = b.dncomm, in.bwd_cncomm_all[q] = b.cncomm;
  }
}

struct HostUnit {
  int dst, cnt;
};

// The most rows a rank receives (Mpad_max) for a call's count matrix all[source][expert].
static int peak_rows(const KosmosMoeConfig &g, const int *all) {
  const int E = g.E_loc, EG = MOE_EP * E;
  const bool p32 = mx_pad32(g);
  int mx = 0;
  for (int d = 0; d < MOE_EP; d++) {
    int acc = 0;
    for (int e = 0; e < E; e++) {
      int cc = 0;
      for (int s = 0; s < MOE_EP; s++) {
        const int n = all[(size_t)s * EG + d * E + e];
        cc += p32 ? plan_pad32(n) : n;
      }
      acc += (cc + EXPERT_PADDING - 1) / EXPERT_PADDING * EXPERT_PADDING;
    }
    mx = std::max(mx, acc);
  }
  return mx;
}

// The layout the forward derives on the device, from the all-gathered counts all[s][eg], for info and the mirrors.
static void host_layout(KosmosMoe *c, const std::vector<int> &all, MoePlanRec *r, std::vector<HostUnit> &snd,
                        std::vector<HostUnit> &rcv) {
  const KosmosMoeConfig &g = c->cfg;
  const int E = g.E_loc, EG = MOE_EP * E, me = g.coll.rank;
  const bool p32 = mx_pad32(g);
  std::vector<int> col((size_t)EG, 0), colp((size_t)EG, 0);
  for (int s = 0; s < MOE_EP; s++) {
    for (int eg = 0; eg < EG; eg++) {
      col[(size_t)eg] += all[(size_t)s * EG + eg];
      colp[(size_t)eg] += p32 ? plan_pad32(all[(size_t)s * EG + eg]) : all[(size_t)s * EG + eg];
    }
  }
  int mx = 0, mine = 0;
  for (int d = 0; d < MOE_EP; d++) {
    int n = 0;
    for (int e = 0; e < E; e++) {
      n += (colp[(size_t)d * E + e] + EXPERT_PADDING - 1) / EXPERT_PADDING * EXPERT_PADDING;
    }
    mx = std::max(mx, n);
    mine = d == me ? n : mine;
  }
  const bool ovf = mx > c->arena_rows;
  memset(r, 0, sizeof *r);
  r->status = ovf ? PLAN_OVERFLOW : 0;
  r->Mpad = ovf ? 0 : mine, r->Mpad_max = mx;
  r->nt1 = r->Mpad / BLOCK_ROW * (2 * g.I / BLOCK_COL), r->nt2 = r->Mpad / BLOCK_ROW * (g.H / BLOCK_COL);
  c->pool_cnt.assign((size_t)E, 0);
  snd.assign((size_t)EG, HostUnit{0, 0});
  rcv.assign((size_t)EG, HostUnit{0, 0});
  int drows = 0, crows = 0;
  for (int e = 0; e < E && !ovf; e++) {
    c->pool_cnt[(size_t)e] = colp[(size_t)me * E + e];
    for (int x = 0; x < MOE_EP; x++) {
      const size_t u = (size_t)e * MOE_EP + x;
      snd[u] = HostUnit{x, all[(size_t)me * EG + x * E + e]};
      rcv[u] = HostUnit{x, all[(size_t)x * EG + me * E + e]};
      drows += snd[u].cnt, crows += rcv[u].cnt;
      r->dself += x == me ? snd[u].cnt : 0;
      r->cself += x == me ? rcv[u].cnt : 0;
    }
  }
  r->offrows = drows - r->dself, r->crows = crows;
}

// The most rows one call can need on a rank: every rank's T k rows, each (source, expert) segment padded.
static int rows_max(const KosmosMoeConfig &g) {
  const long long n = (long long)MOE_EP * g.T * g.k + (long long)g.E_loc * (MOE_EP * 31 + EXPERT_PADDING);
  return (int)((n + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW);
}

// The device bytes at rows rows of what the capacity sizes: the workspace's buffers, its pool's least size (one call's
// block) and n contexts' arenas and tables.
static size_t rows_bytes(const KosmosMoeWorkspace *w, int rows, size_t n) {
  const KosmosMoeConfig &g = w->cfg;
  const size_t H = g.H, I = g.I, N1 = 2 * I, r1 = (size_t)rows + 1, rs = (size_t)rows / 32 + 1;
  const size_t nrbm = (size_t)rows / BLOCK_ROW, nrbs = g.shared ? (size_t)g.T / BLOCK_ROW : 0;
  size_t b = r1 * H * 2;
  if (g.backward) {
    b += (!w->comm || w->own_comm ? arena_layout(g, rows, true).total : 0) + 2 * nrbm * 4 + r1 * N1 * 2;
    const size_t pr = w->prows ? (size_t)w->prows : g.pool_bytes / (2 * H);
    b += (g.mxfp8 ? 0 : r1 * I * 2) + (nrbm + 1) * 4 + (RINFO_WM + MOE_BWD_MAX_EXPERTS) * 4;
    b += std::max(pr, (size_t)pool_head_rows(g, rows)) * 2 * H;
    if (g.mxfp8) {
      b += std::max(r1, (size_t)g.T * (g.probs_in && g.shared ? 2 : 1)) * 2 * H + r1 * H + rs * H + r1 * N1 + rs * N1;
      b += nrbm * (H / MX_QCOLS) * 4 + r1 * I * 2;
      b += r1 * H * 2 + nrbm * (H / MX_QCOLS) * 4;
    }
  }
  size_t x = arena_layout(g, rows, false).total + (2 * r1 + rows + 2 * nrbm) * 4;
  x += ((nrbm + nrbs) * (N1 / BLOCK_COL + H / BLOCK_COL) + nrbs * (N1 / BLOCK_COL)) * sizeof(TileDesc);
  x += g.backward ? 0 : r1 * I * 2;
  return b + n * x;
}

// Collective: the KosmosMoeCapacityStat of the open contexts: forwards keep a running max of their peaks in the plan
// record (read and reset here).
static hipError_t capacity_query(KosmosMoe *c, KosmosMoeCapacityStat *out) {
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = w->cfg;
  const size_t per = 2, pk = offsetof(MoePlanRec, peak);
  c->err.clear();
  const hipError_t ge = ws_gate(c, "capacity query", false);
  if (ge != hipSuccess) {
    return ge;
  }
  std::vector<KosmosMoe *> open;
  for (KosmosMoe *x : w->ctxs) {
    if (x->arena_rows) {
      open.push_back(x);
    }
  }
  bool ok = true;
  if (w->hstage_n < open.size() * per) {
    if (w->hstage) {
      (void)hipHostFree(w->hstage);
    }
    w->hstage = nullptr;
    ok = hipHostMalloc((void **)&w->hstage, open.size() * per * sizeof(int), hipHostMallocDefault) == hipSuccess;
    w->hstage_n = ok ? open.size() * per : 0;
  }
  for (size_t i = 0; i < open.size() && ok; i++) {
    const KosmosMoe *x = open[i];
    int *h = w->hstage + i * per;
    ok = hipMemcpyAsync(h, x->p.status, sizeof(int), hipMemcpyDeviceToHost, nullptr) == hipSuccess &&
         hipMemcpyAsync(h + 1, (char *)x->p.rec + pk, sizeof(int), hipMemcpyDeviceToHost, nullptr) == hipSuccess &&
         hipMemsetAsync(x->p.status, 0, sizeof(int), nullptr) == hipSuccess &&
         hipMemsetAsync((char *)x->p.rec + pk, 0, sizeof(int), nullptr) == hipSuccess;
  }
  ok = ok && hipStreamSynchronize(nullptr) == hipSuccess;
  int mine[5] = {ok ? 1 : 0, 0, 0, 0, 0};
  for (size_t i = 0; i < open.size() && ok; i++) {
    const int *h = w->hstage + i * per;
    const int p = h[1];
    open[i]->status_seen |= h[0];
    mine[1] = std::max(mine[1], p);
    mine[2] += (h[0] & PLAN_ROUNDS) != 0;
    mine[3] += (h[0] & PLAN_OVERFLOW) != 0;
    mine[4] += (h[0] & PLAN_ROUNDS) && p > open[i]->arena_rows;
  }
  std::vector<int> all((size_t)5 * MOE_EP);
  if (!allgather(c, mine, all.data(), sizeof mine)) {
    return set_error(c->err, hipErrorUnknown, "capacity query: all-gather failed");
  }
  KosmosMoeCapacityStat r = {0, (g.capacity_rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW, rows_max(g),
                             (int)open.size(), 0, 0, 0};
  for (int q = 0; q < MOE_EP; q++) {
    const int *a = &all[(size_t)5 * q];
    if (!a[0]) {
      return set_error(c->err, hipErrorUnknown, "capacity query: a status or count read failed (rank %d)", q);
    }
    r.peak = std::max(r.peak, a[1]), r.rounds = std::max(r.rounds, a[2]), r.overflow = std::max(r.overflow, a[3]);
    r.cap_rounds = std::max(r.cap_rounds, a[4]);
  }
  *out = r;
  return hipSuccess;
}

// Collective: closes every open context's arenas (keeping its status bits) and the workspace (and the pool if it would
// not hold one call's block), then opens them at rows rows; every context's device arguments are built again.
static hipError_t ws_reopen(KosmosMoe *c, int rows, const std::vector<KosmosMoe *> &open) {
  KosmosMoeWorkspace *w = c->ws;
  hipError_t e = hipSuccess;
  for (KosmosMoe *x : open) {
    int st = 0;
    if (x->p.status && hipMemcpy(&st, x->p.status, sizeof st, hipMemcpyDeviceToHost) == hipSuccess) {
      x->status_seen |= st;
    }
    const hipError_t e2 = close_arenas(x);
    e = e != hipSuccess ? e : e2;
  }
  const hipError_t e3 = ws_close(w);
  e = e != hipSuccess ? e : e3;
  if (w->pool && w->prows < pool_head_rows(w->cfg, rows)) {
    pool_close(w);
  }
  if (!agree(c, e == hipSuccess)) {
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  e = ws_open(c, rows);
  for (size_t i = 0; i < open.size() && e == hipSuccess; i++) {
    e = open_arenas(open[i], rows);
    if (e != hipSuccess && open[i] != c) {
      c->err = open[i]->err;
    }
  }
  return e;
}

static void set_rows(KosmosMoeWorkspace *w, int rows) {
  w->cfg.capacity_rows = rows;
  for (KosmosMoe *x : w->ctxs) {
    x->cfg.capacity_rows = rows;
    x->info.g_bytes = g_bytes(x->cfg, rows);
  }
}

// Collective: see kosmos_moe_set_capacity. Every step agrees across ranks, so a failure is seen alike by every rank.
static hipError_t set_capacity(KosmosMoe *c, int rows) {
  KosmosMoeWorkspace *w = c->ws;
  c->err.clear();
  const hipError_t ge = ws_gate(c, "capacity resize", true);
  if (ge != hipSuccess) {
    return ge;
  }
  std::vector<int> rq;
  if (!gather_all(c, rows, rq)) {
    return set_error(c->err, hipErrorUnknown, "capacity resize: all-gather failed");
  }
  rows = *std::max_element(rq.begin(), rq.end());
  if (rows <= 0) {
    return set_error(c->err, hipErrorInvalidValue, "capacity resize: rows must be positive");
  }
  rows = std::min((rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW, rows_max(w->cfg));
  if (rows_a32(w->cfg, rows)) {
    return set_error(c->err, hipErrorInvalidValue, "capacity resize to %d rows refused: A-operand offsets exceed 32"
                     " bits", rows);
  }
  std::vector<KosmosMoe *> open;
  bool same = true;
  for (KosmosMoe *x : w->ctxs) {
    if (x->arena_rows) {
      open.push_back(x);
      same = same && x->arena_rows == rows;
    }
  }
  const int old = w->cfg.capacity_rows, orows = w->rows;
  set_rows(w, rows);
  if (!orows || (orows == rows && same)) {
    return hipSuccess;
  }
  for (KosmosMoe *x : open) {
    x->forwarded = 0;
  }
  hipError_t e = ws_reopen(c, rows, open);
  if (e == hipSuccess) {
    return e;
  }
  const std::string why = c->err;
  (void)hipGetLastError();
  set_rows(w, old);
  e = ws_reopen(c, orows, open);
  (void)hipGetLastError();
  const bool kept = agree(c, e == hipSuccess);
  return set_error(c->err, hipErrorOutOfMemory, "capacity resize to %d rows failed (%s); %s", rows, why.c_str(),
                   kept ? "the old capacity is kept" : "the old capacity re-opens at the contexts' next forward");
}

}

hipError_t kosmos_moe_validate(const KosmosMoeConfig *cfg) {
  return kosmos_moe_impl::validate(cfg, nullptr);
}

hipError_t kosmos_moe_arena_bytes(const KosmosMoeConfig *cfg, int rows, size_t *layer_bytes, size_t *shared_bytes) {
  if (kosmos_moe_impl::validate(cfg, nullptr) != hipSuccess || rows < 0) {
    return hipErrorInvalidValue;
  }
  if (layer_bytes) {
    *layer_bytes = kosmos_moe_impl::arena_layout(*cfg, rows, false).total;
  }
  if (shared_bytes) {
    *shared_bytes = cfg->backward ? kosmos_moe_impl::arena_layout(*cfg, rows, true).total : 0;
  }
  return hipSuccess;
}

hipError_t kosmos_moe_workspace_create(KosmosMoeWorkspace **ws, KosmosComm *comm, const KosmosMoeConfig *cfg) {
  if (!ws) {
    return hipErrorInvalidValue;
  }
  *ws = nullptr;
  const hipError_t e = kosmos_moe_impl::validate(cfg, nullptr);
  if (e != hipSuccess) {
    return e;
  }
  KosmosCommInfo ci;
  if (comm && (kosmos_comm_info(comm, &ci) != hipSuccess || ci.rank != cfg->coll.rank ||
               ci.nranks != cfg->coll.nranks)) {
    return hipErrorInvalidValue;
  }
  KosmosMoeWorkspace *w = new KosmosMoeWorkspace();
  w->cfg = *cfg;
  w->comm = comm;
  *ws = w;
  return hipSuccess;
}

hipError_t kosmos_moe_workspace_destroy(KosmosMoeWorkspace *ws) {
  if (!ws) {
    return hipSuccess;
  }
  if (!ws->ctxs.empty()) {
    return hipErrorInvalidValue;
  }
  hipError_t e = hipSuccess;
  if (ws->rows || ws->own_comm || ws->pool) {
    e = hipDeviceSynchronize();
    const hipError_t e2 = kosmos_moe_impl::ws_close(ws);
    e = e != hipSuccess ? e : e2;
    kosmos_moe_impl::pool_close(ws);
  }
  if (ws->hstage) {
    const hipError_t e2 = hipHostFree(ws->hstage);
    e = e != hipSuccess ? e : e2;
  }
  delete ws;
  return e;
}

hipError_t kosmos_moe_pool_resize(KosmosMoeWorkspace *ws, size_t bytes) {
  using namespace kosmos_moe_impl;
  if (!ws) {
    return hipErrorInvalidValue;
  }
  if (!pool_on(ws->cfg)) {
    ws->err = "pool resize: the workspace has no pool (training only)";
    return hipErrorInvalidValue;
  }
  KosmosMoe tmp = KosmosMoe();
  tmp.cfg = ws->cfg, tmp.ws = ws;
  KosmosMoe *c = ws->ctxs.empty() ? &tmp : ws->ctxs[0];
  const hipError_t e = pool_resize(c, bytes);
  ws->err = c->err;
  return e;
}

hipError_t kosmos_moe_pool_info(const KosmosMoeWorkspace *ws, size_t *bytes, size_t *head_bytes,
                                size_t *reserve_bytes) {
  using namespace kosmos_moe_impl;
  if (!ws || !bytes || !head_bytes || !reserve_bytes) {
    return hipErrorInvalidValue;
  }
  const size_t row = 2 * (size_t)ws->cfg.H;
  const bool on = pool_on(ws->cfg) && ws->rows;
  *bytes = ws->pool ? (size_t)ws->prows * row : 0;
  *head_bytes = on ? (size_t)pool_head(ws) * row : 0;
  *reserve_bytes = on && !ws->cfg.mxfp8 ? ((size_t)ws->rows + BLOCK_ROW) * row : 0;
  return hipSuccess;
}

hipError_t kosmos_moe_capacity_query(KosmosMoeWorkspace *ws, KosmosMoeCapacityStat *stat) {
  using namespace kosmos_moe_impl;
  if (!ws || !stat) {
    return hipErrorInvalidValue;
  }
  KosmosMoe tmp = KosmosMoe();
  tmp.cfg = ws->cfg, tmp.ws = ws;
  KosmosMoe *c = ws->ctxs.empty() ? &tmp : ws->ctxs[0];
  const hipError_t e = capacity_query(c, stat);
  ws->err = c->err;
  return e;
}

hipError_t kosmos_moe_set_capacity(KosmosMoeWorkspace *ws, int rows) {
  using namespace kosmos_moe_impl;
  if (!ws) {
    return hipErrorInvalidValue;
  }
  KosmosMoe tmp = KosmosMoe();
  tmp.cfg = ws->cfg, tmp.ws = ws;
  KosmosMoe *c = ws->ctxs.empty() ? &tmp : ws->ctxs[0];
  const hipError_t e = set_capacity(c, rows);
  ws->err = c->err;
  return e;
}

hipError_t kosmos_moe_capacity_bytes(const KosmosMoeWorkspace *ws, int rows, size_t *bytes) {
  using namespace kosmos_moe_impl;
  if (!ws || !bytes || rows <= 0) {
    return hipErrorInvalidValue;
  }
  rows = std::min((rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW, rows_max(ws->cfg));
  *bytes = rows_bytes(ws, rows, ws->ctxs.size());
  return hipSuccess;
}

const char *kosmos_moe_workspace_error(const KosmosMoeWorkspace *ws) {
  return ws ? ws->err.c_str() : "null workspace";
}

hipError_t kosmos_moe_create(KosmosMoe **ctx, KosmosMoeWorkspace *ws) {
  using namespace kosmos_moe_impl;
  if (!ctx || !ws) {
    return hipErrorInvalidValue;
  }
  int slot = 0, n = 0;
  for (bool taken = true; taken; slot += taken) {
    taken = false;
    for (const KosmosMoe *x : ws->ctxs) {
      taken |= x->pslot == slot;
    }
  }
  if (pool_on(ws->cfg) && slot >= POOL_MAXCTX) {
    return hipErrorInvalidValue;
  }
  for (const KosmosMoe *x : ws->ctxs) {
    n = std::max(n, x->pslot + 1);
  }
  n = std::max(n, slot + 1);
  if (ws->pool) {
    const hipError_t e = hipMemcpy(ws->pool + 1, &n, sizeof n, hipMemcpyHostToDevice);
    if (e != hipSuccess) {
      return e;
    }
  }
  KosmosMoe *c = new KosmosMoe();
  c->cfg = ws->cfg;
  c->ws = ws;
  c->pslot = slot;
  if (c->cfg.capacity_rows > 0) {
    c->info.g_bytes = g_bytes(c->cfg, (c->cfg.capacity_rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW);
  }
  if (c->cfg.backward) {
    bwd_shape_plan(c);
  }
  ws->ctxs.push_back(c);
  *ctx = c;
  return hipSuccess;
}

const KosmosMoeInfo *kosmos_moe_info(const KosmosMoe *ctx) {
  return ctx ? &ctx->info : nullptr;
}

const char *kosmos_moe_error(const KosmosMoe *ctx) {
  return ctx ? ctx->err.c_str() : "null context";
}

hipError_t kosmos_moe_plan(KosmosMoe *c, const int *topk_idx, const float *topk_w) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  const KosmosMoeConfig &g = c->cfg;
  const int E = g.E_loc, k = g.k, T = g.T;
  const int EP = MOE_EP;
  c->err.clear();
  c->planned = 0;
  if (hipDeviceSynchronize() != hipSuccess) {
    set_error(c->err, hipErrorUnknown, "device synchronize failed before the plan");
  }
  bool ok = c->err.empty() && topk_idx && topk_w;
  for (int t = 0; ok && t < T; t++) {
    for (int j = 0; ok && j < k; j++) {
      const int eg = topk_idx[(size_t)t * k + j];
      ok = eg >= 0 && eg < EP * E;
      for (int q = 0; ok && q < j; q++) {
        ok = topk_idx[(size_t)t * k + q] != eg;
      }
    }
  }
  if (!ok && c->err.empty()) {
    set_error(c->err, hipErrorInvalidValue,
              "routing: null, an expert id outside [0, %d), or a token routed twice to one expert", EP * E);
  }
  if (!agree(c, ok)) {
    return hipErrorInvalidValue;
  }
  std::vector<int> mine((size_t)EP * E, 0), all((size_t)EP * EP * E);
  for (size_t x = 0; x < (size_t)T * k; x++) {
    mine[(size_t)topk_idx[x]]++;
  }
  if (!allgather(c, mine.data(), all.data(), mine.size() * sizeof(int))) {
    return set_error(c->err, hipErrorUnknown, "routing count all-gather failed");
  }
  const int Mpad_max = peak_rows(g, all.data());
  const int need = (std::max(g.capacity_rows, Mpad_max) + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW;
  if (c->arena_rows < need) {
    (void)close_arenas(c);
    const hipError_t e = open_arenas(c, need);
    if (e != hipSuccess) {
      return e;
    }
  }
  // The agreement is also the barrier: no rank starts a call before every rank has zeroed its arenas (a peer's first
  // call writes into them).
  const hipError_t re = reset_state(c);
  if (!agree(c, re == hipSuccess)) {
    return re != hipSuccess ? re : hipErrorUnknown;
  }
  KosmosMoePlan &p = c->p;
  KosmosMoeFwd &f = c->f;
  KosmosMoeBwd &b = c->b;
  hipError_t pe = hipMemcpy(p.idx, topk_idx, (size_t)T * k * sizeof(int), hipMemcpyHostToDevice);
  if (pe == hipSuccess) {
    pe = hipMemcpy(p.w, topk_w, (size_t)T * k * sizeof(float), hipMemcpyHostToDevice);
  }
  if (pe != hipSuccess) {
    set_error(c->err, pe, "routing copy: %s", hipGetErrorString(pe));
  }
  if (!agree(c, pe == hipSuccess)) {
    return pe != hipSuccess ? pe : hipErrorInvalidValue;
  }
  MoePlanRec r;
  std::vector<HostUnit> sunits, runits;
  host_layout(c, all, &r, sunits, runits);
  set_mirrors(c, r);
  KosmosMoeInfo &in = c->info;
  {
    double fmean;
    int fmx;
    census(runits, MOE_EP, &fmean, &fmx);
    std::vector<int> mxs;
    if (!gather_all(c, fmx, mxs)) {
      return set_error(c->err, hipErrorUnknown, "census all-gather failed");
    }
    in.fwd_census_max = *std::max_element(mxs.begin(), mxs.end());
    if (in.fwd_census_max > 1) {
      return set_error(c->err, hipErrorInvalidValue, "leg-B census: a combine lane serves %d destinations",
                       in.fwd_census_max);
    }
  }
  if (g.backward) {
    double dmean, cmean;
    int dmx, cmx;
    census(sunits, EP, &dmean, &dmx);
    census(runits, b.cncomm, &cmean, &cmx);
    struct Row {
      double dmean, cmean;
      long long dself, drows, cself, crows;
      int dmx, cmx;
    } row{dmean, cmean, r.dself, (long long)r.dself + r.offrows, r.cself, r.crows, dmx, cmx};
    std::vector<Row> rows(EP);
    if (!allgather(c, &row, rows.data(), sizeof(Row))) {
      return set_error(c->err, hipErrorUnknown, "census all-gather failed");
    }
    in.bwd_census_dmean = in.bwd_census_cmean = 0.0, in.bwd_census_dmax = in.bwd_census_cmax = 0;
    in.disp_self = in.disp_rows = in.comb_self = in.comb_rows = 0;
    for (int q = 0; q < EP; q++) {
      const Row &x = rows[(size_t)q];
      in.bwd_census_dmean = std::max(in.bwd_census_dmean, x.dmean), in.bwd_census_cmean = std::max(in.bwd_census_cmean, x.cmean);
      in.bwd_census_dmax = std::max(in.bwd_census_dmax, x.dmx), in.bwd_census_cmax = std::max(in.bwd_census_cmax, x.cmx);
      in.disp_self += x.dself, in.disp_rows += x.drows, in.comb_self += x.cself, in.comb_rows += x.crows;
    }
    if (in.bwd_census_dmax > 1 || in.bwd_census_cmax > 1) {
      return set_error(c->err, hipErrorInvalidValue,
                       "backward census FAILED: a comm workgroup serves more than one destination (dispatch %d,"
                       " combine %d)", in.bwd_census_dmax, in.bwd_census_cmax);
    }
  }
  in.plans++;
  c->planned = 1;
  return hipSuccess;
}

hipError_t kosmos_moe_plan_status(KosmosMoe *c, int *status) {
  using namespace kosmos_moe_impl;
  if (!c || !status) {
    return hipErrorInvalidValue;
  }
  *status = c->status_seen;
  c->status_seen = 0;
  if (!c->arena_rows) {
    return hipSuccess;
  }
  int st = 0;
  KM_TRY(hipDeviceSynchronize());
  KM_TRY(read_plan(c));
  KM_TRY(hipMemcpy(&st, c->p.status, sizeof(int), hipMemcpyDeviceToHost));
  KM_TRY(hipMemset(c->p.status, 0, sizeof(int)));
  *status |= st;
  return hipSuccess;
}

namespace kosmos_moe_impl {

// Collective and host-blocking: maps the arenas and allocates every table at capacity_rows. Refused inside a stream
// capture (s), which it would invalidate: kosmos_moe_prepare does it beforehand.
static hipError_t open_capacity(KosmosMoe *c, hipStream_t s) {
  c->err.clear();
  hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) {
    return set_error(c->err, hipErrorStreamCaptureUnsupported, "first forward inside a stream capture: call"
                                                               " kosmos_moe_prepare (or one forward) on every rank"
                                                               " before capturing");
  }
  const bool ok = c->cfg.capacity_rows > 0;
  if (!agree(c, ok)) {
    if (!ok) {
      set_error(c->err, hipErrorInvalidValue, "a forward with a device routing needs capacity_rows > 0");
    }
    return hipErrorInvalidValue;
  }
  return open_arenas(c, (c->cfg.capacity_rows + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW);
}

// A forward's routing: the caller's device arrays (mapping the arenas on first use), or the latest plan's.
static hipError_t forward_routing(KosmosMoe *c, const int **topk_idx, const float **topk_w, hipStream_t s) {
  if (!*topk_idx != !*topk_w) {
    return set_error(c->err, hipErrorInvalidValue, "forward: topk_idx and topk_w must both be set or both be null");
  }
  if (!*topk_idx) {
    if (!c->planned) {
      return set_error(c->err, hipErrorInvalidValue, "forward without a routing and without kosmos_moe_plan");
    }
    *topk_idx = c->p.idx, *topk_w = c->p.w;
    return hipSuccess;
  }
  return c->arena_rows ? hipSuccess : open_capacity(c, s);
}

// The fused router's forward kernels with score function SC (ks: k in {1, 2, 4}; mxt: MXFP8 training, its shared call
// a ShCallMx).
template <typename ACT, int SC>
static void router_fwd(bool ks, int burst, int grid, hipStream_t s, const Gl &A, const bf16 *x, const bf16 *w1,
                       const bf16 *w2, bf16 *o, bf16 *gp, const int *idx, const float *w, int *tc,
                       const MoeFwdArgs *args, const RouterCall &rc, const ShCall *sc, const MxTrainCall *mxt,
                       bool q8) {
  const ShCallMx *sx = static_cast<const ShCallMx *>(sc);
  if (mxt && ks && sc) {
    moe_mx_router_launch_fwd<ACT, SC, 1, true>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *mxt, sx, q8);
  } else if (mxt && ks) {
    moe_mx_router_launch_fwd<ACT, SC, 1, false>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *mxt,
                                                nullptr, q8);
  } else if (mxt && sc) {
    moe_mx_router_launch_fwd<ACT, SC, 0, true>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *mxt, sx, q8);
  } else if (mxt) {
    moe_mx_router_launch_fwd<ACT, SC, 0, false>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *mxt,
                                                nullptr, q8);
  } else if (ks && sc) {
    moe_shared_router_launch_fwd_ks<ACT, SC>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *sc);
  } else if (ks) {
    moe_router_launch_fwd_ks<ACT, SC>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc);
  } else if (sc) {
    moe_shared_router_launch_fwd<ACT, SC>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, *sc);
  } else {
    moe_router_launch_fwd<ACT, SC>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc);
  }
}

template <typename ACT>
static void router_fwd_score(int score, bool ks, int burst, int grid, hipStream_t s, const Gl &A, const bf16 *x,
                             const bf16 *w1, const bf16 *w2, bf16 *o, bf16 *gp, const int *idx, const float *w, int *tc,
                             const MoeFwdArgs *args, const RouterCall &rc, const ShCall *sc, const MxTrainCall *mxt,
                             bool q8) {
  switch (score) {
    case KOSMOS_ROUTER_SIGMOID:
      router_fwd<ACT, RT_SIGMOID>(ks, burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, sc, mxt, q8);
      break;
    case KOSMOS_ROUTER_SQRTSOFTPLUS:
      router_fwd<ACT, RT_SQRTSOFTPLUS>(ks, burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, sc, mxt, q8);
      break;
    case KOSMOS_ROUTER_SOFTMAX_PRE:
      router_fwd<ACT, RT_SOFTMAX_PRE>(ks, burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, sc, mxt, q8);
      break;
    default:
      router_fwd<ACT, RT_SOFTMAX>(ks, burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, rc, sc, mxt, q8);
      break;
  }
}

// The router-free forward kernels with activation ACT (ks: k in {1, 2, 4}; pin: probs on input).
template <typename ACT>
static void fwd_kernels(bool ks, bool pin, int burst, int grid, hipStream_t s, const Gl &A, const bf16 *x,
                        const bf16 *w1, const bf16 *w2, bf16 *o, bf16 *gp, const int *idx, const float *w, int *tc,
                        const MoeFwdArgs *args, const MxCall *mx, const MxTrainCall *mxt, bool q8, const ShCall *sc) {
  if (sc && mxt) {
    // MXFP8 training's shared call is a ShCallMx (kosmos_moe_forward_shared_mxfp8_train).
    const ShCallMx &sx = static_cast<const ShCallMx &>(*sc);
    if (pin) {
      moe_mx_shared_launch_fwd<ACT, 2>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *mxt, sx, q8);
    } else if (ks) {
      moe_mx_shared_launch_fwd<ACT, 1>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *mxt, sx, q8);
    } else {
      moe_mx_shared_launch_fwd<ACT, 0>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *mxt, sx, q8);
    }
  } else if (pin && sc) {
    moe_shared_pin_launch_fwd_ks<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *sc);
  } else if (pin && (mx || mxt)) {
    moe_mx_launch_fwd<ACT, 2>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, mx, mxt, q8);
  } else if (pin) {
    moe_ks_pin_launch_fwd<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args);
  } else if (sc && ks) {
    moe_shared_launch_fwd_ks<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *sc);
  } else if (sc) {
    moe_shared_launch_fwd<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, *sc);
  } else if ((mx || mxt) && ks) {
    moe_mx_launch_fwd<ACT, 1>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, mx, mxt, q8);
  } else if (mx || mxt) {
    moe_mx_launch_fwd<ACT, 0>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args, mx, mxt, q8);
  } else if (ks) {
    moe_ks_launch_fwd<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args);
  } else {
    moe_plain_launch_fwd<ACT>(burst, grid, s, A, x, w1, w2, o, gp, idx, w, tc, args);
  }
}

// rc: the fused router's call, or null (the caller's routing). mx: the MXFP8 scales (X, W1, W2 fp8), or null.
static hipError_t launch_forward(KosmosMoe *c, const void *X, const void *W1, const void *W2, const int *topk_idx,
                                 const float *topk_w, void *out, void *G, const RouterCall *rc, const MxCall *mx,
                                 const MxTrainCall *mxt, hipStream_t s, const hipEvent_t *ev,
                                 const ShCall *sc = nullptr) {
  KosmosMoeFwd &f = c->f;
  const KosmosMoeConfig &g = c->cfg;
  if (pool_on(g) && !c->ws->pool) {
    return set_error(c->err, hipErrorOutOfMemory, "forward: no pool (kosmos_moe_pool_resize failed)");
  }
  const Gl A_gl((bf16 *)f.recv, nullptr, nullptr, (size_t)c->arena_rows + 1, (size_t)g.H);
  const bf16 *x = (const bf16 *)X, *w1 = (const bf16 *)W1, *w2 = (const bf16 *)W2;
  bf16 *o = (bf16 *)out, *gp = pool_on(g) ? (G ? c->ws->xpool : nullptr) : (bf16 *)G;
  KM_TRY(record(ev, 0, s));
  if (rc && g.act) {
    router_fwd_score<GluClamp10>(g.router.score, g.k <= 4, f.burst, g.wgcap, s, A_gl, x, w1, w2, o, gp, topk_idx,
                                 topk_w, f.t1, f.args, *rc, sc, mxt, g.fp8_combine);
  } else if (rc) {
    router_fwd_score<SwiGLU>(g.router.score, g.k <= 4, f.burst, g.wgcap, s, A_gl, x, w1, w2, o, gp, topk_idx, topk_w,
                             f.t1, f.args, *rc, sc, mxt, g.fp8_combine);
  } else if (g.act) {
    fwd_kernels<GluClamp10>(g.k <= 4, g.probs_in, f.burst, g.wgcap, s, A_gl, x, w1, w2, o, gp, topk_idx, topk_w, f.t1,
                            f.args, mx, mxt, g.fp8_combine, sc);
  } else {
    fwd_kernels<SwiGLU>(g.k <= 4, g.probs_in, f.burst, g.wgcap, s, A_gl, x, w1, w2, o, gp, topk_idx, topk_w, f.t1,
                        f.args, mx, mxt, g.fp8_combine, sc);
  }
  c->forwarded = 1;
  c->fwd_g = G;
  c->fwd_x = X;
  c->fwd_w1 = mxt ? W1 : nullptr, c->fwd_w1s = mxt ? (const void *)mxt->w1s : nullptr;
  c->fwd_gs = sc ? (const void *)sc->g : nullptr;
  c->fwd_sg = sc ? sc->sg : nullptr;
  c->fwd_mx = mxt ? 2 : mx ? 1 : 0;
  c->r.routed = rc != nullptr;
  KM_TRY(record(ev, 1, s));
  return hipGetLastError();
}

// The fused router's backward kernels with score function SC (ks: k in {1, 2, 4}; mxc: MXFP8 training, with the shared
// expert a MxBwdCallSh).
template <typename ACT, int SC>
static void router_bwd(bool ks, int grid, hipStream_t s, const Gl &dY, int *tc, const SingleArgs *args,
                       const SingleCall &call, const SingleRouterCall &rc, const SingleShCall *sc, const MxBwdCall *mxc,
                       bool q8) {
  if (mxc && ks && sc) {
    moe_mx_router_launch_bwd<ACT, SC, 1, true, false>(grid, s, dY, tc, args, call, sc, rc, *mxc, q8);
  } else if (mxc && ks) {
    moe_mx_router_launch_bwd<ACT, SC, 1, false, false>(grid, s, dY, tc, args, call, sc, rc, *mxc, q8);
  } else if (mxc && sc && sc->sg) {
    moe_mx_router_launch_bwd<ACT, SC, 0, true, true>(grid, s, dY, tc, args, call, sc, rc, *mxc, q8);
  } else if (mxc && sc) {
    moe_mx_router_launch_bwd<ACT, SC, 0, true, false>(grid, s, dY, tc, args, call, sc, rc, *mxc, q8);
  } else if (mxc) {
    moe_mx_router_launch_bwd<ACT, SC, 0, false, false>(grid, s, dY, tc, args, call, sc, rc, *mxc, q8);
  } else if (ks && sc) {
    moe_shared_router_launch_bwd_ks<ACT, SC>(grid, s, dY, tc, args, SingleCallSh{call, *sc}, rc);
  } else if (ks) {
    moe_router_launch_bwd_ks<ACT, SC>(grid, s, dY, tc, args, call, rc);
  } else if (sc && sc->sg) {
    moe_shared_router_launch_bwd<ACT, SC, true>(grid, s, dY, tc, args, SingleCallSh{call, *sc}, rc);
  } else if (sc) {
    moe_shared_router_launch_bwd<ACT, SC, false>(grid, s, dY, tc, args, SingleCallSh{call, *sc}, rc);
  } else {
    moe_router_launch_bwd<ACT, SC>(grid, s, dY, tc, args, call, rc);
  }
}

template <typename ACT>
static void router_bwd_score(int score, bool ks, int grid, hipStream_t s, const Gl &dY, int *tc, const SingleArgs *args,
                             const SingleCall &call, const SingleRouterCall &rc, const SingleShCall *sc,
                             const MxBwdCall *mxc, bool q8) {
  switch (score) {
    case KOSMOS_ROUTER_SIGMOID:
      router_bwd<ACT, RT_SIGMOID>(ks, grid, s, dY, tc, args, call, rc, sc, mxc, q8);
      break;
    case KOSMOS_ROUTER_SQRTSOFTPLUS:
      router_bwd<ACT, RT_SQRTSOFTPLUS>(ks, grid, s, dY, tc, args, call, rc, sc, mxc, q8);
      break;
    case KOSMOS_ROUTER_SOFTMAX_PRE:
      router_bwd<ACT, RT_SOFTMAX_PRE>(ks, grid, s, dY, tc, args, call, rc, sc, mxc, q8);
      break;
    default:
      router_bwd<ACT, RT_SOFTMAX>(ks, grid, s, dY, tc, args, call, rc, sc, mxc, q8);
      break;
  }
}

// The router-free backward kernels with activation ACT (ks: k in {1, 2, 4}; px, pw: probs on input's X and weights).
template <typename ACT>
static void bwd_kernels(bool ks, int grid, hipStream_t s, const Gl &dY, int *tc, const SingleArgs *args,
                        const SingleCall &call, const MxBwdCall *mxc, bool q8, const SingleShCall *sc, const bf16 *px,
                        const float *pw) {
  if (sc && mxc) {
    // MXFP8 training's shared call is a MxBwdCallSh (kosmos_moe_backward_shared_mxfp8).
    const MxBwdCallSh &mx = static_cast<const MxBwdCallSh &>(*mxc);
    const SingleCallSh cs = {call, *sc};
    if (px) {
      moe_mx_shared_launch_bwd<ACT, 2, false>(grid, s, dY, tc, args, cs, mx, q8, pw);
    } else if (ks) {
      moe_mx_shared_launch_bwd<ACT, 1, false>(grid, s, dY, tc, args, cs, mx, q8, pw);
    } else if (sc->sg) {
      moe_mx_shared_launch_bwd<ACT, 0, true>(grid, s, dY, tc, args, cs, mx, q8, pw);
    } else {
      moe_mx_shared_launch_bwd<ACT, 0, false>(grid, s, dY, tc, args, cs, mx, q8, pw);
    }
  } else if (px && sc) {
    moe_shared_pin_launch_bwd_ks<ACT>(grid, s, dY, tc, args, SingleCallSh{call, *sc}, px, pw);
  } else if (px && mxc) {
    moe_mx_launch_bwd<ACT, 2>(grid, s, dY, tc, args, call, *mxc, q8, pw);
  } else if (px) {
    moe_ks_pin_launch_bwd<ACT>(grid, s, dY, tc, args, call, px, pw);
  } else if (sc && ks) {
    moe_shared_launch_bwd_ks<ACT>(grid, s, dY, tc, args, SingleCallSh{call, *sc});
  } else if (sc && sc->sg) {
    moe_shared_launch_bwd<ACT, true>(grid, s, dY, tc, args, SingleCallSh{call, *sc});
  } else if (sc) {
    moe_shared_launch_bwd<ACT, false>(grid, s, dY, tc, args, SingleCallSh{call, *sc});
  } else if (mxc && ks) {
    moe_mx_launch_bwd<ACT, 1>(grid, s, dY, tc, args, call, *mxc, q8, nullptr);
  } else if (mxc) {
    moe_mx_launch_bwd<ACT, 0>(grid, s, dY, tc, args, call, *mxc, q8, nullptr);
  } else if (ks) {
    moe_ks_launch_bwd<ACT>(grid, s, dY, tc, args, call);
  } else {
    moe_plain_launch_bwd<ACT>(grid, s, dY, tc, args, call);
  }
}

// rc: the fused router's backward call, or null.
static hipError_t launch_backward(KosmosMoe *c, const void *dout, const void *G, const void *W1, const void *W2,
                                  void *dX, float *dprobs, void *dW1, void *dW2, const SingleRouterCall *rc,
                                  const MxBwdCall *mxc, hipStream_t s, const hipEvent_t *ev,
                                  const SingleShCall *sc = nullptr) {
  if (!c->forwarded || !c->cfg.backward) {
    return set_error(c->err, hipErrorInvalidValue, "backward before a forward, or created without backward");
  }
  if ((mxc != nullptr) != (c->fwd_mx == 2)) {
    return set_error(c->err, hipErrorInvalidValue,
                     "backward: an MXFP8 backward needs a training MXFP8 forward, a bf16 backward a bf16 forward");
  }
  if (pool_on(c->cfg) ? !c->fwd_g : !G || G != c->fwd_g) {
    return set_error(c->err, hipErrorInvalidValue, "backward: the latest forward must have stored (G ="
                     " KOSMOS_MOE_SAVE)");
  }
  if (sc ? (!sc->g || sc->g != c->fwd_gs || !sc->x || !sc->dw1 || !sc->dw2) : c->fwd_gs != nullptr) {
    return set_error(c->err, hipErrorInvalidValue, "backward: a shared forward (with its Gs) needs the shared backward");
  }
  if (sc && (sc->sg != c->fwd_sg || (sc->sg && c->cfg.k <= 4))) {
    return set_error(c->err, hipErrorInvalidValue,
                     "backward: sgate must be the forward's (or null for both); the gate needs k in {6, 8, 10}");
  }
  KosmosMoeWorkspace *w = c->ws;
  const KosmosMoeConfig &g = c->cfg;
  const Gl dY_gl((bf16 *)w->recv, nullptr, nullptr, (size_t)w->rows + 1, (size_t)g.H);
  const SingleCall call = {(const unsigned short *)dout, (const bf16 *)G, (const bf16 *)W1, (const bf16 *)W2,
                           (bf16 *)dX, w->dtk, (bf16 *)dW1, (bf16 *)dW2, dprobs};
  const bf16 *px = g.probs_in ? (const bf16 *)c->fwd_x : nullptr;
  if (rc && g.act) {
    router_bwd_score<GluClamp10>(g.router.score, g.k <= 4, g.wgcap, s, dY_gl, w->t, c->b.args, call, *rc, sc, mxc,
                                 g.fp8_combine);
  } else if (rc) {
    router_bwd_score<SwiGLU>(g.router.score, g.k <= 4, g.wgcap, s, dY_gl, w->t, c->b.args, call, *rc, sc, mxc,
                             g.fp8_combine);
  } else if (g.act) {
    bwd_kernels<GluClamp10>(g.k <= 4, g.wgcap, s, dY_gl, w->t, c->b.args, call, mxc, g.fp8_combine, sc, px, c->f.wt);
  } else {
    bwd_kernels<SwiGLU>(g.k <= 4, g.wgcap, s, dY_gl, w->t, c->b.args, call, mxc, g.fp8_combine, sc, px, c->f.wt);
  }
  KM_TRY(record(ev, 0, s));
  return hipGetLastError();
}

}

hipError_t kosmos_moe_forward(KosmosMoe *c, const void *X, const void *W1, const void *W2, const int *topk_idx,
                              const float *topk_w, void *out, void *G, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  const hipError_t e = forward_routing(c, &topk_idx, &topk_w, s);
  if (e != hipSuccess) {
    return e;
  }
  return launch_forward(c, X, W1, W2, topk_idx, topk_w, out, G, nullptr, nullptr, nullptr, s, ev);
}

hipError_t kosmos_moe_forward_mxfp8(KosmosMoe *c, const void *X, const void *X_scale, const void *W1,
                                    const void *W1_scale, const void *W2, const void *W2_scale, const int *topk_idx,
                                    const float *topk_w, void *out, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!X_scale || !W1_scale || !W2_scale) {
    return set_error(c->err, hipErrorInvalidValue, "forward_mxfp8: a scale tensor is null");
  }
  if (c->cfg.router.on || c->cfg.H % 512) {
    return set_error(c->err, hipErrorNotSupported, "forward_mxfp8: needs no fused router and H a multiple of 512");
  }
  const hipError_t e = forward_routing(c, &topk_idx, &topk_w, s);
  if (e != hipSuccess) {
    return e;
  }
  const MxCall mx = {(const uint8_t *)X_scale, (const uint8_t *)W1_scale, (const uint8_t *)W2_scale};
  return launch_forward(c, X, W1, W2, topk_idx, topk_w, out, nullptr, nullptr, &mx, nullptr, s, ev);
}

hipError_t kosmos_moe_forward_mxfp8_train(KosmosMoe *c, const void *X, const void *W1, const void *W1_scale,
                                          const void *W2, const void *W2_scale, const int *topk_idx,
                                          const float *topk_w, void *out, void *G, hipStream_t s,
                                          const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.mxfp8 || !c->cfg.backward || !W1_scale || !W2_scale) {
    return set_error(c->err, hipErrorInvalidValue,
                     "forward_mxfp8_train: needs a config with mxfp8 and backward, and both scale tensors");
  }
  const hipError_t e = forward_routing(c, &topk_idx, &topk_w, s);
  if (e != hipSuccess) {
    return e;
  }
  const MxTrainCall mx = {(const uint8_t *)W1_scale, (const uint8_t *)W2_scale};
  return launch_forward(c, X, W1, W2, topk_idx, topk_w, out, G, nullptr, nullptr, &mx, s, ev);
}

hipError_t kosmos_moe_forward_shared(KosmosMoe *c, const void *X, const void *W1, const void *W2, const void *Ws1,
                                     const void *Ws2, const int *topk_idx, const float *topk_w, void *out, void *G,
                                     void *Gs, const float *sgate, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!Ws1 || !Ws2 || !c->cfg.shared || (uintptr_t)sgate % 16 || (sgate && c->cfg.k <= 4)) {
    return set_error(c->err, hipErrorInvalidValue, "forward_shared: Ws1 and Ws2 must be set, on a cfg.shared context;"
                                                   " sgate 16-byte aligned, with k in {6, 8, 10}");
  }
  if (!topk_idx != !topk_w) {
    return set_error(c->err, hipErrorInvalidValue, "forward: topk_idx and topk_w must both be set or both be null");
  }
  if (!topk_idx) {
    if (!c->planned) {
      return set_error(c->err, hipErrorInvalidValue, "forward without a routing and without kosmos_moe_plan");
    }
    topk_idx = c->p.idx, topk_w = c->p.w;
  } else if (!c->arena_rows) {
    const hipError_t e = open_capacity(c, s);
    if (e != hipSuccess) {
      return e;
    }
  }
  const ShCall sc = {(const bf16 *)Ws1, (const bf16 *)Ws2, (bf16 *)Gs, sgate};
  return launch_forward(c, X, W1, W2, topk_idx, topk_w, out, G, nullptr, nullptr, nullptr, s, ev, &sc);
}

hipError_t kosmos_moe_forward_shared_mxfp8_train(KosmosMoe *c, const void *X, const void *W1, const void *W1_scale,
                                                 const void *W2, const void *W2_scale, const void *Ws1,
                                                 const void *Ws1_scale, const void *Ws2, const void *Ws2_scale,
                                                 const int *topk_idx, const float *topk_w, void *out, void *G, void *Gs,
                                                 const float *sgate, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.mxfp8 || !c->cfg.backward || !c->cfg.shared || !Gs || !W1_scale || !W2_scale || !Ws1 ||
      !Ws1_scale || !Ws2 || !Ws2_scale || (uintptr_t)sgate % 16 || (sgate && c->cfg.k <= 4)) {
    return set_error(c->err, hipErrorInvalidValue, "forward_shared_mxfp8_train: needs a config with mxfp8,"
                     " backward and shared, Gs and every weight with its scales; sgate 16-byte aligned, with k in"
                     " {6, 8, 10}");
  }
  const hipError_t e = forward_routing(c, &topk_idx, &topk_w, s);
  if (e != hipSuccess) {
    return e;
  }
  const MxTrainCall mx = {(const uint8_t *)W1_scale, (const uint8_t *)W2_scale};
  const ShCallMx sc = {{(const bf16 *)Ws1, (const bf16 *)Ws2, (bf16 *)Gs, sgate}, (const uint8_t *)Ws1_scale,
                       (const uint8_t *)Ws2_scale};
  return launch_forward(c, X, W1, W2, topk_idx, topk_w, out, G, nullptr, nullptr, &mx, s, ev, &sc);
}

hipError_t kosmos_moe_backward_shared_mxfp8(KosmosMoe *c, const void *dout, const void *G, const void *Gs,
                                            const void *X, const void *W1c, const void *W1c_scale, const void *W2c,
                                            const void *W2c_scale, const void *Ws1c, const void *Ws1c_scale,
                                            const void *Ws2c, const void *Ws2c_scale, void *dX, float *dprobs,
                                            void *dW1, void *dW2, void *dWs1, void *dWs2, int acc,
                                            const float *sgate, float *dsgate, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!X || !W1c_scale || !W2c_scale || !Ws1c || !Ws1c_scale || !Ws2c || !Ws2c_scale || acc < 0 || acc > 2) {
    return set_error(c->err, hipErrorInvalidValue,
                     "backward_shared_mxfp8: X and every weight with its scales must be set and acc in [0, 2]");
  }
  const MxBwdCallSh mx = {{(const uint8_t *)W1c_scale, (const uint8_t *)W2c_scale, (const bf16 *)X,
                           (const uint8_t *)c->fwd_w1, (const uint8_t *)c->fwd_w1s},
                          (const uint8_t *)Ws1c_scale, (const uint8_t *)Ws2c_scale};
  const SingleShCall sc = {(const bf16 *)X, (const bf16 *)Ws1c, (const bf16 *)Ws2c, (const bf16 *)Gs, dWs1, dWs2,
                           acc, sgate, dsgate};
  return launch_backward(c, dout, G, W1c, W2c, dX, dprobs, dW1, dW2, nullptr, &mx, s, ev, &sc);
}

hipError_t kosmos_moe_forward_router_shared(KosmosMoe *c, const void *X, const void *Wr, const float *bias,
                                            const float *logit_bias, const void *W1, const void *W2, const void *Ws1,
                                            const void *Ws2, void *out, void *G, void *Gs, const float *sgate,
                                            float *aux_loss, int *tpe, const unsigned *seed, hipStream_t s,
                                            const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.router.on || c->cfg.mxfp8 || !X || !Wr || !Ws1 || !Ws2 || !c->cfg.shared || (uintptr_t)sgate % 16 ||
      (sgate && c->cfg.k <= 4)) {
    return set_error(c->err, hipErrorInvalidValue, "forward_router_shared: needs the router (bf16), cfg.shared, X,"
                                                   " Wr, Ws1, Ws2; sgate 16-byte aligned, with k in {6, 8, 10}");
  }
  if (!c->arena_rows) {
    const hipError_t e = open_capacity(c, s);
    if (e != hipSuccess) {
      return e;
    }
  }
  const RouterCall rc = {(const bf16 *)Wr, bias, logit_bias, aux_loss, tpe, seed};
  const ShCall sc = {(const bf16 *)Ws1, (const bf16 *)Ws2, (bf16 *)Gs, sgate};
  return launch_forward(c, X, W1, W2, c->p.idx, c->p.w, out, G, &rc, nullptr, nullptr, s, ev, &sc);
}

hipError_t kosmos_moe_backward_shared(KosmosMoe *c, const void *dout, const void *G, const void *Gs, const void *X,
                                      const void *W1, const void *W2, const void *Ws1, const void *Ws2, void *dX,
                                      float *dprobs, void *dW1, void *dW2, void *dWs1, void *dWs2, int acc,
                                      const float *sgate, float *dsgate, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!Ws1 || !Ws2 || acc < 0 || acc > 2) {
    return set_error(c->err, hipErrorInvalidValue, "backward_shared: Ws1 and Ws2 must be set and acc in [0, 2]");
  }
  const SingleShCall sc = {(const bf16 *)X, (const bf16 *)Ws1, (const bf16 *)Ws2, (const bf16 *)Gs, dWs1, dWs2,
                           acc, sgate, dsgate};
  return launch_backward(c, dout, G, W1, W2, dX, dprobs, dW1, dW2, nullptr, nullptr, s, ev, &sc);
}

hipError_t kosmos_moe_forward_router(KosmosMoe *c, const void *X, const void *Wr, const float *bias,
                                     const float *logit_bias, const void *W1, const void *W2, void *out, void *G,
                                     float *aux_loss, int *tpe, const unsigned *seed, hipStream_t s,
                                     const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.router.on || c->cfg.mxfp8 || !X || !Wr) {
    return set_error(c->err, hipErrorInvalidValue,
                     "forward_router: the context has no router or is MXFP8, or X or Wr is null");
  }
  if (!c->arena_rows) {
    const hipError_t e = open_capacity(c, s);
    if (e != hipSuccess) {
      return e;
    }
  }
  const RouterCall rc = {(const bf16 *)Wr, bias, logit_bias, aux_loss, tpe, seed};
  return launch_forward(c, X, W1, W2, c->p.idx, c->p.w, out, G, &rc, nullptr, nullptr, s, ev);
}

hipError_t kosmos_moe_backward(KosmosMoe *c, const void *dout, const void *G, const void *W1, const void *W2, void *dX,
                               float *dprobs, void *dW1, void *dW2, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  return launch_backward(c, dout, G, W1, W2, dX, dprobs, dW1, dW2, nullptr, nullptr, s, ev);
}

hipError_t kosmos_moe_backward_mxfp8(KosmosMoe *c, const void *dout, const void *G, const void *X, const void *W1c,
                                     const void *W1c_scale, const void *W2c, const void *W2c_scale, void *dX,
                                     float *dprobs, void *dW1, void *dW2, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!X || !W1c_scale || !W2c_scale) {
    return set_error(c->err, hipErrorInvalidValue, "backward_mxfp8: X or a scale tensor is null");
  }
  const MxBwdCall mx = {(const uint8_t *)W1c_scale, (const uint8_t *)W2c_scale, (const bf16 *)X,
                        (const uint8_t *)c->fwd_w1, (const uint8_t *)c->fwd_w1s};
  return launch_backward(c, dout, G, W1c, W2c, dX, dprobs, dW1, dW2, nullptr, &mx, s, ev);
}

hipError_t kosmos_moe_backward_router(KosmosMoe *c, const void *dout, const void *G, const void *X, const void *W1,
                                      const void *W2, const void *Wr, const float *aux_scale, void *dX, void *dW1,
                                      void *dW2, void *dWr, float *dlogit_bias, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->r.routed || !X || !Wr || !dWr || !aux_scale) {
    return set_error(c->err, hipErrorInvalidValue,
                     "backward_router: the latest forward did not route, or X, Wr, dWr or aux_scale is null");
  }
  const SingleRouterCall rc = {(const bf16 *)X, (const bf16 *)Wr, (bf16 *)dWr, aux_scale};
  return launch_backward(c, dout, G, W1, W2, dX, dlogit_bias, dW1, dW2, &rc, nullptr, s, ev);
}

hipError_t kosmos_moe_backward_router_shared(KosmosMoe *c, const void *dout, const void *G, const void *Gs,
                                             const void *X, const void *W1, const void *W2, const void *Ws1,
                                             const void *Ws2, const void *Wr, const float *aux_scale, void *dX,
                                             void *dW1, void *dW2, void *dWr, float *dlogit_bias, void *dWs1,
                                             void *dWs2, int acc, const float *sgate, float *dsgate, hipStream_t s,
                                             const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->r.routed || !X || !Wr || !dWr || !Ws1 || !Ws2 || !aux_scale || acc < 0 || acc > 2) {
    return set_error(c->err, hipErrorInvalidValue, "backward_router_shared: the latest forward did not route, X, Wr,"
                                                   " dWr, Ws1, Ws2 or aux_scale is null, or acc not in [0, 2]");
  }
  const SingleRouterCall rc = {(const bf16 *)X, (const bf16 *)Wr, (bf16 *)dWr, aux_scale};
  const SingleShCall sc = {(const bf16 *)X, (const bf16 *)Ws1, (const bf16 *)Ws2, (const bf16 *)Gs, dWs1, dWs2,
                           acc, sgate, dsgate};
  return launch_backward(c, dout, G, W1, W2, dX, dlogit_bias, dW1, dW2, &rc, nullptr, s, ev, &sc);
}

hipError_t kosmos_moe_forward_router_mxfp8_train(KosmosMoe *c, const void *X, const void *Wr, const float *bias,
                                                 const float *logit_bias, const void *W1, const void *W1_scale,
                                                 const void *W2, const void *W2_scale, void *out, void *G,
                                                 float *aux_loss, int *tpe, const unsigned *seed, hipStream_t s,
                                                 const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.router.on || !c->cfg.mxfp8 || !X || !Wr || !W1_scale || !W2_scale) {
    return set_error(c->err, hipErrorInvalidValue, "forward_router_mxfp8_train: needs a config with the router and"
                                                   " mxfp8, X, Wr and the weights' scales");
  }
  if (!c->arena_rows) {
    const hipError_t e = open_capacity(c, s);
    if (e != hipSuccess) {
      return e;
    }
  }
  const RouterCall rc = {(const bf16 *)Wr, bias, logit_bias, aux_loss, tpe, seed};
  const MxTrainCall mx = {(const uint8_t *)W1_scale, (const uint8_t *)W2_scale};
  return launch_forward(c, X, W1, W2, c->p.idx, c->p.w, out, G, &rc, nullptr, &mx, s, ev);
}

hipError_t kosmos_moe_backward_router_mxfp8(KosmosMoe *c, const void *dout, const void *G, const void *X,
                                            const void *W1c, const void *W1c_scale, const void *W2c,
                                            const void *W2c_scale, const void *Wr, const float *aux_scale, void *dX,
                                            void *dW1, void *dW2, void *dWr, float *dlogit_bias, hipStream_t s,
                                            const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->r.routed || !X || !Wr || !dWr || !aux_scale || !W1c_scale || !W2c_scale) {
    return set_error(c->err, hipErrorInvalidValue, "backward_router_mxfp8: the latest forward did not route, or X,"
                                                   " Wr, dWr, aux_scale or a scale tensor is null");
  }
  const SingleRouterCall rc = {(const bf16 *)X, (const bf16 *)Wr, (bf16 *)dWr, aux_scale};
  const MxBwdCall mx = {(const uint8_t *)W1c_scale, (const uint8_t *)W2c_scale, (const bf16 *)X,
                        (const uint8_t *)c->fwd_w1, (const uint8_t *)c->fwd_w1s};
  return launch_backward(c, dout, G, W1c, W2c, dX, dlogit_bias, dW1, dW2, &rc, &mx, s, ev);
}

hipError_t kosmos_moe_forward_router_shared_mxfp8_train(KosmosMoe *c, const void *X, const void *Wr,
                                                        const float *bias, const float *logit_bias, const void *W1,
                                                        const void *W1_scale, const void *W2, const void *W2_scale,
                                                        const void *Ws1, const void *Ws1_scale, const void *Ws2,
                                                        const void *Ws2_scale, void *out, void *G, void *Gs,
                                                        const float *sgate, float *aux_loss, int *tpe,
                                                        const unsigned *seed, hipStream_t s, const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->cfg.router.on || !c->cfg.mxfp8 || !c->cfg.shared || !X || !Wr || !Gs || !W1_scale || !W2_scale ||
      !Ws1 || !Ws1_scale || !Ws2 || !Ws2_scale || (uintptr_t)sgate % 16 || (sgate && c->cfg.k <= 4)) {
    return set_error(c->err, hipErrorInvalidValue, "forward_router_shared_mxfp8_train: needs a config with the router,"
                                                   " mxfp8 and shared, X, Wr, Gs and every weight with its scales;"
                                                   " sgate 16-byte aligned, with k in {6, 8, 10}");
  }
  if (!c->arena_rows) {
    const hipError_t e = open_capacity(c, s);
    if (e != hipSuccess) {
      return e;
    }
  }
  const RouterCall rc = {(const bf16 *)Wr, bias, logit_bias, aux_loss, tpe, seed};
  const MxTrainCall mx = {(const uint8_t *)W1_scale, (const uint8_t *)W2_scale};
  const ShCallMx sc = {{(const bf16 *)Ws1, (const bf16 *)Ws2, (bf16 *)Gs, sgate}, (const uint8_t *)Ws1_scale,
                       (const uint8_t *)Ws2_scale};
  return launch_forward(c, X, W1, W2, c->p.idx, c->p.w, out, G, &rc, nullptr, &mx, s, ev, &sc);
}

hipError_t kosmos_moe_backward_router_shared_mxfp8(KosmosMoe *c, const void *dout, const void *G, const void *Gs,
                                                   const void *X, const void *W1c, const void *W1c_scale,
                                                   const void *W2c, const void *W2c_scale, const void *Ws1c,
                                                   const void *Ws1c_scale, const void *Ws2c, const void *Ws2c_scale,
                                                   const void *Wr, const float *aux_scale, void *dX, void *dW1,
                                                   void *dW2, void *dWr, float *dlogit_bias, void *dWs1, void *dWs2,
                                                   int acc, const float *sgate, float *dsgate, hipStream_t s,
                                                   const hipEvent_t *ev) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  if (!c->r.routed || !X || !Wr || !dWr || !aux_scale || !W1c_scale || !W2c_scale || !Ws1c || !Ws1c_scale || !Ws2c ||
      !Ws2c_scale || acc < 0 || acc > 2) {
    return set_error(c->err, hipErrorInvalidValue, "backward_router_shared_mxfp8: the latest forward did not route,"
                                                   " X, Wr, dWr, aux_scale or a weight or scale tensor is null, or acc"
                                                   " not in [0, 2]");
  }
  const SingleRouterCall rc = {(const bf16 *)X, (const bf16 *)Wr, (bf16 *)dWr, aux_scale};
  const MxBwdCallSh mx = {{(const uint8_t *)W1c_scale, (const uint8_t *)W2c_scale, (const bf16 *)X,
                           (const uint8_t *)c->fwd_w1, (const uint8_t *)c->fwd_w1s},
                          (const uint8_t *)Ws1c_scale, (const uint8_t *)Ws2c_scale};
  const SingleShCall sc = {(const bf16 *)X, (const bf16 *)Ws1c, (const bf16 *)Ws2c, (const bf16 *)Gs, dWs1, dWs2,
                           acc, sgate, dsgate};
  return launch_backward(c, dout, G, W1c, W2c, dX, dlogit_bias, dW1, dW2, &rc, &mx, s, ev, &sc);
}

hipError_t kosmos_moe_prepare(KosmosMoe *c) {
  using namespace kosmos_moe_impl;
  if (!c) {
    return hipErrorInvalidValue;
  }
  return c->arena_rows ? hipSuccess : open_capacity(c, nullptr);
}

hipError_t kosmos_moe_quantize_mxfp8(const void *W, int nmat, int rows, int cols, int col, void *q, void *s,
                                     hipStream_t stream) {
  using namespace kosmos_moe_impl;
  if (!W || !q || !s || nmat <= 0 || rows <= 0 || cols <= 0 || (col ? rows : cols) % 32 || (uintptr_t)q % 4) {
    return hipErrorInvalidValue;
  }
  moe_mx_launch_quant(W, nmat, rows, cols, col != 0, q, s, stream);
  return hipGetLastError();
}

hipError_t kosmos_moe_destroy(KosmosMoe *c) {
  if (!c) {
    return hipSuccess;
  }
  hipError_t e = hipSuccess;
  if (c->arena_rows || c->comm) {
    e = hipDeviceSynchronize();
    const hipError_t e2 = kosmos_moe_impl::close_arenas(c);
    e = e != hipSuccess ? e : e2;
  }
  kosmos_moe_impl::free_tables(c);
  if (c->ws->pool) {
    const int none[3] = {0, 0, 0};
    const hipError_t e2 = hipMemcpy(c->ws->pool + 2 + 3 * c->pslot, none, sizeof none, hipMemcpyHostToDevice);
    e = e != hipSuccess ? e : e2;
  }
  std::vector<KosmosMoe *> &v = c->ws->ctxs;
  v.erase(std::remove(v.begin(), v.end(), c), v.end());
  delete c;
  return e;
}

namespace kosmos_moe_impl {

// The training API's configuration: bf16 training without the router, the shared expert, MXFP8 or probs on input, with
// a capacity.
static hipError_t inst_validate(const KosmosMoeConfig *g, std::string *why) {
  const hipError_t e = validate(g, why);
  if (e != hipSuccess) {
    return e;
  }
  if (!g->backward || g->capacity_rows <= 0 || g->router.on || g->shared || g->mxfp8 || g->probs_in) {
    return refuse(why, "instance: needs backward and capacity_rows > 0, and no router, shared expert, MXFP8 or probs"
                       " on input");
  }
  return hipSuccess;
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
  return inst_foff(g, rows) + arena_layout(g, rows, false, true).total;
}

// One table the record keeps: its base, and its bytes at the capacity (rounded to 16; the allocations' tails are
// padding).
struct RecTab {
  const void *p;
  size_t bytes;
};

// The tables a backward reads from its forward (single pass; a call in rounds re-derives all but the routing order's
// from the record's count matrix), in a fixed order.
static std::vector<RecTab> rec_tables(const KosmosMoe *c, const KosmosMoeConfig &g, int rows) {
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
static size_t rec_layout(const KosmosMoeConfig &g, int rows, const std::vector<RecTab> *tabs, size_t *off_slot,
                         std::vector<RecSeg> *segs) {
  const size_t EG = (size_t)MOE_EP * g.E_loc, E = g.E_loc, TK = (size_t)g.T * g.k, nu = E * MOE_EP;
  const size_t R = rows, nrow = R + 1;
  const size_t sizes[] = {E * 4,  E * 4,  nrow * 4, nrow * 4, TK * 4, R * 4, MOE_EP * 4, nu * 4,
                          TK * 4, nu * 20, nu * 20, TK * 4,  (size_t)MOE_EP * SIG_STRIDE * 4,
                          (size_t)MOE_EP * SIG_STRIDE * 4};
  *off_slot = align_up(REC_CNT + (size_t)MOE_EP * EG * 4, 256);
  size_t off = align_up(*off_slot + R * 4, 256);
  constexpr size_t PIECE = 65536;
  for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
    const size_t n = align_up(sizes[i], 16);
    for (size_t q = 0; tabs && segs && q < n; q += PIECE) {
      segs->push_back(RecSeg{(char *)(*tabs)[i].p + q, (unsigned)(off + q), (unsigned)std::min(PIECE, n - q)});
    }
    off = align_up(off + n, 256);
  }
  return off;
}

static void inst_free(KosmosMoeInstance *in) {
  dfree(in->ctr), dfree(in->seg), dfree(in->scnt), dfree(in->sep);
  if (in->ctx) {
    dfree(in->ctx->inst_fa);
  }
}

// After the context's arenas and tables: the backward's forward arguments (count slots and epochs staged), the counters
// and the record's segments.
static hipError_t inst_setup(KosmosMoeInstance *in) {
  KosmosMoe *c = in->ctx;
  const KosmosMoeConfig &g = in->cfg;
  const size_t EG = (size_t)MOE_EP * g.E_loc;
  MoeFwdArgs ha;
  KM_TRY(hipMemcpy(&ha, c->f.args, sizeof ha, hipMemcpyDeviceToHost));
  ha.p.cnts[g.coll.rank] = in->scnt, ha.ep = in->sep;
  KM_TRY(hipMemcpy(c->inst_fa, &ha, sizeof ha, hipMemcpyHostToDevice));
  std::vector<RecSeg> segs;
  const std::vector<RecTab> tabs = rec_tables(c, g, in->rows);
  in->rec_bytes = rec_layout(g, in->rows, &tabs, &in->off_slot, &segs);
  in->nseg = (int)segs.size();
  KM_TRY(dupload(&in->seg, segs));
  KM_TRY(hipMemset(in->scnt, 0, 2 * MOE_EP * EG * sizeof(int)));
  KM_TRY(hipDeviceSynchronize());
  return hipSuccess;
}

}

hipError_t kosmos_moe_instance_bytes(const KosmosMoeConfig *cfg, size_t *bytes) {
  using namespace kosmos_moe_impl;
  if (!bytes || inst_validate(cfg, nullptr) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  *bytes = inst_bytes(*cfg, inst_rows(*cfg));
  return hipSuccess;
}

hipError_t kosmos_moe_saved_bytes(const KosmosMoeConfig *cfg, int rows, KosmosMoeSavedBytes *bytes) {
  using namespace kosmos_moe_impl;
  if (!bytes || inst_validate(cfg, nullptr) != hipSuccess || rows < 1 || rows > inst_rows(*cfg)) {
    return hipErrorInvalidValue;
  }
  size_t off_slot;
  const size_t H = cfg->H, I = cfg->I, r = (size_t)rows;
  bytes->xp = align_up((r + 1) * H * 2, 256);
  bytes->g = align_up(r * 2 * I * 2, 256);
  bytes->a = align_up(r * I * 2, 256);
  bytes->y = align_up((size_t)cfg->T * cfg->k * H * 2, 256);
  bytes->rec = rec_layout(*cfg, inst_rows(*cfg), nullptr, &off_slot, nullptr);
  return hipSuccess;
}

hipError_t kosmos_moe_instance_create(KosmosMoeInstance **inst, KosmosComm *comm, const KosmosMoeConfig *cfg) {
  using namespace kosmos_moe_impl;
  if (!inst) {
    return hipErrorInvalidValue;
  }
  *inst = nullptr;
  std::string why;
  if (inst_validate(cfg, &why) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  KosmosMoeInstance *in = new KosmosMoeInstance();
  in->cfg = *cfg;
  in->cfg.pool_bytes = 0, in->cfg.comm_offset = 0;
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
    e = kosmos_moe_workspace_create(&in->ws, in->comm, &g);
  }
  if (e == hipSuccess) {
    in->ws->inst = 1, in->ws->inst_foff = inst_foff(g, in->rows);
    e = kosmos_moe_create(&in->ctx, in->ws);
  }
  if (e == hipSuccess) {
    KosmosMoe *c = in->ctx;
    e = dzero(&in->ctr, ICTR_N);
    e = e == hipSuccess ? dzero(&in->scnt, 2 * MOE_EP * EG) : e;
    e = e == hipSuccess ? dzero(&in->sep, 2) : e;
    e = e == hipSuccess ? dalloc(&c->inst_fa, 1) : e;
    if (e != hipSuccess) {
      set_error(in->err, e, "instance state: %s", hipGetErrorString(e));
    }
  }
  // Collective from here: every rank takes the same steps (open_capacity agrees across ranks).
  const KosmosCollective &coll = g.coll;
  if (coll_agree(coll, e == hipSuccess) <= 0) {
    if (e == hipSuccess) {
      in->err = "instance: refused on another rank";
      e = hipErrorUnknown;
    }
  } else {
    e = open_capacity(in->ctx, nullptr);
    if (e != hipSuccess) {
      in->err = in->ctx->err;
    } else {
      e = inst_setup(in);
      if (e != hipSuccess) {
        in->err = in->ctx->err;
      }
    }
    if (coll_agree(coll, e == hipSuccess) <= 0 && e == hipSuccess) {
      in->err = "instance: setup refused on another rank";
      e = hipErrorUnknown;
    }
  }
  if (e != hipSuccess) {
    *inst = in;
    (void)kosmos_moe_instance_destroy(in);
    *inst = nullptr;
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
  hipError_t e = hipDeviceSynchronize();
  inst_free(in);
  if (in->ctx) {
    const hipError_t e2 = kosmos_moe_destroy(in->ctx);
    e = e != hipSuccess ? e : e2;
  }
  if (in->ws) {
    const hipError_t e2 = kosmos_moe_workspace_destroy(in->ws);
    e = e != hipSuccess ? e : e2;
  }
  if (in->own_comm && in->comm) {
    const hipError_t e2 = kosmos_comm_destroy(in->comm);
    e = e != hipSuccess ? e : e2;
  }
  delete in;
  return e;
}

const char *kosmos_moe_instance_error(const KosmosMoeInstance *in) {
  return !in ? "null instance" : !in->err.empty() || !in->ctx ? in->err.c_str() : in->ctx->err.c_str();
}

const KosmosMoeInfo *kosmos_moe_instance_info(const KosmosMoeInstance *in) {
  return in && in->ctx ? &in->ctx->info : nullptr;
}

namespace kosmos_moe_impl {

// A saved state's pointers against its tier (every one 256-byte aligned) and rows.
static hipError_t saved_ok(KosmosMoeInstance *in, const KosmosMoeSaved *sv) {
  const auto al = [](const void *p) { return (uintptr_t)p % 256 == 0; };
  if (sv->tier < KOSMOS_MOE_SAVE_NONE || sv->tier > KOSMOS_MOE_SAVE_ALL || sv->rows < 1 || sv->rows > in->rows ||
      !sv->y || !sv->rec || (sv->tier >= KOSMOS_MOE_SAVE_XP && !sv->xp) ||
      (sv->tier == KOSMOS_MOE_SAVE_ALL && (!sv->g || !sv->a)) || !al(sv->xp) || !al(sv->g) || !al(sv->a) ||
      !al(sv->y) || !al(sv->rec)) {
    return set_error(in->err, hipErrorInvalidValue, "saved: tier in [0, 2], rows in [1, %d], y and rec set, xp (tier >="
                     " 1), g and a (tier 2) set, every pointer 256-byte aligned", in->rows);
  }
  return hipSuccess;
}

}

hipError_t kosmos_moe_train_forward(KosmosMoeInstance *in, const void *X, const void *W1, const void *W2,
                                    const int *topk_idx, const float *topk_w, void *out, const KosmosMoeSaved *saved,
                                    hipStream_t s) {
  using namespace kosmos_moe_impl;
  if (!in || !in->ctx) {
    return hipErrorInvalidValue;
  }
  in->err.clear();
  if (!X || !W1 || !W2 || !topk_idx || !topk_w || !out) {
    return set_error(in->err, hipErrorInvalidValue, "train_forward: X, W1, W2, topk_idx, topk_w and out must be set");
  }
  if (saved && saved_ok(in, saved) != hipSuccess) {
    return hipErrorInvalidValue;
  }
  KosmosMoe *c = in->ctx;
  const KosmosMoeConfig &g = in->cfg;
  const KosmosMoeFwd &f = c->f;
  const Gl A_gl((bf16 *)f.recv, nullptr, nullptr, (size_t)c->arena_rows + 1, (size_t)g.H);
  const TrainApiCall v = {saved ? (bf16 *)saved->xp : nullptr, saved ? (bf16 *)saved->g : nullptr,
                          saved ? (bf16 *)saved->a : nullptr, saved ? (bf16 *)saved->y : nullptr,
                          saved ? (uint8_t *)saved->rec : nullptr, in->ctr, in->seg, in->nseg, saved ? saved->tier : -1,
                          saved ? saved->rows : 0, (int)in->off_slot};
  const bf16 *x = (const bf16 *)X, *w1 = (const bf16 *)W1, *w2 = (const bf16 *)W2;
  bf16 *o = (bf16 *)out;
  if (g.k <= 4 && g.act) {
    moe_train_api_launch_fwd_ks<GluClamp10>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else if (g.k <= 4) {
    moe_train_api_launch_fwd_ks<SwiGLU>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else if (g.act) {
    moe_train_api_launch_fwd<GluClamp10>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  } else {
    moe_train_api_launch_fwd<SwiGLU>(f.burst, g.wgcap, s, A_gl, x, w1, w2, o, topk_idx, topk_w, f.t1, f.args, v);
  }
  c->forwarded = 1;
  const hipError_t e = hipGetLastError();
  return e == hipSuccess ? e : set_error(in->err, e, "train_forward launch: %s", hipGetErrorString(e));
}

hipError_t kosmos_moe_train_backward(KosmosMoeInstance *in, const void *dout, const void *X, const void *W1,
                                     const void *W2, const int *topk_idx, const float *topk_w,
                                     const KosmosMoeSaved *saved, void *dX, float *dprobs, void *dW1, void *dW2,
                                     int acc, hipStream_t s) {
  using namespace kosmos_moe_impl;
  if (!in || !in->ctx) {
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
  KosmosMoe *c = in->ctx;
  KosmosMoeWorkspace *w = in->ws;
  const KosmosMoeConfig &g = in->cfg;
  const Gl dY_gl((bf16 *)w->recv, nullptr, nullptr, (size_t)w->rows + 1, (size_t)g.H);
  const SingleCall call = {(const unsigned short *)dout, (const bf16 *)saved->g, (const bf16 *)W1, (const bf16 *)W2,
                           (bf16 *)dX, w->dtk, (bf16 *)dW1, (bf16 *)dW2, dprobs};
  const TrainApiBwdCall v = {(const bf16 *)saved->xp, (const bf16 *)saved->g, (const bf16 *)saved->a,
                             (const bf16 *)saved->y, (const bf16 *)X, (const uint8_t *)saved->rec, in->ctr, in->seg,
                             in->scnt, in->sep, in->nseg, (int)in->off_slot, acc};
  if (g.k <= 4 && g.act) {
    moe_train_api_launch_bwd_ks<GluClamp10>(g.wgcap, s, dY_gl, w->t, c->b.args, call, v);
  } else if (g.k <= 4) {
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
  if (!in || !in->ctx || !status) {
    return hipErrorInvalidValue;
  }
  in->err.clear();
  return kosmos_moe_plan_status(in->ctx, status);
}

#undef KM_TRY
