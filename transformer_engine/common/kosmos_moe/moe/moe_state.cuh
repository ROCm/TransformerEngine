/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The MoE library's contexts and their shared workspace (not public API); include after the device sections.
#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "moe.h"

struct KosmosMoeArena {
  void *base;
  size_t bytes;
};

struct KosmosMoeFwd {
  KosmosMoeArena arena;
  size_t comb_bytes;
  hk_moe::DispPeers dpeers;
  hk_moe::LegBPeers lpeers;
  int *cnts[hk_moe::MOE_EP], *cflag[hk_moe::MOE_EP], *slot[hk_moe::MOE_EP];
  int *rmask[hk_moe::MOE_EP];
  kittens::bf16 *recv, *comb;
  int *sig;
  int Mpad, nrow, nrb, nt1, nt2, ncomm, fcomm, burst;
  unsigned *ep;
  int *cnt, *roff, *gather, *gid, *startby, *cntby, *sendslot, *lbsrc, *owed, *rbhi, *sbase;
  int *tiledone, *adone, *t1, *t2, *blkctr;
  hk_moe::RbBlk *rbs;
  hk_moe::TileDesc *q1, *q2;
  hk_moe::MoeFwdArgs *args;
  kittens::bf16 *act;
  float *wt;
  kittens::bf16 *sact;
  int *sctr;
  uint8_t *xc, *xcp[hk_moe::MOE_EP], *xshp[hk_moe::MOE_EP];
  size_t xcs_off;
  uint8_t *sactc, *sactcs;
};

struct KosmosMoeBwd {
  int dncomm, cncomm, wg2run, wg1run;
  hk_moe::SingleArgs *args;
  hk_moe::SendBlk *blks;
  hk_comm::CCombBlk *cblks;
  int *cntbysrc, *startbysrc;
  float *sendw;
};

// The fused router's per-call scratch: the forward's aux partials, the backward's dlogits (MXFP8: also transposed),
// dX and counters.
struct KosmosMoeWsRouter {
  float *psum, *dlbp;
  int *pcnt, *rid, *bctr, *rpdone, *rdgdone;
  kittens::bf16 *dl, *dxr, *dlt;
};

struct KosmosMoePlan {
  int *hist, *pk, *status, *idx;
  float *w;
  hk_moe::MoePlanRec *rec;
};

// What one call needs only while it runs, shared by every context of the workspace: the backward's peer-written arena
// and counters, dG, Y, which the forward writes and the backward reuses as dXp, and the slots' dtopkw and expert ids.
// MXFP8 training also keeps here the forward's receive arena and row-wise A (the shared expert's too), and for I >= 1024
// the column-wise Xp; training keeps every context's saved state in the pool.
struct KosmosMoeWorkspace {
  KosmosMoeConfig cfg;
  KosmosComm *comm;
  int own_comm;
  int rows;
  unsigned *wep;
  std::vector<KosmosMoe *> ctxs;
  KosmosMoeArena arena;
  size_t off_sig, off_comb, off_csig;
  hk_moe::DispPeers dpeers;
  hk_moe::LegBPeers cpeers;
  kittens::bf16 *recv;
  int *sig;
  int sigb;
  int *t, *tdone, *dgdone;
  kittens::bf16 *y, *dg;
  float *dtk;
  KosmosMoeWsRouter r;
  kittens::bf16 *sbuf, *sdg;
  float *spart, *sdsp;
  int *sctr;
  uint8_t *mxq, *dyc, *dycs, *dgc, *dgcs;
  int *qdy, *qfx;
  kittens::bf16 *xrecv, *xrecvp[hk_moe::MOE_EP], *act;
  int *xsigp[hk_moe::MOE_EP];
  uint8_t *xc, *xcp[hk_moe::MOE_EP], *xshp[hk_moe::MOE_EP];
  size_t xcs_off;
  // MXFP8 training's shared expert: dOut row-wise and column-wise, X and dGs column-wise, the forward's row-wise A.
  uint8_t *sdoq, *sdoc, *sdocs, *sxc, *sxcs, *sdgc, *sdgcs;
  kittens::bf16 *sact;
  // Training's pool of prows rows of 2H bytes (its own comm: peers write the Xp rows, MXFP8 with I < 1024 the
  // column-wise Xp), and its device state (the next base, the slots in use, {base, xtop, top} per context slot).
  KosmosComm *pcomm;
  kittens::bf16 *xpool, *xpoolp[hk_moe::MOE_EP];
  // A layer that kept Xp only: its backward's recomputed A and its row blocks' counts. MXFP8 training in rounds: the
  // landed X rows quantized row-wise and the count per quantization item (rgdone, rinfo: as bf16's).
  kittens::bf16 *ascr;
  int *rgdone, *rinfo;
  uint8_t *xq8;
  int *qx;
  int *pool;
  int prows;
  // Pinned host words for kosmos_moe_capacity_query, hstage_n of them.
  int *hstage;
  size_t hstage_n;
  // Agreed release (kosmos_moe_release): the slots this rank released (a bit per pool slot), and the forwards launched
  // whose agreed masks are still to be read, oldest first (read in launch order, so alike on every rank).
  // rel_next: the next forward launch's sequence number; a release is in the masks of the launches from it on.
  std::mutex rel_mu;
  unsigned rel_pending[hk_moe::MOE_RM_WORDS];
  unsigned long long rel_next;
  // A failure after a forward was enqueued or inside a drain leaves this rank's view of the agreements incomplete: the
  // workspace refuses further forwards until it is destroyed.
  int rel_broken;
  struct RelCall {
    KosmosMoe *c;
    unsigned epoch;
    unsigned long long seq;
    hipEvent_t ev;
  };
  std::deque<RelCall> rel_calls;
  std::vector<hipEvent_t> rel_events;
  std::string err;
};

// The fused router's per-layer state: what a forward leaves for its backward, and the forward's counters.
struct KosmosMoeRouter {
  float *logits;
  int *cnt, *rbdone;
  unsigned *seedv;
  float aux_c;
  int routed;
};

struct KosmosMoe {
  KosmosMoeConfig cfg;
  KosmosMoeInfo info;
  std::string err;
  int planned, forwarded;
  // Status bits kosmos_moe_capacity_query read and cleared, for kosmos_moe_plan_status.
  int status_seen;
  int fwd_mx;
  const void *fwd_g, *fwd_gs;
  const float *fwd_sg;
  const void *fwd_x;
  // MXFP8 training: the forward's row-wise W1 and its scales (a backward in rounds recomputes GEMM1).
  const void *fwd_w1, *fwd_w1s;
  int arena_rows;
  int pslot;
  // Agreed release: the pinned rings (host and device addresses), the forwards launched since the epoch was reset, and
  // the saved state's owner (KOSMOS_MOE_REL_*).
  unsigned *rel_in, *rel_out, *rel_in_d, *rel_out_d;
  unsigned fwd_epochs;
  int rel_state;
  // The first launch whose mask carries this context's latest release: an agreement read from an earlier launch is
  // about a release before it (the context since reused), and leaves the context alone.
  unsigned long long rel_seq;
  // Off for good once a forward of the context was captured into a graph (its replays find no matching ring entry).
  int rel_off;
  KosmosComm *comm;
  KosmosMoeWorkspace *ws;
  std::vector<int> pool_cnt;
  KosmosMoeFwd f;
  KosmosMoeBwd b;
  KosmosMoePlan p;
  KosmosMoeRouter r;
};
