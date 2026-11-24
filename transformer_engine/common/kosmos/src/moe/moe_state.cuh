/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The training instance's host state (not public API); include after the device sections.
#pragma once

#include <string>

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
  kittens::bf16 *recv, *comb;
  int *sig;
  int ncomm, fcomm, burst;
  unsigned *ep;
  int *cnt, *roff, *gather, *gid, *startby, *cntby, *sendslot, *lbsrc, *owed, *rbhi, *sbase;
  int *tiledone, *adone, *t1, *t2, *blkctr;
  hk_moe::RbBlk *rbs;
  hk_moe::TileDesc *q1, *q2;
  hk_moe::MoeFwdArgs *args;
  kittens::bf16 *act;
  float *wt;
};

struct KosmosMoeBwd {
  int dncomm, cncomm, wg2run, wg1run;
  hk_moe::SingleArgs *args;
  hk_moe::SendBlk *blks;
  hk_comm::CCombBlk *cblks;
  int *cntbysrc, *startbysrc;
  float *sendw;
};

struct KosmosMoePlan {
  int *hist, *pk, *status;
  hk_moe::MoePlanRec *rec;
};

// What one call needs only while it runs: the backward's peer-written arena and counters, dG, Y, which the forward
// writes and the backward reuses as dXp, the slots' dtopkw, and the backward's recomputed A of a forward that kept less
// than ALL. Its arena is the instance's comm from byte 0, the context's forward arena from inst_foff.
struct KosmosMoeWorkspace {
  KosmosMoeConfig cfg;
  KosmosComm *comm;
  int rows;
  unsigned *wep;
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
  kittens::bf16 *xrecv, *xrecvp[hk_moe::MOE_EP];
  int *xsigp[hk_moe::MOE_EP];
  kittens::bf16 *ascr;
  int *rgdone, *rinfo;
  size_t inst_foff;
};

struct KosmosMoe {
  KosmosMoeConfig cfg;
  KosmosMoeInfo info;
  std::string err;
  int arena_rows;
  KosmosMoeWorkspace *ws;
  // Its backward's forward arguments for the re-plan of a call in rounds (a copy of f.args whose count slots and epochs
  // are staged from the call's record).
  hk_moe::MoeFwdArgs *replan_args;
  KosmosMoeFwd f;
  KosmosMoeBwd b;
  KosmosMoePlan p;
};

// The training API's instance: a workspace and one context (its scratch) on one comm, the route record's layout and
// table segments, and the copy and re-plan state its kernels share.
struct KosmosMoeInstance {
  KosmosMoeConfig cfg;
  KosmosComm *comm;
  int own_comm;
  int rows;
  KosmosMoeWorkspace *ws;
  KosmosMoe *scratch;
  int *ctr;
  hk_moe::RouteSeg *seg;
  int nseg;
  size_t off_slot, route_bytes;
  int *scnt;
  unsigned *sep;
  std::string err;
};
