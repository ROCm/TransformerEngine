/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// Link check: every MoE and communicator entry point resolves against libkosmos_moe.a alone.
#include "moe.h"

#include <cstdio>

static int no_allgather(void *, const void *, void *, size_t) {
  return 1;
}

int main() {
  struct Shape {
    const char *name;
    int H, I, E_loc, k;
  };
  const Shape shapes[] = {{"dsv3", 7168, 2048, 32, 8},   {"dsv4pro", 7168, 3072, 48, 6}, {"qwen235", 4096, 1536, 16, 8},
                          {"qwen30", 2048, 768, 16, 8},  {"kimi27", 7168, 2048, 48, 8},  {"glm52", 6144, 2048, 32, 8},
                          {"qwen35", 4096, 1024, 64, 10}, {"dsv3k1", 7168, 2048, 32, 1},  {"dsv3k2", 7168, 2048, 32, 2},
                          {"dsv3k4", 7168, 2048, 32, 4}};
  int n = 0;
  int iok = 0;
  for (const Shape &s : shapes) {
    KosmosMoeConfig cfg = {};
    cfg.H = s.H, cfg.I = s.I, cfg.E_loc = s.E_loc, cfg.k = s.k, cfg.T = 8192, cfg.wgcap = 256;
    cfg.coll = KosmosCollective{no_allgather, nullptr, 0, KOSMOS_MOE_EP};
    const int rows = 8 * 8192 * s.k / KOSMOS_MOE_EP + 256 * s.E_loc;
    // The size queries: at the capacity, refused past it and without one.
    cfg.capacity_rows = rows;
    size_t ib = 0;
    KosmosMoeSavedBytes svb = {};
    bool ia = kosmos_moe_instance_bytes(&cfg, &ib) == hipSuccess && ib > 0;
    ia = ia && kosmos_moe_saved_bytes(&cfg, rows, &svb) == hipSuccess && svb.xp && svb.g && svb.a && svb.y && svb.route;
    ia = ia && kosmos_moe_saved_bytes(&cfg, rows + 256, &svb) != hipSuccess;
    cfg.capacity_rows = 0;
    ia = ia && kosmos_moe_instance_bytes(&cfg, &ib) != hipSuccess;
    printf("%-8s instance bytes at %d rows: %zu  instance sizes %d\n", s.name, rows, ib, ia);
    iok += ia;
    n++;
  }
  KosmosMoeConfig bad = {};
  bad.H = 7168, bad.I = 2048, bad.E_loc = 32, bad.k = 8, bad.T = 8192, bad.wgcap = 256, bad.capacity_rows = 65536;
  bad.coll = KosmosCollective{no_allgather, nullptr, 0, KOSMOS_MOE_EP};
  KosmosMoeConfig b[6] = {bad, bad, bad, bad, bad, bad};
  b[0].k = 3;
  b[1].H = 7000;
  b[2].coll.nranks = 7;
  b[3].wgcap = 8;
  b[4].act = 2;
  b[5].E_loc = 65;
  size_t bytes = 0;
  int refused = 0;
  for (const KosmosMoeConfig &c : b) {
    refused += kosmos_moe_instance_bytes(&c, &bytes) != hipSuccess;
  }
  void *fns[] = {(void *)&kosmos_moe_instance_bytes,   (void *)&kosmos_moe_instance_create,
                 (void *)&kosmos_moe_instance_destroy, (void *)&kosmos_moe_instance_error,
                 (void *)&kosmos_moe_saved_bytes,      (void *)&kosmos_moe_train_forward,
                 (void *)&kosmos_moe_train_backward,   (void *)&kosmos_moe_instance_status,
                 (void *)&kosmos_moe_instance_info,    (void *)&kosmos_comm_create,
                 (void *)&kosmos_comm_wrap,            (void *)&kosmos_comm_reset,
                 (void *)&kosmos_comm_destroy,         (void *)&kosmos_comm_info,
                 (void *)&kosmos_comm_error,           (void *)&kosmos_comm_flag_bytes};
  const int nall = (int)(sizeof(fns) / sizeof(fns[0]));
  int nfn = 0;
  for (void *f : fns) {
    nfn += f != nullptr;
  }
  printf("instance sizes %d/%d, bad configs refused %d/6, entry points linked %d/%d\n", iok, n, refused, nfn, nall);
  return (iok == n && refused == 6 && nfn == nall) ? 0 : 1;
}
