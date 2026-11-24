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
  int ok = 0, n = 0, rok = 0, iok = 0;
  for (const Shape &s : shapes) {
    KosmosMoeConfig cfg = {};
    cfg.H = s.H, cfg.I = s.I, cfg.E_loc = s.E_loc, cfg.k = s.k, cfg.T = 8192, cfg.wgcap = 256, cfg.backward = 1;
    cfg.coll = KosmosCollective{no_allgather, nullptr, 0, KOSMOS_MOE_EP};
    size_t lb = 0, sb = 0;
    const int rows = 8 * 8192 * s.k / KOSMOS_MOE_EP + 256 * s.E_loc;
    cfg.pool_bytes = (size_t)2 * rows * (2 * (size_t)s.H + 6 * (size_t)s.I);
    const bool v = kosmos_moe_validate(&cfg) == hipSuccess;
    const bool a = kosmos_moe_arena_bytes(&cfg, rows, &lb, &sb) == hipSuccess;
    KosmosMoeWorkspace *ws = nullptr;
    KosmosMoe *ctx[2] = {nullptr, nullptr};
    bool c = kosmos_moe_workspace_create(&ws, nullptr, &cfg) == hipSuccess && ws;
    c = c && kosmos_moe_create(&ctx[0], ws) == hipSuccess && kosmos_moe_create(&ctx[1], ws) == hipSuccess;
    size_t cb[2] = {0, 0};
    c = c && kosmos_moe_capacity_bytes(ws, rows, &cb[0]) == hipSuccess;
    c = c && kosmos_moe_capacity_bytes(ws, 2 * rows, &cb[1]) == hipSuccess && cb[0] && cb[1] > cb[0];
    c = c && kosmos_moe_workspace_destroy(ws) == hipErrorInvalidValue;
    c = c && kosmos_moe_destroy(ctx[0]) == hipSuccess && kosmos_moe_destroy(ctx[1]) == hipSuccess;
    c = c && kosmos_moe_workspace_destroy(ws) == hipSuccess;
    KosmosMoeConfig rs = cfg, rx = cfg;
    rs.router = KosmosMoeRouterConfig{1, KOSMOS_ROUTER_SIGMOID, 8, s.k > 1 ? s.k / 2 : 1, 2, 1e-2f, 1.0f, 1};
    rx.router = KosmosMoeRouterConfig{1, KOSMOS_ROUTER_SOFTMAX, 0, 0, 1, 1e-3f, 1.0f, 0};
    const int r = (kosmos_moe_validate(&rs) == hipSuccess) + (kosmos_moe_validate(&rx) == hipSuccess);
    // The training API's size queries: at the capacity, refused past it and without one.
    KosmosMoeConfig ic = cfg;
    ic.capacity_rows = rows;
    size_t ib = 0;
    KosmosMoeSavedBytes svb = {};
    bool ia = kosmos_moe_instance_bytes(&ic, &ib) == hipSuccess && ib > lb + sb;
    ia = ia && kosmos_moe_saved_bytes(&ic, rows, &svb) == hipSuccess && svb.xp && svb.g && svb.a && svb.y && svb.rec;
    ia = ia && kosmos_moe_saved_bytes(&ic, rows + 256, &svb) != hipSuccess;
    ic.capacity_rows = 0;
    ia = ia && kosmos_moe_instance_bytes(&ic, &ib) != hipSuccess;
    printf("%-8s validate %d  arena bytes at %d rows: layer %zu shared %zu  create/destroy %d  router configs %d/2"
           "  instance sizes %d\n", s.name, v, rows, lb, sb, c, r, ia);
    ok += v && a && c;
    rok += r;
    iok += ia;
    n++;
  }
  KosmosMoeConfig bad = {};
  bad.H = 7168, bad.I = 2048, bad.E_loc = 32, bad.k = 8, bad.T = 8192, bad.wgcap = 256, bad.backward = 1;
  bad.coll = KosmosCollective{no_allgather, nullptr, 0, KOSMOS_MOE_EP};
  bad.pool_bytes = (size_t)1 << 34;
  KosmosMoeConfig b[6] = {bad, bad, bad, bad, bad, bad};
  b[0].k = 3;
  b[1].H = 7000;
  b[2].coll.nranks = 7;
  b[3].wgcap = 8;
  b[4].comm_offset = 100;
  b[5].pool_bytes = (size_t)1 << 50;
  int refused = 0;
  for (const KosmosMoeConfig &c : b) {
    refused += kosmos_moe_validate(&c) != hipSuccess;
  }
  bad.router = KosmosMoeRouterConfig{1, KOSMOS_ROUTER_SIGMOID, 8, 4, 2, 1e-2f, 1.0f, 1};
  KosmosMoeConfig rb[5] = {bad, bad, bad, bad, bad};
  rb[0].router.score = 4;
  rb[1].router.groups = 7;
  rb[2].router.aux_groups = 5;
  rb[3].T = 8000;
  rb[4].k = 2;
  int rrefused = 0;
  for (const KosmosMoeConfig &c : rb) {
    rrefused += kosmos_moe_validate(&c) != hipSuccess;
  }
  void *fns[] = {(void *)&kosmos_moe_validate,         (void *)&kosmos_moe_arena_bytes,
                 (void *)&kosmos_moe_workspace_create, (void *)&kosmos_moe_workspace_destroy,
                 (void *)&kosmos_moe_create,           (void *)&kosmos_moe_plan,
                 (void *)&kosmos_moe_plan_status,      (void *)&kosmos_moe_forward,
                 (void *)&kosmos_moe_forward_mxfp8,    (void *)&kosmos_moe_backward,
                 (void *)&kosmos_moe_forward_router,   (void *)&kosmos_moe_backward_router,
                 (void *)&kosmos_moe_destroy,          (void *)&kosmos_moe_info,
                 (void *)&kosmos_moe_error,            (void *)&kosmos_moe_forward_mxfp8_train,
                 (void *)&kosmos_moe_backward_mxfp8,   (void *)&kosmos_moe_quantize_mxfp8,
                 (void *)&kosmos_comm_create,          (void *)&kosmos_comm_wrap,
                 (void *)&kosmos_comm_reset,           (void *)&kosmos_comm_destroy,
                 (void *)&kosmos_comm_info,            (void *)&kosmos_comm_error,
                 (void *)&kosmos_comm_flag_bytes,      (void *)&kosmos_moe_prepare,
                 (void *)&kosmos_moe_pool_resize,      (void *)&kosmos_moe_pool_info,
                 (void *)&kosmos_moe_workspace_error,  (void *)&kosmos_moe_set_capacity,
                 (void *)&kosmos_moe_capacity_query,   (void *)&kosmos_moe_capacity_bytes,
                 (void *)&kosmos_moe_instance_bytes,   (void *)&kosmos_moe_instance_create,
                 (void *)&kosmos_moe_instance_destroy, (void *)&kosmos_moe_instance_error,
                 (void *)&kosmos_moe_saved_bytes,      (void *)&kosmos_moe_train_forward,
                 (void *)&kosmos_moe_train_backward,   (void *)&kosmos_moe_instance_status,
                 (void *)&kosmos_moe_instance_info};
  const int nall = (int)(sizeof(fns) / sizeof(fns[0]));
  int nfn = 0;
  for (void *f : fns) {
    nfn += f != nullptr;
  }
  printf("shapes accepted %d/%d, router configs accepted %d/%d, bad configs refused %d/6, bad router configs refused"
         " %d/5, instance sizes %d/%d, entry points linked %d/%d\n", ok, n, rok, 2 * n, refused, rrefused, iok, n, nfn,
         nall);
  return (ok == n && rok == 2 * n && refused == 6 && rrefused == 5 && iok == n && nfn == nall) ? 0 : 1;
}
