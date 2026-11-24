/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// KOSMOS MoE library (gfx950, EP=8): the bf16 MoE training forward and backward, one launch each.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

#include "../utils/kosmos_comm.h"

constexpr int KOSMOS_MOE_EP = 8;

// The experts' GLU activation of the [gate; up] fc1 output g, u: SwiGLU silu(g) u, or DeepSeek-V4's clamped SwiGLU
// silu(min(g, 10)) clamp(u, -10, 10) (TE clamped_swiglu_v2 with limit 10, alpha 1, offset 0).
constexpr int KOSMOS_MOE_ACT_SWIGLU         = 0;
constexpr int KOSMOS_MOE_ACT_SWIGLU_CLAMP10 = 1;

struct KosmosMoeConfig {
  int H;
  int I;
  int E_loc;
  int k;
  int T;
  int wgcap;
  int capacity_rows;
  KosmosCollective coll;
  int act;
};

struct KosmosMoeInfo {
  int Mpad, Mpad_max;
  int fwd_ncomm, fwd_fcomm, fwd_burst;
  int nt1, nt2;
  int bwd_cncomm_clamped, bwd_grid;
  int bwd_dncomm_all[KOSMOS_MOE_EP], bwd_cncomm_all[KOSMOS_MOE_EP];
  double bwd_dpush_gb, bwd_lega_tflop;
  double bwd_cpush_gb;
  double bwd_wg2_tiles, bwd_wg2_meank;
  int bwd_wg2run;
  int bwd_wg1run;
  int arena_rows;
};

// The training API: everything a backward reads from its forward lives in caller-owned buffers (KosmosMoeSaved), so any
// number of forwards may be in flight, backwards may come in any order, and a forward whose backward never runs leaves
// nothing behind. One instance per EP group and configuration (bf16, cfg.capacity_rows > 0); it owns the kernels'
// settings and the symmetric buffers of the call in progress, allocated once at create for cfg.capacity_rows rows and
// never re-allocated: a call over capacity runs in rounds inside its launch, keeps KOSMOS_MOE_SAVE_NONE and sets status
// bits. Every rank issues the same calls in the same order on one stream.
struct KosmosMoeInstance;

// Bytes of the symmetric buffer the comm passed to kosmos_moe_instance_create must hold, as kosmos_comm_info reports
// them; needed only when the caller allocates that buffer (kosmos_comm_wrap without a flag area: add
// kosmos_comm_flag_bytes(8)). With comm null the instance allocates it.
hipError_t kosmos_moe_instance_bytes(const KosmosMoeConfig *cfg, size_t *bytes);
// Collective and host-blocking. comm: null (the instance allocates with kosmos_comm_create) or a comm holding at least
// kosmos_moe_instance_bytes (e.g. kosmos_comm_wrap of a torch symmetric-memory tensor), kept by the caller until
// kosmos_moe_instance_destroy. Does all setup a first call would do, so later calls enqueue one kernel each and are
// capture-safe.
hipError_t kosmos_moe_instance_create(KosmosMoeInstance **inst, KosmosComm *comm, const KosmosMoeConfig *cfg);
hipError_t kosmos_moe_instance_destroy(KosmosMoeInstance *inst);
const char *kosmos_moe_instance_error(const KosmosMoeInstance *inst);

enum { KOSMOS_MOE_SAVE_NONE = 0, KOSMOS_MOE_SAVE_XP = 1, KOSMOS_MOE_SAVE_ALL = 2 };

struct KosmosMoeSavedBytes {
  size_t xp, g, a, y, route;
};
// Bytes of each saved buffer for saved capacity rows (1 <= rows <= the instance's capacity, cfg.capacity_rows rounded
// up to 256): xp, g, a hold a call whose busiest rank routes at most rows rows (every expert padded to 256).
hipError_t kosmos_moe_saved_bytes(const KosmosMoeConfig *cfg, int rows, KosmosMoeSavedBytes *bytes);

// One forward's saved state, caller-owned. Every pointer 256-byte aligned; tier and rows equal on every rank.
struct KosmosMoeSaved {
  int tier;     // the most the forward may keep: KOSMOS_MOE_SAVE_*
  int rows;     // the rows xp, g, a were sized for
  void *xp;     // may be null when tier < XP
  void *g;      // may be null when tier < ALL
  void *a;      // may be null when tier < ALL
  void *y;      // Y by token slot t k + j; always
  void *route;  // the route record: routing layout, receive slots, the tier kept, the call's status; always
};

// One launch. saved null: inference, nothing kept. The forward keeps saved->tier if its rows fit saved->rows and it
// runs in one pass, else nothing (KOSMOS_MOE_SAVE_NONE; status bit 32 if it asked for more), and records it in
// saved->route. X, W1, W2, topk_idx and topk_w must be unchanged at the backward.
hipError_t kosmos_moe_train_forward(KosmosMoeInstance *inst, const void *X, const void *W1, const void *W2,
                                    const int *topk_idx, const float *topk_w, void *out, const KosmosMoeSaved *saved,
                                    hipStream_t stream);
// One launch, the same saved as its forward (the buffers may have moved). acc: 0 dW1, dW2 bf16, overwritten; 1 fp32,
// added into (main_grad); 2 bf16, added into.
hipError_t kosmos_moe_train_backward(KosmosMoeInstance *inst, const void *dout, const void *X, const void *W1,
                                     const void *W2, const int *topk_idx, const float *topk_w,
                                     const KosmosMoeSaved *saved, void *dX, float *dprobs, void *dW1, void *dW2,
                                     int acc, hipStream_t stream);
// Host-blocking: OR of the status bits of every call since the last query; clears them. Errors 1 bad expert id,
// 2 duplicate; notes 8 ran in rounds, 32 kept nothing (in rounds, or rows past saved->rows).
hipError_t kosmos_moe_instance_status(KosmosMoeInstance *inst, int *status);
// The instance's derived settings.
const KosmosMoeInfo *kosmos_moe_instance_info(const KosmosMoeInstance *inst);
