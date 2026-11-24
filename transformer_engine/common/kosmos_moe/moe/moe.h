/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// KOSMOS MoE library (gfx950, EP=8): the MoE forward and backward, one launch each.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

#include "../utils/kosmos_comm.h"

constexpr int KOSMOS_MOE_EP = 8;

// Router score functions (Megatron's): softmax over the top k logits; sigmoid or sqrt(softplus) (DeepSeek-V4) scores,
// top k of score + expert bias, weights the scores normalized over the top k when k > 1; softmax over all experts,
// then top k, not renormalized (Grok, OLMoE; renormalized it is KOSMOS_ROUTER_SOFTMAX).
constexpr int KOSMOS_ROUTER_SOFTMAX      = 0;
constexpr int KOSMOS_ROUTER_SIGMOID      = 1;
constexpr int KOSMOS_ROUTER_SQRTSOFTPLUS = 2;
constexpr int KOSMOS_ROUTER_SOFTMAX_PRE  = 3;

// The experts' GLU activation of the [gate; up] fc1 output g, u: SwiGLU silu(g) u, or DeepSeek-V4's clamped SwiGLU
// silu(min(g, 10)) clamp(u, -10, 10) (TE clamped_swiglu_v2 with limit 10, alpha 1, offset 0).
constexpr int KOSMOS_MOE_ACT_SWIGLU         = 0;
constexpr int KOSMOS_MOE_ACT_SWIGLU_CLAMP10 = 1;

struct KosmosMoeRouterConfig {
  int on;
  int score;
  int groups, group_topk;
  int aux_groups;
  float aux_coeff;
  float scale;
  int force;
};

struct KosmosMoeConfig {
  int H;
  int I;
  int E_loc;
  int k;
  int T;
  int wgcap;
  int backward;
  int capacity_rows;
  size_t comm_offset;
  KosmosCollective coll;
  KosmosMoeRouterConfig router;
  int shared;
  int mxfp8;
  int fp8_combine;
  int act;
  int probs_in;
  size_t pool_bytes;
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
  double bwd_census_dmean, bwd_census_cmean;
  int bwd_census_dmax, bwd_census_cmax;
  long long disp_self, disp_rows, comb_self, comb_rows;
  int fwd_census_max;
  size_t g_bytes;
  int arena_rows;
  int plans;
};

struct KosmosMoe;
struct KosmosMoeWorkspace;

// Training keeps what a forward leaves for its backward in the workspace's pool, cfg.pool_bytes per rank (bf16: Xp, A,
// G; MXFP8: G, the column-wise A and with I < 1024 the column-wise Xp): a forward stores with G = KOSMOS_MOE_SAVE
// (null: not stored); the backward's G is not read.
#define KOSMOS_MOE_SAVE ((void *)1)

hipError_t kosmos_moe_validate(const KosmosMoeConfig *cfg);
// As kosmos_moe_validate; a refusal's reason goes to why (len bytes, NUL-terminated).
hipError_t kosmos_moe_validate_why(const KosmosMoeConfig *cfg, char *why, size_t len);
hipError_t kosmos_moe_arena_bytes(const KosmosMoeConfig *cfg, int rows, size_t *layer_bytes, size_t *shared_bytes);

hipError_t kosmos_moe_workspace_create(KosmosMoeWorkspace **ws, KosmosComm *comm, const KosmosMoeConfig *cfg);
hipError_t kosmos_moe_workspace_destroy(KosmosMoeWorkspace *ws);
// Collective and host-blocking (training): re-allocates the workspace's pool at the ranks' least bytes (at least
// one call's block; halved while the allocation fails) and re-points every context; the calls captured into a graph
// before keep the old pool's addresses. Refused on every rank, nothing changed, with hipErrorStreamCaptureUnsupported
// inside a stream capture and hipErrorNotReady while a forward's stored block awaits its backward. On an error the
// workspace may have no pool (its forwards are refused); kosmos_moe_workspace_error says why.
hipError_t kosmos_moe_pool_resize(KosmosMoeWorkspace *ws, size_t bytes);
// The pool's bytes per rank (0: not open), one call's block and (bf16) the Xp rows reserved for each layer still to
// store (both 0 before the first forward).
hipError_t kosmos_moe_pool_info(const KosmosMoeWorkspace *ws, size_t *bytes, size_t *head_bytes,
                                size_t *reserve_bytes);
// Since the previous query or capacity change: the largest rows a rank needed (as Mpad_max, also over capacity) in any
// call (0: none), the capacity and its largest value (what one call can need), the open contexts, how many set status 8
// (ran in rounds) or 4 (overflow: no call sets it any more, always 0), and how many of those in rounds needed more than
// their capacity (the others the pool forced into rounds).
struct KosmosMoeCapacityStat {
  int peak, rows, max_rows, nctx, rounds, overflow, cap_rounds;
};
// Collective and host-blocking, between calls: reads and clears each open context's status (kosmos_moe_plan_status
// still returns those bits). Refused inside a stream capture.
hipError_t kosmos_moe_capacity_query(KosmosMoeWorkspace *ws, KosmosMoeCapacityStat *stat);
// Collective and host-blocking: re-opens the workspace and its open contexts at the ranks' largest rows (rounded up to
// 256, at most what one call can need), its new capacity; graphs captured before keep the old buffers, a backward of a
// forward before it is refused. Refused as kosmos_moe_pool_resize. hipErrorOutOfMemory: an allocation failed on some
// rank and the old capacity is kept (re-opened, or at the contexts' next forward; kosmos_moe_workspace_error says).
hipError_t kosmos_moe_set_capacity(KosmosMoeWorkspace *ws, int rows);
// The device bytes that kosmos_moe_set_capacity allocates at rows rows (with the pool's least size).
hipError_t kosmos_moe_capacity_bytes(const KosmosMoeWorkspace *ws, int rows, size_t *bytes);
const char *kosmos_moe_workspace_error(const KosmosMoeWorkspace *ws);
hipError_t kosmos_moe_create(KosmosMoe **ctx, KosmosMoeWorkspace *ws);
// Collective: what a context's first device-routed forward does (arenas and tables at capacity_rows), so that its
// calls can be captured into a graph; a capture that still needs it fails with hipErrorStreamCaptureUnsupported.
hipError_t kosmos_moe_prepare(KosmosMoe *ctx);
hipError_t kosmos_moe_plan(KosmosMoe *ctx, const int *topk_idx, const float *topk_w);
// The OR over the context's calls: errors 1 bad expert id, 2 duplicate (4, over capacity, is no longer set: every
// forward over capacity runs in rounds); notes 8 ran in rounds, (training) 16 kept Xp only (bf16), 32 kept nothing (in
// rounds; 64, a single pass keeping nothing, is an error).
hipError_t kosmos_moe_plan_status(KosmosMoe *ctx, int *status);
hipError_t kosmos_moe_forward(KosmosMoe *ctx, const void *X, const void *W1, const void *W2, const int *topk_idx,
                              const float *topk_w, void *out, void *G, hipStream_t stream,
                              const hipEvent_t *ev = nullptr);
// With cfg.probs_in, X and X_scale are the MXFP8 of bf16(w X) (the probs applied before the quantization).
hipError_t kosmos_moe_forward_mxfp8(KosmosMoe *ctx, const void *X, const void *X_scale, const void *W1,
                                    const void *W1_scale, const void *W2, const void *W2_scale, const int *topk_idx,
                                    const float *topk_w, void *out, hipStream_t stream,
                                    const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_forward_mxfp8_train(KosmosMoe *ctx, const void *X, const void *W1, const void *W1_scale,
                                          const void *W2, const void *W2_scale, const int *topk_idx,
                                          const float *topk_w, void *out, void *G, hipStream_t stream,
                                          const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_mxfp8(KosmosMoe *ctx, const void *dout, const void *G, const void *X, const void *W1c,
                                     const void *W1c_scale, const void *W2c, const void *W2c_scale, void *dX,
                                     float *dprobs, void *dW1, void *dW2, int acc, hipStream_t stream,
                                     const hipEvent_t *ev = nullptr);
// With cfg.probs_in (Llama 4, k = 1: the forward computes out = sum_j Expert_j(bf16(w_j X)), unweighted) the backward
// also reads the forward's X (keep it unchanged until then): dX = bf16(w dXp) and dprobs = <dXp, X>.
hipError_t kosmos_moe_backward(KosmosMoe *ctx, const void *dout, const void *G, const void *W1, const void *W2,
                               void *dX, float *dprobs, void *dW1, void *dW2, int acc, hipStream_t stream,
                               const hipEvent_t *ev = nullptr);
// The shared expert (cfg.shared): fc1 Ws1 [2I][H] in Megatron's [gate; up] row layout, fc2 Ws2 [H][I], run in the same
// launch; out = routed + sgate * shared. Gs: [T][2I] bf16 pre-activations ([gate; up] columns) for the backward (null:
// none). sgate: [T] fp32 per-token gates, 16-byte aligned (Qwen3-Next / Qwen3.5: sigmoid(X w_sg), computed by the
// caller; null: ungated); kept until the backward.
hipError_t kosmos_moe_forward_shared(KosmosMoe *ctx, const void *X, const void *W1, const void *W2, const void *Ws1,
                                     const void *Ws2, const int *topk_idx, const float *topk_w, void *out, void *G,
                                     void *Gs, const float *sgate, hipStream_t stream, const hipEvent_t *ev = nullptr);
// MXFP8 training with the shared expert: the routed operands as forward_mxfp8_train / backward_mxfp8, the shared
// expert's as forward_shared / backward_shared with fc1 and fc2 MXFP8: Ws1 [2I][H] (rows [gate; up]) and Ws2 [H][I]
// row-wise (scales [2I][H/32], [H][I/32]), Ws1c and Ws2c column-wise (scales [2I/32][H], [H/32][I]). Gs is required.
hipError_t kosmos_moe_forward_shared_mxfp8_train(KosmosMoe *ctx, const void *X, const void *W1, const void *W1_scale,
                                                 const void *W2, const void *W2_scale, const void *Ws1,
                                                 const void *Ws1_scale, const void *Ws2, const void *Ws2_scale,
                                                 const int *topk_idx, const float *topk_w, void *out, void *G, void *Gs,
                                                 const float *sgate, hipStream_t stream,
                                                 const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_shared_mxfp8(KosmosMoe *ctx, const void *dout, const void *G, const void *Gs,
                                            const void *X, const void *W1c, const void *W1c_scale, const void *W2c,
                                            const void *W2c_scale, const void *Ws1c, const void *Ws1c_scale,
                                            const void *Ws2c, const void *Ws2c_scale, void *dX, float *dprobs,
                                            void *dW1, void *dW2, void *dWs1, void *dWs2, int acc,
                                            const float *sgate, float *dsgate, hipStream_t stream,
                                            const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_forward_router_shared(KosmosMoe *ctx, const void *X, const void *Wr, const float *bias,
                                            const float *logit_bias, const void *W1, const void *W2, const void *Ws1,
                                            const void *Ws2, void *out, void *G, void *Gs, const float *sgate,
                                            float *aux_loss, int *tpe, const unsigned *seed, hipStream_t stream,
                                            const hipEvent_t *ev = nullptr);
// The backward of a forward_shared: dX also gets the shared expert's dX. acc = 0: dWs1 [2I][H] and dWs2 [H][I] are bf16
// and overwritten; acc = 1: they are fp32 (main_grad) and the gradients are added into them; acc = 2: bf16, added into.
// sgate: the forward's (null if it was ungated); dsgate: [T] fp32, the gradient in the gates, <dout_t, shared_t> (null:
// not computed). The gate weight's gradient and its dX term (dsgate sgate (1 - sgate) through X w_sg) are the caller's.
hipError_t kosmos_moe_backward_shared(KosmosMoe *ctx, const void *dout, const void *G, const void *Gs, const void *X,
                                      const void *W1, const void *W2, const void *Ws1, const void *Ws2, void *dX,
                                      float *dprobs, void *dW1, void *dW2, void *dWs1, void *dWs2, int acc,
                                      const float *sgate, float *dsgate, hipStream_t stream,
                                      const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_router_shared(KosmosMoe *ctx, const void *dout, const void *G, const void *Gs,
                                             const void *X, const void *W1, const void *W2, const void *Ws1,
                                             const void *Ws2, const void *Wr, const float *aux_scale, void *dX,
                                             void *dW1, void *dW2, void *dWr, float *dlogit_bias, void *dWs1,
                                             void *dWs2, int acc, const float *sgate, float *dsgate, hipStream_t stream,
                                             const hipEvent_t *ev = nullptr);
// The fused router: bias [8 E_loc] fp32, the expert bias (sigmoid and sqrtsoftplus: added to the scores for the
// selection only; may be null); logit_bias [8 E_loc] fp32, the router linear's bias (added to the logits; may be null).
// The backward's dlogit_bias [8 E_loc] fp32 (null: not computed) is its gradient, the logits' gradient summed over T;
// its aux_scale is a device fp32 scalar (the aux loss's gradient scale), read when the kernel runs.
hipError_t kosmos_moe_forward_router(KosmosMoe *ctx, const void *X, const void *Wr, const float *bias,
                                     const float *logit_bias, const void *W1, const void *W2, void *out, void *G,
                                     float *aux_loss, int *tpe, const unsigned *seed, hipStream_t stream,
                                     const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_router(KosmosMoe *ctx, const void *dout, const void *G, const void *X, const void *W1,
                                      const void *W2, const void *Wr, const float *aux_scale, void *dX, void *dW1,
                                      void *dW2, void *dWr, float *dlogit_bias, int acc, hipStream_t stream,
                                      const hipEvent_t *ev = nullptr);
// MXFP8 training with the fused router: the router (logits, scores, routing, aux loss; its backward's dlogits, dWr and
// dX term) as forward_router / backward_router, in bf16; the experts as forward_mxfp8_train / backward_mxfp8, the
// shared expert as forward_shared_mxfp8_train / backward_shared_mxfp8.
hipError_t kosmos_moe_forward_router_mxfp8_train(KosmosMoe *ctx, const void *X, const void *Wr, const float *bias,
                                                 const float *logit_bias, const void *W1, const void *W1_scale,
                                                 const void *W2, const void *W2_scale, void *out, void *G,
                                                 float *aux_loss, int *tpe, const unsigned *seed, hipStream_t stream,
                                                 const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_router_mxfp8(KosmosMoe *ctx, const void *dout, const void *G, const void *X,
                                            const void *W1c, const void *W1c_scale, const void *W2c,
                                            const void *W2c_scale, const void *Wr, const float *aux_scale, void *dX,
                                            void *dW1, void *dW2, void *dWr, float *dlogit_bias, int acc,
                                            hipStream_t stream,
                                            const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_forward_router_shared_mxfp8_train(KosmosMoe *ctx, const void *X, const void *Wr,
                                                        const float *bias, const float *logit_bias, const void *W1,
                                                        const void *W1_scale, const void *W2, const void *W2_scale,
                                                        const void *Ws1, const void *Ws1_scale, const void *Ws2,
                                                        const void *Ws2_scale, void *out, void *G, void *Gs,
                                                        const float *sgate, float *aux_loss, int *tpe,
                                                        const unsigned *seed, hipStream_t stream,
                                                        const hipEvent_t *ev = nullptr);
hipError_t kosmos_moe_backward_router_shared_mxfp8(KosmosMoe *ctx, const void *dout, const void *G, const void *Gs,
                                                   const void *X, const void *W1c, const void *W1c_scale,
                                                   const void *W2c, const void *W2c_scale, const void *Ws1c,
                                                   const void *Ws1c_scale, const void *Ws2c, const void *Ws2c_scale,
                                                   const void *Wr, const float *aux_scale, void *dX, void *dW1,
                                                   void *dW2, void *dWr, float *dlogit_bias, void *dWs1, void *dWs2,
                                                   int acc, const float *sgate, float *dsgate, hipStream_t stream,
                                                   const hipEvent_t *ev = nullptr);
// nmat bf16 matrices [rows][cols] to MXFP8 (the weights of the _mxfp8 calls): q [nmat][rows][cols] e4m3 and e8m0
// scales s, row-wise [nmat][rows][cols / 32] (col = 0) or column-wise [nmat][rows / 32][cols] (col = 1).
hipError_t kosmos_moe_quantize_mxfp8(const void *W, int nmat, int rows, int cols, int col, void *q, void *s,
                                     hipStream_t stream);
hipError_t kosmos_moe_destroy(KosmosMoe *ctx);

// Agreed release (training): a stored forward whose backward will not run (a caller dropped it) gives its pool block
// back once every rank released it. The ranks agree inside their forwards (a mask published with the counts); each
// forward's agreed mask is read on the host KOSMOS_MOE_REL_LAG forwards later (waiting for that forward), in launch
// order, so the states below change alike on every rank at the same point of the call sequence. Off for calls captured
// into a graph.
enum { KOSMOS_MOE_REL_IDLE = 0, KOSMOS_MOE_REL_STORED = 1, KOSMOS_MOE_REL_PENDING = 2, KOSMOS_MOE_REL_RECLAIMED = 3 };
constexpr int KOSMOS_MOE_REL_LAG = 4;
// Local, host-only, thread-safe: the context's stored forward gets no backward (STORED to PENDING; otherwise nothing;
// nothing either for a context with a forward captured into a graph, whose saved state stays until its backward).
hipError_t kosmos_moe_release(KosmosMoe *ctx);
// The context's state (KOSMOS_MOE_REL_*): RECLAIMED once every rank released it; its next forward makes it STORED or
// IDLE again. A backward of a PENDING or RECLAIMED context is refused.
hipError_t kosmos_moe_release_state(const KosmosMoe *ctx, int *state);
// Host-blocking: reads the agreed masks of every forward launched on the workspace (waits for them).
hipError_t kosmos_moe_release_sync(KosmosMoeWorkspace *ws);

const KosmosMoeInfo *kosmos_moe_info(const KosmosMoe *ctx);
const char *kosmos_moe_error(const KosmosMoe *ctx);
