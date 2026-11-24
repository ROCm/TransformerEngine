/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// nvte_moe_*: the fused expert-parallel MoE (KOSMOS kernels in common/kosmos_moe) behind TE's C API.

#include <transformer_engine/moe.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../common.h"
#include "../util/logging.h"

#ifdef USE_KOSMOS_MOE

#include "../kosmos_moe/moe/moe.h"

struct NVTEMoeWorkspace_ {
  KosmosMoeWorkspace *ws = nullptr;
  KosmosMoeConfig kcfg{};
  NVTEMoeConfig cfg{};
  KosmosAllocator alloc{};  // the comm's allocator (unset: hipMalloc)
  int nlayers = 0;
};

struct NVTEMoeLayer_ {
  KosmosMoe *ctx = nullptr;
  NVTEMoeWorkspace_ *ws = nullptr;
};

namespace {

// Every KOSMOS call of a workspace allocates through its allocator (set on the calling thread for the call).
class AllocScope {
 public:
  explicit AllocScope(const NVTEMoeWorkspace_ *w)
      : prev_(kosmos_set_thread_allocator(w && w->alloc.alloc ? &w->alloc : nullptr)) {}
  ~AllocScope() { kosmos_set_thread_allocator(prev_); }
  AllocScope(const AllocScope &) = delete;
  AllocScope &operator=(const AllocScope &) = delete;

 private:
  const KosmosAllocator *prev_;
};

using transformer_engine::DType;
using transformer_engine::Tensor;

constexpr size_t kConfigMinSize = offsetof(NVTEMoeConfig, router_config) + sizeof(NVTEMoeRouterConfig);
constexpr size_t kCommMinSize = offsetof(NVTEMoeComm, free) + sizeof(void *);
constexpr size_t kFwdArgsMinSize = offsetof(NVTEMoeForwardArgs, save_for_backward) + sizeof(int);
constexpr size_t kBwdArgsMinSize = offsetof(NVTEMoeBackwardArgs, wgrad_accumulate) + sizeof(int);

// A caller's versioned struct in the current layout: fields it did not provide are zero.
template <typename S>
S normalize(const S *user, size_t min_size, const char *name) {
  NVTE_CHECK(user != nullptr, name, " must not be null");
  const size_t want = user->struct_size == 0 ? min_size : user->struct_size;
  NVTE_CHECK(want >= min_size, name, ".struct_size (", user->struct_size, ") is below the minimum ", min_size);
  S s{};
  std::memcpy(&s, user, std::min(want, sizeof(S)));
  s.struct_size = sizeof(S);
  return s;
}

// Stands in for the EP group's all-gather where a configuration is only validated; never called.
int no_allgather(void *, const void *, void *, size_t) {
  return 1;
}

KosmosMoeConfig to_kosmos(const NVTEMoeConfig &c, const NVTEMoeComm *comm) {
  KosmosMoeConfig k;
  std::memset(&k, 0, sizeof(k));
  k.H = c.hidden_size;
  k.I = c.ffn_hidden_size;
  k.E_loc = c.num_local_experts;
  k.k = c.topk;
  k.T = c.num_tokens;
  k.wgcap = c.wgcap;
  k.backward = c.training != 0;
  k.capacity_rows = c.capacity_rows;
  k.pool_bytes = c.pool_bytes;
  k.shared = c.shared_expert != 0;
  k.mxfp8 = c.mxfp8 != 0;
  k.fp8_combine = c.fp8_combine != 0;
  k.act = c.activation;
  k.probs_in = c.probs_on_input != 0;
  if (c.router) {
    k.router.on = 1;
    k.router.score = c.router_config.score;
    k.router.groups = c.router_config.groups;
    k.router.group_topk = c.router_config.group_topk;
    k.router.aux_groups = c.router_config.aux_groups;
    k.router.aux_coeff = c.router_config.aux_coeff;
    k.router.scale = c.router_config.scale;
  }
  if (comm) {
    k.coll.allgather = comm->allgather;
    k.coll.ctx = comm->ctx;
    k.coll.rank = comm->rank;
    k.coll.nranks = comm->nranks;
  } else {
    k.coll.allgather = no_allgather;
    k.coll.nranks = NVTE_MOE_EP_SIZE;
  }
  return k;
}

const Tensor *get(NVTETensor t) {
  return t ? transformer_engine::convertNVTETensor(t) : nullptr;
}

bool present(const Tensor *t) {
  return t != nullptr && (t->data.dptr != nullptr || t->columnwise_data.dptr != nullptr);
}

// Row-wise data of a plain (or row-wise MXFP8) tensor of numel n and dtype dt; null when absent and optional.
void *rowwise(NVTETensor h, const char *name, DType dt, size_t n, bool required) {
  const Tensor *t = get(h);
  if (!present(t)) {
    NVTE_CHECK(!required, "nvte_moe: ", name, " is required");
    return nullptr;
  }
  NVTE_CHECK(t->data.dptr != nullptr, "nvte_moe: ", name, " has no row-wise data");
  NVTE_CHECK(t->data.dtype == dt, "nvte_moe: ", name, " must be ", transformer_engine::to_string(dt), ", got ",
             transformer_engine::to_string(t->data.dtype));
  NVTE_CHECK(t->data.numel() == n, "nvte_moe: ", name, " must have ", n, " elements, got ",
             t->data.numel());
  return t->data.dptr;
}

struct Mx {
  void *data = nullptr;
  void *scale = nullptr;
};

// MXFP8 data and unswizzled e8m0 scales (one per 32 elements along the row, or along the column when columnwise).
Mx mxfp8(NVTETensor h, const char *name, size_t n, bool columnwise, bool required) {
  const Tensor *t = get(h);
  if (!present(t)) {
    NVTE_CHECK(!required, "nvte_moe: ", name, " is required");
    return {};
  }
  NVTE_CHECK(t->scaling_mode == NVTE_MXFP8_1D_SCALING, "nvte_moe: ", name, " must be an MXFP8 tensor");
  NVTE_CHECK(!t->with_gemm_swizzled_scales, "nvte_moe: ", name, " must have unswizzled scales");
  const transformer_engine::SimpleTensor &d = columnwise ? t->columnwise_data : t->data;
  const transformer_engine::SimpleTensor &s = columnwise ? t->columnwise_scale_inv : t->scale_inv;
  const char *usage = columnwise ? "column-wise" : "row-wise";
  NVTE_CHECK(d.dptr != nullptr && s.dptr != nullptr, "nvte_moe: ", name, " has no ", usage, " data or scales");
  NVTE_CHECK(d.dtype == DType::kFloat8E4M3, "nvte_moe: ", name, " must be e4m3");
  NVTE_CHECK(d.numel() == n, "nvte_moe: ", name, " must have ", n, " elements, got ", d.numel());
  NVTE_CHECK(s.numel() == n / 32, "nvte_moe: ", name, " ", usage, " scales must have ", n / 32,
             " elements (unpadded), got ", s.numel());
  return {d.dptr, s.dptr};
}

void check(hipError_t e, const char *what, const NVTEMoeLayer_ *l) {
  if (e != hipSuccess) {
    (void)hipGetLastError();  // a failed HIP call's sticky error must not fail the caller's next unrelated call
    NVTE_ERROR(what, ": ", hipGetErrorString(e), " / ", l && l->ctx ? kosmos_moe_error(l->ctx) : "");
  }
}

void check_ws(hipError_t e, const char *what, const NVTEMoeWorkspace_ *w) {
  if (e != hipSuccess) {
    (void)hipGetLastError();
    NVTE_ERROR(what, ": ", hipGetErrorString(e), " / ", w && w->ws ? kosmos_moe_workspace_error(w->ws) : "");
  }
}

struct Dims {
  size_t T, H, I, E, k, Eg;
  explicit Dims(const NVTEMoeConfig &c)
      : T(c.num_tokens), H(c.hidden_size), I(c.ffn_hidden_size), E(c.num_local_experts), k(c.topk),
        Eg((size_t)NVTE_MOE_EP_SIZE * c.num_local_experts) {}
};

}  // namespace

int nvte_moe_is_supported(void) {
  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    return 0;
  }
  return std::strncmp(prop.gcnArchName, "gfx950", 6) == 0;
}

int nvte_moe_config_valid(const NVTEMoeConfig *cfg, char *why, size_t why_len) {
  const NVTEMoeConfig c = normalize(cfg, kConfigMinSize, "cfg");
  const KosmosMoeConfig k = to_kosmos(c, nullptr);
  char buf[512] = {0};
  hipError_t e = kosmos_moe_validate_why(&k, buf, sizeof(buf));
  if (e == hipSuccess && c.capacity_rows <= 0) {
    std::snprintf(buf, sizeof(buf), "capacity_rows must be > 0 (the routing is on the device)");
    e = hipErrorInvalidValue;
  }
  if (why && why_len) {
    std::snprintf(why, why_len, "%s", e == hipSuccess ? "" : buf);
  }
  return e == hipSuccess;
}

NVTEMoeWorkspace nvte_moe_workspace_create(const NVTEMoeConfig *cfg, const NVTEMoeComm *comm) {
  NVTE_CHECK(nvte_moe_is_supported(), "nvte_moe: the fused MoE needs a gfx950 device");
  const NVTEMoeConfig c = normalize(cfg, kConfigMinSize, "cfg");
  const NVTEMoeComm m = normalize(comm, kCommMinSize, "comm");
  NVTE_CHECK(m.allgather != nullptr, "nvte_moe: comm.allgather is required");
  NVTE_CHECK((m.alloc == nullptr) == (m.free == nullptr), "nvte_moe: comm.alloc and comm.free go together");
  NVTE_CHECK(m.nranks == NVTE_MOE_EP_SIZE, "nvte_moe: the EP group must have ", NVTE_MOE_EP_SIZE, " ranks, got ",
             m.nranks);
  char why[512];
  NVTE_CHECK(nvte_moe_config_valid(&c, why, sizeof(why)), "nvte_moe: configuration refused: ", why);
  auto *w = new NVTEMoeWorkspace_();
  w->cfg = c;
  w->kcfg = to_kosmos(c, &m);
  w->alloc = KosmosAllocator{m.alloc, m.free, m.ctx};
  AllocScope scope(w);
  const hipError_t e = kosmos_moe_workspace_create(&w->ws, nullptr, &w->kcfg);
  if (e != hipSuccess) {
    delete w;
    (void)hipGetLastError();
    NVTE_ERROR("kosmos_moe_workspace_create: ", hipGetErrorString(e), " / ", kosmos_comm_create_error());
  }
  return w;
}

void nvte_moe_workspace_destroy(NVTEMoeWorkspace ws) {
  AllocScope scope(ws);
  if (!ws) {
    return;
  }
  NVTE_CHECK(ws->nlayers == 0, "nvte_moe_workspace_destroy: ", ws->nlayers, " layer handles still open");
  const hipError_t e = kosmos_moe_workspace_destroy(ws->ws);
  delete ws;
  NVTE_CHECK(e == hipSuccess, "kosmos_moe_workspace_destroy: ", hipGetErrorString(e));
}

NVTEMoeLayer nvte_moe_layer_create(NVTEMoeWorkspace ws) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_layer_create: null workspace");
  auto *l = new NVTEMoeLayer_();
  l->ws = ws;
  const hipError_t e = kosmos_moe_create(&l->ctx, ws->ws);
  if (e != hipSuccess) {
    delete l;
    check_ws(e, (std::string("kosmos_moe_create: ") + kosmos_comm_create_error()).c_str(), ws);
  }
  ws->nlayers++;
  return l;
}

void nvte_moe_layer_destroy(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  if (!layer) {
    return;
  }
  const hipError_t e = kosmos_moe_destroy(layer->ctx);
  layer->ws->nlayers--;
  delete layer;
  NVTE_CHECK(e == hipSuccess, "kosmos_moe_destroy: ", hipGetErrorString(e));
}

void nvte_moe_layer_prepare(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_layer_prepare: null layer");
  check(kosmos_moe_prepare(layer->ctx), "kosmos_moe_prepare", layer);
}

void nvte_moe_layer_reset(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_layer_reset: null layer");
  const Dims d(layer->ws->cfg);
  std::vector<int> idx(d.T * d.k);
  std::vector<float> w(d.T * d.k, 1.0f / d.k);
  for (size_t t = 0; t < d.T; t++) {
    for (size_t j = 0; j < d.k; j++) {
      idx[t * d.k + j] = (int)((t + j) % d.Eg);
    }
  }
  check(kosmos_moe_plan(layer->ctx, idx.data(), w.data()), "kosmos_moe_plan (reset)", layer);
}

int nvte_moe_layer_status(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_layer_status: null layer");
  int status = 0;
  check(kosmos_moe_plan_status(layer->ctx, &status), "kosmos_moe_plan_status", layer);
  return status;
}

void nvte_moe_layer_release(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  if (layer && layer->ctx) {
    (void)kosmos_moe_release(layer->ctx);
  }
}

int nvte_moe_layer_release_state(NVTEMoeLayer layer) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_layer_release_state: null layer");
  int state = 0;
  check(kosmos_moe_release_state(layer->ctx, &state), "kosmos_moe_release_state", layer);
  return state;
}

void nvte_moe_release_sync(NVTEMoeWorkspace ws) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_release_sync: null workspace");
  check_ws(kosmos_moe_release_sync(ws->ws), "kosmos_moe_release_sync", ws);
}

void nvte_moe_forward(NVTEMoeLayer layer, const NVTEMoeForwardArgs *args, cudaStream_t stream) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_forward: null layer");
  const NVTEMoeForwardArgs a = normalize(args, kFwdArgsMinSize, "args");
  const NVTEMoeConfig &c = layer->ws->cfg;
  const Dims d(c);
  KosmosMoe *ctx = layer->ctx;
  NVTE_CHECK(!a.save_for_backward || c.training, "nvte_moe_forward: save_for_backward needs a training config");
  void *G = a.save_for_backward ? KOSMOS_MOE_SAVE : nullptr;
  void *out = rowwise(a.out, "out", DType::kBFloat16, d.T * d.H, true);
  const int *idx = nullptr;
  const float *tw = nullptr;
  if (c.router) {
    NVTE_CHECK(!present(get(a.topk_idx)) && !present(get(a.topk_weights)),
               "nvte_moe_forward: topk_idx / topk_weights must be absent with the fused router");
  } else {
    idx = (const int *)rowwise(a.topk_idx, "topk_idx", DType::kInt32, d.T * d.k, true);
    tw = (const float *)rowwise(a.topk_weights, "topk_weights", DType::kFloat32, d.T * d.k, true);
  }
  const void *Wr = nullptr;
  const float *bias = nullptr, *lbias = nullptr;
  float *aux = nullptr;
  int *tpe = nullptr;
  const unsigned *seed = nullptr;
  if (c.router) {
    Wr = rowwise(a.router_weight, "router_weight", DType::kBFloat16, d.Eg * d.H, true);
    bias = (const float *)rowwise(a.expert_bias, "expert_bias", DType::kFloat32, d.Eg, false);
    lbias = (const float *)rowwise(a.router_bias, "router_bias", DType::kFloat32, d.Eg, false);
    aux = (float *)rowwise(a.aux_loss, "aux_loss", DType::kFloat32, 1, c.router_config.aux_groups > 0);
    tpe = (int *)rowwise(a.tokens_per_expert, "tokens_per_expert", DType::kInt32, d.Eg, true);
    seed = (const unsigned *)rowwise(a.seed, "seed", DType::kInt32, 1, false);
  }
  const float *sgate = nullptr;
  void *Gs = nullptr;
  if (c.shared_expert) {
    sgate = (const float *)rowwise(a.shared_gate, "shared_gate", DType::kFloat32, d.T, false);
    NVTE_CHECK(((uintptr_t)sgate & 15) == 0, "nvte_moe_forward: shared_gate must be 16-byte aligned");
    Gs = rowwise(a.shared_pre_act, "shared_pre_act", DType::kBFloat16, d.T * 2 * d.I, c.mxfp8 && c.training);
  }
  const size_t n1 = d.E * 2 * d.I * d.H, n2 = d.E * d.H * d.I, s1 = 2 * d.I * d.H, s2 = d.H * d.I;
  hipError_t e;
  if (!c.mxfp8) {
    const void *X = rowwise(a.x, "x", DType::kBFloat16, d.T * d.H, true);
    const void *W1 = rowwise(a.fc1_weight, "fc1_weight", DType::kBFloat16, n1, true);
    const void *W2 = rowwise(a.fc2_weight, "fc2_weight", DType::kBFloat16, n2, true);
    const void *Ws1 = nullptr, *Ws2 = nullptr;
    if (c.shared_expert) {
      Ws1 = rowwise(a.shared_fc1_weight, "shared_fc1_weight", DType::kBFloat16, s1, true);
      Ws2 = rowwise(a.shared_fc2_weight, "shared_fc2_weight", DType::kBFloat16, s2, true);
    }
    if (c.router && c.shared_expert) {
      e = kosmos_moe_forward_router_shared(ctx, X, Wr, bias, lbias, W1, W2, Ws1, Ws2, out, G, Gs, sgate, aux, tpe,
                                           seed, stream);
    } else if (c.router) {
      e = kosmos_moe_forward_router(ctx, X, Wr, bias, lbias, W1, W2, out, G, aux, tpe, seed, stream);
    } else if (c.shared_expert) {
      e = kosmos_moe_forward_shared(ctx, X, W1, W2, Ws1, Ws2, idx, tw, out, G, Gs, sgate, stream);
    } else {
      e = kosmos_moe_forward(ctx, X, W1, W2, idx, tw, out, G, stream);
    }
  } else {
    const Mx W1 = mxfp8(a.fc1_weight, "fc1_weight", n1, false, true);
    const Mx W2 = mxfp8(a.fc2_weight, "fc2_weight", n2, false, true);
    if (!c.training) {
      NVTE_CHECK(!c.router && !c.shared_expert,
                 "nvte_moe_forward: MXFP8 inference supports neither the fused router nor the shared expert");
      const Mx X = mxfp8(a.x, "x", d.T * d.H, false, true);
      e = kosmos_moe_forward_mxfp8(ctx, X.data, X.scale, W1.data, W1.scale, W2.data, W2.scale, idx, tw, out, stream);
    } else {
      const void *X = rowwise(a.x, "x", DType::kBFloat16, d.T * d.H, true);
      Mx Ws1, Ws2;
      if (c.shared_expert) {
        Ws1 = mxfp8(a.shared_fc1_weight, "shared_fc1_weight", s1, false, true);
        Ws2 = mxfp8(a.shared_fc2_weight, "shared_fc2_weight", s2, false, true);
      }
      if (c.router && c.shared_expert) {
        e = kosmos_moe_forward_router_shared_mxfp8_train(ctx, X, Wr, bias, lbias, W1.data, W1.scale, W2.data,
                                                         W2.scale, Ws1.data, Ws1.scale, Ws2.data, Ws2.scale, out, G,
                                                         Gs, sgate, aux, tpe, seed, stream);
      } else if (c.router) {
        e = kosmos_moe_forward_router_mxfp8_train(ctx, X, Wr, bias, lbias, W1.data, W1.scale, W2.data, W2.scale, out,
                                                  G, aux, tpe, seed, stream);
      } else if (c.shared_expert) {
        e = kosmos_moe_forward_shared_mxfp8_train(ctx, X, W1.data, W1.scale, W2.data, W2.scale, Ws1.data, Ws1.scale,
                                                  Ws2.data, Ws2.scale, idx, tw, out, G, Gs, sgate, stream);
      } else {
        e = kosmos_moe_forward_mxfp8_train(ctx, X, W1.data, W1.scale, W2.data, W2.scale, idx, tw, out, G, stream);
      }
    }
  }
  check(e, "nvte_moe_forward", layer);
}

void nvte_moe_backward(NVTEMoeLayer layer, const NVTEMoeBackwardArgs *args, cudaStream_t stream) {
  AllocScope scope(layer ? layer->ws : nullptr);
  NVTE_CHECK(layer != nullptr, "nvte_moe_backward: null layer");
  const NVTEMoeBackwardArgs a = normalize(args, kBwdArgsMinSize, "args");
  const NVTEMoeConfig &c = layer->ws->cfg;
  const Dims d(c);
  KosmosMoe *ctx = layer->ctx;
  NVTE_CHECK(c.training, "nvte_moe_backward: the workspace was created without training");
  const void *G = KOSMOS_MOE_SAVE;
  const size_t n1 = d.E * 2 * d.I * d.H, n2 = d.E * d.H * d.I, s1 = 2 * d.I * d.H, s2 = d.H * d.I;
  const void *dout = rowwise(a.dout, "dout", DType::kBFloat16, d.T * d.H, true);
  const void *X = rowwise(a.x, "x", DType::kBFloat16, d.T * d.H, c.router || c.shared_expert || c.mxfp8);
  void *dX = rowwise(a.dx, "dx", DType::kBFloat16, d.T * d.H, true);
  // main_grad accumulation (one acc for every weight gradient): 0 bf16 overwritten, 1 fp32 / 2 bf16 added into.
  const int acc = a.wgrad_accumulate;
  NVTE_CHECK(acc >= 0 && acc <= 2, "nvte_moe_backward: wgrad_accumulate must be 0, 1 or 2");
  const DType wdt = acc == 1 ? DType::kFloat32 : DType::kBFloat16;
  void *dW1 = rowwise(a.fc1_wgrad, "fc1_wgrad", wdt, n1, true);
  void *dW2 = rowwise(a.fc2_wgrad, "fc2_wgrad", wdt, n2, true);
  float *dprobs = nullptr;
  const void *Wr = nullptr;
  const float *aux_scale = nullptr;
  void *dWr = nullptr;
  float *dlbias = nullptr;
  if (c.router) {
    Wr = rowwise(a.router_weight, "router_weight", DType::kBFloat16, d.Eg * d.H, true);
    aux_scale = (const float *)rowwise(a.aux_loss_scale, "aux_loss_scale", DType::kFloat32, 1, true);
    dWr = rowwise(a.router_wgrad, "router_wgrad", wdt, d.Eg * d.H, true);
    dlbias = (float *)rowwise(a.router_bias_grad, "router_bias_grad", acc == 2 ? DType::kBFloat16 : DType::kFloat32,
                              d.Eg, false);
  } else {
    dprobs = (float *)rowwise(a.dprobs, "dprobs", DType::kFloat32, d.T * d.Eg, true);
  }
  const void *Gs = nullptr;
  const float *sgate = nullptr;
  float *dsgate = nullptr;
  void *dWs1 = nullptr, *dWs2 = nullptr;
  if (c.shared_expert) {
    Gs = rowwise(a.shared_pre_act, "shared_pre_act", DType::kBFloat16, d.T * 2 * d.I, true);
    sgate = (const float *)rowwise(a.shared_gate, "shared_gate", DType::kFloat32, d.T, false);
    dsgate = (float *)rowwise(a.shared_gate_grad, "shared_gate_grad", DType::kFloat32, d.T, false);
    dWs1 = rowwise(a.shared_fc1_wgrad, "shared_fc1_wgrad", wdt, s1, true);
    dWs2 = rowwise(a.shared_fc2_wgrad, "shared_fc2_wgrad", wdt, s2, true);
  }
  hipError_t e;
  if (!c.mxfp8) {
    const void *W1 = rowwise(a.fc1_weight, "fc1_weight", DType::kBFloat16, n1, true);
    const void *W2 = rowwise(a.fc2_weight, "fc2_weight", DType::kBFloat16, n2, true);
    const void *Ws1 = nullptr, *Ws2 = nullptr;
    if (c.shared_expert) {
      Ws1 = rowwise(a.shared_fc1_weight, "shared_fc1_weight", DType::kBFloat16, s1, true);
      Ws2 = rowwise(a.shared_fc2_weight, "shared_fc2_weight", DType::kBFloat16, s2, true);
    }
    if (c.router && c.shared_expert) {
      e = kosmos_moe_backward_router_shared(ctx, dout, G, Gs, X, W1, W2, Ws1, Ws2, Wr, aux_scale, dX, dW1, dW2, dWr,
                                            dlbias, dWs1, dWs2, acc, sgate, dsgate, stream);
    } else if (c.router) {
      e = kosmos_moe_backward_router(ctx, dout, G, X, W1, W2, Wr, aux_scale, dX, dW1, dW2, dWr, dlbias, acc, stream);
    } else if (c.shared_expert) {
      e = kosmos_moe_backward_shared(ctx, dout, G, Gs, X, W1, W2, Ws1, Ws2, dX, dprobs, dW1, dW2, dWs1, dWs2, acc,
                                     sgate, dsgate, stream);
    } else {
      e = kosmos_moe_backward(ctx, dout, G, W1, W2, dX, dprobs, dW1, dW2, acc, stream);
    }
  } else {
    const Mx W1 = mxfp8(a.fc1_weight, "fc1_weight", n1, true, true);
    const Mx W2 = mxfp8(a.fc2_weight, "fc2_weight", n2, true, true);
    Mx Ws1, Ws2;
    if (c.shared_expert) {
      Ws1 = mxfp8(a.shared_fc1_weight, "shared_fc1_weight", s1, true, true);
      Ws2 = mxfp8(a.shared_fc2_weight, "shared_fc2_weight", s2, true, true);
    }
    if (c.router && c.shared_expert) {
      e = kosmos_moe_backward_router_shared_mxfp8(ctx, dout, G, Gs, X, W1.data, W1.scale, W2.data, W2.scale, Ws1.data,
                                                  Ws1.scale, Ws2.data, Ws2.scale, Wr, aux_scale, dX, dW1, dW2, dWr,
                                                  dlbias, dWs1, dWs2, acc, sgate, dsgate, stream);
    } else if (c.router) {
      e = kosmos_moe_backward_router_mxfp8(ctx, dout, G, X, W1.data, W1.scale, W2.data, W2.scale, Wr, aux_scale, dX,
                                           dW1, dW2, dWr, dlbias, acc, stream);
    } else if (c.shared_expert) {
      e = kosmos_moe_backward_shared_mxfp8(ctx, dout, G, Gs, X, W1.data, W1.scale, W2.data, W2.scale, Ws1.data,
                                           Ws1.scale, Ws2.data, Ws2.scale, dX, dprobs, dW1, dW2, dWs1, dWs2, acc,
                                           sgate, dsgate, stream);
    } else {
      e = kosmos_moe_backward_mxfp8(ctx, dout, G, X, W1.data, W1.scale, W2.data, W2.scale, dX, dprobs, dW1, dW2,
                                    acc, stream);
    }
  }
  check(e, "nvte_moe_backward", layer);
}

int nvte_moe_pool_resize(NVTEMoeWorkspace ws, size_t bytes) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_pool_resize: null workspace");
  const hipError_t e = kosmos_moe_pool_resize(ws->ws, bytes);
  if (e == hipErrorNotReady || e == hipErrorStreamCaptureUnsupported) {
    return e == hipErrorNotReady ? 1 : 2;
  }
  check_ws(e, "kosmos_moe_pool_resize", ws);
  return 0;
}

void nvte_moe_pool_info(NVTEMoeWorkspace ws, size_t *bytes, size_t *head_bytes, size_t *reserve_bytes) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_pool_info: null workspace");
  size_t b = 0, h = 0, r = 0;
  check_ws(kosmos_moe_pool_info(ws->ws, &b, &h, &r), "kosmos_moe_pool_info", ws);
  if (bytes) {
    *bytes = b;
  }
  if (head_bytes) {
    *head_bytes = h;
  }
  if (reserve_bytes) {
    *reserve_bytes = r;
  }
}

int nvte_moe_set_capacity(NVTEMoeWorkspace ws, int rows) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_set_capacity: null workspace");
  const hipError_t e = kosmos_moe_set_capacity(ws->ws, rows);
  if (e == hipErrorNotReady || e == hipErrorStreamCaptureUnsupported || e == hipErrorOutOfMemory) {
    return e == hipErrorNotReady ? 1 : e == hipErrorStreamCaptureUnsupported ? 2 : 3;
  }
  check_ws(e, "kosmos_moe_set_capacity", ws);
  return 0;
}

void nvte_moe_capacity_query(NVTEMoeWorkspace ws, NVTEMoeCapacityStat *stat) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr && stat != nullptr, "nvte_moe_capacity_query: null argument");
  KosmosMoeCapacityStat s;
  check_ws(kosmos_moe_capacity_query(ws->ws, &s), "kosmos_moe_capacity_query", ws);
  stat->peak = s.peak;
  stat->rows = s.rows;
  stat->max_rows = s.max_rows;
  stat->nlayers = s.nctx;
  stat->rounds = s.rounds;
  stat->cap_rounds = s.cap_rounds;
}

size_t nvte_moe_capacity_bytes(NVTEMoeWorkspace ws, int rows) {
  AllocScope scope(ws);
  NVTE_CHECK(ws != nullptr, "nvte_moe_capacity_bytes: null workspace");
  size_t bytes = 0;
  check_ws(kosmos_moe_capacity_bytes(ws->ws, rows, &bytes), "kosmos_moe_capacity_bytes", ws);
  return bytes;
}

size_t nvte_moe_untracked_bytes(void) { return kosmos_untracked_bytes(); }

#else  // !USE_KOSMOS_MOE

struct NVTEMoeWorkspace_ {};
struct NVTEMoeLayer_ {};

#define NVTE_MOE_UNAVAILABLE() \
  NVTE_ERROR("TransformerEngine was built without the fused MoE (needs ROCm, gfx950 and USE_KOSMOS_MOE)")

int nvte_moe_is_supported(void) { return 0; }
int nvte_moe_config_valid(const NVTEMoeConfig *, char *why, size_t why_len) {
  if (why && why_len) {
    std::snprintf(why, why_len, "built without the fused MoE");
  }
  return 0;
}
NVTEMoeWorkspace nvte_moe_workspace_create(const NVTEMoeConfig *, const NVTEMoeComm *) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_workspace_destroy(NVTEMoeWorkspace) {}
NVTEMoeLayer nvte_moe_layer_create(NVTEMoeWorkspace) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_layer_destroy(NVTEMoeLayer) {}
void nvte_moe_layer_prepare(NVTEMoeLayer) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_layer_reset(NVTEMoeLayer) { NVTE_MOE_UNAVAILABLE(); }
int nvte_moe_layer_status(NVTEMoeLayer) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_layer_release(NVTEMoeLayer) {}
int nvte_moe_layer_release_state(NVTEMoeLayer) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_release_sync(NVTEMoeWorkspace) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_forward(NVTEMoeLayer, const NVTEMoeForwardArgs *, cudaStream_t) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_backward(NVTEMoeLayer, const NVTEMoeBackwardArgs *, cudaStream_t) { NVTE_MOE_UNAVAILABLE(); }
int nvte_moe_pool_resize(NVTEMoeWorkspace, size_t) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_pool_info(NVTEMoeWorkspace, size_t *, size_t *, size_t *) { NVTE_MOE_UNAVAILABLE(); }
int nvte_moe_set_capacity(NVTEMoeWorkspace, int) { NVTE_MOE_UNAVAILABLE(); }
void nvte_moe_capacity_query(NVTEMoeWorkspace, NVTEMoeCapacityStat *) { NVTE_MOE_UNAVAILABLE(); }
size_t nvte_moe_capacity_bytes(NVTEMoeWorkspace, int) { NVTE_MOE_UNAVAILABLE(); }
size_t nvte_moe_untracked_bytes(void) { return 0; }

#endif  // USE_KOSMOS_MOE
