/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

// nvte_rocm_moe_*: the KOSMOS MoE training library (common/kosmos, built with USE_KOSMOS_MOE)
// behind Transformer Engine's C API.

#include <transformer_engine/rocm_moe.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../common.h"
#include "../util/logging.h"

namespace {

// A caller's versioned struct in the current layout: fields past its struct_size read as zero.
template <typename S>
S read_versioned(const S *user, const char *name) {
  NVTE_CHECK(user != nullptr, "nvte_rocm_moe: ", name, " must not be null");
  const size_t n = user->struct_size == 0 ? sizeof(S) : user->struct_size;
  NVTE_CHECK(n >= sizeof(S), "nvte_rocm_moe: ", name, ".struct_size ", user->struct_size,
             " is below the base layout's ", sizeof(S));
  S s{};
  std::memcpy(&s, user, sizeof(S));
  s.struct_size = sizeof(S);
  return s;
}

// Writes an output struct up to the caller's struct_size.
template <typename S>
void write_versioned(S *user, const S &value, const char *name) {
  NVTE_CHECK(user != nullptr, "nvte_rocm_moe: ", name, " must not be null");
  const size_t caller = user->struct_size;
  const size_t n = caller == 0 ? sizeof(S) : std::min(caller, sizeof(S));
  std::memcpy(user, &value, n);
  user->struct_size = caller;
}

}  // namespace

#ifdef USE_KOSMOS_MOE

#include "../kosmos/src/moe/moe.h"

struct NVTERocmMoeInstance_ {
  KosmosMoeInstance *inst = nullptr;
  KosmosComm *comm = nullptr;  // the caller's symmetric buffer, wrapped; null: KOSMOS allocated one
  KosmosMoeConfig cfg{};       // kosmos_comm_wrap keeps a pointer to cfg.coll
  NVTERocmMoeConfig config{};
};

namespace {

using transformer_engine::DType;
using transformer_engine::Tensor;

static_assert(NVTE_ROCM_MOE_EP_SIZE == KOSMOS_MOE_EP);
static_assert(kNVTERocmMoeSwiGLU == KOSMOS_MOE_ACT_SWIGLU);
static_assert(kNVTERocmMoeSwiGLUClamp10 == KOSMOS_MOE_ACT_SWIGLU_CLAMP10);
static_assert(kNVTERocmMoeSaveNone == KOSMOS_MOE_SAVE_NONE);
static_assert(kNVTERocmMoeSaveXp == KOSMOS_MOE_SAVE_XP);
static_assert(kNVTERocmMoeSaveAll == KOSMOS_MOE_SAVE_ALL);

// Stands in for the EP group's all-gather where a configuration is only sized or checked.
int no_allgather(void *, const void *, void *, size_t) { return 1; }

KosmosMoeConfig to_kosmos(const NVTERocmMoeConfig &c, const NVTERocmMoeComm *comm) {
  KosmosMoeConfig k;
  std::memset(&k, 0, sizeof(k));
  k.H = c.hidden_size;
  k.I = c.ffn_hidden_size;
  k.E_loc = c.num_local_experts;
  k.k = c.topk;
  k.T = c.num_tokens;
  k.wgcap = c.wgcap;
  k.capacity_rows = c.capacity_rows;
  k.act = c.activation;
  if (comm != nullptr) {
    k.coll.allgather = comm->allgather;
    k.coll.ctx = comm->ctx;
    k.coll.rank = comm->rank;
    k.coll.nranks = comm->nranks;
  } else {
    k.coll.allgather = no_allgather;
    k.coll.nranks = KOSMOS_MOE_EP;
  }
  return k;
}

KosmosMoeConfig sizing_config(const NVTERocmMoeConfig *cfg) {
  return to_kosmos(read_versioned(cfg, "cfg"), nullptr);
}

std::string shape_str(const std::vector<size_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    s += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return s + "]";
}

// The data of a required plain tensor of this dtype and shape.
void *data_of(const NVTETensor h, const char *name, DType dtype, const std::vector<size_t> &shape) {
  NVTE_CHECK(h != nullptr, "nvte_rocm_moe: ", name, " is required");
  const Tensor *t = transformer_engine::convertNVTETensorCheck(h);
  NVTE_CHECK(t->data.dptr != nullptr, "nvte_rocm_moe: ", name, " has no data");
  NVTE_CHECK(t->data.dtype == dtype, "nvte_rocm_moe: ", name, " must be ",
             transformer_engine::to_string(dtype), ", got ",
             transformer_engine::to_string(t->data.dtype));
  NVTE_CHECK(t->data.shape == shape, "nvte_rocm_moe: ", name, " must have shape ", shape_str(shape),
             ", got ", shape_str(std::vector<size_t>(t->data.shape.begin(), t->data.shape.end())));
  return t->data.dptr;
}

// A saved buffer: absent (null) or a byte tensor of at least bytes bytes, 256-byte aligned.
void *saved_buffer(const NVTETensor h, const char *name, size_t bytes, bool required) {
  if (h == nullptr) {
    NVTE_CHECK(!required, "nvte_rocm_moe: saved.", name, " is required at this tier");
    return nullptr;
  }
  const Tensor *t = transformer_engine::convertNVTETensorCheck(h);
  NVTE_CHECK(t->data.dptr != nullptr || !required, "nvte_rocm_moe: saved.", name,
             " is required at this tier");
  if (t->data.dptr == nullptr) {
    return nullptr;
  }
  NVTE_CHECK(t->data.dtype == DType::kByte, "nvte_rocm_moe: saved.", name,
             " must be a byte tensor");
  NVTE_CHECK(t->data.numel() >= bytes, "nvte_rocm_moe: saved.", name, " holds ", t->data.numel(),
             " bytes, needs ", bytes);
  NVTE_CHECK(reinterpret_cast<uintptr_t>(t->data.dptr) % 256 == 0, "nvte_rocm_moe: saved.", name,
             " must be 256-byte aligned");
  return t->data.dptr;
}

KosmosMoeSaved saved_of(const NVTERocmMoeInstance_ *in, const NVTERocmMoeSaved *user) {
  const NVTERocmMoeSaved s = read_versioned(user, "saved");
  NVTE_CHECK(s.tier >= kNVTERocmMoeSaveNone && s.tier <= kNVTERocmMoeSaveAll,
             "nvte_rocm_moe: saved.tier must be 0, 1 or 2, got ", s.tier);
  KosmosMoeSavedBytes b;
  NVTE_CHECK(kosmos_moe_saved_bytes(&in->cfg, s.rows, &b) == hipSuccess,
             "nvte_rocm_moe: saved.rows must be in [1, the capacity rows], got ", s.rows);
  KosmosMoeSaved k;
  k.tier = s.tier;
  k.rows = s.rows;
  k.xp = saved_buffer(s.xp, "xp", b.xp, s.tier >= kNVTERocmMoeSaveXp);
  k.g = saved_buffer(s.g, "g", b.g, s.tier >= kNVTERocmMoeSaveAll);
  k.a = saved_buffer(s.a, "a", b.a, s.tier >= kNVTERocmMoeSaveAll);
  k.y = saved_buffer(s.y, "y", b.y, true);
  k.route = saved_buffer(s.route, "route", b.route, true);
  return k;
}

void check_call(hipError_t e, const char *what, const NVTERocmMoeInstance_ *in) {
  if (e != hipSuccess) {
    NVTE_ERROR(what, ": ", hipGetErrorString(e), " / ", kosmos_moe_instance_error(in->inst));
  }
}

NVTERocmMoeInstance_ *live(NVTERocmMoeInstance inst, const char *what) {
  NVTE_CHECK(inst != nullptr && inst->inst != nullptr, what, ": null instance");
  return inst;
}

}  // namespace

int nvte_rocm_moe_is_built(void) { return 1; }

int nvte_rocm_moe_config_valid(const NVTERocmMoeConfig *cfg) {
  NVTE_API_CALL(nvte_rocm_moe_config_valid);
  const KosmosMoeConfig k = sizing_config(cfg);
  size_t bytes = 0;
  return kosmos_moe_instance_bytes(&k, &bytes) == hipSuccess;
}

size_t nvte_rocm_moe_instance_bytes(const NVTERocmMoeConfig *cfg) {
  NVTE_API_CALL(nvte_rocm_moe_instance_bytes);
  const KosmosMoeConfig k = sizing_config(cfg);
  size_t bytes = 0;
  NVTE_CHECK(
      kosmos_moe_instance_bytes(&k, &bytes) == hipSuccess,
      "nvte_rocm_moe_instance_bytes: configuration refused (see nvte_rocm_moe_config_valid)");
  return bytes;
}

size_t nvte_rocm_moe_symmetric_buffer_bytes(const NVTERocmMoeConfig *cfg) {
  NVTE_API_CALL(nvte_rocm_moe_symmetric_buffer_bytes);
  // kosmos_comm_wrap carves the peer flags from the end of a buffer given without a flag area
  const size_t n = nvte_rocm_moe_instance_bytes(cfg) + kosmos_comm_flag_bytes(KOSMOS_MOE_EP) + 256;
  return (n + 4095) / 4096 * 4096;
}

NVTERocmMoeInstance nvte_rocm_moe_instance_create(const NVTERocmMoeConfig *cfg,
                                                  const NVTERocmMoeComm *comm,
                                                  const NVTERocmMoeSymmetricBuffer *buffer) {
  NVTE_API_CALL(nvte_rocm_moe_instance_create);
  auto *in = new NVTERocmMoeInstance_();
  try {
    in->config = read_versioned(cfg, "cfg");
    const NVTERocmMoeComm m = read_versioned(comm, "comm");
    NVTE_CHECK(m.allgather != nullptr, "nvte_rocm_moe_instance_create: comm.allgather is required");
    NVTE_CHECK(m.nranks == NVTE_ROCM_MOE_EP_SIZE,
               "nvte_rocm_moe_instance_create: the EP group must have ", NVTE_ROCM_MOE_EP_SIZE,
               " ranks, got ", m.nranks);
    in->cfg = to_kosmos(in->config, &m);
    size_t need = 0;
    NVTE_CHECK(
        kosmos_moe_instance_bytes(&in->cfg, &need) == hipSuccess,
        "nvte_rocm_moe_instance_create: configuration refused (see nvte_rocm_moe_config_valid)");
    if (buffer != nullptr) {
      const NVTERocmMoeSymmetricBuffer b = read_versioned(buffer, "buffer");
      KosmosExternalBuffers eb;
      std::memset(&eb, 0, sizeof(eb));
      eb.local = b.local;
      eb.peers = b.peers;
      eb.bytes = b.bytes;
      eb.rank = m.rank;
      eb.nranks = m.nranks;
      eb.coll = &in->cfg.coll;
      const hipError_t e = kosmos_comm_wrap(&in->comm, &eb);
      NVTE_CHECK(e == hipSuccess,
                 "nvte_rocm_moe_instance_create: kosmos_comm_wrap: ", hipGetErrorString(e), " / ",
                 in->comm ? kosmos_comm_error(in->comm) : "");
    }
    const hipError_t e = kosmos_moe_instance_create(&in->inst, in->comm, &in->cfg);
    NVTE_CHECK(e == hipSuccess,
               "nvte_rocm_moe_instance_create: kosmos_moe_instance_create: ", hipGetErrorString(e));
  } catch (...) {
    if (in->comm != nullptr) {
      (void)kosmos_comm_destroy(in->comm);
    }
    delete in;
    throw;
  }
  return in;
}

void nvte_rocm_moe_instance_destroy(NVTERocmMoeInstance inst) {
  NVTE_API_CALL(nvte_rocm_moe_instance_destroy);
  if (inst == nullptr) {
    return;
  }
  const hipError_t e = kosmos_moe_instance_destroy(inst->inst);
  if (inst->comm != nullptr) {
    (void)kosmos_comm_destroy(inst->comm);
  }
  delete inst;
  NVTE_CHECK(e == hipSuccess, "nvte_rocm_moe_instance_destroy: ", hipGetErrorString(e));
}

const char *nvte_rocm_moe_instance_error(NVTERocmMoeInstance inst) {
  return kosmos_moe_instance_error(live(inst, "nvte_rocm_moe_instance_error")->inst);
}

void nvte_rocm_moe_instance_info(NVTERocmMoeInstance inst, NVTERocmMoeInfo *info) {
  NVTE_API_CALL(nvte_rocm_moe_instance_info);
  const KosmosMoeInfo *k =
      kosmos_moe_instance_info(live(inst, "nvte_rocm_moe_instance_info")->inst);
  NVTE_CHECK(k != nullptr, "nvte_rocm_moe_instance_info: no info");
  NVTERocmMoeInfo r{};
  r.struct_size = sizeof(r);
  r.Mpad = k->Mpad;
  r.Mpad_max = k->Mpad_max;
  r.fwd_ncomm = k->fwd_ncomm;
  r.fwd_fcomm = k->fwd_fcomm;
  r.fwd_burst = k->fwd_burst;
  r.nt1 = k->nt1;
  r.nt2 = k->nt2;
  r.bwd_cncomm_clamped = k->bwd_cncomm_clamped;
  r.bwd_grid = k->bwd_grid;
  std::copy(k->bwd_dncomm_all, k->bwd_dncomm_all + KOSMOS_MOE_EP, r.bwd_dncomm_all);
  std::copy(k->bwd_cncomm_all, k->bwd_cncomm_all + KOSMOS_MOE_EP, r.bwd_cncomm_all);
  r.bwd_dpush_gb = k->bwd_dpush_gb;
  r.bwd_lega_tflop = k->bwd_lega_tflop;
  r.bwd_cpush_gb = k->bwd_cpush_gb;
  r.bwd_wg2_tiles = k->bwd_wg2_tiles;
  r.bwd_wg2_meank = k->bwd_wg2_meank;
  r.bwd_wg2run = k->bwd_wg2run;
  r.bwd_wg1run = k->bwd_wg1run;
  r.arena_rows = k->arena_rows;
  write_versioned(info, r, "info");
}

int nvte_rocm_moe_instance_status(NVTERocmMoeInstance inst) {
  NVTE_API_CALL(nvte_rocm_moe_instance_status);
  NVTERocmMoeInstance_ *in = live(inst, "nvte_rocm_moe_instance_status");
  int status = 0;
  check_call(kosmos_moe_instance_status(in->inst, &status), "nvte_rocm_moe_instance_status", in);
  return status;
}

void nvte_rocm_moe_saved_bytes(const NVTERocmMoeConfig *cfg, int rows,
                               NVTERocmMoeSavedBytes *bytes) {
  NVTE_API_CALL(nvte_rocm_moe_saved_bytes);
  const KosmosMoeConfig k = sizing_config(cfg);
  KosmosMoeSavedBytes b;
  NVTE_CHECK(kosmos_moe_saved_bytes(&k, rows, &b) == hipSuccess,
             "nvte_rocm_moe_saved_bytes: rows must be in [1, the capacity rows], got ", rows,
             " (or the configuration is refused)");
  NVTERocmMoeSavedBytes r{};
  r.struct_size = sizeof(r);
  r.xp = b.xp;
  r.g = b.g;
  r.a = b.a;
  r.y = b.y;
  r.route = b.route;
  write_versioned(bytes, r, "bytes");
}

void nvte_rocm_moe_train_forward(NVTERocmMoeInstance inst, const NVTETensor x, const NVTETensor w1,
                                 const NVTETensor w2, const NVTETensor topk_idx,
                                 const NVTETensor topk_weights, NVTETensor out,
                                 const NVTERocmMoeSaved *saved, cudaStream_t stream) {
  NVTE_API_CALL(nvte_rocm_moe_train_forward);
  NVTERocmMoeInstance_ *in = live(inst, "nvte_rocm_moe_train_forward");
  const KosmosMoeConfig &g = in->cfg;
  const size_t T = g.T;
  const size_t H = g.H;
  const size_t I = g.I;
  const size_t E = g.E_loc;
  const size_t k = g.k;
  const void *px = data_of(x, "x", DType::kBFloat16, {T, H});
  const void *pw1 = data_of(w1, "w1", DType::kBFloat16, {E, 2 * I, H});
  const void *pw2 = data_of(w2, "w2", DType::kBFloat16, {E, H, I});
  const int *pidx = static_cast<const int *>(data_of(topk_idx, "topk_idx", DType::kInt32, {T, k}));
  const float *pw =
      static_cast<const float *>(data_of(topk_weights, "topk_weights", DType::kFloat32, {T, k}));
  void *pout = data_of(out, "out", DType::kBFloat16, {T, H});
  KosmosMoeSaved sv{};
  if (saved != nullptr) {
    sv = saved_of(in, saved);
  }
  check_call(kosmos_moe_train_forward(in->inst, px, pw1, pw2, pidx, pw, pout,
                                      saved != nullptr ? &sv : nullptr, stream),
             "nvte_rocm_moe_train_forward", in);
}

void nvte_rocm_moe_train_backward(NVTERocmMoeInstance inst, const NVTETensor dout,
                                  const NVTETensor x, const NVTETensor w1, const NVTETensor w2,
                                  const NVTETensor topk_idx, const NVTETensor topk_weights,
                                  const NVTERocmMoeSaved *saved, NVTETensor dx, NVTETensor dprobs,
                                  NVTETensor dw1, NVTETensor dw2, int accumulate,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_rocm_moe_train_backward);
  NVTERocmMoeInstance_ *in = live(inst, "nvte_rocm_moe_train_backward");
  NVTE_CHECK(saved != nullptr, "nvte_rocm_moe_train_backward: saved is required");
  NVTE_CHECK(accumulate >= 0 && accumulate <= 2,
             "nvte_rocm_moe_train_backward: accumulate must be 0, 1 or 2, got ", accumulate);
  const KosmosMoeConfig &g = in->cfg;
  const size_t T = g.T;
  const size_t H = g.H;
  const size_t I = g.I;
  const size_t E = g.E_loc;
  const size_t k = g.k;
  const DType wdt = accumulate == 1 ? DType::kFloat32 : DType::kBFloat16;
  const void *pdout = data_of(dout, "dout", DType::kBFloat16, {T, H});
  const void *px = data_of(x, "x", DType::kBFloat16, {T, H});
  const void *pw1 = data_of(w1, "w1", DType::kBFloat16, {E, 2 * I, H});
  const void *pw2 = data_of(w2, "w2", DType::kBFloat16, {E, H, I});
  const int *pidx = static_cast<const int *>(data_of(topk_idx, "topk_idx", DType::kInt32, {T, k}));
  const float *pw =
      static_cast<const float *>(data_of(topk_weights, "topk_weights", DType::kFloat32, {T, k}));
  void *pdx = data_of(dx, "dx", DType::kBFloat16, {T, H});
  float *pdprobs = static_cast<float *>(
      data_of(dprobs, "dprobs", DType::kFloat32, {T, NVTE_ROCM_MOE_EP_SIZE * E}));
  void *pdw1 = data_of(dw1, "dw1", wdt, {E, 2 * I, H});
  void *pdw2 = data_of(dw2, "dw2", wdt, {E, H, I});
  const KosmosMoeSaved sv = saved_of(in, saved);
  check_call(kosmos_moe_train_backward(in->inst, pdout, px, pw1, pw2, pidx, pw, &sv, pdx, pdprobs,
                                       pdw1, pdw2, accumulate, stream),
             "nvte_rocm_moe_train_backward", in);
}

#else  // !USE_KOSMOS_MOE

struct NVTERocmMoeInstance_ {};

#define NVTE_ROCM_MOE_UNAVAILABLE()                                                 \
  NVTE_ERROR(                                                                       \
      "Transformer Engine was built without the fused MoE (needs ROCm, gfx950 and " \
      "USE_KOSMOS_MOE)")

int nvte_rocm_moe_is_built(void) { return 0; }
int nvte_rocm_moe_config_valid(const NVTERocmMoeConfig *) { return 0; }
size_t nvte_rocm_moe_instance_bytes(const NVTERocmMoeConfig *) { NVTE_ROCM_MOE_UNAVAILABLE(); }
size_t nvte_rocm_moe_symmetric_buffer_bytes(const NVTERocmMoeConfig *) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}
NVTERocmMoeInstance nvte_rocm_moe_instance_create(const NVTERocmMoeConfig *,
                                                  const NVTERocmMoeComm *,
                                                  const NVTERocmMoeSymmetricBuffer *) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}
void nvte_rocm_moe_instance_destroy(NVTERocmMoeInstance) {}
const char *nvte_rocm_moe_instance_error(NVTERocmMoeInstance) { NVTE_ROCM_MOE_UNAVAILABLE(); }
void nvte_rocm_moe_instance_info(NVTERocmMoeInstance, NVTERocmMoeInfo *) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}
int nvte_rocm_moe_instance_status(NVTERocmMoeInstance) { NVTE_ROCM_MOE_UNAVAILABLE(); }
void nvte_rocm_moe_saved_bytes(const NVTERocmMoeConfig *, int, NVTERocmMoeSavedBytes *) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}
void nvte_rocm_moe_train_forward(NVTERocmMoeInstance, const NVTETensor, const NVTETensor,
                                 const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor,
                                 const NVTERocmMoeSaved *, cudaStream_t) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}
void nvte_rocm_moe_train_backward(NVTERocmMoeInstance, const NVTETensor, const NVTETensor,
                                  const NVTETensor, const NVTETensor, const NVTETensor,
                                  const NVTETensor, const NVTERocmMoeSaved *, NVTETensor,
                                  NVTETensor, NVTETensor, NVTETensor, int, cudaStream_t) {
  NVTE_ROCM_MOE_UNAVAILABLE();
}

#endif  // USE_KOSMOS_MOE
