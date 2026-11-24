/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// Framework-free Python binding of the KOSMOS MoE library: device pointers and streams pass as integers.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include "moe.h"

namespace py = pybind11;

struct PyCollective {
  py::function fn;
  std::string err;
};

// The library's host all-gather: fn(send: bytes) returns the 8 ranks' sends concatenated (any bytes-like object).
static int py_allgather(void *ctx, const void *send, void *recv, size_t bytes) {
  PyCollective *c = (PyCollective *)ctx;
  py::gil_scoped_acquire gil;
  try {
    py::object res = c->fn(py::bytes((const char *)send, bytes));
    py::bytes all = py::reinterpret_steal<py::bytes>(PyBytes_FromObject(res.ptr()));
    if (!all) {
      throw py::error_already_set();
    }
    const std::string_view v = all;
    if (v.size() != bytes * KOSMOS_MOE_EP) {
      c->err = "allgather returned " + std::to_string(v.size()) + " bytes, expected " +
               std::to_string(bytes * KOSMOS_MOE_EP);
      return 1;
    }
    memcpy(recv, v.data(), v.size());
    return 0;
  } catch (const std::exception &e) {
    c->err = e.what();
    return 1;
  }
}

static void check(hipError_t e, const char *what, const KosmosMoe *m, PyCollective *coll) {
  if (e == hipSuccess) {
    return;
  }
  std::string msg = std::string(what) + ": " + hipGetErrorString(e);
  if (m && *kosmos_moe_error(m)) {
    msg += " / " + std::string(kosmos_moe_error(m));
  }
  if (!coll->err.empty()) {
    msg += " / allgather: " + coll->err;
    coll->err.clear();
  }
  throw std::runtime_error(msg);
}

struct MoeWorkspace {
  KosmosMoeWorkspace *w = nullptr;
  PyCollective coll;

  MoeWorkspace(int H, int I, int E_loc, int k, int T, int rank, py::function allgather, int wgcap, int capacity_rows,
               bool backward, const KosmosMoeRouterConfig *router, bool shared, bool mxfp8, bool fp8_combine,
               int act, bool probs_in, size_t pool_bytes) {
    KosmosMoeConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    coll.fn = allgather;
    cfg.H = H, cfg.I = I, cfg.E_loc = E_loc, cfg.k = k, cfg.T = T, cfg.wgcap = wgcap;
    cfg.backward = backward, cfg.capacity_rows = capacity_rows, cfg.shared = shared, cfg.mxfp8 = mxfp8;
    cfg.fp8_combine = fp8_combine, cfg.act = act, cfg.probs_in = probs_in, cfg.pool_bytes = pool_bytes;
    cfg.coll.allgather = py_allgather, cfg.coll.ctx = &coll, cfg.coll.rank = rank, cfg.coll.nranks = KOSMOS_MOE_EP;
    if (router) {
      cfg.router = *router;
    }
    if (kosmos_moe_validate(&cfg) != hipSuccess) {
      throw std::invalid_argument("KOSMOS MoE configuration refused by kosmos_moe_validate (see src/moe/README.md)");
    }
    check(kosmos_moe_workspace_create(&w, nullptr, &cfg), "kosmos_moe_workspace_create", nullptr, &coll);
  }

  void destroy() {
    check(kosmos_moe_workspace_destroy(w), "kosmos_moe_workspace_destroy", nullptr, &coll);
    w = nullptr;
  }

  KosmosMoeWorkspace *live() {
    if (!w) {
      throw std::runtime_error("MoeWorkspace used after destroy()");
    }
    return w;
  }

  // 0: resized; refused, nothing changed: 1 a stored block awaits its backward, 2 inside a stream capture.
  void fail(hipError_t e, const char *what) {
    if (e == hipSuccess) {
      return;
    }
    std::string msg = std::string(what) + ": " + hipGetErrorString(e) + " / " + kosmos_moe_workspace_error(w);
    if (!coll.err.empty()) {
      msg += " / allgather: " + coll.err;
      coll.err.clear();
    }
    throw std::runtime_error(msg);
  }

  int pool_resize(size_t bytes) {
    const hipError_t e = kosmos_moe_pool_resize(live(), bytes);
    if (e == hipErrorNotReady || e == hipErrorStreamCaptureUnsupported) {
      return e == hipErrorNotReady ? 1 : 2;
    }
    fail(e, "kosmos_moe_pool_resize");
    return 0;
  }

  // As pool_resize; 3: an allocation failed on some rank, the old capacity kept.
  int set_capacity(int rows) {
    const hipError_t e = kosmos_moe_set_capacity(live(), rows);
    if (e == hipErrorNotReady || e == hipErrorStreamCaptureUnsupported || e == hipErrorOutOfMemory) {
      return e == hipErrorNotReady ? 1 : e == hipErrorStreamCaptureUnsupported ? 2 : 3;
    }
    fail(e, "kosmos_moe_set_capacity");
    return 0;
  }

  py::dict capacity_query() {
    KosmosMoeCapacityStat st;
    fail(kosmos_moe_capacity_query(live(), &st), "kosmos_moe_capacity_query");
    py::dict d;
    d["peak"] = st.peak, d["rows"] = st.rows, d["max_rows"] = st.max_rows, d["nctx"] = st.nctx;
    d["rounds"] = st.rounds, d["cap_rounds"] = st.cap_rounds;
    d["overflow"] = st.overflow;
    return d;
  }

  size_t capacity_bytes(int rows) {
    size_t b = 0;
    check(kosmos_moe_capacity_bytes(live(), rows, &b), "kosmos_moe_capacity_bytes", nullptr, &coll);
    return b;
  }

  py::dict pool_info() {
    size_t b = 0, head = 0, res = 0;
    check(kosmos_moe_pool_info(live(), &b, &head, &res), "kosmos_moe_pool_info", nullptr, &coll);
    py::dict d;
    d["bytes"] = b, d["head_bytes"] = head, d["reserve_bytes"] = res;
    return d;
  }
};

struct MoeContext {
  KosmosMoe *m = nullptr;
  py::object ws_ref;
  MoeWorkspace *ws;

  explicit MoeContext(py::object ws_obj) : ws_ref(ws_obj), ws(&ws_obj.cast<MoeWorkspace &>()) {
    if (!ws->w) {
      throw std::runtime_error("MoeContext on a destroyed workspace");
    }
    check(kosmos_moe_create(&m, ws->w), "kosmos_moe_create", nullptr, &ws->coll);
  }

  KosmosMoe *live() {
    if (!m) {
      throw std::runtime_error("MoeContext used after destroy()");
    }
    return m;
  }

  void forward(uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t topk_idx, uintptr_t topk_w, uintptr_t out,
               uintptr_t G, uintptr_t stream) {
    check(kosmos_moe_forward(live(), (const void *)X, (const void *)W1, (const void *)W2, (const int *)topk_idx,
                             (const float *)topk_w, (void *)out, (void *)G, (hipStream_t)stream),
          "kosmos_moe_forward", m, &ws->coll);
  }

  void forward_mxfp8_train(uintptr_t X, uintptr_t W1, uintptr_t W1_scale, uintptr_t W2, uintptr_t W2_scale,
                           uintptr_t topk_idx, uintptr_t topk_w, uintptr_t out, uintptr_t G, uintptr_t stream) {
    check(kosmos_moe_forward_mxfp8_train(live(), (const void *)X, (const void *)W1, (const void *)W1_scale,
                                         (const void *)W2, (const void *)W2_scale, (const int *)topk_idx,
                                         (const float *)topk_w, (void *)out, (void *)G, (hipStream_t)stream),
          "kosmos_moe_forward_mxfp8_train", m, &ws->coll);
  }

  void backward_mxfp8(uintptr_t dout, uintptr_t G, uintptr_t X, uintptr_t W1c, uintptr_t W1c_scale, uintptr_t W2c,
                      uintptr_t W2c_scale, uintptr_t dX, uintptr_t dprobs, uintptr_t dW1, uintptr_t dW2,
                      uintptr_t stream) {
    check(kosmos_moe_backward_mxfp8(live(), (const void *)dout, (const void *)G, (const void *)X, (const void *)W1c,
                                    (const void *)W1c_scale, (const void *)W2c, (const void *)W2c_scale, (void *)dX,
                                    (float *)dprobs, (void *)dW1, (void *)dW2, (hipStream_t)stream),
          "kosmos_moe_backward_mxfp8", m, &ws->coll);
  }

  void forward_router(uintptr_t X, uintptr_t Wr, uintptr_t bias, uintptr_t W1, uintptr_t W2, uintptr_t out,
                      uintptr_t G, uintptr_t aux_loss, uintptr_t tpe, uintptr_t seed, uintptr_t stream,
                      uintptr_t logit_bias) {
    check(kosmos_moe_forward_router(live(), (const void *)X, (const void *)Wr, (const float *)bias,
                                    (const float *)logit_bias, (const void *)W1, (const void *)W2, (void *)out,
                                    (void *)G, (float *)aux_loss, (int *)tpe, (const unsigned *)seed,
                                    (hipStream_t)stream),
          "kosmos_moe_forward_router", m, &ws->coll);
  }

  void backward(uintptr_t dout, uintptr_t G, uintptr_t W1, uintptr_t W2, uintptr_t dX, uintptr_t dprobs, uintptr_t dW1,
                uintptr_t dW2, uintptr_t stream) {
    check(kosmos_moe_backward(live(), (const void *)dout, (const void *)G, (const void *)W1, (const void *)W2,
                              (void *)dX, (float *)dprobs, (void *)dW1, (void *)dW2, (hipStream_t)stream),
          "kosmos_moe_backward", m, &ws->coll);
  }

  void backward_router(uintptr_t dout, uintptr_t G, uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t Wr,
                       uintptr_t aux_scale, uintptr_t dX, uintptr_t dW1, uintptr_t dW2, uintptr_t dWr, uintptr_t stream,
                       uintptr_t dlogit_bias) {
    check(kosmos_moe_backward_router(live(), (const void *)dout, (const void *)G, (const void *)X, (const void *)W1,
                                     (const void *)W2, (const void *)Wr, (const float *)aux_scale, (void *)dX,
                                     (void *)dW1, (void *)dW2, (void *)dWr, (float *)dlogit_bias, (hipStream_t)stream),
          "kosmos_moe_backward_router", m, &ws->coll);
  }

  void forward_shared(uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t Ws1, uintptr_t Ws2, uintptr_t topk_idx,
                      uintptr_t topk_w, uintptr_t out, uintptr_t G, uintptr_t Gs, uintptr_t stream, uintptr_t sgate) {
    check(kosmos_moe_forward_shared(live(), (const void *)X, (const void *)W1, (const void *)W2, (const void *)Ws1,
                                    (const void *)Ws2, (const int *)topk_idx, (const float *)topk_w, (void *)out,
                                    (void *)G, (void *)Gs, (const float *)sgate, (hipStream_t)stream),
          "kosmos_moe_forward_shared", m, &ws->coll);
  }

  void forward_router_shared(uintptr_t X, uintptr_t Wr, uintptr_t bias, uintptr_t W1, uintptr_t W2, uintptr_t Ws1,
                             uintptr_t Ws2, uintptr_t out, uintptr_t G, uintptr_t Gs, uintptr_t aux_loss,
                             uintptr_t tpe, uintptr_t seed, uintptr_t stream, uintptr_t logit_bias, uintptr_t sgate) {
    check(kosmos_moe_forward_router_shared(live(), (const void *)X, (const void *)Wr, (const float *)bias,
                                           (const float *)logit_bias, (const void *)W1, (const void *)W2,
                                           (const void *)Ws1, (const void *)Ws2, (void *)out, (void *)G, (void *)Gs,
                                           (const float *)sgate, (float *)aux_loss, (int *)tpe, (const unsigned *)seed,
                                           (hipStream_t)stream),
          "kosmos_moe_forward_router_shared", m, &ws->coll);
  }

  void backward_shared(uintptr_t dout, uintptr_t G, uintptr_t Gs, uintptr_t X, uintptr_t W1, uintptr_t W2,
                       uintptr_t Ws1, uintptr_t Ws2, uintptr_t dX, uintptr_t dprobs, uintptr_t dW1, uintptr_t dW2,
                       uintptr_t dWs1, uintptr_t dWs2, int acc, uintptr_t stream, uintptr_t sgate, uintptr_t dsgate) {
    check(kosmos_moe_backward_shared(live(), (const void *)dout, (const void *)G, (const void *)Gs, (const void *)X,
                                     (const void *)W1, (const void *)W2, (const void *)Ws1, (const void *)Ws2,
                                     (void *)dX, (float *)dprobs, (void *)dW1, (void *)dW2, (void *)dWs1, (void *)dWs2,
                                     acc, (const float *)sgate, (float *)dsgate, (hipStream_t)stream),
          "kosmos_moe_backward_shared", m, &ws->coll);
  }

  void backward_router_shared(uintptr_t dout, uintptr_t G, uintptr_t Gs, uintptr_t X, uintptr_t W1, uintptr_t W2,
                              uintptr_t Ws1, uintptr_t Ws2, uintptr_t Wr, uintptr_t aux_scale, uintptr_t dX,
                              uintptr_t dW1, uintptr_t dW2, uintptr_t dWr, uintptr_t dWs1, uintptr_t dWs2, int acc,
                              uintptr_t stream, uintptr_t dlogit_bias, uintptr_t sgate, uintptr_t dsgate) {
    check(kosmos_moe_backward_router_shared(live(), (const void *)dout, (const void *)G, (const void *)Gs,
                                            (const void *)X, (const void *)W1, (const void *)W2, (const void *)Ws1,
                                            (const void *)Ws2, (const void *)Wr, (const float *)aux_scale, (void *)dX,
                                            (void *)dW1, (void *)dW2, (void *)dWr, (float *)dlogit_bias, (void *)dWs1,
                                            (void *)dWs2, acc, (const float *)sgate, (float *)dsgate,
                                            (hipStream_t)stream),
          "kosmos_moe_backward_router_shared", m, &ws->coll);
  }

  void forward_shared_mxfp8_train(uintptr_t X, uintptr_t W1, uintptr_t W1_scale, uintptr_t W2, uintptr_t W2_scale,
                                  uintptr_t Ws1, uintptr_t Ws1_scale, uintptr_t Ws2, uintptr_t Ws2_scale,
                                  uintptr_t topk_idx, uintptr_t topk_w, uintptr_t out, uintptr_t G, uintptr_t Gs,
                                  uintptr_t stream, uintptr_t sgate) {
    check(kosmos_moe_forward_shared_mxfp8_train(live(), (const void *)X, (const void *)W1, (const void *)W1_scale,
                                                (const void *)W2, (const void *)W2_scale, (const void *)Ws1,
                                                (const void *)Ws1_scale, (const void *)Ws2, (const void *)Ws2_scale,
                                                (const int *)topk_idx, (const float *)topk_w, (void *)out, (void *)G,
                                                (void *)Gs, (const float *)sgate, (hipStream_t)stream),
          "kosmos_moe_forward_shared_mxfp8_train", m, &ws->coll);
  }

  void backward_shared_mxfp8(uintptr_t dout, uintptr_t G, uintptr_t Gs, uintptr_t X, uintptr_t W1c,
                             uintptr_t W1c_scale, uintptr_t W2c, uintptr_t W2c_scale, uintptr_t Ws1c,
                             uintptr_t Ws1c_scale, uintptr_t Ws2c, uintptr_t Ws2c_scale, uintptr_t dX, uintptr_t dprobs,
                             uintptr_t dW1, uintptr_t dW2, uintptr_t dWs1, uintptr_t dWs2, int acc, uintptr_t stream,
                             uintptr_t sgate, uintptr_t dsgate) {
    check(kosmos_moe_backward_shared_mxfp8(live(), (const void *)dout, (const void *)G, (const void *)Gs,
                                           (const void *)X, (const void *)W1c, (const void *)W1c_scale,
                                           (const void *)W2c, (const void *)W2c_scale, (const void *)Ws1c,
                                           (const void *)Ws1c_scale, (const void *)Ws2c, (const void *)Ws2c_scale,
                                           (void *)dX, (float *)dprobs, (void *)dW1, (void *)dW2, (void *)dWs1,
                                           (void *)dWs2, acc, (const float *)sgate, (float *)dsgate,
                                           (hipStream_t)stream),
          "kosmos_moe_backward_shared_mxfp8", m, &ws->coll);
  }

  void forward_router_mxfp8_train(uintptr_t X, uintptr_t Wr, uintptr_t bias, uintptr_t W1, uintptr_t W1_scale,
                                  uintptr_t W2, uintptr_t W2_scale, uintptr_t out, uintptr_t G, uintptr_t aux_loss,
                                  uintptr_t tpe, uintptr_t seed, uintptr_t stream, uintptr_t logit_bias) {
    check(kosmos_moe_forward_router_mxfp8_train(live(), (const void *)X, (const void *)Wr, (const float *)bias,
                                                (const float *)logit_bias, (const void *)W1, (const void *)W1_scale,
                                                (const void *)W2, (const void *)W2_scale, (void *)out, (void *)G,
                                                (float *)aux_loss, (int *)tpe, (const unsigned *)seed,
                                                (hipStream_t)stream),
          "kosmos_moe_forward_router_mxfp8_train", m, &ws->coll);
  }

  void backward_router_mxfp8(uintptr_t dout, uintptr_t G, uintptr_t X, uintptr_t W1c, uintptr_t W1c_scale,
                             uintptr_t W2c, uintptr_t W2c_scale, uintptr_t Wr, uintptr_t aux_scale, uintptr_t dX,
                             uintptr_t dW1, uintptr_t dW2, uintptr_t dWr, uintptr_t stream, uintptr_t dlogit_bias) {
    check(kosmos_moe_backward_router_mxfp8(live(), (const void *)dout, (const void *)G, (const void *)X,
                                           (const void *)W1c, (const void *)W1c_scale, (const void *)W2c,
                                           (const void *)W2c_scale, (const void *)Wr, (const float *)aux_scale,
                                           (void *)dX, (void *)dW1, (void *)dW2, (void *)dWr, (float *)dlogit_bias,
                                           (hipStream_t)stream),
          "kosmos_moe_backward_router_mxfp8", m, &ws->coll);
  }

  void forward_router_shared_mxfp8_train(uintptr_t X, uintptr_t Wr, uintptr_t bias, uintptr_t W1, uintptr_t W1_scale,
                                         uintptr_t W2, uintptr_t W2_scale, uintptr_t Ws1, uintptr_t Ws1_scale,
                                         uintptr_t Ws2, uintptr_t Ws2_scale, uintptr_t out, uintptr_t G, uintptr_t Gs,
                                         uintptr_t aux_loss, uintptr_t tpe, uintptr_t seed, uintptr_t stream,
                                         uintptr_t logit_bias, uintptr_t sgate) {
    check(kosmos_moe_forward_router_shared_mxfp8_train(
              live(), (const void *)X, (const void *)Wr, (const float *)bias, (const float *)logit_bias,
              (const void *)W1, (const void *)W1_scale, (const void *)W2, (const void *)W2_scale, (const void *)Ws1,
              (const void *)Ws1_scale, (const void *)Ws2, (const void *)Ws2_scale, (void *)out, (void *)G, (void *)Gs,
              (const float *)sgate, (float *)aux_loss, (int *)tpe, (const unsigned *)seed, (hipStream_t)stream),
          "kosmos_moe_forward_router_shared_mxfp8_train", m, &ws->coll);
  }

  void backward_router_shared_mxfp8(uintptr_t dout, uintptr_t G, uintptr_t Gs, uintptr_t X, uintptr_t W1c,
                                    uintptr_t W1c_scale, uintptr_t W2c, uintptr_t W2c_scale, uintptr_t Ws1c,
                                    uintptr_t Ws1c_scale, uintptr_t Ws2c, uintptr_t Ws2c_scale, uintptr_t Wr,
                                    uintptr_t aux_scale, uintptr_t dX, uintptr_t dW1, uintptr_t dW2, uintptr_t dWr,
                                    uintptr_t dWs1, uintptr_t dWs2, int acc, uintptr_t stream, uintptr_t dlogit_bias,
                                    uintptr_t sgate, uintptr_t dsgate) {
    check(kosmos_moe_backward_router_shared_mxfp8(
              live(), (const void *)dout, (const void *)G, (const void *)Gs, (const void *)X, (const void *)W1c,
              (const void *)W1c_scale, (const void *)W2c, (const void *)W2c_scale, (const void *)Ws1c,
              (const void *)Ws1c_scale, (const void *)Ws2c, (const void *)Ws2c_scale, (const void *)Wr,
              (const float *)aux_scale, (void *)dX, (void *)dW1, (void *)dW2, (void *)dWr, (float *)dlogit_bias,
              (void *)dWs1, (void *)dWs2, acc, (const float *)sgate, (float *)dsgate, (hipStream_t)stream),
          "kosmos_moe_backward_router_shared_mxfp8", m, &ws->coll);
  }

  void prepare() {
    check(kosmos_moe_prepare(live()), "kosmos_moe_prepare", m, &ws->coll);
  }

  void plan(uintptr_t topk_idx, uintptr_t topk_w) {
    check(kosmos_moe_plan(live(), (const int *)topk_idx, (const float *)topk_w), "kosmos_moe_plan", m, &ws->coll);
  }

  int plan_status() {
    int st = 0;
    check(kosmos_moe_plan_status(live(), &st), "kosmos_moe_plan_status", m, &ws->coll);
    return st;
  }

  py::dict info() {
    const KosmosMoeInfo *i = kosmos_moe_info(live());
    py::dict d;
    d["Mpad"] = i->Mpad, d["Mpad_max"] = i->Mpad_max, d["fwd_ncomm"] = i->fwd_ncomm, d["fwd_fcomm"] = i->fwd_fcomm;
    d["fwd_burst"] = i->fwd_burst, d["nt1"] = i->nt1, d["nt2"] = i->nt2;
    d["bwd_cncomm_clamped"] = i->bwd_cncomm_clamped, d["bwd_grid"] = i->bwd_grid;
    d["bwd_dncomm_all"] = py::cast(std::vector<int>(i->bwd_dncomm_all, i->bwd_dncomm_all + KOSMOS_MOE_EP));
    d["bwd_cncomm_all"] = py::cast(std::vector<int>(i->bwd_cncomm_all, i->bwd_cncomm_all + KOSMOS_MOE_EP));
    d["bwd_dpush_gb"] = i->bwd_dpush_gb, d["bwd_lega_tflop"] = i->bwd_lega_tflop, d["bwd_cpush_gb"] = i->bwd_cpush_gb;
    d["bwd_wg2_tiles"] = i->bwd_wg2_tiles, d["bwd_wg2_meank"] = i->bwd_wg2_meank;
    d["bwd_wg2run"] = i->bwd_wg2run, d["bwd_wg1run"] = i->bwd_wg1run;
    d["bwd_census_dmean"] = i->bwd_census_dmean, d["bwd_census_cmean"] = i->bwd_census_cmean;
    d["bwd_census_dmax"] = i->bwd_census_dmax, d["bwd_census_cmax"] = i->bwd_census_cmax;
    d["disp_self"] = i->disp_self, d["disp_rows"] = i->disp_rows;
    d["comb_self"] = i->comb_self, d["comb_rows"] = i->comb_rows;
    d["fwd_census_max"] = i->fwd_census_max, d["g_bytes"] = i->g_bytes, d["arena_rows"] = i->arena_rows;
    d["plans"] = i->plans;
    return d;
  }

  size_t g_bytes() {
    return kosmos_moe_info(live())->g_bytes;
  }

  std::string error() {
    return kosmos_moe_error(live());
  }

  void destroy() {
    check(kosmos_moe_destroy(m), "kosmos_moe_destroy", nullptr, &ws->coll);
    m = nullptr;
  }
};

static KosmosMoeConfig instance_config(int H, int I, int E_loc, int k, int T, int rank, PyCollective *coll, int wgcap,
                                       int capacity_rows, int act) {
  KosmosMoeConfig cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.H = H, cfg.I = I, cfg.E_loc = E_loc, cfg.k = k, cfg.T = T, cfg.wgcap = wgcap, cfg.backward = 1;
  cfg.capacity_rows = capacity_rows, cfg.act = act;
  cfg.coll.allgather = py_allgather, cfg.coll.ctx = coll, cfg.coll.rank = rank, cfg.coll.nranks = KOSMOS_MOE_EP;
  return cfg;
}

// The bytes of a caller's symmetric buffer for MoeInstance(ext_*): the instance's, then the comm's flags
// (kosmos_comm_wrap carves them from the buffer's end when there is no separate signal pad).
static size_t instance_bytes(int H, int I, int E_loc, int k, int T, int capacity_rows, int rank, py::function allgather,
                             int wgcap, int act) {
  PyCollective pc;
  pc.fn = allgather;
  const KosmosMoeConfig cfg = instance_config(H, I, E_loc, k, T, rank, &pc, wgcap, capacity_rows, act);
  size_t b = 0;
  if (kosmos_moe_instance_bytes(&cfg, &b) != hipSuccess) {
    throw std::invalid_argument("kosmos_moe_instance_bytes refused the configuration (bf16 training, capacity_rows > 0;"
                                " see src/moe/README.md)");
  }
  const size_t n = b + kosmos_comm_flag_bytes(KOSMOS_MOE_EP) + 256;
  return (n + 4095) / 4096 * 4096;
}

// The training API (kosmos_moe_instance_*): saved state in caller buffers, every pointer an integer address.
struct MoeInstance {
  KosmosMoeInstance *in = nullptr;
  KosmosComm *ext = nullptr;
  KosmosMoeConfig cfg;
  PyCollective coll;

  MoeInstance(int H, int I, int E_loc, int k, int T, int rank, py::function allgather, int wgcap, int capacity_rows,
              int act, uintptr_t ext_local, uintptr_t ext_peers, size_t ext_bytes) {
    coll.fn = allgather;
    cfg = instance_config(H, I, E_loc, k, T, rank, &coll, wgcap, capacity_rows, act);
    size_t need = 0;
    if (kosmos_moe_instance_bytes(&cfg, &need) != hipSuccess) {
      throw std::invalid_argument("MoeInstance: configuration refused (bf16 training, capacity_rows > 0; see"
                                  " src/moe/README.md)");
    }
    if (ext_local) {
      KosmosExternalBuffers eb;
      memset(&eb, 0, sizeof eb);
      eb.local = (void *)ext_local;
      eb.peers = (void *const *)ext_peers;
      eb.bytes = ext_bytes;
      eb.rank = rank;
      eb.nranks = KOSMOS_MOE_EP;
      eb.coll = &cfg.coll;
      check(kosmos_comm_wrap(&ext, &eb), "kosmos_comm_wrap", nullptr, &coll);
    }
    const hipError_t e = kosmos_moe_instance_create(&in, ext, &cfg);
    if (e != hipSuccess) {
      if (ext) {
        (void)kosmos_comm_destroy(ext);
        ext = nullptr;
      }
      fail(e, "kosmos_moe_instance_create", nullptr);
    }
  }

  void fail(hipError_t e, const char *what, const KosmosMoeInstance *m) {
    if (e == hipSuccess) {
      return;
    }
    std::string msg = std::string(what) + ": " + hipGetErrorString(e);
    if (m && *kosmos_moe_instance_error(m)) {
      msg += " / " + std::string(kosmos_moe_instance_error(m));
    }
    if (!coll.err.empty()) {
      msg += " / allgather: " + coll.err;
      coll.err.clear();
    }
    throw std::runtime_error(msg);
  }

  KosmosMoeInstance *live() {
    if (!in) {
      throw std::runtime_error("MoeInstance used after destroy()");
    }
    return in;
  }

  py::dict saved_bytes(int rows) {
    KosmosMoeSavedBytes b;
    if (kosmos_moe_saved_bytes(&cfg, rows, &b) != hipSuccess) {
      throw std::invalid_argument("saved_bytes: rows must be in [1, the instance's capacity rows]");
    }
    py::dict d;
    d["xp"] = b.xp, d["g"] = b.g, d["a"] = b.a, d["y"] = b.y, d["rec"] = b.rec;
    return d;
  }

  static KosmosMoeSaved saved(int tier, int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y, uintptr_t rec) {
    return {tier, rows, (void *)xp, (void *)g, (void *)a, (void *)y, (void *)rec};
  }

  void forward(uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t topk_idx, uintptr_t topk_w, uintptr_t out, int tier,
               int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y, uintptr_t rec, uintptr_t stream) {
    const KosmosMoeSaved sv = saved(tier, rows, xp, g, a, y, rec);
    fail(kosmos_moe_train_forward(live(), (const void *)X, (const void *)W1, (const void *)W2, (const int *)topk_idx,
                                  (const float *)topk_w, (void *)out, tier < 0 ? nullptr : &sv, (hipStream_t)stream),
         "kosmos_moe_train_forward", in);
  }

  void backward(uintptr_t dout, uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t topk_idx, uintptr_t topk_w, int tier,
                int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y, uintptr_t rec, uintptr_t dX,
                uintptr_t dprobs, uintptr_t dW1, uintptr_t dW2, int acc, uintptr_t stream) {
    const KosmosMoeSaved sv = saved(tier, rows, xp, g, a, y, rec);
    fail(kosmos_moe_train_backward(live(), (const void *)dout, (const void *)X, (const void *)W1, (const void *)W2,
                                   (const int *)topk_idx, (const float *)topk_w, &sv, (void *)dX, (float *)dprobs,
                                   (void *)dW1, (void *)dW2, acc, (hipStream_t)stream),
         "kosmos_moe_train_backward", in);
  }

  int status() {
    int st = 0;
    fail(kosmos_moe_instance_status(live(), &st), "kosmos_moe_instance_status", in);
    return st;
  }

  py::dict info() {
    const KosmosMoeInfo *i = kosmos_moe_instance_info(live());
    py::dict d;
    d["Mpad"] = i->Mpad, d["Mpad_max"] = i->Mpad_max, d["fwd_ncomm"] = i->fwd_ncomm, d["fwd_fcomm"] = i->fwd_fcomm;
    d["fwd_burst"] = i->fwd_burst, d["bwd_grid"] = i->bwd_grid, d["arena_rows"] = i->arena_rows;
    d["bwd_dncomm"] = i->bwd_dncomm_all[0], d["bwd_cncomm"] = i->bwd_cncomm_all[0];
    d["bwd_wg2run"] = i->bwd_wg2run, d["bwd_wg1run"] = i->bwd_wg1run;
    size_t b = 0;
    (void)kosmos_moe_instance_bytes(&cfg, &b);
    d["instance_bytes"] = b;
    return d;
  }

  std::string error() {
    return kosmos_moe_instance_error(live());
  }

  void destroy() {
    const hipError_t e = kosmos_moe_instance_destroy(in);
    in = nullptr;
    if (ext) {
      (void)kosmos_comm_destroy(ext);
      ext = nullptr;
    }
    fail(e, "kosmos_moe_instance_destroy", nullptr);
  }
};

static void quantize_mxfp8(uintptr_t W, int nmat, int rows, int cols, bool col, uintptr_t q, uintptr_t s,
                           uintptr_t stream) {
  const hipError_t e =
      kosmos_moe_quantize_mxfp8((const void *)W, nmat, rows, cols, col, (void *)q, (void *)s, (hipStream_t)stream);
  if (e != hipSuccess) {
    throw std::runtime_error(std::string("kosmos_moe_quantize_mxfp8: ") + hipGetErrorString(e));
  }
}

static KosmosMoeRouterConfig router_config(int score, int groups, int group_topk, int aux_groups, float aux_coeff,
                                           float scale, int force) {
  return {1, score, groups, group_topk, aux_groups, aux_coeff, scale, force};
}

PYBIND11_MODULE(kosmos, mod) {
  mod.attr("EP") = KOSMOS_MOE_EP;
  mod.attr("ROUTER_SOFTMAX") = KOSMOS_ROUTER_SOFTMAX;
  mod.attr("ROUTER_SIGMOID") = KOSMOS_ROUTER_SIGMOID;
  mod.attr("ROUTER_SQRTSOFTPLUS") = KOSMOS_ROUTER_SQRTSOFTPLUS;
  mod.attr("ROUTER_SOFTMAX_PRE") = KOSMOS_ROUTER_SOFTMAX_PRE;
  mod.attr("ACT_SWIGLU") = KOSMOS_MOE_ACT_SWIGLU;
  mod.attr("ACT_SWIGLU_CLAMP10") = KOSMOS_MOE_ACT_SWIGLU_CLAMP10;
  mod.attr("MOE_SAVE") = (uintptr_t)KOSMOS_MOE_SAVE;
  mod.attr("SAVE_NONE") = (int)KOSMOS_MOE_SAVE_NONE;
  mod.attr("SAVE_XP") = (int)KOSMOS_MOE_SAVE_XP;
  mod.attr("SAVE_ALL") = (int)KOSMOS_MOE_SAVE_ALL;
  mod.def("instance_bytes", &instance_bytes, py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"),
          py::arg("capacity_rows"), py::arg("rank"), py::arg("allgather"), py::arg("wgcap") = 256,
          py::arg("act") = KOSMOS_MOE_ACT_SWIGLU);
  mod.def("quantize_mxfp8", &quantize_mxfp8, py::arg("W"), py::arg("nmat"), py::arg("rows"), py::arg("cols"),
          py::arg("col"), py::arg("q"), py::arg("s"), py::arg("stream"));

  py::class_<KosmosMoeRouterConfig>(mod, "MoeRouterConfig")
      .def(py::init(&router_config), py::arg("score"), py::arg("groups") = 0, py::arg("group_topk") = 0,
           py::arg("aux_groups") = 0, py::arg("aux_coeff") = 0.0f, py::arg("scale") = 1.0f, py::arg("force") = 0)
      .def_readonly("score", &KosmosMoeRouterConfig::score)
      .def_readonly("groups", &KosmosMoeRouterConfig::groups)
      .def_readonly("group_topk", &KosmosMoeRouterConfig::group_topk)
      .def_readonly("aux_groups", &KosmosMoeRouterConfig::aux_groups)
      .def_readonly("aux_coeff", &KosmosMoeRouterConfig::aux_coeff)
      .def_readonly("scale", &KosmosMoeRouterConfig::scale)
      .def_readonly("force", &KosmosMoeRouterConfig::force);

  py::class_<MoeWorkspace>(mod, "MoeWorkspace")
      .def(py::init<int, int, int, int, int, int, py::function, int, int, bool, const KosmosMoeRouterConfig *, bool,
                    bool, bool, int, bool, size_t>(),
           py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"), py::arg("rank"),
           py::arg("allgather"), py::arg("wgcap") = 256, py::arg("capacity_rows") = 0, py::arg("backward") = true,
           py::arg("router") = nullptr, py::arg("shared") = false, py::arg("mxfp8") = false,
           py::arg("fp8_combine") = false, py::arg("act") = KOSMOS_MOE_ACT_SWIGLU, py::arg("probs_in") = false,
           py::arg("pool_bytes") = 0)
      .def("pool_resize", &MoeWorkspace::pool_resize, py::arg("bytes"))
      .def("pool_info", &MoeWorkspace::pool_info)
      .def("set_capacity", &MoeWorkspace::set_capacity, py::arg("rows"))
      .def("capacity_query", &MoeWorkspace::capacity_query)
      .def("capacity_bytes", &MoeWorkspace::capacity_bytes, py::arg("rows"))
      .def("destroy", &MoeWorkspace::destroy);

  py::class_<MoeContext>(mod, "MoeContext")
      .def(py::init<py::object>(), py::arg("workspace"))
      .def("forward", &MoeContext::forward, py::arg("X"), py::arg("W1"), py::arg("W2"), py::arg("topk_idx"),
           py::arg("topk_w"), py::arg("out"), py::arg("G"), py::arg("stream"))
      .def("forward_mxfp8_train", &MoeContext::forward_mxfp8_train, py::arg("X"), py::arg("W1"), py::arg("W1_scale"),
           py::arg("W2"), py::arg("W2_scale"), py::arg("topk_idx"), py::arg("topk_w"), py::arg("out"), py::arg("G"),
           py::arg("stream"))
      .def("backward_mxfp8", &MoeContext::backward_mxfp8, py::arg("dout"), py::arg("G"), py::arg("X"), py::arg("W1c"),
           py::arg("W1c_scale"), py::arg("W2c"), py::arg("W2c_scale"), py::arg("dX"), py::arg("dprobs"),
           py::arg("dW1"), py::arg("dW2"), py::arg("stream"))
      .def("forward_router", &MoeContext::forward_router, py::arg("X"), py::arg("Wr"), py::arg("bias"), py::arg("W1"),
           py::arg("W2"), py::arg("out"), py::arg("G"), py::arg("aux_loss"), py::arg("tpe"), py::arg("seed"),
           py::arg("stream"), py::arg("logit_bias") = 0)
      .def("backward", &MoeContext::backward, py::arg("dout"), py::arg("G"), py::arg("W1"), py::arg("W2"),
           py::arg("dX"), py::arg("dprobs"), py::arg("dW1"), py::arg("dW2"), py::arg("stream"))
      .def("backward_router", &MoeContext::backward_router, py::arg("dout"), py::arg("G"), py::arg("X"),
           py::arg("W1"), py::arg("W2"), py::arg("Wr"), py::arg("aux_scale"), py::arg("dX"), py::arg("dW1"),
           py::arg("dW2"), py::arg("dWr"), py::arg("stream"), py::arg("dlogit_bias") = 0)
      .def("forward_shared", &MoeContext::forward_shared, py::arg("X"), py::arg("W1"), py::arg("W2"), py::arg("Ws1"),
           py::arg("Ws2"), py::arg("topk_idx"), py::arg("topk_w"), py::arg("out"), py::arg("G"), py::arg("Gs"),
           py::arg("stream"), py::arg("sgate") = 0)
      .def("forward_router_shared", &MoeContext::forward_router_shared, py::arg("X"), py::arg("Wr"), py::arg("bias"),
           py::arg("W1"), py::arg("W2"), py::arg("Ws1"), py::arg("Ws2"), py::arg("out"), py::arg("G"), py::arg("Gs"),
           py::arg("aux_loss"), py::arg("tpe"), py::arg("seed"), py::arg("stream"), py::arg("logit_bias") = 0,
           py::arg("sgate") = 0)
      .def("backward_shared", &MoeContext::backward_shared, py::arg("dout"), py::arg("G"), py::arg("Gs"), py::arg("X"),
           py::arg("W1"), py::arg("W2"), py::arg("Ws1"), py::arg("Ws2"), py::arg("dX"), py::arg("dprobs"),
           py::arg("dW1"), py::arg("dW2"), py::arg("dWs1"), py::arg("dWs2"), py::arg("acc"), py::arg("stream"),
           py::arg("sgate") = 0, py::arg("dsgate") = 0)
      .def("backward_router_shared", &MoeContext::backward_router_shared, py::arg("dout"), py::arg("G"), py::arg("Gs"),
           py::arg("X"), py::arg("W1"), py::arg("W2"), py::arg("Ws1"), py::arg("Ws2"), py::arg("Wr"),
           py::arg("aux_scale"), py::arg("dX"), py::arg("dW1"), py::arg("dW2"), py::arg("dWr"), py::arg("dWs1"),
           py::arg("dWs2"), py::arg("acc"), py::arg("stream"), py::arg("dlogit_bias") = 0, py::arg("sgate") = 0,
           py::arg("dsgate") = 0)
      .def("forward_shared_mxfp8_train", &MoeContext::forward_shared_mxfp8_train, py::arg("X"), py::arg("W1"),
           py::arg("W1_scale"), py::arg("W2"), py::arg("W2_scale"), py::arg("Ws1"), py::arg("Ws1_scale"),
           py::arg("Ws2"), py::arg("Ws2_scale"), py::arg("topk_idx"), py::arg("topk_w"), py::arg("out"), py::arg("G"),
           py::arg("Gs"), py::arg("stream"), py::arg("sgate") = 0)
      .def("backward_shared_mxfp8", &MoeContext::backward_shared_mxfp8, py::arg("dout"), py::arg("G"), py::arg("Gs"),
           py::arg("X"), py::arg("W1c"), py::arg("W1c_scale"), py::arg("W2c"), py::arg("W2c_scale"), py::arg("Ws1c"),
           py::arg("Ws1c_scale"), py::arg("Ws2c"), py::arg("Ws2c_scale"), py::arg("dX"), py::arg("dprobs"),
           py::arg("dW1"), py::arg("dW2"), py::arg("dWs1"), py::arg("dWs2"), py::arg("acc"), py::arg("stream"),
           py::arg("sgate") = 0, py::arg("dsgate") = 0)
      .def("forward_router_mxfp8_train", &MoeContext::forward_router_mxfp8_train, py::arg("X"), py::arg("Wr"),
           py::arg("bias"), py::arg("W1"), py::arg("W1_scale"), py::arg("W2"), py::arg("W2_scale"), py::arg("out"),
           py::arg("G"), py::arg("aux_loss"), py::arg("tpe"), py::arg("seed"), py::arg("stream"),
           py::arg("logit_bias") = 0)
      .def("backward_router_mxfp8", &MoeContext::backward_router_mxfp8, py::arg("dout"), py::arg("G"), py::arg("X"),
           py::arg("W1c"), py::arg("W1c_scale"), py::arg("W2c"), py::arg("W2c_scale"), py::arg("Wr"),
           py::arg("aux_scale"), py::arg("dX"), py::arg("dW1"), py::arg("dW2"), py::arg("dWr"), py::arg("stream"),
           py::arg("dlogit_bias") = 0)
      .def("forward_router_shared_mxfp8_train", &MoeContext::forward_router_shared_mxfp8_train, py::arg("X"),
           py::arg("Wr"), py::arg("bias"), py::arg("W1"), py::arg("W1_scale"), py::arg("W2"), py::arg("W2_scale"),
           py::arg("Ws1"), py::arg("Ws1_scale"), py::arg("Ws2"), py::arg("Ws2_scale"), py::arg("out"), py::arg("G"),
           py::arg("Gs"), py::arg("aux_loss"), py::arg("tpe"), py::arg("seed"), py::arg("stream"),
           py::arg("logit_bias") = 0, py::arg("sgate") = 0)
      .def("backward_router_shared_mxfp8", &MoeContext::backward_router_shared_mxfp8, py::arg("dout"), py::arg("G"),
           py::arg("Gs"), py::arg("X"), py::arg("W1c"), py::arg("W1c_scale"), py::arg("W2c"), py::arg("W2c_scale"),
           py::arg("Ws1c"), py::arg("Ws1c_scale"), py::arg("Ws2c"), py::arg("Ws2c_scale"), py::arg("Wr"),
           py::arg("aux_scale"), py::arg("dX"), py::arg("dW1"), py::arg("dW2"), py::arg("dWr"), py::arg("dWs1"),
           py::arg("dWs2"), py::arg("acc"), py::arg("stream"), py::arg("dlogit_bias") = 0, py::arg("sgate") = 0,
           py::arg("dsgate") = 0)
      .def("prepare", &MoeContext::prepare)
      .def("plan", &MoeContext::plan, py::arg("topk_idx"), py::arg("topk_w"))
      .def("plan_status", &MoeContext::plan_status)
      .def("info", &MoeContext::info)
      .def_property_readonly("g_bytes", &MoeContext::g_bytes)
      .def("error", &MoeContext::error)
      .def("destroy", &MoeContext::destroy);

  py::class_<MoeInstance>(mod, "MoeInstance")
      .def(py::init<int, int, int, int, int, int, py::function, int, int, int, uintptr_t, uintptr_t, size_t>(),
           py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"), py::arg("rank"),
           py::arg("allgather"), py::arg("wgcap") = 256, py::arg("capacity_rows") = 0,
           py::arg("act") = KOSMOS_MOE_ACT_SWIGLU, py::arg("ext_local") = 0, py::arg("ext_peers") = 0,
           py::arg("ext_bytes") = 0)
      .def("saved_bytes", &MoeInstance::saved_bytes, py::arg("rows"))
      .def("forward", &MoeInstance::forward, py::arg("X"), py::arg("W1"), py::arg("W2"), py::arg("topk_idx"),
           py::arg("topk_w"), py::arg("out"), py::arg("tier"), py::arg("rows"), py::arg("xp"), py::arg("g"),
           py::arg("a"), py::arg("y"), py::arg("rec"), py::arg("stream"))
      .def("backward", &MoeInstance::backward, py::arg("dout"), py::arg("X"), py::arg("W1"), py::arg("W2"),
           py::arg("topk_idx"), py::arg("topk_w"), py::arg("tier"), py::arg("rows"), py::arg("xp"), py::arg("g"),
           py::arg("a"), py::arg("y"), py::arg("rec"), py::arg("dX"), py::arg("dprobs"), py::arg("dW1"),
           py::arg("dW2"), py::arg("acc"), py::arg("stream"))
      .def("status", &MoeInstance::status)
      .def("info", &MoeInstance::info)
      .def("error", &MoeInstance::error)
      .def("destroy", &MoeInstance::destroy);
}
