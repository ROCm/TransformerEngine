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

static void check(hipError_t e, const char *what, PyCollective *coll) {
  if (e == hipSuccess) {
    return;
  }
  std::string msg = std::string(what) + ": " + hipGetErrorString(e);
  if (!coll->err.empty()) {
    msg += " / allgather: " + coll->err;
    coll->err.clear();
  }
  throw std::runtime_error(msg);
}

static KosmosMoeConfig instance_config(int H, int I, int E_loc, int k, int T, int rank, PyCollective *coll, int wgcap,
                                       int capacity_rows, int act) {
  KosmosMoeConfig cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.H = H, cfg.I = I, cfg.E_loc = E_loc, cfg.k = k, cfg.T = T, cfg.wgcap = wgcap;
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
      check(kosmos_comm_wrap(&ext, &eb), "kosmos_comm_wrap", &coll);
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
    d["xp"] = b.xp, d["g"] = b.g, d["a"] = b.a, d["y"] = b.y, d["route"] = b.route;
    return d;
  }

  static KosmosMoeSaved saved(int tier, int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y,
                              uintptr_t route) {
    return {tier, rows, (void *)xp, (void *)g, (void *)a, (void *)y, (void *)route};
  }

  void forward(uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t topk_idx, uintptr_t topk_w, uintptr_t out, int tier,
               int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y, uintptr_t route, uintptr_t stream) {
    const KosmosMoeSaved sv = saved(tier, rows, xp, g, a, y, route);
    fail(kosmos_moe_train_forward(live(), (const void *)X, (const void *)W1, (const void *)W2, (const int *)topk_idx,
                                  (const float *)topk_w, (void *)out, tier < 0 ? nullptr : &sv, (hipStream_t)stream),
         "kosmos_moe_train_forward", in);
  }

  void backward(uintptr_t dout, uintptr_t X, uintptr_t W1, uintptr_t W2, uintptr_t topk_idx, uintptr_t topk_w, int tier,
                int rows, uintptr_t xp, uintptr_t g, uintptr_t a, uintptr_t y, uintptr_t route, uintptr_t dX,
                uintptr_t dprobs, uintptr_t dW1, uintptr_t dW2, int acc, uintptr_t stream) {
    const KosmosMoeSaved sv = saved(tier, rows, xp, g, a, y, route);
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

PYBIND11_MODULE(kosmos, mod) {
  mod.attr("EP") = KOSMOS_MOE_EP;
  mod.attr("ACT_SWIGLU") = KOSMOS_MOE_ACT_SWIGLU;
  mod.attr("ACT_SWIGLU_CLAMP10") = KOSMOS_MOE_ACT_SWIGLU_CLAMP10;
  mod.attr("SAVE_NONE") = (int)KOSMOS_MOE_SAVE_NONE;
  mod.attr("SAVE_XP") = (int)KOSMOS_MOE_SAVE_XP;
  mod.attr("SAVE_ALL") = (int)KOSMOS_MOE_SAVE_ALL;
  mod.def("instance_bytes", &instance_bytes, py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"),
          py::arg("capacity_rows"), py::arg("rank"), py::arg("allgather"), py::arg("wgcap") = 256,
          py::arg("act") = KOSMOS_MOE_ACT_SWIGLU);

  py::class_<MoeInstance>(mod, "MoeInstance")
      .def(py::init<int, int, int, int, int, int, py::function, int, int, int, uintptr_t, uintptr_t, size_t>(),
           py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"), py::arg("rank"),
           py::arg("allgather"), py::arg("wgcap") = 256, py::arg("capacity_rows") = 0,
           py::arg("act") = KOSMOS_MOE_ACT_SWIGLU, py::arg("ext_local") = 0, py::arg("ext_peers") = 0,
           py::arg("ext_bytes") = 0)
      .def("saved_bytes", &MoeInstance::saved_bytes, py::arg("rows"))
      .def("forward", &MoeInstance::forward, py::arg("X"), py::arg("W1"), py::arg("W2"), py::arg("topk_idx"),
           py::arg("topk_w"), py::arg("out"), py::arg("tier"), py::arg("rows"), py::arg("xp"), py::arg("g"),
           py::arg("a"), py::arg("y"), py::arg("route"), py::arg("stream"))
      .def("backward", &MoeInstance::backward, py::arg("dout"), py::arg("X"), py::arg("W1"), py::arg("W2"),
           py::arg("topk_idx"), py::arg("topk_w"), py::arg("tier"), py::arg("rows"), py::arg("xp"), py::arg("g"),
           py::arg("a"), py::arg("y"), py::arg("route"), py::arg("dX"), py::arg("dprobs"), py::arg("dW1"),
           py::arg("dW2"), py::arg("acc"), py::arg("stream"))
      .def("status", &MoeInstance::status)
      .def("info", &MoeInstance::info)
      .def("error", &MoeInstance::error)
      .def("destroy", &MoeInstance::destroy);
}
