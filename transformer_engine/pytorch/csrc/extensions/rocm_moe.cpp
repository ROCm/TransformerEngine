/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

// transformer_engine_torch.rocm_moe: PyTorch binding of the fused MoE training (nvte_rocm_moe_*),
// used by transformer_engine.pytorch.kosmos.

#include <transformer_engine/rocm_moe.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

namespace {

// The EP group's host all-gather: fn(send: bytes) returns the ranks' sends concatenated in rank
// order (any bytes-like object).
struct PyAllgather {
  py::function fn;
  std::string err;
};

int py_allgather(void *ctx, const void *send, void *recv, size_t bytes) {
  auto *c = static_cast<PyAllgather *>(ctx);
  py::gil_scoped_acquire gil;
  try {
    py::object res = c->fn(py::bytes(static_cast<const char *>(send), bytes));
    py::bytes all = py::reinterpret_steal<py::bytes>(PyBytes_FromObject(res.ptr()));
    if (!all) {
      throw py::error_already_set();
    }
    const std::string_view v = all;
    if (v.size() != bytes * NVTE_ROCM_MOE_EP_SIZE) {
      c->err = "allgather returned " + std::to_string(v.size()) + " bytes, expected " +
               std::to_string(bytes * NVTE_ROCM_MOE_EP_SIZE);
      return 1;
    }
    std::memcpy(recv, v.data(), v.size());
    return 0;
  } catch (const std::exception &e) {
    c->err = e.what();
    return 1;
  }
}

NVTERocmMoeConfig make_config(int H, int I, int E_loc, int k, int T, int capacity_rows, int wgcap,
                              int act) {
  NVTERocmMoeConfig cfg{};
  cfg.struct_size = sizeof(cfg);
  cfg.hidden_size = H;
  cfg.ffn_hidden_size = I;
  cfg.num_local_experts = E_loc;
  cfg.topk = k;
  cfg.num_tokens = T;
  cfg.capacity_rows = capacity_rows;
  cfg.wgcap = wgcap;
  cfg.activation = act;
  return cfg;
}

// The bytes of a caller's symmetric buffer for RocmMoeInstance(symmetric_*).
size_t symmetric_buffer_bytes(int H, int I, int E_loc, int k, int T, int capacity_rows, int wgcap,
                              int act) {
  const NVTERocmMoeConfig cfg = make_config(H, I, E_loc, k, T, capacity_rows, wgcap, act);
  if (!nvte_rocm_moe_config_valid(&cfg)) {
    throw std::invalid_argument(
        "rocm_moe: configuration refused (bf16 training, capacity_rows > 0; see "
        "transformer_engine/common/include/transformer_engine/rocm_moe.h)");
  }
  return nvte_rocm_moe_symmetric_buffer_bytes(&cfg);
}

TensorWrapper wrap(const at::Tensor &t, const char *name) {
  NVTE_CHECK(t.is_contiguous(), "rocm_moe: ", name, " must be contiguous");
  return makeTransformerEngineTensor(t);
}

}  // namespace

// One fused-MoE training instance (nvte_rocm_moe_instance_*). Collective to construct and destroy;
// an instance that is never destroyed is freed at process exit.
class RocmMoeInstance {
 public:
  RocmMoeInstance(int H, int I, int E_loc, int k, int T, int rank, py::function allgather,
                  int wgcap, int capacity_rows, int act, uintptr_t symmetric_local,
                  uintptr_t symmetric_peers, size_t symmetric_bytes)
      : cfg_(make_config(H, I, E_loc, k, T, capacity_rows, wgcap, act)),
        coll_(std::make_unique<PyAllgather>()) {
    if (!nvte_rocm_moe_config_valid(&cfg_)) {
      throw std::invalid_argument(
          "RocmMoeInstance: configuration refused (bf16 training, capacity_rows > 0; see "
          "transformer_engine/common/include/transformer_engine/rocm_moe.h)");
    }
    coll_->fn = std::move(allgather);
    NVTERocmMoeComm comm{};
    comm.struct_size = sizeof(comm);
    comm.ctx = coll_.get();
    comm.rank = rank;
    comm.nranks = NVTE_ROCM_MOE_EP_SIZE;
    comm.allgather = py_allgather;
    NVTERocmMoeSymmetricBuffer buf{};
    buf.struct_size = sizeof(buf);
    buf.local = reinterpret_cast<void *>(symmetric_local);
    buf.peers = reinterpret_cast<void *const *>(symmetric_peers);
    buf.bytes = symmetric_bytes;
    run([&] {
      inst_ = nvte_rocm_moe_instance_create(&cfg_, &comm, symmetric_local ? &buf : nullptr);
    });
  }

  py::dict saved_bytes(int rows) {
    NVTERocmMoeSavedBytes b{};
    b.struct_size = sizeof(b);
    try {
      nvte_rocm_moe_saved_bytes(&cfg_, rows, &b);
    } catch (const std::exception &) {
      throw std::invalid_argument("saved_bytes: rows must be in [1, the instance's capacity rows]");
    }
    py::dict d;
    d["xp"] = b.xp;
    d["g"] = b.g;
    d["a"] = b.a;
    d["y"] = b.y;
    d["route"] = b.route;
    return d;
  }

  // tier < 0: an inference forward, nothing kept.
  void forward(const at::Tensor &x, const at::Tensor &w1, const at::Tensor &w2,
               const at::Tensor &topk_idx, const at::Tensor &topk_w, const at::Tensor &out,
               int tier, int rows, const std::optional<at::Tensor> &xp,
               const std::optional<at::Tensor> &g, const std::optional<at::Tensor> &a,
               const std::optional<at::Tensor> &y, const std::optional<at::Tensor> &route) {
    std::vector<TensorWrapper> keep;
    NVTERocmMoeSaved sv = saved(tier, rows, xp, g, a, y, route, &keep);
    TensorWrapper tx = wrap(x, "x");
    TensorWrapper tw1 = wrap(w1, "w1");
    TensorWrapper tw2 = wrap(w2, "w2");
    TensorWrapper tidx = wrap(topk_idx, "topk_idx");
    TensorWrapper tw = wrap(topk_w, "topk_w");
    TensorWrapper tout = wrap(out, "out");
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    run([&] {
      nvte_rocm_moe_train_forward(live(), tx.data(), tw1.data(), tw2.data(), tidx.data(), tw.data(),
                                  tout.data(), tier < 0 ? nullptr : &sv, stream);
    });
  }

  void backward(const at::Tensor &dout, const at::Tensor &x, const at::Tensor &w1,
                const at::Tensor &w2, const at::Tensor &topk_idx, const at::Tensor &topk_w,
                int tier, int rows, const std::optional<at::Tensor> &xp,
                const std::optional<at::Tensor> &g, const std::optional<at::Tensor> &a,
                const std::optional<at::Tensor> &y, const std::optional<at::Tensor> &route,
                const at::Tensor &dx, const at::Tensor &dprobs, const at::Tensor &dw1,
                const at::Tensor &dw2, int acc) {
    std::vector<TensorWrapper> keep;
    NVTERocmMoeSaved sv = saved(tier, rows, xp, g, a, y, route, &keep);
    TensorWrapper tdout = wrap(dout, "dout");
    TensorWrapper tx = wrap(x, "x");
    TensorWrapper tw1 = wrap(w1, "w1");
    TensorWrapper tw2 = wrap(w2, "w2");
    TensorWrapper tidx = wrap(topk_idx, "topk_idx");
    TensorWrapper tw = wrap(topk_w, "topk_w");
    TensorWrapper tdx = wrap(dx, "dx");
    TensorWrapper tdprobs = wrap(dprobs, "dprobs");
    TensorWrapper tdw1 = wrap(dw1, "dw1");
    TensorWrapper tdw2 = wrap(dw2, "dw2");
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    run([&] {
      nvte_rocm_moe_train_backward(live(), tdout.data(), tx.data(), tw1.data(), tw2.data(),
                                   tidx.data(), tw.data(), &sv, tdx.data(), tdprobs.data(),
                                   tdw1.data(), tdw2.data(), acc, stream);
    });
  }

  int status() {
    int st = 0;
    run([&] { st = nvte_rocm_moe_instance_status(live()); });
    return st;
  }

  py::dict info() {
    NVTERocmMoeInfo i{};
    i.struct_size = sizeof(i);
    run([&] { nvte_rocm_moe_instance_info(live(), &i); });
    py::dict d;
    d["Mpad"] = i.Mpad;
    d["Mpad_max"] = i.Mpad_max;
    d["fwd_ncomm"] = i.fwd_ncomm;
    d["fwd_fcomm"] = i.fwd_fcomm;
    d["fwd_burst"] = i.fwd_burst;
    d["bwd_grid"] = i.bwd_grid;
    d["arena_rows"] = i.arena_rows;
    d["bwd_dncomm"] = i.bwd_dncomm_all[0];
    d["bwd_cncomm"] = i.bwd_cncomm_all[0];
    d["bwd_wg2run"] = i.bwd_wg2run;
    d["bwd_wg1run"] = i.bwd_wg1run;
    d["instance_bytes"] = nvte_rocm_moe_instance_bytes(&cfg_);
    return d;
  }

  std::string error() { return nvte_rocm_moe_instance_error(live()); }

  void destroy() {
    NVTERocmMoeInstance in = inst_;
    inst_ = nullptr;
    run([&] { nvte_rocm_moe_instance_destroy(in); });
  }

 private:
  NVTERocmMoeInstance live() const {
    if (inst_ == nullptr) {
      throw std::runtime_error("RocmMoeInstance used after destroy()");
    }
    return inst_;
  }

  // Runs a C API call, adding the all-gather callback's failure (if any) to its error.
  template <typename F>
  void run(F &&f) {
    try {
      f();
    } catch (const std::exception &e) {
      std::string msg = e.what();
      if (!coll_->err.empty()) {
        msg += " / allgather: " + coll_->err;
        coll_->err.clear();
      }
      throw std::runtime_error(msg);
    }
  }

  // The saved state's tensors, wrapped into keep (which must outlive the call).
  static NVTERocmMoeSaved saved(int tier, int rows, const std::optional<at::Tensor> &xp,
                                const std::optional<at::Tensor> &g,
                                const std::optional<at::Tensor> &a,
                                const std::optional<at::Tensor> &y,
                                const std::optional<at::Tensor> &route,
                                std::vector<TensorWrapper> *keep) {
    NVTERocmMoeSaved sv{};
    sv.struct_size = sizeof(sv);
    sv.tier = tier;
    sv.rows = rows;
    keep->reserve(5);
    auto t = [keep](const std::optional<at::Tensor> &o, const char *name) -> NVTETensor {
      if (!o.has_value()) {
        return nullptr;
      }
      keep->push_back(wrap(*o, name));
      return keep->back().data();
    };
    sv.xp = t(xp, "xp");
    sv.g = t(g, "g");
    sv.a = t(a, "a");
    sv.y = t(y, "y");
    sv.route = t(route, "route");
    return sv;
  }

  NVTERocmMoeInstance inst_ = nullptr;
  NVTERocmMoeConfig cfg_;
  std::unique_ptr<PyAllgather> coll_;
};

void register_rocm_moe_bindings(py::module_ &m) {
  auto mod = m.def_submodule(
      "rocm_moe", "Fused expert-parallel MoE training on ROCm (KOSMOS, gfx950): nvte_rocm_moe_*.");
  mod.def("is_built", &nvte_rocm_moe_is_built,
          "Whether Transformer Engine was built with the fused MoE.");
  mod.attr("EP_SIZE") = NVTE_ROCM_MOE_EP_SIZE;
  mod.attr("ACT_SWIGLU") = static_cast<int>(kNVTERocmMoeSwiGLU);
  mod.attr("ACT_SWIGLU_CLAMP10") = static_cast<int>(kNVTERocmMoeSwiGLUClamp10);
  mod.attr("SAVE_NONE") = static_cast<int>(kNVTERocmMoeSaveNone);
  mod.attr("SAVE_XP") = static_cast<int>(kNVTERocmMoeSaveXp);
  mod.attr("SAVE_ALL") = static_cast<int>(kNVTERocmMoeSaveAll);
  mod.def("symmetric_buffer_bytes", &symmetric_buffer_bytes, py::arg("H"), py::arg("I"),
          py::arg("E_loc"), py::arg("k"), py::arg("T"), py::arg("capacity_rows"),
          py::arg("wgcap") = 256, py::arg("act") = static_cast<int>(kNVTERocmMoeSwiGLU),
          "Bytes to allocate for the symmetric buffer of RocmMoeInstance(symmetric_*).");
  py::class_<RocmMoeInstance>(mod, "RocmMoeInstance")
      .def(py::init<int, int, int, int, int, int, py::function, int, int, int, uintptr_t, uintptr_t,
                    size_t>(),
           py::arg("H"), py::arg("I"), py::arg("E_loc"), py::arg("k"), py::arg("T"),
           py::arg("rank"), py::arg("allgather"), py::arg("wgcap") = 256,
           py::arg("capacity_rows") = 0, py::arg("act") = static_cast<int>(kNVTERocmMoeSwiGLU),
           py::arg("symmetric_local") = 0, py::arg("symmetric_peers") = 0,
           py::arg("symmetric_bytes") = 0)
      .def("saved_bytes", &RocmMoeInstance::saved_bytes, py::arg("rows"))
      .def("forward", &RocmMoeInstance::forward, py::arg("x"), py::arg("w1"), py::arg("w2"),
           py::arg("topk_idx"), py::arg("topk_w"), py::arg("out"), py::arg("tier"), py::arg("rows"),
           py::arg("xp"), py::arg("g"), py::arg("a"), py::arg("y"), py::arg("route"))
      .def("backward", &RocmMoeInstance::backward, py::arg("dout"), py::arg("x"), py::arg("w1"),
           py::arg("w2"), py::arg("topk_idx"), py::arg("topk_w"), py::arg("tier"), py::arg("rows"),
           py::arg("xp"), py::arg("g"), py::arg("a"), py::arg("y"), py::arg("route"), py::arg("dx"),
           py::arg("dprobs"), py::arg("dw1"), py::arg("dw2"), py::arg("acc"))
      .def("status", &RocmMoeInstance::status)
      .def("info", &RocmMoeInstance::info)
      .def("error", &RocmMoeInstance::error)
      .def("destroy", &RocmMoeInstance::destroy);
}

}  // namespace transformer_engine::pytorch
