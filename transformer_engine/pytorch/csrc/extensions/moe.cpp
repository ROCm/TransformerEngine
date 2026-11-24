/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// PyTorch binding of the fused expert-parallel MoE (nvte_rocm_moe_*): workspace and layer handles.

#include <c10/core/Allocator.h>
#include <transformer_engine/moe.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../extensions.h"

namespace transformer_engine::pytorch {

namespace {

// The host all-gather over the EP group: fn(send: bytes) returns the ranks' sends concatenated in rank order.
// The EP group's host all-gather, and the allocator of the buffers the ranks share (ipc_alloc(bytes) -> pointer,
// ipc_free(pointer)): PyTorch tensors in a dedicated memory pool of plain hipMalloc segments.
struct PyAllgather {
  py::function fn;
  py::function ipc_alloc;
  py::function ipc_free;
  int nranks = 0;
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
    if (v.size() != bytes * c->nranks) {
      c->err = "allgather returned " + std::to_string(v.size()) + " bytes, expected " +
               std::to_string(bytes * c->nranks);
      return 1;
    }
    std::memcpy(recv, v.data(), v.size());
    return 0;
  } catch (const std::exception &e) {
    c->err = e.what();
    return 1;
  }
}

// Every byte the fused MoE allocates comes from PyTorch, so torch.cuda.memory_allocated and the allocator's
// out-of-memory handling count it. Buffers the ranks share (ipc) come from a dedicated MemPool of plain hipMalloc
// segments (IPC handles work on them whatever the allocator's expandable_segments setting, and they hold nothing
// else a peer could reach); the rest from the caching allocator.
int torch_alloc(void *ctx, size_t bytes, int ipc, void **ptr) {
  *ptr = nullptr;
  try {
    if (ipc) {
      auto *c = static_cast<PyAllgather *>(ctx);
      py::gil_scoped_acquire gil;
      *ptr = reinterpret_cast<void *>(c->ipc_alloc(bytes).cast<uintptr_t>());
    } else {
      *ptr = c10::GetAllocator(c10::DeviceType::CUDA)->raw_allocate(bytes);
    }
    return *ptr ? 0 : 1;
  } catch (const std::exception &e) {
    if (ipc) {
      static_cast<PyAllgather *>(ctx)->err = e.what();
    }
    *ptr = nullptr;
    return 1;
  }
}

// The registry (KOSMOS) remembers which kind each pointer is: ipc pointers are the pool's.
void torch_free(void *ctx, void *ptr) {
  // Frees happen at teardown or a resize: finish the device's work first, whatever stream it ran on, before the
  // caching allocator may hand the block to another stream.
  (void)cudaDeviceSynchronize();
  auto *c = static_cast<PyAllgather *>(ctx);
  {
    py::gil_scoped_acquire gil;
    if (c->ipc_free(reinterpret_cast<uintptr_t>(ptr)).cast<bool>()) {
      return;
    }
  }
  c10::GetAllocator(c10::DeviceType::CUDA)->raw_deallocate(ptr);
}

TensorWrapper wrap(const py::object &t) {
  if (t.is_none()) {
    return TensorWrapper();
  }
  return makeTransformerEngineTensor(t, py::none());
}

}  // namespace

class MoeWorkspace {
 public:
  MoeWorkspace(int hidden_size, int ffn_hidden_size, int num_local_experts, int topk, int num_tokens, int rank,
               py::function allgather, py::function ipc_alloc, py::function ipc_free, int capacity_rows, int wgcap,
               bool training, size_t pool_bytes, int activation, bool mxfp8, bool fp8_combine, bool probs_on_input,
               bool shared_expert, bool router, int router_score, int router_groups, int router_group_topk,
               int router_aux_groups, float router_aux_coeff, float router_scale)
      : coll_(std::make_unique<PyAllgather>()) {
    NVTERocmMoeConfig cfg = NVTE_ROCM_MOE_CONFIG_INIT;
    cfg.hidden_size = hidden_size;
    cfg.ffn_hidden_size = ffn_hidden_size;
    cfg.num_local_experts = num_local_experts;
    cfg.topk = topk;
    cfg.num_tokens = num_tokens;
    cfg.wgcap = wgcap;
    cfg.capacity_rows = capacity_rows;
    cfg.pool_bytes = pool_bytes;
    cfg.training = training;
    cfg.activation = activation;
    cfg.mxfp8 = mxfp8;
    cfg.fp8_combine = fp8_combine;
    cfg.probs_on_input = probs_on_input;
    cfg.shared_expert = shared_expert;
    cfg.router = router;
    cfg.router_config = {router_score, router_groups, router_group_topk, router_aux_groups, router_aux_coeff,
                         router_scale};
    char why[512];
    if (!nvte_rocm_moe_config_valid(&cfg, why, sizeof(why))) {
      throw std::invalid_argument(std::string("fused MoE configuration refused: ") + why);
    }
    coll_->fn = std::move(allgather);
    coll_->ipc_alloc = std::move(ipc_alloc);
    coll_->ipc_free = std::move(ipc_free);
    coll_->nranks = NVTE_ROCM_MOE_EP_SIZE;
    NVTERocmMoeComm comm = NVTE_ROCM_MOE_COMM_INIT;
    comm.ctx = coll_.get();
    comm.rank = rank;
    comm.nranks = NVTE_ROCM_MOE_EP_SIZE;
    comm.allgather = py_allgather;
    comm.alloc = torch_alloc;
    comm.free = torch_free;
    run([&] { ws_ = nvte_rocm_moe_workspace_create(&cfg, &comm); });
  }

  // Collective; every layer handle on the workspace must be destroyed first.
  void destroy() {
    if (ws_) {
      NVTERocmMoeWorkspace ws = ws_;
      ws_ = nullptr;
      run([&] { nvte_rocm_moe_workspace_destroy(ws); });
    }
  }

  int pool_resize(size_t bytes) {
    int r = 0;
    run([&] { r = nvte_rocm_moe_pool_resize(live(), bytes); });
    return r;
  }

  py::dict pool_info() {
    size_t b = 0, h = 0, r = 0;
    run([&] { nvte_rocm_moe_pool_info(live(), &b, &h, &r); });
    py::dict d;
    d["bytes"] = b;
    d["head_bytes"] = h;
    d["reserve_bytes"] = r;
    return d;
  }

  int set_capacity(int rows) {
    int r = 0;
    run([&] { r = nvte_rocm_moe_set_capacity(live(), rows); });
    return r;
  }

  py::dict capacity_query() {
    NVTERocmMoeCapacityStat s{};
    run([&] { nvte_rocm_moe_capacity_query(live(), &s); });
    py::dict d;
    d["peak"] = s.peak;
    d["rows"] = s.rows;
    d["max_rows"] = s.max_rows;
    d["nlayers"] = s.nlayers;
    d["rounds"] = s.rounds;
    d["cap_rounds"] = s.cap_rounds;
    return d;
  }

  // Reads every launched forward's release agreement (host-blocking; call it alike on every rank).
  void release_sync() {
    run([&] { nvte_rocm_moe_release_sync(live()); });
  }

  size_t capacity_bytes(int rows) {
    size_t b = 0;
    run([&] { b = nvte_rocm_moe_capacity_bytes(live(), rows); });
    return b;
  }

  NVTERocmMoeWorkspace live() const {
    if (!ws_) {
      throw std::runtime_error("MoeWorkspace used after destroy()");
    }
    return ws_;
  }

  // Runs a call, adding the all-gather callback's failure (if any) to its error.
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

 private:
  NVTERocmMoeWorkspace ws_ = nullptr;
  std::unique_ptr<PyAllgather> coll_;
};

class MoeLayer {
 public:
  explicit MoeLayer(py::object workspace) : ws_obj_(std::move(workspace)) {
    ws_ = ws_obj_.cast<MoeWorkspace *>();
    ws_->run([&] { layer_ = nvte_rocm_moe_layer_create(ws_->live()); });
  }

  // Collective.
  void destroy() {
    if (layer_) {
      NVTERocmMoeLayer l = layer_;
      layer_ = nullptr;
      ws_->run([&] { nvte_rocm_moe_layer_destroy(l); });
    }
  }

  void prepare() {
    ws_->run([&] { nvte_rocm_moe_layer_prepare(live()); });
  }

  void reset() {
    ws_->run([&] { nvte_rocm_moe_layer_reset(live()); });
  }

  int status() {
    int s = 0;
    ws_->run([&] { s = nvte_rocm_moe_layer_status(live()); });
    return s;
  }

  // The saving forward will get no backward: local and thread-safe; a no-op on a destroyed handle (finalizers).
  void release() {
    if (layer_) {
      nvte_rocm_moe_layer_release(layer_);
    }
  }

  int release_state() {
    int s = 0;
    ws_->run([&] { s = nvte_rocm_moe_layer_release_state(live()); });
    return s;
  }

  void forward(py::object x, py::object topk_idx, py::object topk_weights, py::object fc1_weight,
               py::object fc2_weight, py::object out, bool save_for_backward, py::object router_weight,
               py::object expert_bias, py::object router_bias, py::object seed, py::object aux_loss,
               py::object tokens_per_expert, py::object shared_fc1_weight, py::object shared_fc2_weight,
               py::object shared_gate, py::object shared_pre_act) {
    std::vector<TensorWrapper> keep;
    keep.reserve(16);
    auto t = [&keep](const py::object &o) {
      keep.push_back(wrap(o));
      return keep.back().data();
    };
    NVTERocmMoeForwardArgs a{};
    a.struct_size = sizeof(a);
    a.x = t(x);
    a.topk_idx = t(topk_idx);
    a.topk_weights = t(topk_weights);
    a.router_weight = t(router_weight);
    a.expert_bias = t(expert_bias);
    a.router_bias = t(router_bias);
    a.seed = t(seed);
    a.fc1_weight = t(fc1_weight);
    a.fc2_weight = t(fc2_weight);
    a.shared_fc1_weight = t(shared_fc1_weight);
    a.shared_fc2_weight = t(shared_fc2_weight);
    a.shared_gate = t(shared_gate);
    a.shared_pre_act = t(shared_pre_act);
    a.out = t(out);
    a.aux_loss = t(aux_loss);
    a.tokens_per_expert = t(tokens_per_expert);
    a.save_for_backward = save_for_backward;
    const auto stream = at::cuda::getCurrentCUDAStream();
    // Without the GIL: the call may wait for an earlier launch (the release bookkeeping, KOSMOS_MOE_REL_LAG
    // launches back); the callbacks that need Python take it back.
    ws_->run([&] {
      py::gil_scoped_release nogil;
      nvte_rocm_moe_forward(live(), &a, stream);
    });
  }

  void backward(py::object dout, py::object x, py::object fc1_weight, py::object fc2_weight, py::object dx,
                py::object dprobs, py::object fc1_wgrad, py::object fc2_wgrad, py::object router_weight,
                py::object aux_loss_scale, py::object router_wgrad, py::object router_bias_grad,
                py::object shared_fc1_weight, py::object shared_fc2_weight, py::object shared_gate,
                py::object shared_pre_act, py::object shared_fc1_wgrad, py::object shared_fc2_wgrad,
                py::object shared_gate_grad, int wgrad_accumulate) {
    std::vector<TensorWrapper> keep;
    keep.reserve(20);
    auto t = [&keep](const py::object &o) {
      keep.push_back(wrap(o));
      return keep.back().data();
    };
    NVTERocmMoeBackwardArgs a{};
    a.struct_size = sizeof(a);
    a.dout = t(dout);
    a.x = t(x);
    a.router_weight = t(router_weight);
    a.aux_loss_scale = t(aux_loss_scale);
    a.fc1_weight = t(fc1_weight);
    a.fc2_weight = t(fc2_weight);
    a.shared_fc1_weight = t(shared_fc1_weight);
    a.shared_fc2_weight = t(shared_fc2_weight);
    a.shared_gate = t(shared_gate);
    a.shared_pre_act = t(shared_pre_act);
    a.dx = t(dx);
    a.dprobs = t(dprobs);
    a.fc1_wgrad = t(fc1_wgrad);
    a.fc2_wgrad = t(fc2_wgrad);
    a.router_wgrad = t(router_wgrad);
    a.router_bias_grad = t(router_bias_grad);
    a.shared_fc1_wgrad = t(shared_fc1_wgrad);
    a.shared_fc2_wgrad = t(shared_fc2_wgrad);
    a.shared_gate_grad = t(shared_gate_grad);
    a.wgrad_accumulate = wgrad_accumulate;
    const auto stream = at::cuda::getCurrentCUDAStream();
    // Without the GIL: the call may wait for an earlier launch (the release bookkeeping, KOSMOS_MOE_REL_LAG
    // launches back); the callbacks that need Python take it back.
    ws_->run([&] {
      py::gil_scoped_release nogil;
      nvte_rocm_moe_backward(live(), &a, stream);
    });
  }

 private:
  NVTERocmMoeLayer live() const {
    if (!layer_) {
      throw std::runtime_error("MoeLayer used after destroy()");
    }
    return layer_;
  }

  py::object ws_obj_;
  MoeWorkspace *ws_ = nullptr;
  NVTERocmMoeLayer layer_ = nullptr;
};

void register_moe_bindings(py::module_ &m) {
  m.def("moe_is_supported", &nvte_rocm_moe_is_supported, "Whether the fused MoE is built and the device supports it.");
  m.attr("MOE_EP_SIZE") = NVTE_ROCM_MOE_EP_SIZE;
  m.def("moe_untracked_bytes", &nvte_rocm_moe_untracked_bytes,
        "Device bytes the fused MoE holds outside PyTorch's allocator (0 for workspaces created from Python).");
  py::class_<MoeWorkspace>(m, "MoeWorkspace")
      .def(py::init<int, int, int, int, int, int, py::function, py::function, py::function, int, int, bool, size_t,
                    int, bool, bool, bool, bool, bool, int, int, int, int, float, float>(),
           py::arg("hidden_size"), py::arg("ffn_hidden_size"), py::arg("num_local_experts"), py::arg("topk"),
           py::arg("num_tokens"), py::arg("rank"), py::arg("allgather"), py::arg("ipc_alloc"), py::arg("ipc_free"),
           py::arg("capacity_rows"),
           py::arg("wgcap") = 256, py::arg("training") = true, py::arg("pool_bytes") = 0, py::arg("activation") = 0,
           py::arg("mxfp8") = false, py::arg("fp8_combine") = false, py::arg("probs_on_input") = false,
           py::arg("shared_expert") = false, py::arg("router") = false, py::arg("router_score") = 0,
           py::arg("router_groups") = 0, py::arg("router_group_topk") = 0, py::arg("router_aux_groups") = 0,
           py::arg("router_aux_coeff") = 0.0f, py::arg("router_scale") = 1.0f)
      .def("destroy", &MoeWorkspace::destroy)
      .def("pool_resize", &MoeWorkspace::pool_resize, py::arg("bytes"))
      .def("pool_info", &MoeWorkspace::pool_info)
      .def("set_capacity", &MoeWorkspace::set_capacity, py::arg("rows"))
      .def("capacity_query", &MoeWorkspace::capacity_query)
      .def("capacity_bytes", &MoeWorkspace::capacity_bytes, py::arg("rows"))
      .def("release_sync", &MoeWorkspace::release_sync);
  py::class_<MoeLayer>(m, "MoeLayer")
      .def(py::init<py::object>(), py::arg("workspace"))
      .def("destroy", &MoeLayer::destroy)
      .def("prepare", &MoeLayer::prepare)
      .def("reset", &MoeLayer::reset)
      .def("status", &MoeLayer::status)
      .def("release", &MoeLayer::release)
      .def("release_state", &MoeLayer::release_state)
      .def("forward", &MoeLayer::forward, py::arg("x"), py::arg("topk_idx"), py::arg("topk_weights"),
           py::arg("fc1_weight"), py::arg("fc2_weight"), py::arg("out"), py::arg("save_for_backward"),
           py::arg("router_weight") = py::none(), py::arg("expert_bias") = py::none(),
           py::arg("router_bias") = py::none(), py::arg("seed") = py::none(), py::arg("aux_loss") = py::none(),
           py::arg("tokens_per_expert") = py::none(), py::arg("shared_fc1_weight") = py::none(),
           py::arg("shared_fc2_weight") = py::none(), py::arg("shared_gate") = py::none(),
           py::arg("shared_pre_act") = py::none())
      .def("backward", &MoeLayer::backward, py::arg("dout"), py::arg("x"), py::arg("fc1_weight"),
           py::arg("fc2_weight"), py::arg("dx"), py::arg("dprobs"), py::arg("fc1_wgrad"), py::arg("fc2_wgrad"),
           py::arg("router_weight") = py::none(), py::arg("aux_loss_scale") = py::none(),
           py::arg("router_wgrad") = py::none(), py::arg("router_bias_grad") = py::none(),
           py::arg("shared_fc1_weight") = py::none(), py::arg("shared_fc2_weight") = py::none(),
           py::arg("shared_gate") = py::none(), py::arg("shared_pre_act") = py::none(),
           py::arg("shared_fc1_wgrad") = py::none(), py::arg("shared_fc2_wgrad") = py::none(),
           py::arg("shared_gate_grad") = py::none(), py::arg("wgrad_accumulate") = 0);
}

}  // namespace transformer_engine::pytorch

// The fused MoE's IPC memory pool (torch.cuda.MemPool over torch.cuda.memory.CUDAPluggableAllocator): plain
// hipMalloc segments, whose IPC handles work whatever the caching allocator's expandable_segments setting.
extern "C" __attribute__((visibility("default"))) void *nvte_rocm_moe_ipc_malloc(ssize_t size, int, cudaStream_t) {
  void *p = nullptr;
  return cudaMalloc(&p, static_cast<size_t>(size)) == cudaSuccess ? p : nullptr;
}

extern "C" __attribute__((visibility("default"))) void nvte_rocm_moe_ipc_free(void *ptr, ssize_t, int, cudaStream_t) {
  (void)cudaFree(ptr);
}
