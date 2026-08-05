/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Thin pybind11/torch binding exposing xAttention's flash-attn-shaped
// interface (xattn::interface) to Python for the TransformerEngine
// "XAttention" attention backend. Built in-tree as a self-contained second
// extension (transformer_engine_xattention) against the 3rdparty/xAttention
// submodule; see build_tools/xattention.py.

#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

#include "interface/attention.h"

namespace xi = xattn::interface;

namespace {

// xAttention launches asynchronously on the stream it is handed and does no
// internal synchronization, so every call must run on torch's current stream to
// stay ordered against the surrounding ops. c10::hip's masquerading-as-CUDA
// stream (what the hipified parts of TE use) wraps this same handle.
hipStream_t current_stream() { return c10::hip::getCurrentHIPStream().stream(); }

// Scratch that xAttention carves its dQ accumulator and amax partials out of.
// Taken from torch's caching allocator, whose stream-ordered reuse is what
// keeps the block alive across the asynchronous launches: it can only be handed
// out again behind work already queued on the allocating stream.
struct Workspace {
  Workspace(const at::Tensor &like, size_t bytes) : size(bytes) {
    if (bytes == 0) return;
    buffer = torch::empty({static_cast<int64_t>(bytes)}, like.options().dtype(torch::kUInt8));
    ptr = buffer.data_ptr();
  }

  at::Tensor buffer;
  void *ptr = nullptr;
  size_t size = 0;
};

// q is (b, s, h, d) when BSHD and (b, h, s, d) otherwise; k supplies the KV
// head count and sequence length.
Workspace fwd_workspace(const at::Tensor &q, bool input_bshd) {
  const auto s = q.sizes();
  return Workspace(
      q, xi::get_mha_fwd_workspace_size(s[0], input_bshd ? s[2] : s[1], input_bshd ? s[1] : s[2]));
}

Workspace bwd_workspace(const at::Tensor &q, const at::Tensor &k, bool input_bshd) {
  const auto s = q.sizes();
  return Workspace(q, xi::get_mha_bwd_workspace_size(
                          s[0], input_bshd ? s[2] : s[1], input_bshd ? k.size(2) : k.size(1),
                          input_bshd ? s[1] : s[2], input_bshd ? k.size(1) : k.size(2), s[3]));
}

// ---- forward (fp16/bf16) --------------------------------------------------
std::vector<at::Tensor> xattn_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                  std::optional<at::Tensor> out, double softmax_scale,
                                  bool is_causal, int64_t window_size_left,
                                  int64_t window_size_right, bool input_bshd, bool output_bshd) {
  // {out, softmax_lse}
  Workspace ws = fwd_workspace(q, input_bshd);
  return xi::mha_fwd(q, k, v, std::move(out), static_cast<float>(softmax_scale), is_causal,
                     /*return_softmax=*/false, input_bshd, output_bshd,
                     static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                     ws.ptr, ws.size, current_stream());
}

// ---- backward (fp16/bf16) -------------------------------------------------
std::vector<at::Tensor> xattn_bwd(at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v,
                                  at::Tensor out, at::Tensor softmax_lse,
                                  std::optional<at::Tensor> dq, std::optional<at::Tensor> dk,
                                  std::optional<at::Tensor> dv,
                                  std::optional<at::Tensor> alibi_slopes, double p_dropout,
                                  double softmax_scale, bool is_causal, int64_t window_size_left,
                                  int64_t window_size_right, double softcap, bool deterministic,
                                  bool input_bshd, bool output_bshd) {
  // {dq, dk, dv, softmax_d}
  Workspace ws = bwd_workspace(q, k, input_bshd);
  return xi::mha_bwd(dout, q, k, v, out, softmax_lse, std::move(dq), std::move(dk), std::move(dv),
                     std::move(alibi_slopes), static_cast<float>(p_dropout),
                     static_cast<float>(softmax_scale), is_causal,
                     static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                     static_cast<float>(softcap), deterministic, input_bshd, output_bshd, ws.ptr,
                     ws.size, current_stream());
}

// ---- forward (per-tensor fp8 quant) ---------------------------------------
std::vector<at::Tensor> xattn_fwd_quant(at::Tensor q, at::Tensor k, at::Tensor v, double descale_q,
                                        double descale_k, double descale_v, double scale_s,
                                        double descale_s, double scale_o,
                                        std::optional<at::Tensor> out, double softmax_scale,
                                        bool is_causal, int64_t window_size_left,
                                        int64_t window_size_right, bool input_bshd,
                                        bool output_bshd) {
  // {out, lse, amax_s, amax_o}
  Workspace ws = fwd_workspace(q, input_bshd);
  return xi::mha_fwd_quant(
      q, k, v, static_cast<float>(descale_q), static_cast<float>(descale_k),
      static_cast<float>(descale_v), static_cast<float>(scale_s), static_cast<float>(descale_s),
      static_cast<float>(scale_o), std::move(out), static_cast<float>(softmax_scale), is_causal,
      /*return_softmax=*/false, input_bshd, output_bshd, static_cast<int>(window_size_left),
      static_cast<int>(window_size_right), ws.ptr, ws.size, current_stream());
}

// ---- forward (MXFP8 block-scaled) -----------------------------------------
std::vector<at::Tensor> xattn_fwd_mx(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_scale,
                                     at::Tensor k_scale, at::Tensor v_scale,
                                     std::optional<at::Tensor> out, double softmax_scale,
                                     bool is_causal, int64_t window_size_left,
                                     int64_t window_size_right, bool input_bshd, bool output_bshd) {
  // {out, lse}
  Workspace ws = fwd_workspace(q, input_bshd);
  return xi::mha_fwd_mx(q, k, v, q_scale, k_scale, v_scale, std::move(out),
                        static_cast<float>(softmax_scale), is_causal, /*return_softmax=*/false,
                        input_bshd, output_bshd, static_cast<int>(window_size_left),
                        static_cast<int>(window_size_right), ws.ptr, ws.size, current_stream());
}

// ---- backward (per-tensor fp8 quant) --------------------------------------
std::vector<at::Tensor> xattn_bwd_quant(
    at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
    at::Tensor softmax_lse, double descale_q, double descale_k, double descale_v, double descale_o,
    double descale_do, double scale_s, double descale_s, double scale_ds, double descale_ds,
    double scale_dq, double scale_dk, double scale_dv, std::optional<at::Tensor> dq,
    std::optional<at::Tensor> dk, std::optional<at::Tensor> dv,
    std::optional<at::Tensor> alibi_slopes, double p_dropout, double softmax_scale, bool is_causal,
    int64_t window_size_left, int64_t window_size_right, double softcap, bool deterministic,
    bool input_bshd, bool output_bshd) {
  // {dq, dk, dv, softmax_d, amax_dq, amax_dk, amax_dv, amax_ds}
  Workspace ws = bwd_workspace(q, k, input_bshd);
  return xi::mha_bwd_quant(
      dout, q, k, v, out, softmax_lse, static_cast<float>(descale_q), static_cast<float>(descale_k),
      static_cast<float>(descale_v), static_cast<float>(descale_o), static_cast<float>(descale_do),
      static_cast<float>(scale_s), static_cast<float>(descale_s), static_cast<float>(scale_ds),
      static_cast<float>(descale_ds), static_cast<float>(scale_dq), static_cast<float>(scale_dk),
      static_cast<float>(scale_dv), std::move(dq), std::move(dk), std::move(dv),
      std::move(alibi_slopes), static_cast<float>(p_dropout), static_cast<float>(softmax_scale),
      is_causal, static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<float>(softcap), deterministic, input_bshd, output_bshd, ws.ptr, ws.size,
      current_stream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "xAttention <-> TransformerEngine PyTorch binding";
  namespace py = pybind11;

  m.def("fwd", &xattn_fwd, "xAttention forward (fp16/bf16)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("out"), py::arg("softmax_scale"), py::arg("is_causal"),
        py::arg("window_size_left"), py::arg("window_size_right"), py::arg("input_bshd"),
        py::arg("output_bshd"));

  m.def("bwd", &xattn_bwd, "xAttention backward (fp16/bf16)", py::arg("dout"), py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("out"), py::arg("softmax_lse"), py::arg("dq"),
        py::arg("dk"), py::arg("dv"), py::arg("alibi_slopes"), py::arg("p_dropout"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("softcap"), py::arg("deterministic"),
        py::arg("input_bshd"), py::arg("output_bshd"));

  m.def("fwd_quant", &xattn_fwd_quant, "xAttention forward (per-tensor fp8)", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("descale_q"), py::arg("descale_k"), py::arg("descale_v"),
        py::arg("scale_s"), py::arg("descale_s"), py::arg("scale_o"), py::arg("out"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("input_bshd"), py::arg("output_bshd"));

  m.def("fwd_mx", &xattn_fwd_mx, "xAttention forward (MXFP8)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"), py::arg("out"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("input_bshd"), py::arg("output_bshd"));

  m.def("bwd_quant", &xattn_bwd_quant, "xAttention backward (per-tensor fp8)", py::arg("dout"),
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("softmax_lse"),
        py::arg("descale_q"), py::arg("descale_k"), py::arg("descale_v"), py::arg("descale_o"),
        py::arg("descale_do"), py::arg("scale_s"), py::arg("descale_s"), py::arg("scale_ds"),
        py::arg("descale_ds"), py::arg("scale_dq"), py::arg("scale_dk"), py::arg("scale_dv"),
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("alibi_slopes"), py::arg("p_dropout"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("softcap"), py::arg("deterministic"),
        py::arg("input_bshd"), py::arg("output_bshd"));
}
