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

// xAttention takes the amax out-params as raw device pointers and validates
// nothing about them, so a host tensor or a non-fp32 one would corrupt memory
// instead of raising. Check here, at the boundary we own.
std::optional<at::Tensor> checked_amax(std::optional<at::Tensor> t, const char *name) {
  if (t.has_value()) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1 &&
                    t->is_contiguous(),
                name, " must be a contiguous 1-element fp32 device tensor");
  }
  return t;
}

// xAttention reads each tensor through the strides implied by its layout, so a
// caller holding sbhd data passes it as-is rather than transposing to bshd.
xattn::MHADataLayout layout_from_name(const std::string &name) {
  if (name == "bshd") return xattn::MHADataLayout::BSHD;
  if (name == "sbhd") return xattn::MHADataLayout::SBHD;
  if (name == "bhsd") return xattn::MHADataLayout::BHSD;
  TORCH_CHECK(false, "unknown attention data layout '", name, "'; expected bshd, sbhd or bhsd");
}

// (batch, seqlen, nhead) of a rank-4 attention tensor under `layout`; head_dim
// is the last dimension in every layout.
struct MHADims {
  int64_t batch, seqlen, nhead;
};

MHADims dims_of(const at::Tensor &t, xattn::MHADataLayout layout) {
  const auto s = t.sizes();
  TORCH_CHECK(s.size() == 4, "attention data tensor must be rank 4, got rank ", s.size());
  switch (layout) {
    case xattn::MHADataLayout::BSHD:
      return {s[0], s[1], s[2]};
    case xattn::MHADataLayout::BHSD:
      return {s[0], s[2], s[1]};
    case xattn::MHADataLayout::SBHD:
      return {s[1], s[0], s[2]};
  }
  TORCH_CHECK(false, "unknown attention data layout");
}

Workspace fwd_workspace(const at::Tensor &q, xattn::MHADataLayout layout) {
  const auto q_dims = dims_of(q, layout);
  return Workspace(
      q, xi::get_mha_fwd_workspace_size(q_dims.batch, q_dims.nhead, q_dims.seqlen));
}

Workspace bwd_workspace(const at::Tensor &q, const at::Tensor &k, xattn::MHADataLayout layout) {
  const auto q_dims = dims_of(q, layout);
  const auto k_dims = dims_of(k, layout);
  return Workspace(q, xi::get_mha_bwd_workspace_size(q_dims.batch, q_dims.nhead, k_dims.nhead,
                                                     q_dims.seqlen, k_dims.seqlen, q.size(3)));
}

// ---- forward (fp16/bf16) --------------------------------------------------
std::vector<at::Tensor> xattn_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                  std::optional<at::Tensor> out, double softmax_scale,
                                  bool is_causal, int64_t window_size_left,
                                  int64_t window_size_right, const std::string &input_layout,
                                  const std::string &output_layout) {
  // {out, softmax_lse}
  const auto in_l = layout_from_name(input_layout);
  const auto out_l = layout_from_name(output_layout);
  Workspace ws = fwd_workspace(q, in_l);
  return xi::mha_fwd(q, k, v, std::move(out), /*softmax_lse_=*/std::nullopt,
                     static_cast<float>(softmax_scale), is_causal,
                     /*return_softmax=*/false, in_l, out_l,
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
                                  const std::string &input_layout,
                                  const std::string &output_layout) {
  // {dq, dk, dv}
  const auto in_l = layout_from_name(input_layout);
  const auto out_l = layout_from_name(output_layout);
  Workspace ws = bwd_workspace(q, k, in_l);
  return xi::mha_bwd(dout, q, k, v, out, softmax_lse, std::move(dq), std::move(dk), std::move(dv),
                     std::move(alibi_slopes), static_cast<float>(p_dropout),
                     static_cast<float>(softmax_scale), is_causal,
                     static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                     static_cast<float>(softcap), deterministic, in_l, out_l, ws.ptr,
                     ws.size, current_stream());
}

// ---- forward (per-tensor fp8 quant) ---------------------------------------
std::vector<at::Tensor> xattn_fwd_quant(at::Tensor q, at::Tensor k, at::Tensor v, double descale_q,
                                        double descale_k, double descale_v, double scale_s,
                                        double descale_s, double scale_o,
                                        std::optional<at::Tensor> out,
                                        std::optional<at::Tensor> amax_s,
                                        std::optional<at::Tensor> amax_o, double softmax_scale,
                                        bool is_causal, int64_t window_size_left,
                                        int64_t window_size_right,
                                        const std::string &input_layout,
                                        const std::string &output_layout) {
  // {out, lse, amax_s, amax_o}. Supplying amax_s/amax_o lets the caller hand in
  // its quantizers' amax slots so the reduction lands there directly.
  const auto in_l = layout_from_name(input_layout);
  const auto out_l = layout_from_name(output_layout);
  Workspace ws = fwd_workspace(q, in_l);
  return xi::mha_fwd_quant(
      q, k, v, static_cast<float>(descale_q), static_cast<float>(descale_k),
      static_cast<float>(descale_v), static_cast<float>(scale_s), static_cast<float>(descale_s),
      static_cast<float>(scale_o), std::move(out), /*softmax_lse_=*/std::nullopt,
      checked_amax(std::move(amax_s), "amax_s"), checked_amax(std::move(amax_o), "amax_o"),
      static_cast<float>(softmax_scale), is_causal,
      /*return_softmax=*/false, in_l, out_l, static_cast<int>(window_size_left),
      static_cast<int>(window_size_right), ws.ptr, ws.size, current_stream());
}

// ---- forward (MXFP8 block-scaled) -----------------------------------------
std::vector<at::Tensor> xattn_fwd_mx(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_scale,
                                     at::Tensor k_scale, at::Tensor v_scale,
                                     std::optional<at::Tensor> out, double softmax_scale,
                                     bool is_causal, int64_t window_size_left,
                                     int64_t window_size_right, const std::string &input_layout,
                                     const std::string &output_layout) {
  // {out, lse}
  const auto in_l = layout_from_name(input_layout);
  const auto out_l = layout_from_name(output_layout);
  Workspace ws = fwd_workspace(q, in_l);
  return xi::mha_fwd_mx(q, k, v, q_scale, k_scale, v_scale, std::move(out),
                        /*softmax_lse_=*/std::nullopt,
                        static_cast<float>(softmax_scale), is_causal, /*return_softmax=*/false,
                        in_l, out_l, static_cast<int>(window_size_left),
                        static_cast<int>(window_size_right), ws.ptr, ws.size, current_stream());
}

// ---- backward (per-tensor fp8 quant) --------------------------------------
std::vector<at::Tensor> xattn_bwd_quant(
    at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
    at::Tensor softmax_lse, double descale_q, double descale_k, double descale_v, double descale_o,
    double descale_do, double scale_s, double descale_s, double scale_ds, double descale_ds,
    double scale_dq, double scale_dk, double scale_dv, std::optional<at::Tensor> dq,
    std::optional<at::Tensor> dk, std::optional<at::Tensor> dv, std::optional<at::Tensor> amax_dq,
    std::optional<at::Tensor> amax_dk, std::optional<at::Tensor> amax_dv,
    std::optional<at::Tensor> amax_ds, std::optional<at::Tensor> alibi_slopes, double p_dropout,
    double softmax_scale, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    double softcap, bool deterministic, const std::string &input_layout,
    const std::string &output_layout) {
  // {dq, dk, dv, amax_dq, amax_dk, amax_dv, amax_ds}. The four amax reductions
  // write to distinct addresses, so a caller folding them together must still
  // pass four separate slots (they are plain stores, not atomic maxima).
  const auto in_l = layout_from_name(input_layout);
  const auto out_l = layout_from_name(output_layout);
  Workspace ws = bwd_workspace(q, k, in_l);
  return xi::mha_bwd_quant(
      dout, q, k, v, out, softmax_lse, static_cast<float>(descale_q), static_cast<float>(descale_k),
      static_cast<float>(descale_v), static_cast<float>(descale_o), static_cast<float>(descale_do),
      static_cast<float>(scale_s), static_cast<float>(descale_s), static_cast<float>(scale_ds),
      static_cast<float>(descale_ds), static_cast<float>(scale_dq), static_cast<float>(scale_dk),
      static_cast<float>(scale_dv), std::move(dq), std::move(dk), std::move(dv),
      checked_amax(std::move(amax_dq), "amax_dq"), checked_amax(std::move(amax_dk), "amax_dk"),
      checked_amax(std::move(amax_dv), "amax_dv"), checked_amax(std::move(amax_ds), "amax_ds"),
      std::move(alibi_slopes), static_cast<float>(p_dropout), static_cast<float>(softmax_scale),
      is_causal, static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<float>(softcap), deterministic, in_l, out_l, ws.ptr, ws.size,
      current_stream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "xAttention <-> TransformerEngine PyTorch binding";
  namespace py = pybind11;

  m.def("fwd", &xattn_fwd, "xAttention forward (fp16/bf16)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("out"), py::arg("softmax_scale"), py::arg("is_causal"),
        py::arg("window_size_left"), py::arg("window_size_right"), py::arg("input_layout"),
        py::arg("output_layout"));

  m.def("bwd", &xattn_bwd, "xAttention backward (fp16/bf16)", py::arg("dout"), py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("out"), py::arg("softmax_lse"), py::arg("dq"),
        py::arg("dk"), py::arg("dv"), py::arg("alibi_slopes"), py::arg("p_dropout"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("softcap"), py::arg("deterministic"),
        py::arg("input_layout"), py::arg("output_layout"));

  m.def("fwd_quant", &xattn_fwd_quant, "xAttention forward (per-tensor fp8)", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("descale_q"), py::arg("descale_k"), py::arg("descale_v"),
        py::arg("scale_s"), py::arg("descale_s"), py::arg("scale_o"), py::arg("out"),
        py::arg("amax_s"), py::arg("amax_o"), py::arg("softmax_scale"), py::arg("is_causal"),
        py::arg("window_size_left"), py::arg("window_size_right"), py::arg("input_layout"),
        py::arg("output_layout"));

  m.def("fwd_mx", &xattn_fwd_mx, "xAttention forward (MXFP8)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"), py::arg("out"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("input_layout"), py::arg("output_layout"));

  m.def("bwd_quant", &xattn_bwd_quant, "xAttention backward (per-tensor fp8)", py::arg("dout"),
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("softmax_lse"),
        py::arg("descale_q"), py::arg("descale_k"), py::arg("descale_v"), py::arg("descale_o"),
        py::arg("descale_do"), py::arg("scale_s"), py::arg("descale_s"), py::arg("scale_ds"),
        py::arg("descale_ds"), py::arg("scale_dq"), py::arg("scale_dk"), py::arg("scale_dv"),
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("amax_dq"), py::arg("amax_dk"),
        py::arg("amax_dv"), py::arg("amax_ds"), py::arg("alibi_slopes"), py::arg("p_dropout"),
        py::arg("softmax_scale"), py::arg("is_causal"), py::arg("window_size_left"),
        py::arg("window_size_right"), py::arg("softcap"), py::arg("deterministic"),
        py::arg("input_layout"), py::arg("output_layout"));
}
