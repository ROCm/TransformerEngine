# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""xAttention fp8 attention backend for TransformerEngine (ROCm / gfx950).

xAttention is an fp8-only flash-attention library (MI350/MI450). This module
adapts TE's fp8 DPA plumbing (Float8Tensors + quantizers) to xAttention's
per-tensor quant kernels (mha_fwd_quant / mha_bwd_quant), so that on ROCm it can
serve as the fp8 kernel underneath the FusedAttention backend.

All xAttention-specific logic lives here; FusedAttnFunc only calls
``fp8_forward`` / ``fp8_backward`` behind ``is_available()``. The binding module
(``transformer_engine_xattention``) is built out-of-tree — see
``transformer_engine/pytorch/csrc/xattention/``.

Scope (fp8 DelayedScaling): bshd/sbhd, non-padding, non-CP, head_dim 64/128,
causal / no-mask / sliding-window. Anything else must be filtered out by
``get_attention_backend`` before we get here.
"""

import os
from typing import List, Optional, Tuple

import torch

try:
    import transformer_engine_xattention as _xattn  # noqa: F401

    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on out-of-tree build
    _xattn = None
    _IMPORT_ERROR = e


def is_installed() -> bool:
    """Whether the xAttention binding module is importable (ignores the env gate)."""
    return _xattn is not None


def is_available() -> bool:
    """Whether the xAttention binding is importable and enabled (NVTE_XATTENTION=1)."""
    if _xattn is None:
        return False
    return os.getenv("NVTE_XATTENTION", "0") == "1"


def import_error() -> Optional[BaseException]:
    """The import error, if the binding failed to load (for diagnostics)."""
    return _IMPORT_ERROR


def _qkv_format(qkv_layout: str) -> str:
    """'sbhd' or 'bshd' from a qkv_layout string (first group's alpha chars)."""
    return "".join(c for c in qkv_layout.split("_")[0] if c.isalpha())


def _fp8_data(t) -> torch.Tensor:
    """Reinterpret a Float8Tensor's uint8 ``_data`` as its fp8 torch dtype.

    TE stores fp8 bytes as uint8; xAttention requires a float8_e4m3fn/e5m2 tensor.
    """
    import transformer_engine_torch as tex  # pylint: disable=import-outside-toplevel

    torch_dtype = (
        torch.float8_e5m2 if t._fp8_dtype == tex.DType.kFloat8E5M2 else torch.float8_e4m3fn
    )
    return t._data.view(torch_dtype)


def _to_bshd(x: torch.Tensor, fmt: str) -> torch.Tensor:
    # x is per-tensor [*, h, d] laid out as fmt (sbhd or bshd); return bshd contig.
    if fmt == "sbhd":
        x = x.transpose(0, 1)
    return x.contiguous()


def _from_bshd(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "sbhd":
        return x.transpose(0, 1).contiguous()
    return x


def _window(window_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if window_size is None:
        return -1, -1
    return int(window_size[0]), int(window_size[1])


def fp8_forward(
    q_fp8,
    k_fp8,
    v_fp8,
    qkv_layout: str,
    s_quantizer,
    o_quantizer,
    softmax_scale: float,
    attn_mask_type: str,
    window_size: Optional[Tuple[int, int]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run the xAttention per-tensor fp8 forward.

    Inputs q/k/v are Float8Tensors (already quantized by the caller). Returns a
    bf16 output plus aux_ctx_tensors ``[softmax_lse]`` for the backward. Writes
    amax_s/amax_o back into the S/O quantizers for delayed-scaling history.
    """
    assert _xattn is not None, f"xAttention binding not available: {_IMPORT_ERROR}"
    causal = "causal" in attn_mask_type
    wl, wr = _window(window_size)
    fmt = _qkv_format(qkv_layout)

    qd = _to_bshd(_fp8_data(q_fp8), fmt)
    kd = _to_bshd(_fp8_data(k_fp8), fmt)
    vd = _to_bshd(_fp8_data(v_fp8), fmt)

    descale_q = q_fp8._scale_inv.item()
    descale_k = k_fp8._scale_inv.item()
    descale_v = v_fp8._scale_inv.item()
    scale_s = s_quantizer.scale.item()
    descale_s = 1.0 / scale_s
    scale_o = o_quantizer.scale.item()

    out = torch.empty_like(qd, dtype=torch.bfloat16)
    res = _xattn.fwd_quant(
        qd, kd, vd, descale_q, descale_k, descale_v, scale_s, descale_s, scale_o,
        out, float(softmax_scale), causal, wl, wr, True, True,
    )
    out_bshd, softmax_lse, amax_s, amax_o = res[0], res[1], res[2], res[3]

    s_quantizer.amax.copy_(amax_s.reshape(s_quantizer.amax.shape).to(s_quantizer.amax.dtype))
    o_quantizer.amax.copy_(amax_o.reshape(o_quantizer.amax.shape).to(o_quantizer.amax.dtype))

    out_hp = _from_bshd(out_bshd, fmt)
    return out_hp, [softmax_lse.contiguous()]


def fp8_backward(
    d_out_fp8,
    q_fp8,
    k_fp8,
    v_fp8,
    out_fp8,
    softmax_lse: torch.Tensor,
    qkv_layout: str,
    s_quantizer,
    dp_quantizer,
    dqkv_quantizer,
    o_quantizer,
    do_quantizer,
    softmax_scale: float,
    attn_mask_type: str,
    window_size: Optional[Tuple[int, int]],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the xAttention per-tensor fp8 backward, returning bf16 dq/dk/dv.

    Writes amax_dq/dk/dv into dqkv_quantizer and amax_ds into dp_quantizer.
    """
    assert _xattn is not None, f"xAttention binding not available: {_IMPORT_ERROR}"
    causal = "causal" in attn_mask_type
    wl, wr = _window(window_size)
    fmt = _qkv_format(qkv_layout)

    qd = _to_bshd(_fp8_data(q_fp8), fmt)
    kd = _to_bshd(_fp8_data(k_fp8), fmt)
    vd = _to_bshd(_fp8_data(v_fp8), fmt)
    od = _to_bshd(_fp8_data(out_fp8), fmt)
    dod = _to_bshd(_fp8_data(d_out_fp8), fmt)

    descale_q = q_fp8._scale_inv.item()
    descale_k = k_fp8._scale_inv.item()
    descale_v = v_fp8._scale_inv.item()
    descale_o = out_fp8._scale_inv.item()
    descale_do = d_out_fp8._scale_inv.item()
    scale_s = s_quantizer.scale.item()
    descale_s = 1.0 / scale_s
    scale_ds = dp_quantizer.scale.item()
    descale_ds = 1.0 / scale_ds

    b, s, h, d = qd.shape
    dq = torch.empty(b, s, h, d, device=qd.device, dtype=torch.bfloat16)
    dk = torch.empty_like(kd, dtype=torch.bfloat16)
    dv = torch.empty_like(vd, dtype=torch.bfloat16)

    res = _xattn.bwd_quant(
        dod, qd, kd, vd, od, softmax_lse,
        descale_q, descale_k, descale_v, descale_o, descale_do,
        scale_s, descale_s, scale_ds, descale_ds,
        1.0, 1.0, 1.0,  # dq/dk/dv are bf16 -> output scales unused
        dq, dk, dv, None,
        0.0, float(softmax_scale), causal, wl, wr, 0.0, bool(deterministic), True, True,
    )
    dq_o, dk_o, dv_o = res[0], res[1], res[2]
    amax_dq, amax_dk, amax_dv, amax_ds = res[4], res[5], res[6], res[7]

    # Delayed-scaling history: dQKV quantizer tracks a single amax across dq/dk/dv.
    amax_dqkv = torch.max(torch.stack([amax_dq.reshape(()), amax_dk.reshape(()), amax_dv.reshape(())]))
    dqkv_quantizer.amax.copy_(amax_dqkv.reshape(dqkv_quantizer.amax.shape).to(dqkv_quantizer.amax.dtype))
    dp_quantizer.amax.copy_(amax_ds.reshape(dp_quantizer.amax.shape).to(dp_quantizer.amax.dtype))

    return _from_bshd(dq_o, fmt), _from_bshd(dk_o, fmt), _from_bshd(dv_o, fmt)
