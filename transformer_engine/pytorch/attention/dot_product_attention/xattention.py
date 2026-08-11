# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""xAttention fp8 attention backend for TransformerEngine (ROCm / gfx950).

xAttention is an fp8-only flash-attention library (MI350/MI450). This module
adapts TE's fp8 DPA plumbing (Float8Tensors + quantizers) to xAttention's
per-tensor quant kernels (mha_fwd_quant / mha_bwd_quant), so that on ROCm it can
serve as the fp8 kernel underneath the FusedAttention backend.

All xAttention-specific logic lives here; FusedAttnFunc only calls
``fp8_forward`` / ``fp8_backward`` behind the ``NVTE_XAttn`` backend selection.
The binding module (``transformer_engine_xattention``) is built in-tree as a
self-contained second extension against the ``3rdparty/xAttention`` submodule —
see ``build_tools/xattention.py``.

Scope (fp8 DelayedScaling): bshd/sbhd, non-padding, non-CP, head_dim 64/128,
causal / no-mask / sliding-window. Anything else must be filtered out by
``get_attention_backend`` before we get here.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch


def _xattention_root_and_arch() -> Tuple[Optional[str], Optional[str]]:
    """Locate the installed xAttention data root and packaged arch.

    Order: explicit env override, then the build-time generated paths module,
    then the in-tree submodule relative to this package (editable install).
    """
    root = os.getenv("NVTE_XATTENTION_SOURCE_DIR")
    arch = os.getenv("NVTE_XATTENTION_ARCH")
    if root is None:
        try:
            from ._xattention_paths import (  # pylint: disable=import-outside-toplevel
                XATTENTION_ROOT,
                XATTENTION_ARCH,
            )

            root = XATTENTION_ROOT
            arch = arch or XATTENTION_ARCH
        except ImportError:  # pragma: no cover - generated only by a real build
            pass
    if root is None:
        # dot_product_attention -> attention -> pytorch -> transformer_engine -> repo root
        cand = Path(__file__).resolve().parents[4] / "3rdparty" / "xAttention"
        if cand.is_dir():
            root = str(cand)
    return root, arch


def _configure_runtime_env() -> None:
    """Point the prebuilt closed core at its SP3 toolchain and writable scratch.

    The core bakes in absolute paths from the packaging machine, so the SP3
    toolchain dir and (for JIT) writable scratch/kernel-cache dirs must be set
    via ``XATT_*`` env vars before it runs. User-set values are always honored.
    Runs at import (before the binding is loaded); failures are non-fatal.
    """
    try:
        root, arch = _xattention_root_and_arch()
        if root is None:
            return
        root = Path(root)
        if arch is None:
            sp3_base = root / "sp3"
            sp3_dirs = [p.name for p in sp3_base.glob("*") if p.is_dir()] if sp3_base.is_dir() else []
            if len(sp3_dirs) == 1:
                arch = sp3_dirs[0]

        # SP3 toolchain (required for JIT codegen; harmless to set for AOT).
        if arch and "XATT_SP3_DIR" not in os.environ:
            sp3 = root / "sp3" / arch
            if sp3.is_dir():
                os.environ["XATT_SP3_DIR"] = str(sp3)

        # Kernel build mode + prebuilt kernel dir (AOT) recorded at build time.
        kernel_mode, kernel_dir = "AOT", None
        try:
            from ._xattention_paths import (  # pylint: disable=import-outside-toplevel
                XATTENTION_KERNEL_MODE,
                XATTENTION_KERNEL_DIR,
            )

            kernel_mode = (XATTENTION_KERNEL_MODE or "AOT").upper()
            kernel_dir = XATTENTION_KERNEL_DIR
        except ImportError:  # pragma: no cover - generated only by a real build
            pass

        # Writable scratch must not live in the (possibly read-only) install tree.
        cache = (
            Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "transformer_engine"
            / "xattention"
        )
        if "XATT_TMP_DIR" not in os.environ:
            tmp = cache / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            os.environ["XATT_TMP_DIR"] = str(tmp)

        # XATT_KERNEL_DIR is the parent of the "<arch>" dir (the runtime appends
        # "/<arch>" itself). AOT: point at the prebuilt kernels; JIT: a writable
        # cache the runtime codegen can populate.
        if "XATT_KERNEL_DIR" not in os.environ:
            if kernel_mode == "AOT" and kernel_dir and Path(kernel_dir).is_dir():
                os.environ["XATT_KERNEL_DIR"] = str(kernel_dir)
            else:
                kern = cache / "kernels"
                kern.mkdir(parents=True, exist_ok=True)
                os.environ["XATT_KERNEL_DIR"] = str(kern)
    except Exception:  # pragma: no cover - defensive; never break import
        pass


_configure_runtime_env()

try:
    import transformer_engine_xattention as _xattn  # noqa: F401

    _IMPORT_ERROR = None
except (ImportError, OSError) as e:  # pragma: no cover - depends on the optional build
    _xattn = None
    _IMPORT_ERROR = e


def is_installed() -> bool:
    """Whether the xAttention binding module is importable (ignores the env gate)."""
    return _xattn is not None


def import_error() -> Optional[BaseException]:
    """The import error, if the binding failed to load (for diagnostics)."""
    return _IMPORT_ERROR


def _qkv_format(qkv_layout: str) -> str:
    """'sbhd' or 'bshd' from a qkv_layout string (first group's alpha chars)."""
    return "".join(c for c in qkv_layout.split("_")[0] if c.isalpha())


def _torch_fp8_dtype(fp8_dtype) -> torch.dtype:
    """The torch fp8 dtype matching a TE ``tex.DType``."""
    import transformer_engine_torch as tex  # pylint: disable=import-outside-toplevel

    return torch.float8_e5m2 if fp8_dtype == tex.DType.kFloat8E5M2 else torch.float8_e4m3fn


def _fp8_data(t) -> torch.Tensor:
    """Reinterpret a Float8Tensor's uint8 ``_data`` as its fp8 torch dtype.

    TE stores fp8 bytes as uint8; xAttention requires a float8_e4m3fn/e5m2 tensor.
    """
    return t._data.view(_torch_fp8_dtype(t._fp8_dtype))


def _host_scalars(tensors: List[torch.Tensor]) -> List[float]:
    """Read several device scalars to the host in a single synchronization.

    xAttention takes its scales as host floats, and the per-tensor fp8 path needs
    five to seven of them per call. Read one at a time, each is its own
    device-to-host copy that drains the stream; packing them first costs one small
    kernel and leaves a single sync.
    """
    return torch.cat([t.detach().reshape(1) for t in tensors]).tolist()


def _has_static_scale(quantizer) -> bool:
    """Whether ``quantizer``'s scale is known before the kernel runs.

    DelayedScaling carries a scale forward from the previous iteration's amax
    history, so xAttention can quantize as it writes. CurrentScaling derives the
    scale from the tensor's own amax, which does not exist until afterwards.
    """
    from ...tensor.float8_tensor import (  # pylint: disable=import-outside-toplevel
        Float8Quantizer,
    )

    return isinstance(quantizer, Float8Quantizer)


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
    fp8_output: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run the xAttention per-tensor fp8 forward.

    Inputs q/k/v are Float8Tensors (already quantized by the caller). Returns the
    output plus aux_ctx_tensors ``[softmax_lse]`` for the backward, and writes
    amax_s/amax_o back into the S/O quantizers for delayed-scaling history.

    The output is a Float8Tensor when ``fp8_output`` is set and the O quantizer's
    scale is known ahead of the kernel, sparing the caller a requantization pass;
    otherwise it is bf16.
    """
    assert _xattn is not None, f"xAttention binding not available: {_IMPORT_ERROR}"
    causal = "causal" in attn_mask_type
    wl, wr = _window(window_size)
    fmt = _qkv_format(qkv_layout)

    qd = _to_bshd(_fp8_data(q_fp8), fmt)
    kd = _to_bshd(_fp8_data(k_fp8), fmt)
    vd = _to_bshd(_fp8_data(v_fp8), fmt)

    # The kernel folds scale_o into the output store whenever per-tensor quant is
    # on, so it may only be non-unit when we are actually asking for fp8 out.
    quantize_out = fp8_output and _has_static_scale(o_quantizer)

    scales = [q_fp8._scale_inv, k_fp8._scale_inv, v_fp8._scale_inv, s_quantizer.scale]
    if quantize_out:
        scales.append(o_quantizer.scale)
    descale_q, descale_k, descale_v, scale_s, *rest = _host_scalars(scales)
    descale_s = 1.0 / scale_s
    scale_o = rest[0] if quantize_out else 1.0

    if quantize_out:
        # TE keeps fp8 payloads as uint8; the kernel wants the fp8 view of them.
        out_data = torch.empty_like(qd, dtype=torch.uint8)
        out = out_data.view(_torch_fp8_dtype(o_quantizer.dtype))
    else:
        out = torch.empty_like(qd, dtype=torch.bfloat16)

    # The kernel reduces amax straight into the quantizers' slots, so there is no
    # allocation to make here and nothing to copy back afterwards. amax_o is
    # captured before scale_o is applied, so the history stays in the output's
    # true units whichever output dtype we asked for.
    res = _xattn.fwd_quant(
        qd, kd, vd, descale_q, descale_k, descale_v, scale_s, descale_s, scale_o,
        out, s_quantizer.amax, o_quantizer.amax,
        float(softmax_scale), causal, wl, wr, True, True,
    )
    out_bshd, softmax_lse = res[0], res[1]

    if quantize_out:
        # scale_inv is derived from the quantizer on device; no host round trip.
        return (
            o_quantizer.create_tensor_from_data(
                _from_bshd(out_data, fmt), fake_dtype=q_fp8.dtype
            ),
            [softmax_lse.contiguous()],
        )
    return _from_bshd(out_bshd, fmt), [softmax_lse.contiguous()]


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

    descale_q, descale_k, descale_v, descale_o, descale_do, scale_s, scale_ds = _host_scalars(
        [
            q_fp8._scale_inv,
            k_fp8._scale_inv,
            v_fp8._scale_inv,
            out_fp8._scale_inv,
            d_out_fp8._scale_inv,
            s_quantizer.scale,
            dp_quantizer.scale,
        ]
    )
    descale_s = 1.0 / scale_s
    descale_ds = 1.0 / scale_ds

    b, s, h, d = qd.shape
    dq = torch.empty(b, s, h, d, device=qd.device, dtype=torch.bfloat16)
    dk = torch.empty_like(kd, dtype=torch.bfloat16)
    dv = torch.empty_like(vd, dtype=torch.bfloat16)

    # Delayed-scaling history: the dQKV quantizer tracks a single amax across
    # dq/dk/dv, but the kernel's three reductions are plain stores to distinct
    # addresses, so aliasing them onto one slot would race. Give them adjacent
    # slots in one buffer and fold it afterwards. dS maps 1:1 and goes direct.
    amax_dqkv = torch.empty(3, device=qd.device, dtype=torch.float32)

    res = _xattn.bwd_quant(
        dod, qd, kd, vd, od, softmax_lse,
        descale_q, descale_k, descale_v, descale_o, descale_do,
        scale_s, descale_s, scale_ds, descale_ds,
        1.0, 1.0, 1.0,  # dq/dk/dv are bf16 -> output scales unused
        dq, dk, dv,
        amax_dqkv[0:1], amax_dqkv[1:2], amax_dqkv[2:3], dp_quantizer.amax,
        None,
        0.0, float(softmax_scale), causal, wl, wr, 0.0, bool(deterministic), True, True,
    )
    dq_o, dk_o, dv_o = res[0], res[1], res[2]

    torch.amax(amax_dqkv, dim=0, keepdim=True, out=dqkv_quantizer.amax)

    return _from_bshd(dq_o, fmt), _from_bshd(dk_o, fmt), _from_bshd(dv_o, fmt)
