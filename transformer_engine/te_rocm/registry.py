# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Op-family implementation registry (plugin plan Stage 6, proposal sec 3.5).

Binding contract, per family:
- ``supports(context)`` is a PURE predicate - no launches, no allocation;
- selection happens strictly before launch;
- strict failure is the default: if policy requests an implementation its predicate rejects,
  selection RAISES with the reason - no silent fallback;
- policy is FROZEN at first selection (which also covers first compile/hipGraph capture):
  the policy env is read once per process, matching how CI toggles these flags (per process,
  never mid-run);
- every selection and rejection is recorded for the diagnostics snapshot.

Family 1: ``quantize`` - compiled ``tex.quantize`` (default) vs ``te_quantize_triton``
(policy: ``NVTE_USE_CAST_TRANSPOSE_TRITON=1``), the eight scattered env-dispatch sites
replaced by ``select_quantize()``.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

__all__ = ["select_quantize", "registry_state"]

_LOG: Dict[str, dict] = {}  # family -> {"selected": name, "policy": ..., "rejections": [...]}


def _quantize_triton_supports(quantizer: Any) -> tuple[bool, str]:
    """Pure predicate: which quantizers the Triton cast path handles."""
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Quantizer,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.te_rocm.tensors.mxfp4_tensor import MXFP4Quantizer

    ok = isinstance(
        quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer, MXFP8Quantizer, MXFP4Quantizer)
    )
    if ok:
        return True, ""
    return False, f"te_quantize_triton has no kernel for {type(quantizer).__name__}"


_POLICY_TRITON: Optional[bool] = None  # frozen at first selection


def select_quantize(quantizer: Any) -> Callable:
    """Select the quantize implementation for this quantizer. Pure; call before launch."""
    global _POLICY_TRITON
    import transformer_engine_torch as tex

    if _POLICY_TRITON is None:
        _POLICY_TRITON = bool(int(os.environ.get("NVTE_USE_CAST_TRANSPOSE_TRITON", "0")))
        _LOG["quantize"] = {"policy_triton": _POLICY_TRITON, "rejections": []}
    if not _POLICY_TRITON:
        _LOG["quantize"]["selected"] = "compiled(tex.quantize)"
        return tex.quantize
    ok, reason = _quantize_triton_supports(quantizer)
    if not ok:
        # strict: the user asked for the triton path; a silent fallback would hide a
        # performance or numerics surprise. Refuse with the reason instead.
        _LOG["quantize"]["rejections"].append(reason)
        raise RuntimeError(
            f"NVTE_USE_CAST_TRANSPOSE_TRITON=1 but the triton quantize path rejects this"
            f" call: {reason}. Unset the flag or use a supported quantizer type."
        )
    from transformer_engine.te_rocm.triton_kernels.cast import te_quantize_triton

    _LOG["quantize"]["selected"] = "triton(te_quantize_triton)"
    return te_quantize_triton


def registry_state() -> dict:
    """For the diagnostics snapshot: selections and rejection reasons so far."""
    return dict(_LOG)
