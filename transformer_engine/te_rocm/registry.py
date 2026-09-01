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


_NORM_POLICY: Dict[str, Optional[bool]] = {"layernorm": None, "rmsnorm": None}
_NORM_ENV = {"layernorm": "NVTE_USE_LAYERNORM_TRITON", "rmsnorm": "NVTE_USE_RMSNORM_TRITON"}


def select_norm(op: str, forward: bool) -> Callable:
    """Family 2: norms. Select the layernorm/rmsnorm implementation. Pure; call before launch.

    Policy env per op, frozen at first selection for that op (covers capture). The triton
    norms have no shape/dtype eligibility limits beyond running on HIP; a non-HIP build with
    the policy set gets a strict refusal, not a silent fallback.
    """
    import transformer_engine_torch as tex
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    assert op in _NORM_ENV, f"unknown norm op {op!r}"
    if _NORM_POLICY[op] is None:
        _NORM_POLICY[op] = bool(int(os.environ.get(_NORM_ENV[op], "0")))
        _LOG.setdefault("norms", {})[op] = {"policy_triton": _NORM_POLICY[op], "rejections": []}
    compiled = {
        ("layernorm", True): tex.layernorm_fwd, ("layernorm", False): tex.layernorm_bwd,
        ("rmsnorm", True): tex.rmsnorm_fwd, ("rmsnorm", False): tex.rmsnorm_bwd,
    }[(op, forward)]
    if not _NORM_POLICY[op]:
        _LOG["norms"][op]["selected"] = "compiled"
        return compiled
    if not IS_HIP_EXTENSION:
        reason = f"{_NORM_ENV[op]}=1 but this is not a HIP build"
        _LOG["norms"][op]["rejections"].append(reason)
        raise RuntimeError(reason)
    from transformer_engine.te_rocm.triton_kernels.norms_common import (
        te_layernorm_bwd_triton,
        te_layernorm_fwd_triton,
        te_rmsnorm_bwd_triton,
        te_rmsnorm_fwd_triton,
    )

    triton = {
        ("layernorm", True): te_layernorm_fwd_triton, ("layernorm", False): te_layernorm_bwd_triton,
        ("rmsnorm", True): te_rmsnorm_fwd_triton, ("rmsnorm", False): te_rmsnorm_bwd_triton,
    }[(op, forward)]
    _LOG["norms"][op]["selected"] = "triton"
    return triton


_GEMM_POLICY: Dict[str, Optional[bool]] = {"generic": None, "grouped": None, "dequantize": None}


def _policy(kind: str, env: str) -> bool:
    if _GEMM_POLICY[kind] is None:
        _GEMM_POLICY[kind] = bool(int(os.environ.get(env, "0")))
    return _GEMM_POLICY[kind]


def generic_gemm_use_triton(a_row_scaled_nvfp4: bool, b_row_scaled_nvfp4: bool) -> bool:
    """Family 3: policy + predicate for the generic-GEMM triton path (single call site).

    STRICT: row-scaled NVFP4 inputs have no triton kernel; requesting the triton path for
    them raises with the reason rather than silently computing something else.
    """
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    use = _policy("generic", "NVTE_USE_GEMM_TRITON") and IS_HIP_EXTENSION
    log = _LOG.setdefault("generic_gemm", {})
    log["policy_triton"] = _GEMM_POLICY["generic"]
    if use and (a_row_scaled_nvfp4 or b_row_scaled_nvfp4):
        reason = "row-scaled NVFP4 GEMM has no triton kernel"
        log.setdefault("rejections", []).append(reason)
        raise RuntimeError(f"NVTE_USE_GEMM_TRITON=1 but {reason}")
    log["selected"] = "triton" if use else "compiled"
    return use


def grouped_gemm_use_triton(*, fp8: bool, fuse_wgrad_accumulation: bool) -> bool:
    """Family 4: policy + predicate for grouped GEMM (single decision site).

    SOFT fallback by design: the pre-registry env contract allowed one process to mix
    eligible and ineligible calls (fp8 / wgrad-fused fall back to the compiled path
    silently), so a strict refusal here would break existing runs. The rejection reason is
    logged for the diagnostics snapshot instead.
    """
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    use = _policy("grouped", "NVTE_USE_GROUPED_GEMM_TRITON") and IS_HIP_EXTENSION
    log = _LOG.setdefault("grouped_gemm", {})
    log["policy_triton"] = _GEMM_POLICY["grouped"]
    if use and (fp8 or fuse_wgrad_accumulation):
        log.setdefault("rejections", []).append(
            f"soft fallback to compiled: fp8={fp8}, fuse_wgrad_accumulation={fuse_wgrad_accumulation}"
        )
        log["selected"] = "compiled (soft fallback)"
        return False
    log["selected"] = "triton" if use else "compiled"
    return use


def dequantize_use_triton() -> bool:
    """Family 1 (dequantize direction): env policy, frozen at first selection."""
    from torch.utils.cpp_extension import IS_HIP_EXTENSION

    use = _policy("dequantize", "NVTE_USE_DEQUANTIZE_TRITON") and IS_HIP_EXTENSION
    _LOG.setdefault("dequantize", {})["selected"] = "triton" if use else "compiled"
    return use


def registry_state() -> dict:
    """For the diagnostics snapshot: selections and rejection reasons so far."""
    return dict(_LOG)
