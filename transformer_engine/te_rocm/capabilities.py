# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Capability provider (plugin plan S5.3 / proposal sec 3.6).

``supports(op, context) -> Decision`` with MANDATORY rejection reasons. The provider
distinguishes static hardware capability, version-dependent capability, and dynamic operation
eligibility; policy stays with the caller. First registered capability: ``fp8.fnuz`` - the
smallest end-to-end proof, replacing three parallel implementations (common ctypes /
pytorch device-arch check / jax SUBPROCESS shell-out) with one library-backed answer.

Framework-neutral: imports only transformer_engine.common, and lazily.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

__all__ = ["Decision", "supports", "fnuz", "register"]


@dataclasses.dataclass(frozen=True)
class Decision:
    """Outcome of a capability query. ``reason`` is mandatory when unsupported."""

    supported: bool
    reason: str  # empty only when supported
    constraints: Optional[dict] = None
    implementation_version: str = "te_rocm.capabilities/1"

    def __bool__(self) -> bool:
        return self.supported

    def __post_init__(self) -> None:
        assert self.supported or self.reason, "rejections must carry a reason"


def _fp8_fnuz(_context: Optional[dict]) -> Decision:
    # Single source of truth: the BUILT LIBRARY (S3.3 introspection symbol
    # nvte_uses_fp8_fnuz via common's ctypes handle) - not a device-arch table. The retired
    # pytorch implementation answered `arch == gfx942`; the library answer agrees on every
    # supported arch and cannot drift from what the kernels were compiled for.
    import transformer_engine.common as te_common  # lazy; loads the core library once

    if te_common.is_fp8_fnuz():
        return Decision(True, "")
    return Decision(False, "core library built for OCP FP8 formats (e4m3fn/e5m2), not FNUZ")


_REGISTRY: Dict[str, Callable[[Optional[dict]], Decision]] = {
    "fp8.fnuz": _fp8_fnuz,
}


def register(op: str, fn: Callable[[Optional[dict]], Decision]) -> None:
    """Register a capability resolver. Later Stage-5/6 op families land here."""
    assert op not in _REGISTRY, f"capability {op!r} already registered"
    _REGISTRY[op] = fn


def supports(op: str, context: Optional[dict] = None) -> Decision:
    """Query one capability. Unknown ops get a reasoned rejection, never a KeyError."""
    fn = _REGISTRY.get(op)
    if fn is None:
        return Decision(False, f"unknown capability {op!r}; registered: {sorted(_REGISTRY)}")
    return fn(context)


def fnuz() -> bool:
    """Convenience: does this build use FNUZ FP8 formats?"""
    return supports("fp8.fnuz").supported
