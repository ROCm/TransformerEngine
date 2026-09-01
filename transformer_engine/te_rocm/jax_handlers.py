# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX handler-dict seam (plugin plan S7.1 / F16).

``handlers()`` returns a dict merged OVER ``transformer_engine_jax.registrations()`` in
``jax/cpp_extensions/base.py`` before ``ffi.register_ffi_target`` runs. That is the whole
seam: no synthesized module, selection binds at trace, and policy freezes per executable (a
jitted function never re-selects). Ships EMPTY - overrides land one op family at a time under
the Stage-6 dispatch contract, each with its paired conformance test.

``register_override`` exists for tests and for the future families; it must be called before
``transformer_engine.jax`` is imported (registration is an import-time event - that is a
property of upstream's loop, not of this seam).
"""
from __future__ import annotations

from typing import Any, Dict

_OVERRIDES: Dict[str, Any] = {}


def register_override(name: str, capsule: Any) -> None:
    """Install one handler override. Import-order sensitive by upstream's design."""
    _OVERRIDES[name] = capsule


def handlers() -> Dict[str, Any]:
    """The dict merged over upstream's registrations(). Empty until a family lands."""
    return dict(_OVERRIDES)
