# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Lazy loading of the KOSMOS binding.

``transformer_engine.transformer_engine_kosmos`` is KOSMOS's own pointer-based binding
(``python/kosmos.cpp`` of the vendored copy), built by TE with ``USE_KOSMOS_MOE`` under that module
name. Without it, the module ``kosmos`` that KOSMOS's own ``make python`` builds is used if it is
importable and has the training-instance API (``MoeInstance``). Nothing is imported until the first
call that needs KOSMOS.
"""

import importlib
from types import ModuleType
from typing import Optional

_MODULE: Optional[ModuleType] = None
_MODULE_NAMES = ("transformer_engine.transformer_engine_kosmos", "kosmos")


def _has_instance_api(mod: ModuleType) -> bool:
    # The caller-owned training API: an instance, its size query and the save tiers.
    return all(
        hasattr(mod, name)
        for name in ("MoeInstance", "instance_bytes", "SAVE_NONE", "SAVE_XP", "SAVE_ALL")
    )


def load() -> ModuleType:
    """The KOSMOS binding module; raises ImportError naming every candidate that failed."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    errors = []
    for name in _MODULE_NAMES:
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            errors.append(f"{name}: {e}")
            continue
        if not _has_instance_api(mod):
            errors.append(
                f"{name}: {getattr(mod, '__file__', '?')} has no KOSMOS MoeInstance training API"
            )
            continue
        _MODULE = mod
        return mod
    raise ImportError(
        "KOSMOS MoE is not available. Build TransformerEngine for gfx950 with"
        " NVTE_CMAKE_EXTRA_ARGS=-DUSE_KOSMOS_MOE=ON, or put KOSMOS's own `make python` module on"
        " sys.path. Tried: "
        + "; ".join(errors)
    )


def module_or_none() -> Optional[ModuleType]:
    """The binding, or None if it cannot be imported."""
    try:
        return load()
    except ImportError:
        return None
