# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""The binding of the fused MoE: ``transformer_engine_torch.rocm_moe``.

KOSMOS is linked into libtransformer_engine behind the C API ``nvte_rocm_moe_*``
(``transformer_engine/rocm_moe.h``) when Transformer Engine is built for gfx950 with
``USE_KOSMOS_MOE`` (the default); the submodule exists in every build and reports whether it is.
"""

from types import ModuleType
from typing import Optional

import transformer_engine_torch as tex

_MODULE: Optional[ModuleType] = None


def load() -> ModuleType:
    """The binding module; raises ImportError if Transformer Engine was built without KOSMOS."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    mod = getattr(tex, "rocm_moe", None)
    if mod is None or not mod.is_built():
        raise ImportError(
            "KOSMOS MoE is not available: this Transformer Engine was built without it. It needs"
            " ROCm and gfx950 in NVTE_ROCM_ARCH; NVTE_CMAKE_EXTRA_ARGS=-DUSE_KOSMOS_MOE=OFF leaves"
            " it out."
        )
    _MODULE = mod
    return mod


def module_or_none() -> Optional[ModuleType]:
    """The binding, or None if Transformer Engine was built without KOSMOS."""
    try:
        return load()
    except ImportError:
        return None
