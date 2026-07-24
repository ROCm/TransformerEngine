# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""ROCm SDK initialization for Transformer Engine wheels.

See https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
"""

from __future__ import annotations

import os

# Libraries TE native code may resolve at runtime. Preload before libtransformer_engine.so.
_PRELOAD_LIBS = (
    "amd_comgr",
    "amdhip64",
    "hiprtc",
    "roctx64",
    "hipblaslt",
)


def initialize() -> None:
    """Preload ROCm runtime wheels before TE native libraries are loaded."""
    try:
        import rocm_sdk
        from rocm_sdk._devel import get_devel_root
    except ImportError:
        return

    if not os.getenv("ROCM_PATH"):
        os.environ["ROCM_PATH"] = str(get_devel_root())
    rocm_sdk.initialize_process(preload_shortnames=list(_PRELOAD_LIBS))
