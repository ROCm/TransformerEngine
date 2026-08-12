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
        # Prefer a system ROCm tree when present: FlyDSL's MLIR linker resolution
        # expects that layout to locate ld.lld. Probe the standard system roots in
        # the same order as the rest of TE (see common/__init__.py and
        # build_tools/utils.py::rocm_path), then fall back to the rocm-sdk devel
        # wheel for wheel-only environments with no system tree.
        for _candidate in ("/opt/rocm/core", "/opt/rocm"):
            if os.path.exists(_candidate):
                os.environ["ROCM_PATH"] = _candidate
                break
        else:
            os.environ["ROCM_PATH"] = str(get_devel_root())
    rocm_sdk.initialize_process(preload_shortnames=list(_PRELOAD_LIBS))
