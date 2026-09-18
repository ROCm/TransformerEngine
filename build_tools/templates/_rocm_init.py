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


def _profiler_owns_rocm_mappings() -> bool:
    """True when rocprofv3 already mapped the devel-tree ROCm/LLVM libraries.

    The TheRock split puts identical SONAMEs in both ``_rocm_sdk_core`` and
    ``_rocm_sdk_devel`` as *different inodes*. ``rocm_sdk.preload_libraries``
    dlopens the core copies by absolute path. ``rocprofv3`` LD_PRELOADs
    ``librocprofiler-sdk.so`` from the devel tree, which already pulled in
    devel ``libamd_comgr`` / ``libLLVM``. A second core mapping of LLVM then
    aborts with ``CommandLine option registered more than once``, and
    rocprofv3's SIGABRT handler appears to hang.

    Skip TE's core-wheel preload in that process so the linker keeps the
    single devel mapping rocprofv3 already established.
    """
    if os.getenv("NVTE_ROCM_SKIP_SDK_PRELOAD", "0") == "1":
        return True
    ld_preload = os.getenv("LD_PRELOAD", "")
    if "rocprofiler" in ld_preload:
        return True
    # rocprofv3 always writes this into the child environment.
    if os.getenv("ROCPROFILER_LIBRARY_CTOR") is not None:
        return True
    return False


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
    if _profiler_owns_rocm_mappings():
        return
    rocm_sdk.initialize_process(preload_shortnames=list(_PRELOAD_LIBS))
