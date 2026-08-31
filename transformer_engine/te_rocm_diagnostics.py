# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""ROCm plugin diagnostics snapshot (plugin plan S3.5). What a bug report attaches.

    python -m transformer_engine.te_rocm_diagnostics
or  from transformer_engine.te_rocm_diagnostics import snapshot; snapshot()
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - diagnostics must never raise
        return f"<error: {type(e).__name__}: {e}>" if default is None else default


def snapshot() -> dict:
    """Collect the state that explains 'what am I actually running?'. Import side effects are
    limited to importing transformer_engine itself (which installs the seam)."""
    import transformer_engine as te  # noqa: PLC0415

    root = Path(te.__file__).resolve().parent.parent
    out = {
        "python": platform.python_version(),
        "host": platform.node(),
        "transformer_engine_file": te.__file__,
        "te_rocm_build": _safe(lambda: te.common.te_rocm_build),
        "is_fp8_fnuz": _safe(lambda: te.common.is_fp8_fnuz()),
        "core_abi_version": _safe(lambda: int(te.common._TE_LIB_CTYPES.nvte_rocm_core_abi_version())),
        "hip": _safe(lambda: subprocess.run(["hipconfig", "--version"], capture_output=True,
                                            text=True, timeout=10).stdout.strip()),
        "gpu_arch": _safe(lambda: "gfx" + subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        ).stdout.split("gfx", 1)[1].split()[0]),
        "torch": _safe(lambda: __import__("torch").__version__),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("NVTE_", "HIP_VISIBLE", "ROCR_VISIBLE"))},
    }
    # seam state
    out["seam"] = {
        "compiled_module": _safe(lambda: sys.modules["transformer_engine_rocm_torch"].__name__),
        "alias_identity": _safe(lambda: sys.modules.get("transformer_engine_torch")
                                is sys.modules.get("transformer_engine_rocm_torch")),
    }
    # overlay bundle: package-dir copy (installed wheel) or overlay root (dev PYTHONPATH tree)
    om = Path(te.__file__).resolve().parent / "_overlay_manifest.json"
    if not om.exists():
        om = root / "overlay-manifest.json"
    if om.exists():
        m = json.loads(om.read_text())
        out["overlay"] = {"bundle_hash": m["bundle_hash"], "upstream_sha": m["upstream_sha"],
                          "patches": len(m["patches"]), "built": m["built"],
                          "patch_ids": sorted(p["id"] for p in m["patches"])}
    else:
        out["overlay"] = None  # fork tree / installed wheel without an overlay manifest
    return out


def main() -> int:
    print(json.dumps(snapshot(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
