# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from pathlib import Path
import subprocess


def is_local_version_used() -> bool:
    return not bool(int(os.getenv("NVTE_NO_LOCAL_VERSION", "0"))) and (
        not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0")))
        or bool(int(os.getenv("NVTE_USE_LOCAL_VERSION", "0"))))


def version_file(base: str | Path) -> Path:
    return Path(base).resolve() / "build_tools" / "VERSION.txt"


def te_version() -> str:
    """Transformer Engine version string

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    root_path = Path(__file__).resolve().parent
    with open(root_path / "VERSION.txt", "r") as f:
        version = f.readline().strip()
    if is_local_version_used():
        try:
            output = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                cwd=root_path,
                check=True,
                universal_newlines=True,
            )
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            commit = output.stdout.strip()
            version += f"+{commit}"
    return version
