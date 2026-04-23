# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""JAX-side helpers shared across TE JAX code (ROCm detection, FP8 dtypes)."""

import importlib.metadata
import re
import subprocess
import sys
from functools import cache

import jax.numpy as jnp


@cache
def is_hip_extension() -> bool:
    """Return True when the JAX ROCm plugin distribution is installed."""
    return any(
        re.match(r"jax-rocm\d+-plugin", d.metadata["Name"])
        for d in importlib.metadata.distributions()
    )


if is_hip_extension():

    @cache
    def is_mi200():
        """Return True when running on AMD Instinct MI200-class hardware."""
        import jax

        return re.search(r"AMD Instinct MI2\.0", jax.devices()[0].device_kind) is not None


@cache
def is_fp8_fnuz():
    """Return True when TE core reports FP8 FNUZ usage."""
    if not is_hip_extension():
        return False
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; "
            "import os; os.environ['NVTE_FRAMEWORK']='none'; "
            "import transformer_engine as te; exit(not te.common.is_fp8_fnuz())",
        ],
        check=False,
    )
    return proc.returncode == 0

get_jnp_float8_e4m3_type = lambda: jnp.float8_e4m3fnuz if is_fp8_fnuz() else jnp.float8_e4m3fn
get_jnp_float8_e5m2_type = lambda: jnp.float8_e5m2fnuz if is_fp8_fnuz() else jnp.float8_e5m2
