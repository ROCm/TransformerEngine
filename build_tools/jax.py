# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import rocm_build, rocm_path
from .utils import all_files_in_dir, get_cuda_include_dirs, debug_build_enabled
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/JAX extensions."""
    if rocm_build():
        return jax_install_requires(["flax>=0.7.1"])
    else:
        return ["jax", "flax>=0.7.1"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/JAX extensions."""
    return ["numpy"]


def xla_path() -> str:
    """XLA root path lookup.
    Throws FileNotFoundError if XLA source is not found."""

    try:
        import jax
        from packaging import version
        if version.parse(jax.__version__) >= version.parse("0.5.0"):
            from jax import ffi
        else:
            from jax.extend import ffi
    except ImportError:
        if os.getenv("XLA_HOME"):
            xla_home = Path(os.getenv("XLA_HOME"))
        else:
            xla_home = "/opt/xla"
    else:
        xla_home = ffi.include_dir()

    if not os.path.isdir(xla_home):
        raise FileNotFoundError("Could not find xla source.")
    return xla_home


def setup_jax_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup PyBind11 extension for JAX support"""
    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = all_files_in_dir(extensions_dir, name_extension="cpp")

    # Header files
    if rocm_build():
        hip_root, _ = rocm_path()
        include_dirs = [hip_root / "include"]
    else:
        include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
            xla_path(),
        ]
    )

    # If NVTE_RELEASE_BUILD is set, we assume not building but sources packaging
    # and we do not hipify the sources
    if rocm_build() and not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        from .hipify.hipify import hipify_sources as hipify
        base_dir = Path(__file__).parent.parent.resolve()
        sources = hipify(base_dir, csrc_source_files, common_header_files, sources, base_dir)

    # Compile flags
    cxx_flags = ["-O3"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")
    
    if rocm_build():
        cxx_flags.extend(["-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM"])

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    return Pybind11Extension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args=cxx_flags,
    )


def jax_install_requires(reqs: List[str]) -> List[str]:
    """Update requirements with current JAX version to avoid undesired update."""
    try:
        import jax
    except ImportError:
        return []
    return reqs + [f"jax=={jax.__version__}"]
