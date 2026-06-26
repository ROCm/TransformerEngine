# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX related extensions."""

import os
from pathlib import Path

import setuptools

from .utils import rocm_build, rocm_path
from .utils import get_cuda_include_dirs, all_files_in_dir, debug_build_enabled, setup_mpi_flags
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/JAX extensions."""
    # If NVTE_RELEASE_BUILD is set, we assume not building but sources packaging
    if rocm_build() and not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        """Update requirements with current JAX version to avoid undesired update."""
        try:
            import jax
            return [f"jax=={jax.__version__}", "flax>=0.7.1"]
        except ImportError:
            pass
    return ["jax", "flax>=0.7.1"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/JAX extensions.

    Triton Package Selection:
        The triton package is selected based on NVTE_USE_PYTORCH_TRITON environment variable:

        Default (NVTE_USE_PYTORCH_TRITON unset or "0"):
            Returns 'triton' - OpenAI's standard package from PyPI.
            Install with: pip install triton

        NVTE_USE_PYTORCH_TRITON=1:
            Returns 'pytorch-triton' - for mixed JAX+PyTorch environments.
            Install with: pip install pytorch-triton --index-url https://download.pytorch.org/whl/cu121

            Note: Do NOT install pytorch-triton from PyPI directly - that's a placeholder.
    """
    use_pytorch_triton = bool(int(os.environ.get("NVTE_USE_PYTORCH_TRITON", "0")))

    triton_package = "pytorch-triton" if use_pytorch_triton else "triton"

    return [
        "numpy",
        triton_package,
    ]


def xla_path() -> str:
    """XLA root path lookup.
    Throws FileNotFoundError if XLA source is not found."""

    try:
        from jax import ffi
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

    setup_mpi_flags(include_dirs, cxx_flags)

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    return Pybind11Extension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args=cxx_flags,
        libraries=["nccl"] if not rocm_build() else [],
    )
