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
from .utils import all_files_in_dir, get_cuda_include_dirs, debug_build_enabled
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

    # Link against the TE core library so the jax extension resolves core
    # symbols via the ELF NEEDED graph rather than via RTLD_GLOBAL. This
    # avoids transitively exposing librocroller.so symbols, which interpose
    # with HIP's internal helpers and cause hipModuleLoad to abort with
    # `free(): invalid size`.
    #
    # The CMake build of the core library runs before this extension is
    # linked (see `_CMakeBuildExtension.run` in build_ext.py), and the
    # CMake install directory is injected into `library_dirs` there so the
    # linker can find `libtransformer_engine.so` even on a clean build. We
    # additionally try to resolve a previously-built copy here so that
    # incremental builds and tooling that links this extension in isolation
    # still work.
    libraries = ["nccl"] if not rocm_build() else []
    libraries.append("transformer_engine")
    library_dirs: List[str] = []
    try:
        from transformer_engine.common import _get_shared_object_file
        core_lib_path = Path(_get_shared_object_file("core"))
        library_dirs.append(str(core_lib_path.parent))
    except (ImportError, FileNotFoundError):
        pass

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    return Pybind11Extension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args=cxx_flags,
        libraries=libraries,
        library_dirs=library_dirs,
    )
