# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""

import os
from pathlib import Path
from importlib import metadata

import setuptools

from .utils import (
    rocm_build,
    rocm_path,
    all_files_in_dir,
    cuda_version,
    get_cuda_include_dirs,
    debug_build_enabled,
    setup_mpi_flags,
)
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    requirements = [
        "torch>=2.1",
        "einops",
        "onnxscript",
        "onnx",
        "packaging",
        "pydantic",
    ]
    if not rocm_build():
        # NVIDIA-only: nvdlfw-inspect is CUDA framework-inspect; nvidia-cudnn-frontend
        # supplies the cuDNN headers for the CUDA build (ROCm uses a stub).
        requirements += [
            "nvdlfw-inspect",
            "nvidia-cudnn-frontend>=1.25.0",
        ]
    return requirements


def test_requirements() -> List[str]:
    """Test dependencies for TE/PyTorch extensions."""
    return [
        "numpy",
        "torchvision",
        "transformers",
        "torchao==0.13",
        "onnxruntime",
        "onnxruntime_extensions",
    ]


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    sources = all_files_in_dir(Path(csrc_source_files), name_extension="cpp")

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
        ]
    )

    # If NVTE_RELEASE_BUILD is set, we assume not building but sources packaging
    # and we do not hipify the sources
    if rocm_build() and not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        from .hipify.hipify import hipify_sources as hipify
        base_dir = Path(__file__).parent.parent.resolve()
        sources = hipify(base_dir, csrc_source_files, common_header_files, sources, base_dir)

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden"]

    # Build-compat tuple (plugin plan S4.6): embed what this extension was built against so the
    # loader can refuse a mismatched runtime at import instead of failing strangely later.
    # Python ABI is already enforced by the cpython so-name tag; arches are runtime-dispatched.
    if rocm_build():
        import torch as _torch
        from .utils import rocm_version as _rocm_version
        _torch_mm = ".".join(_torch.__version__.split("+")[0].split(".")[:2])
        _rocm_mm = ".".join(str(v) for v in _rocm_version()[:2])
        cxx_flags.append(f'-DNVTE_ROCM_BUILD_COMPAT="torch{_torch_mm}-rocm{_rocm_mm}"')
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Version-dependent CUDA options
    if not rocm_build():
        try:
            version = cuda_version()
        except FileNotFoundError:
            print("Could not determine CUDA version")
        else:
            if version < (12, 0):
                raise RuntimeError("Transformer Engine requires CUDA 12.0 or newer")

    setup_mpi_flags(include_dirs, cxx_flags)

    # Mirror the NCCL EP gate from setup.py / common CMake. When disabled, the
    # ep.cpp source no-ops at the #ifdef boundary; without the define it would
    # produce undefined references to nvte_ep_*.
    # Disabled on ROCm
    if not rocm_build() and bool(int(os.getenv("NVTE_WITH_NCCL_EP", "1"))):
        cxx_flags.append("-DNVTE_WITH_NCCL_EP")
        # PyTorch's symm-mem headers gate the NCCL_HAS_SYMMEM_* feature macros on
        # USE_NCCL. The EP extension shares the symm-mem NCCL comm with torch, so
        # it needs those macros visible.
        cxx_flags.append("-DUSE_NCCL")

    library_dirs = []
    libraries = []
    if bool(int(os.getenv("NVTE_ENABLE_NVSHMEM", 0))):
        assert (
            os.getenv("NVSHMEM_HOME") is not None
        ), "NVSHMEM_HOME must be set when compiling with NVTE_ENABLE_NVSHMEM=1"
        nvshmem_home = Path(os.getenv("NVSHMEM_HOME"))
        include_dirs.append(nvshmem_home / "include")
        library_dirs.append(nvshmem_home / "lib")
        libraries.append("nvshmem_host")
        cxx_flags.append("-DNVTE_ENABLE_NVSHMEM")

    if bool(int(os.getenv("NVTE_ENABLE_ROCSHMEM", 0))):
        mpi_home = Path(os.getenv("MPI_HOME", "/usr/lib/x86_64-linux-gnu/openmpi"))
        include_dirs.append(mpi_home / "include")
        library_dirs.append(mpi_home / "lib")
        libraries.append("mpi")
        cxx_flags.extend(["-DNVTE_ENABLE_ROCSHMEM", "-DOMPI_SKIP_MPICXX"])

    if bool(int(os.getenv("NVTE_WITH_CUBLASMP", 0))):
        cxx_flags.append("-DNVTE_WITH_CUBLASMP")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="transformer_engine_rocm_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
