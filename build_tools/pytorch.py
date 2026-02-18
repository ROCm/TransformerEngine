# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import (
    rocm_build,
    rocm_path,
    hipify,
)
from .utils import (
    all_files_in_dir,
    cuda_archs,
    cuda_version,
    get_cuda_include_dirs,
    debug_build_enabled,
)
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    return ["torch>=2.1", "einops", "onnxscript==0.3.1", "onnx"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/JAX extensions."""
    return ["numpy", "torchvision", "transformers"]


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files - include both .cpp and .cu files
    # .cu files will be hipified to .hip for ROCm builds
    sources = all_files_in_dir(Path(csrc_source_files), name_extension="cpp")
    cu_sources = all_files_in_dir(Path(csrc_source_files), name_extension="cu")
    sources.extend(cu_sources)

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
        current_file_path = Path(__file__).parent.resolve()
        base_dir = current_file_path.parent
        sources = hipify(base_dir, csrc_source_files, sources, include_dirs[1:])

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden"]
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

    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME=/path/to/mpi must be set when compiling with NVTE_UB_WITH_MPI=1!"
        mpi_path = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_path / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")

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
        cxx_flags.append("-DNVTE_ENABLE_ROCSHMEM")
        mpi_home = Path(os.getenv("MPI_HOME", "/usr/lib/x86_64-linux-gnu/openmpi"))
        include_dirs.append(mpi_home / "include")
        library_dirs.append(mpi_home / "lib")
        libraries.append("mpi_cxx")


    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
