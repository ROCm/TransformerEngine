# This file was modified for portability to AMDGPU
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import (
    rocm_build,
    hipify,
    all_files_in_dir,
    cuda_version,
    cuda_path,
)


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = [
        csrc_source_files / "common.cu",
        csrc_source_files / "ts_fp8_op.cpp",
    ] + all_files_in_dir(extensions_dir)

    if not rocm_build():
        sources.extend([
            csrc_source_files / "userbuffers" / "ipcsocket.cc",
            csrc_source_files / "userbuffers" / "userbuffers.cu",
            csrc_source_files / "userbuffers" / "userbuffers-host.cpp",
        ])
        
    # Header files
    include_dirs = [
        common_header_files,
        common_header_files / "common",
        common_header_files / "common" / "include",
        csrc_header_files,
    ]

    if rocm_build():
        current_file_path = Path(__file__).parent.resolve()
        base_dir = current_file_path.parent
        sources = hipify(base_dir, csrc_source_files, sources, include_dirs)

    # Compiler flags
    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
    ]
    if rocm_build():
        nvcc_flags = [
            "-O3",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT16_OPERATORS__",
            "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT162_OPERATORS__",
            "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        ]
    else:
        nvcc_flags = [
            "-O3",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ]

    # Version-dependent CUDA options
    if rocm_build():
        ##TODO: Figure out which hipcc version starts to support this parallel compilation
        nvcc_flags.extend(["-parallel-jobs=4"])
    else:
        try:
            version = cuda_version()
        except FileNotFoundError:
            print("Could not determine CUDA Toolkit version")
        else:
            if version >= (11, 2):
                nvcc_flags.extend(["--threads", "4"])
            if version >= (11, 0):
                nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
            if version >= (11, 8):
                nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Libraries
    library_dirs = []
    libraries = []
    if os.getenv("NVTE_UB_WITH_MPI"):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME must be set when compiling with NVTE_UB_WITH_MPI=1"
        mpi_home = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_home / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")
        nvcc_flags.append("-DNVTE_UB_WITH_MPI")
        library_dirs.append(mpi_home / "lib")
        libraries.append("mpi")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
