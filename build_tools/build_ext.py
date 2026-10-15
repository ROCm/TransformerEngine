# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import copy
import os
import subprocess
import sys
import sysconfig
import tempfile
import time

from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Type

import setuptools
from setuptools.command.sdist import sdist as _sdist

from .te_version import te_version, is_local_version_used, version_file
from .utils import (
    rocm_build,
    rocm_path,
    cmake_bin,
    debug_build_enabled,
    found_ninja,
    nvcc_path,
    get_max_jobs_for_parallel_build,
)


class CMakeExtension(setuptools.Extension):
    """CMake extension module"""

    def __init__(
        self,
        name: str,
        cmake_path: Path,
        cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])  # No work for base class
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = [] if cmake_flags is None else cmake_flags

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:
        # Make sure paths are str
        _cmake_bin = str(cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if debug_build_enabled() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DPython_SITEARCH={sysconfig.get_path('platlib')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        if bool(int(os.getenv("NVTE_USE_CCACHE", "0"))):
            ccache_bin = os.getenv("NVTE_CCACHE_BIN", "ccache")
            configure_command += [
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache_bin}",
                f"-DCMAKE_CUDA_COMPILER_LAUNCHER={ccache_bin}",
            ]
        configure_command += self.cmake_flags

        #ROCm: below variable is not used by CMake build. Leave it here for reference
        #import pybind11
        #pybind11_dir = Path(pybind11.__file__).resolve().parent
        #pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
        #configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir, "--verbose"]
        install_command = [_cmake_bin, "--install", build_dir, "--verbose"]

        # Check whether parallel build is restricted
        max_jobs = get_max_jobs_for_parallel_build()
        if found_ninja():
            configure_command.append("-GNinja")
        elif rocm_build():
            raise RuntimeError(f"This project requires the Ninja build system. Install it using 'pip install ninja'.")
        build_command.append("--parallel")
        if max_jobs > 0:
            build_command.append(str(max_jobs))

        # Run CMake commands
        start_time = time.perf_counter()
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")

        total_time = time.perf_counter() - start_time
        print(f"Time for build_ext: {total_time:.2f} seconds")


def get_build_ext(
    extension_cls: Type[setuptools.Extension], framework_extension_only: bool = False
):
    class _CMakeBuildExtension(extension_cls):
        """Setuptools command with support for CMake extension modules"""

        def finalize_options(self) -> None:
            super().finalize_options()
            # Persist the intermediate object directory (build_temp) across builds.
            # Framework extensions (e.g. transformer_engine_torch) are compiled by
            # setuptools/pip into build_temp, but pip normally points it at an
            # ephemeral temp dir, so every `pip install` recompiles all objects from
            # scratch. Rooting it at a stable location lets the underlying ninja skip
            # unchanged objects. Mirrors the persistent CMake build dir above.
            root_dir = Path(__file__).resolve().parent.parent
            build_temp = root_dir / "build" / "ext_temp"
            build_temp.mkdir(parents=True, exist_ok=True)
            self.build_temp = str(build_temp)

        def run(self) -> None:
            # Build CMake extensions
            for ext in self.extensions:
                package_path = Path(self.get_ext_fullpath(ext.name))
                install_dir = package_path.resolve().parent
                if not isinstance(ext, CMakeExtension):
                    continue

                print(f"Building CMake extension {ext.name}")
                configured_build_dir = os.getenv("NVTE_CMAKE_BUILD_DIR")
                if not configured_build_dir and self.inplace:
                    root_dir = Path(__file__).resolve().parent.parent
                    configured_build_dir = root_dir / "build" / "cmake"

                if configured_build_dir:
                    # A persistent build directory enables incremental builds.
                    build_dir = Path(configured_build_dir).resolve()
                    build_dir.mkdir(parents=True, exist_ok=True)
                    ext._build_cmake(
                        build_dir=build_dir,
                        install_dir=install_dir,
                    )
                    continue

                # Isolate CMake state between concurrent and successive builds.
                build_temp = Path(self.build_temp)
                build_temp.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"cmake-build-{ext.name}-",
                    dir=build_temp,
                ) as build_dir:
                    print(f"Building CMake extension {ext.name} in temporary directory {build_dir}")
                    ext._build_cmake(
                        build_dir=Path(build_dir),
                        install_dir=install_dir,
                    )

            # Build non-CMake extensions as usual
            all_extensions = self.extensions
            self.extensions = [
                ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
            ]
            super().run()
            self.extensions = all_extensions

            # Ensure that shared objects files for source and PyPI installations live
            # in separate directories to avoid conflicts during install and runtime.
            lib_dir = (
                "wheel_lib"
                if (not rocm_build() and bool(int(os.getenv("NVTE_RELEASE_BUILD", "0")))) or framework_extension_only
                else ""
            )

            # Ensure that binaries are not in global package space.
            # For editable/inplace builds this is not a concern as
            # the SOs will be in a local directory anyway.
            if not self.inplace:
                target_dir = install_dir / "transformer_engine" / lib_dir
                target_dir.mkdir(exist_ok=True, parents=True)

                for ext in Path(self.build_lib).glob("*.so"):
                    self.copy_file(ext, target_dir)
                    os.remove(ext)

        def build_extensions(self):
            # For core lib + JAX install, fix build_ext from pybind11.setup_helpers
            # to handle CUDA files correctly.
            # Upstream uses get_frameworks() here which is incorrectly works when install from
            # release (sdist) wheel on a system with both frameworks installed.
            ext_names = [ext.name for ext in self.extensions]
            if ("transformer_engine_torch" not in ext_names and
                "transformer_engine_rocm_torch" not in ext_names):
                # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
                # extra_compile_args is a dict.
                for ext in self.extensions:
                    if isinstance(ext.extra_compile_args, dict):
                        for target in ["cxx", "nvcc"]:
                            if target not in ext.extra_compile_args.keys():
                                ext.extra_compile_args[target] = []

                # Define new _compile method that redirects to NVCC for .cu and .cuh files.
                # Also redirect .hip files to HIPCC
                original_compile_fn = self.compiler._compile
                if not framework_extension_only:
                    self.compiler.src_extensions += [".cu", ".cuh", ".hip"]

                def _compile_fn(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
                    # Copy before we make any modifications.
                    cflags = copy.deepcopy(extra_postargs)
                    original_compiler = self.compiler.compiler_so
                    try:

                        if rocm_build():
                            _, nvcc_bin = rocm_path()
                        else:
                            nvcc_bin = nvcc_path()
                        original_compiler = self.compiler.compiler_so

                        if (
                            os.path.splitext(src)[1] in [".cu", ".cuh"]
                            and not framework_extension_only
                        ):
                            self.compiler.set_executable("compiler_so", str(nvcc_bin))
                            if isinstance(cflags, dict):
                                cflags = cflags["nvcc"]

                            # Add -fPIC if not already specified
                            if not any("-fPIC" in flag for flag in cflags):
                                if rocm_build():
                                    cflags.append("-fPIC")
                                else:
                                    cflags.extend(["--compiler-options", "'-fPIC'"])

                            if not rocm_build():
                                # Forward unknown options
                                if not any("--forward-unknown-opts" in flag for flag in cflags):
                                    cflags.append("--forward-unknown-opts")

                        elif isinstance(cflags, dict):
                            cflags = cflags["cxx"]

                        # Append -std=c++17 if not already in flags
                        if not any(flag.startswith("-std=") for flag in cflags):
                            cflags.append("-std=c++17")

                        return original_compile_fn(obj, src, ext, cc_args, cflags, pp_opts)

                    finally:
                        # Put the original compiler back in place.
                        self.compiler.set_executable("compiler_so", original_compiler)

                self.compiler._compile = _compile_fn

            super().build_extensions()

    return _CMakeBuildExtension


class SdistWithLocalVersion(_sdist):
    """
    Override sdist to modify the *staged* copy of VERSION.txt.
    """
    def make_release_tree(self, base_dir, files):
        # First let setuptools stage the files into base_dir
        super().make_release_tree(base_dir, files)

        if is_local_version_used():
            version_file(base_dir).write_text(te_version() + "\n", encoding="utf-8")
