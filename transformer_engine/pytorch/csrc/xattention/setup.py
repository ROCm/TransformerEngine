# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Standalone build for the xAttention <-> TransformerEngine PyTorch binding.

This is intentionally NOT wired into the main TE build. It compiles a small
pybind11/torch extension (`transformer_engine_xattention`) against a prebuilt
xAttention checkout and installs it into the active environment, where the
`transformer_engine.pytorch.xattention` wrapper imports it (guarded).

Usage (inside the build container):

    NVTE_XATTENTION_SOURCE_DIR=/path/to/xAttention \
        python setup.py build_ext --inplace

The xAttention checkout must already be built for the target arch, i.e. it must
contain:
    build/lib/libinterface.a
    build/lib/libcodeGen.a
    build/include/Config.h
    lib/<arch>/xattention_tileiras.a
"""

import glob
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

XA = os.environ.get("NVTE_XATTENTION_SOURCE_DIR")
if not XA:
    raise SystemExit("NVTE_XATTENTION_SOURCE_DIR must point to the xAttention checkout")
XA = os.path.abspath(XA)

build_lib = os.path.join(XA, "build", "lib")
interface_a = os.path.join(build_lib, "libinterface.a")
codegen_a = os.path.join(build_lib, "libcodeGen.a")

core_candidates = sorted(glob.glob(os.path.join(XA, "lib", "*", "xattention_tileiras.a")))
if not core_candidates:
    raise SystemExit(f"prebuilt core xattention_tileiras.a not found under {XA}/lib/<arch>/")
core_a = core_candidates[0]

for path in (interface_a, codegen_a, core_a):
    if not os.path.exists(path):
        raise SystemExit(f"missing xAttention artifact: {path}\nBuild xAttention first.")

rocm_home = os.environ.get("ROCM_HOME", os.environ.get("ROCM_PATH", "/opt/rocm"))

ext = CppExtension(
    name="transformer_engine_xattention",
    sources=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "xattention_binding.cpp"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "xattention_torch_shim.cpp"),
    ],
    include_dirs=[
        os.path.join(XA, "include"),
        os.path.join(XA, "build", "include"),
        os.path.join(rocm_home, "include"),
    ],
    extra_compile_args=["-std=c++20", "-D__HIP_PLATFORM_AMD__"],
    # Static archives have interdependencies (interface -> codeGen -> core and
    # back), so wrap them in a link group rather than relying on order.
    extra_link_args=[
        "-Wl,--start-group",
        interface_a,
        codegen_a,
        core_a,
        "-Wl,--end-group",
        f"-L{os.path.join(rocm_home, 'lib')}",
        "-lamdhip64",
        f"-Wl,-rpath,{os.path.join(rocm_home, 'lib')}",
    ],
)

setup(
    name="transformer_engine_xattention",
    version="0.1.0",
    description="xAttention backend binding for TransformerEngine (PyTorch)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    py_modules=[],
)

if __name__ == "__main__" and len(sys.argv) == 1:
    print(__doc__)
