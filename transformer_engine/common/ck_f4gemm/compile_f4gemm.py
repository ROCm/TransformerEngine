# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT build script for AITER f4gemm kernels (ASM + CK blockscale).

Mirrors the pattern of 3rdparty/aiter/op_tests/cpp/mha/compile.py.
Calls build_module() directly (not @compile_ops) because f4gemm modules
are pybind11 extensions that should only be compiled, not loaded/invoked
during the build step.

Produces two pybind11 extension modules:
  - module_gemm_a4w4_asm.so       (ASM pre-compiled .co kernel path)
  - module_gemm_a4w4_blockscale.so (CK blockscale templated path)
"""

import sys
import os
import argparse

# Never import aiter at module level — manipulate sys.path to reach jit.core
this_dir = os.path.dirname(os.path.abspath(__file__))
aiter_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "3rdparty", "aiter"))
sys.path.insert(0, os.path.join(aiter_root, "aiter"))
from jit.core import (  # noqa: E402
    build_module,
    AITER_CSRC_DIR,
    AITER_META_DIR,
    AITER_CONFIG_GEMM_A4W4,
)

BLOCKSCALE_TUNE_FILE = os.environ.get(
    "AITER_CONFIG_GEMM_A4W4", AITER_CONFIG_GEMM_A4W4
)


def compile_f4gemm_asm():
    """Build the module_gemm_a4w4_asm pybind11 extension."""
    build_module(
        md_name="module_gemm_a4w4_asm",
        srcs=[
            f"{AITER_CSRC_DIR}/pybind/gemm_a4w4_asm_pybind.cu",
            f"{AITER_CSRC_DIR}/py_itfs_cu/asm_gemm_a4w4.cu",
        ],
        flags_extra_cc=[],
        flags_extra_hip=[],
        blob_gen_cmd=[
            f"{AITER_META_DIR}/hsa/codegen.py -m f4gemm --output_dir {{}}"
        ],
        extra_include=[],
        extra_ldflags=[],
        verbose=True,
        is_python_module=True,
        is_standalone=False,
        torch_exclude=False,
    )


def compile_f4gemm_blockscale():
    """Build the module_gemm_a4w4_blockscale pybind11 extension."""
    flags_extra_hip = [
        "-mllvm -greedy-reverse-local-assignment=1",
        "-mllvm --amdgpu-use-amdgpu-trackers=1",
    ]
    hip_clang_path = os.environ.get("GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH")
    if hip_clang_path and os.path.exists(hip_clang_path):
        prev = os.environ.get("HIP_CLANG_PATH")
        os.environ["HIP_CLANG_PATH"] = hip_clang_path

    build_module(
        md_name="module_gemm_a4w4_blockscale",
        srcs=[
            f"{AITER_CSRC_DIR}/pybind/gemm_a4w4_blockscale_pybind.cu",
            f"{AITER_CSRC_DIR}/py_itfs_cu/gemm_common.cu",
            f"{AITER_CSRC_DIR}/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu",
        ],
        flags_extra_cc=[],
        flags_extra_hip=flags_extra_hip,
        blob_gen_cmd=[
            f"{AITER_CSRC_DIR}/ck_gemm_a4w4_blockscale/gen_instances.py"
            f" --working_path {{}} --tune_file {BLOCKSCALE_TUNE_FILE}"
        ],
        extra_include=[f"{AITER_CSRC_DIR}/ck_gemm_a4w4_blockscale/include"],
        extra_ldflags=[],
        verbose=True,
        is_python_module=True,
        is_standalone=False,
        torch_exclude=False,
    )

    if hip_clang_path and os.path.exists(hip_clang_path):
        if prev is not None:
            os.environ["HIP_CLANG_PATH"] = prev
        else:
            os.environ.pop("HIP_CLANG_PATH", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compile_f4gemm",
        description="AOT compile AITER f4gemm kernel modules",
    )
    parser.add_argument(
        "--api",
        default="",
        required=False,
        help="Which module(s) to build: 'asm', 'blockscale', or '' for both (default: both).",
    )
    args = parser.parse_args()

    if args.api == "asm":
        compile_f4gemm_asm()
    elif args.api == "blockscale":
        compile_f4gemm_blockscale()
    elif args.api == "":
        compile_f4gemm_asm()
        compile_f4gemm_blockscale()
    else:
        raise ValueError(
            f"Invalid --api value '{args.api}': must be 'asm', 'blockscale', or '' (both)"
        )
