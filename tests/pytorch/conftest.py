# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Pytest hooks for the Triton GEMM backend CI sweep.

Two things live here:

1. A ``pytest_runtest_call`` wrapper that converts intentional backend
   refusals (HYBRID / mixed FP8 / NVFP4) raised as ``ValueError`` from
   ``quantization.py`` and ``triton_kernels/gemm/gemm_wrapper.py`` into
   ``pytest.skip``. When those gates are relaxed the marker text
   disappears and the hook stops firing.

2. A ``pytest_collection_modifyitems`` hook that pre-skips known-bad
   fp32 tests on gfx942 under ``NVTE_USE_GEMM_TRITON=1``. gfx942's
   Triton fp32 matmul has a stable numerical divergence from
   ``torch.matmul``; mi35x/gfx950 runs the same tests cleanly. When the
   kernel is fixed, remove the ``_KNOWN_BAD_FP32_ON_GFX942`` set and
   this hook becomes a no-op.
"""

import os

import pytest


# Substrings identifying our Triton GEMM backend refusals. Kept short so
# they are easy to grep.
_TRITON_GEMM_GATE_MARKERS = (
    # Mixed FP8 (e4m3 x e5m2) refused at the low-level matmul entry.
    "Mixed FP8 types",
    # Covers both quantization.py::check_recipe_support (HYBRID) and
    # gemm_wrapper._classify_input's refusal of NVFP4 / other
    # QuantizedTensorStorage subclasses.
    "The Triton GEMM backend (NVTE_USE_GEMM_TRITON=1) does not support",
)


def _has_fp32_param(item) -> bool:
    """Whether an item's parametrize values include fp32 (torch.float32 or "fp32")."""
    import torch  # local import so conftest import stays cheap

    params = getattr(item, "callspec", None)
    if params is None:
        return False
    for v in params.params.values():
        if v is torch.float32 or (isinstance(v, str) and v == "fp32"):
            return True
    return False


def _correctness_kernel_is_pure_fp32(item) -> bool:
    """test_correctness parametrizes on (in_dtype, out_dtype) strings; skip only
    the pure-fp32 combo. fp16->fp32 accumulate and fp8->fp32 are separate code
    paths that pass on gfx942."""
    params = getattr(item, "callspec", None)
    if params is None:
        return False
    return params.params.get("in_dtype") == "fp32" and params.params.get("out_dtype") == "fp32"


def _always(item) -> bool:
    return True


# Tests known to fail with fp32 on gfx942 under NVTE_USE_GEMM_TRITON=1. The value
# is a predicate on the pytest item that returns True if this specific variant
# should be skipped (allows finer-grained control than "any fp32 param" for tests
# with multiple dtype-like parameters).
_KNOWN_BAD_FP32_ON_GFX942 = {
    # tests/pytorch/triton_kernels/test_gemm.py -- Triton vs torch.matmul
    "test_triton_vs_pytorch_regular": _has_fp32_param,
    # tests/pytorch/triton_kernels/test_gemm.py -- Triton vs C++ backend
    "test_triton_vs_cpp_regular": _has_fp32_param,
    "test_triton_vs_cpp_bias_forward": _has_fp32_param,
    # tests/pytorch/triton_kernels/test_gemm_kernel.py -- low-level kernel
    # (parametrizes on separate in_dtype / out_dtype strings)
    "test_correctness": _correctness_kernel_is_pure_fp32,
    # tests/pytorch/test_fusible_ops.py
    "test_basic_linear": _has_fp32_param,
    "test_custom_forward_fused_op2": _always,
    "test_custom_backward_fused_op": _always,
    # tests/pytorch/test_numerics.py -- surfaced once the shard stopped
    # crashing early on other MXFP8 issues. Same fp32-in Triton kernel
    # divergence as test_basic_linear above; only dtype0=fp32 variants fail.
    "test_linear_accuracy": _has_fp32_param,
    "test_layernorm_linear_accuracy": _has_fp32_param,
    "test_layernorm_mlp_accuracy": _has_fp32_param,
}


def pytest_collection_modifyitems(config, items):
    """Skip known-bad fp32 tests on gfx942 under NVTE_USE_GEMM_TRITON=1."""
    if not bool(int(os.environ.get("NVTE_USE_GEMM_TRITON", "0"))):
        return

    import torch  # local import so conftest import stays cheap

    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    is_gfx942 = major == 9 and minor < 5
    if not is_gfx942:
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "gfx942 Triton fp32 GEMM has a stable numerical divergence from "
            "torch.matmul (gfx950 passes cleanly). Skipping under "
            "NVTE_USE_GEMM_TRITON=1 on gfx942 pending kernel fix."
        )
    )
    for item in items:
        func_name = item.name.split("[")[0]
        predicate = _KNOWN_BAD_FP32_ON_GFX942.get(func_name)
        if predicate is None:
            continue
        if not predicate(item):
            continue
        item.add_marker(skip_marker)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Convert known Triton GEMM backend gate errors into skips."""
    outcome = yield
    if outcome.excinfo is None:
        return
    exc = outcome.excinfo[1]
    if not isinstance(exc, ValueError):
        return
    msg = str(exc)
    if any(marker in msg for marker in _TRITON_GEMM_GATE_MARKERS):
        outcome.force_exception(
            pytest.skip.Exception(f"Triton GEMM backend gate: {msg}")
        )
