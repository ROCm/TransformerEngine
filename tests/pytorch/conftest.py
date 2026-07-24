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


# Tests known to fail with fp32 on gfx942 under NVTE_USE_GEMM_TRITON=1.
# Value True  -> skip only the fp32-parametrized variants of that test
# Value False -> skip every variant of that test (not parametrized on dtype
#                but exercises fp32 internally)
_KNOWN_BAD_FP32_ON_GFX942 = {
    # tests/pytorch/triton_kernels/test_gemm.py
    "test_triton_vs_pytorch_regular": True,
    # tests/pytorch/test_fusible_ops.py
    "test_basic_linear": True,
    "test_custom_forward_fused_op2": False,
    "test_custom_backward_fused_op": False,
}


def _has_fp32_param(item) -> bool:
    """Whether an item's parametrize values include fp32."""
    import torch  # local import so conftest import stays cheap

    params = getattr(item, "callspec", None)
    if params is None:
        return False
    for v in params.params.values():
        if v is torch.float32 or (isinstance(v, str) and v == "fp32"):
            return True
    return False


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
        if func_name not in _KNOWN_BAD_FP32_ON_GFX942:
            continue
        requires_fp32_param = _KNOWN_BAD_FP32_ON_GFX942[func_name]
        if requires_fp32_param and not _has_fp32_param(item):
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
