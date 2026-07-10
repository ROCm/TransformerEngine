# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Convert Triton GEMM backend refusals (HYBRID, mixed FP8, NVFP4, ...)
into pytest.skip so the CI sweep under NVTE_USE_GEMM_TRITON=1 does not
flag intentionally-unsupported combinations as failures.

The gates raise ValueError from quantization.py and
triton_kernels/gemm/gemm_wrapper.py; when they are relaxed the marker
text disappears and this hook stops firing.
"""

import pytest


# Substrings identifying our Triton GEMM backend refusals. Kept short so
# they are easy to grep.
_TRITON_GEMM_GATE_MARKERS = (
    # Mixed FP8 (e4m3 x e5m2) refused at the low-level matmul entry.
    "Mixed FP8 types",
    # Covers both quantization.py::check_recipe_support (HYBRID) and
    # Float8TensorWrapper's refusal of NVFP4 / other QuantizedTensorStorage.
    "The Triton GEMM backend (NVTE_USE_GEMM_TRITON=1) does not support",
)


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
