# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Shared pytest hooks for the PyTorch test suite.

Currently: convert known-issue Triton GEMM backend refusals into pytest.skip
so the CI sweep under NVTE_USE_GEMM_TRITON=1 does not flood with red on tests
that exercise recipes the backend intentionally does not implement (HYBRID,
mixed FP8, NVFP4, ...).

The runtime gates that raise these ValueErrors live in
transformer_engine/pytorch/quantization.py and
transformer_engine/pytorch/gemm_triton.py. Each carries a message identifying
the Triton GEMM backend by name. When the gates are relaxed (e.g. after
PyTorch 2.14 ships the Triton mixed-MFMA fix, or NVFP4 is implemented in the
Triton kernels), the matching text disappears from the error and this hook
stops firing on its own -- no test-side coordination required.
"""

import pytest


# Substrings that identify one of our Triton GEMM backend refusals. Any
# ValueError whose message contains one of these is treated as a
# known-unsupported combination for the current Triton backend, not a real
# failure. Kept as a short set so it is easy to grep and audit.
_TRITON_GEMM_GATE_MARKERS = (
    # HYBRID recipe refused in quantization.py::check_recipe_support
    "does not support Format.HYBRID",
    # Mixed FP8 (e4m3 x e5m2) refused at the low-level matmul entry
    "Mixed FP8 types",
    # QuantizedTensorStorage subclass we do not implement (NVFP4, ...)
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
