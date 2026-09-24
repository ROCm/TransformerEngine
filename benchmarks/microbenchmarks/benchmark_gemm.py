#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Dense GEMM micro-benchmark using te.Linear across precisions and backends.

Run with ``python benchmark_gemm.py`` (a pytest module under the hood; see
conftest.py). Sweeps the shared model GEMM shapes over BF16 (the high-precision
baseline) plus every supported low-precision recipe (FP8, MXFP8, MXFP4, NVFP4)
via te.autocast, crossed with a selectable kernel Backend. Precisions whose
hardware/runtime support is unavailable on the current device are skipped
automatically.

Examples::

    python benchmark_gemm.py --csv                   # -> benchmark_gemm.csv
    python benchmark_gemm.py -k "bf16 and QKV"       # select shapes/precisions
    python benchmark_gemm.py -k triton               # select the Triton backend
    python benchmark_gemm.py -k flydsl --run-flydsl  # select the FlyDSL backend

Output: benchmark_gemm.csv (written to cwd when --csv is passed).
"""

import functools
import os
import warnings

import pytest
import torch
import transformer_engine.pytorch as te
from utils import (
    build_recipes,
    generate_gemm_test_cases,
    apply_backend_env, compute_tflops, direction_records,
    make_input, te_honors_env,
)

BENCHMARK_LABEL = "GEMM"

RECIPES = build_recipes()

# Env recipes to force a dense-GEMM kernel backend (None unsets the var). Per the
# dispatch: bf16 defaults to hipBLASLt; Triton is NVTE_GEMM_BACKEND=TRITON (PR #676,
# unified selector) or the legacy NVTE_USE_GEMM_TRITON=1 toggle (PR #667), resolved
# per build by _triton_gemm_env. mxfp8 defaults to HipKittens, forced to hipBLASLt via
# NVTE_ROCM_USE_HIPBLASLT_MXFP8 (rocm_gemm.cu). fp8 has a single backend. FlyDSL
# (gfx950-only, BF16/FP8/MXFP8) is NVTE_GEMM_BACKEND=FLYDSL (cpp_extensions/gemm.py).
_GEMM_TRITON = "NVTE_USE_GEMM_TRITON"
_HIPBLASLT_MXFP8 = "NVTE_ROCM_USE_HIPBLASLT_MXFP8"
_GEMM_BACKEND = "NVTE_GEMM_BACKEND"

# Triton's env is resolved at runtime by _triton_gemm_env (mechanism varies by build).
GEMM_BACKENDS = {
    "hipblaslt":  {_GEMM_TRITON: None, _HIPBLASLT_MXFP8: "1", _GEMM_BACKEND: None},
    "hipkittens": {_GEMM_TRITON: None, _HIPBLASLT_MXFP8: None, _GEMM_BACKEND: None},
    "flydsl":     {_GEMM_TRITON: None, _HIPBLASLT_MXFP8: None, _GEMM_BACKEND: "FLYDSL"},
}

# Backends with a real choice per precision (the supported-backends table).
_BACKENDS_BY_PRECISION = {
    "bf16": ["hipblaslt", "triton", "flydsl"],
    "fp8": ["hipblaslt", "flydsl"],
    "mxfp8": ["hipblaslt", "hipkittens", "flydsl"],
}


def _backends_for(precision):
    return _BACKENDS_BY_PRECISION.get(precision, ["hipblaslt"])


@functools.lru_cache(maxsize=1)
def _triton_gemm_env():
    """Env that forces the Triton GEMM backend on the installed build, or None.

    PR #676 unified selection under NVTE_GEMM_BACKEND=TRITON; older builds (PR
    #667) use the legacy NVTE_USE_GEMM_TRITON=1 toggle. Prefer the unified selector.
    """
    if te_honors_env(_GEMM_BACKEND):
        return {_GEMM_BACKEND: "TRITON", _GEMM_TRITON: None, _HIPBLASLT_MXFP8: None}
    if te_honors_env(_GEMM_TRITON):
        return {_GEMM_TRITON: "1", _GEMM_BACKEND: None, _HIPBLASLT_MXFP8: None}
    return None


def _triton_gemm_supported():
    return _triton_gemm_env() is not None


@functools.lru_cache(maxsize=1)
def _flydsl_gemm_supported():
    if not te_honors_env(_GEMM_BACKEND):
        return False
    try:
        from transformer_engine.pytorch.utils import get_device_compute_capability
        if get_device_compute_capability() != (9, 5):
            return False
        from transformer_engine.pytorch.flydsl_kernels.gemm import (  # noqa: F401
            te_generic_gemm_flydsl,
        )
        return True
    except Exception:
        return False


def _flydsl_fell_back(fn):
    """Run *fn* once with FlyDSL fallback warnings on; True if FlyDSL fell back to C++."""
    prev = os.environ.get("NVTE_FLYDSL_GEMM_WARN_FALLBACK")
    os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = "1"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        return any("FlyDSL GEMM does not support" in str(w.message) for w in caught)
    finally:
        if prev is None:
            os.environ.pop("NVTE_FLYDSL_GEMM_WARN_FALLBACK", None)
        else:
            os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = prev


def generate_cases():
    """Cross the shared dense GEMM shapes with each precision, backend, direction."""
    cases = []
    for base in generate_gemm_test_cases():
        for precision in RECIPES:
            for backend in _backends_for(precision):
                for direction in ("fwd", "bwd"):
                    cases.append({
                        "Case": base["Case"],
                        "Precision": precision,
                        "Backend": backend,
                        "Direction": direction,
                        "M": base["M"],
                        "N": base["N"],
                        "K": base["K"],
                        "dtype": base["dtype"],
                    })
    return cases


def _case_id(c):
    return f"{c['Case']}-{c['Precision']}-{c['Backend']}-{c['Direction']}-M{c['M']}"


def bench_gemm(Case, Precision, Direction, M, N, K, dtype):
    device = "cuda"

    recipe = RECIPES[Precision]
    use_fp8 = recipe is not None

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    next_x = make_input((M, K), dtype, device=device, requires_grad=True)

    def fwd_func():
        with te.autocast(enabled=use_fp8, recipe=recipe):
            return linear(next_x())

    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        xb = next_x()
        with te.autocast(enabled=use_fp8, recipe=recipe):
            o = linear(xb)
            o.backward(grad_out)
        xb.grad = None
        linear.weight.grad = None

    if os.environ.get("NVTE_GEMM_BACKEND") == "FLYDSL" and _flydsl_fell_back(
        fwd_bwd_func if Direction == "bwd" else fwd_func
    ):
        pytest.skip("FlyDSL GEMM fell back to C++ for this shape/direction")

    fwd_flops = 2 * M * N * K
    # fp8/mxfp8 multi-streams the operand cast; the profiler under-counts those
    # concurrent kernels, so measure elapsed device time by makespan. bf16 is a
    # single GEMM kernel -- keep the profiler, which isolates sub-wall device time.
    return direction_records(
        Direction, BENCHMARK_LABEL, "TFLOPS", compute_tflops,
        fwd_func, fwd_bwd_func, fwd_flops, 2 * fwd_flops,
        kernel_method="event" if use_fp8 else None,
    )


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        params = [
            pytest.param(
                c, id=_case_id(c),
                marks=pytest.mark.flydsl if c["Backend"] == "flydsl" else (),
            )
            for c in cases
        ]
        metafunc.parametrize("case", params)


@pytest.mark.benchmark
def test_gemm(microbench, case, monkeypatch):
    if case["Precision"] == "mxfp4" and any(
        dim % 32 for dim in (case["M"], case["N"], case["K"])
    ):
        pytest.skip("MXFP4 GEMM needs M/N/K divisible by 32")
    if case["Backend"] == "triton" and not _triton_gemm_supported():
        pytest.skip("Triton GEMM backend not available in this TE build")
    if case["Backend"] == "hipkittens" and not te_honors_env(_HIPBLASLT_MXFP8):
        pytest.skip("HipKittens GEMM backend not available in this TE build")
    if case["Backend"] == "hipkittens" and (case["N"] % 256 or case["K"] % 256):
        pytest.skip("HipKittens GEMM needs 256-aligned N/K")
    if case["Backend"] == "flydsl" and not _flydsl_gemm_supported():
        pytest.skip("FlyDSL GEMM backend not available (needs gfx950 + flydsl package)")
    # Triton's selector varies by TE version (NVTE_GEMM_BACKEND=TRITON or legacy toggle).
    backend_env = _triton_gemm_env() if case["Backend"] == "triton" else GEMM_BACKENDS[case["Backend"]]
    apply_backend_env(monkeypatch, backend_env)
    microbench.run(
        case,
        lambda: bench_gemm(
            case["Case"], case["Precision"], case["Direction"],
            case["M"], case["N"], case["K"], case["dtype"],
        ),
    )


if __name__ == "__main__":
    import sys
    # Make the file runnable directly: python benchmark_gemm.py [--csv -k ...].
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
