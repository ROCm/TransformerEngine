#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Dense GEMM micro-benchmark using te.Linear across precisions and backends.

Runs under pytest (see conftest.py). Sweeps the shared model GEMM shapes over
BF16 (the high-precision baseline) plus every supported low-precision recipe
(FP8, MXFP8, MXFP4, NVFP4) via te.autocast, crossed with a selectable kernel
Backend. Precisions whose hardware/runtime support is unavailable on the current
device are skipped automatically.

Examples::

    pytest benchmark_gemm.py --csv                 # -> benchmark_gemm.csv
    pytest benchmark_gemm.py -k "bf16 and QKV"     # select shapes/precisions
    pytest benchmark_gemm.py -k triton             # select the triton backend

Output: benchmark_gemm.csv (written to cwd when --csv is passed).
"""

import pytest
import torch
import transformer_engine.pytorch as te
from utils import (
    build_recipes,
    generate_gemm_test_cases,
    apply_backend_env, compute_tflops, direction_records,
    make_input,
)

BENCHMARK_LABEL = "GEMM"

RECIPES = build_recipes()

# Env recipes to force a dense-GEMM kernel backend (None unsets the var). Per the
# C++ dispatch: bf16 defaults to hipBLASLt, forced to Triton via NVTE_USE_GEMM_TRITON;
# mxfp8 defaults to HipKittens, forced to hipBLASLt via NVTE_ROCM_USE_HIPBLASLT_MXFP8
# (rocm_gemm.cu). fp8 has a single backend.
_GEMM_TRITON = "NVTE_USE_GEMM_TRITON"
_HIPBLASLT_MXFP8 = "NVTE_ROCM_USE_HIPBLASLT_MXFP8"

GEMM_BACKENDS = {
    "hipblaslt":  {_GEMM_TRITON: None, _HIPBLASLT_MXFP8: "1"},
    "triton":     {_GEMM_TRITON: "1", _HIPBLASLT_MXFP8: None},
    "hipkittens": {_GEMM_TRITON: None, _HIPBLASLT_MXFP8: None},
}

# Backends with a real choice per precision (the supported-backends table).
_BACKENDS_BY_PRECISION = {
    "bf16": ["hipblaslt", "triton"],
    "fp8": ["hipblaslt"],
    "mxfp8": ["hipblaslt", "hipkittens"],
}


def _backends_for(precision):
    return _BACKENDS_BY_PRECISION.get(precision, ["hipblaslt"])


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

    fwd_flops = 2 * M * N * K
    return direction_records(
        Direction, BENCHMARK_LABEL, "TFLOPS", compute_tflops,
        fwd_func, fwd_bwd_func, fwd_flops, 2 * fwd_flops,
    )


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])


@pytest.mark.benchmark
def test_gemm(microbench, case, monkeypatch):
    apply_backend_env(monkeypatch, GEMM_BACKENDS[case["Backend"]])
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
