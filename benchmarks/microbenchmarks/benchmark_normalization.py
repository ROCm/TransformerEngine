#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Normalization micro-benchmark using the fusible-ops LayerNorm / RMSNorm.

Sweeps BF16 plus the quantized-output precisions (FP8, MXFP8) that TE training
recipes produce. In FP8/FP4 training the norm is fused with the following
Linear (LayerNormLinear / LayerNormMLP) and writes its output already
quantized. That is reproduced here with ``ops.Sequential(Norm, Quantize)``: the
op fuser threads the Quantize op's input quantizer into the norm, so the norm
writes the quantized output directly under ``autocast``. The Quantize op is an
identity outside ``autocast``, so the bf16 baseline uses the same harness.

Forward only: the quantize epilogue is a forward-pass phenomenon; the norm
backward reads/writes high precision and is precision-independent. NVFP4/MXFP4
norm outputs are not swept (no validated norm->FP4 cast path). Precisions
unsupported on the current device are skipped.

These are memory-bound; we report GB/s (input read + output write).
Output: benchmark_normalization.csv (written to cwd)
"""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops
from utils import (
    MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    build_recipes,
    apply_backend_env, time_func, compute_gbps, make_metric_record,
    make_input,
)

NORM_TYPES = [
    ("RMSNorm",   ops.RMSNorm),
    ("LayerNorm", ops.LayerNorm),
]

BENCHMARK_LABEL = "Normalization Forward"

# Quantized-output precisions validated for the norm cast; tests/pytorch/
# triton_kernels/test_norms.py exercises fp8 and mxfp8 norm quantizers. bf16 is
# the plain baseline (Quantize is an identity outside autocast).
RECIPES = build_recipes(names=("bf16", "fp8", "mxfp8"))

# Forward output bytes/elem by precision. Under autocast the norm quantizes
# through the recipe-created quantizers, which default to columnwise usage on, so
# it writes BOTH rowwise and columnwise data (plus both MXFP8 scale buffers) --
# the norm->Linear training layout (rowwise feeds fprop, columnwise feeds wgrad).
# Input is always bf16 (2 bytes/elem).
#   bf16  : 2.0                    (single bf16 output; Quantize is identity)
#   fp8   : 2.0                    (rowwise + columnwise E4M3/E5M2 data; scale ~ 0)
#   mxfp8 : 2.0 + 2/32 = 2 + 1/16  (rowwise + columnwise data + both E8M0 scales)
_FWD_WRITE_BYTES = {
    "bf16": 2.0,
    "fp8": 2.0,
    "mxfp8": 2.0 + 1.0 / 16,
}


# Backend axis (None unsets, so "default" is the native path even if the ambient
# env has a toggle set). "triton" flips the Triton RMSNorm/LayerNorm kernel (each
# NormType reads only its own toggle).
NORM_BACKENDS = {
    "default": {"NVTE_USE_RMSNORM_TRITON": None, "NVTE_USE_LAYERNORM_TRITON": None},
    "triton": {"NVTE_USE_RMSNORM_TRITON": "1", "NVTE_USE_LAYERNORM_TRITON": "1"},
}

_NORM_CLS = {name: cls for name, cls in NORM_TYPES}


def generate_cases():
    """Cross models x norm type x precision x backend x M (forward only)."""
    cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for norm_name in _NORM_CLS:
            for precision in RECIPES:
                for backend in NORM_BACKENDS:
                    for M in M_SIZE_LIST:
                        cases.append({
                            "Case": model_name,
                            "NormType": norm_name,
                            "Precision": precision,
                            "Backend": backend,
                            "M": M,
                            "hidden_size": hidden,
                        })
    return cases


def _case_id(c):
    return f"{c['Case']}-{c['NormType']}-{c['Precision']}-{c['Backend']}-M{c['M']}"


def bench_norm(NormType, Precision, M, hidden_size):
    device = "cuda"
    dtype = torch.bfloat16

    recipe = RECIPES[Precision]
    use_fp8 = recipe is not None
    norm_op_cls = _NORM_CLS[NormType]

    # Norm followed by Quantize so the norm writes its output directly in the
    # target precision under autocast (identity when use_fp8 is False).
    model = ops.Sequential(
        norm_op_cls(hidden_size, device=device, dtype=dtype),
        ops.Quantize(),
    )
    next_x = make_input((M, hidden_size), dtype, device=device, requires_grad=False)

    def fwd_func():
        with te.autocast(enabled=use_fp8, recipe=recipe):
            return model(next_x())

    fwd_func()

    # BF16 read + quantized write.
    fwd_bytes = int(M * hidden_size * (2 + _FWD_WRITE_BYTES[Precision]))

    fwd_ms, fwd_measurement = time_func(fwd_func)
    return [make_metric_record(
        BENCHMARK_LABEL, fwd_ms, "GB/s", compute_gbps(fwd_bytes, fwd_ms),
        measurement=fwd_measurement,
    )]


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])


@pytest.mark.benchmark
def test_norm(microbench, case, monkeypatch):
    apply_backend_env(monkeypatch, NORM_BACKENDS[case["Backend"]])
    microbench.run(
        case,
        lambda: bench_norm(case["NormType"], case["Precision"], case["M"], case["hidden_size"]),
    )


if __name__ == "__main__":
    import sys
    # Make the file runnable directly: python benchmark_normalization.py [--csv -k ...].
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
