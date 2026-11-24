#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Normalization micro-benchmark using the fusible-ops LayerNorm / RMSNorm.

Run with ``python benchmark_normalization.py`` (a pytest module under the hood;
see conftest.py).

Sweeps BF16 plus the quantized-output precisions (FP8, blockwise FP8, MXFP8,
MXFP4, NVFP4) that TE training recipes produce. In FP8/FP4 training the norm is
fused with the following
Linear (LayerNormLinear / LayerNormMLP) and writes its output already
quantized. That is reproduced here with ``ops.Sequential(Norm, Quantize)``: the
op fuser threads the Quantize op's input quantizer into the norm, so the norm
writes the quantized output directly under ``autocast``. The Quantize op is an
identity outside ``autocast``, so the bf16 baseline uses the same harness.

The backward (input and weight gradients) runs for bf16 only: the norm saves
its bf16 input and the Quantize op passes the gradient through, so the backward
reads and writes bf16 whatever the forward output precision. On ROCm, blockwise
FP8 and MXFP8/MXFP4/NVFP4 norms run unfused (bf16 norm + separate cast kernel),
so those forward rows measure norm + cast as LayerNormLinear sees it. Precisions
unsupported on the current device are skipped.

These are memory-bound; we report GB/s (forward: input read + output write;
backward: input and output-gradient read + input-gradient write).
Output: benchmark_normalization.csv (written to cwd)
"""

if __name__ == "__main__":
    # Enter pytest before torch touches the GPU, so conftest's neighbor snapshot excludes us.
    import sys

    import pytest

    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

import functools
import sys
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops
from utils import (
    MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    build_recipes,
    apply_backend_env, compute_gbps, direction_records,
    make_input, te_honors_env,
)

NORM_TYPES = [
    ("RMSNorm",   ops.RMSNorm),
    ("LayerNorm", ops.LayerNorm),
]

BENCHMARK_LABEL = "Normalization"

# bf16 is the plain baseline (Quantize is an identity outside autocast).
# The blockwise FP8 scale format (E8M0 vs FP32) only matters to the GEMMs, which run both.
RECIPES = build_recipes(names=("bf16", "fp8", "fp8_block_e8m0", "mxfp8", "mxfp4", "nvfp4"))

# Forward output bytes/elem by precision. Under autocast the norm quantizes
# through the recipe-created quantizers, which default to columnwise usage on, so
# it writes BOTH rowwise and columnwise data (plus both MXFP8 scale buffers) --
# the norm->Linear training layout (rowwise feeds fprop, columnwise feeds wgrad).
# Input is always bf16 (2 bytes/elem).
#   bf16  : 2.0                    (single bf16 output; Quantize is identity)
#   fp8   : 2.0                    (rowwise + columnwise E4M3/E5M2 data; scale ~ 0)
#   fp8_block_e8m0: 2.0 + 2*4/128 = 2 + 1/16  (rowwise + columnwise data, power-of-2 scale
#                   per 1x128, stored as FP32)
#   mxfp8 : 2.0 + 2/32 = 2 + 1/16  (rowwise + columnwise data + both E8M0 scales)
#   mxfp4 : 1.0 + 2/32 = 1 + 1/16  (packed E2M1 data x2 + E8M0 scale per 32)
#   nvfp4 : 1.0 + 2/16 = 1 + 1/8   (packed E2M1 data x2 + E4M3 scale per 16)
_FWD_WRITE_BYTES = {
    "bf16": 2.0,
    "fp8": 2.0,
    "fp8_block_e8m0": 2.0 + 1.0 / 16,
    "mxfp8": 2.0 + 1.0 / 16,
    "mxfp4": 1.0 + 1.0 / 16,
    "nvfp4": 1.0 + 1.0 / 8,
}


# Backend axis (None unsets, so "hip" is the native C++ path even if the ambient
# env has a toggle set). "triton" flips the Triton RMSNorm/LayerNorm kernel (each
# NormType reads only its own toggle).
NORM_BACKENDS = {
    "hip": {"NVTE_USE_RMSNORM_TRITON": None, "NVTE_USE_LAYERNORM_TRITON": None},
    "triton": {"NVTE_USE_RMSNORM_TRITON": "1", "NVTE_USE_LAYERNORM_TRITON": "1"},
}

_NORM_CLS = {name: cls for name, cls in NORM_TYPES}


@functools.lru_cache(maxsize=1)
def _triton_norm_handles_mxfp4():
    """False on TE builds whose Triton norm predates MXFP4 output support."""
    try:
        src = (Path(te.__file__).parent / "triton_kernels/norms_common.py").read_text()
    except OSError:
        return False
    return "MXFP4Quantizer" in src


def generate_cases():
    """Cross models x norm type x precision x backend x direction x M (bwd for bf16 only)."""
    cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for norm_name in _NORM_CLS:
            for precision in RECIPES:
                directions = ("fwd", "bwd") if precision == "bf16" else ("fwd",)
                for backend in NORM_BACKENDS:
                    for direction in directions:
                        for M in M_SIZE_LIST:
                            cases.append({
                                "Case": model_name,
                                "NormType": norm_name,
                                "Precision": precision,
                                "Backend": backend,
                                "Direction": direction,
                                "M": M,
                                "hidden_size": hidden,
                            })
    return cases


def _case_id(c):
    return (f"{c['Case']}-{c['NormType']}-{c['Precision']}-{c['Backend']}-{c['Direction']}"
            f"-M{c['M']}")


def bench_norm(NormType, Precision, Direction, M, hidden_size):
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
    params = list(model.parameters())
    next_x = make_input((M, hidden_size), dtype, device=device,
                        requires_grad=Direction == "bwd")

    def fwd_func():
        xb = next_x()
        with te.autocast(enabled=use_fp8, recipe=recipe):
            return model(xb), xb

    grad_out = torch.randn((M, hidden_size), dtype=dtype, device=device)

    def bwd_func(fwd_out):
        out, xb = fwd_out
        out.backward(grad_out)
        xb.grad = None
        for p in params:
            p.grad = None

    fwd_func()

    # BF16 read + quantized write.
    fwd_bytes = int(M * hidden_size * (2 + _FWD_WRITE_BYTES[Precision]))
    # BF16 x and dy read + dx write; the weight-sized terms are negligible.
    bwd_bytes = M * hidden_size * 6

    return direction_records(
        Direction, BENCHMARK_LABEL, "GB/s", compute_gbps,
        fwd_func, bwd_func, fwd_bytes, bwd_bytes,
    )


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = generate_cases()
        metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])


@pytest.mark.benchmark
def test_norm(request, microbench, case, monkeypatch):
    if case["Backend"] == "triton":
        var = ("NVTE_USE_RMSNORM_TRITON" if case["NormType"] == "RMSNorm"
               else "NVTE_USE_LAYERNORM_TRITON")
        if not te_honors_env(var):
            pytest.skip("Triton norm backend not available in this TE build")
        if case["Precision"] == "mxfp4" and not _triton_norm_handles_mxfp4():
            # Imperative xfail (not run): the GPU memory fault would abort the whole session.
            pytest.xfail("Triton norm lacks MXFP4 support in this TE build (GPU memory fault)")
    if case["Backend"] == "hip" and case["hidden_size"] == 16384:
        # Llama3.1-405B: the hip norm at hidden 16384 is a multi-CTA cooperative launch, which
        # makes HIP's exit-time teardown segfault (see pytest_unconfigure in conftest.py).
        request.config._skip_hip_teardown_session = request.session
    # The dense-GEMM selector also drives autocast's recipe check (TRITON rejects fp8 HYBRID).
    monkeypatch.delenv("NVTE_GEMM_BACKEND", raising=False)
    apply_backend_env(monkeypatch, NORM_BACKENDS[case["Backend"]])
    microbench.run(
        case,
        lambda: bench_norm(case["NormType"], case["Precision"], case["Direction"], case["M"],
                           case["hidden_size"]),
    )
