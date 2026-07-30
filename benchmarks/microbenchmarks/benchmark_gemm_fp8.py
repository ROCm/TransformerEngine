#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
FP8 GEMM micro-benchmark using te.Linear under fp8_autocast.

Same model shapes as benchmark_gemm.py.
Output: benchmark_gemm_fp8.csv (written to cwd)
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from utils import (
    generate_gemm_test_cases,
    time_funcs, compute_tflops, make_forward_backward_metric_records, run_benchmarks,
    make_input,
)

RECIPES = {
    "hybrid": DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
    ),
}

FP8_RECIPE = RECIPES["hybrid"]

BENCHMARK_LABEL = "FP8 GEMM"


def bench_fp8_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    next_x = make_input((M, K), dtype, device=device, requires_grad=True)
    grad_out = torch.randn(M, N, dtype=dtype, device=device)

    def fwd_func():
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            return linear(next_x())

    def fwd_bwd_func():
        x = next_x()
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            out = linear(x)
            out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops

    ms = time_funcs({"fwd": fwd_func, "fwd_bwd": fwd_bwd_func})
    fwd_ms = ms["fwd"].median * 1e3
    bwd_ms = ms["fwd_bwd"].median * 1e3 - fwd_ms

    fwd_tflops = compute_tflops(fwd_flops, fwd_ms)
    bwd_tflops = compute_tflops(bwd_flops, bwd_ms)

    return make_forward_backward_metric_records(
        BENCHMARK_LABEL,
        "TFLOPS",
        fwd_ms,
        fwd_tflops,
        bwd_ms,
        bwd_tflops,
        backward_derived=True,
        fwd_measurement=ms["fwd"],
        fwd_bwd_measurement=ms["fwd_bwd"],
    )


if __name__ == "__main__":
    run_benchmarks(
        test_cases=generate_gemm_test_cases(),
        bench_fn=bench_fp8_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
    )
