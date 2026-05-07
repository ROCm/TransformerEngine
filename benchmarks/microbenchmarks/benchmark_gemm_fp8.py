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
    MODEL_CONFIGS, M_SIZE_LIST, gemm_shapes,
    time_func, compute_tflops, run_benchmarks,
)

FP8_RECIPE = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)

ACTIVE_SHAPES = gemm_shapes(MODEL_CONFIGS)


def _generate_gemm_test_cases():
    test_cases = []
    for M in M_SIZE_LIST:
        for case_name, (N, K) in ACTIVE_SHAPES.items():
            test_cases.append({
                "Case": case_name,
                "M": M,
                "N": N,
                "K": K,
                "dtype": torch.bfloat16,
            })
    return test_cases


def bench_fp8_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn(M, N, dtype=dtype, device=device)

    def fwd_func():
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            return linear(x)

    def fwd_bwd_func():
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            out = linear(x)
        out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops

    fwd_ms = time_func(fwd_func)
    fwd_bwd_ms = time_func(fwd_bwd_func)
    bwd_ms = fwd_bwd_ms - fwd_ms

    fwd_tflops = compute_tflops(fwd_flops, fwd_ms)
    bwd_tflops = compute_tflops(bwd_flops, bwd_ms)

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS (derived)")

    return {
        "FP8 Forward Time (ms)": f"{fwd_ms:.2f}",
        "FP8 Forward TFLOPS": f"{fwd_tflops:.2f}",
        "FP8 Backward Time (ms)": f"{bwd_ms:.2f}",
        "FP8 Backward TFLOPS": f"{bwd_tflops:.2f}",
    }


if __name__ == "__main__":
    run_benchmarks(
        test_cases=_generate_gemm_test_cases(),
        bench_fn=bench_fp8_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
        metric_columns=[
            "FP8 Forward Time (ms)", "FP8 Forward TFLOPS",
            "FP8 Backward Time (ms)", "FP8 Backward TFLOPS",
        ],
        default_csv="benchmark_gemm_fp8.csv",
    )
