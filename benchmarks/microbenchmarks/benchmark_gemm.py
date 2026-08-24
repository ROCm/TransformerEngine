#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Dense GEMM micro-benchmark using te.Linear across precisions.

Sweeps the shared model GEMM shapes over BF16 (the high-precision baseline)
plus every supported low-precision recipe (FP8, MXFP8, MXFP4, NVFP4) via
te.autocast. Precisions whose hardware/runtime support is unavailable on the
current device are skipped automatically.

Note: Transformer Engine exposes no MXFP6 recipe (MXFP6 exists only as a
low-level CK/CUTLASS data type), so it is not part of the sweep.

Output: benchmark_gemm.csv (written to cwd)
"""

import torch
import transformer_engine.pytorch as te
from utils import (
    build_recipes,
    generate_gemm_test_cases,
    time_func, compute_tflops, make_forward_backward_metric_records, run_benchmarks,
    make_input,
)

BENCHMARK_LABEL = "GEMM"

RECIPES = build_recipes()


def generate_precision_gemm_test_cases():
    """Cross the shared dense GEMM shapes with each supported precision."""
    test_cases = []
    for base_case in generate_gemm_test_cases():
        for precision in RECIPES:
            test_cases.append({**base_case, "Precision": precision})
    return test_cases


def bench_gemm(Case, Precision, M, N, K, dtype):
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
            out = linear(xb)
            out.backward(grad_out)
        xb.grad = None
        linear.weight.grad = None

    fwd_bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

    fwd_ms, fwd_measurement = time_func(fwd_func)
    fwd_bwd_ms, fwd_bwd_measurement = time_func(fwd_bwd_func)
    bwd_ms = fwd_bwd_ms - fwd_ms

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
        fwd_measurement=fwd_measurement,
        fwd_bwd_measurement=fwd_bwd_measurement,
    )


if __name__ == "__main__":
    run_benchmarks(
        test_cases=generate_precision_gemm_test_cases(),
        bench_fn=bench_gemm,
        param_columns=["Case", "Precision", "M", "N", "K", "dtype"],
    )
