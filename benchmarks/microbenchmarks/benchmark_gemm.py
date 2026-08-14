#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import torch
import transformer_engine.pytorch as te
from utils import (
    generate_gemm_test_cases,
    time_func, compute_tflops, make_forward_backward_metric_records, run_benchmarks,
    make_input,
)

BENCHMARK_LABEL = "GEMM"


def bench_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    next_x = make_input((M, K), dtype, device=device, requires_grad=True)

    fwd_func = lambda: linear(next_x())
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        xb = next_x()
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
        test_cases=generate_gemm_test_cases(),
        bench_fn=bench_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
    )
