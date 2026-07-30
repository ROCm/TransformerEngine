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
    time_funcs, compute_tflops, make_forward_backward_metric_records, run_benchmarks,
    make_input,
)

BENCHMARK_LABEL = "GEMM"


def bench_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    next_x = make_input((M, K), dtype, device=device, requires_grad=True)

    def fwd_func():
        return linear(next_x())

    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        x = next_x()
        out = linear(x)
        out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    fwd_bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

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
        bench_fn=bench_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
    )
