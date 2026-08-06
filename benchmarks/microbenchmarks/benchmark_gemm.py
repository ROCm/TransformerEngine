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
    time_forward_backward, compute_tflops, make_forward_backward_metric_records, run_benchmarks,
)

BENCHMARK_LABEL = "GEMM"


def bench_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: linear(x)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def zero_grads():
        x.grad = None
        linear.weight.grad = None

    def fwd_bwd_func():
        out = linear(x)
        out.backward(grad_out)
        zero_grads()

    fwd_bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

    fwd_ms, bwd_ms, record_kwargs = time_forward_backward(
        fwd_func, fwd_bwd_func, grad_out
    )

    fwd_tflops = compute_tflops(fwd_flops, fwd_ms)
    bwd_tflops = compute_tflops(bwd_flops, bwd_ms)

    return make_forward_backward_metric_records(
        BENCHMARK_LABEL,
        "TFLOPS",
        fwd_ms,
        fwd_tflops,
        bwd_ms,
        bwd_tflops,
        **record_kwargs,
    )


if __name__ == "__main__":
    run_benchmarks(
        test_cases=generate_gemm_test_cases(),
        bench_fn=bench_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
    )
