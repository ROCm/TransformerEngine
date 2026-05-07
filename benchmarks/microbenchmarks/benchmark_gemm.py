#!/usr/bin/env python
###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import torch
import transformer_engine.pytorch as te
from utils import (
    MODEL_CONFIGS, M_SIZE_LIST, gemm_shapes,
    time_func, compute_tflops, run_benchmarks,
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


def bench_gemm(Case, M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: linear(x)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        out = linear(x)
        out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    fwd_bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

    fwd_ms = time_func(fwd_func)
    fwd_bwd_ms = time_func(fwd_bwd_func)
    bwd_ms = fwd_bwd_ms - fwd_ms

    fwd_tflops = compute_tflops(fwd_flops, fwd_ms)
    bwd_tflops = compute_tflops(bwd_flops, bwd_ms)

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS (derived)")

    return {
        "TE Forward Time (ms)": f"{fwd_ms:.2f}",
        "TE Forward TFLOPS": f"{fwd_tflops:.2f}",
        "TE Backward Time (ms)": f"{bwd_ms:.2f}",
        "TE Backward TFLOPS": f"{bwd_tflops:.2f}",
    }


if __name__ == "__main__":
    run_benchmarks(
        test_cases=_generate_gemm_test_cases(),
        bench_fn=bench_gemm,
        param_columns=["Case", "M", "N", "K", "dtype"],
        metric_columns=[
            "TE Forward Time (ms)", "TE Forward TFLOPS",
            "TE Backward Time (ms)", "TE Backward TFLOPS",
        ],
        default_csv="benchmark_gemm.csv",
    )
