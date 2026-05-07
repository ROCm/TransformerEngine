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

# Select which configs / shapes to run (comment/uncomment as needed)
ACTIVE_CONFIGS = [
    MODEL_CONFIGS[0],   # Llama3-8B/TP1
    # MODEL_CONFIGS[1], # Llama3-8B/TP8
    # MODEL_CONFIGS[2], # Llama3-70B/TP8
    # MODEL_CONFIGS[3], # Llama3-405B/TP8
    # MODEL_CONFIGS[4], # Qwen2.5-7B/TP1
    # MODEL_CONFIGS[5], # Qwen2.5-72B/TP8
]

ACTIVE_SHAPES = gemm_shapes(ACTIVE_CONFIGS)
# To restrict shapes, filter the dict:
ACTIVE_SHAPES = {k: v for k, v in ACTIVE_SHAPES.items() if "QKV" in k}


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

    def bwd_func():
        out = linear(x)
        out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

    fwd_ms = time_func(fwd_func)
    fwd_bwd_ms = time_func(bwd_func)
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
