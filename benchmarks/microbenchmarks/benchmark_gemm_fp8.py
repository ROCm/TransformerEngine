#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
FP8 GEMM micro-benchmark using te.Linear under fp8_autocast.

Same model shapes as benchmark_gemm.py:
  - Llama 3   8B (TP=1, TP=8), 70B (TP=8), 405B (TP=8)
  - Qwen 2.5  7B (TP=1), 72B (TP=8)

Each model contributes four GEMM shapes:
  QKV projection     (column-parallel)  N = (Qheads + 2*KVheads)*head_dim / TP, K = hidden
  Attention output   (row-parallel)     N = hidden, K = Qheads*head_dim / TP
  MLP Gate+Up        (column-parallel)  N = 2*intermediate / TP, K = hidden  (SwiGLU)
  MLP Down           (row-parallel)     N = hidden, K = intermediate / TP

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

Output: benchmark_gemm_fp8.csv (written to cwd)
"""

import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Sequence / batch-token sizes to sweep
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# (name, hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
MODEL_CONFIGS = [
    ("Llama3-8B/TP1",   4096,  14336,  32,  8, 128,  1),
    ("Llama3-8B/TP8",   4096,  14336,  32,  8, 128,  8),
    ("Llama3-70B/TP8",  8192,  28672,  64,  8, 128,  8),
    ("Llama3-405B/TP8", 16384, 53248, 128,  8, 128,  8),
    ("Qwen2.5-7B/TP1",  3584, 18944,  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 8192, 29568,  64,  8, 128,  8),
]

FP8_RECIPE = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)


def _generate_gemm_test_cases():
    test_cases = []
    for (name, hidden, intermediate, n_q, n_kv, hd, tp) in MODEL_CONFIGS:
        shapes = {
            f"{name}-QKV":     ((n_q * hd + 2 * n_kv * hd) // tp, hidden),
            f"{name}-AttnOut": (hidden,                             (n_q * hd) // tp),
            f"{name}-GateUp":  ((2 * intermediate) // tp,           hidden),
            f"{name}-Down":    (hidden,                             intermediate // tp),
        }
        for M in M_SIZE_LIST:
            for case_name, (N, K) in shapes.items():
                test_cases.append({
                    "Case": case_name,
                    "M": M,
                    "N": N,
                    "K": K,
                    "dtype": torch.bfloat16,
                })
    return test_cases


def bench_fp8_gemm(M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn(M, N, dtype=dtype, device=device)

    # Forward under fp8_autocast
    def fwd_func():
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            return linear(x)

    # Combined fwd+bwd (TE consumes saved state on backward, no retain_graph)
    def fwd_bwd_func():
        with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
            out = linear(x)
        out.backward(grad_out)
        x.grad = None
        linear.weight.grad = None

    # Sanity run
    fwd_func()
    fwd_bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops

    # Warmup
    for _ in range(20):
        fwd_func()
        fwd_bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100

    fwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(n_iters).mean * 1e3
    fwd_bwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func}).timeit(n_iters).mean * 1e3

    bwd_ms = max(fwd_bwd_ms - fwd_ms, 0.0)

    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS (derived)")

    return {
        "FP8 Forward Time (ms)": f"{fwd_ms:.2f}",
        "FP8 Forward TFLOPS": f"{fwd_tflops:.2f}",
        "FP8 Backward Time (ms)": f"{bwd_ms:.2f}",
        "FP8 Backward TFLOPS": f"{bwd_tflops:.2f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = _generate_gemm_test_cases()

    columns = [
        "Case", "M", "N", "K", "dtype",
        "FP8 Forward Time (ms)",
        "FP8 Forward TFLOPS",
        "FP8 Backward Time (ms)",
        "FP8 Backward TFLOPS",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*60}")
    print(f"WARMUP: {c}")
    print(f"{'='*60}")
    bench_fp8_gemm(M=c["M"], N=c["N"], K=c["K"], dtype=c["dtype"])

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case}")
        print(f"{'='*60}")
        try:
            metrics = bench_fp8_gemm(
                M=case["M"], N=case["N"], K=case["K"], dtype=case["dtype"]
            )
            row = {
                "Case": case["Case"],
                "M": case["M"],
                "N": case["N"],
                "K": case["K"],
                "dtype": str(case["dtype"]),
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"FAILED: {case}: {e}")
            raise

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_gemm_fp8.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
