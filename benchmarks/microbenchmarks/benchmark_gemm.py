#!/usr/bin/env python
###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te

# Sequence / batch-token sizes to sweep
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# Model configurations
# Sources:
# - Llama 3 8B (hidden=4096, intermediate=14336, heads=32, kv_heads=8, head_dim=128)
#   https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json

# - Llama 3 70B (hidden=8192, intermediate=28672, heads=64, kv_heads=8, head_dim=128)
#   https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json

# - Llama 3 405B (hidden=16384, intermediate=53248, heads=128, kv_heads=8, head_dim=128)
#   https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json

# - Qwen 2.5 7B (hidden=3584, intermediate=18944, heads=28, kv_heads=4, head_dim=128)
#   https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json

# - Qwen 2.5 72B (hidden=8192, intermediate=29568, heads=64, kv_heads=8, head_dim=128)
#   https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

MODEL_CONFIGS = [
    # (name, hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
    ("Llama3-8B/TP1",   4096,  14336,  32,  8, 128,  1),
    ("Llama3-8B/TP8",   4096,  14336,  32,  8, 128,  8),
    ("Llama3-70B/TP8",  8192,  28672,  64,  8, 128,  8),
    ("Llama3-405B/TP8", 16384, 53248, 128,  8, 128,  8),
    ("Qwen2.5-7B/TP1",  3584, 18944,  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 8192, 29568,  64,  8, 128,  8),
]


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


def bench_gemm(M, N, K, dtype):
    device = "cuda"

    linear = te.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: linear(x)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def bwd_func():
        out = linear(x)
        out.backward(grad_out)
        # Clear grads so they don't accumulate across iterations
        x.grad = None
        linear.weight.grad = None

    bwd_func()

    fwd_flops = 2 * M * N * K
    bwd_flops = 2 * fwd_flops  # dX + dW

    # Warmup
    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100
    
    fwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(n_iters).mean * 1e3
    fwd_bwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(n_iters).mean * 1e3

    bwd_ms = max(fwd_bwd_ms - fwd_ms, 0.0)

    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS (derived)")

    return {
        "TE Forward Time (ms)": f"{fwd_ms:.2f}",
        "TE Forward TFLOPS": f"{fwd_tflops:.2f}",
        "TE Backward Time (ms)": f"{bwd_ms:.2f}",
        "TE Backward TFLOPS": f"{bwd_tflops:.2f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = _generate_gemm_test_cases()

    columns = [
        "Case", "M", "N", "K", "dtype",
        "TE Forward Time (ms)",
        "TE Forward TFLOPS",
        "TE Backward Time (ms)",
        "TE Backward TFLOPS",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*60}")
    print(f"WARMUP: {c}")
    print(f"{'='*60}")
    bench_gemm(M=c["M"], N=c["N"], K=c["K"], dtype=c["dtype"])

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case}")
        print(f"{'='*60}")
        try:
            metrics = bench_gemm(
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

    out_csv = "benchmark_gemm.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
