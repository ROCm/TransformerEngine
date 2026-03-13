#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Normalization micro-benchmark using te.LayerNorm and te.RMSNorm.

Shapes are derived from training workloads:
  - Llama 3   8B, 70B, 405B (all use RMSNorm)
  - Qwen 2.5  7B, 72B       (all use RMSNorm)

Modern models predominantly use RMSNorm, but we benchmark both
LayerNorm and RMSNorm since TE supports both and they share the
same kernel infrastructure.

The M dimension (batch * seq_len) is swept across typical training sizes.

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

Output: benchmark_normalization.csv (written to cwd)
"""

import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te

# Sequence / batch-token sizes to sweep
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# (model_name, hidden_size)
MODEL_HIDDEN_SIZES = [
    ("Llama3-8B",   4096),
    ("Llama3-70B",  8192),
    ("Llama3-405B", 16384),
    ("Qwen2.5-7B",  3584),
    ("Qwen2.5-72B", 8192),
]

NORM_TYPES = [
    ("RMSNorm",   te.RMSNorm),
    ("LayerNorm", te.LayerNorm),
]


def _generate_norm_test_cases():
    test_cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for norm_name, norm_cls in NORM_TYPES:
            for M in M_SIZE_LIST:
                test_cases.append({
                    "Case": f"{model_name}/{norm_name}",
                    "M": M,
                    "hidden_size": hidden,
                    "norm_name": norm_name,
                    "norm_cls": norm_cls,
                    "dtype": torch.bfloat16,
                })
    return test_cases


def bench_norm(M, hidden_size, norm_cls, dtype):
    device = "cuda"

    norm = norm_cls(hidden_size).to(device=device, dtype=dtype)
    x = torch.randn(M, hidden_size, dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: norm(x)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        out = norm(x)
        out.backward(grad_out)
        x.grad = None
        for p in norm.parameters():
            p.grad = None

    fwd_bwd_func()

    # Normalization is memory-bound; report bandwidth instead of FLOPS.
    # Each element is read once (fwd) or read+written (bwd), plus the
    # weight/bias vectors. We report effective GB/s based on the
    # minimum data movement: fwd reads x and writes y, bwd reads
    # grad_out+x+saved_stats and writes grad_x+grad_weight.
    elem_bytes = x.element_size()
    fwd_bytes = 2 * M * hidden_size * elem_bytes   # read x, write y
    bwd_bytes = 4 * M * hidden_size * elem_bytes   # read grad+x+y, write grad_x

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

    fwd_gbps = fwd_bytes / (fwd_ms * 1e-3) / 1e9
    bwd_gbps = bwd_bytes / (bwd_ms * 1e-3) / 1e9 if bwd_ms > 0 else 0.0

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_gbps:.1f} GB/s")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_gbps:.1f} GB/s (derived)")

    return {
        "TE Forward Time (ms)": f"{fwd_ms:.4f}",
        "TE Forward GB/s": f"{fwd_gbps:.1f}",
        "TE Backward Time (ms)": f"{bwd_ms:.4f}",
        "TE Backward GB/s": f"{bwd_gbps:.1f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = _generate_norm_test_cases()

    columns = [
        "Case", "M", "hidden_size", "dtype",
        "TE Forward Time (ms)",
        "TE Forward GB/s",
        "TE Backward Time (ms)",
        "TE Backward GB/s",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*60}")
    print(f"WARMUP: {c['Case']} M={c['M']} hidden={c['hidden_size']}")
    print(f"{'='*60}")
    bench_norm(M=c["M"], hidden_size=c["hidden_size"],
               norm_cls=c["norm_cls"], dtype=c["dtype"])

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['Case']} M={case['M']} hidden={case['hidden_size']}")
        print(f"{'='*60}")
        try:
            metrics = bench_norm(
                M=case["M"],
                hidden_size=case["hidden_size"],
                norm_cls=case["norm_cls"],
                dtype=case["dtype"],
            )
            row = {
                "Case": case["Case"],
                "M": case["M"],
                "hidden_size": case["hidden_size"],
                "dtype": str(case["dtype"]),
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"FAILED: {case['Case']}: {e}")
            raise

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_normalization.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
