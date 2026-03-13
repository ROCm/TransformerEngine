#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
FP8 casting micro-benchmark.

Benchmarks quantization (BF16 -> FP8) and dequantization (FP8 -> BF16) for
both E4M3 (activations/weights) and E5M2 (gradients) formats.

Shapes are (M, hidden_size) matching the activation tensors from models:
  - Llama 3   8B, 70B, 405B
  - Qwen 2.5  7B, 72B

These casts are memory-bound; we report GB/s (input + output bytes).

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

Output: benchmark_casting.csv (written to cwd)
"""

import torch
import torch.utils.benchmark as benchmark


# Detect FP8 dtypes (ROCm vs CUDA)
if hasattr(torch, "float8_e4m3fnuz"):
    FP8_E4M3 = torch.float8_e4m3fnuz
    FP8_E5M2 = torch.float8_e5m2fnuz
else:
    FP8_E4M3 = torch.float8_e4m3fn
    FP8_E5M2 = torch.float8_e5m2

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

# (cast_name, src_dtype, dst_dtype)
CAST_CONFIGS = [
    ("BF16-to-FP8-E4M3", torch.bfloat16, FP8_E4M3),
    ("FP8-E4M3-to-BF16", FP8_E4M3,       torch.bfloat16),
    ("BF16-to-FP8-E5M2", torch.bfloat16, FP8_E5M2),
    ("FP8-E5M2-to-BF16", FP8_E5M2,       torch.bfloat16),
]


def _generate_cast_test_cases():
    test_cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for cast_name, src_dtype, dst_dtype in CAST_CONFIGS:
            for M in M_SIZE_LIST:
                test_cases.append({
                    "Case": f"{model_name}/{cast_name}",
                    "M": M,
                    "hidden_size": hidden,
                    "src_dtype": src_dtype,
                    "dst_dtype": dst_dtype,
                    "dtype_str": cast_name,
                })
    return test_cases


def bench_cast(M, hidden_size, src_dtype, dst_dtype):
    device = "cuda"

    # For FP8 source, create via cast from randn
    if src_dtype in (FP8_E4M3, FP8_E5M2):
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device).to(src_dtype)
    else:
        x = torch.randn(M, hidden_size, dtype=src_dtype, device=device)

    cast_func = lambda: x.to(dst_dtype)

    # Sanity check
    cast_func()

    # Total bytes moved: read input + write output
    numel = M * hidden_size
    src_bytes = numel * x.element_size()
    dst_bytes = numel * cast_func().element_size()
    total_bytes = src_bytes + dst_bytes

    # Warmup
    for _ in range(20):
        cast_func()
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100
    ms = benchmark.Timer(stmt="fn()", globals={"fn": cast_func}).timeit(n_iters).mean * 1e3
    gbps = total_bytes / (ms * 1e-3) / 1e9

    print(f"  {ms:.4f} ms | {gbps:.1f} GB/s")

    return {
        "Cast Time (ms)": f"{ms:.4f}",
        "Cast GB/s": f"{gbps:.1f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = _generate_cast_test_cases()

    columns = [
        "Case", "M", "hidden_size", "dtype_str",
        "Cast Time (ms)",
        "Cast GB/s",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*60}")
    print(f"WARMUP: {c['Case']} M={c['M']} hidden={c['hidden_size']}")
    print(f"{'='*60}")
    bench_cast(M=c["M"], hidden_size=c["hidden_size"],
               src_dtype=c["src_dtype"], dst_dtype=c["dst_dtype"])

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['Case']} M={case['M']} hidden={case['hidden_size']}")
        print(f"{'='*60}")
        try:
            metrics = bench_cast(
                M=case["M"],
                hidden_size=case["hidden_size"],
                src_dtype=case["src_dtype"],
                dst_dtype=case["dst_dtype"],
            )
            row = {
                "Case": case["Case"],
                "M": case["M"],
                "hidden_size": case["hidden_size"],
                "dtype_str": case["dtype_str"],
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"FAILED: {case['Case']}: {e}")
            raise

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_casting.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
