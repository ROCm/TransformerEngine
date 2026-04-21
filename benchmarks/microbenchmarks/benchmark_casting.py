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
  - Llama 3.1 8B, 70B, 405B
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
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8Quantizer


TE_FP8_E4M3 = tex.DType.kFloat8E4M3
TE_FP8_E5M2 = tex.DType.kFloat8E5M2

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

CAST_CONFIGS = [
    # (name, direction, fp8_dtype)
    ("BF16-to-FP8-E4M3", "quantize",   TE_FP8_E4M3),
    ("FP8-E4M3-to-BF16", "dequantize", TE_FP8_E4M3),
    ("BF16-to-FP8-E5M2", "quantize",   TE_FP8_E5M2),
    ("FP8-E5M2-to-BF16", "dequantize", TE_FP8_E5M2),
]


def _generate_cast_test_cases():
    test_cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for cast_name, direction, fp8_dtype in CAST_CONFIGS:
            for M in M_SIZE_LIST:
                test_cases.append({
                    "Case": f"{model_name}/{cast_name}",
                    "M": M,
                    "hidden_size": hidden,
                    "direction": direction,
                    "fp8_dtype": fp8_dtype,
                    "dtype_str": cast_name,
                })
    return test_cases


def bench_cast(M, hidden_size, direction, fp8_dtype):
    device = "cuda"

    numel = M * hidden_size
    scale = torch.ones(1, dtype=torch.float32, device=device)
    amax = torch.zeros(1, dtype=torch.float32, device=device)
    quantizer = Float8Quantizer(scale, amax, fp8_dtype)

    if direction == "quantize":
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
        out = quantizer(x)  # pre-allocate output tensor
        cast_func = lambda: quantizer.quantize(x, out=out)

        # BF16 read (2 bytes) + FP8 write (1 byte)
        total_bytes = numel * (2 + 1)
    else:
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
        fp8_tensor = quantizer(x)
        cast_func = lambda: fp8_tensor.dequantize()

        # FP8 read (1 byte) + BF16 write (2 bytes)
        total_bytes = numel * (1 + 2)

    # Benchmark
    ms = benchmark.Timer(stmt="fn()", globals={"fn": cast_func}).blocked_autorange().mean * 1e3
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

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['Case']} M={case['M']} hidden={case['hidden_size']}")
        print(f"{'='*60}")

        metrics = bench_cast(
            M=case["M"],
            hidden_size=case["hidden_size"],
            direction=case["direction"],
            fp8_dtype=case["fp8_dtype"],
        )
        row = {
            "Case": case["Case"],
            "M": case["M"],
            "hidden_size": case["hidden_size"],
            "dtype_str": case["dtype_str"],
            **metrics,
        }
        rows.append(row)

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_casting.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
