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

These casts are memory-bound; we report GB/s (input + output bytes).
Output: benchmark_casting.csv (written to cwd)
"""

import torch
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8Quantizer
from utils import (
    MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    time_func, compute_gbps, make_metric_record, run_benchmarks,
)

TE_FP8_E4M3 = tex.DType.kFloat8E4M3
TE_FP8_E5M2 = tex.DType.kFloat8E5M2

CAST_LABEL = "Cast"

CAST_CONFIGS = [
    # (name, direction, fp8_dtype)
    ("BF16-to-FP8-E4M3", "quantize",   TE_FP8_E4M3),
    ("FP8-E4M3-to-BF16", "dequantize", TE_FP8_E4M3),
    ("BF16-to-FP8-E5M2", "quantize",   TE_FP8_E5M2),
    ("FP8-E5M2-to-BF16", "dequantize", TE_FP8_E5M2),
]


def _generate_test_cases():
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


def bench_cast(Case, M, hidden_size, direction, fp8_dtype, dtype_str):
    device = "cuda"

    numel = M * hidden_size
    scale = torch.ones(1, dtype=torch.float32, device=device)
    amax = torch.zeros(1, dtype=torch.float32, device=device)
    quantizer = Float8Quantizer(scale, amax, fp8_dtype)

    if direction == "quantize":
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
        out = quantizer(x)
        cast_func = lambda: quantizer.quantize(x, out=out)
        total_bytes = numel * (2 + 1)  # BF16 read + FP8 write
    else:
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
        fp8_tensor = quantizer(x)
        cast_func = lambda: fp8_tensor.dequantize()
        total_bytes = numel * (1 + 2)  # FP8 read + BF16 write

    ms, measurement = time_func(cast_func, method="blocked")
    gbps = compute_gbps(total_bytes, ms)

    return [make_metric_record(CAST_LABEL, ms, "GB/s", gbps, measurement=measurement)]


if __name__ == "__main__":
    run_benchmarks(
        test_cases=_generate_test_cases(),
        bench_fn=bench_cast,
        param_columns=["Case", "M", "hidden_size", "dtype_str"],
    )
