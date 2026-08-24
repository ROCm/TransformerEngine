#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Normalization micro-benchmark using the fusible-ops LayerNorm / RMSNorm.

Sweeps BF16 plus the quantized-output precisions (FP8, MXFP8) that TE training
recipes produce. In FP8/FP4 training the norm is fused with the following
Linear (LayerNormLinear / LayerNormMLP) and writes its output already
quantized. That is reproduced here with ``ops.Sequential(Norm, Quantize)``: the
op fuser threads the Quantize op's input quantizer into the norm, so the norm
writes the quantized output directly under ``autocast``. The Quantize op is an
identity outside ``autocast``, so the bf16 baseline uses the same harness.

Forward only: the quantize epilogue is a forward-pass phenomenon; the norm
backward reads/writes high precision and is precision-independent. NVFP4/MXFP4
norm outputs are not swept (no validated norm->FP4 cast path). Precisions
unsupported on the current device are skipped.

These are memory-bound; we report GB/s (input read + output write).
Output: benchmark_normalization.csv (written to cwd)
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops
from utils import (
    MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    build_recipes,
    time_func, compute_gbps, make_metric_record, run_benchmarks,
    make_input,
)

NORM_TYPES = [
    ("RMSNorm",   ops.RMSNorm),
    ("LayerNorm", ops.LayerNorm),
]

BENCHMARK_LABEL = "Normalization Forward"

# Quantized-output precisions validated for the norm cast; tests/pytorch/
# triton_kernels/test_norms.py exercises fp8 and mxfp8 norm quantizers. bf16 is
# the plain baseline (Quantize is an identity outside autocast).
RECIPES = build_recipes(names=("bf16", "fp8", "mxfp8"))

# Forward output bytes/elem by precision (input is always bf16 = 2 bytes/elem):
#   bf16  : 2.0
#   fp8   : 1.0             (E4M3/E5M2 data, per-tensor scale ~ 0)
#   mxfp8 : 1.0 + 1/32      (+ E8M0 1 byte / 32-elem block)
_FWD_WRITE_BYTES = {
    "bf16": 2.0,
    "fp8": 1.0,
    "mxfp8": 1.0 + 1.0 / 32,
}


def _generate_test_cases():
    test_cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for norm_name, norm_op_cls in NORM_TYPES:
            for precision in RECIPES:
                for M in M_SIZE_LIST:
                    test_cases.append({
                        "Case": f"{model_name}/{norm_name}",
                        "Precision": precision,
                        "M": M,
                        "hidden_size": hidden,
                        "norm_op_cls": norm_op_cls,
                        "dtype": torch.bfloat16,
                    })
    return test_cases


def bench_norm(Case, Precision, M, hidden_size, norm_op_cls, dtype):
    device = "cuda"

    recipe = RECIPES[Precision]
    use_fp8 = recipe is not None

    # Norm followed by Quantize so the norm writes its output directly in the
    # target precision under autocast (identity when use_fp8 is False).
    model = ops.Sequential(
        norm_op_cls(hidden_size, device=device, dtype=dtype),
        ops.Quantize(),
    )
    next_x = make_input((M, hidden_size), dtype, device=device, requires_grad=False)

    def fwd_func():
        with te.autocast(enabled=use_fp8, recipe=recipe):
            return model(next_x())

    fwd_func()

    # BF16 read + quantized write.
    fwd_bytes = int(M * hidden_size * (2 + _FWD_WRITE_BYTES[Precision]))

    fwd_ms, fwd_measurement = time_func(fwd_func)
    fwd_gbps = compute_gbps(fwd_bytes, fwd_ms)

    return [make_metric_record(
        BENCHMARK_LABEL, fwd_ms, "GB/s", fwd_gbps, measurement=fwd_measurement,
    )]


if __name__ == "__main__":
    run_benchmarks(
        test_cases=_generate_test_cases(),
        bench_fn=bench_norm,
        param_columns=["Case", "Precision", "M", "hidden_size", "dtype"],
    )
