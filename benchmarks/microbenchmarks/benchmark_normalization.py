#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Normalization micro-benchmark using te.LayerNorm and te.RMSNorm.

Both LayerNorm and RMSNorm share the same kernel infrastructure.
The M dimension (batch * seq_len) is swept across typical training sizes.

Output: benchmark_normalization.csv (written to cwd)
"""

import torch
import transformer_engine.pytorch as te
from utils import (
    DTYPE_LIST, MODEL_HIDDEN_SIZES, M_SIZE_LIST,
    time_forward_backward, compute_gbps, make_forward_backward_metric_records, run_benchmarks,
)

NORM_TYPES = [
    ("RMSNorm",   te.RMSNorm),
    ("LayerNorm", te.LayerNorm),
]

BENCHMARK_LABEL = "Normalization"


def _generate_test_cases():
    test_cases = []
    for model_name, hidden in MODEL_HIDDEN_SIZES:
        for norm_name, norm_cls in NORM_TYPES:
            for M in M_SIZE_LIST:
                for dtype in DTYPE_LIST:
                    test_cases.append({
                        "Case": f"{model_name}/{norm_name}",
                        "M": M,
                        "hidden_size": hidden,
                        "norm_name": norm_name,
                        "norm_cls": norm_cls,
                        "dtype": dtype,
                    })
    return test_cases


def bench_norm(Case, M, hidden_size, norm_name, norm_cls, dtype):
    device = "cuda"

    norm = norm_cls(hidden_size).to(device=device, dtype=dtype)
    x = torch.randn(M, hidden_size, dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: norm(x)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def zero_grads():
        x.grad = None
        for p in norm.parameters():
            p.grad = None

    def fwd_bwd_func():
        out = norm(x)
        out.backward(grad_out)
        zero_grads()

    fwd_bwd_func()

    elem_bytes = x.element_size()
    fwd_bytes = 2 * M * hidden_size * elem_bytes   # read x, write y
    bwd_bytes = 4 * M * hidden_size * elem_bytes   # read grad+x+y, write grad_x

    fwd_ms, bwd_ms, record_kwargs = time_forward_backward(
        fwd_func, fwd_bwd_func, grad_out
    )

    fwd_gbps = compute_gbps(fwd_bytes, fwd_ms)
    bwd_gbps = compute_gbps(bwd_bytes, bwd_ms)

    return make_forward_backward_metric_records(
        BENCHMARK_LABEL,
        "GB/s",
        fwd_ms,
        fwd_gbps,
        bwd_ms,
        bwd_gbps,
        **record_kwargs,
    )


if __name__ == "__main__":
    run_benchmarks(
        test_cases=_generate_test_cases(),
        bench_fn=bench_norm,
        param_columns=["Case", "M", "hidden_size", "dtype"],
    )
