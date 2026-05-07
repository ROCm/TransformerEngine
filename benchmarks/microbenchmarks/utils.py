#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared utilities for microbenchmarks: model configs, timing, throughput, runner."""

import argparse
import torch
import torch.utils.benchmark as benchmark

# ---------------------------------------------------------------------------
# Sequence / batch-token sizes
# ---------------------------------------------------------------------------
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# (name, hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
#
# Sources:
# - Llama 3 8B   https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
# - Llama 3 70B  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
# - Llama 3 405B https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
# - Qwen 2.5 7B  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
# - Qwen 2.5 72B https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

MODEL_CONFIGS = [
    ("Llama3-8B/TP1",   4096,  14336,  32,  8, 128,  1),
    ("Llama3-8B/TP8",   4096,  14336,  32,  8, 128,  8),
    ("Llama3-70B/TP8",  8192,  28672,  64,  8, 128,  8),
    ("Llama3-405B/TP8", 16384, 53248, 128,  8, 128,  8),
    ("Qwen2.5-7B/TP1",  3584, 18944,  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 8192, 29568,  64,  8, 128,  8),
]

# Unique (model_name, hidden_size) pairs for element-wise benchmarks
MODEL_HIDDEN_SIZES = [
    ("Llama3-8B",   4096),
    ("Llama3-70B",  8192),
    ("Llama3-405B", 16384),
    ("Qwen2.5-7B",  3584),
    ("Qwen2.5-72B", 8192),
]


def gemm_shapes(configs=None):
    """Generate {case_name: (N, K)} dict from MODEL_CONFIGS.

    Each model contributes up to four GEMM shapes:
      QKV, AttnOut, GateUp (SwiGLU), Down.
    """
    shapes = {}
    for (name, hidden, intermediate, n_q, n_kv, hd, tp) in (configs or MODEL_CONFIGS):
        shapes[f"{name}-QKV"]     = ((n_q * hd + 2 * n_kv * hd) // tp, hidden)
        shapes[f"{name}-AttnOut"] = (hidden, (n_q * hd) // tp)
        shapes[f"{name}-GateUp"]  = ((2 * intermediate) // tp, hidden)
        shapes[f"{name}-Down"]    = (hidden, intermediate // tp)
    return shapes


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_func(fn, method="adaptive"):
    """Time *fn* and return elapsed milliseconds.

    method: "adaptive" uses adaptive_autorange (good for compute-bound),
            "blocked"  uses blocked_autorange  (good for memory-bound).
    """
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    if method == "blocked":
        return timer.blocked_autorange().mean * 1e3
    return timer.adaptive_autorange().mean * 1e3


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------

def compute_tflops(flops, ms):
    """TFLOPS from operation count and milliseconds."""
    return flops / (ms * 1e-3) / 1e12


def compute_gbps(nbytes, ms):
    """GB/s from byte count and milliseconds."""
    return nbytes / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def add_csv_arg(parser):
    """Add a ``--csv`` flag to an argparse parser."""
    parser.add_argument(
        "--csv", nargs="?", const=True, default=None, metavar="FILE",
        help="Write results to CSV. Optional filename; default derived from script name.",
    )


def run_benchmarks(test_cases, bench_fn, param_columns, metric_columns,
                   default_csv=None):
    """Iterate *test_cases*, call *bench_fn*, and optionally write a CSV.

    Parameters
    ----------
    test_cases : list[dict]
        Each dict has at least the keys in *param_columns* plus any extra
        keys the bench_fn needs (passed as **case).
    bench_fn : callable
        Called as ``bench_fn(**case)`` and must return a dict whose keys
        match *metric_columns*.
    param_columns : list[str]
        Column names to pull from each test case into the output row.
    metric_columns : list[str]
        Column names to pull from the bench_fn return value.
    default_csv : str or None
        Default CSV filename used when ``--csv`` is passed without a
        filename.  CSV output is only written when the caller passes
        ``--csv`` on the command line.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_csv_arg(parser)
    args, _ = parser.parse_known_args()

    columns = param_columns + metric_columns
    rows = []

    for case in test_cases:
        label = "  ".join(f"{k}={case[k]}" for k in param_columns)
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        metrics = bench_fn(**case)

        row = {k: (str(case[k]) if isinstance(case[k], torch.dtype) else case[k])
               for k in param_columns}
        row.update(metrics)
        rows.append(row)

    if args.csv is not None:
        import pandas as pd
        out_csv = args.csv if isinstance(args.csv, str) else default_csv
        results = pd.DataFrame(rows, columns=columns)
        results.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")
