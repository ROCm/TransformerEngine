#!/usr/bin/env python
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Shared result emit for the distributed comm+overlap benchmarks.

Turns a list of per-iteration GPU times (ms) into the same summary statistics the
single-GPU microbenchmarks report (median/mean/stdev/min/max plus p10/p90), and
appends one row to a CSV.  Optionally also writes a per-sample CSV mirroring the
microbenchmarks' ``--csv-samples`` output (one row per timing iteration).
"""

from __future__ import annotations

import csv
import os

import numpy as np

SUMMARY_FIELDS = [
    "benchmark", "variant", "M", "N", "K", "tp", "dtype", "n",
    "median_ms", "mean_ms", "stdev_ms", "min_ms", "max_ms", "p10_ms", "p90_ms",
    "tflops", "gbps", "speedup",  # FIXME: Note that gbps and speedup are not implemented currently
]

SAMPLE_FIELDS = ["benchmark", "variant", "sample_idx", "time_ms"]


def summarize(times_ms) -> dict:
    """Summary statistics (ms) of a per-iteration timing list."""
    s = np.asarray(list(times_ms), dtype=np.float64)
    return {
        "n": int(s.size),
        "median_ms": float(np.median(s)),
        "mean_ms": float(s.mean()),
        "stdev_ms": float(s.std()),
        "min_ms": float(s.min()),
        "max_ms": float(s.max()),
        "p10_ms": float(np.quantile(s, 0.10)),
        "p90_ms": float(np.quantile(s, 0.90)),
    }


def _append_row(path: str, fields, row) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def emit_summary(path, *, benchmark, variant, params, times_ms,
                 flops=None, nbytes=None, baseline_median_ms=None) -> dict:
    """Append a summary row; returns the computed stats.

    Throughput is derived from the *median* time when *flops* / *nbytes* are
    given; *speedup* is ``baseline_median_ms / median`` when a baseline is given.
    """
    stats = summarize(times_ms)
    median_s = stats["median_ms"] * 1e-3
    row = {
        "benchmark": benchmark,
        "variant": variant,
        "M": params.get("M", ""),
        "N": params.get("N", ""),
        "K": params.get("K", ""),
        "tp": params.get("tp", ""),
        "dtype": params.get("dtype", ""),
        "n": stats["n"],
        "median_ms": f"{stats['median_ms']:.6f}",
        "mean_ms": f"{stats['mean_ms']:.6f}",
        "stdev_ms": f"{stats['stdev_ms']:.6f}",
        "min_ms": f"{stats['min_ms']:.6f}",
        "max_ms": f"{stats['max_ms']:.6f}",
        "p10_ms": f"{stats['p10_ms']:.6f}",
        "p90_ms": f"{stats['p90_ms']:.6f}",
        "tflops": f"{flops / median_s / 1e12:.4f}" if flops else "",
        "gbps": f"{nbytes / median_s / 1e9:.4f}" if nbytes else "",
        "speedup": f"{baseline_median_ms / stats['median_ms']:.4f}" if baseline_median_ms else "",
    }
    _append_row(path, SUMMARY_FIELDS, row)
    return stats


def emit_samples(path, *, benchmark, variant, times_ms) -> None:
    """Append one row per timing iteration (mirrors ``--csv-samples``)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        if not exists:
            writer.writeheader()
        for i, t in enumerate(times_ms):
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "sample_idx": i,
                    "time_ms": f"{float(t):.6f}",
                }
            )
