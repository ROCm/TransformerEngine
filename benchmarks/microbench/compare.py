#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compare two microbench CSVs produced by driver.py.

Joins on (suite, class, method, params). For each group, reports baseline and
candidate medians plus a speedup; >1 means candidate is faster (for time) or
higher-throughput (for tflops/gbps).

Usage:
    python compare.py baseline.csv candidate.csv
    python compare.py baseline.csv candidate.csv --metric tflops
    python compare.py baseline.csv candidate.csv --sort speedup --top 20
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict

KEY_COLS = ("suite", "class", "method", "params")


def load(path, metric):
    """Group rows by KEY_COLS, returning {key: [float, ...]} of the metric values.

    Empty cells are skipped (e.g. tflops/gbps may be absent when no work_* is
    defined). The metric is always coerced to float; non-numeric rows are
    skipped silently.
    """
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            val = row.get(metric, "")
            if val == "" or val is None:
                continue
            try:
                groups[tuple(row[k] for k in KEY_COLS)].append(float(val))
            except (ValueError, KeyError):
                continue
    return groups


def summarize(samples):
    """Return (median, mean, n) for a list of samples, or None if empty."""
    if not samples:
        return None
    return statistics.median(samples), statistics.fmean(samples), len(samples)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline_csv")
    p.add_argument("candidate_csv")
    p.add_argument("--metric", default="time_s", choices=["time_s", "tflops", "gbps"],
                   help="column to compare (default: time_s)")
    p.add_argument("--sort", default="suite", choices=["suite", "speedup", "abs_change"],
                   help="row order (default: suite)")
    p.add_argument("--top", type=int, default=None,
                   help="show only the top N rows after sorting")
    p.add_argument("--min-samples", type=int, default=1,
                   help="skip groups with fewer than this many samples in either CSV")
    args = p.parse_args()

    base = load(args.baseline_csv, args.metric)
    cand = load(args.candidate_csv, args.metric)

    # For time: lower is better, so speedup = base / cand.
    # For tflops/gbps: higher is better, so speedup = cand / base.
    lower_is_better = (args.metric == "time_s")

    rows = []
    for key in sorted(base.keys() | cand.keys()):
        b = summarize(base.get(key, []))
        c = summarize(cand.get(key, []))
        if b is None or c is None:
            rows.append({"key": key, "status": "baseline-only" if c is None else "candidate-only",
                         "b": b, "c": c, "speedup": None})
            continue
        if b[2] < args.min_samples or c[2] < args.min_samples:
            continue
        b_med, c_med = b[0], c[0]
        if b_med <= 0 or c_med <= 0:
            speedup = None
        else:
            speedup = (b_med / c_med) if lower_is_better else (c_med / b_med)
        rows.append({"key": key, "status": "matched", "b": b, "c": c, "speedup": speedup})

    matched = [r for r in rows if r["status"] == "matched" and r["speedup"] is not None]
    only_b = [r for r in rows if r["status"] == "baseline-only"]
    only_c = [r for r in rows if r["status"] == "candidate-only"]

    if args.sort == "speedup":
        matched.sort(key=lambda r: r["speedup"])
    elif args.sort == "abs_change":
        matched.sort(key=lambda r: -abs(r["speedup"] - 1.0))
    if args.top is not None:
        matched = matched[:args.top]

    unit = "ms" if args.metric == "time_s" else args.metric
    scale = 1e3 if args.metric == "time_s" else 1.0
    print(f"{'suite':<22} {'class':<22} {'method':<22} {'params':<48} "
          f"{'base ' + unit:>12} {'cand ' + unit:>12} {'speedup':>9}  n_b/n_c")
    print("-" * 160)
    for r in matched:
        s, cls, m, params = r["key"]
        b_med, c_med = r["b"][0] * scale, r["c"][0] * scale
        n_b, n_c = r["b"][2], r["c"][2]
        print(f"{s:<22} {cls:<22} {m:<22} {params:<48} "
              f"{b_med:>12.4f} {c_med:>12.4f} {r['speedup']:>8.3f}x  {n_b}/{n_c}")

    if matched:
        speedups = [r["speedup"] for r in matched]
        print()
        print(f"{len(matched)} matched groups: "
              f"median {statistics.median(speedups):.3f}x  "
              f"min {min(speedups):.3f}x  "
              f"max {max(speedups):.3f}x")

    for label, rows_list in [("baseline only", only_b), ("candidate only", only_c)]:
        if not rows_list:
            continue
        print(f"\n{len(rows_list)} groups {label}:")
        for r in rows_list:
            print("  " + " | ".join(r["key"]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
