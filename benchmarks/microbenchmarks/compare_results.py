#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Compare two CSVs from the same benchmark suite.

Auto-detects metric columns (containing "TFLOPS" or "GB/s") and key columns.
Outputs a markdown <details> block to stdout with per-config results,
and optionally appends a summary table row to --summary-file.

Usage:
    python compare_results.py baseline.csv candidate.csv --bench-name NAME --summary-file FILE
"""

import argparse
import sys

import numpy as np
import pandas as pd

SKIP_COLS = {"TestID", "Label"}
DEFAULT_MIN_BASELINE_METRIC = 0.5


def auto_detect_columns(df):
    metric_cols = [c for c in df.columns if "TFLOPS" in c or "GB/s" in c]
    key_cols = [
        c for c in df.columns
        if c not in metric_cols and c not in SKIP_COLS
        and "Time" not in c
    ]
    return key_cols, metric_cols


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark CSVs")
    parser.add_argument("baseline_csv", help="Baseline CSV")
    parser.add_argument("candidate_csv", help="Candidate CSV")
    parser.add_argument("--bench-name", default="benchmark",
                        help="Benchmark name for markdown output")
    parser.add_argument("--summary-file", default=None,
                        help="Append a summary table row (markdown) to this file")
    parser.add_argument(
        "--min-baseline-metric",
        type=float,
        default=DEFAULT_MIN_BASELINE_METRIC,
        help=(
            "Small baseline metrics can produce noisy speedups; skip speedup "
            "calculations when the baseline metric is below this threshold. "
            "Set to 0 to disable the filter."
        ),
    )
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline_csv)
    candidate_df = pd.read_csv(args.candidate_csv)

    key_cols, metric_cols = auto_detect_columns(baseline_df)

    if not metric_cols:
        print("No metric columns found.")
        return 0

    for col in metric_cols:
        baseline_df[col] = pd.to_numeric(baseline_df[col], errors="coerce")
        candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")

    merged = baseline_df.merge(
        candidate_df,
        on=key_cols,
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        print("WARNING: No matching rows between baseline and candidate CSVs.")
        return 0

    all_speedups = []
    per_row_data = []

    for idx in merged.index:
        row_keys = {k: merged.loc[idx, k] for k in key_cols}
        row_metrics = {}

        for metric in metric_cols:
            baseline_col = f"{metric}_baseline"
            candidate_col = f"{metric}_candidate"
            baseline_value = merged.loc[idx, baseline_col]
            candidate_value = merged.loc[idx, candidate_col]

            if pd.isna(baseline_value) or pd.isna(candidate_value):
                continue
            if not np.isfinite(baseline_value) or not np.isfinite(candidate_value):
                continue
            if baseline_value <= 0:
                continue
            if (
                args.min_baseline_metric > 0
                and baseline_value < args.min_baseline_metric
            ):
                continue

            speedup = candidate_value / baseline_value
            all_speedups.append(speedup)
            row_metrics[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "speedup": speedup,
            }

        if row_metrics:
            per_row_data.append({"keys": row_keys, "metrics": row_metrics})

    if not all_speedups:
        print("WARNING: No valid comparisons found.")
        return 0

    speedups = np.array(all_speedups)
    median_sp = float(np.median(speedups))
    min_sp = float(np.min(speedups))
    max_sp = float(np.max(speedups))

    # Details block
    print("<details>")
    print(f"<summary><b>{args.bench_name}</b> "
          f"(median {median_sp:.3f}x, min {min_sp:.3f}x, max {max_sp:.3f}x)</summary>")
    print()

    header_cols = list(key_cols)
    for m in metric_cols:
        short = m.replace(" TFLOPS", "")
        header_cols.extend([
            f"{short} Baseline",
            f"{short} Candidate",
            f"{short} Speedup",
        ])

    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")

    for row in per_row_data:
        cells = [str(row["keys"].get(k, "")) for k in key_cols]
        for metric in metric_cols:
            if metric in row["metrics"]:
                v = row["metrics"][metric]
                cells.append(f"{v['baseline']:.2f}")
                cells.append(f"{v['candidate']:.2f}")
                cells.append(f"{v['speedup']:.3f}x")
            else:
                cells.extend(["", "", ""])
        print("| " + " | ".join(cells) + " |")

    print()
    print("</details>")
    print()

    # Summary row
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(f"| {args.bench_name} | {median_sp:.3f}x | {min_sp:.3f}x | {max_sp:.3f}x |\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
