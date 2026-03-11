#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Compare two CSVs from the same benchmark (base branch vs PR branch).

Auto-detects metric columns (containing "TFLOPS") and key columns.
Outputs a markdown <details> block to stdout with per-config results,
and optionally appends a summary table row to --summary-file.

Usage:
    python compare_results.py base.csv pr.csv --bench-name NAME --summary-file FILE
"""

import argparse
import sys

import numpy as np
import pandas as pd

SKIP_COLS = {"TestID", "Label"}


def auto_detect_columns(df):
    metric_cols = [c for c in df.columns if "TFLOPS" in c]
    key_cols = [
        c for c in df.columns
        if c not in metric_cols and c not in SKIP_COLS
        and "Time" not in c
    ]
    return key_cols, metric_cols


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark CSVs")
    parser.add_argument("base_csv", help="Base branch CSV")
    parser.add_argument("pr_csv", help="PR branch CSV")
    parser.add_argument("--bench-name", default="benchmark",
                        help="Benchmark name for markdown output")
    parser.add_argument("--summary-file", default=None,
                        help="Append a summary table row (markdown) to this file")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_csv)
    pr_df = pd.read_csv(args.pr_csv)

    key_cols, metric_cols = auto_detect_columns(base_df)

    if not metric_cols:
        print("No metric columns found.")
        return 0

    for col in metric_cols:
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        pr_df[col] = pd.to_numeric(pr_df[col], errors="coerce")

    merged = base_df.merge(pr_df, on=key_cols, suffixes=("_base", "_pr"), how="inner")
    if merged.empty:
        print("WARNING: No matching rows between base and PR.")
        return 0

    all_speedups = []
    per_row_data = []

    for idx in merged.index:
        row_keys = {k: merged.loc[idx, k] for k in key_cols}
        row_metrics = {}

        for metric in metric_cols:
            bc, pc = f"{metric}_base", f"{metric}_pr"
            bv = merged.loc[idx, bc]
            pv = merged.loc[idx, pc]

            if pd.isna(bv) or pd.isna(pv) or bv < 0.5:
                continue

            speedup = pv / bv
            all_speedups.append(speedup)
            row_metrics[metric] = {"base": bv, "pr": pv, "speedup": speedup}

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
        header_cols.extend([f"{short} Base", f"{short} PR", f"{short} Speedup"])

    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")

    for row in per_row_data:
        cells = [str(row["keys"].get(k, "")) for k in key_cols]
        for metric in metric_cols:
            if metric in row["metrics"]:
                v = row["metrics"][metric]
                cells.append(f"{v['base']:.2f}")
                cells.append(f"{v['pr']:.2f}")
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
