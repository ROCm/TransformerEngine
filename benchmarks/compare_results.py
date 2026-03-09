#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import sys

import pandas as pd

SKIP_COLS = {"TestID", "Label"}


def auto_detect_columns(df: pd.DataFrame):
    """Split columns into (key_cols, metric_cols) by naming convention."""
    metric_cols = [c for c in df.columns if "TFLOPS" in c]
    key_cols = [
        c for c in df.columns
        if c not in metric_cols and c not in SKIP_COLS
        and "Time" not in c  # skip timing columns, only compare TFLOPS
    ]
    return key_cols, metric_cols


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark CSVs")
    parser.add_argument("base_csv", help="Base branch CSV")
    parser.add_argument("pr_csv", help="PR branch CSV")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold %% (default: 5.0)")
    parser.add_argument("--key-cols", default=None,
                        help="Comma-separated key columns (auto-detected if omitted)")
    parser.add_argument("--metric-cols", default=None,
                        help="Comma-separated metric columns (auto-detected if omitted)")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_csv)
    pr_df = pd.read_csv(args.pr_csv)

    # Determine columns
    if args.key_cols:
        key_cols = [c.strip() for c in args.key_cols.split(",")]
    else:
        key_cols, _ = auto_detect_columns(base_df)

    if args.metric_cols:
        metric_cols = [c.strip() for c in args.metric_cols.split(",")]
    else:
        _, metric_cols = auto_detect_columns(base_df)

    if not metric_cols:
        print("No metric columns found — nothing to compare.")
        return 0

    print(f"Key columns:    {key_cols}")
    print(f"Metric columns: {metric_cols}")
    print(f"Threshold:      {args.threshold}%")
    print(f"Base rows: {len(base_df)}, PR rows: {len(pr_df)}")
    print()

    # Ensure metric columns are numeric
    for col in metric_cols:
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        pr_df[col] = pd.to_numeric(pr_df[col], errors="coerce")

    # Match rows
    merged = base_df.merge(pr_df, on=key_cols, suffixes=("_base", "_pr"), how="inner")
    if merged.empty:
        print("WARNING: No matching rows between base and PR.")
        return 0

    print(f"Matched rows: {len(merged)}")
    print()

    # Compare
    regressions = []
    for metric in metric_cols:
        bc = f"{metric}_base"
        pc = f"{metric}_pr"
        if bc not in merged.columns or pc not in merged.columns:
            continue

        bv = merged[bc]
        pv = merged[pc]
        delta_pct = ((pv - bv) / bv) * 100.0

        for idx in merged.index:
            if pd.isna(bv[idx]) or pd.isna(pv[idx]) or bv[idx] < 0.5:
                continue
            if delta_pct[idx] < -args.threshold:
                key_info = " | ".join(f"{k}={merged.loc[idx, k]}" for k in key_cols)
                regressions.append({
                    "keys": key_info,
                    "metric": metric,
                    "base": bv[idx],
                    "pr": pv[idx],
                    "delta": delta_pct[idx],
                })

    # Print summary per metric
    for metric in metric_cols:
        bc = f"{metric}_base"
        pc = f"{metric}_pr"
        if bc not in merged.columns:
            continue
        bv = merged[bc].dropna()
        pv = merged[pc].dropna()
        if bv.empty:
            continue
        deltas = ((pv - bv) / bv) * 100.0
        print(f"  {metric}:")
        print(f"    mean base={bv.mean():.2f}  pr={pv.mean():.2f}  delta={deltas.mean():+.2f}%")
        print(f"    min delta={deltas.min():+.2f}%  max delta={deltas.max():+.2f}%")
    print()

    if regressions:
        print(f"REGRESSIONS DETECTED: {len(regressions)}")
        print("-" * 80)
        for r in regressions:
            print(f"  [{r['metric']}] {r['keys']}")
            print(f"    base={r['base']:.2f}  pr={r['pr']:.2f}  delta={r['delta']:+.2f}%")
        print("-" * 80)
        return 1
    else:
        print("No regressions detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
