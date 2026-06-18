#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Statistically compare two result JSON files written by ``driver.py``.

A point-estimate (median) cannot tell a real regression from measurement noise.
This tool compares the raw per-call samples stored in two result files (one per
checkout) with a statistical test (Brunner-Munzel by default) via the benchstats
package. It marks each (benchmark, parameter combination) as faster (``>``),
slower (``<``), or not significantly different (``~``), prints a per-direction
summary, and exits ``1`` when a significant timing difference is found so it can
gate CI. Requires ``pip install -r requirements.txt``.

Usage:
    # run the suite on each checkout (each saves <hash>.json), then:
    python compare_results.py results/<base>.json results/<cand>.json
    python compare_results.py base.json cand.json --alpha 0.01
    python compare_results.py base.json cand.json --export-to report.svg
"""

import argparse
import json
import os
import re
import sys

import numpy as np

_TIME_KEY = "time_s"  # metric exposed to benchstats (seconds, lower is better)


def _load_samples(path, name_filter=None):
    """Load a driver result JSON into ``{bench_name: {"time_s": ndarray}}``.

    One benchstats "benchmark" per (benchmark, parameter combination); the name
    is ``<suite>.<Class>.<method> | name=val, ...``. Only timing is exposed:
    throughput is a constant-work transform of time, so a rank test on it is
    identical.
    """
    with open(path) as f:
        data = json.load(f)
    pattern = re.compile(name_filter) if name_filter else None

    stats = {}
    for bench_key, rec in data.get("results", {}).items():
        param_names = rec.get("param_names") or []
        for combo, samples in zip(rec.get("combos") or [], rec.get("samples") or []):
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            if param_names and len(param_names) == len(combo):
                label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            else:
                label = ", ".join(str(v) for v in combo)
            name = bench_key + (" | " + label if label else "")
            if pattern is not None and pattern.search(name) is None:
                continue
            stats[name] = {_TIME_KEY: arr}
    return stats


def run_stats(args):
    """Compare two result JSONs; return 1 if a significant difference is found."""
    import rich.table  # noqa: F401  benchstats render uses rich.table without importing it
    from benchstats.compare import compareStats
    from benchstats.render import renderComparisonResults
    from benchstats.common import LoggingConsole, detectExportFormat

    main_metrics = [_TIME_KEY]
    export_fmt = detectExportFormat(args.export_to, None) if args.export_to else None
    if export_fmt is not None and os.path.isfile(args.export_to):
        os.remove(args.export_to)

    console = LoggingConsole(
        record=export_fmt is not None, log_level=LoggingConsole.LogLevel.Warning,
    )

    s1 = _load_samples(args.baseline_json, args.filter)
    s2 = _load_samples(args.candidate_json, args.filter)

    cr = compareStats(
        s1, s2, method=args.method, alpha=args.alpha,
        main_metrics=main_metrics, debug_log=console,
    )
    renderComparisonResults(
        cr, console, main_metrics=main_metrics,
        always_show_pvalues=args.always_show_pvalues,
    )

    # benchstats encodes each comparison as baseline-vs-candidate: "<" means
    # baseline < candidate (candidate slower -> regression), ">" means candidate
    # faster, "~" means not significant at alpha.
    for metric in main_metrics:
        counts = {"<": 0, ">": 0, "~": 0}
        for bm_res in cr.results.values():
            res = bm_res.get(metric)
            if res is not None:
                counts[res.result] = counts.get(res.result, 0) + 1
        total = sum(counts.values())
        console.print(
            f"\nSummary for '{metric}' ({cr.method}, alpha={cr.alpha:g}, "
            f"{total} benchmarks):"
        )
        console.print(f"  candidate faster (significant, '>'): {counts['>']}")
        console.print(f"  candidate slower (significant, '<'): {counts['<']}")
        console.print(f"  no significant difference ('~'):     {counts['~']}")

    if export_fmt is not None:
        {"txt": lambda: console.save_text(args.export_to),
         "svg": lambda: console.save_svg(args.export_to, title=""),
         "html": lambda: console.save_html(args.export_to)}[export_fmt]()

    if cr.at_least_one_differs:
        console.warning("At least one significant timing difference was detected (exit 1).")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Statistically compare two driver result JSONs via benchstats.")
    parser.add_argument("baseline_json", help="Baseline result JSON")
    parser.add_argument("candidate_json", help="Candidate result JSON")
    parser.add_argument("--filter", default=None,
                        help="Only compare benchmarks whose name matches this regex.")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Significance level for the test (default: 0.001).")
    parser.add_argument("--method", default="brunnermunzel",
                        help="Statistical test to use (default: brunnermunzel).")
    parser.add_argument("--always-show-pvalues", action="store_true",
                        help="Show p-values for non-significant rows too.")
    parser.add_argument("--export-to", default=None, metavar="FILE",
                        help="Export the report to a .txt/.svg/.html file (format from extension).")
    return run_stats(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
