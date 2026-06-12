#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Statistically compare two ASV result JSON files written by ``driver.py``.

The point-estimate timings in the ASV dashboard cannot tell a real regression
from measurement noise. This tool compares the raw per-call samples stored in
two result files (one per checkout) using a statistical test (Brunner-Munzel by
default) via the benchstats package. It marks each (benchmark, parameter
combination) as faster (``<``), slower (``>``), or not significantly different
(``~``) and exits ``1`` when a significant timing difference is found, so it can
gate CI. Requires ``pip install -r requirements.txt``.

Usage:
    # run the suite on the baseline checkout, then on the candidate checkout,
    # pointing each at its own results file, then:
    python compare_results.py baseline.json candidate.json
    python compare_results.py baseline.json candidate.json --alpha 0.01
    python compare_results.py baseline.json candidate.json --export-to report.svg
"""

import argparse
import os
import sys


def run_stats(args):
    """Compare two ASV result JSONs with a statistical test via benchstats.

    Returns a process exit code: 1 if a significant difference is found in the
    timing metric, else 0.
    """
    import rich.table  # noqa: F401  benchstats 3.4.0 render uses rich.table.Table without importing it
    from parser_TEasv import parser_TEasv
    from benchstats.compare import compareStats
    from benchstats.render import renderComparisonResults
    from benchstats.common import LoggingConsole, detectExportFormat

    main_metrics = ["time_s"]

    export_fmt = detectExportFormat(args.export_to, None) if args.export_to else None
    if export_fmt is not None and os.path.isfile(args.export_to):
        os.remove(args.export_to)

    console = LoggingConsole(
        record=export_fmt is not None,
        log_level=LoggingConsole.LogLevel.Warning,
    )

    s1 = parser_TEasv(args.baseline_json, args.filter, None, debug_log=console).getStats()
    s2 = parser_TEasv(args.candidate_json, args.filter, None, debug_log=console).getStats()

    cr = compareStats(
        s1, s2,
        method=args.method,
        alpha=args.alpha,
        main_metrics=main_metrics,
        debug_log=console,
    )

    renderComparisonResults(
        cr, console,
        main_metrics=main_metrics,
        always_show_pvalues=args.always_show_pvalues,
    )

    if export_fmt is not None:
        if export_fmt == "txt":
            console.save_text(args.export_to)
        elif export_fmt == "svg":
            console.save_svg(args.export_to, title="")
        elif export_fmt == "html":
            console.save_html(args.export_to)

    if cr.at_least_one_differs:
        console.warning(
            "At least one significant timing difference was detected (exit 1)."
        )
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Statistically compare two ASV result JSON files via benchstats.")
    parser.add_argument("baseline_json", help="Baseline ASV result JSON")
    parser.add_argument("candidate_json", help="Candidate ASV result JSON")
    parser.add_argument(
        "--filter", default=None,
        help="Only compare benchmarks whose name matches this regex.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.001,
        help="Significance level for the test (default: 0.001).",
    )
    parser.add_argument(
        "--method", default="brunnermunzel",
        help="Statistical test to use (default: brunnermunzel).",
    )
    parser.add_argument(
        "--always-show-pvalues", action="store_true",
        help="Always show p-values, including for non-significant results.",
    )
    parser.add_argument(
        "--export-to", default=None, metavar="FILE",
        help="Export the report to a .txt/.svg/.html file (format from extension).",
    )
    args = parser.parse_args()

    # The benchstats parser is imported lazily from the script directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    return run_stats(args)


if __name__ == "__main__":
    sys.exit(main())
