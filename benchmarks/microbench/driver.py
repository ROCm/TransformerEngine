#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Microbenchmark driver — runs bench classes via torch.utils.benchmark.Timer
and writes long-format CSV.

Usage:
    python driver.py <suite> [method_filter] [--csv FILE | --no-csv]
    python driver.py --all [--csv FILE | --no-csv]
    python bench_gemm.py [method_filter] [--csv FILE | --no-csv]

CSV schema (one row per Timer block):
    suite, class, method, params, sample_idx, time_s, number_per_run,
    tflops, gbps, commit, machine, started_at_ms

Each row's `time_s` is one block's per-call mean (block_total / number_per_run).
The downstream analysis tool can group by (suite, class, method, params) to
recover the distribution of block-mean per-call times.
"""

import argparse
import csv
import glob
import importlib
import itertools
import os
import platform
import subprocess
import sys
import time

import torch.utils.benchmark as benchmark


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def _get_machine_name():
    return platform.node() or "unknown"


def _get_commit_hash(short=False):
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha[:8] if short else sha
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Timing helper used by bench files
# ---------------------------------------------------------------------------

def time_func(fn, min_run_time=1.0, method="blocked"):
    """Time *fn* with torch.utils.benchmark.Timer and return the Measurement.

    The Measurement object exposes per-block elapsed times (`.times`) and
    `.number_per_run` (kernel invocations averaged per block). The driver
    flattens these into long-format CSV rows.

    method:
        "blocked"  — fixed-block sampling, more samples (recommended for stats).
        "adaptive" — stops when noise threshold is met; fewer, variable samples.
    """
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    if method == "adaptive":
        return timer.adaptive_autorange(min_run_time=min_run_time)
    return timer.blocked_autorange(min_run_time=min_run_time)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "suite", "class", "method", "params", "sample_idx", "time_s",
    "number_per_run", "tflops", "gbps", "commit", "machine", "started_at_ms",
]


def _default_csv_path(script_dir):
    """benchmarks/.bench-results/<machine>/<commit-short>.csv, anchored at the repo root."""
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.join(
        repo_root, "benchmarks", ".bench-results",
        _get_machine_name(), f"{_get_commit_hash(short=True)}.csv",
    )


def save_csv_results(rows, csv_path, append=False):
    """Write sample rows to *csv_path* (long format, one row per Timer block)."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    write_header = not (append and os.path.exists(csv_path))
    with open(csv_path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults {'appended to' if append else 'saved to'} {csv_path}  "
          f"({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _format_params(param_names, combo):
    """Canonical 'k1=v1;k2=v2' string for joining across runs."""
    return ";".join(f"{n}={v}" for n, v in zip(param_names, combo))


def _measurement_to_rows(measurement, *, suite, class_name, method_name,
                        params_str, work, commit, machine, started_at_ms):
    """Flatten a Timer Measurement into one CSV row per block.

    Measurement.times is already per-call (Timer divides by number_per_run
    internally). number_per_run is recorded as metadata so the analysis tool
    knows how many invocations were averaged into each row's time_s.
    """
    n = measurement.number_per_run
    rows = []
    for i, per_call_s in enumerate(measurement.times):
        rows.append({
            "suite": suite,
            "class": class_name,
            "method": method_name,
            "params": params_str,
            "sample_idx": i,
            "time_s": per_call_s,
            "number_per_run": n,
            "tflops": (work["flops"] / per_call_s / 1e12)
                      if "flops" in work and per_call_s > 0 else "",
            "gbps": (work["bytes"] / per_call_s / 1e9)
                    if "bytes" in work and per_call_s > 0 else "",
            "commit": commit,
            "machine": machine,
            "started_at_ms": started_at_ms,
        })
    return rows


def run_class(suite_name, cls, class_name, method_filter=None,
              commit=None, machine=None):
    """Run all benchmarks in a class. Returns a list of CSV row dicts."""
    methods = sorted(m for m in dir(cls) if m.startswith("time_"))
    if method_filter:
        methods = [m for m in methods if method_filter in m]
    if not methods:
        return []

    params = getattr(cls, "params", [[]])
    param_names = getattr(cls, "param_names", [])
    combos = list(itertools.product(*params))

    # Discover throughput columns from work_* companions
    probe_keys = set()
    for m in methods:
        wfn = getattr(cls, "work_" + m[5:], None)
        if wfn:
            try:
                probe_keys.update(wfn(cls(), *combos[0]))
            except Exception:
                pass
    has_tflops = "flops" in probe_keys
    has_gbps = "bytes" in probe_keys

    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods, "
          "Timer-driven)")
    extra_hdr = ""
    if has_tflops:
        extra_hdr += f"  {'TFLOPS':>10}"
    if has_gbps:
        extra_hdr += f"  {'GB/s':>10}"
    HDR = (f"  {'median':>10}  {'mean':>10}  {'iqr':>10}  {'n_blocks':>9}"
           f"  {'per_run':>8}" + extra_hdr + f"  {'method':<30}  params")
    print("-" * len(HDR))
    print(HDR)
    print("-" * len(HDR))

    rows = []
    for method_name in methods:
        started_at_ms = int(time.time() * 1000)
        for combo in combos:
            label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            params_str = _format_params(param_names, combo)
            instance = cls()
            try:
                instance.setup(*combo)
            except Exception as e:
                print(f"  SKIP  {label}  setup failed: {e}")
                continue

            method = getattr(instance, method_name)
            try:
                measurement = method(*combo)
            except Exception as e:
                print(f"  SKIP  {label}  {method_name} failed: {e}")
                continue

            wfn = getattr(instance, "work_" + method_name[5:], None)
            work = {}
            if wfn:
                try:
                    work = wfn(*combo)
                except Exception:
                    pass

            rows.extend(_measurement_to_rows(
                measurement, suite=suite_name, class_name=class_name,
                method_name=method_name, params_str=params_str, work=work,
                commit=commit, machine=machine, started_at_ms=started_at_ms,
            ))

            median_s = measurement.median
            mean_s = measurement.mean
            iqr_s = measurement.iqr
            extra_cols = ""
            if has_tflops:
                extra_cols += (f"  {work['flops'] / median_s / 1e12:>10.1f}"
                               if "flops" in work and median_s > 0 else f"  {'':>10}")
            if has_gbps:
                extra_cols += (f"  {work['bytes'] / median_s / 1e9:>10.1f}"
                               if "bytes" in work and median_s > 0 else f"  {'':>10}")
            print(f"  {median_s*1000:>8.3f}ms  {mean_s*1000:>8.3f}ms  "
                  f"{iqr_s*1000:>8.3f}ms  {len(measurement.times):>9}  "
                  f"{measurement.number_per_run:>8}"
                  f"{extra_cols}  {method_name:<30}  {label}")

    return rows


def run_as_main(caller_file=None):
    """Run benchmarks from a bench file or from the command line.

    When called with a file path (from a bench file's ``__main__`` block),
    the suite is derived from the filename. When called without arguments
    (i.e. ``python driver.py bench_gemm``), the suite is taken from argv.

    Usage from a bench file::

        if __name__ == "__main__":
            from driver import run_as_main
            run_as_main(__file__)
    """
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks via torch.utils.benchmark and emit CSV.")
    if caller_file is None:
        parser.add_argument("suite", nargs="?", default=None,
                            help="Benchmark module name (e.g. bench_casting)")
        parser.add_argument("--all", action="store_true",
                            help="Run all bench_*.py suites in the directory")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="Only run time_* methods containing this string")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="Output CSV path. Default: "
                             "benchmarks/.bench-results/<machine>/<commit>.csv")
    parser.add_argument("--no-csv", action="store_true",
                        help="Don't write CSV (stdout summary only).")
    parser.add_argument("--append", action="store_true",
                        help="Append to the CSV instead of overwriting.")
    args = parser.parse_args()

    if caller_file is not None:
        script_dir = os.path.dirname(os.path.abspath(caller_file))
        suite_names = [os.path.splitext(os.path.basename(caller_file))[0]]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_all = getattr(args, "all", False)
        if run_all:
            suite_names = sorted(
                os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(script_dir, "bench_*.py"))
            )
        elif args.suite:
            suite_names = [args.suite]
        else:
            parser.error("provide a suite name or use --all")

    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    commit = _get_commit_hash(short=True)
    machine = _get_machine_name()

    all_rows = []
    for suite_name in suite_names:
        mod = importlib.import_module(suite_name)
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if isinstance(obj, type) and name.startswith("Bench"):
                all_rows.extend(run_class(
                    suite_name, obj, name, args.method_filter,
                    commit=commit, machine=machine,
                ))

    if all_rows and not args.no_csv:
        csv_path = args.csv or _default_csv_path(script_dir)
        save_csv_results(all_rows, csv_path, append=args.append)


if __name__ == "__main__":
    run_as_main()
