#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Microbenchmark driver — runs Bench* classes via torch.utils.benchmark.

Each bench file declares one or more ``Bench*`` classes with ``params``,
``param_names``, and ``time_*`` methods (optionally paired with ``work_*``
companions returning ``{"flops": ...}`` or ``{"bytes": ...}``). The driver
runs them and writes long-format CSV (one row per Timer block).

Usage:
    python driver.py <suite> [method_filter] [--csv FILE | --no-csv]
    python driver.py --all   [method_filter] [--csv FILE | --no-csv]
    python bench_gemm.py     [method_filter] [--csv FILE | --no-csv]
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


def time_func(fn, min_run_time=1.0):
    """Time *fn* with torch.utils.benchmark.Timer (blocked_autorange)."""
    return benchmark.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(
        min_run_time=min_run_time)


def _commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


CSV_COLUMNS = [
    "suite", "class", "method", "params", "sample_idx", "time_s",
    "number_per_run", "tflops", "gbps", "commit", "machine", "started_at_ms",
]


def _default_csv_path(script_dir):
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.join(repo_root, "benchmarks", ".bench-results",
                        platform.node() or "unknown", f"{_commit()}.csv")


def run_class(suite, cls, class_name, method_filter, commit, machine):
    """Run all time_* methods of *cls* over the param cross-product."""
    methods = sorted(m for m in dir(cls) if m.startswith("time_")
                     and (not method_filter or method_filter in m))
    if not methods:
        return []

    param_names = getattr(cls, "param_names", [])
    combos = list(itertools.product(*getattr(cls, "params", [[]])))

    print(f"\n{class_name}  ({len(combos)} combos x {len(methods)} methods)")
    hdr = (f"  {'median':>10}  {'mean':>10}  {'iqr':>10}  "
           f"{'TFLOPS':>8}  {'GB/s':>8}  {'method':<28}  params")
    print("-" * len(hdr)); print(hdr); print("-" * len(hdr))

    def fmt(val):
        return f"{val:>8.1f}" if val else f"{'':>8}"

    rows = []
    for method_name in methods:
        started_at_ms = int(time.time() * 1000)
        for combo in combos:
            label = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            params_str = ";".join(f"{n}={v}" for n, v in zip(param_names, combo))
            inst = cls()
            try:
                inst.setup(*combo)
                m = getattr(inst, method_name)(*combo)
            except Exception as e:
                print(f"  SKIP  {label}  {method_name}: {e}")
                continue

            wfn = getattr(inst, "work_" + method_name[5:], None)
            work = wfn(*combo) if wfn else {}
            flops, byts = work.get("flops"), work.get("bytes")

            for i, t in enumerate(m.times):
                rows.append({
                    "suite": suite, "class": class_name, "method": method_name,
                    "params": params_str, "sample_idx": i, "time_s": t,
                    "number_per_run": m.number_per_run,
                    "tflops": flops / t / 1e12 if flops and t > 0 else "",
                    "gbps": byts / t / 1e9 if byts and t > 0 else "",
                    "commit": commit, "machine": machine,
                    "started_at_ms": started_at_ms,
                })

            tflops = flops / m.median / 1e12 if flops and m.median > 0 else None
            gbps = byts / m.median / 1e9 if byts and m.median > 0 else None
            print(f"  {m.median*1000:>8.3f}ms  {m.mean*1000:>8.3f}ms  "
                  f"{m.iqr*1000:>8.3f}ms  {fmt(tflops)}  {fmt(gbps)}  "
                  f"{method_name:<28}  {label}")
    return rows


def main(caller_file=None):
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks via torch.utils.benchmark.")
    if caller_file is None:
        parser.add_argument("suite", nargs="?",
                            help="bench module (e.g. bench_gemm)")
        parser.add_argument("--all", action="store_true",
                            help="run all bench_*.py in this directory")
    parser.add_argument("method_filter", nargs="?", default=None,
                        help="only run time_* methods containing this string")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="output CSV path "
                             "(default: benchmarks/.bench-results/<machine>/<commit>.csv)")
    parser.add_argument("--no-csv", action="store_true",
                        help="don't write CSV (stdout summary only)")
    args = parser.parse_args()

    if caller_file is not None:
        script_dir = os.path.dirname(os.path.abspath(caller_file))
        suites = [os.path.splitext(os.path.basename(caller_file))[0]]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if args.all:
            suites = sorted(os.path.splitext(os.path.basename(f))[0]
                            for f in glob.glob(os.path.join(script_dir, "bench_*.py")))
        elif args.suite:
            suites = [args.suite]
        else:
            parser.error("provide a suite name or use --all")

    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    commit, machine = _commit(), platform.node() or "unknown"
    rows = []
    for suite in suites:
        mod = importlib.import_module(suite)
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if isinstance(obj, type) and name.startswith("Bench"):
                rows.extend(run_class(suite, obj, name, args.method_filter,
                                      commit, machine))

    if rows and not args.no_csv:
        path = args.csv or _default_csv_path(script_dir)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
