#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Convert benchmark CSV files (from benchmark_*.py) to ASV-compatible JSON.

Reads one or more CSV files produced by the microbenchmarks and writes
the same JSON result format that asv requires, so results
can be visualised with ``asv publish && asv preview`` or compared with
``asv compare``.

Usage:
    # Convert all CSVs in the current directory
    python csv_to_asv.py benchmark_gemm.csv benchmark_casting.csv ...

    # Convert all benchmark CSVs found in a directory
    python csv_to_asv.py perf_results/pr/*.csv

    # Specify output directory (default: benchmarks/.asv/results)
    python csv_to_asv.py --results-dir ./my_results benchmark_gemm.csv

    # Provide a custom machine name or commit hash
    python csv_to_asv.py --machine mi325 --commit abc1234 *.csv
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import pandas as pd


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Columns that are never parameters and never metrics
_SKIP_COLS = {"TestID", "Label"}


def _classify_columns(df):
    """Split DataFrame columns into (key_cols, time_cols, throughput_cols).

    Heuristic:
      - Columns containing "Time" and "(ms)" are time metrics.
      - Columns containing "TFLOPS" or "GB/s" are throughput metrics.
      - Everything else (except _SKIP_COLS) is a key/parameter column.
    """
    time_cols = []
    throughput_cols = []
    key_cols = []

    for c in df.columns:
        if c in _SKIP_COLS:
            continue
        if "Time" in c and "(ms)" in c:
            time_cols.append(c)
        elif "TFLOPS" in c or "GB/s" in c:
            throughput_cols.append(c)
        else:
            key_cols.append(c)

    return key_cols, time_cols, throughput_cols


def _pair_time_throughput(time_cols, throughput_cols):
    """Pair each time column with its throughput companion, if any.

    Matching heuristic: strip the distinctive suffixes and compare the
    remaining prefix.  E.g.
        "TE Forward Time (ms)"  <->  "TE Forward TFLOPS"
        "Cast Time (ms)"        <->  "Cast GB/s"

    Returns a list of (time_col, throughput_col_or_None) tuples.
    """

    def _time_key(col):
        return col.replace(" Time (ms)", "").strip()

    def _tp_key(col):
        for suffix in (" TFLOPS", " GB/s"):
            if col.endswith(suffix):
                return col[: -len(suffix)].strip()
        return col.strip()

    tp_by_key = {}
    for tc in throughput_cols:
        tp_by_key[_tp_key(tc)] = tc

    pairs = []
    matched_tp = set()
    for tc in time_cols:
        key = _time_key(tc)
        companion = tp_by_key.get(key)
        pairs.append((tc, companion))
        if companion:
            matched_tp.add(companion)

    # Standalone throughput columns (no matching time col)
    for tc in throughput_cols:
        if tc not in matched_tp:
            pairs.append((None, tc))

    return pairs


# ---------------------------------------------------------------------------
# ASV helpers  (mirrored from PR #487 driver.py)
# ---------------------------------------------------------------------------

def _get_machine_info():
    """Build the params / machine dict that ASV expects."""
    machine = platform.node()
    info = {
        "arch": platform.machine(),
        "cpu": "",
        "machine": machine,
        "num_cpu": str(os.cpu_count()),
        "os": f"{platform.system()} {platform.release()}",
        "ram": "",
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram"] = line.split()[1]  # kB
                    break
    except OSError:
        pass
    return machine, info


def _get_commit_hash():
    """Get the current git HEAD hash."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _format_param_value(v):
    """Format a parameter value the way ASV stores it in JSON."""
    if isinstance(v, str):
        return f"'{v}'"
    return repr(v)


def _make_version(suite_name, bench_name, param_names):
    """Deterministic version hash for a benchmark entry."""
    code = f"{suite_name}.{bench_name}({', '.join(param_names)})"
    return hashlib.sha256(code.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def csv_to_asv_entries(csv_path):
    """Convert one CSV file into ASV result + meta dicts.

    Returns (results_dict, meta_dict) where each key is a fully-qualified
    benchmark name like ``benchmark_gemm.time_forward``.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}, {}

    suite_name = os.path.splitext(os.path.basename(csv_path))[0]
    key_cols, time_cols, throughput_cols = _classify_columns(df)
    pairs = _pair_time_throughput(time_cols, throughput_cols)

    # Build the ASV parameter axes from unique values in key columns
    param_names = list(key_cols)
    asv_params = []
    for col in key_cols:
        asv_params.append([_format_param_value(v) for v in df[col].unique().tolist()])

    # Build a lookup from key tuple -> row index for fast access
    key_tuples = [tuple(row) for row in df[key_cols].values]

    # Cross-product of unique param values (in the order they appear)
    unique_per_col = [df[col].unique().tolist() for col in key_cols]
    import itertools

    all_combos = list(itertools.product(*unique_per_col))

    combo_to_idx = {}
    for idx, kt in enumerate(key_tuples):
        combo_to_idx[kt] = idx

    results = {}
    meta = {}
    now_ms = int(time.time() * 1000)

    for time_col, tp_col in pairs:
        # Derive a short benchmark name
        if time_col:
            # e.g. "TE Forward Time (ms)" -> "time_te_forward"
            short = time_col.replace(" Time (ms)", "").strip()
            short = "time_" + short.lower().replace(" ", "_").replace("(", "").replace(")", "")
            bench_key = f"{suite_name}.{short}"
        elif tp_col:
            short = tp_col.strip()
            for suffix in (" TFLOPS", " GB/s"):
                short = short.replace(suffix, "")
            short = short.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            bench_key = f"{suite_name}.throughput_{short}"
        else:
            continue

        version = _make_version(suite_name, short, param_names)

        # Populate values for every combo in the cross-product
        time_values = []
        for combo in all_combos:
            idx = combo_to_idx.get(combo)
            if idx is not None and time_col and time_col in df.columns:
                val = df.loc[idx, time_col]
                try:
                    # Convert ms -> seconds (ASV convention)
                    time_values.append(float(val) / 1000.0)
                except (ValueError, TypeError):
                    time_values.append(None)
            else:
                time_values.append(None)

        # Store the time benchmark
        if time_col:
            n = len(time_values)
            results[bench_key] = [
                time_values,       # result (medians)
                asv_params,        # params
                version,           # version
                now_ms,            # started_at
                0,                 # duration
                [None] * n,        # stats_ci_99_a
                [None] * n,        # stats_ci_99_b
                [None] * n,        # stats_q_25
                [None] * n,        # stats_q_75
                [1] * n,           # stats_number
                [1] * n,           # stats_repeat
            ]
            meta[bench_key] = {
                "code": "",
                "name": bench_key,
                "param_names": param_names,
                "params": asv_params,
                "timeout": 300,
                "type": "time",
                "unit": "seconds",
                "version": version,
            }

        # Store the throughput companion
        if tp_col:
            tp_values = []
            for combo in all_combos:
                idx = combo_to_idx.get(combo)
                if idx is not None and tp_col in df.columns:
                    val = df.loc[idx, tp_col]
                    try:
                        tp_values.append(float(val))
                    except (ValueError, TypeError):
                        tp_values.append(None)
                else:
                    tp_values.append(None)

            if "TFLOPS" in tp_col:
                tp_unit = "TFLOPS"
            elif "GB/s" in tp_col:
                tp_unit = "GB/s"
            else:
                tp_unit = ""

            tp_short = tp_col.strip()
            for suffix in (" TFLOPS", " GB/s"):
                tp_short = tp_short.replace(suffix, "")
            tp_short = tp_short.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            tp_key = f"{suite_name}.throughput_{tp_short}"
            tp_version = _make_version(suite_name, f"throughput_{tp_short}", param_names)

            n = len(tp_values)
            results[tp_key] = [
                tp_values,
                asv_params,
                tp_version,
                now_ms,
                0,
                [None] * n,
                [None] * n,
                [None] * n,
                [None] * n,
                [1] * n,
                [1] * n,
            ]
            meta[tp_key] = {
                "code": "",
                "name": tp_key,
                "param_names": param_names,
                "params": asv_params,
                "timeout": 300,
                "type": "time",
                "unit": tp_unit,
                "version": tp_version,
            }

    return results, meta


def save_asv_results(all_results, all_meta, results_dir, machine_name=None,
                     commit_hash=None):
    """Write results and benchmark index to ASV's results directory."""
    if commit_hash is None:
        commit_hash = _get_commit_hash()
    detected_machine, machine_info = _get_machine_info()
    if machine_name:
        machine_info["machine"] = machine_name
    else:
        machine_name = detected_machine

    env_name = "existing-" + sys.executable.replace("/", "_").strip("_")
    machine_dir = os.path.join(results_dir, machine_name)
    os.makedirs(machine_dir, exist_ok=True)

    # Write machine.json if missing
    machine_json = os.path.join(machine_dir, "machine.json")
    if not os.path.exists(machine_json):
        with open(machine_json, "w") as f:
            json.dump({**machine_info, "version": 1}, f, indent=4)

    # Load existing result file or start fresh
    filename = f"{commit_hash[:8]}-{env_name}.json"
    result_path = os.path.join(machine_dir, filename)
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
    else:
        data = {
            "commit_hash": commit_hash,
            "env_name": env_name,
            "date": int(time.time() * 1000),
            "params": {**machine_info, "python": sys.executable},
            "python": sys.executable,
            "requirements": {},
            "env_vars": {},
            "result_columns": [
                "result", "params", "version",
                "started_at", "duration",
                "stats_ci_99_a", "stats_ci_99_b",
                "stats_q_25", "stats_q_75",
                "stats_number", "stats_repeat",
            ],
            "results": {},
            "durations": {},
            "version": 2,
        }

    # Merge new results
    for bench_key, bench_data in all_results.items():
        data["results"][bench_key] = bench_data

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {result_path}")

    # Update benchmarks.json index
    benchmarks_path = os.path.join(results_dir, "benchmarks.json")
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path) as f:
            benchmarks_data = json.load(f)
    else:
        benchmarks_data = {"version": 2}

    benchmarks_data.update(all_meta)

    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks_data, f, indent=4)

    print(f"Updated {benchmarks_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark CSV files to ASV-compatible JSON.")
    parser.add_argument("csv_files", nargs="+",
                        help="CSV files produced by benchmark_*.py")
    parser.add_argument("--results-dir", default=None,
                        help="ASV results directory "
                             "(default: benchmarks/.asv/results relative to repo root)")
    parser.add_argument("--machine", default=None,
                        help="Machine name for ASV (default: hostname)")
    parser.add_argument("--commit", default=None,
                        help="Commit hash (default: git rev-parse HEAD)")
    args = parser.parse_args()

    if args.results_dir is None:
        # Default: benchmarks/.asv/results relative to repo root
        try:
            repo_root = (
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            repo_root = os.getcwd()
        args.results_dir = os.path.join(repo_root, "benchmarks", ".asv", "results")

    all_results = {}
    all_meta = {}

    for csv_path in args.csv_files:
        for f in glob.glob(csv_path):
            print(f"Processing {f} ...")
            results, meta_data = csv_to_asv_entries(f)
            all_results.update(results)
            all_meta.update(meta_data)
            print(f"  {len(results)} benchmark entries extracted")

    if not all_results:
        print("No benchmark data found.")
        return

    save_asv_results(all_results, all_meta, args.results_dir,
                     machine_name=args.machine, commit_hash=args.commit)


if __name__ == "__main__":
    main()
