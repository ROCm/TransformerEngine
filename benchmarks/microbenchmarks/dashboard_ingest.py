#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ingest microbenchmark CSVs into the CI dashboard's history schema.

The benchmarks in this directory (``benchmark_gemm.py``, ``benchmark_casting.py``,
...) write a results CSV via ``--csv``: one row per config, with paired
``<label> Time (ms)`` and ``<label> <unit>`` columns (unit = ``TFLOPS`` or
``GB/s``), already computed. This script turns one commit's CSV(s) into records
in ``dashboard/data/history.json`` -- the schema the ported FlyDSL dashboard
(``dashboard/app.js``) consumes.

Each invocation is ONE run, with a unique ``run_id``. Pass ``--append`` to
accumulate runs so the dashboard's per-kernel 2-sigma noise band can form (it
needs the latest run plus >= 3 prior runs -- i.e. >= 4 runs -- per kernel). Runs
may be different commits OR repeated runs of the same commit: repeated same-code
runs are exactly the run-to-run noise the band measures. Pin ``--run-id`` (or
``--ts``) to make re-ingesting a specific run idempotent instead of adding one.

Stdlib only (no torch / no pandas); safe to run anywhere the CSVs are visible.

Usage (after producing CSVs -- e.g. in the container where TE is installed):
  python benchmark_gemm.py --csv                 # -> benchmark_gemm.csv
  python benchmark_casting.py --csv              # -> benchmark_casting.csv
  # fold this commit's CSVs into the history (one run):
  python dashboard_ingest.py benchmark_gemm.csv benchmark_casting.csv \\
      --commit "$(git rev-parse HEAD)" --append
  # serve the static dashboard (any Python, no GPU needed):
  (cd dashboard && python -m http.server 8000)   # http://localhost:8000
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Throughput units the dev suite emits; a "<label> <unit>" column pairs with a
# "<label> Time (ms)" column. "higher is better" holds for both, so a drop past
# the noise band is a regression.
METRIC_UNITS = ("TFLOPS", "GB/s")

# The dashboard sizes a kernel's 2-sigma band from prior runs, so it needs the
# latest run plus >= 3 prior runs before a band appears.
MIN_RUNS_FOR_BAND = 4


def _unit_of(col):
    for unit in METRIC_UNITS:
        if col.endswith(unit):
            return unit
    return None


def _num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _run_id_from_ts(ts):
    """Run id (ms since epoch) from an ISO-8601 UTC timestamp."""
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _shape_dtype(row, key_cols):
    """Derive a compact (shape, dtype) label from a CSV row's parameter columns."""
    dtype = "bf16"
    params = {}
    for col in key_cols:
        if col.lower() == "dtype":
            dtype = str(row[col]).replace("torch.", "")
        else:
            params[col] = row[col]
    if "Case" in params and "M" in params:          # dense GEMM: readable + unique
        shape = f"{params['Case']} M{params['M']}"
    elif {"M", "N", "K"} <= set(params):
        shape = f"{params['M']}x{params['N']}x{params['K']}"
    else:
        shape = ", ".join(f"{k}={v}" for k, v in params.items()) or "-"
    return shape, dtype


def records_from_csv(path, arch, runner, commit, ts, run_id):
    """Yield history records for one dev-suite results CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        metric_cols = [c for c in cols if _unit_of(c)]
        key_cols = [c for c in cols if c not in metric_cols and "Time" not in c]
        for row in reader:
            shape, dtype = _shape_dtype(row, key_cols)
            for mcol in metric_cols:
                unit = _unit_of(mcol)
                value = _num(row.get(mcol))
                if value is None or value <= 0:
                    continue
                label = mcol[: -(len(unit) + 1)].rstrip()  # "GEMM Forward TFLOPS" -> "GEMM Forward"
                ms = _num(row.get(f"{label} Time (ms)"))
                yield {
                    "op": label,
                    "shape": shape,
                    "dtype": dtype,
                    "metric": unit,
                    "value": round(value, 4),
                    "status": "ok",
                    "vs_main": None,
                    "vs_tag": None,
                    "regression": False,
                    "extra": ({"median_ms": round(ms, 4)} if ms else {}),
                    "ts": ts,
                    "commit": commit,
                    "pr": None,
                    "run_id": run_id,
                    "source": "ci",
                    "runner": runner,
                    "arch": arch,
                }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="Results CSV(s) for ONE commit")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "dashboard" / "data"),
                        help="Where to write history.json / runs.json")
    parser.add_argument("--arch", default="gfx950", help="GPU arch tag")
    parser.add_argument("--runner", default="local", help="Runner label")
    parser.add_argument("--repo", default="ROCm/TransformerEngine")
    parser.add_argument("--commit", default=None, help="Commit sha (default: git HEAD)")
    parser.add_argument("--ts", default=None, help="ISO-8601 UTC timestamp (default: now)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Explicit run id (default: unique per invocation). Pin it "
                             "to make re-ingesting the same run idempotent.")
    parser.add_argument("--append", action="store_true",
                        help="Merge into existing history.json (accumulate runs)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    commit = args.commit or _git_head()
    if args.ts:
        ts = args.ts
        default_run_id = _run_id_from_ts(ts)
    else:
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        default_run_id = int(now.timestamp() * 1000)  # ms -> unique per invocation
    run_id = args.run_id if args.run_id is not None else default_run_id

    new = []
    for path in args.csv:
        try:
            new.extend(records_from_csv(path, args.arch, args.runner, commit, ts, run_id))
        except (OSError, csv.Error) as exc:
            print(f"  skip {path}: {exc}")
    if not new:
        sys.exit("no throughput (TFLOPS / GB/s) records found in the given CSV(s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.json"

    records = []
    if args.append and hist_path.exists():
        try:
            records = json.loads(hist_path.read_text()).get("records", [])
        except (OSError, json.JSONDecodeError):
            records = []
        # Replace this run's matching kernel points in place; keep everything else.
        newkeys = {(r["run_id"], r["op"], r["shape"], r["dtype"], r["metric"]) for r in new}
        records = [r for r in records
                   if (r.get("run_id"), r.get("op"), r.get("shape"),
                       r.get("dtype"), r.get("metric")) not in newkeys]
    records.extend(new)

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hist_path.write_text(json.dumps(
        {"schema": 1, "updated": updated, "repo": args.repo, "records": records}, indent=1))
    runs_path = out_dir / "runs.json"
    if not runs_path.exists():
        runs_path.write_text(json.dumps(
            {"schema": 1, "updated": updated, "repo": args.repo, "runs": []}, indent=1))

    kernels = {(r["op"], r["shape"], r["dtype"], r["arch"]) for r in records}
    runs = {r["run_id"] for r in records}
    commits = {r["commit"] for r in records}
    print(f"run {commit[:8]} @ {ts} (run_id {run_id}): +{len(new)} records from "
          f"{len(args.csv)} CSV(s)")
    print(f"history now: {len(records)} records, {len(kernels)} kernel series, "
          f"{len(runs)} run(s) across {len(commits)} commit(s)")
    print(f"  wrote {hist_path}")
    if len(runs) < MIN_RUNS_FOR_BAND:
        print(f"  note: {len(runs)}/{MIN_RUNS_FOR_BAND} runs so far -- the 2-sigma band needs "
              "the latest run plus >= 3 prior runs per kernel. Re-run the benchmarks "
              "(even on the same commit) and append again to add more runs.")


if __name__ == "__main__":
    main()
