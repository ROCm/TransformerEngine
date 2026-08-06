#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Append microbenchmark results into the dashboard's per-family CSV shards.

The benchmarks in this directory (``benchmark_gemm.py``, ``benchmark_casting.py``,
...) write a results CSV via ``--csv``: one row per config, with paired
``<label> Time (ms)`` and ``<label> <unit>`` columns (unit = ``TFLOPS`` or
``GB/s``), already computed. This script reshapes each into long ("tidy") rows
and **appends** them to a per-family, per-ref shard the dashboard reads directly:

    dashboard/data/perf-<family>-<ref>.csv     # e.g. perf-gemm-dev.csv, perf-gemm-pr1234.csv
    dashboard/data/index.csv                   # catalog: file,family,ref,pr

There is no derived ``history.json`` -- the CSV shards are the single source of
truth. ``<family>`` comes from the input file name (``benchmark_gemm.csv`` ->
``gemm``); ``<ref>`` is the baseline branch (default ``dev``) or ``pr<N>`` when
``--pr N`` is given. The shard rows are append-only (git-friendly); each ingest
invocation is one run with a unique ``run_id`` (repeated runs of the same commit
accumulate -- that is exactly the run-to-run noise the dashboard's band measures).

``index.csv`` lets the front-end discover shards without a hardcoded list; it is
updated only when a *new* shard first appears (new family, or a new PR).

Stdlib only (no torch / no pandas); safe to run anywhere the CSVs are visible.

Usage:
  python benchmark_gemm.py --csv                 # -> benchmark_gemm.csv (needs TE + GPU)
  python benchmark_casting.py --csv
  # append this run to the dev shards:
  python dashboard_ingest.py benchmark_*.csv --commit "$(git rev-parse HEAD)"
  # a PR run instead (isolated from the dev baseline):
  python dashboard_ingest.py benchmark_*.csv --pr 1234 --commit "$PR_HEAD_SHA"
  # serve the static dashboard (any Python, no GPU needed):
  (cd dashboard && python -m http.server 8000)   # http://localhost:8000
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Throughput units the suite emits; a "<label> <unit>" column pairs with a
# "<label> Time (ms)" column. "higher is better" holds for both, so a drop past
# the noise band is a regression.
METRIC_UNITS = ("TFLOPS", "GB/s")

# Long-format shard columns (one row per measurement) and the shard catalog.
SHARD_HEADER = ["ts", "commit", "run_id", "arch", "model", "runner",
                "op", "shape", "dtype", "metric", "value", "time_ms", "pr"]
INDEX_HEADER = ["file", "family", "ref", "pr"]


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


def _detect_arch():
    """Best-effort GPU arch for this machine (one GPU type per box).

    Tries ``rocminfo`` first (present on any ROCm install, no torch needed), then
    falls back to torch's ``gcnArchName``. Returns a bare ``gfxNNNN`` string, or
    None if nothing could be determined.
    """
    try:
        out = subprocess.run(["rocminfo"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=20).stdout.decode(errors="ignore")
        m = re.search(r"gfx[0-9a-fA-F]+", out)   # first gfx agent name = the GPU
        if m:
            return m.group(0)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return None


def _model_token(name):
    """Short GPU model from a marketing name (e.g. 'MI355X'), or None.

    Only GPU names qualify (an Instinct ``MI###`` code or a Radeon card), so the
    CPU ``Marketing Name`` lines ``rocminfo`` also emits are ignored.
    """
    if not name:
        return None
    m = re.search(r"MI\s?\d{3,4}[A-Za-z]*", name)
    if m:
        return m.group(0).replace(" ", "").upper()
    if "Radeon" in name:
        return name.replace("AMD", "").strip() or None
    return None


def _detect_model():
    """Best-effort short GPU model label (e.g. 'MI355X'), or None.

    Discovered on the GPU box from ``rocminfo`` ("Marketing Name"), then torch,
    so the arch->model label mapping is never hardcoded in the dashboard.
    """
    candidates = []
    try:
        out = subprocess.run(["rocminfo"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=20).stdout.decode(errors="ignore")
        candidates += [ln.split(":", 1)[1].strip()
                       for ln in out.splitlines() if "Marketing Name" in ln]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch
        candidates.append(torch.cuda.get_device_name(0))
    except Exception:
        pass
    for name in candidates:
        tok = _model_token(name)
        if tok:
            return tok
    return None


def _run_id_from_ts(ts):
    """Run id (ms since epoch) from an ISO-8601 UTC timestamp."""
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _family_of(path):
    """Benchmark family from the CSV file name (``benchmark_gemm.csv`` -> ``gemm``)."""
    stem = Path(path).stem
    return stem[len("benchmark_"):] if stem.startswith("benchmark_") else stem


def _shape_dtype(row, key_cols):
    """Derive a compact (shape, dtype) label from a CSV row's parameter columns.

    The label must be *unique* per distinct parameter combination -- otherwise
    several kernels collapse onto one series (e.g. grouped GEMM sweeps an extra
    expert-count column ``B``, so DSV2-Down/M512/B{5,10,20} are three kernels,
    not one). ``Case`` implies ``N``/``K`` for these suites so those stay
    summarized by the readable base; every *other* swept column is appended so
    nothing is silently merged.
    """
    dtype = "bf16"
    params = {}
    for col in key_cols:
        if "dtype" in col.lower():                   # "dtype" or "dtype_str"
            dtype = str(row[col]).replace("torch.", "")
        else:
            params[col] = row[col]
    # Columns already represented by the readable base (Case summarizes N/K here).
    summarized = {"Case", "M", "N", "K"}
    extras = "".join(f" {k}={v}" for k, v in params.items() if k not in summarized)
    if "Case" in params and "M" in params:          # dense/grouped GEMM: readable
        shape = f"{params['Case']} M{params['M']}"
    elif {"M", "N", "K"} <= set(params):
        shape = f"{params['M']}x{params['N']}x{params['K']}"
    else:
        return (", ".join(f"{k}={v}" for k, v in params.items()) or "-"), dtype
    return shape + extras, dtype


def long_rows_from_csv(path, meta):
    """Yield long-format shard rows (dicts keyed by SHARD_HEADER) for one CSV."""
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
                    "ts": meta["ts"], "commit": meta["commit"], "run_id": meta["run_id"],
                    "arch": meta["arch"], "model": meta.get("model", ""),
                    "runner": meta["runner"],
                    "op": label + meta.get("op_suffix", ""), "shape": shape, "dtype": dtype,
                    "metric": unit, "value": round(value, 4),
                    "time_ms": "" if ms is None else round(ms, 4),
                    "pr": meta["pr"],
                }


def append_shard(path, rows):
    """Append *rows* to shard *path*, writing a header if it's new. Returns (was_new, n)."""
    was_new = not path.exists()
    n = 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHARD_HEADER)
        if was_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return was_new, n


def update_index(out_dir, entries):
    """Merge *entries* into ``index.csv`` (add new shards only). Returns count added."""
    idx = out_dir / "index.csv"
    rows, have = [], set()
    if idx.exists():
        with open(idx, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                have.add(row.get("file"))
    added = 0
    for entry in entries:
        if entry["file"] not in have:
            rows.append(entry)
            have.add(entry["file"])
            added += 1
    if added or not idx.exists():
        rows.sort(key=lambda r: (r.get("family", ""), r.get("ref", "")))
        with open(idx, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in INDEX_HEADER})
    return added


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="benchmark_<family>.csv file(s) for ONE run")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "dashboard" / "data"),
                        help="Dashboard data dir (default: dashboard/data)")
    parser.add_argument("--ref", default="dev",
                        help="Baseline branch ref for the shard name (default: dev)")
    parser.add_argument("--pr", type=int, default=None,
                        help="PR number -> writes to perf-<family>-pr<N>.csv instead of -<ref>")
    parser.add_argument("--commit", default=None, help="Commit sha (default: git HEAD)")
    parser.add_argument("--ts", default=None, help="ISO-8601 UTC timestamp (default: now)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Explicit run id (default: unique per invocation)")
    parser.add_argument("--arch", default=None,
                        help="GPU arch tag (e.g. gfx950); auto-detected via rocminfo/torch "
                             "if omitted")
    parser.add_argument("--runner", default="local", help="Runner label")
    parser.add_argument("--model", default=None,
                        help="GPU model label for the dashboard (e.g. MI355X); "
                             "auto-detected from rocminfo/torch if omitted")
    parser.add_argument("--op-suffix", default="",
                        help="Append this string to every op label (e.g. ' [kernel]') so a "
                             "compute-kernel run forms its own dashboard series instead of "
                             "merging into the matching e2e op.")
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

    if args.pr is not None:
        ref, pr_field = f"pr{args.pr}", str(args.pr)
    else:
        ref, pr_field = args.ref, ""
    arch = args.arch or _detect_arch()
    if not arch:
        sys.exit("could not determine the GPU arch; pass --arch (e.g. --arch gfx950)")
    model = args.model or _detect_model()
    meta = {"ts": ts, "commit": commit, "run_id": run_id,
            "arch": arch, "model": model or "", "runner": args.runner,
            "pr": pr_field, "op_suffix": args.op_suffix}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, per_shard, total = [], [], 0
    for path in args.csv:
        family = _family_of(path)
        try:
            rows = list(long_rows_from_csv(path, meta))
        except (OSError, csv.Error) as exc:
            print(f"  skip {Path(path).name}: {exc}")
            continue
        if not rows:
            print(f"  skip {Path(path).name}: no TFLOPS/GB/s columns")
            continue
        shard = out_dir / f"perf-{family}-{ref}.csv"
        _, n = append_shard(shard, rows)
        total += n
        per_shard.append((shard.name, n))
        entries.append({"file": shard.name, "family": family, "ref": ref, "pr": pr_field})

    if total == 0:
        sys.exit("no throughput (TFLOPS / GB/s) rows found in the given CSV(s)")
    added = update_index(out_dir, entries)

    print(f"run {commit[:8]} @ {ts} (run_id {run_id}) ref={ref} "
          f"arch={arch}{'' if args.arch else ' (auto)'}"
          f"{f' model={model}' if model else ''}: +{total} rows")
    for name, n in per_shard:
        print(f"  {name}: +{n}")
    if added:
        print(f"  index.csv: +{added} new shard(s)")
    print(f"  data dir: {out_dir}")


if __name__ == "__main__":
    main()
