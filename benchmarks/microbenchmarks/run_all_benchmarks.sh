#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Run the full microbenchmark suite and (optionally) ingest the results into the
# dashboard's per-family CSV shards so they're ready to redeploy.
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_all_benchmarks.sh [options]

  (no options)     run every microbenchmark -> benchmark_<family>.csv (cwd)
  --ingest         after each pass, append the results to the dashboard shards
                   (one run per pass) -- output ready to redeploy
  --out-dir DIR    shard dir for --ingest (default: dashboard/data)
  --runs N         repeat run+ingest N times to build a baseline (needs --ingest;
                   a noise band needs >=4 runs). Default: 1
  --ref REF        ingest baseline ref (default: dev)
  --pr N           ingest as PR <N> instead of --ref (isolated from the baseline)
  --python-time    record host wall-clock time instead of the default GPU kernel
                   (device) time (benchmarks otherwise run with --kernel-profile
                   so the shards hold kernel time)
  --compute-kernel record compute-kernel-only time (drops host/torch scaffolding
                   like randn/copies) as a separate series (ingested with an
                   op-suffix ' [kernel]'); pair with --ref to keep its own shard
  --bundle         after ingesting, emit a self-contained dashboard/dist/dashboard.html
  -h, --help       show this help

The benchmarks need TE + torch + a GPU; the ingest step is stdlib-only and
auto-detects the GPU model (e.g. MI355X). Override the benchmark
interpreter with PYTHON=... if needed.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

# Canonical microbenchmarks -- each maps to a dashboard family via its file name.
# (benchmark_normalization2.py is an experimental variant and is intentionally
# excluded; add it here if you want it ingested as its own family.)
BENCHMARKS=(
  benchmark_gemm.py
  benchmark_gemm_fp8.py
  benchmark_casting.py
  benchmark_normalization.py
  benchmark_grouped_gemm.py
)

INGEST=0
OUT_DIR="dashboard/data"
RUNS=1
REF="dev"
PR=""
BUNDLE=0
# Dashboard shards hold GPU kernel (device) time by default; --python-time opts out.
KERNEL_PROFILE="--kernel-profile"
# Non-empty for --compute-kernel: op-label suffix so its rows form their own series.
OP_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ingest) INGEST=1; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --pr) PR="$2"; shift 2 ;;
    --kernel-profile) KERNEL_PROFILE="--kernel-profile"; shift ;;
    --python-time) KERNEL_PROFILE=""; shift ;;
    --compute-kernel) KERNEL_PROFILE="--compute-kernel"; OP_SUFFIX=" [kernel]"; shift ;;
    --bundle) BUNDLE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$SCRIPT_DIR"

if [[ "$RUNS" -gt 1 && "$INGEST" != 1 ]]; then
  echo "error: --runs > 1 only makes sense with --ingest (otherwise each pass just overwrites the CSVs)" >&2
  exit 2
fi

# CSV names the benchmarks write (default = <script>.csv).
CSVS=()
for b in "${BENCHMARKS[@]}"; do CSVS+=("${b%.py}.csv"); done

if [[ -n "$PR" ]]; then
  ingest_args=(--pr "$PR" --out-dir "$OUT_DIR")
else
  ingest_args=(--ref "$REF" --out-dir "$OUT_DIR")
fi
if [[ -n "$OP_SUFFIX" ]]; then
  ingest_args+=(--op-suffix "$OP_SUFFIX")
fi

for ((run = 1; run <= RUNS; run++)); do
  [[ "$RUNS" -gt 1 ]] && echo "===== pass $run/$RUNS ====="
  for b in "${BENCHMARKS[@]}"; do
    echo ">>> $b"
    "$PY" "$b" --csv ${KERNEL_PROFILE}
  done
  if [[ "$INGEST" == 1 ]]; then
    echo ">>> ingest -> $OUT_DIR"
    python3 dashboard_ingest.py "${CSVS[@]}" "${ingest_args[@]}"
  fi
done

if [[ "$BUNDLE" == 1 ]]; then
  echo ">>> bundle -> dashboard/dist/dashboard.html"
  python3 build_bundle.py --data-dir "$OUT_DIR"
fi

echo "done."
if [[ "$INGEST" == 1 ]]; then
  echo "ingested into $OUT_DIR -- redeploy with:"
  echo "  ./dashboard_redeploy.sh --dst <gh-pages-checkout>$([[ "$OUT_DIR" == "dashboard/data" ]] && echo ' --with-data')"
else
  echo "wrote: ${CSVS[*]}   (add --ingest to append to the dashboard shards)"
fi
