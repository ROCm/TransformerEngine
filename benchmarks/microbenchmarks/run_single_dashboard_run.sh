#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
###############################################################################
#
# Single-shot microbenchmark run for the dashboard: runs each microbenchmark once
# against the CURRENT TE build and collects annotated CSVs ready for
# dashboard_ingest.py. One data point -- not a weekly series.
#
# Run this inside the TE GPU container (the benchmarks need a real GPU).
#
# Usage: bash run_single_dashboard_run.sh [OUTDIR]
#   KERNEL_PROFILE=0   skip GPU kernel timing (wall time only)
#   BENCH_FILES="..."  override the family list
###############################################################################
set -uo pipefail

MB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-${MB_DIR}/results/single_run}"
KERNEL_PROFILE="${KERNEL_PROFILE:-1}"
BENCH_FILES="${BENCH_FILES:-benchmark_gemm.py benchmark_casting.py benchmark_grouped_gemm.py benchmark_normalization.py}"

sha="$(git -C "${MB_DIR}" rev-parse HEAD)"
cdate="$(git -C "${MB_DIR}" show -s --format=%cI HEAD)"
week="$(date +%F)"
run_dir="${OUTDIR}/${week}_$(git -C "${MB_DIR}" rev-parse --short=12 HEAD)"
mkdir -p "${run_dir}"

# GPU model for the dashboard, detected on THIS machine (amd-smi market name ->
# short MI### token, else torch) so the ingest hint isn't hardcoded to one arch.
model="$(amd-smi static --asic 2>/dev/null | grep -oiE 'MI[0-9]{3,4}[A-Z]*' | head -1)"
[[ -n "${model}" ]] || model="$(python3 -c 'import torch,re; n=torch.cuda.get_device_name(0); m=re.search(r"MI\s?\d{3,4}[A-Za-z]*", n); print(m.group(0).replace(" ","").upper() if m else "")' 2>/dev/null)"
[[ -n "${model}" ]] || model="UNKNOWN"

kp=""; [[ "${KERNEL_PROFILE}" == "1" ]] && kp="--kernel-profile"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

rm -f "${MB_DIR}"/benchmark_*.csv          # clear stale generated CSVs (gitignored)
read -r -a files <<< "${BENCH_FILES}"
for f in "${files[@]}"; do
  echo "== ${f} =="
  ( cd "${MB_DIR}" && python "${f}" --csv ${kp} ) \
    || echo "  (${f} returned non-zero; continuing)"
done

# Collect + tag each CSV with run_week/commit_sha/commit_date -- the per-row
# metadata dashboard_ingest.py reads for the point's timestamp/commit.
shopt -s nullglob
for csv in "${MB_DIR}"/benchmark_*.csv; do
  base="$(basename "${csv}")"
  mv "${csv}" "${run_dir}/${base}"
  python3 - "${run_dir}/${base}" "${week}" "${sha}" "${cdate}" <<'PY'
import csv, sys
path, week, sha, cdate = sys.argv[1:5]
rows = list(csv.reader(open(path, newline="")))
if rows:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_week", "commit_sha", "commit_date", *rows[0]])
        for r in rows[1:]:
            w.writerow([week, sha, cdate, *r])
PY
  echo "  -> ${run_dir}/${base}"
done
shopt -u nullglob

echo
echo "CSVs ready in ${run_dir}"
echo "Ingest (from a checkout that has the dashboard tooling):"
echo "  python dashboard_ingest.py ${run_dir}/benchmark_*.csv --model ${model} --runner local --ref dev"
echo "  python build_bundle.py"
