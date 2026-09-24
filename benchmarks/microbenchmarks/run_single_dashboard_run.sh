#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
###############################################################################
#
# Single-shot microbenchmark run for the dashboard: runs each microbenchmark once
# against the CURRENT TE build and collects annotated CSVs ready for
# dashboard_ingest.py.
#
# Usage: bash run_single_dashboard_run.sh [OUTDIR]
#   KERNEL_PROFILE=0   skip GPU kernel timing (wall time only)
#   RUN_FLYDSL=0       skip FlyDSL-backed cases (long compile; gfx950-only)
#   SAMPLES=0          skip per-iteration timing samples (--csv-samples)
#   BENCH_FILES="..."  override the family list
###############################################################################
set -uo pipefail

MB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-${MB_DIR}/results/single_run}"
KERNEL_PROFILE="${KERNEL_PROFILE:-1}"
RUN_FLYDSL="${RUN_FLYDSL:-1}"        # include FlyDSL-backed cases (--run-flydsl); 0 to skip
SAMPLES="${SAMPLES:-1}"              # also write per-iteration wall-clock samples (--csv-samples); 0 to skip
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
host="$(hostname -s 2>/dev/null || hostname)"
gpu="${HIP_VISIBLE_DEVICES:-}"; gpu="${gpu%%,*}"; gpu="${gpu:-0}"   # first visible GPU id (or 0)
dest="${model}_${host}_gpu${gpu}"                                   # TE-dashboard results/<...> dir name

# Per-run manifest (versions + machine context) archived alongside the CSVs.
_pyver() { python -c "import ${1} as _m; print(getattr(_m, '__version__', '?'))" 2>/dev/null || echo '?'; }
{
  echo "date:       $(date -Is)"
  echo "host:       ${host}"
  echo "gpu_id:     ${gpu}"
  echo "arch:       ${model}"
  echo "te_commit:  ${sha}"
  echo "te:         $(_pyver transformer_engine)"
  echo "pytorch:    $(_pyver torch)"
  echo "triton:     $(_pyver triton)"
  echo "jax:        $(_pyver jax)"
  echo "rocm:       $(cat /opt/rocm/.info/version 2>/dev/null || python -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo '?')"
  echo "amdgpu_drv: $(cat /sys/module/amdgpu/version 2>/dev/null || echo '?')"
  echo "kernel:     $(uname -r)"
} > "${run_dir}/run_info.txt"

kp=""; [[ "${KERNEL_PROFILE}" == "1" ]] && kp="--kernel-profile"
fly=""; [[ "${RUN_FLYDSL}" == "1" ]] && fly="--run-flydsl"
smp=""; [[ "${SAMPLES}" == "1" ]] && smp="--csv-samples"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

rm -f "${MB_DIR}"/benchmark_*.csv          # clear stale generated CSVs (gitignored)
read -r -a files <<< "${BENCH_FILES}"
for f in "${files[@]}"; do
  echo "== ${f} =="
  ( cd "${MB_DIR}" && python "${f}" -v --csv ${kp} ${fly} ${smp} ) \
    || echo "  (${f} returned non-zero; continuing)"
done

# Collect outputs: summary CSVs get run_week/commit_sha/commit_date and feed the
# ingest; per-iteration samples (benchmark_*_samples.csv) are archived untouched
# under samples/ (not ingested yet -- kept for future swarm/violin + A/B use).
shopt -s nullglob
for csv in "${MB_DIR}"/benchmark_*.csv; do
  base="$(basename "${csv}")"
  if [[ "${base}" == *_samples.csv ]]; then
    mkdir -p "${run_dir}/samples"
    mv "${csv}" "${run_dir}/samples/${base}"
    echo "  -> ${run_dir}/samples/${base}"
    continue
  fi
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
echo "Run manifest:  ${run_dir}/run_info.txt"
[[ "${SAMPLES}" == "1" ]] && echo "Per-iteration samples archived in ${run_dir}/samples/ (not ingested yet)"
echo "Publish in a TE-dashboard checkout (github.com/AMD-ROCm-Internal/TE-dashboard):"
echo "  cp -r ${run_dir} <TE-dashboard>/results/${dest}/"
echo "  python dashboard_ingest.py results/${dest}/$(basename "${run_dir}")/benchmark_*.csv --model ${model}"
echo "  python build_bundle.py"
