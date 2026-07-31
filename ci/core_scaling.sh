#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Measure how ci/core.sh scales with ctest job count and OMP thread count.
#
# ci/core.sh hardcodes n_parallel_jobs=4 and derives OMP_NUM_THREADS from it as
# physical_cores/n_parallel_jobs. That 4 is a CPU-partitioning number -- it makes
# jobs*threads equal the core count -- not a measured optimum; it was raised to 8
# once (PR #426) and came back to 4 as part of an unrelated revert (PR #440),
# never having been evaluated with core pinned to a single GPU. This script
# answers the question that revert skipped:
#
#   A  current policy, OMP = cores/j: does raising j help under the coupling?
#   B  decoupled, OMP pinned to 8:    is the coupling what limits j?
#   C  j fixed at 4, OMP swept:       do these tests use OMP threads at all?
#
# The gtest suites are registered one-ctest-test-per-case (gtest_discover_tests),
# so the workload is dominated by process startup rather than GPU work. If C
# shows a flat curve, core's CPU budget is being spent on threads it does not use
# instead of on concurrency it does.
#
# Usage:  [GPU=0] [FILTER='-E GEMMTestSuite'] ci/core_scaling.sh
#
#   GPU     device to pin every run to. Keep this set -- leaving all devices
#           visible is what made PR #426 pile every job onto device 0.
#   FILTER  ctest selection, passed through verbatim. Defaults to the non-GEMM
#           half (the larger one, ~6.4k tests). Use '-R GEMMTestSuite' for the
#           other half, or '' for everything.
#
# Run on an idle machine: a concurrent CI job invalidates every number here.
# Results are appended to /tmp/core_scaling.tsv; per-config ctest output stays in
# /tmp/core_run.<name>.log.

set -u

GPU=${GPU:-0}
FILTER=${FILTER:--E GEMMTestSuite}
OUT=${OUT:-/tmp/core_scaling.tsv}

DIR=$(cd "$(dirname "$0")" && pwd)
TEST_DIR=${TE_PATH:-$(cd "$DIR/.." && pwd)/}tests/cpp

cd "$TEST_DIR" || { echo "cannot cd to $TEST_DIR" >&2; exit 1; }

CORES=$(( $(lscpu | awk '/^CPU\(s\):/{print $2}') \
        / $(lscpu | awk '/Thread\(s\) per core:/{print $NF}') ))

# ---------------------------------------------------------------------------
# Preflight. The previous version of this script reported 0.0s for every config
# because ctest exited immediately and its output was discarded. Refuse to
# produce a number unless tests were actually discovered.
if [ ! -d build ]; then
    echo "== configuring + building (one time, 10-20 min) =="
    if ! cmake -GNinja -Bbuild . > /tmp/core_cfg.log 2>&1; then
        tail -30 /tmp/core_cfg.log >&2; exit 1
    fi
    if ! cmake --build build >> /tmp/core_cfg.log 2>&1; then
        tail -30 /tmp/core_cfg.log >&2; exit 1
    fi
fi

NTESTS=$(HIP_VISIBLE_DEVICES=$GPU ctest --test-dir build -N $FILTER 2>/dev/null \
         | awk '/Total Tests:/{print $3}')
if [ -z "$NTESTS" ] || [ "$NTESTS" -eq 0 ]; then
    echo "ABORT: ctest discovered no tests in $TEST_DIR/build" >&2
    echo "A build configured for a different container path will do this;" >&2
    echo "remove build/ and re-run to reconfigure. Raw ctest output:" >&2
    HIP_VISIBLE_DEVICES=$GPU ctest --test-dir build -N $FILTER 2>&1 | tail -20 >&2
    exit 1
fi

echo "cores=$CORES  gpu=$GPU  filter=${FILTER:-<all>}  discovered=$NTESTS tests"

# ---------------------------------------------------------------------------
run() {
    _name=$1; _jobs=$2; _omp=$3
    _log=/tmp/core_run.${_name}.log
    _smi=/tmp/core_smi.${_name}

    # Sample utilisation and VRAM alongside the run. Peak VRAM is the ceiling on
    # how far j can be raised regardless of what the wall clock says.
    ( while :; do
          rocm-smi --showuse --showmeminfo vram --csv 2>/dev/null \
              | awk -F, -v g="card$GPU" '$1 ~ g {print $2"\t"$4}'
          sleep 2
      done ) > "$_smi" 2>/dev/null &
    _smi_pid=$!

    _t0=$(date +%s)
    OMP_NUM_THREADS=$_omp HIP_VISIBLE_DEVICES=$GPU \
        ctest --test-dir build -j"$_jobs" --timeout 600 $FILTER > "$_log" 2>&1
    _rc=$?
    _t1=$(date +%s)
    kill $_smi_pid 2>/dev/null
    wait $_smi_pid 2>/dev/null

    _secs=$((_t1 - _t0))
    _passed=$(awk '/tests passed/{print $1}' "$_log" | tail -1)
    _gpu=$(awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n}' "$_smi")
    _vram=$(awk '{if ($2 > m) m = $2} END {printf "%.0f", m/1048576}' "$_smi")

    printf '%-9s j=%-3s omp=%-3s %6ss  rc=%-3s %-16s gpu=%3s%%  vram=%sMB\n' \
        "$_name" "$_jobs" "$_omp" "$_secs" "$_rc" "${_passed:-NO-SUMMARY}" \
        "$_gpu" "$_vram" | tee -a "$OUT"

    # A config that finishes implausibly fast did not run the suite. Say so
    # rather than folding a zero into the comparison.
    if [ "$_secs" -lt 30 ]; then
        echo "  !! suspiciously fast -- tail of $_log:" | tee -a "$OUT"
        tail -5 "$_log" | sed 's/^/     /' | tee -a "$OUT"
    fi
}

: > "$OUT"

echo "=== A: current policy (OMP = cores/j) ===" | tee -a "$OUT"
for j in 4 8 16 32; do
    run "A_j$j" $j $((CORES / j))
done

echo "=== B: decoupled (OMP pinned to 8) ===" | tee -a "$OUT"
for j in 8 16 32; do
    run "B_j$j" $j 8
done

echo "=== C: is OMP used at all? (j=4) ===" | tee -a "$OUT"
for o in 32 8 1; do
    run "C_omp$o" 4 $o
done

echo
echo "================ results: $OUT ================"
cat "$OUT"
