#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Run every microbenchmark.
#
#   ./run_all.sh                  # defaults (rotation on)
#   ./run_all.sh --rotating 512   # rotate within a 512 MB budget
#   ./run_all.sh --no-rotating    # disable input rotation
#   ./run_all.sh --csv            # also write per-benchmark CSVs
#
# Set PYTHON to pick a specific interpreter (default: python).

shopt -s nullglob

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

failed=()
for bench in benchmark_*.py; do
    echo
    echo "############################################################"
    echo "# ${bench} $*"
    echo "############################################################"
    if ! "$PYTHON" "$bench" "$@"; then
        echo "!!! ${bench} FAILED" >&2
        failed+=("$bench")
    fi
done

echo
if (( ${#failed[@]} )); then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "All benchmarks completed."
