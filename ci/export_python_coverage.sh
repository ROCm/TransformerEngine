#!/bin/sh
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Combine coverage.py shards from pytest_run and write coverage.json.
#
# This is execution coverage of the installed transformer_engine package
# (which Python functions/lines ran). It is not JUnit: pass/fail of test
# cases stays in the XML artifacts.
#
# Never fails the CI job. Never writes an empty stub — a missing artifact
# is the honest signal that coverage was not produced.

OUT_DIR="${1:-/workspace/python-coverage}"
mkdir -p "$OUT_DIR" || exit 0

if ! command -v coverage >/dev/null 2>&1; then
    echo "coverage is not installed; skipping Python coverage export"
    exit 0
fi

export COVERAGE_FILE="${COVERAGE_FILE:-$OUT_DIR/.coverage}"
JSON="$OUT_DIR/coverage.json"
META="$OUT_DIR/coverage-meta.txt"
rm -f "$JSON" "$META"

coverage combine || true
if ! coverage json -o "$JSON"; then
    echo "coverage json failed; not uploading a stub"
    rm -f "$JSON"
    exit 0
fi

if [ ! -s "$JSON" ]; then
    echo "coverage.json is empty; not uploading a stub"
    rm -f "$JSON"
    exit 0
fi

if ! python3 - "$JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    document = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError, UnicodeError):
    raise SystemExit(1)
if not document.get("files"):
    raise SystemExit(1)
PY
then
    echo "coverage.json has no measured files; not uploading a stub"
    rm -f "$JSON"
    exit 0
fi

COMMIT="unknown"
if command -v git >/dev/null 2>&1; then
    COMMIT=$(git -C "${TE_PATH:-/workspace}" rev-parse HEAD 2>/dev/null || echo unknown)
fi

{
    echo "commit=$COMMIT"
    echo "format=coverage.py JSON"
    echo "source=transformer_engine (installed package measured by CI pytest)"
    echo "not=JUnit pass/fail of test cases"
    echo "not=C++/HIP kernels (.cu/.hip); those need llvm-cov"
    echo "not=torchrun/mpirun child processes"
} > "$META"

echo "Wrote $JSON (commit=$COMMIT)"
exit 0
