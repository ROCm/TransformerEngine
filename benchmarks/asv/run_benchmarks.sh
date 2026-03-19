#!/usr/bin/env bash
# Helper script for common ASV benchmark tasks.
# Run from the repository root (where asv.conf.json lives).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BENCH_DIR="benchmarks/asv"
mapfile -t SUITES < <(find "$BENCH_DIR" -maxdepth 1 -name 'bench_*.py' -printf '%f\n' | sed 's/\.py$//' | sort)

usage() {
    cat <<EOF
Usage: bash benchmarks/asv/run_benchmarks.sh <command> [options]

Commands:
  setup                 Register this machine with ASV
  run [SUITE]           Run all benchmarks, or a single suite (e.g. bench_casting)
  quick [SUITE]         Smoke-test run (single iteration, results not saved)
  direct [-w W] [-n N] SUITE [METHOD]
                      Fast in-process run (no subprocesses, no ASV overhead)
  compare [REF] [NEW]   Compare two commits (default: HEAD~1 vs HEAD)
  view                  Generate HTML dashboard and open preview server
  list                  List available benchmark suites

EOF
}

case "${1:-}" in
    setup)
        MACHINE="${2:-$(hostname)}"
        echo "Registering machine as: $MACHINE"
        asv machine --yes --machine "$MACHINE"
        ;;
    run)
        CMD=(asv run --python=same --launch-method spawn
             --set-commit-hash "$(git rev-parse HEAD)")
        [[ -n "${2:-}" ]] && CMD+=(--bench "$2")
        echo "Running: ${CMD[*]}"
        "${CMD[@]}"
        ;;
    quick)
        CMD=(asv run --python=same --launch-method spawn --quick
             --set-commit-hash "$(git rev-parse HEAD)")
        [[ -n "${2:-}" ]] && CMD+=(--bench "$2")
        echo "Running (quick): ${CMD[*]}"
        "${CMD[@]}"
        ;;
    direct)
        shift
        if [[ $# -eq 0 ]]; then
            echo "Usage: $0 direct [options] SUITE [METHOD]"
            echo "Options: -w WARMUP  -n ITERS"
            exit 1
        fi
        python "$BENCH_DIR/direct_run.py" "$@"
        ;;
    compare)
        REF="${2:-HEAD~1}"
        NEW="${3:-HEAD}"
        echo "Comparing $REF vs $NEW"
        asv continuous --python=same --launch-method spawn "$REF" "$NEW"
        ;;
    view)
        asv publish
        echo "Starting preview server at http://localhost:8080"
        asv preview
        ;;
    list)
        echo "Available benchmark suites:"
        for s in "${SUITES[@]}"; do echo "  $s"; done
        ;;
    *)
        usage
        exit 1
        ;;
esac
