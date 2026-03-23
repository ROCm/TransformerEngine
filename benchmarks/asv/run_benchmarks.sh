#!/usr/bin/env bash
# Helper script for common ASV benchmark tasks.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BENCH_DIR="benchmarks/asv"
ASV_CONF="$(pwd)/$BENCH_DIR/asv.conf.json"
mapfile -t SUITES < <(find "$BENCH_DIR" -maxdepth 1 -name 'bench_*.py' -printf '%f\n' | sed 's/\.py$//' | sort)

usage() {
    cat <<EOF
Usage: bash benchmarks/asv/run_benchmarks.sh <command> [options]

Commands:
  setup                 Register this machine with ASV
  run [-w W] [-n N] [SUITE] [METHOD]
                        Run benchmarks in-process (fast, saves ASV-compatible results)
  run --asv [SUITE]     Run benchmarks via ASV (subprocess isolation per benchmark)
  compare [REF] [NEW]   Compare two commits (default: HEAD~1 vs HEAD)
  view                  Generate HTML dashboard and open preview server
  list                  List available benchmark suites

EOF
}

case "${1:-}" in
    setup)
        MACHINE="${2:-$(hostname)}"
        echo "Registering machine as: $MACHINE"
        asv machine --yes --machine "$MACHINE" --config "$ASV_CONF"
        ;;
    run)
        shift
        if [[ "${1:-}" == "--asv" ]]; then
            shift
            CMD=(asv run --config "$ASV_CONF" --python=same --launch-method spawn
                 --set-commit-hash "$(git rev-parse HEAD)")
            [[ -n "${1:-}" ]] && CMD+=(--bench "$1")
            echo "Running (asv): ${CMD[*]}"
            "${CMD[@]}"
        else
            # Default: fast in-process run
            ARGS=()
            while [[ $# -gt 0 ]]; do
                ARGS+=("$1")
                shift
            done
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                # Run all suites
                for s in "${SUITES[@]}"; do
                    python "$BENCH_DIR/driver.py" "$s"
                done
            else
                python "$BENCH_DIR/driver.py" "${ARGS[@]}"
            fi
        fi
        ;;
    compare)
        REF="${2:-HEAD~1}"
        NEW="${3:-HEAD}"
        echo "Comparing $REF vs $NEW"
        asv continuous --config "$ASV_CONF" --python=same --launch-method spawn "$REF" "$NEW"
        ;;
    view)
        asv publish --config "$ASV_CONF"
        echo "Starting preview server at http://localhost:8080"
        asv preview --config "$ASV_CONF"
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
