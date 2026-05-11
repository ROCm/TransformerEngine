#!/usr/bin/env bash
# Helper script for common benchmark tasks.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BENCH_DIR="benchmarks/microbench"

usage() {
    cat <<EOF
Usage: bash benchmarks/microbench/run_benchmarks.sh <command> [options]

Commands:
  run [-w W] [-n N] [SUITE] [METHOD]
                        Run benchmarks in-process (writes long-format CSV to
                        benchmarks/.bench-results/<machine>/<commit>.csv)
  list                  List available benchmark suites

EOF
}

case "${1:-}" in
    run)
        shift
        if [[ $# -eq 0 ]]; then
            python "$BENCH_DIR/driver.py" --all
        else
            python "$BENCH_DIR/driver.py" "$@"
        fi
        ;;
    list)
        echo "Available benchmark suites:"
        ls "$BENCH_DIR"/bench_*.py 2>/dev/null | sed 's|.*/bench_|  bench_|;s|\.py$||'
        ;;
    *)
        usage
        exit 1
        ;;
esac
