#!/usr/bin/env bash
# Helper script for common benchmark tasks.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BENCH_DIR="benchmarks/asv"
ASV_CONF="$(pwd)/$BENCH_DIR/asv.conf.json"

usage() {
    cat <<EOF
Usage: bash benchmarks/asv/run_benchmarks.sh <command> [options]

Commands:
  run [-w W] [-n N] [SUITE] [METHOD]
                        Run benchmarks in-process (saves ASV-compatible results)
  view                  Build the ASV HTML dashboard from saved results and serve it
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
    view)
        asv publish --config "$ASV_CONF"
        echo "Starting preview server at http://localhost:8080"
        asv preview --config "$ASV_CONF"
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
