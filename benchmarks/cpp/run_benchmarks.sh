#!/bin/bash
# Builds benchmarks, runs them, and consolidates results into a single CSV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULTS_DIR="${SCRIPT_DIR}/results"

main() {
    echo "=== MXFP8 Benchmark Suite ==="

    echo -e "\n[1/3] Building benchmarks..."
    cd "${SCRIPT_DIR}"
    if ! cmake -GNinja -B"${BUILD_DIR}" . || ! cmake --build "${BUILD_DIR}"; then
        echo "Build failed. Fix the build errors and try again."
        return
    fi
    echo "Build complete"

    mkdir -p "${RESULTS_DIR}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_PREFIX="${RESULTS_DIR}/bench_${TIMESTAMP}"

    echo -e "\n[2/3] Running benchmarks..."

    BENCHMARKS=(
        "bench_quantize_mxfp8_fused"
        "bench_dequantize_mxfp8"
        "bench_gated_mxfp8"
        "bench_casttranspose"
    )

    FAILED_BENCHMARKS=()
    for bench in "${BENCHMARKS[@]}"; do
        if [ -f "${BUILD_DIR}/${bench}" ]; then
            echo "  Running ${bench}..."
            if "${BUILD_DIR}/${bench}" \
                --benchmark_out="${RESULT_PREFIX}_${bench}.csv" \
                --benchmark_out_format=csv \
                --benchmark_min_time=0.2s; then
                echo "  Saved to ${RESULT_PREFIX}_${bench}.csv"
            else
                echo "  FAILED: ${bench} (exit code $?), continuing..."
                FAILED_BENCHMARKS+=("${bench}")
            fi
        else
            echo "  ${bench} not found, skipping"
        fi
    done

    echo -e "\n[3/3] Consolidating results..."

    CONSOLIDATED_CSV="${RESULT_PREFIX}_all.csv"
    FIRST_CSV=$(ls "${RESULT_PREFIX}"_*.csv 2>/dev/null | grep -v "_all.csv" | head -1)

    if [ -z "$FIRST_CSV" ]; then
        echo "No CSV files found to consolidate"
        return
    fi

    head -1 "$FIRST_CSV" > "$CONSOLIDATED_CSV"

    for csv in "${RESULT_PREFIX}"_bench_*.csv; do
        if [ "$csv" != "$CONSOLIDATED_CSV" ]; then
            tail -n +2 "$csv" >> "$CONSOLIDATED_CSV"
        fi
    done

    echo "Consolidated CSV: ${CONSOLIDATED_CSV}"

    echo -e "\n=== Summary ==="
    TOTAL_ROWS=$(tail -n +2 "$CONSOLIDATED_CSV" | wc -l)
    echo "Total benchmarks: $TOTAL_ROWS"
    echo "Results saved to: ${RESULTS_DIR}/"
    echo ""
    echo "Files created:"
    for bench in "${BENCHMARKS[@]}"; do
        if [ -f "${RESULT_PREFIX}_${bench}.csv" ]; then
            echo "  - $(basename "${RESULT_PREFIX}_${bench}.csv")"
        fi
    done
    echo "  - $(basename "$CONSOLIDATED_CSV") (consolidated)"
    echo ""

    if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
        echo "Failed benchmarks:"
        for bench in "${FAILED_BENCHMARKS[@]}"; do
            echo "  ${bench}"
        done
        echo ""
    fi
}

main
