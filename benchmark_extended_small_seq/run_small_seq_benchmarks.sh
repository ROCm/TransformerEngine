#!/bin/bash
#
# Benchmark runner for Sciforium small-sequence attention scenarios.
#
# Covers all three in-scope scenarios in a single sweep with BSHD layout: 
#   Scenario 1: sq <= 16, skv <= 16 (self-attention, causal, no padding)
#   Scenario 3: sq = 16, skv = 16 (subset of scenario 1)
#   Scenario 4: sq = 17, skv = 17 (self-attention, causal, no padding)
#
# Backends: JAX unfused attention (reference) and TE/CK fused attention (current)
#
# Usage:
#   bash run_small_seq_benchmarks.sh [--repeats N] [--warmups N] [--batch-sizes "2048 4096"]
#
# Environment variable overrides:
#   REPEATS, WARMUPS, BATCH_SIZES, NHEADS, DIMS, GQA_RATIOS, MODES, KERNELS
#
# Output: CSV file in results/ directory

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPEATS="${REPEATS:-25}"
WARMUPS="${WARMUPS:-5}"
BATCH_SIZES="${BATCH_SIZES:-2048 4096}"
NHEADS="${NHEADS:-32}"
DIMS="${DIMS:-128}"
GQA_RATIOS="${GQA_RATIOS:-1}"
MODES="${MODES:-fwd bwd}"
KERNELS="${KERNELS:-jax te}"
SEQLENS="${SEQLENS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats)      REPEATS="$2"; shift 2 ;;
        --warmups)      WARMUPS="$2"; shift 2 ;;
        --batch-sizes)  BATCH_SIZES="$2"; shift 2 ;;
        --nheads)       NHEADS="$2"; shift 2 ;;
        --dims)         DIMS="$2"; shift 2 ;;
        --modes)        MODES="$2"; shift 2 ;;
        --kernels)      KERNELS="$2"; shift 2 ;;
        --seqlens)      SEQLENS="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV="$RESULTS_DIR/small_seq_sweep_bshd_scenario_1_3_4.csv"

echo "============================================================"
echo "Small-Sequence Attention Benchmarks"
echo "============================================================"
echo "Kernels:      $KERNELS"
echo "Seq lengths:  $SEQLENS"
echo "Batch sizes:  $BATCH_SIZES"
echo "Heads:        $NHEADS"
echo "Dim:          $DIMS"
echo "GQA ratios:   $GQA_RATIOS"
echo "Modes:        $MODES"
echo "Repeats:      $REPEATS"
echo "Warmups:      $WARMUPS"
echo "Output:       $CSV"
echo "============================================================"
echo ""

COMMON_ARGS="--kernel-names $KERNELS \
    --repeats $REPEATS \
    --warmups $WARMUPS \
    --nheads $NHEADS \
    --dims $DIMS \
    --gqa-ratios $GQA_RATIOS \
    --modes $MODES \
    --layouts bshd \
    --nr-segments 1"

for BSZ in $BATCH_SIZES; do
    echo "--- batch_size=$BSZ ---"
    for SEQ in $SEQLENS; do
        echo "  sq=skv=$SEQ"
        python fa_profiling.py \
            $COMMON_ARGS \
            --batch-sizes "$BSZ" \
            --seqlens-q "$SEQ" \
            --seqlens-kv "$SEQ" \
            --csv "$CSV"
    done
done

echo ""
echo "============================================================"
echo "Complete. Results: $CSV"
echo ""
echo "View results:"
echo "  column -s, -t < $CSV"
echo "============================================================"
