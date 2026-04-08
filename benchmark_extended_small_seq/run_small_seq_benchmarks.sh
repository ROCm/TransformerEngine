#!/bin/bash
#
# Benchmark runner for Sciforium small-sequence attention scenarios.
#
# Scenarios:
#   1: sq <= 16, skv <= 16, self-attention (causal), no padding (initial baseline)
#   3: sq = 16, skv = 16, self-attention (causal), fixed length, no padding
#   4: sq = 17, skv = 17, self-attention (causal), fixed length, no padding
#
# Backends: JAX unfused attention (reference) and TE/CK fused attention (current)
#
# Usage:
#   bash run_small_seq_benchmarks.sh [--repeats N] [--warmups N] [--batch-sizes "1 4 8"]
#
# Environment variable overrides:
#   REPEATS, WARMUPS, BATCH_SIZES, NHEADS, DIMS, GQA_RATIOS, MODES, KERNELS
#
# Output: CSV files in results/ directory

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPEATS="${REPEATS:-25}"
WARMUPS="${WARMUPS:-5}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8}"
NHEADS="${NHEADS:-32}"
DIMS="${DIMS:-128}"
GQA_RATIOS="${GQA_RATIOS:-1}"
MODES="${MODES:-fwd bwd}"
KERNELS="${KERNELS:-jax te}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats)      REPEATS="$2"; shift 2 ;;
        --warmups)      WARMUPS="$2"; shift 2 ;;
        --batch-sizes)  BATCH_SIZES="$2"; shift 2 ;;
        --nheads)       NHEADS="$2"; shift 2 ;;
        --dims)         DIMS="$2"; shift 2 ;;
        --modes)        MODES="$2"; shift 2 ;;
        --kernels)      KERNELS="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Small-Sequence Attention Benchmarks"
echo "============================================================"
echo "Kernels:      $KERNELS"
echo "Batch sizes:  $BATCH_SIZES"
echo "Heads:        $NHEADS"
echo "Dim:          $DIMS"
echo "GQA ratios:   $GQA_RATIOS"
echo "Modes:        $MODES"
echo "Repeats:      $REPEATS"
echo "Warmups:      $WARMUPS"
echo "Results dir:  $RESULTS_DIR"
echo "============================================================"
echo ""

COMMON_ARGS="--kernel-names $KERNELS \
    --repeats $REPEATS \
    --warmups $WARMUPS \
    --batch-sizes $BATCH_SIZES \
    --nheads $NHEADS \
    --dims $DIMS \
    --gqa-ratios $GQA_RATIOS \
    --modes $MODES \
    --layouts bshd \
    --nr-segments 1"

# =====================================================================
# Scenario 3: sq=16, skv=16, causal, fixed length, no padding
# =====================================================================
echo "[Scenario 3] sq=16, skv=16 -- fixed length, causal, no padding"
CSV_S3="$RESULTS_DIR/scenario3_sq16_skv16_${TIMESTAMP}.csv"

python fa_profiling.py \
    $COMMON_ARGS \
    --seqlens-q 16 \
    --seqlens-kv 16 \
    --csv "$CSV_S3"

echo "  -> Results written to $CSV_S3"
echo ""

# =====================================================================
# Scenario 4: sq=17, skv=17, causal, fixed length, no padding
# HIGH PRIORITY -- 17 is not aligned to 4x4 or 16x16 matrix-multiply
# tiles, requiring extra handling in the CK kernel.
# =====================================================================
echo "[Scenario 4] sq=17, skv=17 -- fixed length, causal, no padding (HIGH PRIORITY)"
CSV_S4="$RESULTS_DIR/scenario4_sq17_skv17_${TIMESTAMP}.csv"

python fa_profiling.py \
    $COMMON_ARGS \
    --seqlens-q 17 \
    --seqlens-kv 17 \
    --csv "$CSV_S4"

echo "  -> Results written to $CSV_S4"
echo ""

# =====================================================================
# Scenario 1: sq <= 16, skv <= 16, self-attention (sq == skv), causal,
# no padding (initial baseline -- padding/varlen deferred).
#
# fa_profiling.py generates a Cartesian product of --seqlens-q and
# --seqlens-kv, so we loop with matched pairs to enforce sq == skv.
# =====================================================================
echo "[Scenario 1] sq=skv sweep {1..16} -- causal, no padding"
CSV_S1="$RESULTS_DIR/scenario1_small_seq_sweep_${TIMESTAMP}.csv"

for SEQ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
    echo "  sq=skv=$SEQ"
    python fa_profiling.py \
        $COMMON_ARGS \
        --seqlens-q "$SEQ" \
        --seqlens-kv "$SEQ" \
        --csv "$CSV_S1"
done

echo "  -> Results written to $CSV_S1"
echo ""

# =====================================================================
# Summary
# =====================================================================
echo "============================================================"
echo "All scenarios complete."
echo ""
echo "Results:"
echo "  Scenario 1 (sweep):  $CSV_S1"
echo "  Scenario 3 (16x16):  $CSV_S3"
echo "  Scenario 4 (17x17):  $CSV_S4"
echo ""
echo "View results:"
echo "  column -s, -t < $CSV_S3"
echo "============================================================"
