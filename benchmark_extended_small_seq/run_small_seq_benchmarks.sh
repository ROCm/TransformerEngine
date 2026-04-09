#!/bin/bash
#
# Benchmark runner for Sciforium small-sequence attention scenarios.
#
# Covers all three in-scope scenarios in a single sweep:
#   Scenario 1: sq <= 16, skv <= 16 (self-attention, causal, no padding)
#   Scenario 3: sq = 16, skv = 16 (subset of scenario 1)
#   Scenario 4: sq = 17, skv = 17 (self-attention, causal, no padding)
#
# Backends: JAX unfused attention (reference) and TE/CK fused attention (current)
#
# Usage:
#   bash run_small_seq_benchmarks.sh [--repeats N] [--warmups N] \
#       [--batch-sizes "2048 4096"] [--dims "64 128 256"] [--layouts "bshd"] \
#       [--dtypes "bfloat16"]
#
# Environment variable overrides:
#   REPEATS, WARMUPS, BATCH_SIZES, NHEADS, DIMS, DTYPES, LAYOUTS, GQA_RATIOS, MODES, KERNELS, SEQLENS
#
# One CSV per (layout, batch_size, head_dim, dtype):
#   results/small_seq_sweep_<LAYOUT>_bs_<BATCH>_hd_<DIM>_dt_<DTYPE>_scenario_1_3_4.csv
# Example: small_seq_sweep_bshd_bs_2048_hd_64_dt_bfloat16_scenario_1_3_4.csv
#
# Defaults: BATCH_SIZES=2048 4096, DIMS=64 128 256, LAYOUTS=bshd, DTYPES=bfloat16

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPEATS="${REPEATS:-25}"
WARMUPS="${WARMUPS:-5}"
BATCH_SIZES="${BATCH_SIZES:-2048 4096}"
NHEADS="${NHEADS:-32}"
DIMS="${DIMS:-64 128 256}"
DTYPES="${DTYPES:-bfloat16}"
LAYOUTS="${LAYOUTS:-bshd}"
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
        --dtypes)       DTYPES="$2"; shift 2 ;;
        --layouts)      LAYOUTS="$2"; shift 2 ;;
        --modes)        MODES="$2"; shift 2 ;;
        --kernels)      KERNELS="$2"; shift 2 ;;
        --seqlens)      SEQLENS="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Small-Sequence Attention Benchmarks"
echo "============================================================"
echo "Kernels:      $KERNELS"
echo "Layouts:      $LAYOUTS"
echo "Dtypes:       $DTYPES"
echo "Seq lengths:  $SEQLENS"
echo "Batch sizes:  $BATCH_SIZES"
echo "Heads:        $NHEADS"
echo "Head dims:    $DIMS"
echo "GQA ratios:   $GQA_RATIOS"
echo "Modes:        $MODES"
echo "Repeats:      $REPEATS"
echo "Warmups:      $WARMUPS"
echo "Output:       results/small_seq_sweep_<LAYOUT>_bs_<BATCH>_hd_<DIM>_dt_<DTYPE>_scenario_1_3_4.csv"
echo "============================================================"
echo ""

for LAYOUT in $LAYOUTS; do
    for DT in $DTYPES; do
        for DIM in $DIMS; do
            for BSZ in $BATCH_SIZES; do
                CSV="$RESULTS_DIR/small_seq_sweep_${LAYOUT}_bs_${BSZ}_hd_${DIM}_dt_${DT}_scenario_1_3_4.csv"
                echo "============================================================"
                echo "layout=$LAYOUT  dtype=$DT  batch_size=$BSZ  head_dim=$DIM"
                echo "  -> $CSV"
                echo "============================================================"

                COMMON_ARGS="--kernel-names $KERNELS \
                    --repeats $REPEATS \
                    --warmups $WARMUPS \
                    --nheads $NHEADS \
                    --dims $DIM \
                    --batch-sizes $BSZ \
                    --dtypes $DT \
                    --gqa-ratios $GQA_RATIOS \
                    --modes $MODES \
                    --layouts $LAYOUT \
                    --nr-segments 1"

                for SEQ in $SEQLENS; do
                    echo "  sq=skv=$SEQ"
                    python fa_profiling.py \
                        $COMMON_ARGS \
                        --seqlens-q "$SEQ" \
                        --seqlens-kv "$SEQ" \
                        --csv "$CSV"
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Complete. CSV files (one per layout x dtype x batch x head_dim):"
for LAYOUT in $LAYOUTS; do
    for DT in $DTYPES; do
        for DIM in $DIMS; do
            for BSZ in $BATCH_SIZES; do
                echo "  $RESULTS_DIR/small_seq_sweep_${LAYOUT}_bs_${BSZ}_hd_${DIM}_dt_${DT}_scenario_1_3_4.csv"
            done
        done
    done
done
echo ""
echo "View results (example):"
echo "  column -s, -t $RESULTS_DIR/small_seq_sweep_bshd_bs_2048_hd_128_dt_bfloat16_scenario_1_3_4.csv"
echo "============================================================"
