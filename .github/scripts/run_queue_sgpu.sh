#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run all sGPU test suites as one global work queue across N GPUs.
#
# Usage: run_queue_sgpu.sh [options] <config>...
#   -n, --gpus <n>        number of GPUs / concurrent workers (default: 4)
#   -g, --first-gpu <n>   first GPU id to use (default: 0)
#   -l, --log-dir <dir>   where suite and per-item logs are written
#   -w, --weights <file>  "<label>/<tag> <weight>" pairs used to order the queue
#       --only <regex>    keep only items whose "<label>/<tag>" matches (smoke
#                         tests, or re-running just the items that failed)
#       --skip-setup      container prerequisites and the CK JIT cache are
#                         already in place; go straight to the queue
#       --dry-run         build and print the queue, run nothing
#
# Why a queue instead of one suite per GPU (run_parallel_sgpu.sh): the suites
# are very unevenly sized, so a static 1:1 map leaves most GPUs idle waiting for
# the largest suite. Here every suite is expanded into individual work items,
# the items are ordered longest-first, and each worker pulls the next one when
# it goes idle -- so a GPU only stops working when the queue is empty.
#
# Compatibility: after the queue drains, each suite's item logs are concatenated
# into <log-dir>/<logfile> and the worst item exit code is written to
# <logfile>.rc, so existing consumers of those paths keep working unchanged.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_GPUS=${TEST_NUM_GPUS:-4}
FIRST_GPU=${TEST_FIRST_GPU:-0}
LOG_DIR=${LOG_DIR:-/tmp/te_ci_logs}
WEIGHTS_FILE=""
DRY_RUN=""
ONLY_RE=""
SKIP_SETUP=""
# Items with no recorded weight sort first: an unknown item is more likely to be
# a new (or newly slow) one, and a long item started late is what stretches the
# tail. Losing the gamble costs far less than mis-scheduling a genuinely big item.
DEFAULT_WEIGHT=${TE_CI_DEFAULT_WEIGHT:-999999}
# Ceiling for the per-item deadline derived from the weight below. Also the
# deadline for unknown items, whose DEFAULT_WEIGHT makes the derived budget
# meaningless.
MAX_ITEM_SECS=${TE_CI_MAX_ITEM_SECS:-7200}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--gpus)       NUM_GPUS="$2"; shift 2 ;;
        --gpus=*)        NUM_GPUS="${1#*=}"; shift ;;
        -g|--first-gpu)  FIRST_GPU="$2"; shift 2 ;;
        --first-gpu=*)   FIRST_GPU="${1#*=}"; shift ;;
        -l|--log-dir)    LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)     LOG_DIR="${1#*=}"; shift ;;
        -w|--weights)    WEIGHTS_FILE="$2"; shift 2 ;;
        --weights=*)     WEIGHTS_FILE="${1#*=}"; shift ;;
        --only)          ONLY_RE="$2"; shift 2 ;;
        --only=*)        ONLY_RE="${1#*=}"; shift ;;
        --skip-setup)    SKIP_SETUP=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -*)              echo "Unknown option: $1" >&2; exit 1 ;;
        *)               break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: at least one config file is required." >&2
    exit 1
fi

resolved_configs=()
for c in "$@"; do resolved_configs+=( "$(realpath -m "$c")" ); done
[[ -n "$WEIGHTS_FILE" ]] && WEIGHTS_FILE="$(realpath -m "$WEIGHTS_FILE")"
[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(realpath -m "$LOG_DIR")"

ITEM_LOG_DIR="$LOG_DIR/items"
mkdir -p "$ITEM_LOG_DIR"
cd "$REPO_ROOT" || { echo "Error: cannot cd to '${REPO_ROOT}'" >&2; exit 1; }

QUEUE_FILE="$LOG_DIR/queue.tsv"
IDX_FILE="$LOG_DIR/queue.idx"
LOCK_FILE="$LOG_DIR/queue.lock"
: > "$QUEUE_FILE"

# ---------------------------------------------------------------------------
# Phase 1: expand every suite into work items
#
# List mode runs the suite script with TE_CI_LIST_ONLY=1, which makes pytest_run
# echo "TE_CI_ITEM <tag>" for each invocation it would have made instead of
# running it. Level gating, backend matrix and capability probes all still apply,
# so the emitted list is exactly what the suite would have executed.
echo "== Expanding suites into work items =="
declare -a SUITE_LABELS SUITE_LOGFILES
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r label logfile mode cmd rest <<< "$line"
    SUITE_LABELS+=( "$label" )
    SUITE_LOGFILES+=( "$logfile" )
    if [[ "$mode" == "list" ]]; then
        mapfile -t tags < <(TE_CI_LIST_ONLY=1 "$cmd" ${rest:-} 2>/dev/null \
                            | sed -n 's/^TE_CI_ITEM //p')
        if [[ ${#tags[@]} -eq 0 ]]; then
            echo "::error::suite '${label}' (${cmd}) produced no work items" >&2
            exit 1
        fi
        for tag in "${tags[@]}"; do
            printf '%s\t%s\t%s\t%s\n' "$label" "$cmd" "$tag" "${rest:-}" >> "$QUEUE_FILE.raw"
        done
        echo "  ${label}: ${#tags[@]} items"
    else
        printf '%s\t%s\t%s\t%s\n' "$label" "$cmd" "" "${rest:-}" >> "$QUEUE_FILE.raw"
        echo "  ${label}: 1 item (opaque)"
    fi
done < <(cat "${resolved_configs[@]}")

# Duplicate tags within a suite would collide on both the JUnit XML filename and
# the TEST_FILTER dispatch (one invocation would run both). Fail loudly instead.
dupes=$(cut -f1,3 "$QUEUE_FILE.raw" | grep -v '^\S*\t$' | sort | uniq -d)
if [[ -n "$dupes" ]]; then
    echo "::error::duplicate work item tags detected:" >&2
    echo "$dupes" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: weight and order the queue (longest processing time first)
awk -v wf="$WEIGHTS_FILE" -v dw="$DEFAULT_WEIGHT" -v only="$ONLY_RE" '
    BEGIN {
        FS = OFS = "\t"
        if (wf != "" && (getline line < wf) >= 0) {
            do {
                if (line !~ /^[[:space:]]*#/ && line !~ /^[[:space:]]*$/) {
                    split(line, a, " "); if (a[1] != "") w[a[1]] = a[2]
                }
            } while ((getline line < wf) > 0)
        }
    }
    {
        key = $3 == "" ? $1 : $1 "/" $3
        if (only != "" && key !~ only) next
        # Truncate to an integer: the weight is also used in shell arithmetic to
        # derive the per-item deadline, and bash cannot parse "675.0".
        print (key in w ? int(w[key]) : dw), $0
    }
' "$QUEUE_FILE.raw" | sort -t$'\t' -k1,1nr > "$QUEUE_FILE"
rm -f "$QUEUE_FILE.raw"

if [[ ! -s "$QUEUE_FILE" ]]; then
    echo "::error::queue is empty${ONLY_RE:+ (no item matched --only '${ONLY_RE}')}" >&2
    exit 1
fi

TOTAL_ITEMS=$(wc -l < "$QUEUE_FILE")
echo "== Queue: ${TOTAL_ITEMS} items over ${NUM_GPUS} GPUs (first=${FIRST_GPU}) =="
if [[ -n "$WEIGHTS_FILE" && -f "$WEIGHTS_FILE" ]]; then
    echo "   weights: $WEIGHTS_FILE"
else
    echo "   weights: none (unordered -- run ci/build_weights.py after this run)"
fi

# The plan is printed on every run, not just --dry-run: it is the only record of
# what the scheduler intended, and comparing it against the schedule reported at
# the end is how a mis-weighted item is spotted.
echo "== Queue plan (dispatch order; est = weight used to sort) =="
awk -F'\t' -v dw="$DEFAULT_WEIGHT" -v n="$NUM_GPUS" '
    { total += $1
      printf "  %3d. est=%-8s %-9s %s\n", NR, ($1 == dw ? "unknown" : $1 "s"), $2,
             ($4 == "" ? "(whole suite)" : $4) }
    END {
        printf "\n  estimated work %ss over %s GPUs -> lower bound %.0fs", total, n, total / n
        printf " (largest item %ss)\n", first
    }
    NR == 1 { first = ($1 == dw ? "unknown" : $1) }
' "$QUEUE_FILE"

if [[ -n "$DRY_RUN" ]]; then
    echo "== Dry run: nothing executed =="
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 3: one-time, container-wide setup
#
# Prerequisites and the CK JIT blob cache are shared filesystem state, so they
# are built once here rather than by each of the N*items invocations. Running
# the suites sequentially also removes the race they previously had over the
# shared cache directory when launched in parallel.
if [[ -n "$SKIP_SETUP" ]]; then
    echo "== One-time suite setup: skipped (--skip-setup) =="
fi
for i in "${!SUITE_LABELS[@]}"; do
    [[ -n "$SKIP_SETUP" ]] && break
    [[ $i -eq 0 ]] && echo "== One-time suite setup =="
    label="${SUITE_LABELS[$i]}"
    # Columns are: weight, label, cmd, tag, rest. Only list-mode suites (those
    # with a non-empty tag) have a setup/dispatch split; opaque suites are a
    # single invocation that still does its own setup.
    cmd=$(awk -F'\t' -v l="$label" '$2==l && $4!="" {print $3; exit}' "$QUEUE_FILE")
    [[ -z "$cmd" ]] && continue
    echo "  setup: $label ($cmd)"
    if ! HIP_VISIBLE_DEVICES=$FIRST_GPU TE_CI_SETUP_ONLY=1 "$cmd" \
            > "$LOG_DIR/setup.${label}.log" 2>&1; then
        echo "::error::setup failed for ${label}; see $LOG_DIR/setup.${label}.log" >&2
        tail -30 "$LOG_DIR/setup.${label}.log" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Phase 4: run the queue
echo 1 > "$IDX_FILE"
: > "$LOCK_FILE"
START_TS=$(date +%s)

take_next() {
    local i
    {
        flock 9
        i=$(cat "$IDX_FILE")
        echo $((i + 1)) > "$IDX_FILE"
    } 9<>"$LOCK_FILE"
    echo "$i"
}

worker() {
    local gpu=$1
    local i line weight label cmd tag rest itemlog rc safetag junit_dir start end
    while :; do
        i=$(take_next)
        [[ "$i" -gt "$TOTAL_ITEMS" ]] && break
        line=$(sed -n "${i}p" "$QUEUE_FILE")
        [[ -z "$line" ]] && break
        IFS=$'\t' read -r weight label cmd tag rest <<< "$line"

        safetag="${tag:-whole}"
        itemlog="$ITEM_LOG_DIR/${label}.${safetag}.log"
        if [[ -n "${JUNITXML_PREFIX:-}${JUNITXML_SUFFIX:-}" ]]; then
            junit_dir="${JUNITXML_PREFIX:-}${label}/"
            mkdir -p "$junit_dir"
        else
            junit_dir=""
        fi

        # Bound the item at 3x its learned weight, plus slack for the ~5% noise
        # floor and a genuinely slower run; past that it is hung, not slow.
        # Nothing else bounds an *item*: pytest's --timeout is per test, so an
        # item of many just-under-PYTEST_TIMEOUT tests -- or a hang during
        # collection or conftest import, which pytest-timeout does not cover --
        # otherwise runs until the job's timeout-minutes kills all N workers at
        # once, discarding every in-flight item on every GPU. rc 124 here means
        # the deadline fired; build_weights.py drops those rows, since the
        # measurement is the ceiling itself and would compound run over run.
        budget=$(( weight * 3 + 300 ))
        [[ $budget -gt $MAX_ITEM_SECS ]] && budget=$MAX_ITEM_SECS

        start=$(date +%s)
        if [[ -n "$tag" ]]; then
            HIP_VISIBLE_DEVICES=$gpu TE_CI_SKIP_SETUP=1 TEST_FILTER="$tag" \
                JUNITXML_PREFIX="$junit_dir" \
                timeout --kill-after=60 "$budget" "$cmd" ${rest:-} > "$itemlog" 2>&1
        else
            HIP_VISIBLE_DEVICES=$gpu JUNITXML_PREFIX="$junit_dir" \
                timeout --kill-after=60 "$budget" "$cmd" ${rest:-} > "$itemlog" 2>&1
        fi
        rc=$?
        if [[ $rc -eq 124 || $rc -eq 137 ]]; then
            echo "::error::[${label}] ${safetag} killed at its ${budget}s deadline" >&2
            echo "TE_CI_ITEM_DEADLINE_EXCEEDED after ${budget}s" >> "$itemlog"
        fi
        end=$(date +%s)

        echo "$rc" > "${itemlog}.rc"
        # Machine-readable timing record. The first five columns are the stable
        # contract ci/build_weights.py --from-timing reads; the scheduling
        # columns are appended after them so that reader stays unchanged. Written
        # incrementally so a run killed by the job timeout still leaves a usable
        # record of everything that had completed.
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$label" "$safetag" "$gpu" "$((end - start))" "$rc" \
            "$((start - START_TS))" "$((end - START_TS))" "$weight" \
            >> "$LOG_DIR/timings.tsv"
        printf '[%s] gpu%-2s t+%-7s %5ss rc=%-4s est=%-8s %-9s %s\n' \
            "$(date '+%H:%M:%S')" "$gpu" "$((start - START_TS))s" "$((end - start))" \
            "$rc" "$([[ $weight -eq $DEFAULT_WEIGHT ]] && echo unknown || echo "${weight}s")" \
            "$label" "$safetag"
    done
}

: > "$LOG_DIR/timings.tsv"
for k in $(seq 0 $((NUM_GPUS - 1))); do
    worker $((FIRST_GPU + k)) &
done
wait

WALL=$(( $(date +%s) - START_TS ))
# The report divides by WALL in several places; a sub-second run (a --only of one
# trivial item) would otherwise abort the whole report on a division by zero.
[[ $WALL -lt 1 ]] && WALL=1

# ---------------------------------------------------------------------------
# Phase 5: fold per-item results back into per-suite logs and exit codes
OVERALL_RC=0
for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    # A suite with no items in the queue (filtered out by --only) must not get a
    # suite log and rc=0 -- that would read as "passed" to downstream consumers.
    awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE" || continue
    suite_log="$LOG_DIR/${SUITE_LOGFILES[$i]}"
    : > "$suite_log"
    worst=0
    for itemlog in "$ITEM_LOG_DIR/${label}."*.log; do
        [[ -e "$itemlog" ]] || continue
        rc=$(cat "${itemlog}.rc" 2>/dev/null || echo 1)
        echo "##### $(basename "$itemlog" .log) (rc=${rc}) #####" >> "$suite_log"
        cat "$itemlog" >> "$suite_log"
        [[ "$rc" != "0" ]] && worst=$rc
    done
    echo "$worst" > "${suite_log}.rc"
    [[ "$worst" != "0" ]] && OVERALL_RC=$worst
done

# ---------------------------------------------------------------------------
# Phase 6: scheduling report
#
# timings.tsv columns: label, tag, gpu, secs, rc, start_off, end_off, est.
# Everything below is derived from that one file, so the same report can be
# regenerated after the fact from an uploaded artifact.
report() {
    local tf="$LOG_DIR/timings.tsv"

    echo "=== sGPU queue: ${TOTAL_ITEMS} items, ${NUM_GPUS} GPUs, drained in ${WALL}s ==="
    echo
    echo "-- schedule: what ran where, in execution order --"
    sort -t$'\t' -k3,3n -k6,6n "$tf" | awk -F'\t' -v dw="$DEFAULT_WEIGHT" '
        $3 != last { if (NR > 1) print ""; last = $3 }
        {
            est = ($8 == dw ? "unknown" : $8 "s")
            # Estimate error is the scheduler feedback loop made visible: a large
            # positive miss is an item that should have been dispatched earlier.
            err = ($8 > 0 && $8 != dw) ? sprintf("%+.0f%%", ($4 - $8) * 100 / $8) : "n/a"
            printf "  gpu%-2s  t+%-7s %6ss  rc=%-4s est=%-9s %-7s %s/%s\n",
                   $3, $6 "s", $4, $5, est, err, $1, $2
        }'

    echo
    echo "-- per-GPU utilisation --"
    awk -F'\t' -v w="$WALL" '
        { busy[$3] += $4; n[$3]++; if ($5 != 0) bad[$3]++; if ($7 > last[$3]) last[$3] = $7 }
        END {
            for (g in busy)
                printf "  gpu%-3s %3d items  %6ss busy  %5ss idle  %5.1f%% util  %d failed\n",
                       g, n[g], busy[g], w - busy[g], busy[g] * 100 / w, bad[g] + 0
        }' "$tf" | sort

    echo
    echo "-- timeline (one column ~$(( WALL / 60 + 1 ))s; = ran, # failed, . idle) --"
    # Idle gaps are the whole point: a run of dots on one GPU while another is
    # still busy is exactly the tail a better ordering would have filled.
    awk -F'\t' -v w="$WALL" -v cols=60 -v first="$FIRST_GPU" -v n="$NUM_GPUS" '
        { from = int($6 * cols / w); to = int($7 * cols / w)
          if (to >= cols) to = cols - 1
          for (c = from; c <= to; c++) cell[$3, c] = ($5 == 0 ? "=" : "#") }
        END {
            for (g = first; g < first + n; g++) {
                line = ""
                for (c = 0; c < cols; c++) line = line ((g, c) in cell ? cell[g, c] : ".")
                printf "  gpu%-3s |%s|\n", g, line
            }
            printf "  %7s 0s%*ss\n", "", cols - 2, w
        }' "$tf"

    echo
    echo "-- efficiency --"
    awk -F'\t' -v w="$WALL" -v n="$NUM_GPUS" '
        { work += $4; if ($4 > big) { big = $4; bigname = $1 "/" $2 } }
        END {
            bound = work / n; if (big > bound) bound = big
            printf "  total work      %8.0fs across %d items\n", work, NR
            printf "  capacity        %8.0fs (%ss x %d GPUs)\n", w * n, w, n
            printf "  lower bound     %8.0fs  max(work/%d, largest item)\n", bound, n
            printf "  actual makespan %8.0fs  %+.1f%% over the bound\n", w, (w - bound) * 100 / bound
            printf "  utilisation     %8.1f%%\n", work * 100 / (w * n)
            printf "  largest item    %8.0fs  %s%s\n", big, bigname,
                   (big > work / n ? "   <-- item-bound: splitting would now pay" : "")
        }' "$tf"

    echo
    echo "-- weight accuracy (misses over 30s; these are what the next run corrects) --"
    # 30s is roughly one process startup, and below the ~5% run-to-run noise for
    # anything long enough to matter. Listing smaller misses would bury the real
    # ones -- a healthy table has almost every item within a few seconds.
    awk -F'\t' -v dw="$DEFAULT_WEIGHT" '$8 != dw && $8 > 0 {
            d = $4 - $8; a = (d < 0 ? -d : d)
            if (a >= 30) printf "%.0f\t%+.0f\t%s\t%s\t%s\n", a, d, $8, $4, $1 "/" $2 }' "$tf" \
        | sort -k1,1nr | head -15 \
        | awk -F'\t' '{printf "  %+7ss  est %6ss -> actual %6ss  %s\n", $2, $3, $4, $5}'
    awk -F'\t' -v dw="$DEFAULT_WEIGHT" '$8 == dw {printf "  unknown     -> actual %6ss  %s/%s\n", $4, $1, $2}' "$tf"

    echo
    echo "-- failures --"
    if awk -F'\t' '$5 != 0 {found = 1} END {exit !found}' "$tf"; then
        awk -F'\t' '$5 != 0 {
            printf "  rc=%-4s %6ss  %s/%s%s\n", $5, $4, $1, $2,
                   ($5 == 124 || $5 == 137 ? "   (killed at per-item deadline)" : "") }' "$tf"
    else
        echo "  none"
    fi

    # An item that never got a timings.tsv row was never dispatched -- only
    # possible if a worker died outright. Silence here would read as success.
    local ran; ran=$(wc -l < "$tf")
    if [[ "$ran" -ne "$TOTAL_ITEMS" ]]; then
        echo
        echo "  !! ${ran}/${TOTAL_ITEMS} items produced a timing record; the rest never ran:"
        awk -F'\t' 'NR == FNR {seen[$1 "/" $2] = 1; next}
                    { k = ($4 == "" ? $2 "/whole" : $2 "/" $4); if (!(k in seen)) print "     " k }' \
            "$tf" "$QUEUE_FILE"
    fi
}

report | tee "$LOG_DIR/schedule.txt"
{
    echo "## sGPU queue schedule (${WALL}s over ${NUM_GPUS} GPUs)"
    echo
    echo '```'
    cat "$LOG_DIR/schedule.txt"
    echo '```'
} > "$LOG_DIR/schedule.md"

exit $OVERALL_RC
