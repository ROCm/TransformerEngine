#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run all sGPU test suites as one global work queue across N GPUs.
#
# Usage: run_queue_sgpu.sh [-l|--log-dir <dir>] [<config>...]
#
# Config format (one suite per line; # comments and blank lines are ignored):
#   <label>  <logfile>  <mode>  <command> [args...]
#     mode=list    expanded into one work item per test invocation
#     mode=opaque  scheduled as a single work item
#
# With no config given, .github/scripts/ci_sgpu_queue.conf is used. Unlike
# run_parallel_sgpu.sh, line order carries no GPU assignment: every item from
# every config goes into one queue and workers pull on completion.
#
# The queue uses every GPU it can see; restrict it with HIP_VISIBLE_DEVICES.
#
# Example usage:
#   TEST_LEVEL=1 .github/scripts/run_queue_sgpu.sh
#   HIP_VISIBLE_DEVICES=0,1 TEST_LEVEL=1 .github/scripts/run_queue_sgpu.sh
#
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Two directories in the repo root:
#
#   test-results/  everything one run produced; this script adds logs/ under it
#   ci-weights/    the learned weight table, the one thing that must outlive the
#                  run that measured it. Same name the workflow caches.
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/test-results/logs}

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    log_error() { echo "::error::$*" >&2; }
    log_warn()  { echo "::warning::$*" >&2; }
else
    log_error() { echo "Error: $*" >&2; }
    log_warn()  { echo "Warning: $*" >&2; }
fi

# Items with no recorded weight sort first: an unknown item is more likely to be
# a new (or newly slow) one, and a long item started late is what stretches the
# tail. Losing the gamble costs far less than mis-scheduling a genuinely big item.
# With no weights file at all -- the first run on a new arch, or a cache miss --
# every item takes this, the queue keeps its natural order, and the run is simply
# unordered. That costs makespan once; the table it writes fixes the next run.
DEFAULT_WEIGHT=${TE_CI_DEFAULT_WEIGHT:-999999}

# ---------------------------------------------------------------------------
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#*=}"; shift ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [-l|--log-dir <dir>] [<config>...]" >&2
            exit 1 ;;
        *)
            break ;;
    esac
done

# Resolve config paths to absolute against the caller's CWD *before* changing
# directory, so relative paths passed by the caller remain valid: the configs
# are not read until Phase 1, long after the cd to the repo root below.
declare -a CONFIGS=()
if [[ $# -gt 0 ]]; then
    for c in "$@"; do CONFIGS+=( "$(realpath -m "$c")" ); done
else
    CONFIGS=( "${SCRIPT_DIR}/ci_sgpu_queue.conf" )
fi
for c in "${CONFIGS[@]}"; do
    if [[ ! -f "$c" ]]; then
        log_error "suite list not found: $c"
        exit 1
    fi
done

# The pool is every GPU this run can see: HIP_VISIBLE_DEVICES if set
# else what rocminfo counts.
declare -a GPU_IDS=()
GPU_SOURCE=""
detect_gpu_pool() {
    local n k
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a GPU_IDS <<< "$HIP_VISIBLE_DEVICES"
        GPU_SOURCE="HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
        return 0
    fi
    n=$(rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU')
    [[ -n "$n" && "$n" -gt 0 ]] || return 1
    for ((k = 0; k < n; k++)); do GPU_IDS+=( "$k" ); done
    GPU_SOURCE="rocminfo${ROCR_VISIBLE_DEVICES:+, ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES}"
}

detect_arch() {
    rocminfo 2>/dev/null | grep -E "^ *Name: *gfx" | head -1 | sed "s/.*gfx/gfx/;s/[: ].*//"
}

if ! detect_gpu_pool; then
    log_error "no GPU found: rocminfo reports none and HIP_VISIBLE_DEVICES is unset." \
              "Set HIP_VISIBLE_DEVICES to the devices this run may use."
    exit 1
fi

NUM_GPUS=${#GPU_IDS[@]}
echo "== GPUs: ${NUM_GPUS} visible -- ids ${GPU_IDS[*]} (via ${GPU_SOURCE}) =="

ARCH=$(detect_arch)
if [[ -z "$ARCH" ]]; then
    log_error "could not read the GPU arch from rocminfo; it keys the weight table"
    exit 1
fi
echo "== Arch: ${ARCH} =="

# The weight table is keyed by arch and TEST_LEVEL -- the coarsest keying under
# which a weight still means one thing. Read below to order the queue and
# rewritten in Phase 7, so a first local run is unordered and every run after it
# is not.
WEIGHTS_FILE="${REPO_ROOT}/ci-weights/test_weights.${ARCH}.l${TEST_LEVEL:-99}.txt"
mkdir -p "$(dirname "$WEIGHTS_FILE")" 2>/dev/null

[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(realpath -m "$LOG_DIR")"


# One directory per phase, so "where did that come from" is answered by the path
# and a phase can be cleared without touching the others:
#
#   prerequisite_ck_jit_status/ Phase 3  one-time prerequisites per suite
#   items/                      Phase 4  the test output itself -- one file per item
#   suites/                     Phase 5  per-suite verdict: rc + index into items/
#   report/                     Phase 6  the human-readable schedule
#   queue/                               the machine-readable state: what to run,
#                                        what it cost
#
# Phase 1 has no directory: what it produced is queue.tsv and items.tsv, and all
# it has of its own is whatever the suites printed on stderr while listing, which
# is one file.
EXPAND_LOG="$LOG_DIR/expand.log"
SETUP_DIR="$LOG_DIR/prerequisite_ck_jit_status"
ITEM_LOG_DIR="$LOG_DIR/items"
SUITE_LOG_DIR="$LOG_DIR/suites"
REPORT_DIR="$LOG_DIR/report"
QUEUE_DIR="$LOG_DIR/queue"

rm -rf "${REPO_ROOT}/test-results"
rm -rf "$ITEM_LOG_DIR" "$SUITE_LOG_DIR"

mkdir -p "$SETUP_DIR" "$ITEM_LOG_DIR" "$SUITE_LOG_DIR" "$REPORT_DIR" "$QUEUE_DIR"
: > "$EXPAND_LOG"
cd "$REPO_ROOT" || { echo "Error: cannot cd to '${REPO_ROOT}'" >&2; exit 1; }

QUEUE_FILE="$QUEUE_DIR/queue.tsv"
ITEMS_FILE="$QUEUE_DIR/items.tsv"
TIMINGS_FILE="$QUEUE_DIR/timings.tsv"
IDX_FILE="$QUEUE_DIR/queue.idx"
LOCK_FILE="$QUEUE_DIR/queue.lock"
: > "$QUEUE_FILE"
: > "$ITEMS_FILE"
# Phase 2 removes this, but only if it gets that far: an expansion that bails out
# leaves it behind, and re-running into the same log dir would then append to the
# previous attempt and trip the duplicate-tag check. Local runs do that a lot.
: > "$QUEUE_FILE.raw"

# ---------------------------------------------------------------------------
# Phase 1: expand every suite into work items
#
# List mode runs the suite script with TE_CI_LIST_ONLY=1, which makes pytest_run
# echo "TE_CI_ITEM <tag>" for each invocation it would have made instead of
# running it. The suite's own control flow produces the list, so nothing here has
# to know how a suite decomposes.
#
# Every list-mode suite is listed twice, because "will run" and "exists" are
# different questions:
#
#   run 1  LIST_ONLY             what this host will run     -> the queue
#   run 2  LIST_ONLY + LIST_ALL  what exists at this level   -> items.tsv
#
# Run 1 keeps every gate: level, backend matrix, and the host capability probes.
# Run 2 answers the capability probes yes without probing (check_list_all in
# ci/_utils.sh); the level and matrix gates still apply.
#
# So run2 - run1 is what this host skipped -- no flash-attn, say -- and those
# tests do still exist. Only a tag in neither list is gone for good, and that is
# what ci/build_weights.py prunes on.
echo "== Expanding suites into work items =="
declare -a SUITE_LABELS SUITE_LOGFILES
for config in "${CONFIGS[@]}"; do
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        read -r label logfile mode cmd rest <<< "$line"
        SUITE_LABELS+=( "$label" )
        SUITE_LOGFILES+=( "$logfile" )
        if [[ "$mode" == "list" ]]; then
            # The tags themselves are not logged here -- they go straight into
            # queue.tsv and items.tsv, which is where anything reading them
            # looks. What is left is stderr, banner-separated by suite and pass:
            # the suite scripts report a declined capability probe on it, which
            # is noise on a good run but the only explanation of a list that
            # came back short.
            echo "=== ${label}: list -- what this runner will run ===" >> "$EXPAND_LOG"
            mapfile -t tags < <(TE_CI_LIST_ONLY=1 "$cmd" ${rest:-} \
                                2>> "$EXPAND_LOG" \
                                | sed -n 's/^TE_CI_ITEM //p')
            if [[ ${#tags[@]} -eq 0 ]]; then
                log_error "suite '${label}' (${cmd}) produced no work items"
                tail -20 "$EXPAND_LOG" >&2
                exit 1
            fi
            for tag in "${tags[@]}"; do
                printf '%s\t%s\t%s\t%s\n' "$label" "$cmd" "$tag" "${rest:-}" >> "$QUEUE_FILE.raw"
            done
            echo "=== ${label}: list-all -- what exists at this level ===" >> "$EXPAND_LOG"
            mapfile -t all_tags < <(TE_CI_LIST_ONLY=1 TE_CI_LIST_ALL=1 "$cmd" ${rest:-} \
                                    2>> "$EXPAND_LOG" \
                                    | sed -n 's/^TE_CI_ITEM //p')
            # The wider list must be a superset. If it is not, it cannot be
            # trusted as "everything that exists", so the label is left out of
            # items.tsv -- which is exactly what tells build_weights.py to prune
            # nothing for it. Warn rather than fail: this only degrades
            # bookkeeping, and refusing to run the tests would be a worse trade.
            if [[ ${#all_tags[@]} -lt ${#tags[@]} ]]; then
                log_warn "suite '${label}': list-all returned ${#all_tags[@]} items," \
                         "fewer than the ${#tags[@]} queued; its weights will not be pruned"
            else
                for tag in "${all_tags[@]}"; do
                    printf '%s\t%s\n' "$label" "$tag" >> "$ITEMS_FILE"
                done
            fi
            echo "  ${label}: ${#tags[@]} items (${#all_tags[@]} exist at this level)"
        else
            printf '%s\t%s\t%s\t%s\n' "$label" "$cmd" "" "${rest:-}" >> "$QUEUE_FILE.raw"
            printf '%s\t%s\n' "$label" "" >> "$ITEMS_FILE"
            echo "  ${label}: 1 item (opaque)"
        fi
    done < "$config"
done

# A label repeated across configs would install its prerequisites twice in
# Phase 3 and have Phase 5 roll the same item logs into the same suite log
# twice. The tag check below cannot catch it: an opaque suite has no tag.
dupe_labels=$(printf '%s\n' "${SUITE_LABELS[@]}" | sort | uniq -d)
if [[ -n "$dupe_labels" ]]; then
    log_error "duplicate suite labels across configs:"
    echo "$dupe_labels" >&2
    exit 1
fi

# Duplicate tags within a suite would collide on both the JUnit XML filename and
# the TEST_FILTER dispatch (one invocation would run both). Fail loudly instead.
dupes=$(cut -f1,3 "$QUEUE_FILE.raw" | grep -v '^\S*\t$' | sort | uniq -d)
if [[ -n "$dupes" ]]; then
    log_error "duplicate work item tags detected:"
    echo "$dupes" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: weight and order the queue (longest processing time first)
awk -v wf="$WEIGHTS_FILE" -v dw="$DEFAULT_WEIGHT" '
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
        # Truncate to an integer: the weight is also used in shell arithmetic to
        # derive the per-item deadline, and bash cannot parse "675.0".
        print (key in w ? int(w[key]) : dw), $0
    }
' "$QUEUE_FILE.raw" | sort -t$'\t' -k1,1nr > "$QUEUE_FILE"
rm -f "$QUEUE_FILE.raw"

if [[ ! -s "$QUEUE_FILE" ]]; then
    log_error "queue is empty: every suite expanded to nothing"
    exit 1
fi

TOTAL_ITEMS=$(wc -l < "$QUEUE_FILE")
echo "== Queue: ${TOTAL_ITEMS} items over ${NUM_GPUS} GPUs (${GPU_IDS[*]}) =="
if [[ -f "$WEIGHTS_FILE" ]]; then
    echo "   weights: $WEIGHTS_FILE ($(wc -l < "$WEIGHTS_FILE") items)"
else
    echo "   weights: none yet -- this run is unordered and will write $WEIGHTS_FILE"
fi

# The plan is printed before anything runs: it is the only record of what the
# scheduler intended, and comparing it against the schedule reported at the end
# is how a mis-weighted item is spotted.
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

# ---------------------------------------------------------------------------
# Phase 3: one-time, container-wide setup
#
# pip prerequisites and the CK JIT blob cache are container-wide filesystem
# state, so each suite installs and prebuilds them once here.
setup_banner=""
for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    # Columns are: weight, label, cmd, tag, rest. Only list-mode suites (those
    # with a non-empty tag) have a setup/dispatch split; opaque suites are a
    # single invocation that still does its own setup.
    cmd=$(awk -F'\t' -v l="$label" '$2==l && $4!="" {print $3; exit}' "$QUEUE_FILE")
    [[ -z "$cmd" ]] && continue
    if [[ -z "$setup_banner" ]]; then
        echo "== One-time setup: pip prerequisites + CK JIT prebuild (once per suite) =="
        setup_banner=1
    fi
    printf '  %s: installing prerequisites, prebuilding CK JIT blobs (%s) ... ' "$label" "$cmd"
    setup_start=$(date +%s)
    if ! HIP_VISIBLE_DEVICES=${GPU_IDS[0]} TE_CI_SETUP_ONLY=1 "$cmd" \
            > "$SETUP_DIR/${label}.log" 2>&1; then
        echo "FAILED"
        log_error "setup failed for ${label}; see $SETUP_DIR/${label}.log"
        tail -30 "$SETUP_DIR/${label}.log" >&2
        exit 1
    fi
    echo "done in $(( $(date +%s) - setup_start ))s"
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
    local i line weight label cmd tag rest itemlog rc safetag junit_dir start end incomplete
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

        # No scheduler-imposed deadline: the suite scripts' own PYTEST_TIMEOUT and
        # the workflow's timeout-minutes are the only limits, exactly as they are
        # outside the queue.
        start=$(date +%s)
        if [[ -n "$tag" ]]; then
            HIP_VISIBLE_DEVICES=$gpu TE_CI_SKIP_SETUP=1 TEST_FILTER="$tag" \
                JUNITXML_PREFIX="$junit_dir" \
                "$cmd" ${rest:-} > "$itemlog" 2>&1
        else
            HIP_VISIBLE_DEVICES=$gpu JUNITXML_PREFIX="$junit_dir" \
                "$cmd" ${rest:-} > "$itemlog" 2>&1
        fi
        rc=$?
        end=$(date +%s)

        echo "$rc" > "${itemlog}.rc"

        # A te_ci_result_sink sidecar that outlived the process means pytest
        # never reached its end-of-session write -- a --timeout-method=thread
        # expiry, a segfault, or an OOM-kill -- so this duration is where the
        # item was cut off, not what it costs. rc cannot be used to tell: a
        # thread-method timeout exits 1, indistinguishable from an ordinary test
        # failure, which is a perfectly good measurement. build_weights.py drops
        # the flagged rows rather than teaching the table a truncated number.
        incomplete=0
        if [[ -n "$junit_dir" ]]; then
            if [[ -n "$tag" ]]; then
                # Items sharing a label share junit_dir, so only this item's own
                # sidecar may be consulted -- a glob would see the in-flight
                # sidecar of an item still running on another GPU.
                [[ -e "${junit_dir}${tag}${JUNITXML_SUFFIX:-}.partial" ]] && incomplete=1
            elif compgen -G "${junit_dir}*.partial" > /dev/null; then
                incomplete=1   # opaque suite: it is the only item in its dir
            fi
        fi

        # Machine-readable timing record. The first five columns are the stable
        # contract ci/build_weights.py reads; the scheduling columns are appended
        # after them so that reader stays unchanged. Written incrementally so a
        # run killed by the job timeout still leaves a usable record of
        # everything that had completed.
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$label" "$safetag" "$gpu" "$((end - start))" "$rc" \
            "$((start - START_TS))" "$((end - START_TS))" "$weight" "$incomplete" \
            >> "$TIMINGS_FILE"
        printf '[%s] gpu%-2s t+%-7s %5ss rc=%-4s est=%-8s %-9s %s\n' \
            "$(date '+%H:%M:%S')" "$gpu" "$((start - START_TS))s" "$((end - start))" \
            "$rc" "$([[ $weight -eq $DEFAULT_WEIGHT ]] && echo unknown || echo "${weight}s")" \
            "$label" "$safetag"
    done
}

: > "$TIMINGS_FILE"
for gpu in "${GPU_IDS[@]}"; do
    worker "$gpu" &
done
wait

WALL=$(( $(date +%s) - START_TS ))
# The report divides by WALL in several places; a run short enough to round to
# zero seconds would otherwise abort the whole report on a division by zero.
[[ $WALL -lt 1 ]] && WALL=1

# ---------------------------------------------------------------------------
# Phase 5: roll per-item results up into a per-suite verdict
#
# Two outputs per suite, both keyed by the logfile name from the config so the
# workflow's failure gate keeps reading the same name: the exit code it gates on,
# and an index of which items ran and how each one ended.
#
# The item logs are not concatenated into that index. They are the largest files
# the run produces, and copying them would double both the artifact and the
# reader's work -- the index says which item to open, and the item log is right
# there under items/ with only that item's output in it.
OVERALL_RC=0
for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    # Phase 1 fails a suite that expands to nothing, so this should never skip.
    # It stays because the alternative to skipping is an empty suite log and
    # rc=0, which reads as "passed" to the workflow's gate.
    awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE" || continue
    suite_log="$SUITE_LOG_DIR/${SUITE_LOGFILES[$i]}"
    : > "$suite_log"
    worst=0
    for itemlog in "$ITEM_LOG_DIR/${label}."*.log; do
        [[ -e "$itemlog" ]] || continue
        rc=$(cat "${itemlog}.rc" 2>/dev/null || echo 1)
        printf '%-4s rc=%-4s items/%s\n' \
            "$([[ "$rc" == "0" ]] && echo ok || echo FAIL)" "$rc" \
            "$(basename "$itemlog")" >> "$suite_log"
        [[ "$rc" != "0" ]] && worst=$rc
    done
    echo "$worst" > "${suite_log}.rc"
    [[ "$worst" != "0" ]] && OVERALL_RC=$worst
done

# ---------------------------------------------------------------------------
# Phase 6: scheduling report
# ---------------------------------------------------------------------------
if ! python3 "$REPO_ROOT/ci/schedule_report.py" "$LOG_DIR" \
        --gpus "${GPU_IDS[*]}" \
        --wall "$WALL" \
        --default-weight "$DEFAULT_WEIGHT"; then
    log_warn "could not write the scheduling report"
fi

# ---------------------------------------------------------------------------
# Phase 7: Update the learned weight table for the next run. 
# ---------------------------------------------------------------------------
echo
if ! python3 "$REPO_ROOT/ci/build_weights.py" "$TIMINGS_FILE" \
        --items "$ITEMS_FILE" -o "$WEIGHTS_FILE"; then
    log_warn "could not update $WEIGHTS_FILE; the next run will use the table as it stands"
fi

exit $OVERALL_RC
