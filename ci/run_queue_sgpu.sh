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
# The queue uses every GPU it can see; restrict it with HIP_VISIBLE_DEVICES.
#
# Example usage:
#   TEST_LEVEL=1 ci/run_queue_sgpu.sh
#   HIP_VISIBLE_DEVICES=0,1 TEST_LEVEL=1 ci/run_queue_sgpu.sh
set -u
SCRIPT_START_TS=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Three directories in the repo root:
#
#   test-results/  everything one run produced; this script adds logs/ under it
#   ci-weights/    the learned weight table
#   ci-rerun/      what failed, so a re-run of this job can queue only that
: "${LOG_DIR:=${REPO_ROOT}/test-results/logs}"

# Items with no recorded weight sort first: an unknown item is more likely to be
# a new (or newly slow) one, and a long item started late is what stretches the
# tail. Losing the gamble costs far less than mis-scheduling a genuinely big item.
: "${DEFAULT_WEIGHT:=${TE_CI_DEFAULT_WEIGHT:-999999}}"

# Cross-phase state. The SUITE_* arrays are filled once by Phase 0 and share an
# index, so every later phase walks them instead of re-reading the configs.
declare -a CONFIGS=()
declare -a GPU_IDS=()
declare -a SUITE_LABELS=()
declare -a SUITE_LOGFILES=()
declare -a SUITE_MODES=()
declare -a SUITE_CMDS=()
declare -a SUITE_ARGS=()
GPU_SOURCE=""
OVERALL_RC=0

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    log_error() { echo "::error::$*" >&2; }
    log_warn()  { echo "::warning::$*" >&2; }
else
    log_error() { echo "Error: $*" >&2; }
    log_warn()  { echo "Warning: $*" >&2; }
fi

if [[ -n "${TEST_MGPU:-}" ]]; then
    log_warn "ignoring TEST_MGPU=${TEST_MGPU}: this queue only dispatches single-GPU items"
fi
export TEST_SGPU=1
export TEST_MGPU=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
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

# Resolve config paths to absolute
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

# ---------------------------------------------------------------------------
# Phase 0: read the configs and reject a bad one
# ---------------------------------------------------------------------------
for config in "${CONFIGS[@]}"; do
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        read -r label logfile mode cmd rest <<< "$line"
        [[ "$cmd" == /* ]] || cmd="${REPO_ROOT}/${cmd}"
        cmd="$(realpath -m "$cmd")"
        SUITE_LABELS+=( "$label" )
        SUITE_LOGFILES+=( "$logfile" )
        SUITE_MODES+=( "$mode" )
        SUITE_CMDS+=( "$cmd" )
        SUITE_ARGS+=( "${rest:-}" )
    done < "$config"
done

if [[ ${#SUITE_LABELS[@]} -eq 0 ]]; then
    log_error "no suites to run: ${CONFIGS[*]} contain only comments and blank lines"
    exit 1
fi

# Reject duplicate labels across configs: the queue is keyed by label
dupe_labels=$(printf '%s\n' "${SUITE_LABELS[@]}" | sort | uniq -d)
if [[ -n "$dupe_labels" ]]; then
    log_error "duplicate suite labels across configs:"
    echo "$dupe_labels" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Set up the run: GPU pool, arch, weight table, log tree
# ---------------------------------------------------------------------------

# Fill GPU_IDS with every GPU this run can see: HIP_VISIBLE_DEVICES if set
# else what rocminfo counts. Returns 1 if there is none.
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

# The weight table is keyed by arch and TEST_LEVEL
WEIGHTS_FILE="${REPO_ROOT}/ci-weights/test_weights.${ARCH}.l${TEST_LEVEL:-99}.txt"
mkdir -p "$(dirname "$WEIGHTS_FILE")" 2>/dev/null

# What failed last time, keyed the same way. Phase 5 writes it; Phase 2 reads it
# and narrows the queue to it, which is how "re-run failed jobs" re-runs only the
# failed tests. Nothing here decides whether this is a re-run -- the file is
# present only when something put the previous attempt's copy in place, so the
# scheduler needs no flag to tell the two cases apart.
RERUN_FILE="${REPO_ROOT}/ci-rerun/failed.${ARCH}.l${TEST_LEVEL:-99}.tsv"
mkdir -p "$(dirname "$RERUN_FILE")" 2>/dev/null
RERUN_MODE=""   # set by Phase 2 once it has actually narrowed the queue

# A second, finer layer over the same idea: one file of pytest nodeids per item,
# named "<label>.<tag>.txt", narrowing a re-run of that item to the tests that
# actually failed instead of the whole file. Strictly optional -- an item with no
# file here re-runs whole -- so nothing about the queue depends on it being right.
RERUN_TESTS_DIR="${REPO_ROOT}/ci-rerun/tests"

[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(realpath -m "$LOG_DIR")"

# Items no longer run from the repo root (Phase 4), so a relative JUnit prefix
# would land in a different place for every item. Anchor it once, here, keeping
# any trailing slash: the prefix is concatenated with the label, not joined as a
# path, and realpath strips the separator.
if [[ -n "${JUNITXML_PREFIX:-}" && "$JUNITXML_PREFIX" != /* ]]; then
    junit_prefix_sep=""
    [[ "$JUNITXML_PREFIX" == */ ]] && junit_prefix_sep=/
    JUNITXML_PREFIX="$(realpath -m "$JUNITXML_PREFIX")${junit_prefix_sep}"
    export JUNITXML_PREFIX
fi

# Directory Structure under LOG_DIR:
#
#   prerequisite_ck_jit_status/ Phase 3  the CK JIT prebuild log and the cache
#                                        snapshot it leaves for the post-queue
#                                        drift check, plus one pip setup log
#                                        per suite
#   items/                      Phase 4  the test output itself -- one file per item
#   cwd/                        Phase 4  one working directory per item; empty ones
#                                        are removed, so whatever is left here is
#                                        what an item wrote to its CWD
#   suites/                     Phase 5  per-suite verdict: rc + index into items/
#   report/                     Phase 7  the human-readable schedule
#   queue/                               the machine-readable state: what to run,
#                                        what it cost
#
SETUP_DIR="$LOG_DIR/prerequisite_ck_jit_status"
ITEM_LOG_DIR="$LOG_DIR/items"
ITEM_CWD_DIR="$LOG_DIR/cwd"
SUITE_LOG_DIR="$LOG_DIR/suites"
REPORT_DIR="$LOG_DIR/report"
QUEUE_DIR="$LOG_DIR/queue"

rm -rf "$SETUP_DIR" "$ITEM_LOG_DIR" "$ITEM_CWD_DIR" "$SUITE_LOG_DIR" "$REPORT_DIR" "$QUEUE_DIR"

mkdir -p "$SETUP_DIR" "$ITEM_LOG_DIR" "$ITEM_CWD_DIR" "$SUITE_LOG_DIR" "$REPORT_DIR" "$QUEUE_DIR"

# The repo root is where the suites are expanded and where the scheduler's own
# state lives; the items themselves each get their own directory in Phase 4.
cd "$REPO_ROOT" || { echo "Error: cannot cd to '${REPO_ROOT}'" >&2; exit 1; }

QUEUE_FILE="$QUEUE_DIR/queue.tsv"        # Phase 2 writes, Phases 4 and 5 read
ITEMS_FILE="$QUEUE_DIR/items.tsv"        # Phase 1 writes, Phase 6 reads
TIMINGS_FILE="$QUEUE_DIR/timings.tsv"    # Phase 4 writes, Phases 6 and 7 read
: > "$QUEUE_FILE"
: > "$ITEMS_FILE"

# ---------------------------------------------------------------------------
# Phase 1: expand every suite into work items
# ---------------------------------------------------------------------------
#
# List mode runs the suite script with TE_CI_LIST_ITEMS=1, which makes pytest_run
# echo "TE_CI_ITEM <tag>" instead of running it.
#
# Every list-mode suite is listed twice, because "will run" and "exists" are
# different questions:
#
#   run 1  LIST_ONLY             what this host will run     -> the queue
#   run 2  LIST_ONLY + LIST_ALL  what exists at this level   -> items.tsv
#
# So run2 - run1 is what this host skipped -- no flash-attn, say -- and those
# tests do still exist. Only a tag in neither list is gone for good, and that is
# what build_weights.py prunes on.
EXPAND_LOG="$LOG_DIR/expand.log"   # whatever the suites printed while listing
LIST_TMP="$QUEUE_DIR/.expand.tmp"  # one suite's list, reused per suite
: > "$EXPAND_LOG"
# Truncated, not appended to: Phase 2 removes the .raw file only if it gets that
# far, so an expansion that bails out leaves it behind, and the next run would
# carry that attempt's tags into the duplicate-tag check. The startup wipe of
# QUEUE_DIR covers the same case; this keeps the phase correct on its own.
: > "$QUEUE_FILE.raw"

EXPAND_TS=$(date +%s)
echo "== Expanding test suites into work items =="
for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    mode="${SUITE_MODES[$i]}"
    cmd="${SUITE_CMDS[$i]}"
    rest="${SUITE_ARGS[$i]}"
    if [[ "$mode" == "list" ]]; then
        echo "=== ${label}: list -- what this runner will run ===" >> "$EXPAND_LOG"
        TE_CI_LIST_ITEMS=1 "$cmd" ${rest:-} > "$LIST_TMP" 2>> "$EXPAND_LOG"
        list_rc=$?
        mapfile -t tags < <(sed -n 's/^TE_CI_ITEM //p' "$LIST_TMP")

        if [[ $list_rc -ne 0 ]]; then
            log_error "suite '${label}' (${cmd}) reported an error while listing (rc=${list_rc})"
            tail -20 "$EXPAND_LOG" >&2
            exit 1
        fi
        if [[ ${#tags[@]} -eq 0 ]]; then
            log_error "suite '${label}' (${cmd}) produced no work items"
            tail -20 "$EXPAND_LOG" >&2
            exit 1
        fi
        for tag in "${tags[@]}"; do
            printf '%s\t%s\t%s\t%s\n' "$label" "$cmd" "$tag" "${rest:-}" >> "$QUEUE_FILE.raw"
        done

        echo "=== ${label}: list-all -- what exists at this level ===" >> "$EXPAND_LOG"
        TE_CI_LIST_ITEMS=1 TE_CI_SKIP_CHECK_SUPPORTED=1 "$cmd" ${rest:-} > "$LIST_TMP" 2>> "$EXPAND_LOG"
        all_rc=$?
        mapfile -t all_tags < <(sed -n 's/^TE_CI_ITEM //p' "$LIST_TMP")

        if [[ $all_rc -ne 0 ]]; then
            log_warn "suite '${label}': list-all reported an error (rc=${all_rc}) after" \
                     "returning ${#all_tags[@]} items; its weights will not be pruned"
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
done
rm -f "$LIST_TMP"

# ---------------------------------------------------------------------------
# Phase 2: narrow the queue to a re-run, then weight and order it
# ---------------------------------------------------------------------------
#
# Narrowing happens here, after expansion, rather than by expanding less. Phase 1
# still lists every suite in full, so items.tsv stays a complete census of what
# exists at this level -- which is what Phase 6 prunes against, and reading "not
# re-run this attempt" as "deleted" would throw away most of the weight table.
if [[ -s "$RERUN_FILE" ]]; then
    # Columns differ: the failure list is label+tag, the raw queue is
    # label+cmd+tag+rest, so the join is on fields 1,2 against 1,3.
    awk -F'\t' 'NR==FNR { want[$1 FS $2]; next } ($1 FS $3) in want' \
        "$RERUN_FILE" "$QUEUE_FILE.raw" > "$QUEUE_FILE.rerun"
    n_failed=$(wc -l < "$RERUN_FILE")
    n_rerun=$(wc -l < "$QUEUE_FILE.rerun")
    if [[ "$n_rerun" -gt 0 ]]; then
        RERUN_MODE=1
        mv "$QUEUE_FILE.rerun" "$QUEUE_FILE.raw"
        echo "== Re-run: ${n_failed} items failed the previous attempt;" \
             "queueing the ${n_rerun} of them this runner still has =="
    else
        # Every listed item is gone -- the config changed under the re-run, or
        # this runner skips them all. Running everything is the safe reading of
        # that; trusting the empty intersection would test nothing and pass.
        rm -f "$QUEUE_FILE.rerun"
        log_warn "none of the ${n_failed} items in ${RERUN_FILE##*/} are in this" \
                 "runner's queue; running everything instead"
    fi
fi

if ! python3 "$REPO_ROOT/ci/scheduler/build_weights.py" order "$QUEUE_FILE.raw" \
        --weights "$WEIGHTS_FILE" \
        --output "$QUEUE_FILE" \
        --default-weight "$DEFAULT_WEIGHT" \
        --gpus "${GPU_IDS[*]}"; then
    log_error "could not order the queue"
    exit 1
fi
rm -f "$QUEUE_FILE.raw"

TOTAL_ITEMS=$(wc -l < "$QUEUE_FILE")
EXPAND_SECS=$(( $(date +%s) - EXPAND_TS ))   # Phases 1-2: listing and ordering

# ---------------------------------------------------------------------------
# Phase 3: one-time prerequisites
# ---------------------------------------------------------------------------
#
# Two kinds of prerequisite, hoisted to two different levels because their scope
# differs:
#
#   CK JIT blobs   container-wide state keyed by GPU arch, not by framework, so
#                  every suite wants the same cache. Built once for the whole
#                  run, here, and TE_CI_SKIP_CK_JIT keeps the per-suite setup
#                  below from rebuilding what already exists.
#   pip packages   framework-specific (torch vs jax), so each list-mode suite
#                  still installs its own.
#
# Outside the queue nothing changes: a bare ci/pytorch.sh sees neither variable
# and still does both steps inline.
SETUP_TS=$(date +%s)

# ck_jit_prebuild and check_setup_needed live in _utils.sh, the same definitions
# the suite scripts use -- the point of the hoist is to run that code once, not
# to reimplement it. DIR is what _utils.sh resolves TE_PATH from.
DIR="$REPO_ROOT/ci"
# shellcheck source=/dev/null
source "$REPO_ROOT/ci/_utils.sh"

# The build runs in a subshell (so pinning HIP_VISIBLE_DEVICES does not stick to
# the scheduler itself) and the drift check runs after the queue drains, in this
# process. A shell variable cannot cross that boundary, so the snapshot goes to
# a file; _utils.sh falls back to its in-process variable when this is unset.
export TE_CI_CK_JIT_SNAPSHOT="$SETUP_DIR/ck_jit_cache.snapshot"

echo "== One-time setup: CK JIT prebuild (once per run, shared by every suite) =="
ck_jit_start=$(date +%s)
if ! ( export HIP_VISIBLE_DEVICES="${GPU_IDS[0]}"; ck_jit_prebuild build ) \
        > "$SETUP_DIR/ck_jit_prebuild.log" 2>&1; then
    log_error "CK JIT prebuild failed; see $SETUP_DIR/ck_jit_prebuild.log"
    tail -30 "$SETUP_DIR/ck_jit_prebuild.log" >&2
    exit 1
fi
sed -n 's/^/  /p' "$SETUP_DIR/ck_jit_prebuild.log"
echo "  done in $(( $(date +%s) - ck_jit_start ))s"

setup_banner=""
for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    cmd="${SUITE_CMDS[$i]}"
    # Only list-mode suites have a setup/dispatch split; an opaque suite is a
    # single invocation that still does its own setup inline.
    [[ "${SUITE_MODES[$i]}" == "list" ]] || continue
    # Nothing of this suite is queued, so installing its prerequisites would buy
    # nothing. Only a re-run reaches this: Phase 1 fails a suite that expands to
    # nothing, so outside one every list-mode suite has rows.
    awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE" || continue
    if [[ -z "$setup_banner" ]]; then
        echo "== One-time setup: pip prerequisites (once per suite) =="
        setup_banner=1
    fi
    printf '  %s: installing prerequisites (%s) ... ' "$label" "$cmd"
    setup_start=$(date +%s)
    if ! HIP_VISIBLE_DEVICES=${GPU_IDS[0]} TE_CI_SETUP_ONLY=1 TE_CI_SKIP_CK_JIT=1 "$cmd" \
            > "$SETUP_DIR/${label}.log" 2>&1; then
        echo "FAILED"
        log_error "setup failed for ${label}; see $SETUP_DIR/${label}.log"
        tail -30 "$SETUP_DIR/${label}.log" >&2
        exit 1
    fi
    echo "done in $(( $(date +%s) - setup_start ))s"
done
SETUP_SECS=$(( $(date +%s) - SETUP_TS ))   # Phase 3: CK JIT prebuild + pip prerequisites

# ---------------------------------------------------------------------------
# Phase 4: run the queue
# ---------------------------------------------------------------------------
IDX_FILE="$QUEUE_DIR/queue.idx"    # next queue line to hand out
LOCK_FILE="$QUEUE_DIR/queue.lock"  # guards the read-modify-write of IDX_FILE

# Echo the next queue index and advance it. Every worker calls this, so the
# read-modify-write is done under flock.
take_next() {
    local i
    {
        flock 9
        i=$(cat "$IDX_FILE")
        echo $((i + 1)) > "$IDX_FILE"
    } 9<>"$LOCK_FILE"
    echo "$i"
}

# Usage: worker <gpu>
# Pull items off the queue until it is empty, running each one on <gpu> and
# appending its timing record to TIMINGS_FILE.
worker() {
    local gpu=$1
    local i line weight label cmd tag rest itemlog rc safetag junit_dir start end incomplete
    local itemcwd
    while :; do
        i=$(take_next)
        [[ "$i" -gt "$TOTAL_ITEMS" ]] && break
        line=$(sed -n "${i}p" "$QUEUE_FILE")
        [[ -z "$line" ]] && break

        weight="${line%%$'\t'*}"; line="${line#*$'\t'}"
        label="${line%%$'\t'*}"; line="${line#*$'\t'}"
        cmd="${line%%$'\t'*}"; line="${line#*$'\t'}"
        tag="${line%%$'\t'*}"
        rest="${line#*$'\t'}"

        safetag="${tag:-whole}"
        itemlog="$ITEM_LOG_DIR/${label}.${safetag}.log"
        if [[ -n "${JUNITXML_PREFIX:-}${JUNITXML_SUFFIX:-}" ]]; then
            junit_dir="${JUNITXML_PREFIX:-}${label}/"
            mkdir -p "$junit_dir"
        else
            junit_dir=""
        fi

        # Every item gets a working directory to itself
        itemcwd="$ITEM_CWD_DIR/${label}.${safetag}"
        mkdir -p "$itemcwd"

        # Narrow this item to the individual tests that failed the last attempt,
        # if one left a list for it. List-mode items only: an opaque suite runs
        # several pytest invocations and a single nodeid list cannot speak for
        # all of them. Empty reads to the plugin as "run the item whole", which
        # is also what it does with a list that no longer matches anything.
        only_tests="$RERUN_TESTS_DIR/${label}.${tag}.txt"
        [[ -n "$tag" && -n "$junit_dir" && -s "$only_tests" ]] || only_tests=""

        # No scheduler-imposed deadline: the suite scripts' own PYTEST_TIMEOUT and
        # the workflow's timeout-minutes are the only limits, exactly as they are
        # outside the queue.
        start=$(date +%s)
        if [[ -n "$tag" ]]; then
            ( cd "$itemcwd" && HIP_VISIBLE_DEVICES=$gpu TE_CI_SKIP_SETUP=1 TEST_FILTER="$tag" \
                JUNITXML_PREFIX="$junit_dir" TE_CI_ONLY_TESTS="$only_tests" \
                "$cmd" ${rest:-} ) > "$itemlog" 2>&1
        else
            ( cd "$itemcwd" && HIP_VISIBLE_DEVICES=$gpu JUNITXML_PREFIX="$junit_dir" \
                "$cmd" ${rest:-} ) > "$itemlog" 2>&1
        fi
        rc=$?
        end=$(date +%s)

        # Leave the directory only when the item put something in it, so that
        # cwd/ ends up listing exactly the items that write relative paths.
        rmdir "$itemcwd" 2>/dev/null

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

        # Update timings.tsv: Phase 6 learns the next run's weights
        # from it. One row appended per item as
        # it ends, so a killed run still records what had finished.
        #
        #   label  tag  gpu  secs  rc  start_off  end_off  est  incomplete
        #
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$label" "$safetag" "$gpu" "$((end - start))" "$rc" \
            "$((start - START_TS))" "$((end - START_TS))" "$weight" "$incomplete" \
            >> "$TIMINGS_FILE"

        # Console progress: one line per item as it finishes
        #
        #   time  gpu  start offset  duration  rc  estimate  suite  item
        #
        printf '[%s] gpu%-2s t+%-7s %5ss rc=%-4s est=%-8s %-9s %s\n' \
            "$(date '+%H:%M:%S')" "$gpu" "$((start - START_TS))s" "$((end - start))" \
            "$rc" "$([[ $weight -eq $DEFAULT_WEIGHT ]] && echo unknown || echo "${weight}s")" \
            "$label" "$safetag"
    done
}

echo 1 > "$IDX_FILE"
: > "$LOCK_FILE"
: > "$TIMINGS_FILE"
START_TS=$(date +%s)

for gpu in "${GPU_IDS[@]}"; do
    worker "$gpu" &
done
wait

WALL=$(( $(date +%s) - START_TS ))
# The report divides by WALL in several places; a run short enough to round to
# zero seconds would otherwise abort the whole report on a division by zero.
[[ $WALL -lt 1 ]] && WALL=1

# --- CK JIT cache drift ----------------------------------------------------
# What the queue JIT-compiled that ci/ck_jit_prebuild.txt did not cover, i.e.
# what that list is now missing. Run once for the whole queue rather than once
# per item: the cache is container-wide, so every item after the first would
# report the same drift. Silent when the list is complete.
ck_jit_drift="$(ck_jit_prebuild list 2>&1)" \
    || log_warn "CK JIT cache check failed; continuing"
echo "== CK JIT cache =="
if [[ -n "$ck_jit_drift" ]]; then
    sed -n 's/^/  /p' <<< "$ck_jit_drift"
else
    echo "  no drift: ci/ck_jit_prebuild.txt covered every blob the queue used"
fi

# ---------------------------------------------------------------------------
# Phase 5: roll per-item results up into a per-suite verdict
# ---------------------------------------------------------------------------
declare -a FAILED_ITEMS=()   # log path per failed item, for Phase 8
declare -a FAILED_KEYS=()    # label+tag per failed item, for the next attempt

for i in "${!SUITE_LABELS[@]}"; do
    label="${SUITE_LABELS[$i]}"
    suite_log="$SUITE_LOG_DIR/${SUITE_LOGFILES[$i]}"
    if ! awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE"; then
        # Nothing of this suite was queued. On a re-run that is the ordinary
        # case and means every item of it passed the attempt being re-run, so it
        # is recorded as passing: the workflow's gate reads these files, and a
        # missing one reads as a failure rather than as an absence.
        #
        # Outside a re-run the branch is unreachable (Phase 1 fails a suite that
        # expands to nothing) and skipping is the honest answer -- writing rc=0
        # for a suite that genuinely vanished would claim a pass nobody saw.
        [[ -n "$RERUN_MODE" ]] || continue
        echo "not re-run this attempt: no item of this suite failed the last one" > "$suite_log"
        echo 0 > "${suite_log}.rc"
        continue
    fi
    : > "$suite_log"
    worst=0
    while IFS= read -r tag; do
        safetag="${tag:-whole}"
        itemlog="$ITEM_LOG_DIR/${label}.${safetag}.log"
        note=""
        rc=$(cat "${itemlog}.rc" 2>/dev/null)
        if [[ ! "$rc" =~ ^[0-9]+$ ]]; then
            rc=1
            note="  <- no exit code recorded: the item never finished"
        fi
        printf '%-4s rc=%-4s items/%s%s\n' \
            "$([[ "$rc" == "0" ]] && echo ok || echo FAIL)" "$rc" \
            "${label}.${safetag}.log" "$note" >> "$suite_log"
        [[ "$rc" == "0" ]] && continue
        worst=$rc
        FAILED_ITEMS+=( "${itemlog#"${REPO_ROOT}/"}" )
        FAILED_KEYS+=( "$(printf '%s\t%s' "$label" "$tag")" )
    done < <(awk -F'\t' -v l="$label" '$2==l {print $4}' "$QUEUE_FILE")
    echo "$worst" > "${suite_log}.rc"
    [[ "$worst" != "0" ]] && OVERALL_RC=$worst
done

# What a re-run of this job should queue instead of the whole config. Written on
# every run, pass or fail: a green run has to leave an empty list behind rather
# than the previous attempt's stale one.
#
# The list converges across attempts on its own. An item that passed was never
# queued on the next attempt, so it cannot reappear here, and each attempt's list
# is a subset of the one before it.
if [[ ${#FAILED_KEYS[@]} -gt 0 ]]; then
    printf '%s\n' "${FAILED_KEYS[@]}" > "$RERUN_FILE"
else
    : > "$RERUN_FILE"
fi

# And how much of each of those items has to come back. Rebuilt from nothing
# every run: a list left over from an earlier attempt would send an item back to
# tests that are no longer the reason it fails, and an item that has since begun
# failing at collection would keep a narrowing it must no longer get.
rm -rf "$RERUN_TESTS_DIR"
if [[ ${#FAILED_KEYS[@]} -gt 0 && -n "${JUNITXML_PREFIX:-}" ]]; then
    echo "== Re-run granularity: which failed items can come back as single tests =="
    printf '%s\n' "${FAILED_KEYS[@]}" \
        | python3 "$REPO_ROOT/ci/scheduler/rerun_tests.py" \
              --junit-prefix "$JUNITXML_PREFIX" --junit-suffix "${JUNITXML_SUFFIX:-}" \
              -o "$RERUN_TESTS_DIR" \
        || log_warn "could not work out per-test re-runs; each failed item will re-run whole"
fi

# ---------------------------------------------------------------------------
# Phase 6: update the learned weight table for the next run
# ---------------------------------------------------------------------------
echo
if [[ -n "$RERUN_MODE" ]]; then
    # Two reasons, either of which alone is enough. An item narrowed to its
    # failing nodeids ran a handful of its tests, so its duration measures the
    # re-run and not the item -- there is no contention model that recovers the
    # other 300 tests it did not run. And even an item that came back whole ran
    # on a box with most of its GPUs idle, without the CPU, memory and PCIe
    # contention of a full queue, so it lands faster than it would in the run
    # this table is meant to schedule. Both drag weights downward. The attempt
    # being re-run already merged its own measurements, so sitting this one out
    # costs the table nothing.
    echo "== Weights: left alone -- a re-run's timings are not a full queue's =="
elif ! python3 "$REPO_ROOT/ci/scheduler/build_weights.py" update "$TIMINGS_FILE" \
        --items "$ITEMS_FILE" -o "$WEIGHTS_FILE"; then
    log_warn "could not update $WEIGHTS_FILE; the next run will use the table as it stands"
fi

# ---------------------------------------------------------------------------
# Phase 7: scheduling report
# ---------------------------------------------------------------------------
if ! python3 "$REPO_ROOT/ci/scheduler/schedule_report.py" "$LOG_DIR" \
        --gpus "${GPU_IDS[*]}" \
        --wall "$WALL" \
        --expand-secs "$EXPAND_SECS" \
        --setup-secs "$SETUP_SECS" \
        --total-wall "$(( $(date +%s) - SCRIPT_START_TS ))" \
        --default-weight "$DEFAULT_WEIGHT" \
        --weights "$WEIGHTS_FILE" \
        ${RERUN_MODE:+--rerun}; then
    log_warn "could not write the scheduling report"
fi

# ---------------------------------------------------------------------------
# Phase 8: failure summary
# ---------------------------------------------------------------------------
if [[ ${#FAILED_ITEMS[@]} -gt 0 ]]; then
    echo
    echo "== ${#FAILED_ITEMS[@]} of ${TOTAL_ITEMS} items FAILED =="
    printf '  %s\n' "${FAILED_ITEMS[@]}"
    echo "  (re-running this job queues these ${#FAILED_ITEMS[@]} items and nothing else)"
fi

exit $OVERALL_RC
