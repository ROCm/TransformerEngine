#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run test suites as one global work queue across the GPUs of this box.
#
# Usage: run_queue.sh [-q|--queue sgpu|mgpu] [-s|--suites "<label>..."] [-l|--log-dir <dir>]
#                     [<config>...]
#
# Two queues share this script, selected by --queue (default sgpu):
#
#   sgpu  the suites' single-GPU tests (TEST_SGPU=1); every item takes one GPU.
#         Default config ci/ci_sgpu_queue.conf.
#   mgpu  the suites' multi-GPU tests (TEST_MGPU=1); every item takes the whole
#         box. Default config ci/ci_mgpu_queue.conf.
#
# Config format (one suite per line; # comments and blank lines are ignored):
#   <label>  <logfile>  <mode>  <command> [args...]
#     mode=list    expanded into one work item per test invocation
#     mode=opaque  scheduled as a single work item
#
# The queue uses every GPU it can see; restrict it with HIP_VISIBLE_DEVICES.
#
# --suites (or TE_CI_SUITES) runs only the named suites; the others get a passing
# "not selected" verdict. CI sets it from .github/scripts/change_scope.py.
#
# Example usage:
#   TEST_LEVEL=1 ci/run_queue.sh
#   HIP_VISIBLE_DEVICES=0,1 TEST_LEVEL=1 ci/run_queue.sh
#   TEST_LEVEL=3 ci/run_queue.sh --queue mgpu
#   TEST_LEVEL=1 ci/run_queue.sh --suites "torch examples"
set -u
SCRIPT_START_TS=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Phase 4 hands GPUs out with `wait -n -p`, which bash grew in 5.1
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Error: bash 5.1 or newer is required (this is ${BASH_VERSION})" >&2
    exit 1
fi

# All run data lives under TE_CI_DATA_DIR (default ci-data/):
#   test-results/  this run's results; this script writes logs/ under it
#   weights/       learned item durations (sGPU queue only)
#   rerun/         what failed, so a re-run of the job queues only that
: "${TE_CI_DATA_DIR:=ci-data}"
[[ "$TE_CI_DATA_DIR" == /* ]] || TE_CI_DATA_DIR="${REPO_ROOT}/${TE_CI_DATA_DIR}"
: "${LOG_DIR:=${TE_CI_DATA_DIR}/test-results/logs}"

# Weight of an item with no recorded duration. It is large so new items start
# first: a long item started late is what stretches the run.
: "${DEFAULT_WEIGHT:=${TE_CI_DEFAULT_WEIGHT:-999999}}"

# Cross-phase state. The SUITE_* arrays are filled by Phase 0 and share an index.
declare -a CONFIGS=()
declare -a GPU_IDS=()
declare -a SUITE_LABELS=()
declare -a SUITE_LOGFILES=()
declare -a SUITE_MODES=()
declare -a SUITE_CMDS=()
declare -a SUITE_ARGS=()
# Suites --suites left out; each still gets a verdict file
declare -a SKIPPED_LABELS=()
declare -a SKIPPED_LOGFILES=()
declare -a SKIPPED_OPAQUE=()
GPU_SOURCE=""
OVERALL_RC=0

# Run a ci/scheduler/ script with its directory on PYTHONPATH. The scripts import
# their shared queue_files module by bare name, which PYTHONSAFEPATH=1 (exported
# by ci/_utils.sh, sourced in Phase 3) would otherwise prevent.
SCHEDULER_DIR="${REPO_ROOT}/ci/scheduler"
scheduler_py() {
    local script="$1"; shift
    PYTHONPATH="${SCHEDULER_DIR}${PYTHONPATH:+:$PYTHONPATH}" python3 "${SCHEDULER_DIR}/${script}" "$@"
}

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    log_error() { echo "::error::$*" >&2; }
    log_warn()  { echo "::warning::$*" >&2; }
else
    log_error() { echo "Error: $*" >&2; }
    log_warn()  { echo "Warning: $*" >&2; }
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
QUEUE="sgpu"
# Set, even to nothing, means "only these"; unset means every suite
SELECTED_SUITES="${TE_CI_SUITES-}"
FILTER_SUITES="${TE_CI_SUITES+1}"
USAGE="Usage: $0 [-q|--queue sgpu|mgpu] [-s|--suites \"<label>...\"] [-l|--log-dir <dir>] [<config>...]"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--suites)
            SELECTED_SUITES="$2"; FILTER_SUITES=1; shift 2 ;;
        --suites=*)
            SELECTED_SUITES="${1#*=}"; FILTER_SUITES=1; shift ;;
        -q|--queue)
            QUEUE="$2"; shift 2 ;;
        --queue=*)
            QUEUE="${1#*=}"; shift ;;
        -l|--log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#*=}"; shift ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "$USAGE" >&2
            exit 1 ;;
        *)
            break ;;
    esac
done

# The suite scripts pick their sGPU or mGPU tests from TEST_SGPU/TEST_MGPU.
case "$QUEUE" in
    sgpu) export TEST_SGPU=1 TEST_MGPU="" ;;
    mgpu) export TEST_SGPU="" TEST_MGPU=1 ;;
    *)
        echo "Unknown queue: ${QUEUE}" >&2
        echo "$USAGE" >&2
        exit 1 ;;
esac

# Resolve config paths to absolute
if [[ $# -gt 0 ]]; then
    for c in "$@"; do CONFIGS+=( "$(realpath -m "$c")" ); done
else
    CONFIGS=( "${SCRIPT_DIR}/ci_${QUEUE}_queue.conf" )
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

# Keep only the --suites asked for. An unknown name is an error, not a no-op:
# change_scope.py and the configs disagree, and running nothing would pass the PR.
if [[ -n "$FILTER_SUITES" ]]; then
    read -r -a wanted <<< "$SELECTED_SUITES"
    for name in "${wanted[@]}"; do
        if ! printf '%s\n' "${SUITE_LABELS[@]}" | grep -qxF -- "$name"; then
            log_error "--suites names '${name}', which ${CONFIGS[*]} do not define"
            exit 1
        fi
    done
    declare -a kept=()
    for i in "${!SUITE_LABELS[@]}"; do
        if printf '%s\n' "${wanted[@]}" | grep -qxF -- "${SUITE_LABELS[$i]}"; then
            kept+=( "$i" )
        else
            SKIPPED_LABELS+=( "${SUITE_LABELS[$i]}" )
            SKIPPED_LOGFILES+=( "${SUITE_LOGFILES[$i]}" )
            # Keep a skipped opaque suite in items.tsv so Phase 6 does not
            # prune its weight as if the suite were gone.
            [[ "${SUITE_MODES[$i]}" == "list" ]] || SKIPPED_OPAQUE+=( "${SUITE_LABELS[$i]}" )
        fi
    done
    for array in SUITE_LABELS SUITE_LOGFILES SUITE_MODES SUITE_CMDS SUITE_ARGS; do
        declare -n src=$array
        declare -a picked=()
        for i in "${kept[@]}"; do picked+=( "${src[$i]}" ); done
        src=( "${picked[@]}" )
        unset -n src
    done
    echo "== Suites: running ${SUITE_LABELS[*]:-none}; not selected: ${SKIPPED_LABELS[*]:-none} =="
fi

# Usage: write_skipped_verdicts <suite log dir>
# Give each suite --suites left out a passing verdict; the workflow treats a
# missing verdict as a failure.
write_skipped_verdicts() {
    local k
    for k in "${!SKIPPED_LABELS[@]}"; do
        echo "not run: not selected (--suites), e.g. the PR changes nothing it tests" \
            > "$1/${SKIPPED_LOGFILES[$k]}"
        echo 0 > "$1/${SKIPPED_LOGFILES[$k]}.rc"
    done
}

# Nothing left to run needs no GPU, no setup and no report
if [[ ${#SUITE_LABELS[@]} -eq 0 ]]; then
    mkdir -p "$LOG_DIR/suites"
    write_skipped_verdicts "$LOG_DIR/suites"
    echo "== No suite selected: nothing to run =="
    exit 0
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

# Learned item durations, keyed by arch, TEST_LEVEL and GPU count. Only the sGPU
# queue keeps them: mGPU items run one at a time on the whole box, so their order
# does not change the wall time.
WEIGHTS_FILE=""
if [[ "$QUEUE" == "sgpu" ]]; then
    WEIGHTS_FILE="${TE_CI_DATA_DIR}/weights/test_weights.${QUEUE}.${ARCH}.l${TEST_LEVEL:-99}.g${NUM_GPUS}.txt"
    mkdir -p "$(dirname "$WEIGHTS_FILE")" 2>/dev/null
fi

# Items that failed the previous attempt (label, tag). Phase 5 writes it; when it
# is non-empty, Phase 2 queues only those items ("Re-run failed jobs").
RERUN_DIR="${TE_CI_DATA_DIR}/rerun/${QUEUE}"
RERUN_FILE="${RERUN_DIR}/failed.${ARCH}.l${TEST_LEVEL:-99}.tsv"
mkdir -p "$RERUN_DIR" 2>/dev/null
RERUN_MODE=""   # set by Phase 2 once it has actually narrowed the queue

# Per item, the failing nodeids to re-run (<label>.<tag>.txt). Optional: an item
# with no file here re-runs whole.
RERUN_TESTS_DIR="${RERUN_DIR}/tests"

# GPUs per item: one in the sGPU queue, the whole box in the mGPU queue (the
# tests adapt to the GPU count they see, skipping cases when given fewer).
case "$QUEUE" in
    sgpu) DEFAULT_ITEM_GPUS=1 ;;
    mgpu) DEFAULT_ITEM_GPUS=$NUM_GPUS ;;
esac

[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(realpath -m "$LOG_DIR")"

# Items run in their own directories (Phase 4), so make a relative JUnit prefix
# absolute. Keep any trailing slash: the prefix is concatenated, not joined.
if [[ -n "${JUNITXML_PREFIX:-}" && "$JUNITXML_PREFIX" != /* ]]; then
    junit_prefix_sep=""
    [[ "$JUNITXML_PREFIX" == */ ]] && junit_prefix_sep=/
    JUNITXML_PREFIX="$(realpath -m "$JUNITXML_PREFIX")${junit_prefix_sep}"
    export JUNITXML_PREFIX
fi

# Layout of LOG_DIR:
#   prerequisite_ck_jit_status/  Phase 3  CK JIT prebuild and pip setup logs
#   items/                       Phase 4  one log per item
#   cwd/                         Phase 4  per-item working dirs (empty ones removed)
#   suites/                      Phase 5  per-suite verdict: rc and item index
#   report/                      Phase 7  the schedule report
#   queue/                                queue, census and timings (TSV)
SETUP_DIR="$LOG_DIR/prerequisite_ck_jit_status"
ITEM_LOG_DIR="$LOG_DIR/items"
ITEM_CWD_DIR="$LOG_DIR/cwd"
SUITE_LOG_DIR="$LOG_DIR/suites"
REPORT_DIR="$LOG_DIR/report"
QUEUE_DIR="$LOG_DIR/queue"

rm -rf "$SETUP_DIR" "$ITEM_LOG_DIR" "$ITEM_CWD_DIR" "$SUITE_LOG_DIR" "$REPORT_DIR" "$QUEUE_DIR"

mkdir -p "$SETUP_DIR" "$ITEM_LOG_DIR" "$ITEM_CWD_DIR" "$SUITE_LOG_DIR" "$REPORT_DIR" "$QUEUE_DIR"

# Suites are expanded from the repo root; items run in their own dirs (Phase 4).
cd "$REPO_ROOT" || { echo "Error: cannot cd to '${REPO_ROOT}'" >&2; exit 1; }

QUEUE_FILE="$QUEUE_DIR/queue.tsv"        # Phase 2 writes, Phases 4 and 5 read
ITEMS_FILE="$QUEUE_DIR/items.tsv"        # Phase 1 writes, Phase 6 reads
TIMINGS_FILE="$QUEUE_DIR/timings.tsv"    # Phase 4 writes, Phases 6 and 7 read
: > "$QUEUE_FILE"
: > "$ITEMS_FILE"
write_skipped_verdicts "$SUITE_LOG_DIR"
for label in "${SKIPPED_OPAQUE[@]}"; do
    printf '%s\t%s\n' "$label" "" >> "$ITEMS_FILE"
done

# ---------------------------------------------------------------------------
# Phase 1: expand every suite into work items
# ---------------------------------------------------------------------------
# With TE_CI_LIST_ITEMS=1 a suite script prints "TE_CI_ITEM <tag>" for each test
# invocation instead of running it. Each list-mode suite is listed twice:
#
#   list      what this host will run    -> the queue
#   list-all  what exists at this level  -> items.tsv (TE_CI_SKIP_CHECK_SUPPORTED=1)
#
# Phase 6 prunes weights against items.tsv, so a test this host skips (no
# flash-attn, say) keeps its weight.
EXPAND_LOG="$LOG_DIR/expand.log"   # whatever the suites printed while listing
LIST_TMP="$QUEUE_DIR/.expand.tmp"  # one suite's list, reused per suite
: > "$EXPAND_LOG"
# Start from an empty raw queue, so tags from an aborted run cannot leak in
: > "$QUEUE_FILE.raw"

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
            printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$cmd" "$tag" "${rest:-}" "$DEFAULT_ITEM_GPUS" \
                >> "$QUEUE_FILE.raw"
        done

        echo "=== ${label}: list-all -- what exists at this level ===" >> "$EXPAND_LOG"
        TE_CI_LIST_ITEMS=1 TE_CI_SKIP_CHECK_SUPPORTED=1 "$cmd" ${rest:-} > "$LIST_TMP" 2>> "$EXPAND_LOG"
        all_rc=$?
        mapfile -t all_tags < <(sed -n 's/^TE_CI_ITEM \([^ ]*\).*/\1/p' "$LIST_TMP")

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
        printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$cmd" "" "${rest:-}" "$DEFAULT_ITEM_GPUS" \
            >> "$QUEUE_FILE.raw"
        printf '%s\t%s\n' "$label" "" >> "$ITEMS_FILE"
        echo "  ${label}: 1 item (opaque)"
    fi
done
rm -f "$LIST_TMP"

# A tag is both the item's TEST_FILTER key and its JUnit XML name, so two call
# lines in a suite must not share one; give such lines distinct labels.
dupe_tags=$(awk -F'\t' '$3 != "" { print $1 "/" $3 }' "$QUEUE_FILE.raw" | sort | uniq -d)
if [[ -n "$dupe_tags" ]]; then
    log_error "duplicate work item tags; give each call site a distinct label:"
    echo "$dupe_tags" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: narrow the queue to a re-run, then weight and order it
# ---------------------------------------------------------------------------
# On a re-run, keep only the items that failed last time. Phase 1 still listed
# every suite, so items.tsv remains a full census.
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
        # None of them is in this queue any more (the config changed). Run
        # everything: an empty queue would test nothing and pass.
        rm -f "$QUEUE_FILE.rerun"
        log_warn "none of the ${n_failed} items in ${RERUN_FILE##*/} are in this" \
                 "runner's queue; running everything instead"
    fi
fi

if ! scheduler_py build_weights.py order "$QUEUE_FILE.raw" \
        ${WEIGHTS_FILE:+--weights "$WEIGHTS_FILE"} \
        --output "$QUEUE_FILE" \
        --default-weight "$DEFAULT_WEIGHT" \
        --gpus "${GPU_IDS[*]}"; then
    log_error "could not order the queue"
    exit 1
fi
rm -f "$QUEUE_FILE.raw"

TOTAL_ITEMS=$(wc -l < "$QUEUE_FILE")

# ---------------------------------------------------------------------------
# Phase 3: one-time prerequisites
# ---------------------------------------------------------------------------
# The CK JIT cache is keyed by GPU arch and shared by every suite, so it is built
# once here; TE_CI_SKIP_CK_JIT stops the suites rebuilding it. Pip prerequisites
# are per framework, so each list-mode suite installs its own once
# (TE_CI_SETUP_ONLY). A suite script run on its own still does both inline.

# ck_jit_prebuild comes from _utils.sh, which resolves TE_PATH from DIR.
DIR="$REPO_ROOT/ci"
# shellcheck source=/dev/null
source "$REPO_ROOT/ci/_utils.sh"

# The prebuild runs in a subshell and the drift check after the queue, so the
# cache snapshot between them goes through a file.
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
    rest="${SUITE_ARGS[$i]}"
    # Opaque suites do their own setup when they run
    [[ "${SUITE_MODES[$i]}" == "list" ]] || continue
    # Skip a suite with nothing queued (a re-run where all of it passed)
    awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE" || continue
    if [[ -z "$setup_banner" ]]; then
        echo "== One-time setup: pip prerequisites (once per suite) =="
        setup_banner=1
    fi
    printf '  %s: installing prerequisites (%s) ... ' "$label" "$cmd"
    setup_start=$(date +%s)
    if ! HIP_VISIBLE_DEVICES=${GPU_IDS[0]} TE_CI_SETUP_ONLY=1 TE_CI_SKIP_CK_JIT=1 "$cmd" ${rest:-} \
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
# ---------------------------------------------------------------------------
# Items are dispatched strictly in queue order (widest first, then longest first;
# see build_weights.py), each to as many GPUs as it asks for. When the next item
# does not fit, the dispatcher waits rather than letting a later one jump ahead,
# so narrow items cannot starve a wide one. In the sGPU queue every item takes
# one GPU, so the next item simply goes to whichever GPU frees first.

# Base of the torchrun rendezvous port each item is given, see run_item
: "${TE_CI_PORT_BASE:=29600}"

# Usage: run_item <gpus> <weight> <label> <cmd> <tag> <rest>
# Run one item on <gpus> (comma-separated device ids) and append its timing
# record to TIMINGS_FILE.
run_item() {
    local gpus=$1 weight=$2 label=$3 cmd=$4 tag=$5 rest=$6
    local itemlog rc safetag junit_dir start end incomplete itemcwd only_tests port

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

    # On a re-run, limit a list-mode item to its failing tests if a list was
    # left for it; empty runs the whole item.
    only_tests="$RERUN_TESTS_DIR/${label}.${tag}.txt"
    [[ -n "$tag" && -n "$junit_dir" && -s "$only_tests" ]] || only_tests=""

    # Give each item its own default torchrun port (PET_MASTER_PORT), so items
    # running at once do not collide on 29500. Running items never share a
    # GPU, so the first GPU id makes the port unique.
    port=$(( TE_CI_PORT_BASE + ${gpus%%,*} ))

    # No timeout here: the suites' PYTEST_TIMEOUT and the workflow's limits apply
    start=$(date +%s)
    if [[ -n "$tag" ]]; then
        ( cd "$itemcwd" && HIP_VISIBLE_DEVICES=$gpus PET_MASTER_PORT=$port \
            TE_CI_SKIP_SETUP=1 TEST_FILTER="$tag" \
            JUNITXML_PREFIX="$junit_dir" TE_CI_ONLY_TESTS="$only_tests" \
            "$cmd" ${rest:-} ) < /dev/null > "$itemlog" 2>&1
    else
        ( cd "$itemcwd" && HIP_VISIBLE_DEVICES=$gpus PET_MASTER_PORT=$port \
            JUNITXML_PREFIX="$junit_dir" \
            "$cmd" ${rest:-} ) < /dev/null > "$itemlog" 2>&1
    fi
    rc=$?
    end=$(date +%s)

    # Keep the working directory only if the item wrote something to it
    rmdir "$itemcwd" 2>/dev/null

    echo "$rc" > "${itemlog}.rc"

    # A leftover result-sink sidecar (.partial) means pytest died mid-run
    # (timeout, segfault, OOM), so this duration is not a full measurement and
    # build_weights.py skips it. rc cannot tell: a thread timeout also exits 1.
    incomplete=0
    if [[ -n "$junit_dir" ]]; then
        if [[ -n "$tag" ]]; then
            # Check only this item's sidecar: a suite's items share junit_dir
            [[ -e "${junit_dir}${tag}${JUNITXML_SUFFIX:-}.partial" ]] && incomplete=1
        elif compgen -G "${junit_dir}*.partial" > /dev/null; then
            incomplete=1   # opaque suite: it is the only item in its dir
        fi
    fi

    # One timings.tsv row per item as it ends (Phase 6 learns weights from it):
    #   label  tag  gpus  secs  rc  start_off  end_off  est  incomplete
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$safetag" "$gpus" "$((end - start))" "$rc" \
        "$((start - START_TS))" "$((end - START_TS))" "$weight" "$incomplete" \
        >> "$TIMINGS_FILE"

    # Console progress, one line per item (no estimate without a weight table):
    #   time  gpus  start offset  duration  rc  estimate  suite  item
    local est=""
    if [[ -n "$WEIGHTS_FILE" ]]; then
        est="est=$([[ $weight -eq $DEFAULT_WEIGHT ]] && echo unknown || echo "${weight}s")"
    fi
    printf '[%s] gpu%-8s t+%-7s %5ss rc=%-4s %-12s %-9s %s\n' \
        "$(date '+%H:%M:%S')" "$gpus" "$((start - START_TS))s" "$((end - start))" \
        "$rc" "$est" "$label" "$safetag"
}

# GPUs held by each running item, keyed by its run_item pid
declare -A HELD_BY_PID=()
declare -A BUSY_GPU=()

# Echo the free GPUs, in GPU_IDS order
free_gpus() {
    local gpu
    for gpu in "${GPU_IDS[@]}"; do
        [[ -n "${BUSY_GPU[$gpu]:-}" ]] || echo "$gpu"
    done
}

# Block until one running item finishes, and give its GPUs back
reap_one() {
    local pid="" gpu
    local -a held
    wait -n -p pid "${!HELD_BY_PID[@]}"
    [[ -n "$pid" ]] || return 1
    IFS=, read -r -a held <<< "${HELD_BY_PID[$pid]}"
    for gpu in "${held[@]}"; do unset 'BUSY_GPU[$gpu]'; done
    unset 'HELD_BY_PID[$pid]'
}

: > "$TIMINGS_FILE"
START_TS=$(date +%s)

mapfile -t QUEUE_LINES < "$QUEUE_FILE"
for line in "${QUEUE_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    # read collapses consecutive tabs (an empty tag or rest), so split on \x1f
    IFS=$'\x1f' read -r weight label cmd tag rest gpus_wanted <<< "${line//$'\t'/$'\x1f'}"
    mapfile -t free < <(free_gpus)
    while (( ${#free[@]} < gpus_wanted )); do
        reap_one || { log_error "dispatcher lost track of its running items"; exit 1; }
        mapfile -t free < <(free_gpus)
    done
    assigned=( "${free[@]:0:gpus_wanted}" )
    for gpu in "${assigned[@]}"; do BUSY_GPU[$gpu]=1; done
    assigned_csv=$(IFS=,; echo "${assigned[*]}")
    run_item "$assigned_csv" "$weight" "$label" "$cmd" "$tag" "$rest" &
    HELD_BY_PID[$!]=$assigned_csv
done
while (( ${#HELD_BY_PID[@]} > 0 )); do
    reap_one || break
done
wait

WALL=$(( $(date +%s) - START_TS ))
# The report divides by WALL; keep it nonzero for a sub-second run
[[ $WALL -lt 1 ]] && WALL=1

# --- CK JIT cache drift ----------------------------------------------------
# Blobs the queue JIT-compiled that ci/ck_jit_prebuild.txt does not list. Checked
# once after the queue, since every item shares the cache.
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
    # A marker in the suite's JUnit dir tells junit_report.py the suite was not
    # re-run, so its empty results dir is not reported as a crash
    suite_junit_dir=""
    if [[ -n "${JUNITXML_PREFIX:-}${JUNITXML_SUFFIX:-}" ]]; then
        suite_junit_dir="${JUNITXML_PREFIX:-}${label}/"
        rm -f "${suite_junit_dir}not_rerun"
    fi
    if ! awk -F'\t' -v l="$label" '$2==l {found=1} END {exit !found}' "$QUEUE_FILE"; then
        # Nothing of this suite was queued: on a re-run, all of it passed last
        # time, so record a pass (the workflow reads a missing verdict as a
        # failure). Outside a re-run Phase 1 already failed an empty suite.
        [[ -n "$RERUN_MODE" ]] || continue
        echo "not re-run this attempt: no item of this suite failed the last one" > "$suite_log"
        echo 0 > "${suite_log}.rc"
        if [[ -n "$suite_junit_dir" ]]; then
            mkdir -p "$suite_junit_dir"
            cp "$suite_log" "${suite_junit_dir}not_rerun"
        fi
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

# What a re-run of this job should queue. Written even when empty, so a passing
# run clears the previous attempt's list.
if [[ ${#FAILED_KEYS[@]} -gt 0 ]]; then
    printf '%s\n' "${FAILED_KEYS[@]}" > "$RERUN_FILE"
else
    : > "$RERUN_FILE"
fi

# Which of those items can re-run just their failing tests. Rebuilt every run,
# so no list from an earlier attempt survives.
rm -rf "$RERUN_TESTS_DIR"
if [[ ${#FAILED_KEYS[@]} -gt 0 && -n "${JUNITXML_PREFIX:-}" ]]; then
    echo "== Re-run granularity: which failed items can come back as single tests =="
    printf '%s\n' "${FAILED_KEYS[@]}" \
        | scheduler_py rerun_tests.py \
              --junit-prefix "$JUNITXML_PREFIX" --junit-suffix "${JUNITXML_SUFFIX:-}" \
              -o "$RERUN_TESTS_DIR" \
        || log_warn "could not work out per-test re-runs; each failed item will re-run whole"
fi

# ---------------------------------------------------------------------------
# Phase 6: update the learned weight table for the next run
# ---------------------------------------------------------------------------
[[ -n "$WEIGHTS_FILE" ]] && echo
if [[ -z "$WEIGHTS_FILE" ]]; then
    :   # this queue keeps no weight table
elif [[ -n "$RERUN_MODE" ]]; then
    # A re-run's timings understate a full queue's: narrowed items ran only a
    # few tests, on a mostly idle box. The attempt being re-run already updated
    # the table.
    echo "== Weights: left alone -- a re-run's timings are not a full queue's =="
elif ! scheduler_py build_weights.py update "$TIMINGS_FILE" \
        --items "$ITEMS_FILE" -o "$WEIGHTS_FILE"; then
    log_warn "could not update $WEIGHTS_FILE; the next run will use the table as it stands"
fi

# ---------------------------------------------------------------------------
# Phase 7: scheduling report
# ---------------------------------------------------------------------------
case "$QUEUE" in
    sgpu) REPORT_TITLE="sGPU queue schedule" ;;
    mgpu) REPORT_TITLE="mGPU queue schedule" ;;
esac
if ! scheduler_py schedule_report.py "$LOG_DIR" \
        --title "$REPORT_TITLE" \
        --gpus "${GPU_IDS[*]}" \
        --wall "$WALL" \
        --total-wall "$(( $(date +%s) - SCRIPT_START_TS ))" \
        --default-weight "$DEFAULT_WEIGHT" \
        ${WEIGHTS_FILE:+--weights "$WEIGHTS_FILE"} \
        --skipped "${SKIPPED_LABELS[*]}" \
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
