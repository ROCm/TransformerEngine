#!/bin/bash
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run multiple sGPU jobs in parallel from one or more config files.
#
# Usage: run_parallel_sgpu.sh [-g|--first-gpu <n>] [-l|--log-dir <dir>] <config>...
#
# Config format (one job per line; # comments and blank lines are ignored):
#   <label>  <logfile>  <command> [args...]
#
# GPUs are assigned sequentially starting from --first-gpu across all config
# entries in the order configs are passed on the command line.
#
# Each job's exit code is written to <log-dir>/<logfile>.rc by the job runner.

# Resolve repo root relative to this script's location (.github/scripts/ -> ../..)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# How often (seconds) to poll child processes
: "${POLL_INTERVAL:=5}"
# Warn if a log file has not been updated in this many seconds
: "${STALL_WARN_SECS:=180}"
# Number of tail lines to show for stalled logs
: "${STALL_TAIL_LINES:=3}"
# Number of new lines to show for resumed logs
: "${STALL_RESUME_CONTEXT_LINES:=2}"

# Associative arrays:  _JOB_PIDS[name]=pid   _JOB_LOGS[pid]=logfile
declare -A _JOB_PIDS
declare -A _JOB_LOGS
_OVERALL_RC=0

# Launch a background job and register it.
# Usage: launch_job <name> <logfile> <cmd> [args...]
launch_job() {
    local name="$1"
    local logfile="$2"
    local rcfile="${logfile}.rc"
    shift 2
    rm -f "$rcfile"
    "$@" >"$logfile" 2>&1 &
    local pid=$!
    _JOB_PIDS["$name"]=$pid
    _JOB_LOGS[$pid]="$logfile"
    echo "Started '${name}' (pid ${pid}) -> ${logfile}"
}

# Wait for all currently registered jobs, polling every POLL_INTERVAL seconds.
# Writes <logfile>.rc for every finished job; warns about stalled logs.
# Clears _JOB_PIDS/_JOB_LOGS when done.
wait_for_jobs() {
    local -A remaining
    local -A stall_mtime
    local -A stall_lineno
    for name in "${!_JOB_PIDS[@]}"; do
        remaining["$name"]="${_JOB_PIDS[$name]}"
    done

    while [ ${#remaining[@]} -gt 0 ]; do
        sleep "$POLL_INTERVAL"

        for name in "${!remaining[@]}"; do
            local pid="${remaining[$name]}"

            if ! kill -0 "$pid" 2>/dev/null; then
                # Process has exited — capture its return code
                wait "$pid"
                local rc=$?
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] '${name}' (pid ${pid}) finished with rc=${rc}"
                if [ $rc -ne 0 ]; then
                    _OVERALL_RC=$rc
                fi
                echo "$rc" > "${_JOB_LOGS[$pid]}.rc"
                unset "remaining[$name]"
            else
                # Process still running — check for log staleness
                local logfile="${_JOB_LOGS[$pid]}"
                if [ -f "$logfile" ]; then
                    local now mtime age
                    now=$(date +%s)
                    mtime=$(stat -c '%Y' "$logfile" 2>/dev/null || echo "$now")
                    age=$(( now - mtime ))
                    if [ -n "${stall_mtime[$pid]+set}" ]; then
                        if [ "$mtime" -gt "${stall_mtime[$pid]}" ]; then
                            local frozen_secs=$(( mtime - stall_mtime[$pid] ))
                            local freeze_line="${stall_lineno[$pid]}"
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: '${name}' (pid ${pid}) log '${logfile}' resumed updating after ${frozen_secs}s"
                            echo "--- first ${STALL_RESUME_CONTEXT_LINES} lines of ${logfile} starting ${freeze_line} ---"                            
                            tail -n "+${freeze_line}" "$logfile" | head -n ${STALL_RESUME_CONTEXT_LINES}
                            echo "---"
                            unset "stall_mtime[$pid]"
                            unset "stall_lineno[$pid]"
                        fi
                        # else: still stalled but already warned — do nothing
                    elif [ "$age" -ge "$STALL_WARN_SECS" ]; then
                        # don't use wc here because it does not count the last line if it doesn't end with a newline
                        local freeze_line=$(grep -c '' < "$logfile")
                        stall_mtime[$pid]=$mtime
                        stall_lineno[$pid]=$freeze_line
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: '${name}' (pid ${pid}) log '${logfile}' has not been updated for ${age}s"
                        echo "--- last ${STALL_TAIL_LINES} lines of ${logfile} up to ${freeze_line} ---"
                        head -n "${freeze_line}" "$logfile" | tail -n "$STALL_TAIL_LINES"
                        echo "--- end of ${logfile} ---"
                    fi
                fi
            fi
        done
    done

    # Reset for next batch
    unset _JOB_PIDS
    unset _JOB_LOGS
    declare -gA _JOB_PIDS
    declare -gA _JOB_LOGS
}

FIRST_GPU=${TEST_FIRST_GPU:-0}
LOG_DIR=${LOG_DIR:-/tmp/te_ci_logs}

# ---------------------------------------------------------------------------
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--first-gpu)
            FIRST_GPU="$2"; shift 2 ;;
        --first-gpu=*)
            FIRST_GPU="${1#*=}"; shift ;;
        -l|--log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#*=}"; shift ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [-g|--first-gpu <n>] [-l|--log-dir <dir>] <config>..." >&2
            exit 1 ;;
        *)
            break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: at least one config file is required." >&2
    echo "Usage: $0 [-g|--first-gpu <n>] [-l|--log-dir <dir>] <config>..." >&2
    exit 1
fi

# Resolve config paths and LOG_DIR to absolute against the caller's CWD
# *before* changing directory, so relative paths passed by the caller remain valid.
resolved_configs=()
for c in "$@"; do
    resolved_configs+=( "$(realpath -m "$c")" )
done
[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(realpath -m "$LOG_DIR")"

mkdir -p "$LOG_DIR"

# cd to repo root so that commands in configs (e.g. ci/pytorch.sh) resolve correctly
cd "$REPO_ROOT" || { echo "Error: cannot cd to '${REPO_ROOT}'" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Launch all jobs, assigning GPUs sequentially across all config files
gpu=$FIRST_GPU
for config in "${resolved_configs[@]}"; do
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip blank lines and comments
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]]  && continue
        read -r label logfile rest <<< "$line"
        # Each suite appends its own subdir to the inherited base prefix so the base path is defined once.
        if [ -n "${JUNITXML_PREFIX}${JUNITXML_SUFFIX}" ]; then
            junitxml_dir="${JUNITXML_PREFIX}${label}/"
            mkdir -p "${junitxml_dir}"
        else
            junitxml_dir=""
        fi
        # shellcheck disable=SC2086  # $rest is intentionally word-split
        HIP_VISIBLE_DEVICES=$gpu JUNITXML_PREFIX="${junitxml_dir}" launch_job "$label" "${LOG_DIR}/$logfile" $rest
        (( gpu++ ))
    done < "$config"
done

wait_for_jobs
exit $_OVERALL_RC
