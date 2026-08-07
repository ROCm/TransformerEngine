# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# ROCM_PATH resolution
resolve_rocm_path() {
    if [ -n "${ROCM_PATH:-}" ]; then
        echo "$ROCM_PATH"
        return 0
    fi
    if command -v rocm-sdk >/dev/null 2>&1; then
        local _root
        _root="$(rocm-sdk path --root)"
        if [ -n "$_root" ] && [ -f "${_root}/bin/hipcc" ]; then
            echo "$_root"
            return 0
        fi
    fi
    if [ -d "/opt/rocm/core" ]; then
        echo /opt/rocm/core
        return 0
    fi
    if [ -d "/opt/rocm" ]; then
        echo /opt/rocm
        return 0
    fi
    echo "Could not find ROCm installation" >&2
    exit 1
}

REALPATH=realpath
realpath $DIR >/dev/null 2>/dev/null
test $? -ne 0 && REALPATH=echo

: ${TE_PATH:=`$REALPATH $DIR/..`/}
export TE_PATH
TEST_DIR=${TE_PATH}tests/

: ${TEST_LEVEL:=99} #Run all tests by default
TEST_JOBS_MODE=""

if [ -z "${TEST_SGPU}${TEST_MGPU}" ]; then
    TEST_SGPU=1
    TEST_MGPU=1
fi

TEST_START_TS=`date +%s`

#To disable some logs trimming
export CI=1

# Crash/hang visibility and bounding:
# - PYTHONFAULTHANDLER dumps a Python traceback on fatal signals (segfaults).
# - PYTEST_TIMEOUT bounds every individual test item so a single hang cannot
#   stall the whole CI job; the offending test is recorded as a failure with a
#   traceback instead of the run silently timing out hours later.
# Note: the 'thread' method bounds only the pytest process itself. Tests that
# launch torchrun/mpirun children (tests/pytorch/distributed) are reaped
# separately by tests/pytorch/distributed/conftest.py, which reads PYTEST_TIMEOUT
# to bound each child below this outer limit -- hence the exports below.
# All are overridable from the environment.
export PYTHONFAULTHANDLER=1
export PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-600}               # per-test (per-parametrization) timeout, seconds
export PYTEST_TIMEOUT_METHOD=${PYTEST_TIMEOUT_METHOD:-thread} # unstick a hung main thread; see note above
export CTEST_TIMEOUT=${CTEST_TIMEOUT:-300}                 # per-cpp-test timeout, seconds
# Tests run from the TE root, where the checkout would otherwise shadow the installed
# package: `python -m pytest` prepends cwd to sys.path, and `python <script>` prepends
# the script's directory. The checkout has no compiled libraries and none of the files
# generated at build time, while the .so lookup falls back to site-packages -- so an
# import landing there runs source-tree Python against installed native libraries, with
# mismatched halves. On ROCm wheels it also skips the rocm-sdk preload, which segfaults
# during test collection. PYTHONSAFEPATH drops both implicit sys.path entries, so the
# installed package always wins; being an env var, it applies to torchrun/mpirun children
# and subprocesses too. Editable installs are unaffected: they resolve through a finder
# in site-packages, not through cwd. Note that any non-empty value enables it -- unset
# it, rather than setting 0, to opt out.
export PYTHONSAFEPATH=${PYTHONSAFEPATH:-1}

_script_error_count=0
_run_error_count=0
_ignored_error_count=0
_seen_test_tags=""
TEST_ERROR_IGNORE=""

script_error() {
    _script_error_count=$((_script_error_count+1))
    test "$@" && echo $@ >&2
}

test_run_error() {
    if [ -n "$TEST_ERROR_IGNORE" ]; then
        _ignored_error_count=$((_ignored_error_count+1))
        test -n "$@" && echo "Ignore error in test $@" >&2
        return
    fi
    _run_error_count=$((_run_error_count+1))
    test -n "$@" && echo "Error in test $@" >&2
}

return_run_results() {
    test $_script_error_count -ne 0 && echo Detected $_script_error_count script errors during tests run at level $TEST_LEVEL >&2
    test $_run_error_count -ne 0 && echo Got $_run_error_count test errors during run at level $TEST_LEVEL >&2
    test $_ignored_error_count -ne 0 && echo Ignored $_ignored_error_count test errors during run at level $TEST_LEVEL >&2
    test $_run_error_count -eq 0 -a $_script_error_count -eq 0
}

configure_fused_attn_env() {
    case "$1" in
        "auto")
            unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_FUSED_ATTN_AOTRITON NVTE_FUSED_ATTN_CK
        ;;
        "aotriton")
            export NVTE_FLASH_ATTN=0
            export NVTE_FUSED_ATTN_CK=0
            unset NVTE_FUSED_ATTN NVTE_FUSED_ATTN_AOTRITON
        ;;
        "ck")
            export NVTE_FLASH_ATTN=0
            export NVTE_FUSED_ATTN_AOTRITON=0
            unset NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK
        ;;
        "flash")
            export NVTE_FUSED_ATTN=0 NVTE_FUSED_ATTN_CK=0 NVTE_FUSED_ATTN_AOTRITON=0
            unset NVTE_FLASH_ATTN
        ;;
        "unfused")
            export NVTE_FLASH_ATTN=0
            export NVTE_FUSED_ATTN=0
            unset NVTE_FUSED_ATTN_AOTRITON NVTE_FUSED_ATTN_CK
        ;;
        *)
            script_error "Error unknown fused attention config $1"
            return 1
        ;;
    esac
    return 0
}

check_level() {
    test $TEST_LEVEL -ge $1
}

# TE_CI_LIST_ALL widens list mode from "what this host would run" to "what this
# TEST_LEVEL could run anywhere". Level gating, the backend matrix and TEST_FILTER
# still apply -- only host capability probes are answered yes without probing.
# The scheduler uses the wider list as the authoritative set of work items that
# exist, so .github/scripts/scheduler/build_weights.py can tell a test that was deleted or
# renamed (drop its weight) from one this host merely skipped (keep it). A probe
# result varies by machine; whether a test file exists does not.
check_list_all() {
    test -n "$TE_CI_LIST_ONLY" -a -n "$TE_CI_LIST_ALL"
}

# Every host capability probe goes through here, selected by label. Returns 0 to
# run the guarded tests, 1 to skip them.
check_supported() {
    check_list_all && return 0
    case "$1" in
        "mxfp8")
            #MXFP8-only test filters collect no tests on unsupported archs
            _probe_result=$(NVTE_ROCM_ENABLE_MXFP8=1 python -c "${PYTHON_TE_IMPORT}; from transformer_engine.pytorch.quantization import is_mxfp8_available; print(is_mxfp8_available())" 2>/dev/null)
            _probe_message="MXFP8 is not supported on this device, skipping MXFP8-only tests"
        ;;
        "flash_attn")
            _probe_result=$(python -c "${PYTHON_TE_IMPORT}; from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils; print(FlashAttentionUtils.is_installed)" 2>/dev/null)
            _probe_message="Flash attention is not installed"
        ;;
        *)
            #A mistyped label must not read as "unsupported": that would skip the
            #guarded tests on every host and look exactly like a machine that
            #cannot run them. Count it so the run's exit code reports it.
            script_error "check_supported: unknown capability $1"
            return 1
        ;;
    esac
    test "$_probe_result" = "True" && return 0
    echo "$_probe_message" >&2
    return 1
}

check_test_jobs_requested() {
    return 1 #Disable parallel jobs because some tests do not support parallel execution
    test -z "$SINGLE_CONFIG" -a -n "$TEST_JOBS" || return 1
    # TEST_JOBS - number of test configurations running in parallel
    # change below condition to -gt 0 to enable single job mode for functionality testing
    test $TEST_JOBS -gt 1 || return 1
    return 0
}

calculate_test_jobs_count() {
    test -n "$1" && test $1 -gt 0 || return 1
    check_test_jobs_requested || return 1

    _device_count=$1
    _devlist=""
    for f in "$HIP_VISIBLE_DEVICES" "$ROCR_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES"; do
        test -z "$f" && continue
        if [ -z "$_devlist" ]; then
            _devlist="$f"
        elif [ "$_devlist" != "$f" ]; then
            echo "Failed to determine visible devices: multiple filering. Disable parallel jobs" >&2
            return 1
        fi
    done
    if [ -n "$_devlist" ]; then
        _f=`echo $_devlist | cut -d, -f$_device_count`
        if [ -z "$_f" ]; then
            echo "Failed to determine visible devices: list lenght mismatch. Disable parallel jobs" >&2
            return 1
        fi
    fi

    test $_device_count -le $TEST_JOBS && TEST_JOBS=$_device_count
    if [ -n "$_devlist" ]; then
        TEST_GPUS=`echo $_devlist | cut -d, -f1-$TEST_JOBS`
    else
        TEST_GPUS=`seq -s, 0 $((TEST_JOBS-1))`
    fi
    test -n "$TEST_GPUS" || return 1
    return 0
}

init_test_jobs() {
    # Call calculate_test_jobs_count and the check_test_jobs_requested because
    # The former can update TEST_JOBS count
    calculate_test_jobs_count $1 && check_test_jobs_requested || return
    : ${WAIT_POLL:=60} #Job count polling interval when cannot use wait
    set -m
    _TEST_JOB_DIR=`mktemp -d`
    test -d "$_TEST_JOB_DIR" || exit 1
    _TEST_CONFIG_LIST=""
    TEST_JOBS_MODE=1
    echo "Init test jobs: TEST_JOBS=$TEST_JOBS GPUs=$TEST_GPUS WAIT_POLL=$WAIT_POLL"
}

wait_for_job_slot() {
    _JOB_IDX=0
    while [ true ]; do
        jobs > /dev/null 2>&1
        for job in `seq 1 $TEST_JOBS`; do
            jobs %$job > /dev/null 2>&1
            if [ $? -eq 2 ]; then
                _JOB_IDX=$job
                return
            fi
        done
        sleep $WAIT_POLL
    done
}

run_test_job() {
    test -n "$TEST_JOBS_MODE" || return 1
    wait_for_job_slot
    _GPU_ID=`echo $TEST_GPUS | cut -d, -f$_JOB_IDX`
    echo "***** Run job on GPU $_GPU_ID for test config $1 at `date` *****"
    (HIP_VISIBLE_DEVICES=$_GPU_ID SINGLE_CONFIG="$1" TEST_LEVEL=$TEST_LEVEL $0; echo RC=$?) > "$_TEST_JOB_DIR/$1.log" 2>&1 &
    _TEST_CONFIG_LIST="$_TEST_CONFIG_LIST $1"
}

finish_test_jobs() {
    test -n "$TEST_JOBS_MODE" || return 1
    TEST_JOBS_MODE=""
    wait > /dev/null; jobs > /dev/null
    echo "All test jobs completed at `date`"
    for _config in $_TEST_CONFIG_LIST; do
        rc=`tail -1 "$_TEST_JOB_DIR/$_config.log"`
        if [ "$rc" != "RC=0" ]; then
            echo "Test config $_config finished with error $rc" >&2
            test_run_error
        fi
        echo "##### $_config log begin #####"
        cat "$_TEST_JOB_DIR/$_config.log"
        echo "##### $_config log end #####"
    done
    rm -rf "$_TEST_JOB_DIR"
}

get_test_config_list() {
    echo $_TEST_CONFIG_LIST
}

get_test_variant_tag() {
    if [ -n "$1" -a -n "$2" ]; then
        echo "$1/$2"
    else
        echo "$1$2"
    fi
}

get_test_name_tag() {
    _fname=${1##*/}
    _test_name=${_fname%%.*}
    _dir=${1%$_fname}
    if [ -n "$2" ]; then
        _tag="$_dir$_test_name.$2"
    else
        _tag="$_dir$_test_name"
    fi
    echo "$(echo $_tag | tr '/' '.')"
}

get_pytest_junitxml() {
    if [ -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" ]; then
        echo "--junitxml=$JUNITXML_PREFIX$1$JUNITXML_SUFFIX"
    fi
}

# Pytest can exit *before* it writes its --junitxml report: a usage error
# (exit 4), a conftest import failure, a hard per-test timeout that os._exit()s
# the process, or a segfault/OOM-kill during collection. The file then silently
# never appears, junit_report.py only sees the other (passing) files' XML, and
# the job summary shows green even though pytest_run already recorded the failure
# via test_run_error. Synthesize a minimal JUnit XML for the missing file so the
# report surfaces it as an error instead of dropping it (keeps the summary
# consistent with the suite exit-code gate).
# args: test_name_tag exit_code
write_missing_junitxml() {
    test -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" || return 0   # XML not requested (local run)
    _missing_xml="$JUNITXML_PREFIX$1$JUNITXML_SUFFIX"
    test -s "$_missing_xml" && return 0   # pytest already wrote a (non-empty) report
    # A non-empty sidecar means te_ci_result_sink captured per-test progress;
    # junit_report.py reconstructs the run from it, so don't shadow that richer
    # record with a coarse whole-file stub.
    test -s "$_missing_xml.partial" && return 0
    mkdir -p "$(dirname "$_missing_xml")" 2>/dev/null
    # Keep the words 'timeout'/'timed out' out of this message: junit_report.py's
    # is_timeout() substring-matches the message and would mislabel the entry.
    _missing_msg="pytest exited $2 without writing JUnit XML (crash, usage/conftest error, or process hard-exit before reporting); see the suite .log artifact"
    cat > "$_missing_xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="0" skipped="0" tests="1">
<testcase classname="$1" name="(no JUnit XML written)"><error message="$_missing_msg"></error></testcase>
</testsuite></testsuites>
EOF
}

get_ctest_junitxml() {
    if [ -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" ]; then
        echo "--output-junit ${JUNITXML_PREFIX}$1${JUNITXML_SUFFIX}"
    fi
}

check_test_filter() {
    test -z "$TEST_FILTER" && return 0
    for _tf in $TEST_FILTER; do
        case "$1" in
        $_tf) return 0
        esac
    done
    return 1
}

# The test name tag names the JUnit XML file and is the TEST_FILTER key a
# scheduler re-dispatches on, so two call lines may not share one: the second
# would overwrite the first's results, and one dispatch would run both. Anything
# that changes what a line runs -- an env prefix, a -k expression, extra pytest
# args -- needs its own label to keep the tags apart.
#
# Fatal rather than counted-and-continued. Skipping the offending call line
# leaves a run that looks complete but is quietly short by an item, and in list
# mode that short list is what the scheduler queues -- so the tests the
# duplicate displaced never run at all, on this run or any after it. The only
# signal would be a non-zero exit code at the very end, long after the message
# scrolled past.
check_test_tag_unique() {
    case " $_seen_test_tags " in
    *" $1 "*)
        script_error "Duplicate test tag $1: give the call site a distinct label"
        exit 1
    esac
    _seen_test_tags="$_seen_test_tags $1"
}

start_message() {
    echo "Started with TEST_LEVEL=$TEST_LEVEL sGPU='$TEST_SGPU' mGPU='$TEST_MGPU' at `date`"
    _rocm_path=$(resolve_rocm_path)
    _rocm_path=`$REALPATH "$_rocm_path" 2>/dev/null || echo "$_rocm_path"`
    echo "ROCM PATH: $_rocm_path"
    python3 --version
}

get_cpu_count() {
    n_vcpus=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    cpus_per_core=$(lscpu | grep "Thread(s) per core:" | awk '{print $NF}')

    echo $((n_vcpus / cpus_per_core))
}

configure_omp_threads() {
    n_physical_cores=`get_cpu_count`
    n_parallel_jobs=$1

    if [ -z ${OMP_NUM_THREADS} ]; then
        export OMP_NUM_THREADS=$((n_physical_cores / n_parallel_jobs))
	echo "Setting OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    else
        echo "Using OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    fi
}

time_elapsed() {
    _arg=$1
    date -d @$((`date +%s` - _arg)) +%${2:-T}
}

pytest_run() {
    #args: tag1 tag2 level ...
    check_level $3 || return
    _test_variant_tag=`get_test_variant_tag $1 $2`
    shift 3
    _test_name_tag=`get_test_name_tag $1 $_test_variant_tag`
    check_test_filter $_test_name_tag || return
    check_test_tag_unique $_test_name_tag || return
    # List mode: emit the work item instead of running it, so an external
    # scheduler can pack items across GPUs. The tag alone is enough to
    # re-dispatch the item: setting TEST_FILTER to it and re-entering this
    # script replays the very same call line, so the inline NVTE_* prefixes and
    # -k expressions are reapplied by the script itself and never have to be
    # serialized here. The suite scripts stay the single source of truth for
    # what runs at each TEST_LEVEL.
    if [ -n "$TE_CI_LIST_ONLY" ]; then
        echo "TE_CI_ITEM $_test_name_tag"
        return
    fi
    _start_ts=`date +%s`
    echo "Run [$_test_variant_tag] $@ at `time_elapsed $TEST_START_TS`"
    # A per-test timeout is applied to every item. Callers may still append their
    # own --timeout/--timeout-method (e.g. distributed tests); since argparse
    # takes the last value, a caller-supplied override wins over these defaults.
    #
    # te_ci_result_sink streams per-test progress to a <junitxml>.partial sidecar
    # so a hard --timeout-method=thread exit (or segfault/OOM) that skips pytest's
    # end-of-session JUnit XML write stays reconstructable by junit_report.py
    # instead of vanishing from the summary. Enabled only when JUnit XML is
    # requested (CI); a plain local run is unaffected.
    _junitxml_arg=`get_pytest_junitxml $_test_name_tag`
    _sink_plugin=""
    _result_sink=""
    _pytest_pythonpath="$PYTHONPATH"
    if [ -n "$_junitxml_arg" ]; then
        _result_sink="${JUNITXML_PREFIX}${_test_name_tag}${JUNITXML_SUFFIX}.partial"
        rm -f "$_result_sink" 2>/dev/null
        _sink_plugin="-p te_ci_result_sink"
        _pytest_pythonpath="${TE_PATH}ci${PYTHONPATH:+:$PYTHONPATH}"
    fi
    TE_RESULT_SINK="$_result_sink" PYTHONPATH="$_pytest_pythonpath" \
        python3 -m pytest -v -rfEs \
        --timeout=$PYTEST_TIMEOUT --timeout-method=$PYTEST_TIMEOUT_METHOD \
        $_sink_plugin $_junitxml_arg $TEST_PYTEST_ARGS "$TEST_DIR/$@"
    _pytest_rc=$?
    if [ $_pytest_rc -ne 0 ]; then
        test_run_error "[$_test_variant_tag] $1"
        write_missing_junitxml "$_test_name_tag" "$_pytest_rc"
    fi
    echo "Done [$_test_variant_tag] $1 in `time_elapsed $_start_ts`"
}

PYTHON_TE_IMPORT="import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; import transformer_engine"
ck_jit_prebuild() {
    _prebuild_list="${TE_PATH}ci/ck_jit_prebuild.txt"
    if [ ! -f "$_prebuild_list" ]; then
        script_error "ck_jit_prebuild: blob list not found: $_prebuild_list"
        return 1
    fi
    _gpu_arch=$(rocminfo | grep -E "^ *Name: *gfx" | head -1 | sed "s/.*gfx/gfx/;s/ .*//" 2>/dev/null)
    if [ -n "$_gpu_arch" ]; then
        _arch_arg="--arch $_gpu_arch"
    else
        script_error "ck_jit_prebuild: GPU architecture not detected, omitting --arch"
        _arch_arg=""
    fi
    _te_install_dir=$(python -c "${PYTHON_TE_IMPORT}; import os; print(os.path.dirname(transformer_engine.__file__))" 2>/dev/null)
    if [ -z "$_te_install_dir" ]; then
        script_error "ck_jit_prebuild: failed to determine transformer_engine installation directory"
        return 1
    fi
    _prebuild_py="$_te_install_dir/lib/ck_jit/ck_jit_prebuild.py"
    if [ ! -f "$_prebuild_py" ]; then
        script_error "ck_jit_prebuild: prebuild script not found: $_prebuild_py"
        return 1
    fi
    _cpu_count=$(get_cpu_count)
    if [ -n "$_cpu_count" -a "$_cpu_count" != "0" ]; then
        _jobs_arg="--jobs $((_cpu_count/2))"
    fi
    if [ "$1" = "build" ]; then
        echo "Building CK JIT cache for arch=${_gpu_arch:-<not detected>}..."
        python "$_prebuild_py" build --blob-list "$_prebuild_list" $_arch_arg $_jobs_arg > /dev/null
        _CK_JIT_CACHE_SNAPSHOT=$(python "$_prebuild_py" cache)
        echo "$_CK_JIT_CACHE_SNAPSHOT" | grep Cache
    else
        if [ -z "${_CK_JIT_CACHE_SNAPSHOT+set}" ]; then
            python "$_prebuild_py" cache | grep Cache
        else
            _cache_now=$(python "$_prebuild_py" cache)
            if [ "$_CK_JIT_CACHE_SNAPSHOT" != "$_cache_now" ]; then
                echo "Cache diff (build -> now):"
                _diff_tmp=$(mktemp)
                echo "$_CK_JIT_CACHE_SNAPSHOT" > "$_diff_tmp"
                echo "$_cache_now" | diff -u "$_diff_tmp" -
                rm -f "$_diff_tmp"
            fi
        fi
    fi
}
