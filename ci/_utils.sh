# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

REALPATH=realpath
realpath $DIR >/dev/null 2>/dev/null
test $? -ne 0 && REALPATH=echo

: ${TE_PATH:=`$REALPATH $DIR/..`/}
export TE_PATH
TEST_DIR=${TE_PATH}tests/

: ${TEST_LEVEL:=99} #Run all tests by default
TEST_JOBS_MODE=""

_script_error_count=0
_run_error_count=0

script_error() {
    _script_error_count=$((_script_error_count+1))
    test "$@" && echo $@ >&2
}

test_run_error() {
    _run_error_count=$((_run_error_count+1))
}

return_run_results() {
    test $_script_error_count -ne 0 && echo Detected $_script_error_count script errors during tests run at level $TEST_LEVEL >&2
    test $_run_error_count -ne 0 && echo Got $_run_error_count test errors during run at level $TEST_LEVEL >&2
    test $_run_error_count -eq 0 -a $_script_error_count -eq 0
}

configure_gemm_env() {
    case "$1" in
        "auto")
            unset NVTE_USE_HIPBLASLT NVTE_USE_ROCBLAS ROCBLAS_STREAM_ORDER_ALLOC
        ;;
        "hipblaslt")
            export NVTE_USE_HIPBLASLT=1
            unset NVTE_USE_ROCBLAS ROCBLAS_STREAM_ORDER_ALLOC
        ;;
        "rocblas")
            export NVTE_USE_ROCBLAS=1 ROCBLAS_STREAM_ORDER_ALLOC=1
            unset NVTE_USE_HIPBLASLT
        ;;
        *)
            script_error "Error unknown GEMM config $1"
            return 1
        ;;
    esac
    return 0
}

configure_fused_attn_env() {
    case "$1" in
        "auto")
            unset NVTE_FUSED_ATTN NVTE_FUSED_ATTN_AOTRITON NVTE_FUSED_ATTN_CK
        ;;
        "aotriton")
            export NVTE_FUSED_ATTN_CK=0
            unset NVTE_FUSED_ATTN NVTE_FUSED_ATTN_AOTRITON
        ;;
        "ck")
            export NVTE_FUSED_ATTN_AOTRITON=0
            unset NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK
        ;;
        "unfused")
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

get_test_name_tag() {
    _fname=${1##*/}
    _test_name=${_fname%%.*}
    test -n "$2" && _test_suffix=.$2
    echo "$_test_name$_test_suffix"
}

get_pytest_junitxml() {
    if [ -n "$JUNITXML_PREFIX$JUNITXML_SUFFIX" ]; then
        echo "--junitxml=$JUNITXML_PREFIX$1$JUNITXML_SUFFIX"
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

start_message() {
    echo "Started with TEST_LEVEL=$TEST_LEVEL at `date`"
    echo "ROCm: `ls -d /opt/rocm-*`"
    python --version
}
