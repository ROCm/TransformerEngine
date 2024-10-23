# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

REALPATH=realpath
realpath $DIR >/dev/null 2>/dev/null
test $? -ne 0 && REALPATH=echo

: ${TE_PATH:=`$REALPATH $DIR/..`/}
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
            unset NVTE_USE_HIPBLASLT NVTE_USE_ROCBLAS
        ;;
        "hipblaslt")
            export NVTE_USE_HIPBLASLT=1
            unset NVTE_USE_ROCBLAS
        ;;
        "rocblas")
            export NVTE_USE_ROCBLAS=1
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

init_test_jobs() {
    test -z "$SINGLE_CONFIG" || return
    : ${TEST_JOBS:=0} #Number of test configurations running in parallel
    test $TEST_JOBS -gt 0 || return
    _JOB_CNT=$((TEST_JOBS-1))
    : ${WAIT_POLL:=60} #Job count polling interval when cannot use wait
    set -m
    _TEST_JOB_DIR=`mktemp -d`
    test -d "$_TEST_JOB_DIR" || exit 1
    _TEST_CONFIG_LIST=""
    TEST_JOBS_MODE=1
    echo "Init test jobs: TEST_JOBS=$TEST_JOBS WAIT_POLL=$WAIT_POLL at `date`"
}

wait_for_jobs_count() {
    jobs > "$_TEST_JOB_DIR/jobs.lst"
    _cnt=`grep Running "$_TEST_JOB_DIR/jobs.lst" | wc -l`
    while [ $_cnt -gt $1 ]; do
        sleep "$WAIT_POLL"
        jobs > "$_TEST_JOB_DIR/jobs.lst"
        _cnt=`grep Running "$_TEST_JOB_DIR/jobs.lst" | wc -l`
    done
}

run_test_job() {
    test -n "$TEST_JOBS_MODE" || return 1
    wait_for_jobs_count $_JOB_CNT
    echo "***** Run job for test config $1 at `date` *****"
    (SINGLE_CONFIG="$1" TEST_LEVEL=$TEST_LEVEL $0; echo RC=$?) > "$_TEST_JOB_DIR/$1.log" 2>&1 &
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