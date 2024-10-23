# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

REALPATH=realpath
realpath $DIR >/dev/null 2>/dev/null
test $? -ne 0 && REALPATH=echo

: ${TE_PATH:=`$REALPATH $DIR/..`/}
TEST_DIR=${TE_PATH}tests/

: ${TEST_LEVEL:=99} #Run all tests by default
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
