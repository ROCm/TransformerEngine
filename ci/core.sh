#!/bin/sh
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

start_message
if [ -z "$TEST_SGPU" ]; then
    return_run_results
    exit 0
fi

n_parallel_jobs=4

configure_omp_threads $n_parallel_jobs

TEST_DIR=${TE_PATH}tests/cpp

cd $TEST_DIR
(cmake -GNinja -Bbuild . && cmake --build build) > build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    script_error "Failed to build cpp test suite"
    cat build.log
    exit $rc
fi

check_test_filter "nongemm"
if [ $? -eq 0 ]; then
    echo ===== Run non GEMM tests =====
    ctest --test-dir build -j"$n_parallel_jobs" -V --output-on-failure -E "GEMMTestSuite"
    test $? -eq 0 || test_run_error "non-GEMM"
fi

check_test_filter "gemm"
if [ $? -eq 0 ]; then
    echo  ===== Run GEMM tests =====
    ctest --test-dir build -j"$n_parallel_jobs" -V --output-on-failure -R "GEMMTestSuite"
    test $? -eq 0 || test_run_error "GEMM"
fi

return_run_results
