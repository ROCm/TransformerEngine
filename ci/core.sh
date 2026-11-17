#!/bin/sh
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

start_message
if [ -z "$TEST_SGPU" ]; then
    return_run_results
    exit 0
fi

# ci/run_queue.sh runs the two ctest runs below as separate work items. An item's
# name is its TEST_FILTER key and the name of its JUnit XML file.
if [ -n "$TE_CI_LIST_ITEMS" ]; then
    echo "TE_CI_ITEM nongemm"
    echo "TE_CI_ITEM gemm"
    exit 0
fi
# Nothing to prepare before the queue starts: each item builds what it needs below.
test -n "$TE_CI_SETUP_ONLY" && exit 0

n_parallel_jobs=4

configure_omp_threads $n_parallel_jobs

TEST_DIR=${TE_PATH}tests/cpp

cd $TEST_DIR
# Both items share this build tree and the queue may start them together, so the
# build holds a lock: the first item builds and the second finds it up to date. The
# build itself runs without the lock's descriptor, so nothing it leaves running
# can keep holding it. A failed build fails these items and nothing else.
mkdir -p build
exec 9> build/.te_ci_build.lock
flock 9
(exec 9>&-; cmake -GNinja -Bbuild . && cmake --build build) > build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    script_error "Failed to build cpp test suite"
    cat build.log
    exit $rc
fi
exec 9>&-

check_test_filter "nongemm"
if [ $? -eq 0 ]; then
    echo ===== Run non GEMM tests =====
    ctest --test-dir build -j"$n_parallel_jobs" -V --output-on-failure \
        --timeout $CTEST_TIMEOUT $(get_ctest_junitxml nongemm) -E "GEMMTestSuite"
    test $? -eq 0 || test_run_error "non-GEMM"
fi

check_test_filter "gemm"
if [ $? -eq 0 ]; then
    echo  ===== Run GEMM tests =====
    ctest --test-dir build -j"$n_parallel_jobs" -V --output-on-failure \
        --timeout $CTEST_TIMEOUT $(get_ctest_junitxml gemm) -R "GEMMTestSuite"
    test $? -eq 0 || test_run_error "GEMM"
fi

return_run_results
