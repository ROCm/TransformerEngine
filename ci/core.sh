#!/bin/sh
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/cpp

cd $TEST_DIR
(cmake -GNinja -Bbuild . && cmake --build build) > build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    script_error "Failed to build cpp test suite"
    cat build.log
    exit $rc
fi

echo ===== Run non GEMM tests =====
ctest --test-dir build -j4 -E "OperatorTest/GEMMTestSuite"
test $? -eq 0 || test_run_error

for _gemm in hipblaslt rocblas; do
    configure_gemm_env $_gemm || continue
    _exclude=""
    if [ $_gemm = "hipblaslt" ]; then
        _exclude="-E Test(.*bf16/.*X.X1|.*fp8.*fp16/.*X1X0|.*fp8.*X.X1|.*fp8/|.*bf8/)"
    fi
    echo  ===== Run GEMM $_gemm tests =====
    ctest --test-dir build -j4 -R "OperatorTest/GEMMTestSuite" $_exclude
    test $? -eq 0 || test_run_error
done

return_run_results
