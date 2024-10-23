#!/bin/sh
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

install_prerequisites() {
    pip install numpy==1.22.4 onnx onnxruntime
    rc=$?
    if [ $rc -ne 0 ]; then
        script_error "Failed to install test prerequisites"
        exit $rc
    fi
}

run_1() {
    check_level 1 || return
    echo "Run [$_gemm, $_fus_attn] $@"
    pytest "$TEST_DIR/$@" || test_run_error
}

run_3() {
    check_level 3 || return
    echo "Run [$_gemm, $_fus_attn] $@"
    pytest "$TEST_DIR/$@" || test_run_error
}

echo "Started with TEST_LEVEL=$TEST_LEVEL at `date`"
install_prerequisites

for _gemm in hipblaslt rocblas; do
    configure_gemm_env $_gemm || continue
    
    for _fus_attn in auto ck aotriton unfused; do
        configure_fused_attn_env $_fus_attn || continue

        #On basic (1) test level rocBLAS tests are run with default FUSED_ATTN flags only
        #On normal (3) level it runs with all but default backend configuration
        #hipBlasLt tests are run with ck/aotriton/unfused on test level 1
        #and with auto/aotriton/unfused on test level 3
        if [ $TEST_LEVEL -ge 3 ]; then
            test $_gemm = "rocblas" -a $_fus_attn = "auto" && continue
            test $_gemm = "hipblaslt" -a $_fus_attn = "ck" && continue
        else
            test $_gemm = "rocblas" -a $_fus_attn != "auto" && continue
            test $_gemm = "hipblaslt" -a $_fus_attn = "auto" && continue
        fi

        echo ====== Run with GEMM backend: $_gemm and Fused attention backend: $_fus_attn =====

        if [ $_gemm != "rocblas" ]; then
            run_1 test_cuda_graphs.py
            _graph_filter=""
        else
            _graph_filter="not graph"
        fi
        run_1 test_deferred_init.py
        run_1 test_float8tensor.py
        run_1 test_fused_optimizer.py
        run_1 test_fused_rope.py
        run_1 test_gqa.py
        run_1 test_jit.py
        run_1 test_multi_tensor.py
        run_1 test_numerics.py -k "$_graph_filter"
        run_3 test_onnx_export.py
        run_1 test_recipe.py
        run_1 test_sanity.py -k "$_graph_filter"
        run_1 test_torch_save_load.py
        if [ $_fus_attn != "unfused" ]; then
            run_1 fused_attn/test_fused_attn.py
            run_1 fused_attn/test_fused_attn_with_cp.py
        fi
    done
done

return_run_results
