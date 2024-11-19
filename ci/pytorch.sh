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

run() {
    check_level $1 || return
    shift
    _test_name_tag=`get_test_name_tag $1 $_gemm-$_fus_attn`
    check_test_filter $_test_name_tag || return
    echo "Run [$_gemm, $_fus_attn] $@"
    : ${_WORKERS_COUNT:=1}
    pytest `get_pytest_junitxml $_test_name_tag` \
           -n$_WORKERS_COUNT --max-worker-restart=$_WORKERS_COUNT "$TEST_DIR/$@" || test_run_error
    echo "Done [$_gemm, $_fus_attn] $1"
}

run_test_config(){
    echo ====== Run with GEMM backend: $_gemm and Fused attention backend: $_fus_attn =====
    _WORKERS_COUNT=4
    if [ $_fus_attn = "ck" -o $_fus_attn = "auto" ]; then 
        _is_default_fa="1"
    else
        _is_default_fa=""
    fi
    if [ $_gemm != "rocblas" ]; then
        #test -n "$_is_default_fa" && run 1 test_cast_transpose_triton.py
        run 1 test_cuda_graphs.py
        _graph_filter=""
    else
        _graph_filter="not graph"
    fi
    run 1 test_deferred_init.py
    run 1 test_float8tensor.py
    run 1 test_fused_rope.py
    test $_gemm = "hipblaslt" && run 1 test_fusible_ops.py #TODO: Run on RocBLAS with supported subtests
    test $_gemm = "hipblaslt" -a -n "$_is_default_fa" && run 3 test_gemm_autotune.py
    run 1 test_gqa.py
    run 1 test_jit.py
    run 1 test_multi_tensor.py
    run 1 test_numerics.py -k "$_graph_filter"
    run 3 test_onnx_export.py
    run 1 test_recipe.py
    run 1 test_sanity.py -k "$_graph_filter"
    run 1 test_torch_save_load.py
    test $_fus_attn != "unfused" && run 1 fused_attn/test_fused_attn.py
}

run_test_config_mgpu(){
    echo ====== Run mGPU with GEMM backend: $_gemm and Fused attention backend: $_fus_attn =====
    _WORKERS_COUNT=1
    run 3 test_fused_optimizer.py
    run 3 test_fusible_ops_distributed.py
    if [ $_fus_attn != "unfused" ]; then
        run 3 fused_attn/test_fused_attn_with_cp.py
    fi
}

# Single config mode, run it and return result
if [ -n "$SINGLE_CONFIG" ]; then
    _gemm=`echo $SINGLE_CONFIG | cut -d- -f1`
    _fus_attn=`echo $SINGLE_CONFIG | cut -d- -f2`
    configure_gemm_env $_gemm && configure_fused_attn_env $_fus_attn && run_test_config
    return_run_results
    exit $?
fi

#Master script mode: prepare testing prerequisites first
echo "Started with TEST_LEVEL=$TEST_LEVEL at `date`"
install_prerequisites
check_test_jobs_requested && init_test_jobs `python -c "import torch; print(torch.cuda.device_count())"`

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


        if [ -n "$TEST_JOBS_MODE" ]; then
            run_test_job "$_gemm-$_fus_attn"
        else
            run_test_config
            run_test_config_mgpu
        fi
    done
done

if [ -n "$TEST_JOBS_MODE" ]; then
    finish_test_jobs
    for _cfg in $(get_test_config_list); do
        _gemm=`echo $_cfg | cut -d- -f1`
        _fus_attn=`echo $_cfg | cut -d- -f2`
        configure_gemm_env $_gemm && configure_fused_attn_env $_fus_attn && run_test_config_mgpu;
    done
fi
return_run_results
