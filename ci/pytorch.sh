#!/bin/sh
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

#: ${TEST_WORKERS:=4}

install_prerequisites() {
    pip install 'numpy>=1.22.4,<2.0' onnx onnxruntime
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
    #: ${_WORKERS_COUNT:=1}
    #_args=-n$_WORKERS_COUNT --max-worker-restart=$_WORKERS_COUNT
    pytest -v `get_pytest_junitxml $_test_name_tag` "$TEST_DIR/$@" || test_run_error
    echo "Done [$_gemm, $_fus_attn] $1"
}

run_default_fa() {
    #Run tests that do not use fused attention or control backend selection
    #with default backend only
    if [ $_fus_attn = "auto" ]; then
        run $*
    fi
}

run_test_config(){
    echo ==== Run with GEMM backend: $_gemm and Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    #run_default_fa 1 test_cast_transpose_triton.py
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8tensor.py
    run_default_fa 1 test_fused_rope.py
    run_default_fa 1 test_fusible_ops.py
    test $_gemm = "hipblaslt" && run_default_fa 3 test_gemm_autotune.py
    run 1 test_gqa.py
    run 1 test_jit.py
    run_default_fa 1 test_multi_tensor.py
    run 1 test_numerics.py
    run_default_fa 3 test_onnx_export.py # All FA are disabled in ONNX export mode
    run_default_fa 1 test_permutation.py
    run_default_fa 1 test_recipe.py
    run 1 test_sanity.py
    run_default_fa 1 test_torch_save_load.py
    run_default_fa 1 fused_attn/test_fused_attn.py # Backend selection is controlled by the test
}

run_test_config_mgpu(){
    #_WORKERS_COUNT=1
    #test $TEST_WORKERS = 0 && _WORKERS_COUNT=0
    if [ $_fus_attn = "auto" -a $_gemm = "hipblaslt" ]; then
        echo ==== Run mGPU with GEMM backend: $_gemm and Fused attention backend: $_fus_attn ====
        run 3 test_fused_optimizer.py
        run 3 test_fusible_ops_distributed.py
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
start_message
install_prerequisites
pip list | egrep "flash|ml_dtypes|numpy|onnx|torch|transformer_e|typing_ext"
#check_test_jobs_requested && init_test_jobs `python -c "import torch; print(torch.cuda.device_count())"`

for _gemm in hipblaslt rocblas; do
    configure_gemm_env $_gemm || continue
    
    for _fus_attn in auto ck aotriton unfused; do
        configure_fused_attn_env $_fus_attn || continue

        #Auto - default mode with all Fused attentions backends enabled
        #CK/AOTriton - only corresponding Fused attention backend is enabled
        #Unfused - Fused attention is disabled
        #Level 1 - run hipBlasLt in auto and unfused modes, rocBlas in auto mode
        #Level 3 - run hipBlasLt in all but unfused modes, rocBlas in auto and unfused modes
        if [ $TEST_LEVEL -ge 3 ]; then
            test $_gemm = hipblaslt -a $_fus_attn = unfused && continue
            test $_gemm = rocblas -a $_fus_attn != auto -a $_fus_attn != unfused && continue
        else
            test $_gemm = hipblaslt -a $_fus_attn != auto -a $_fus_attn != unfused && continue
            test $_gemm = rocblas -a $_fus_attn != auto && continue
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
