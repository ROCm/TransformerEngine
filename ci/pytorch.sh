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
    NVTE_USE_ROCM=1 bash $TEST_DIR/custom_ort_ops/build.sh
}

run() {
    check_level $1 || return
    shift
    _test_name_tag=`get_test_name_tag $1 $_gemm-$_fus_attn`
    check_test_filter $_test_name_tag || return
    echo "Run [$_gemm, $_fus_attn] $@"
    #: ${_WORKERS_COUNT:=1}
    #_args=-n$_WORKERS_COUNT --max-worker-restart=$_WORKERS_COUNT
    pytest -v `get_pytest_junitxml $_test_name_tag` "$TEST_DIR/$@" || test_run_error "[$_gemm, $_fus_attn] $1"
    echo "Done [$_gemm, $_fus_attn] $1"
}

run_default_fa() {
    #Run tests that do not use fused attention or control backend selection
    #with default backend only
    if [ $_fus_attn = "auto" ]; then
        run "$@"
    fi
}

run_test_config(){
    echo ==== Run with GEMM backend: $_gemm and Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8tensor.py
    run_default_fa 1 test_fused_rope.py
    # test_fusible_ops now contains fp8+grad and other gemm configs not supported by rocblas gemm path
    test $_gemm = "hipblaslt" && run_default_fa 1 test_fusible_ops.py
    test $_gemm = "hipblaslt" && run_default_fa 3 test_gemm_autotune.py
    run 1 test_gqa.py
    run 1 test_jit.py
    run_default_fa 1 test_multi_tensor.py
    # test_numerics now contains fp8+grad and other gemm configs not supported by rocblas gemm path
    test $_gemm = "hipblaslt" && run 1 test_numerics.py
    # TODO: release test_permutation_mask_map_fp8 until upstream fixes the to_float8 error
    run_default_fa 1 test_permutation.py -k "not test_permutation_mask_map_fp8 and not test_permutation_single_case"
    # test_recipe now contains fp8+grad and other gemm configs not supported by rocblas gemm path
    test $_gemm = "hipblaslt" && run_default_fa 1 test_recipe.py
    # test_sanity now contains fp8+grad and other gemm configs not supported by rocblas gemm path
    test $_gemm = "hipblaslt" && run 1 test_sanity.py
    run_default_fa 1 fused_attn/test_fused_attn.py # Backend selection is controlled by the test
    if [ $_gemm = "hipblaslt" ]; then
        #TODO: bring back cast transpose kernels after triton kernels for transformer_engine::pytorch::quantize
        #TODO: release rmsnorm_fwd_bwd_triton tests after the memory issue in hsa-runtime fixed
        run_default_fa 1 triton_kernels/test_rmsnorm_triton.py
        NVTE_USE_RMSNORM_TRITON=1 run_default_fa 3 test_numerics.py
    fi
}

run_test_config_mgpu(){
    #_WORKERS_COUNT=1
    #test $TEST_WORKERS = 0 && _WORKERS_COUNT=0
    if [ $_fus_attn = "auto" -a $_gemm = "hipblaslt" ]; then
        echo ==== Run mGPU with GEMM backend: $_gemm and Fused attention backend: $_fus_attn ====
        run 3 test_fused_optimizer.py
        run 3 distributed/test_fusible_ops.py
        run 3 fused_attn/test_fused_attn_with_cp.py
        run 3 distributed/test_numerics.py
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
    
    for _fus_attn in auto flash ck aotriton unfused; do
        configure_fused_attn_env $_fus_attn || continue

        #Auto - default mode with all Flash and Fused attention backends enabled
        #Flash - Fused attention is disabled
        #CK/AOTriton - no Flash attention and only corresponding Fused attention backend is enabled
        #Unfused - Flash and Fused attentions are disabled
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
            test -n "$TEST_SGPU" && run_test_job "$_gemm-$_fus_attn"
        else
            test -n "$TEST_SGPU" && run_test_config
            test -n "$TEST_MGPU" && run_test_config_mgpu
        fi
    done
done

if [ -n "$TEST_JOBS_MODE" -a -n "$TEST_MGPU" ]; then
    finish_test_jobs
    for _cfg in $(get_test_config_list); do
        _gemm=`echo $_cfg | cut -d- -f1`
        _fus_attn=`echo $_cfg | cut -d- -f2`
        configure_gemm_env $_gemm && configure_fused_attn_env $_fus_attn && run_test_config_mgpu;
    done
fi
return_run_results
