#!/bin/sh
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

#: ${TEST_WORKERS:=4}

install_prerequisites() {
    pip install 'numpy>=1.22.4,<2.0' onnx onnxruntime pandas
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
    _test_name_tag=`get_test_name_tag $1 $_fus_attn`
    check_test_filter $_test_name_tag || return
    echo "Run [$_fus_attn] $@"
    #: ${_WORKERS_COUNT:=1}
    #_args=-n$_WORKERS_COUNT --max-worker-restart=$_WORKERS_COUNT
    pytest -v `get_pytest_junitxml $_test_name_tag` "$TEST_DIR/$@" || test_run_error "[$_fus_attn] $1"
    echo "Done [$_fus_attn] $1"
}

run_default_fa() {
    #Run tests that do not use fused attention or control backend selection
    #with default backend only
    if [ $_fus_attn = "auto" ]; then
        run "$@"
    fi
}

run_test_config(){
    echo ==== Run with Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8tensor.py
    run_default_fa 1 test_fused_rope.py
    run_default_fa 1 test_fusible_ops.py
    run_default_fa 3 test_gemm_autotune.py
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
    # TODO: bring back cast transpose kernels after triton kernels for transformer_engine::pytorch::quantize
    #run_default_fa 1 triton_kernels/test_cast_transpose.py
    run_default_fa 1 triton_kernels/test_rmsnorm.py
    run_default_fa 1 triton_kernels/test_layernorm.py
    run_default_fa 1 triton_kernels/test_norm_common.py
    #NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 NVTE_USE_LAYERNORM_TRITON=1 run_default_fa 3 test_numerics.py
    NVTE_USE_RMSNORM_TRITON=1 NVTE_USE_LAYERNORM_TRITON=1 run_default_fa 3 test_numerics.py
}

run_test_config_mgpu(){
    #_WORKERS_COUNT=1
    #test $TEST_WORKERS = 0 && _WORKERS_COUNT=0
    if [ $_fus_attn = "auto" ]; then
        echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
        run 3 test_fused_optimizer.py
        run 3 distributed/test_fusible_ops.py
        run 3 fused_attn/test_fused_attn_with_cp.py
        run 3 distributed/test_numerics.py
    fi
}

run_benchmark() {
    check_test_filter benchmark || return
    echo "\n============= Running benchmarks attention script ============="
    BENCH_SCRIPT="$DIR/../benchmarks/attention/benchmark_attention_rocm.py"
    
    if command -v realpath >/dev/null 2>&1; then
        BENCH_SCRIPT=$(realpath "$DIR/../benchmarks/attention/benchmark_attention_rocm.py")
    fi

    if [ ! -f "$BENCH_SCRIPT" ]; then
        echo "Benchmark script not found: $BENCH_SCRIPT"
        return
    fi

    python "$BENCH_SCRIPT" --use_ck_bwd_v3 --run_sanity_checks || test_run_error $BENCH_SCRIPT
}

# Single config mode, run it and return result
if [ -n "$SINGLE_CONFIG" ]; then
    _fus_attn="$SINGLE_CONFIG"
    configure_fused_attn_env $_fus_attn && run_test_config
    return_run_results
    exit $?
fi

#Master script mode: prepare testing prerequisites first
start_message
install_prerequisites
pip list | egrep "flash|ml_dtypes|numpy|onnx|torch|transformer_e|typing_ext"
#check_test_jobs_requested && init_test_jobs `python -c "import torch; print(torch.cuda.device_count())"`
    
for _fus_attn in auto flash ck aotriton unfused; do
    configure_fused_attn_env $_fus_attn || continue

    #Auto - default mode with all Flash and Fused attention backends enabled
    #Flash - Fused attention is disabled
    #CK/AOTriton - no Flash attention and only corresponding Fused attention backend is enabled
    #Unfused - Flash and Fused attentions are disabled
    #Level 1 - run in auto and unfused modes
    #Level 3 - run in all but unfused modes
    if [ $TEST_LEVEL -ge 3 ]; then
        test $_fus_attn = unfused && continue
    else
        test $_fus_attn != auto -a $_fus_attn != unfused && continue
    fi

    if [ -n "$TEST_JOBS_MODE" ]; then
        test -n "$TEST_SGPU" && run_test_job "$_fus_attn"
    else
        test -n "$TEST_SGPU" && run_test_config
        test -n "$TEST_MGPU" && run_test_config_mgpu
    fi
done

if [ -n "$TEST_JOBS_MODE" -a -n "$TEST_MGPU" ]; then
    finish_test_jobs
    for _fus_attn in $(get_test_config_list); do
        configure_fused_attn_env $_fus_attn && run_test_config_mgpu;
    done
fi

#run benchmark script
if [ $TEST_LEVEL -ge 3 ] && [ -n "$TEST_SGPU" ]; then
    run_benchmark
fi

return_run_results
