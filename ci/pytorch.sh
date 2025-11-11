#!/bin/sh
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

#: ${TEST_WORKERS:=4}

install_prerequisites() {
    pip install 'numpy>=1.22.4' pandas
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
    _test_variant_tag=`get_test_variant_tag $_fus_attn $_test_label`
    _test_name_tag=`get_test_name_tag $1 $_test_variant_tag`
    check_test_filter $_test_name_tag || return
    echo "Run [$_test_variant_tag] $@"
    #: ${_WORKERS_COUNT:=1}
    #_args=-n$_WORKERS_COUNT --max-worker-restart=$_WORKERS_COUNT
    pytest -v -rfEs `get_pytest_junitxml $_test_name_tag` "$TEST_DIR/$@" || test_run_error "[$_test_variant_tag] $1"
    echo "Done [$_test_variant_tag] $1"
}

run_default_fa() {
    #Run tests that do not use fused attention or control backend selection
    #with default backend only
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        run "$@"
    fi
}

run_default_fa_lbl() {
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        _test_label="$1"
        shift
        run "$@"
        _test_label=""
    fi
}

run_test_config(){
    echo ==== Run with Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8tensor.py
    run_default_fa 1 test_float8_current_scaling_exact.py
    test $_fus_attn = auto -o $_fus_attn = ck -o $_fus_attn = aotriton && NVTE_FLASH_ATTN=0 run 1 test_cpu_offloading.py
    run_default_fa 1 test_fused_rope.py
    run_default_fa 1 test_fusible_ops.py
    run_default_fa 3 test_gemm_autotune.py
    run_default_fa 3 test_gemm_sm_count.py
    run 1 test_gqa.py
    run 1 test_jit.py
    run_default_fa 1 test_multi_tensor.py
    run 1 test_numerics.py
    run_default_fa 1 test_permutation.py
    run_default_fa 1 test_recipe.py
    run 1 test_sanity.py
    run_default_fa 1 test_sanity_import.py
    run_default_fa 1 fused_attn/test_fused_attn.py # Backend selection is controlled by the test
    NVTE_CK_USES_FWD_V3=1 NVTE_CK_USES_BWD_V3=1 run_default_fa_lbl "v3" 1 fused_attn/test_fused_attn.py # Using FAv3 for forward and backward pass
    run_default_fa 1 triton_kernels/test_cast.py
    run_default_fa 1 triton_kernels/test_cast_mxfp8.py
    run_default_fa 1 triton_kernels/test_norm_common.py
    run_default_fa 1 triton_kernels/test_norms.py
    NVTE_TEST_TRITON_AUTOTUNE=1 run_default_fa_lbl "autotune" 3 triton_kernels/test_norms.py
    run_default_fa 1 test_parallel_cross_entropy.py
    NVTE_USE_DEQUANTIZE_TRITON=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 NVTE_USE_LAYERNORM_TRITON=1 run_default_fa_lbl "triton" 1 test_numerics.py
    NVTE_USE_RMSNORM_TRITON=1 run_default_fa_lbl "triton" 1 test_fusible_ops.py
}

run_test_config_mgpu(){
    #_WORKERS_COUNT=1
    #test $TEST_WORKERS = 0 && _WORKERS_COUNT=0
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
        run 3 test_fused_optimizer.py
        run 3 test_sanity_import.py
        run 3 distributed/test_fusible_ops.py
        run 3 distributed/test_numerics.py
        run 3 distributed/test_torch_fsdp2.py
        run 3 distributed/test_torch_fsdp2_fp8.py
        run 3 fused_attn/test_fused_attn_with_cp.py
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
pip list | egrep "flash|ml_dtypes|numpy|torch|transformer_e|typing_ext"
#check_test_jobs_requested && init_test_jobs `python -c "import torch; print(torch.cuda.device_count())"`

for _fus_attn in auto flash ck aotriton unfused; do
    configure_fused_attn_env $_fus_attn || continue

    #Auto - default mode with all Flash and Fused attention backends enabled
    #Flash - Fused attention is disabled
    #CK/AOTriton - no Flash attention and only corresponding Fused attention backend is enabled
    #Unfused - Flash and Fused attentions are disabled
    #Level 1 - run in auto and unfused modes
    #Level 3 - run in all but auto and unfused modes
    if [ $TEST_LEVEL -ge 3 ]; then
        test $_fus_attn = auto -o $_fus_attn = unfused && continue
        _DEFAULT_FUSED_ATTN="ck"
    else
        test $_fus_attn != auto -a $_fus_attn != unfused && continue
        _DEFAULT_FUSED_ATTN="auto"
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
if [ $TEST_LEVEL -ge 3 ]; then
    if [ -n "$TEST_SGPU" ]; then
        run_benchmark   
    fi
fi

return_run_results
