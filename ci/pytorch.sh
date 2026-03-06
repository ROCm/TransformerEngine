#!/bin/sh
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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
}

run() {
    pytest_run $_fus_attn "" "$@"
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
        pytest_run $_fus_attn "$@"
    fi
}

run_test_config(){
    echo ==== Run with Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        mkdir -p ${TEST_DIR}/checkpoint
        python ${TEST_DIR}/test_checkpoint.py --save-checkpoint all --checkpoint-dir ${TEST_DIR}/checkpoint
        NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=${TEST_DIR}/checkpoint run 1 test_checkpoint.py
        rm -rf ${TEST_DIR}/checkpoint
    fi
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8tensor.py
    run_default_fa 1 test_float8_current_scaling_exact.py
    run 1 test_cpu_offloading.py
    test $_fus_attn = auto -o $_fus_attn = ck -o $_fus_attn = aotriton && NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 run 3 test_cpu_offloading_v1.py
    run_default_fa 1 test_fused_rope.py
    run_default_fa 1 test_fused_router.py
    run_default_fa 1 test_fusible_ops.py
    run_default_fa 1 test_gemm_autotune.py
    run 1 test_gqa.py
    run 1 test_jit.py
    run_default_fa 1 test_multi_tensor.py
    run 1 test_numerics.py
    run_default_fa 1 test_permutation.py
    run_default_fa 1 test_recipe.py
    run 1 test_sanity.py
    run_default_fa 1 test_sanity_import.py
    run_default_fa 1 attention/test_attention.py # Backend selection is controlled by the test
    run_default_fa 1 attention/test_cp_utils.py
    run_default_fa 1 attention/test_kv_cache.py
    run_default_fa 1 triton_kernels/test_cast.py
    run_default_fa 1 triton_kernels/test_cast_mxfp8.py
    run_default_fa 1 triton_kernels/test_grouped_gemm.py
    run_default_fa 1 triton_kernels/test_utils.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 triton_kernels/test_norms.py
    NVTE_ROCM_ENABLE_MXFP8=1 NVTE_TEST_TRITON_AUTOTUNE=1 run_default_fa_lbl "autotune" 3 triton_kernels/test_norms.py
    run_default_fa 1 test_parallel_cross_entropy.py
    NVTE_USE_DEQUANTIZE_TRITON=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 NVTE_USE_LAYERNORM_TRITON=1 run_default_fa_lbl "triton" 3 test_numerics.py
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 run_default_fa_lbl "triton" 1 test_fusible_ops.py
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "triton" 1 test_float8_current_scaling_exact.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 test_numerics.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 test_fusible_ops.py
    NVTE_USE_ATOMIC_AMAX=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "amax+triton" 3 test_numerics.py
    NVTE_USE_ATOMIC_AMAX=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "amax+triton" 3 test_fusible_ops.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 triton_kernels/test_cast.py
}

run_test_config_mgpu(){
    echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
    configure_omp_threads 8
    run_default_fa 1 test_fused_optimizer.py
    #this test is not really mGPU but time sensitive so run it here because sGPU tests
    #run in parallel on CI and it affects timing
    run_default_fa 1 test_gemm_sm_count.py
    run_default_fa 3 test_sanity_import.py
    run_default_fa 2 distributed/test_fusible_ops.py
    run_default_fa 2 distributed/test_numerics.py
    run_default_fa 1 distributed/test_torch_fsdp2.py
    run_default_fa 2 distributed/test_torch_fsdp2_fp8.py
    run_default_fa_lbl "flash" 3 attention/test_attention_with_cp.py -k "with_flash"
    run_default_fa_lbl "fused" 2 attention/test_attention_with_cp.py -k "with_fused"
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

    python "$BENCH_SCRIPT" --run_sanity_checks || test_run_error $BENCH_SCRIPT
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
    #Level 1 - run in auto mode only
    #Level 2 - run in ck and aotriton modes
    #Level 3 - run in all but auto modes
    if [ $TEST_LEVEL -ge 3 ]; then
        test $_fus_attn = auto && continue
        _DEFAULT_FUSED_ATTN="ck"
    elif [ $TEST_LEVEL -eq 2 ]; then
        test $_fus_attn != aotriton -a $_fus_attn != ck && continue
        _DEFAULT_FUSED_ATTN="ck"
    else
        test $_fus_attn != auto && continue
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
