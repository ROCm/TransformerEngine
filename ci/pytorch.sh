#!/bin/sh
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/pytorch

#: ${TEST_WORKERS:=4}

install_prerequisites() {
    pip install 'numpy>=1.22.4' pandas safetensors pyyaml pytest-timeout
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

check_mxfp8_supported() {
    #Guard MXFP8-only test filters, which collect no tests on unsupported archs
    _result=$(NVTE_ROCM_ENABLE_MXFP8=1 python -c "${PYTHON_TE_IMPORT}; from transformer_engine.pytorch.quantization import is_mxfp8_available; print(is_mxfp8_available())" 2>/dev/null)
    if [ "$_result" = "True" ]; then
        return 0
    else
        echo "MXFP8 is not supported on this device, skipping MXFP8-only tests" >&2
        return 1
    fi
}

run_test_config(){
    echo ==== Run with Fused attention backend: $_fus_attn ====
    #_WORKERS_COUNT=$TEST_WORKERS
    # Enable GroupedLinear single-param feature
    export NVTE_GROUPED_LINEAR_SINGLE_PARAM=1
    run_default_fa 1 test_backward_override.py
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        mkdir -p ${TEST_DIR}/checkpoint
        python ${TEST_DIR}/test_checkpoint.py --save-checkpoint all --checkpoint-dir ${TEST_DIR}/checkpoint
        NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=${TEST_DIR}/checkpoint run 1 test_checkpoint.py
        rm -rf ${TEST_DIR}/checkpoint
    fi
    run 1 test_cuda_graphs.py
    run_default_fa 1 test_deferred_init.py
    run_default_fa 1 test_float8_current_scaling_exact.py
    run_default_fa 1 test_float8blockwisetensor.py
    run_default_fa 1 test_float8_blockwise_scaling_exact.py
    run_default_fa 1 test_float8_blockwise_gemm_exact.py
    run_default_fa 1 test_quantized_tensor.py
    test $_fus_attn = auto -o $_fus_attn = ck && run 1 test_cpu_offloading.py
    test $_fus_attn = auto -o $_fus_attn = ck -o $_fus_attn = aotriton && NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 run 3 test_cpu_offloading_v1.py
    run_default_fa 1 test_fused_rope.py
    run_default_fa 1 test_fused_router.py
    run_default_fa 1 test_fusible_ops.py
    run_default_fa 1 test_gemm_autotune.py
    NVTE_USE_GEMM_TRITON=1 run_default_fa_lbl "triton" 1 triton_kernels/test_gemm.py
    NVTE_USE_GEMM_TRITON=1 run_default_fa_lbl "triton" 1 triton_kernels/test_gemm_kernel.py
    run 1 test_gqa.py
    run 1 test_grouped_linear.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 test_grouped_tensor.py
    run 1 test_jit.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 test_multi_tensor.py
    run 1 test_numerics.py
    check_mxfp8_supported && NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa_lbl "mxfp8" 1 test_numerics.py -k "MXFP8BlockScaling and 126m and not grouped"
    run_default_fa 1 test_nvfp4_fsdp2_hooks.py
    run_default_fa 1 test_permutation.py
    run_default_fa 1 test_recipe.py
    run 1 test_sanity.py
    run_default_fa 3 test_sanity_hipified_cast_transpose.py
    run_default_fa 1 test_sanity_import.py
    run_default_fa 1 test_torch_compile.py
    run_default_fa 1 attention/test_attention.py # Backend selection is controlled by the test
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 run_default_fa_lbl "deterministic" 3 attention/test_attention.py -k "test_deterministic_bwd_ck"
    run_default_fa 1 attention/test_cp_utils.py
    run_default_fa 1 attention/test_kv_cache.py
    run_default_fa 1 attention/test_cu_seqlens_cache.py
    run_default_fa 1 triton_kernels/test_cast.py
    run_default_fa 1 triton_kernels/test_cast_mxfp8.py
    run_default_fa 1 triton_kernels/test_cast_mxfp4.py
    run_default_fa 1 triton_kernels/test_grouped_gemm.py
    run_default_fa 1 triton_kernels/test_utils.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 triton_kernels/test_norms.py
    NVTE_ROCM_ENABLE_MXFP8=1 NVTE_TEST_TRITON_AUTOTUNE=1 run_default_fa_lbl "autotune" 3 triton_kernels/test_norms.py
    run_default_fa 1 test_parallel_cross_entropy.py
    NVTE_USE_DEQUANTIZE_TRITON=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 NVTE_USE_LAYERNORM_TRITON=1 run_default_fa_lbl "triton" 3 test_numerics.py
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 NVTE_USE_RMSNORM_TRITON=1 run_default_fa_lbl "triton" 1 test_fusible_ops.py
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "triton" 1 test_float8_current_scaling_exact.py
    # NVTE_ROCM_ENABLE_MXFP8=1 opens up MXFP8 recipe parametrizations
    # (is_mxfp8_available() on ROCm gates on this env var). Restricted to
    # test_numerics.py for now -- test_fusible_ops.py MXFP8 exercises fused-op
    # paths that hit dev-side C++ bugs (gated_mxfp8 swizzle assert, grouped
    # GEMM bias assert) that fail identically under hipBLASLt / HipKittens /
    # Triton, so leave it alone until dev fixes those.
    NVTE_ROCM_ENABLE_MXFP8=1 NVTE_USE_GEMM_TRITON=1 run_default_fa_lbl "gemm-triton" 3 test_numerics.py
    NVTE_USE_GEMM_TRITON=1 run_default_fa_lbl "gemm-triton" 1 test_fusible_ops.py
    NVTE_USE_GEMM_TRITON=1 run_default_fa_lbl "gemm-triton" 1 test_float8_current_scaling_exact.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 test_numerics.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 test_fusible_ops.py
    NVTE_USE_ATOMIC_AMAX=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "amax+triton" 3 test_numerics.py
    NVTE_USE_ATOMIC_AMAX=1 NVTE_USE_CAST_TRANSPOSE_TRITON=1 run_default_fa_lbl "amax+triton" 3 test_fusible_ops.py
    NVTE_USE_ATOMIC_AMAX=1 run_default_fa_lbl "amax" 3 triton_kernels/test_cast.py
    run_default_fa 1 nvfp4/
    run_default_fa 1 mxfp4/
    run_default_fa 1 test_qk_norm.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 test_partial_cast.py
    NVTE_DISABLE_TRITON_AUTOTUNING=1 run_default_fa 1 test_mhc.py
    run_default_fa 1 layernorm_mlp/test_selective_activation_checkpoint.py
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 test_custom_recipe.py
    #optimize_for_gemm cases self-skip on ROCm: MXFP8 scale swizzle fusion is unimplemented
    NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 mxfp8/
    #Upstream PR3122 moved the grouped MLP cases out of test_fusible_ops.py into this file.
    #mxfp8-True variants are deselected: hipBLASLt MXFP8 GEMM does not support bias on ROCm.
    #Scoped to TestGroupedMLPFusedOp; the TestGroupedLinearOp sweep in the same file is ~4.7k cases.
    check_mxfp8_supported && NVTE_ROCM_ENABLE_MXFP8=1 run_default_fa 1 test_grouped_mlp.py -k "TestGroupedMLPFusedOp and not mxfp8-True"
    #NVIDIA-DL-Framework-Inspect suite. One file per invocation: TEDebugState is process global,
    #and debug/test_numerics.py must stay separate as its basename collides with test_numerics.py
    _dbg_args="--feature_dirs=${TE_PATH}transformer_engine/debug/features --configs_dir=${TE_PATH}tests/pytorch/debug/test_configs/"
    NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_config.py $_dbg_args
    NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_sanity.py $_dbg_args
    NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_api_features.py $_dbg_args
    NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_perf.py $_dbg_args
    NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_numerics.py $_dbg_args
    #test_log.py keeps the arch guard: test_fp8_stats_allows_nvfp4_with_recipe_prefix requests
    #mxfp8 stats and fails, rather than skipping, when MXFP8 is unavailable
    check_mxfp8_supported && NVTE_ROCM_ENABLE_MXFP8=1 NVTE_TORCH_COMPILE=0 run_default_fa 1 debug/test_log.py $_dbg_args
}

run_test_config_mgpu(){
    echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
    configure_omp_threads 8
    run_default_fa 1 test_fused_optimizer.py
    #this test is not really mGPU but time sensitive so run it here because sGPU tests
    #run in parallel on CI and it affects timing
    run_default_fa 1 test_gemm_sm_count.py
    run_default_fa 3 test_sanity_import.py
    run_default_fa 3 distributed/test_cast_master_weights_to_fp8.py
    run_default_fa 3 distributed/test_comm_gemm_overlap.py
    run_default_fa 2 distributed/test_fusible_ops.py
    run_default_fa 2 distributed/test_numerics.py
    #mGPU only: on a single GPU this file asserts rather than skips
    run_default_fa 2 distributed/test_sanity.py
    run_default_fa 2 distributed/test_numerics_exact.py
    NVTE_TORCH_COMPILE=0 run_default_fa 2 debug/test_distributed.py --feature_dirs=${TE_PATH}transformer_engine/debug/features --configs_dir=${TE_PATH}tests/pytorch/debug/test_configs/
    run_default_fa 1 distributed/test_torch_fsdp2.py
    run_default_fa 2 distributed/test_torch_fsdp2_fp8.py
    if [ $_fus_attn = ck ]; then
        run 2 attention/test_attention_with_cp.py -k "with_fused"
    elif [ $_fus_attn = flash ]; then
        run 3 attention/test_attention_with_cp.py -k "with_flash"
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

    python "$BENCH_SCRIPT" --run_sanity_checks || test_run_error $BENCH_SCRIPT
}

# Single config mode, run it and return result
if [ -n "$SINGLE_CONFIG" ]; then
    _fus_attn="$SINGLE_CONFIG"
    configure_fused_attn_env $_fus_attn && run_test_config
    return_run_results
    exit $?
fi

check_flash_attn_installed() {
    _result=$(python -c "${PYTHON_TE_IMPORT}; from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils; print(FlashAttentionUtils.is_installed)" 2>/dev/null)
    if [ "$_result" = "True" ]; then
        return 0
    else
        echo "Flash attention is not installed" >&2
        return 1
    fi
}

#Master script mode: prepare testing prerequisites first
start_message
install_prerequisites
pip list | egrep "flash|ml_dtypes|numpy|torch|transformer_e|typing_ext"
#check_test_jobs_requested && init_test_jobs `python -c "import torch; print(torch.cuda.device_count())"`
ck_jit_prebuild build || exit $?

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

    if [ $_fus_attn = flash ]; then
        check_flash_attn_installed || continue
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

ck_jit_prebuild list
return_run_results
