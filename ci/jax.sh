#!/bin/sh
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

export NVTE_FRAMEWORK=jax

DIR=`dirname $0`

. $DIR/_utils.sh

install_prerequisites() {
    _req_file=$(mktemp)
    pip list | awk '$1=="jax" || $1=="jaxlib" { print $1"=="$2 }' > "$_req_file"
    echo "flax>=0.7.1" >> "$_req_file"
    echo "typing_extensions>=4.12.2" >> "$_req_file"
    pip install -r "$_req_file"
    rc=$?
    rm -f "$_req_file"
    if [ $rc -ne 0 ]; then
        script_error "Failed to install Flax and dependencies"
        return $rc
    fi
}

TEST_DIR=${TE_PATH}tests/jax

run() {
    check_level $1 || return
    shift
    _test_name_tag=`get_test_name_tag $1 $_fus_attn`
    check_test_filter $_test_name_tag || return
    echo "Run [$_fus_attn] $*"
    pytest -v `get_pytest_junitxml $_test_name_tag` "$TEST_DIR/$@" || test_run_error "[$_fus_attn] $1"
    echo "Done [$_fus_attn] $1"
}

run_default_fa() {
    #Run tests that do not use fused attention with only one backend
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        run $*
    fi
}

run_test_config() {
    echo ==== Run with Fused attention backend: $_fus_attn ====
    run_default_fa 1 test_custom_call_compute.py
    run_default_fa 1 test_functions.py
    run 1 test_fused_attn.py
    NVTE_CK_USES_FWD_V3=1 NVTE_CK_USES_BWD_V3=1 run 1 test_fused_attn.py # Using FAv3 for forward and backward pass
    run_default_fa 1 test_helper.py
    run_default_fa 1 test_layer.py #it effectevly always uses unfused attention
    run_default_fa 1 test_sharding.py
    run_default_fa 1 test_softmax.py
}

run_test_config_mgpu() {
    echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
    
    _JAX_DISABLE_JIT_FLAG=${JAX_DISABLE_JIT:-0}
    _ver=$(pip show jaxlib | grep Version)
    case "$_ver" in
    *0.4.35*)
        # Workaround for distributed tests hang with JIT enabled
        JAX_DISABLE_JIT=1 run 3 test_distributed_fused_attn.py -k 'not (test_context_parallel_allgather_attn[BALANCED or test_context_parallel_ring_attn)'
        _JAX_DISABLE_JIT_FLAG=1

        # Run tests that fail with JIT disabled
        run 3 test_distributed_fused_attn.py -k 'test_context_parallel_allgather_attn[BALANCED'

        # Test ring attention with xla_flag --xla_experimental_ignore_channel_id only
        # TODO: remove this flag after jax/xla update
        XLA_FLAGS="--xla_experimental_ignore_channel_id" run 3 test_distributed_fused_attn.py -k test_context_parallel_ring_attn
        ;;
    *0.4.31*)
        #Workaround for JAX 0.4.31 regression: crash in test_destributed_fused_attn and test_distributed_layernorm_mlp
        export XLA_FLAGS="--xla_gpu_enable_dot_strength_reduction=false --xla_gpu_enable_command_buffer=CUSTOM_CALL"
        run 3 test_distributed_fused_attn.py
        ;;
    esac
    
    run_default_fa 3 test_distributed_layernorm.py
    JAX_DISABLE_JIT=$_JAX_DISABLE_JIT_FLAG run_default_fa 3 test_distributed_layernorm_mlp.py
    run_default_fa 3 test_distributed_softmax.py
    unset XLA_FLAGS

    run_default_fa 3 test_sanity_import.py
}

# Single config mode, run it synchroniously and return result
if [ -n "$SINGLE_CONFIG" ]; then
    _fus_attn="$SINGLE_CONFIG"
    configure_fused_attn_env $_fus_attn && run_test_config
    return_run_results
    exit $?
fi

#Master script mode: prepares testing prerequisites
start_message
install_prerequisites
pip list | egrep "flax|fidle|jax|ml_dtypes|numpy|transformer_e|typing_ext"
#check_test_jobs_requested
#test $? -eq 0 && init_test_jobs `python -c "import jax; print(len([d for d in jax.devices() if 'rocm' in d.client.platform_version]))"`

for _fus_attn in auto ck aotriton; do
    configure_fused_attn_env $_fus_attn || continue

    #On basic (1) level tests are run with ck
    #On full (3) level they are run with auto/aotriton
    #Do not use unfused becaue JAX tests either do not use FA or enforce it
    if [ $TEST_LEVEL -ge 3 ]; then
        _DEFAULT_FUSED_ATTN="auto"
        test $_fus_attn = "ck" && continue
    else
        _DEFAULT_FUSED_ATTN="ck"
        test $_fus_attn != "ck" && continue
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
        configure_fused_attn_env $_fus_attn && run_test_config_mgpu
    done
fi
return_run_results
