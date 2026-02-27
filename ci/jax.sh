#!/bin/sh
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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
    pip install pytest-timeout
    rc=$?
    if [ $rc -ne 0 ]; then
        script_error "Failed to install test prerequisites"
        exit $rc
    fi
}

TEST_DIR=${TE_PATH}tests/jax

run() {
    pytest_run $_fus_attn "" "$@"
}

run_default_fa() {
    #Run tests that do not use fused attention with only one backend
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        run $*
    fi
}

run_lbl() {
    pytest_run $_fus_attn "$@"
}

run_default_fa_lbl() {
    if [ $_fus_attn = "$_DEFAULT_FUSED_ATTN" ]; then
        run_lbl "$@"
    fi
}

run_test_config() {
    echo ==== Run with Fused attention backend: $_fus_attn ====
    export NVTE_JAX_UNITTEST_LEVEL=L0 # this env variable controls parameters set for some tests
    run_default_fa 1 test_custom_call_compute.py
    run_default_fa 1 test_functions.py
    run 1 test_fused_attn.py
    XLA_FLAGS='--xla_gpu_graph_level=0' run 1 test_fused_attn.py -k 'test_ck_unfused_smallseq_backend' # CK smallseq with GPU graph disabled
    NVTE_CK_USES_FWD_V3=0 NVTE_CK_USES_BWD_V3=0 run_default_fa_lbl "v2" 3 test_fused_attn.py # Using FAv2 for forward and backward pass
    run_default_fa 1 test_helper.py
    run_default_fa 1 test_layer.py #it effectevly always uses unfused attention
    run_default_fa 1 test_sanity_import.py
    run_default_fa 1 test_softmax.py
}

run_test_config_mgpu() {
    echo ==== Run mGPU with Fused attention backend: $_fus_attn ====
    configure_omp_threads 8

    # Mitigate distributed tests hang by adding 5min timeout
    _timeout_args="--timeout 300 --timeout-method thread"
    # Workaround for some distributed tests hang/abotrion
    export XLA_FLAGS="--xla_gpu_enable_nccl_comm_splitting=false"

    if [ $_fus_attn = $_DEFAULT_FUSED_ATTN ]; then
        _dfa_level=2
        export NVTE_JAX_UNITTEST_LEVEL=L1
    else
        _dfa_level=3
        export NVTE_JAX_UNITTEST_LEVEL=L2
    fi
    # Do not fail automated CI if test_distributed_fused_attn is hung
    # If the sctipt run w/o TEST_LEVEL the test error will be honored
    if [ "$TEST_LEVEL" -le 3 ]; then
        TEST_ERROR_IGNORE="1"
    fi
    run $_dfa_level test_distributed_fused_attn.py $_timeout_args
    TEST_ERROR_IGNORE=""
    run_default_fa 3 test_distributed_layernorm.py
    run_default_fa 2 test_distributed_layernorm_mlp.py $_timeout_args
    run_default_fa 3 test_distributed_softmax.py

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

    #On basic (1) level tests are run with auto
    #On medium (2) level they are run with ck and aotriton
    #On full (3) level they are run with auto and aotriton
    #Do not use unfused becaue JAX tests either do not use FA or enforce it
    if [ $TEST_LEVEL -ge 3 ]; then
        _DEFAULT_FUSED_ATTN="auto"
        test $_fus_attn = "ck" && continue
    elif [ $TEST_LEVEL -ge 2 ]; then
        _DEFAULT_FUSED_ATTN="ck"
        test $_fus_attn = "auto" && continue
    else
        _DEFAULT_FUSED_ATTN="auto"
        test $_fus_attn != "auto" && continue
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
