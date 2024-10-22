#!/bin/sh
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

DIR=`dirname $0`

. $DIR/_utils.sh

TEST_DIR=${TE_PATH}tests/jax

run_1() {
    check_level 1 || return
    echo "Run [$_fus_attn] $*"
    pytest "$TEST_DIR/$@" || test_run_error
}

for _fus_attn in auto ck aotriton unfused; do
    configure_fused_attn_env $_fus_attn || continue

    #On basic (1) level tests are run with ck/aotriton/unfused
    #On full (3) level they are run with auto/aotriton/unfused
    if [ $TEST_LEVEL -ge 3 ]; then
        test $_fus_attn = "ck" && continue
    else
        test $_fus_attn = "auto" && continue
    fi

    run_1 test_custom_call_compute.py
    run_1 test_distributed_fused_attn.py
    run_1 test_distributed_layernorm.py
    run_1 test_distributed_layernorm_mlp.py
    run_1 test_distributed_softmax.py
    run_1 test_functions.py
    run_1 test_fused_attn.py
    run_1 test_helper.py
    if [ $_fus_attn != "unfused" ]; then
        #Layer tests control Fused attn so we can only play with backend
        run_1 test_layer.py
        run_1 test_praxis_layers.py
    fi
    run_1 test_sharding.py
    run_1 test_softmax.py
done

return_run_results