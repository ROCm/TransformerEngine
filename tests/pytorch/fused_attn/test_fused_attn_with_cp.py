# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from test_fused_attn import (
    ModelConfig,
    _is_flash_attention_2_available,
)
if not IS_HIP_EXTENSION:
    from test_fused_attn import (
        _cudnn_version,
    )
#TODO: release GQA tests once CK/AOTriton support GQA/MQA
if IS_HIP_EXTENSION:
    model_configs = {
        #   test:             b,  h, hg,   d,    sq,   skv,   p,      mask,      bias
        "cp_1_0": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # MHA
        "cp_1_1": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # MHA
    }
else:
    model_configs = {
        #   test:             b,  h, hg,   d,    sq,   skv,   p,      mask,      bias
        "cp_1_0": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # MHA
        "cp_1_1": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # MHA
        "cp_2_0": ModelConfig(1, 12,  1, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # GQA
        "cp_2_1": ModelConfig(1, 12,  1, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # GQA
    }

def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2"]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args

@pytest.mark.skipif(not _is_flash_attention_2_available(), reason="Flash-attn 2.0+ is required.")
@pytest.mark.parametrize("dtype", ['bf16', 'fp16'])
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("qkv_format", ['bshd', 'sbhd'])
def test_cp_with_flash_attention(dtype, model, qkv_format):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend='FlashAttention'
        ),
        check=True
    )

@pytest.mark.skipif(False if IS_HIP_EXTENSION else _cudnn_version() < (8,9,7), reason="cuDNN 8.9.7+ is required for NVTE.")
@pytest.mark.parametrize("dtype", ['bf16', 'fp16'])
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("qkv_format", ['bshd', 'sbhd'])
def test_cp_with_fused_attention(dtype, model, qkv_format):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend='FusedAttention'
        ),
        check=True
    )
