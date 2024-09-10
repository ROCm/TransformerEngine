# This file was modified for portability to AMDGPU
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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
    from test_fused_attn import _cudnn_version
    from transformer_engine.pytorch.utils import get_device_compute_capability

model_configs_flash_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0,  "causal", "no_bias"), # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"), # MHA
    "cp_2_0": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0,  "causal", "no_bias"), # GQA
    "cp_2_1": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"), # GQA
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
@pytest.mark.skipif(IS_HIP_EXTENSION or get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ['bf16', 'fp16'])
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", ['bshd', 'sbhd', 'thd'])
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

#TODO: release GQA tests once CK/AOTriton support GQA/MQA
if IS_HIP_EXTENSION:
    model_configs_fused_attn = {
        #   test:             b,  h, hg,   d,    sq,   skv,   p,      mask,      bias
        "cp_1_0": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # MHA
        "cp_1_1": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # MHA
    }
else:
    model_configs_fused_attn = {
        #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,              bias
        "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0,  "causal",         "no_bias"), # MHA
        "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask",         "no_bias"), # MHA
        "cp_1_2": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0,  "causal", "post_scale_bias"), # MHA
        "cp_1_3": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"), # MHA
        "cp_2_0": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0,  "causal",         "no_bias"), # GQA
        "cp_2_1": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0, "no_mask",         "no_bias"), # GQA
        "cp_2_2": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0,  "causal", "post_scale_bias"), # GQA
        "cp_2_3": ModelConfig(2, 12,  1, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"), # GQA
    }

@pytest.mark.skipif(not IS_HIP_EXTENSION and _cudnn_version() < (8,9,7), reason="cuDNN 8.9.7+ is required for NVTE.")
@pytest.mark.skipif(not IS_HIP_EXTENSION and get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ['bf16', 'fp16'])
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ['bshd', 'sbhd', 'thd'])
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
