# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.common import torch_dtype_to_te_dtype
from transformer_engine.pytorch.triton_kernels.utils import get_num_sms

def test_sm_margin():
    num_sms = get_num_sms()
    assert num_sms > 0
    assert get_num_sms(0) == num_sms
    assert get_num_sms(-5) == num_sms
    assert get_num_sms(1) == num_sms - 1
    assert get_num_sms(100 * num_sms) == 1


def test_torch_dtype_to_te_dtype():
    assert torch_dtype_to_te_dtype(torch.float32) == tex.DType.kFloat32
    assert torch_dtype_to_te_dtype(torch.float16) == tex.DType.kFloat16
    assert torch_dtype_to_te_dtype(torch.bfloat16) == tex.DType.kBFloat16
    assert torch_dtype_to_te_dtype(torch.float8_e4m3fnuz) == tex.DType.kFloat8E4M3
    assert torch_dtype_to_te_dtype(torch.float8_e4m3fn) == tex.DType.kFloat8E4M3
    assert torch_dtype_to_te_dtype(torch.float8_e5m2fnuz) == tex.DType.kFloat8E5M2
    assert torch_dtype_to_te_dtype(torch.float8_e5m2) == tex.DType.kFloat8E5M2
