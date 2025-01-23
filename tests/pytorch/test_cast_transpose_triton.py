# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.cast_transpose_triton import te_cast_transpose_noop_triton, te_cast_transpose_dbias_triton, get_te_dtype
from transformer_engine.pytorch.cpp_extensions import fused_cast_transpose_noop, fused_cast_transpose_bgrad

def get_tolerances(in_dtype):
    if in_dtype == torch.float32:
        return 1e-6, 5e-6
    elif in_dtype == torch.float16:
        return 1e-5, 1e-3
    elif in_dtype == torch.bfloat16:
        return 1e-5, 1e-2

@pytest.mark.parametrize("M, N", 
                         [(2048, 12288),
                          (768, 1024),
                          (256, 65536),
                          (65536, 128),
                          (256, 256),
                          (120, 2080),
                          (8, 8),
                          (1, 3221),
                          (2333, 1),
                          (1481, 677),
                          (40960, 128256)
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])
def test_cast_tranpose_triton(M, N, in_dtype, out_dtype):
    ## Unit distribution between [-2.0, 1.0]
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    scale_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    noop_flag = torch.Tensor()

    casted_tensor = torch.empty(M, N, dtype=torch.uint8, device='cuda')
    transposed_tensor = torch.empty(N, M, dtype=torch.uint8, device='cuda')
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    scale_inv_tensor = torch.empty(1, dtype=torch.float32, device='cuda')
    
    casted_tensor_triton = torch.empty(M, N, dtype=torch.uint8, device='cuda')
    transposed_tensor_triton = torch.empty(N, M, dtype=torch.uint8, device='cuda')
    amax_tensor_triton = torch.zeros(1, dtype=torch.float32, device='cuda')
    scale_inv_tensor_triton = torch.empty(1, dtype=torch.float32, device='cuda')

    fused_cast_transpose_noop(input_tensor, noop_flag, scale_tensor, amax_tensor, scale_inv_tensor, casted_tensor, transposed_tensor, get_te_dtype(out_dtype), 0, 0, 0)
    te_cast_transpose_noop_triton(input_tensor, noop_flag, scale_tensor, casted_tensor_triton, transposed_tensor_triton, amax_tensor_triton, scale_inv_tensor_triton, get_te_dtype(out_dtype))

    assert torch.equal(casted_tensor, casted_tensor_triton), 'Casted results do not match!'
    assert torch.equal(transposed_tensor, transposed_tensor_triton), 'transposed results do not match!'
    assert torch.allclose(amax_tensor, amax_tensor_triton, atol=1e-6, rtol=5e-6), 'Amax results do not match!'
    assert torch.allclose(scale_inv_tensor, scale_inv_tensor_triton, atol=1e-6, rtol=5e-6), 'Scale inverse results do not match!'

@pytest.mark.parametrize("M, N", 
                         [(64, 400),
                          (2048, 12288),
                          (768, 1024),
                          (256, 65536),
                          (65536, 128),
                          (256, 256),
                          (40960, 128256)
                          ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])
def test_cast_tranpose_dbias_triton(M, N, in_dtype, out_dtype):
    ## Unit distribution between [-2.0, 1.0]
    input_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    input_tensor = input_tensor.to(in_dtype)
    scale_tensor = torch.rand(M, N, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    noop_flag = torch.Tensor()

    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    scale_inv_tensor = torch.empty(1, dtype=torch.float32, device='cuda')
    
    amax_tensor_triton = torch.zeros(1, dtype=torch.float32, device='cuda')
    scale_inv_tensor_triton = torch.empty(1, dtype=torch.float32, device='cuda')

    dbias_tensor, casted_tensor, transposed_tensor = fused_cast_transpose_bgrad(input_tensor, scale_tensor, amax_tensor, scale_inv_tensor, get_te_dtype(out_dtype), 0, 0, 0)
    dbias_tensor_triton, casted_tensor_triton, transposed_tensor_triton = te_cast_transpose_dbias_triton(input_tensor, scale_tensor, amax_tensor_triton, scale_inv_tensor_triton, get_te_dtype(out_dtype))

    assert torch.equal(casted_tensor, casted_tensor_triton), 'Casted results do not match!'
    assert torch.equal(transposed_tensor, transposed_tensor_triton), 'transposed results do not match!'
    assert torch.allclose(amax_tensor, amax_tensor_triton, atol=1e-6, rtol=5e-6), 'Amax results do not match!'
    atol, rtol = get_tolerances(in_dtype)
    rtol *= 4
    assert torch.allclose(dbias_tensor, dbias_tensor_triton, atol=atol, rtol=rtol), 'Amax results do not match!'
    assert torch.allclose(scale_inv_tensor, scale_inv_tensor_triton, atol=1e-6, rtol=5e-6), 'Scale inverse results do not match!'

