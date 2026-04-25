# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""
Non-regression tests for the hipified cast_transpose kernel dispatch.

Verifies that tex.quantize (which routes through cast_transpose on AMD)
dispatches exactly one GPU kernel per call.  A previous bug (commit 542b8c7b)
caused the NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE path to launch both
cast_transpose_optimized_kernel AND cast_transpose_general_kernel per call,
doubling the GPU work.
"""

import pytest
import torch
from torch.profiler import profile, ProfilerActivity

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer


def _fill_uniform(shape, dtype):
    """Create a deterministic random tensor on GPU."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(12345)
    return torch.empty(shape, dtype=dtype, device="cuda").uniform_(-2.0, 2.0, generator=gen)


@pytest.mark.parametrize("shape", [
    (128, 128),
    (2048, 12288),
    (256, 256),
])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_single_kernel_dispatch(shape, in_dtype, out_dtype, monkeypatch):
    """
    Verify that tex.quantize dispatches exactly one cast_transpose GPU kernel when using PR #89 hipified path
    (NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1).
    """
    input_tensor = _fill_uniform(shape, dtype=in_dtype)
    scale = torch.rand(1, dtype=torch.float32, device="cuda") * 3.0 - 2.0
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")

    monkeypatch.setenv("NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE", "1")

    # Warmup (also triggers hipRTC compilation)
    q = Float8Quantizer(scale=scale.clone(), amax=amax.clone(), fp8_dtype=out_dtype)
    tex.quantize(input_tensor, q)
    torch.cuda.synchronize()

    # Profiled run
    q = Float8Quantizer(scale=scale.clone(), amax=amax.clone(), fp8_dtype=out_dtype)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tex.quantize(input_tensor, q)
        torch.cuda.synchronize()

    ct_kernels = [
        evt.name for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA
        and "cast_transpose" in evt.name
    ]

    assert len(ct_kernels) == 1, (
        f"Expected exactly 1 cast_transpose kernel, got {len(ct_kernels)}: {ct_kernels}"
    )
