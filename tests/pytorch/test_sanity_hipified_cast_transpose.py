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


@pytest.mark.parametrize("num_experts", [4, 8, 64])
@pytest.mark.parametrize("hidden_dim", [256, 1536, 4096])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_fused_vs_unfused_padding_mct(num_experts, hidden_dim, fp8_dtype):
    """Compare fused padding+MCT against unfused (pad BF16 then MCT)."""
    actual_splits = [torch.randint(1, 200, (1,)).item() for _ in range(num_experts)]
    align         = 16
    padded_splits = [(m + align - 1) // align * align for m in actual_splits]
    total_actual  = sum(actual_splits)
    total_padded  = sum(padded_splits)

    inp_unpadded = _fill_uniform((total_actual, hidden_dim), torch.bfloat16)

    inp_padded = torch.zeros((total_padded, hidden_dim), dtype=torch.bfloat16, device="cuda")
    src_offset = 0
    dst_offset = 0
    for actual, padded in zip(actual_splits, padded_splits):
        inp_padded[dst_offset:dst_offset + actual] = inp_unpadded[src_offset:src_offset + actual]
        src_offset += actual
        dst_offset += padded

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    amax  = torch.zeros(1, dtype=torch.float32, device="cuda")

    def make_quantizers():
        return [Float8Quantizer(scale=scale.clone(), amax=amax.clone(), fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
                for _ in range(num_experts)]

    outputs_unfused = tex.split_quantize(inp_padded, padded_splits, make_quantizers())

    outputs_fused = tex.split_quantize(inp_unpadded, padded_splits, make_quantizers(), valid_split_sections=actual_splits)

    assert len(outputs_unfused) == len(outputs_fused) == num_experts
    for e in range(num_experts):
        data_u, trans_u = outputs_unfused[e].get_data_tensors(rowwise_data=True, columnwise_data=True)
        data_f, trans_f = outputs_fused[e].get_data_tensors(rowwise_data=True, columnwise_data=True)
        assert data_u.shape == data_f.shape, f"Expert {e}: rowwise shape mismatch"
        assert torch.equal(data_u, data_f), f"Expert {e}: rowwise data mismatch"
        assert trans_u.shape == trans_f.shape, f"Expert {e}: columnwise shape mismatch"
        assert torch.equal(trans_u, trans_f), f"Expert {e}: columnwise data mismatch"
