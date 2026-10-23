# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

import transformer_engine.pytorch as te
import transformer_engine.pytorch.jit as te_jit

# Model names for test_torch_dynamo
_model_factory = {
    "Linear": [(lambda: te.Linear(16, 16)), [16, 16]],
    "LayerNorm": [(lambda: te.LayerNorm(16)), [16, 16]],
    "LayerNormLinear": [(lambda: te.LayerNormLinear(16, 16)), [16, 16]],
    "LayerNormMLP": [(lambda: te.LayerNormMLP(16, 16)), [16, 16]],
    "TransformerLayer": [(lambda: te.TransformerLayer(128, 128, 2)), [4, 1, 128]],
}


@pytest.mark.skipif(torch.__version__ < "2", reason="torch.compile not available")
@pytest.mark.parametrize("model_name", list(_model_factory.keys()))
def test_torch_dynamo(model_name: str):
    """Test compatibility with Torch Dynamo

    Construct model, optimize with Torch Dynamo, and perform a single
    forward and backward pass.

    """

    # Helper function to construct tensor with default options
    def make_tensor(
        dims: Tuple[int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        requires_grad: bool = True,
        **kwargs,
    ):
        return torch.zeros(
            dims,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            **kwargs,
        )

    # Construct model and input tensors
    model_builder, input_builder = _model_factory[model_name]
    model = model_builder()
    inputs = [make_tensor(input_builder)]

    # Optimize model with TorchDynamo
    torch.compile(model)

    # Forward and backward pass
    out = model(*inputs)
    out.backward(torch.zeros_like(out))


def test_lazy_compile():
    """Smoke test to ensure lazy compilation is working."""
    from transformer_engine.pytorch.jit import dgelu_fused_

    dgelu_fused_(torch.randn(10, 10), torch.randn(10, 10))


def _lse_ref(softmax_lse, softmax_lse_per_step):
    return torch.logaddexp(softmax_lse.double(), softmax_lse_per_step.double()).float()


def _run_lse_correction(func, seqlen, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    make = lambda *shape: torch.randn(*shape, device="cuda", generator=gen) * 3 + 5
    if func.__name__ == "flash_attn_fwd_softmax_lse_correction":
        # [h, t]
        softmax_lse, softmax_lse_per_step = make(8, seqlen), make(8, seqlen)
        ref = _lse_ref(softmax_lse, softmax_lse_per_step)
        func(softmax_lse, softmax_lse_per_step)
    else:
        # [b, h, s] merged with the second half of each sequence, [b, h, s//2]
        softmax_lse, softmax_lse_per_step = make(2, 8, seqlen), make(2, 8, seqlen // 2)
        ref = softmax_lse.clone()
        ref[..., seqlen // 2 :] = _lse_ref(softmax_lse[..., seqlen // 2 :], softmax_lse_per_step)
        func(softmax_lse.view(*softmax_lse.shape[:-1], 2, -1), softmax_lse_per_step)
    return softmax_lse, ref


@pytest.mark.skipif(not IS_HIP_EXTENSION, reason="Dynamic-shape compilation is ROCm only")
@pytest.mark.skipif(
    te_jit.jit_fuser_dynamic is not te_jit.lazy_compile_dynamic,
    reason="torch.compile is disabled",
)
@pytest.mark.parametrize(
    "func_name",
    [
        "flash_attn_fwd_softmax_lse_correction",
        "flash_attn_fwd_second_half_softmax_lse_correction",
    ],
)
def test_cp_softmax_lse_correction_single_compile(func_name):
    """CP softmax LSE merges must compile once and stay bit-exact across input shapes.

    See: https://github.com/ROCm/TransformerEngine/issues/693
    """
    import torch._dynamo
    from torch._dynamo.utils import counters
    from transformer_engine.pytorch.attention.dot_product_attention import context_parallel

    func = getattr(context_parallel, func_name)
    torch._dynamo.reset()
    counters.clear()

    out_first, ref = _run_lse_correction(func, 4096, seed=0)
    for i, seqlen in enumerate((3072, 5120, 2000)):
        _run_lse_correction(func, seqlen, seed=i + 1)
    out_again, _ = _run_lse_correction(func, 4096, seed=0)

    num_graphs = counters["stats"]["unique_graphs"]
    assert num_graphs == 1, f"{func_name} was compiled into {num_graphs} graphs, expected 1"
    torch.testing.assert_close(out_again, out_first, rtol=0, atol=0)
    torch.testing.assert_close(out_first, ref, rtol=1e-6, atol=1e-6)


def test_l2normalization_fused():
    """Smoke test for L2Normalization fusion functions."""
    from transformer_engine.pytorch.jit import (
        l2normalization_fused,
        l2normalization_fwd_fused,
        l2normalization_backward_fused,
    )

    # Basic smoke test like other JIT functions
    x = torch.randn(10, 128, device="cuda", dtype=torch.float32)
    eps = 1e-6

    # Test inference version
    output_inf = l2normalization_fused(x, eps)

    # Test training version with backward
    x_train = torch.randn(10, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    output_train, rsqrt_norm = l2normalization_fwd_fused(x_train, eps)
    grad_output = torch.randn_like(output_train)
    grad_input = l2normalization_backward_fused(grad_output, x_train, rsqrt_norm, eps)


def test_l2normalization_fused_correctness():
    """Simple verification that L2Normalization fusion matches reference implementation."""
    from transformer_engine.pytorch.jit import (
        l2normalization_fwd_fused,
        l2normalization_backward_fused,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 64, device=device, dtype=torch.float32, requires_grad=True)
    eps = 1e-6

    # Test fused forward
    output_fused, rsqrt_norm = l2normalization_fwd_fused(x, eps)

    # Reference implementation
    x_ref = x.clone().detach().requires_grad_(True)
    x_squared = x_ref.pow(2)
    l2_norm_squared = x_squared.sum(dim=-1, keepdim=True)
    rsqrt_norm_ref = torch.rsqrt(l2_norm_squared + eps)
    output_ref = x_ref * rsqrt_norm_ref

    # Check forward pass matches
    torch.testing.assert_close(output_fused, output_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rsqrt_norm, rsqrt_norm_ref, atol=1e-6, rtol=1e-5)

    # Test fused backward
    grad_output = torch.randn_like(output_fused)
    grad_input_fused = l2normalization_backward_fused(grad_output, x, rsqrt_norm, eps)

    # Reference backward
    output_ref.backward(grad_output)
    grad_input_ref = x_ref.grad

    # Check backward pass matches
    torch.testing.assert_close(grad_input_fused, grad_input_ref, atol=1e-5, rtol=1e-4)
