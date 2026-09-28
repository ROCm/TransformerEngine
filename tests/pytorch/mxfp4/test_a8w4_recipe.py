# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""End-to-end test: te.Linear under the a8w4 CustomRecipe."""

import gc

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4_grouped import (
    MXFP8E4M3QuantizerRef,
    a8w4_quantizer_factory,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

_recipe_ok, _reason = te.is_mxfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not _recipe_ok, reason=_reason)


def _isolate():
    FP8GlobalStateManager.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _e2m1_ref():
    return MXFP4QuantizerRef(
        rowwise=True,
        columnwise=True,
        shuffle_rowwise_data=False,
        shuffle_columnwise_data=False,
        with_gemm_swizzled_scales=False,
        use_hadamard=False,
    )


def test_a8w4_custom_recipe_linear_forward():
    """te.Linear forward under CustomRecipe(a8w4) == the recipe's qgemm reference."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    B, S, K, N = 2, 128, 256, 256  # K=256 -> loop_k>1 in the eventual fast path
    torch.manual_seed(0)

    module = te.Linear(K, N, bias=False, device=device, params_dtype=dtype)
    x = torch.randn(B, S, K, device=device, dtype=dtype)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with te.autocast(enabled=True, recipe=rec):
        y = module(x)

    # Reference
    xq = MXFP8E4M3QuantizerRef(rowwise=True, columnwise=False).quantize(x.reshape(-1, K))
    wq = _e2m1_ref().quantize(module.weight.detach())
    y_ref = MXFP8E4M3QuantizerRef().qgemm(
        xq.data, wq.data, None, dtype, xq.scale, wq.scale, qresult_x=xq, qresult_w=wq,
    ).reshape(B, S, N)

    rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-12)
    assert y.shape == (B, S, N)
    assert rel < 5e-2, f"a8w4 recipe forward disagrees with its qgemm reference: rel={rel:.4f}"


def test_a8w4_custom_recipe_linear_runs_backward():
    """Smoke: fwd+bwd under the a8w4 recipe complete without error and grads are finite."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    B, S, K, N = 2, 128, 256, 256
    torch.manual_seed(0)

    module = te.Linear(K, N, bias=False, device=device, params_dtype=dtype)
    x = torch.randn(B, S, K, device=device, dtype=dtype, requires_grad=True)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with te.autocast(enabled=True, recipe=rec):
        y = module(x)
    y.backward(torch.randn_like(y))

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
