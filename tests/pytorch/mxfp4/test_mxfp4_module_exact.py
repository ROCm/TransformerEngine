# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""End-to-end tests for TE modules with MXFP4 recipe vs CustomRecipe reference.

Compares native MXFP4BlockScaling recipe output against a CustomRecipe that
uses MXFP4QuantizerRef.  Tests Linear and LayerNormLinear modules with fwd+bwd.
"""

import gc

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import (
    mxfp4_ref_quantizer_factory,
)
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


recipe_available, reason_for_no_recipe = te.is_mxfp4_available(return_reason=True)

try:
    import aiter  # noqa: F401

    _aiter_available = True
except ImportError:
    _aiter_available = False


class GetRecipes:
    @staticmethod
    def mxfp4_vanilla():
        return recipe.MXFP4BlockScaling(use_hadamard=False)

    @staticmethod
    def mxfp4_hadamard():
        return recipe.MXFP4BlockScaling(use_hadamard=True)

    @staticmethod
    def mxfp4_recipe_to_test(use_hadamard: bool = False):
        if use_hadamard:
            return GetRecipes.mxfp4_hadamard()
        return GetRecipes.mxfp4_vanilla()


def get_mxfp4_quantizer_factory(use_hadamard: bool = False):
    """Create a quantizer factory for MXFP4 reference implementation."""
    def factory(role):  
        if role == "linear_input":
            return MXFP4QuantizerRef(rowwise=True, columnwise=True, shuffle_rowwise_data=False, shuffle_columnwise_data=False, with_gemm_swizzled_scales=False, use_hadamard=use_hadamard, use_te_quantizer=True)
        if role == "linear_weight":
            return MXFP4QuantizerRef(rowwise=True, columnwise=True, shuffle_rowwise_data=False, shuffle_columnwise_data=False, with_gemm_swizzled_scales=False, use_hadamard=use_hadamard, use_te_quantizer=True)
        elif role == "linear_output":
            # Output quantization not used
            return None
        if role == "linear_grad_output":
            return MXFP4QuantizerRef(rowwise=True, columnwise=True, shuffle_rowwise_data=False, shuffle_columnwise_data=False, with_gemm_swizzled_scales=False, use_hadamard=use_hadamard, use_te_quantizer=True)
        elif role == "linear_grad_input":
            # Grad input quantization not used
            return None
        else:
            return None
    return factory


def reset_rng_states():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def isolate_test_state():
    """Clear all global state that can leak between tests."""
    FP8GlobalStateManager.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def check_mxfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    use_hadamard: bool = False,
):
    """Compare native MXFP4 module against reference implementation."""
    isolate_test_state()

    device = "cuda"
    batch_size = 32
    seq_len = 128

    reset_rng_states()

    if module_class == te.Linear:
        native_module = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    elif module_class == te.LayerNormLinear:
        native_module = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    else:
        raise ValueError(f"Unsupported module class: {module_class}")

    reset_rng_states()

    if module_class == te.Linear:
        ref_module = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    elif module_class == te.LayerNormLinear:
        ref_module = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )

    # Sync weights
    with torch.no_grad():
        if hasattr(native_module, "weight") and hasattr(ref_module, "weight"):
            ref_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)
        if hasattr(native_module, "layer_norm_weight") and hasattr(ref_module, "layer_norm_weight"):
            if native_module.layer_norm_weight is not None:
                ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias") and hasattr(ref_module, "layer_norm_bias"):
            if native_module.layer_norm_bias is not None and ref_module.layer_norm_bias is not None:
                ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    mxfp4_recipe = GetRecipes.mxfp4_recipe_to_test(use_hadamard)
    mxfp4_ref_factory = get_mxfp4_quantizer_factory(use_hadamard)
    mxfp4_ref_recipe = recipe.CustomRecipe(qfactory=mxfp4_ref_factory)

    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_shape = (batch_size, seq_len, in_features)
        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output = grad_output_val.clone().detach()

        with te.autocast(enabled=True, recipe=mxfp4_recipe):
            y_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        with te.autocast(enabled=True, recipe=mxfp4_ref_recipe):
            y_ref = ref_module(x_ref)
        y_ref.backward(grad_output)
        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    native_module.bias.grad.detach().clone()
                    if bias and native_module.bias.grad is not None
                    else None
                ),
            }
        )

        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    ref_module.bias.grad.detach().clone()
                    if bias and ref_module.bias.grad is not None
                    else None
                ),
            }
        )

    for step in range(num_steps):
        native_out = native_outputs[step]
        ref_out = ref_outputs[step]

        torch.testing.assert_close(
            native_out["output"],
            ref_out["output"],
            atol=8e-3,
            rtol=8e-3,
        )

        torch.testing.assert_close(
            native_out["input_grad"],
            ref_out["input_grad"],
            atol=8e-3,
            rtol=8e-3,
            msg=f"Input gradient mismatch at step {step}",
        )


        torch.testing.assert_close(
            native_out["weight_grad"],
            ref_out["weight_grad"],
            atol=8e-3,
            rtol=8e-3,
            msg=f"Weight gradient mismatch at step {step}",
        )

        if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
            torch.testing.assert_close(
                native_out["bias_grad"],
                ref_out["bias_grad"],
                atol=8e-3,
                rtol=8e-3,
                msg=f"Bias gradient mismatch at step {step}",
            )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not _aiter_available, reason="aiter package not available")
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
        (512, 512),
        (768, 3072),
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1], ids=["single_step"])
@pytest.mark.parametrize("use_hadamard", [True, False], ids=["with_hadamard", "no_hadamard"])
def test_mxfp4_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int,
    use_hadamard: bool,
):
    """Test MXFP4 Linear module against reference implementation."""
    if x_dtype != torch.bfloat16:
        pytest.skip("MXFP4 quantization is only supported for bfloat16 input")

    check_mxfp4_module_versus_reference(
        module_class=te.Linear,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        num_steps=num_steps,
        use_hadamard=use_hadamard,
    )


def check_mxfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    use_hadamard: bool = False,
):
    """Compare native MXFP4 LayerNormLinear against reference, including ln_out."""
    isolate_test_state()

    device = "cuda"
    batch_size = 32
    seq_len = 128

    reset_rng_states()

    native_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    reset_rng_states()
    ref_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    with torch.no_grad():
        if hasattr(native_module, "weight") and hasattr(ref_module, "weight"):
            ref_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)
        if hasattr(native_module, "layer_norm_weight") and hasattr(ref_module, "layer_norm_weight"):
            if native_module.layer_norm_weight is not None:
                ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias") and hasattr(ref_module, "layer_norm_bias"):
            if native_module.layer_norm_bias is not None and ref_module.layer_norm_bias is not None:
                ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    mxfp4_recipe = GetRecipes.mxfp4_recipe_to_test(use_hadamard)
    mxfp4_ref_factory = get_mxfp4_quantizer_factory(use_hadamard)
    mxfp4_ref_recipe = recipe.CustomRecipe(qfactory=mxfp4_ref_factory)

    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_shape = (batch_size, seq_len, in_features)
        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output = grad_output_val.clone().detach()

        with te.autocast(enabled=True, recipe=mxfp4_recipe):
            y_native, ln_out_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        with te.autocast(enabled=True, recipe=mxfp4_ref_recipe):
            y_ref, ln_out_ref = ref_module(x_ref)
        y_ref.backward(grad_output)

        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "ln_out": ln_out_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
            }
        )
        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "ln_out": ln_out_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
            }
        )

    for step in range(num_steps):
        n = native_outputs[step]
        r = ref_outputs[step]
        torch.testing.assert_close(
            n["output"], r["output"], atol=8e-3, rtol=8e-3,
            msg=f"Output mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["ln_out"], r["ln_out"], atol=8e-3, rtol=8e-3,
            msg=f"LN output mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["input_grad"], r["input_grad"], atol=8e-3, rtol=8e-3,
            msg=f"Input gradient mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["weight_grad"], r["weight_grad"], atol=8e-3, rtol=8e-3,
            msg=f"Weight gradient mismatch at step {step}",
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not _aiter_available, reason="aiter package not available")
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
    ],
)
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1], ids=["single_step"])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"], ids=["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("use_hadamard", [True, False], ids=["with_hadamard", "no_hadamard"])
def test_mxfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int,
    use_hadamard: bool,
):
    check_mxfp4_layernorm_linear_versus_reference(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        normalization=normalization,
        x_dtype=x_dtype,
        num_steps=num_steps,
        use_hadamard=use_hadamard,
    )
