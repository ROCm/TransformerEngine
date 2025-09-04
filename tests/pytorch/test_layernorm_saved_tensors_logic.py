import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from transformer_engine.pytorch import LayerNormLinear, LayerNormMLP, fp8_autocast
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

@pytest.fixture(scope="module", autouse=True)
def _check_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device with BF16 support required for this FP8 test.")
    fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

class SimpleModel(nn.Module):
    def __init__(self, module_class, *module_args, **module_kwargs):
        super().__init__()
        self.ln_layer = module_class(*module_args, **module_kwargs)
    def forward(self, x):
        return self.ln_layer(x)


TEST_CONFIGS = {
    "linear": {
        "module_class": LayerNormLinear,
        "backward_target": 'transformer_engine.pytorch.module.layernorm_linear._LayerNormLinear.backward',
        "ln_out_index": -3,
        "num_grads": 41,
        "weight_name": "weight",
    },
    "mlp": {
        "module_class": LayerNormMLP,
        "backward_target": 'transformer_engine.pytorch.module.layernorm_mlp._LayerNormMLP.backward',
        "ln_out_index": 2,
        "num_grads": 46,
        "weight_name": "fc1_weight",
    },
}


@pytest.mark.parametrize("layernorm_type", ["linear", "mlp"])
def test_saved_tensors_logic(layernorm_type):
    """
    Directly tests if `ln_out` is saved for backward by spying on the
    `ctx.saved_tensors` tuple. This test is parameterized to run for
    both LayerNormLinear and LayerNormMLP.
    """
    config = TEST_CONFIGS[layernorm_type]
    M, N, H = 1024, 2048, 4096
    dtype = torch.bfloat16
    device = "cuda"

    if layernorm_type == "linear":
        model_args = (N, H)
        grad_output_shape = (M, H)
    else:  # mlp
        model_args = (N, H)  # hidden_size, ffn_hidden_size
        grad_output_shape = (M, N)

    model = SimpleModel(
        config["module_class"],
        *model_args,
        return_layernorm_output=True
    ).to(device=device, dtype=dtype)

    inp = torch.randn(M, N, requires_grad=True, device=device, dtype=dtype)
    grad_output = torch.randn(*grad_output_shape, device=device, dtype=dtype)

    saved_ln_out_container = []
    def spy_on_ctx(ctx, *args, **kwargs):
        saved_tensor = ctx.saved_tensors[config["ln_out_index"]]
        saved_ln_out_container.append(saved_tensor)
        # Backward must return a certain amount of gradients, thus, this mock returns the same amount of None
        return (None,) * config["num_grads"]

    # Baseline (Gradients ON for the fc1_weight)
    # We expect the full QuantizedTensor to be saved.
    weight_tensor = getattr(model.ln_layer, config["weight_name"])
    weight_tensor.requires_grad_(True)

    with patch(config["backward_target"], side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output, retain_graph=True)

    assert mock_backward.called, "The backward method was not called for the baseline test."
    assert len(saved_ln_out_container) == 1
    # Assert that a real QuantizedTensor was saved
    assert isinstance(saved_ln_out_container[0], QuantizedTensor), f"A QuantizedTensor should be saved for `ln_out` when grads are ON, got {type(saved_ln_out_container[0])} instead."

    # Gradients OFF
    # We expect `None` to be saved instead of the tensor.
    weight_tensor.requires_grad_(False)
    saved_ln_out_container.clear()

    with patch(config["backward_target"], side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output)

    assert mock_backward.called, "The backward method was not called for the `Gradients OFF` test."
    assert len(saved_ln_out_container) == 1
    # Assert that `None` was saved
    assert saved_ln_out_container[0] is None, f"`None` should be saved for `ln_out` when grads are OFF, got {type(saved_ln_out_container[0])} instead."
