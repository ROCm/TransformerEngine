import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from transformer_engine.pytorch import LayerNormLinear, LayerNormMLP, fp8_autocast
from transformer_engine.pytorch.tensor import QuantizedTensor

@pytest.fixture(scope="module", autouse=True)
def _check_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device with BF16 support required for this FP8 test.")

# A simple model containing the LayerNormMLP layer we want to test
class SimpleModel(nn.Module):
    def __init__(self, M, N, layernorm_type="mlp"):
        super().__init__()
        if layernorm_type == "mlp":
            self.ln_layer = LayerNormMLP(
                hidden_size=M,
                ffn_hidden_size=N,
                return_layernorm_output=True
            )
        elif layernorm_type == "linear":
            self.ln_layer = LayerNormLinear(
                in_features=M,
                out_features=N,
                return_layernorm_output=True
            )
        else:
            assert False, "Wrong layernorm_type. Choose between `mlp` or `linear`."

    def forward(self, x):
        return self.ln_layer(x)

def test_layernorm_mlp_saved_tensors_logic():
    """
    Tests if `ln_out` is saved for backward in LayerNormMLP by
    spying on `ctx.saved_tensors` inside the backward pass.
    """
    tokens, hidden_size, ffn_hidden_size = 256, 1024, 4096
    dtype = torch.bfloat16
    device = "cuda"

    model = SimpleModel(hidden_size, ffn_hidden_size, layernorm_type="mlp").to(device=device, dtype=dtype)
    inp = torch.randn(tokens, hidden_size, requires_grad=True, device=device, dtype=dtype)
    grad_output = torch.randn(tokens, hidden_size, device=device, dtype=dtype) # MLP output is same shape as input

    # The full path to the internal backward method we want to spy on
    backward_method_target = 'transformer_engine.pytorch.module.layernorm_mlp._LayerNormMLP.backward'
    saved_ln_out_container = []

    def spy_on_ctx(ctx, *args, **kwargs):
        """This spy function inspects ctx.saved_tensors and stores the result."""
        # For _LayerNormMLP, ln_out is the 3rd tensor saved (index 2)
        saved_tensor = ctx.saved_tensors[2]
        saved_ln_out_container.append(saved_tensor)
        # _LayerNormMLP.forward takes 46 arguments, so backward must return 46 gradients
        return (None,) * 46

    # Baseline (Gradients ON for the fc1_weight)
    # We expect the full QuantizedTensor to be saved.
    model.ln_layer.fc1_weight.requires_grad_(True)

    with patch(backward_method_target, side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output, retain_graph=True)

    assert mock_backward.called, "The backward method was not called for the baseline test."
    assert len(saved_ln_out_container) == 1
    # Assert that a real QuantizedTensor was saved
    assert isinstance(saved_ln_out_container[0], QuantizedTensor), f"A QuantizedTensor should be saved for `ln_out` when grads are ON, got {type(saved_ln_out_container[0])} instead."

    # Gradients OFF
    # We expect `None` to be saved instead of the tensor.
    model.ln_layer.fc1_weight.requires_grad_(False)
    saved_ln_out_container.clear() # Reset for the next run

    with patch(backward_method_target, side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output)

    assert mock_backward.called, "The backward method was not called for the `Gradients OFF` test."
    assert len(saved_ln_out_container) == 1
    # Assert that `None` was saved
    assert saved_ln_out_container[0] is None, f"`None` should be saved for `ln_out` when grads are OFF, got {type(saved_ln_out_container[0])} instead."


def test_layernorm_linear_saved_tensors_logic():
    """
    Tests if `ln_out` is saved for backward in LayerNormLinear by
    spying on `ctx.saved_tensors` inside the backward pass.
    """
    M, N, H = 1024 * 4, 2048 * 4, 4096 * 4
    dtype = torch.bfloat16
    device = "cuda"

    model = SimpleModel(N, H, layernorm_type="linear").to(device=device, dtype=dtype)
    inp = torch.randn(M, N, requires_grad=True, device=device, dtype=dtype)
    grad_output = torch.randn(M, H, device=device, dtype=dtype)

    # The full path to the internal backward method we want to spy on
    backward_method_target = 'transformer_engine.pytorch.module.layernorm_linear._LayerNormLinear.backward'
    saved_ln_out_container = []

    def spy_on_ctx(ctx, *args, **kwargs):
        """This spy function runs instead of the real backward method."""
        # For _LayerNormLinear, we know ln_out is the 3rd last tensor saved (index -3)
        saved_tensor = ctx.saved_tensors[-3]
        saved_ln_out_container.append(saved_tensor)
        # _LayerNormLinear.forward takes 41 arguments, so backward must return 41 gradients
        return (None,) * 41

    # Baseline (gradients ON)
    # We expect the full QuantizedTensor to be saved.
    model.ln_layer.weight.requires_grad_(True)

    with patch(backward_method_target, side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output, retain_graph=True)

    assert mock_backward.called, "The backward method was not called for the baseline test."
    assert len(saved_ln_out_container) == 1
    # Assert that a real QuantizedTensor was saved
    assert isinstance(saved_ln_out_container[0], QuantizedTensor), f"A QuantizedTensor should be saved for `ln_out` when grads are ON, got {type(saved_ln_out_container[0])} instead."

    # Gradients OFF
    # We expect `None` to be saved instead of the tensor.
    model.ln_layer.weight.requires_grad_(False)
    inp = torch.randn(M, N, requires_grad=False, device=device, dtype=dtype)
    saved_ln_out_container.clear()  # Reset for the next run

    with patch(backward_method_target, side_effect=spy_on_ctx) as mock_backward:
        with fp8_autocast(enabled=True):
            out, ln_out_returned = model(inp)
            out.backward(grad_output, retain_graph=True)

    assert mock_backward.called, "The backward method was not called for the `Gradients OFF` test."
    assert len(saved_ln_out_container) == 1
    # Assert that `None` was saved
    assert saved_ln_out_container[0] is None, f"`None` should be saved for `ln_out` when grads are OFF, got {type(saved_ln_out_container[0])} instead."
