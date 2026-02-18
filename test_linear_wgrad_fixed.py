"""
Test the actual Linear module's wgrad to understand the shape issue.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"
os.environ["DEBUG_MXFP8_SELECT"] = "1"

from transformer_engine.pytorch import Linear
from transformer_engine.pytorch.fp8 import fp8_autocast, fp8_model_init
import transformer_engine.common.recipe as te_recipe

device = torch.device("cuda")
torch.manual_seed(42)

print("=" * 80)
print("Testing Linear module wgrad")
print("=" * 80)

# Parameters matching Llama-8B MLP
in_features = 14336
out_features = 4096
batch = 2
seq_len = 2048

print(f"\nModel parameters:")
print(f"  in_features: {in_features}")
print(f"  out_features: {out_features}")
print(f"  batch: {batch}")
print(f"  seq_len: {seq_len}")

# Create Linear module with MXFP8
recipe = te_recipe.MXFP8BlockScaling()

with fp8_model_init(enabled=True):
    linear = Linear(in_features, out_features, bias=False, device=device)

print(f"\nLinear weight shape: {linear.weight.shape}")

# Create input with shape [batch, seq_len, in_features]
input_tensor = torch.randn(batch, seq_len, in_features, dtype=torch.bfloat16, device=device, requires_grad=True)
print(f"Input shape: {input_tensor.shape}")

# Forward pass with MXFP8
with fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = linear(input_tensor)
    print(f"Output shape: {output.shape}")

    # Create grad output
    grad_output = torch.randn_like(output)
    print(f"Grad output shape: {grad_output.shape}")

    # Backward pass
    try:
        output.backward(grad_output)
        print(f"SUCCESS! Weight grad computed")
        if linear.weight.grad is not None:
            print(f"Weight grad shape: {linear.weight.grad.shape}")
    except Exception as e:
        print(f"ERROR during backward: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Testing with flattened input (manual reshape)")
print("=" * 80)

# Reset gradients
if linear.weight.grad is not None:
    linear.weight.grad.zero_()
if input_tensor.grad is not None:
    input_tensor.grad.zero_()

# Manually flatten input to [batch*seq_len, in_features]
input_flat = input_tensor.reshape(-1, in_features).detach().requires_grad_()
print(f"Flattened input shape: {input_flat.shape}")

# Forward pass with flattened input
with fp8_autocast(enabled=True, fp8_recipe=recipe):
    output_flat = linear(input_flat)
    print(f"Output shape: {output_flat.shape}")

    # Create grad output
    grad_output_flat = torch.randn_like(output_flat)
    print(f"Grad output shape: {grad_output_flat.shape}")

    # Backward pass
    try:
        output_flat.backward(grad_output_flat)
        print(f"SUCCESS! Weight grad computed")
        if linear.weight.grad is not None:
            print(f"Weight grad shape: {linear.weight.grad.shape}")
    except Exception as e:
        print(f"ERROR during backward: {e}")
        import traceback
        traceback.print_exc()