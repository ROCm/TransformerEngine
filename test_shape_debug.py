"""
Debug shape handling for wgrad operation.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

device = torch.device("cuda")
torch.manual_seed(42)

# Test wgrad operation with strange shapes
print("Testing wgrad shape issue")
print("=" * 80)

# wgrad layout: NT (transa=False, transb=True)
# grad_weight = general_gemm(input, grad_output, layout="NT")
# Expected: dW = X^T @ dY (in row-major)

# Simulate the shapes from Megatron-LM error
# A (input): [14336, 1, 2048]
# B (grad_output): [2048, 4096]
# Expected output: [4096, 14336]

# Create tensors with these shapes
input_tensor = torch.randn(14336, 1, 2048, dtype=torch.bfloat16, device=device)
grad_output_tensor = torch.randn(2048, 4096, dtype=torch.bfloat16, device=device)

print(f"Input shape: {input_tensor.shape}")
print(f"Grad output shape: {grad_output_tensor.shape}")

# Quantize to MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

input_mxfp8 = quantizer.quantize(input_tensor)
grad_output_mxfp8 = quantizer.quantize(grad_output_tensor)

print(f"\nMXFP8 storage shapes:")
print(f"  Input rowwise: {input_mxfp8._rowwise_data.shape}, columnwise: {input_mxfp8._columnwise_data.shape}")
print(f"  Grad output rowwise: {grad_output_mxfp8._rowwise_data.shape}, columnwise: {grad_output_mxfp8._columnwise_data.shape}")

# Try wgrad GEMM with layout NT
try:
    print(f"\nCalling general_gemm with layout='NT'")
    result = general_gemm(
        input_mxfp8,
        grad_output_mxfp8,
        torch.empty(0, device=device),  # workspace
        layout="NT",
        out_dtype=torch.bfloat16,
        grad=True,
    )
    print(f"Result shape: {result[0].shape}")
except Exception as e:
    print(f"ERROR: {e}")

# Now let's understand what the correct interpretation should be
print("\n" + "=" * 80)
print("Understanding the shape issue:")
print("-" * 80)

# The issue is that input has an extra dimension [14336, 1, 2048]
# This seems to be [seqlen*batch, 1, in_features] where the middle dim is 1
# But grad_output is [out_features, in_features] transposed to [2048, 4096]

# Let's test with the flattened version
input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])  # [14336, 2048]
print(f"\nFlattened input shape: {input_flat.shape}")

# Quantize the flattened version
input_flat_mxfp8 = quantizer.quantize(input_flat)

# Try wgrad GEMM with flattened input
try:
    print(f"\nCalling general_gemm with flattened input, layout='NT'")
    result2 = general_gemm(
        input_flat_mxfp8,
        grad_output_mxfp8,
        torch.empty(0, device=device),  # workspace
        layout="NT",
        out_dtype=torch.bfloat16,
        grad=True,
    )
    print(f"Result shape: {result2[0].shape}")
    print(f"Expected shape: [4096, 14336] or [14336, 4096]")
except Exception as e:
    print(f"ERROR: {e}")

# Check what the reference computation would give
print("\n" + "=" * 80)
print("Reference computation:")
print("-" * 80)

# NT layout means: A no transpose, B transpose
# In row-major: X @ dY^T
ref_result = input_flat @ grad_output_tensor.T
print(f"Reference (X @ dY^T) shape: {ref_result.shape}")
print(f"This is wrong! We want dW = X^T @ dY")

# Actually for wgrad we want X^T @ dY
ref_correct = input_flat.T @ grad_output_tensor.T  # Wait, this doesn't match either

# Actually, grad_output might already be the right orientation
# If grad_output is [2048, 4096], that's [batch*seq, out_features] flattened
# No wait, that doesn't make sense either

print("\nLet me reconsider the shapes:")
print(f"Input: [14336, 2048] = [batch*seq, in_features]")
print(f"Grad output: [2048, 4096] - this seems wrong!")
print(f"  Should grad_output be [batch*seq, out_features] = [14336, 4096]?")

# Actually looking at the error message from Megatron:
# "got [4096, 2048] but expected shape compatible with [4096, 14336]"
# So the weight should be [out_features, in_features] = [4096, 14336]
# But that means in_features = 14336, which contradicts input being [batch*seq, 2048]

print("\nActual interpretation:")
print("Weight should be [4096, 14336] = [out_features, in_features]")
print("So in_features = 14336, out_features = 4096")
print("Input should be [batch*seq, in_features] = [batch*seq, 14336]")
print("But we have input [14336, 2048]")
print("This suggests the dimensions are swapped somewhere!")