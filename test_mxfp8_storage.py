"""
Check how MXFP8 actually stores columnwise data.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Create a simple matrix to understand the storage
M, K = 128, 256  # Must be divisible by 32 for MXFP8
torch.manual_seed(42)

# Create distinct values to track
a_fp32 = torch.arange(M * K, dtype=torch.float32, device=device).reshape(M, K) * 0.1

print("Original matrix A:")
print(a_fp32)
print(f"Shape: {a_fp32.shape}")

# Quantize
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32.to(torch.bfloat16))

print("\nMXFP8 storage:")
print(f"Rowwise data shape: {a_mxfp8._rowwise_data.shape}")
print(f"Rowwise scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"Columnwise data shape: {a_mxfp8._columnwise_data.shape if a_mxfp8._columnwise_data is not None else None}")
print(f"Columnwise scale shape: {a_mxfp8._columnwise_scale_inv.shape if a_mxfp8._columnwise_scale_inv is not None else None}")

# Check if columnwise is actually transposed
if a_mxfp8._columnwise_data is not None:
    print("\nChecking if columnwise is transposed:")
    # Dequantize rowwise
    row_dequant = a_mxfp8.dequantize()
    print(f"Rowwise dequantized shape: {row_dequant.shape}")

    # Dequantize columnwise manually
    VEC_SIZE = 32
    col_data = a_mxfp8._columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)
    col_scale = a_mxfp8._columnwise_scale_inv

    print(f"\nColumnwise data first few elements:")
    print(f"col_data[0, :4] = {col_data[0, :4]}")
    print(f"col_data[1, :4] = {col_data[1, :4]}")

    print(f"\nOriginal data first few elements:")
    print(f"a_fp32[0, :4] = {a_fp32[0, :4]}")
    print(f"a_fp32[:4, 0] = {a_fp32[:4, 0]}")

    # Check if columnwise[i,j] corresponds to original[j,i] (transposed)
    # or original[i,j] (not transposed)

    # Simple test: is columnwise row 0 the same as original column 0?
    # Note: need to account for quantization differences

print("\n" + "=" * 60)
print("Testing larger matrix:")
M, N, K = 128, 128, 256

# Test with actual use case
a_shape = (M, K)
b_shape = (K, N)

a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"\nA: {a_shape}")
print(f"  Rowwise: {a_mxfp8._rowwise_data.shape}")
print(f"  Columnwise: {a_mxfp8._columnwise_data.shape if a_mxfp8._columnwise_data is not None else None}")

print(f"\nB: {b_shape}")
print(f"  Rowwise: {b_mxfp8._rowwise_data.shape}")
print(f"  Columnwise: {b_mxfp8._columnwise_data.shape if b_mxfp8._columnwise_data is not None else None}")

# The key question: Is columnwise actually transposed?
# From the documents, it should be stored transposed
# But our debug output shows both have the same shape!

print("\nConclusion:")
print("If columnwise has the same shape as rowwise, it's NOT transposed!")
print("This would explain why our scale assumptions are wrong.")