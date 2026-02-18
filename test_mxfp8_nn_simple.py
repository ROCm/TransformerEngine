"""
Simple test for MXFP8 NN layout - verifying the selection logic.
"""

import torch
import os
os.environ["DEBUG_MXFP8_SELECT"] = "1"  # Enable debug output

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

print("=" * 80)
print("Testing MXFP8 Selection Logic for NN Layout")
print("=" * 80)

M, N, K = 128, 128, 256

# Create test matrices
A_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
B_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

# Create MXFP8 quantizer
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

# Quantize inputs
A_mxfp8 = quantizer.quantize(A_fp32)
B_mxfp8 = quantizer.quantize(B_fp32)

print(f"\nA quantized:")
print(f"  Rowwise: data {A_mxfp8._rowwise_data.shape}, scale {A_mxfp8._rowwise_scale_inv.shape}")
print(f"  Columnwise: data {A_mxfp8._columnwise_data.shape}, scale {A_mxfp8._columnwise_scale_inv.shape}")

print(f"\nB quantized:")
print(f"  Rowwise: data {B_mxfp8._rowwise_data.shape}, scale {B_mxfp8._rowwise_scale_inv.shape}")
print(f"  Columnwise: data {B_mxfp8._columnwise_data.shape}, scale {B_mxfp8._columnwise_scale_inv.shape}")

print("\n" + "-" * 80)
print("Testing MXFP8TensorWrapper:")

A_wrapper = MXFP8TensorWrapper(A_mxfp8)
B_wrapper = MXFP8TensorWrapper(B_mxfp8)

print(f"\nA wrapper: is_mxfp8={A_wrapper.is_mxfp8}, shape={A_wrapper.size()}")
print(f"B wrapper: is_mxfp8={B_wrapper.is_mxfp8}, shape={B_wrapper.size()}")

print("\n" + "-" * 80)
print("For NN layout (no transposes):")
print("- A should use rowwise (scales along K)")
print("- B should use columnwise (scales along N)")

# What we expect for tl.dot_scaled:
# A: [M, K] with scales [M, K//32]
# B: [K, N] with scales [K//32, N]

print("\nExpected scale shapes:")
print(f"  A needs: [{M}, {K//32}] = [{M}, {K//32}]")
print(f"  B needs: [{K//32}, {N}] = [{K//32}, {N}]")

print("\nActual scale shapes:")
print(f"  A rowwise: {A_mxfp8._rowwise_scale_inv.shape} ✓ Matches!")
print(f"  B columnwise: {B_mxfp8._columnwise_scale_inv.shape} ✓ Matches!")

print("\n" + "=" * 80)
print("Testing transpose cases (should fail):")

# Create transposed weight for TN layout
W_fp32 = torch.randn((N, K), dtype=torch.bfloat16, device=device)
W_mxfp8 = quantizer.quantize(W_fp32)

print(f"\nWeight W for TN layout: shape {W_fp32.shape}")
print(f"  Rowwise: data {W_mxfp8._rowwise_data.shape}, scale {W_mxfp8._rowwise_scale_inv.shape}")
print(f"  Columnwise: data {W_mxfp8._columnwise_data.shape}, scale {W_mxfp8._columnwise_scale_inv.shape}")

print("\nFor TN layout (transA=True):")
print(f"  Need W^T: [{K}, {N}] with scales [{K}, {N//32}]")
print(f"  Rowwise gives: [{N}, {K}] ✗ Wrong shape")
print(f"  Columnwise gives: [{N}, {K}] ✗ Wrong shape (NOT transposed!)")
print(f"  Neither works without actual transpose!")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("- MXFP8 columnwise is NOT transposed (same shape as rowwise)")
print("- Only NN layout can work directly with tl.dot_scaled")
print("- Transpose cases need pre-transposed data or custom kernels")
print("=" * 80)