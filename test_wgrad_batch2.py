"""
Test wgrad with batch size 2 to understand the shape issue.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"
os.environ["DEBUG_MXFP8_SELECT"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

device = torch.device("cuda")
torch.manual_seed(42)

print("=" * 80)
print("Testing wgrad with batch size 2")
print("=" * 80)

# Parameters matching Llama-8B
out_features = 4096
in_features = 14336
batch = 2
seq_len = 2048  # Typical sequence length

print(f"\nModel parameters:")
print(f"  out_features: {out_features}")
print(f"  in_features: {in_features}")
print(f"  batch: {batch}")
print(f"  seq_len: {seq_len}")

# The Linear module flattens [batch, seq_len, features] to [batch*seq_len, features]
batch_seq = batch * seq_len  # 2 * 2048 = 4096

# Create test tensors
# Input: [batch*seq, in_features]
input_tensor = torch.randn(batch_seq, in_features, dtype=torch.bfloat16, device=device)
# Grad output: [batch*seq, out_features]
grad_output_tensor = torch.randn(batch_seq, out_features, dtype=torch.bfloat16, device=device)

print(f"\nFlattened tensor shapes (what wgrad expects):")
print(f"  Input: {input_tensor.shape} = [batch*seq, in_features]")
print(f"  Grad output: {grad_output_tensor.shape} = [batch*seq, out_features]")

# Quantize to MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

input_mxfp8 = quantizer.quantize(input_tensor)
grad_output_mxfp8 = quantizer.quantize(grad_output_tensor)

print(f"\nMXFP8 storage shapes:")
print(f"  Input rowwise: {input_mxfp8._rowwise_data.shape}")
print(f"  Input columnwise: {input_mxfp8._columnwise_data.shape}")
print(f"  Grad output rowwise: {grad_output_mxfp8._rowwise_data.shape}")
print(f"  Grad output columnwise: {grad_output_mxfp8._columnwise_data.shape}")

# Test wgrad GEMM with layout NT
print(f"\n" + "=" * 80)
print("Testing wgrad GEMM (layout='NT')")
print("=" * 80)

try:
    result = general_gemm(
        input_mxfp8,
        grad_output_mxfp8,
        torch.empty(0, device=device),  # workspace
        layout="NT",
        out_dtype=torch.bfloat16,
        grad=True,
    )
    print(f"SUCCESS! Result shape: {result[0].shape}")
    print(f"Expected shape: [{out_features}, {in_features}]")
    if result[0].shape == torch.Size([out_features, in_features]):
        print("✓ Shape matches expected weight gradient!")
except Exception as e:
    print(f"ERROR: {e}")

# Now let's test what happens if we have multi-dimensional input
# with shape [batch, seq_len, in_features]
print(f"\n" + "=" * 80)
print("Testing with multi-dimensional input (not flattened)")
print("=" * 80)

input_3d = torch.randn(batch, seq_len, in_features, dtype=torch.bfloat16, device=device)
grad_output_3d = torch.randn(batch, seq_len, out_features, dtype=torch.bfloat16, device=device)

print(f"3D tensor shapes:")
print(f"  Input: {input_3d.shape} = [batch, seq_len, in_features]")
print(f"  Grad output: {grad_output_3d.shape} = [batch, seq_len, out_features]")

input_3d_mxfp8 = quantizer.quantize(input_3d)
grad_output_3d_mxfp8 = quantizer.quantize(grad_output_3d)

print(f"\n3D MXFP8 storage shapes:")
print(f"  Input rowwise: {input_3d_mxfp8._rowwise_data.shape}")
print(f"  Input columnwise: {input_3d_mxfp8._columnwise_data.shape}")
print(f"  Grad output rowwise: {grad_output_3d_mxfp8._rowwise_data.shape}")
print(f"  Grad output columnwise: {grad_output_3d_mxfp8._columnwise_data.shape}")

try:
    result_3d = general_gemm(
        input_3d_mxfp8,
        grad_output_3d_mxfp8,
        torch.empty(0, device=device),  # workspace
        layout="NT",
        out_dtype=torch.bfloat16,
        grad=True,
    )
    print(f"Result shape: {result_3d[0].shape}")
    print(f"Expected shape: [{out_features}, {in_features}]")
except Exception as e:
    print(f"ERROR: {e}")