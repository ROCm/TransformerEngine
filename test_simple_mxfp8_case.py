import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

# Use very simple dimensions: 32x32 matrices (one block each)
M, N, K = 32, 32, 32

# Create simple test data: all ones
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Simple test: all ones, single block per matrix")
print("=" * 60)

print(f"\nA [{M}, {K}]: all ones")
print(f"  Rowwise scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  Expected: [{M}, {K//32}] = [32, 1]")

print(f"\nB [{K}, {N}]: all ones")
print(f"  Columnwise scale shape: {b_mxfp8._columnwise_scale_inv.shape}")
print(f"  Expected: [{K//32}, {N}] = [1, 32]")

# Expected result: C[i,j] = sum_k A[i,k] * B[k,j] = 32 * 1 * 1 = 32
expected = torch.full((M, N), 32.0, dtype=torch.bfloat16, device=device)

# Reference
ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
print(f"\nReference result:")
print(f"  All should be ~32: min={ref.min().item():.2f}, max={ref.max().item():.2f}")

# Kernel
a_fp8 = reinterpret_as_fp8_tensor(a_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
b_fp8 = reinterpret_as_fp8_tensor(b_mxfp8._columnwise_data, tex.DType.kFloat8E4M3)

c_kernel = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

mxfp8_matmul(
    a_fp8, a_mxfp8._rowwise_scale_inv,
    b_fp8, b_mxfp8._columnwise_scale_inv,
    c_kernel,
    M, N, K,
    tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
)

print(f"\nKernel result:")
print(f"  All should be ~32: min={c_kernel.min().item():.2f}, max={c_kernel.max().item():.2f}")

print(f"\nComparison:")
print(f"  Max diff from expected (32.0): {torch.max(torch.abs(c_kernel - expected)).item():.4f}")
print(f"  Max diff from reference: {torch.max(torch.abs(c_kernel - ref)).item():.4f}")

# Check specific values
print(f"\nFirst few values:")
print(f"  Expected: {expected[0, :5]}")
print(f"  Reference: {ref[0, :5]}")
print(f"  Kernel: {c_kernel[0, :5]}")
