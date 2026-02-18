import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

M, N, K = 128, 128, 128

# Simple controlled test
torch.manual_seed(42)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Direct kernel vs manual computation")
print("=" * 60)

# Method 1: Dequantize then matmul (reference)
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()
ref = torch.matmul(a_dequant, b_dequant)

print(f"\n1. Reference (dequantize + matmul):")
print(f"  Result[0,0]: {ref[0, 0].item():.4f}")

# Method 2: Direct kernel call
a_fp8 = reinterpret_as_fp8_tensor(a_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
b_fp8 = reinterpret_as_fp8_tensor(b_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
c_kernel = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

mxfp8_matmul(
    a_fp8, a_mxfp8._rowwise_scale_inv,
    b_fp8, b_mxfp8._rowwise_scale_inv,
    c_kernel,
    M, N, K,
    tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
)

print(f"\n2. Direct kernel call:")
print(f"  Result[0,0]: {c_kernel[0, 0].item():.4f}")

# Method 3: Manual FP32 matmul with FP8 values (no scaling)
a_fp8_as_fp32 = a_fp8.to(torch.float32)
b_fp8_as_fp32 = b_fp8.to(torch.float32)
unscaled = torch.matmul(a_fp8_as_fp32, b_fp8_as_fp32)

print(f"\n3. Unscaled FP8 matmul (FP8→FP32, no scales):")
print(f"  Result[0,0]: {unscaled[0, 0].item():.4f}")

# Method 4: Manual scaling application
# For block-scaled matmul, we need to apply scales properly
# Let me try to manually apply the scales like tl.dot_scaled should

# Each element C[i,j] = sum_k(A[i,k] * B[k,j])
# With MXFP8: C[i,j] = sum_{block_k} sum_{within_block} (A[i,k] * scale_A[i, block_k] * B[k,j] * scale_B[block_k, j])

# Hmm, actually tl.dot_scaled should handle this. Let me just check the difference
print(f"\n4. Comparison:")
print(f"  Ref vs Kernel diff: {abs(ref[0, 0] - c_kernel[0, 0]).item():.4f}")
print(f"  Max diff: {torch.max(torch.abs(ref - c_kernel)).item():.4f}")

# Check a few elements
print(f"\n5. First row comparison:")
print(f"  Ref:    {ref[0, :5]}")
print(f"  Kernel: {c_kernel[0, :5]}")

# Check scale shapes
print(f"\n6. Scale shapes:")
print(f"  A scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  B scale: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"  Expected A: [{M}, {K//32}] = [{M}, {K//32}]")
print(f"  Expected B: [{K}, {N//32}] = [{K}, {N//32}]")

# Check if B scales are right
if b_mxfp8._rowwise_scale_inv.shape != (K, N//32):
    print(f"  ⚠ B scale shape mismatch!")
    print(f"    Got {b_mxfp8._rowwise_scale_inv.shape}, need ({K}, {N//32})")
