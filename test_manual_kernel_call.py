import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

# Create tensors with the dimensions we expect the kernel to see
# For NN layout: we want kernel to compute (M, K) @ (K, N) = (M, N)
M, N, K = 128, 256, 512

print("Creating test tensors...")
print(f"Want to compute: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

# Create FP32 tensors
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

# Quantize
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,  # Only rowwise for simplicity
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"\nMXFP8 tensors:")
print(f"A data: {a_mxfp8._rowwise_data.shape}, scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"B data: {b_mxfp8._rowwise_data.shape}, scale: {b_mxfp8._rowwise_scale_inv.shape}")

# Convert to native FP8
a_data_fp8 = reinterpret_as_fp8_tensor(a_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
b_data_fp8 = reinterpret_as_fp8_tensor(b_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)

# Output tensor
c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

print(f"\nCalling mxfp8_matmul directly...")
print(f"  a: {a_data_fp8.shape}, a_scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  b: {b_data_fp8.shape}, b_scale: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"  c: {c.shape}")
print(f"  M={M}, N={N}, K={K}")

try:
    mxfp8_matmul(
        a_data_fp8, a_mxfp8._rowwise_scale_inv,
        b_data_fp8, b_mxfp8._rowwise_scale_inv,
        c,
        M, N, K,
        tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
    )
    print(f"\n✓ Kernel succeeded!")
    print(f"Output shape: {c.shape}")

    # Check against reference
    a_dequant = a_mxfp8.dequantize() if hasattr(a_mxfp8, 'dequantize') else a_fp32
    b_dequant = b_mxfp8.dequantize() if hasattr(b_mxfp8, 'dequantize') else b_fp32
    ref = torch.matmul(a_dequant, b_dequant)

    max_diff = torch.max(torch.abs(c.float() - ref.float())).item()
    print(f"Max difference from reference: {max_diff}")

except Exception as e:
    print(f"\n✗ Kernel failed: {e}")
    import traceback
    traceback.print_exc()
