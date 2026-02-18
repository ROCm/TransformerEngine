import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

M, N, K = 32, 32, 32

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
print("Test with manually sliced scales")
print("=" * 60)

# Get scales
a_scale_padded = a_mxfp8._rowwise_scale_inv
b_scale_padded = b_mxfp8._columnwise_scale_inv

print(f"\nPadded scales:")
print(f"  A: {a_scale_padded.shape}")
print(f"  B: {b_scale_padded.shape}")

# Manually slice to correct sizes
VEC_SIZE = 32
a_scale = a_scale_padded[:M, :K//VEC_SIZE].contiguous()
b_scale = b_scale_padded[:K//VEC_SIZE, :N].contiguous()

print(f"\nSliced scales:")
print(f"  A: {a_scale.shape} (expected [{M}, {K//VEC_SIZE}] = [32, 1])")
print(f"  B: {b_scale.shape} (expected [{K//VEC_SIZE}, {N}] = [1, 32])")

# Get data
a_fp8 = reinterpret_as_fp8_tensor(a_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
b_fp8 = reinterpret_as_fp8_tensor(b_mxfp8._columnwise_data, tex.DType.kFloat8E4M3)

# Run kernel with sliced scales
c_kernel = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

mxfp8_matmul(
    a_fp8, a_scale,
    b_fp8, b_scale,
    c_kernel,
    M, N, K,
    tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
)

# Reference
ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())

print(f"\nResults:")
print(f"  Reference: all values should be 32.0")
print(f"    min={ref.min().item():.2f}, max={ref.max().item():.2f}")
print(f"  Kernel with sliced scales:")
print(f"    min={c_kernel.min().item():.2f}, max={c_kernel.max().item():.2f}")

print(f"\nFirst row:")
print(f"  Ref:    {ref[0, :5]}")
print(f"  Kernel: {c_kernel[0, :5]}")

max_diff = torch.max(torch.abs(c_kernel - ref)).item()
print(f"\nMax diff: {max_diff:.4f}")
