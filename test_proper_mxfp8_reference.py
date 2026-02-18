import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

M, N, K = 128, 256, 512

torch.manual_seed(42)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Testing MXFP8 GEMM with proper reference")
print("=" * 60)

# Kernel uses:
# - A: rowwise data + rowwise scale
# - B: columnwise data + columnwise scale

# So reference should dequantize using the same quantizations:
# But there's no direct API to dequantize columnwise separately...

# Let's use dequantize() which should give the right answer
# Actually, let me manually dequantize columnwise for B

# For now, use the standard dequantize as reference
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()

ref = torch.matmul(a_dequant, b_dequant)

print(f"\nReference using dequantize():")
print(f"  Result[0,0]: {ref[0, 0].item():.4f}")

# Now run kernel with columnwise B
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
print(f"  Result[0,0]: {c_kernel[0, 0].item():.4f}")

diff = torch.max(torch.abs(c_kernel - ref)).item()
print(f"\nMax diff: {diff:.4f}")

# The question is: what's the right reference?
# If b_mxfp8.dequantize() uses rowwise, but kernel uses columnwise,
# they're computing slightly different things!

# Let me check if the issue is that we should be using rowwise B data
# with columnwise scales (which doesn't make sense)

print(f"\n" + "=" * 60)
print("Trying rowwise B data with columnwise scales (shouldn't work):")
print("=" * 60)

b_fp8_rowwise = reinterpret_as_fp8_tensor(b_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
c_kernel2 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

try:
    mxfp8_matmul(
        a_fp8, a_mxfp8._rowwise_scale_inv,
        b_fp8_rowwise, b_mxfp8._columnwise_scale_inv,  # Mismatched!
        c_kernel2,
        M, N, K,
        tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
    )
    print(f"Result[0,0]: {c_kernel2[0, 0].item():.4f}")
    diff2 = torch.max(torch.abs(c_kernel2 - ref)).item()
    print(f"Max diff: {diff2:.4f}")
except Exception as e:
    print(f"Error: {e}")

print(f"\n" + "=" * 60)
print("Trying rowwise B data with rowwise scales:")
print("=" * 60)

c_kernel3 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

# But rowwise scales have wrong shape [K, N//32] instead of [K//32, N]
# So this should fail or give wrong results
print(f"B rowwise scale shape: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"Expected by kernel: [{K//32}, {N}] = [16, 256]")
print(f"Rowwise scale is [{K}, {N//32}] = [512, 8] - WRONG SHAPE!")
