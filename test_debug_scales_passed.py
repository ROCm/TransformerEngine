import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Simple case: 32x32, all ones
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
print("Debugging scale values")
print("=" * 60)

print(f"\nA rowwise scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"A rowwise scale (E8M0):")
print(a_mxfp8._rowwise_scale_inv)

print(f"\nB columnwise scale shape: {b_mxfp8._columnwise_scale_inv.shape}")
print(f"B columnwise scale (E8M0):")
print(b_mxfp8._columnwise_scale_inv)

# Convert E8M0 to actual scale values
a_scale_fp32 = 2.0 ** (a_mxfp8._rowwise_scale_inv.float() - 127.0)
b_scale_fp32 = 2.0 ** (b_mxfp8._columnwise_scale_inv.float() - 127.0)

print(f"\nA rowwise scale (FP32):")
print(a_scale_fp32)

print(f"\nB columnwise scale (FP32):")
print(b_scale_fp32)

# For all ones quantized to FP8, the scale should be close to 1.0 (E8M0 = 127)
# Let's see what we actually get

# Check FP8 values
a_fp8_as_fp32 = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
b_fp8_as_fp32 = b_mxfp8._columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)

print(f"\nA FP8 values (as FP32, first row):")
print(a_fp8_as_fp32[0, :])

print(f"\nB FP8 values (as FP32, first column):")
print(b_fp8_as_fp32[:, 0])

# Expected: for value 1.0, FP8 representation should be 1.0, scale should be ~1.0
# So E8M0 should be ~127

print(f"\nTo reconstruct original value:")
print(f"  Original = FP8_value * scale")
print(f"  For A[0,0]: {a_fp8_as_fp32[0,0].item():.4f} * {a_scale_fp32[0,0].item():.4f} = {(a_fp8_as_fp32[0,0] * a_scale_fp32[0,0]).item():.4f}")
print(f"  Should be: 1.0")

# Check which block [0,0] belongs to
print(f"\nFor A[0,0], block index in K dimension: {0 // 32} (out of {K//32} blocks)")
print(f"  So scale index should be [0, 0]")
print(f"  Scale E8M0: {a_mxfp8._rowwise_scale_inv[0, 0].item()}")
print(f"  Scale FP32: {a_scale_fp32[0, 0].item():.6f}")
