import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N = 64, 128

# Create simple test matrix
a_fp32 = torch.randn((M, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)

print("=" * 60)
print("Understanding rowwise vs columnwise quantization")
print("=" * 60)

print(f"\nOriginal matrix: [{M}, {N}]")

print(f"\nRowwise quantization:")
print(f"  Data shape: {a_mxfp8._rowwise_data.shape}")
print(f"  Scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  Interpretation: {M} rows, each with {N//32} scale blocks")

print(f"\nColumnwise quantization:")
print(f"  Data shape: {a_mxfp8._columnwise_data.shape}")
print(f"  Scale shape: {a_mxfp8._columnwise_scale_inv.shape}")
print(f"  Interpretation: {N} columns, each with {M//32} scale blocks")

# Check if data shapes differ
print(f"\nData shapes are same: {a_mxfp8._rowwise_data.shape == a_mxfp8._columnwise_data.shape}")

# Check if data content is the same
data_same = torch.allclose(
    a_mxfp8._rowwise_data.float(),
    a_mxfp8._columnwise_data.float()
)
print(f"Data content is same: {data_same}")

# The scale shapes should be transposed
expected_colwise_scale = (M//32, N)
print(f"\nColumnwise scale expected: {expected_colwise_scale}")
print(f"Columnwise scale actual: {a_mxfp8._columnwise_scale_inv.shape}")
print(f"Match: {a_mxfp8._columnwise_scale_inv.shape == expected_colwise_scale}")

# Now test with transpose
print(f"\n" + "=" * 60)
print(f"If we want matrix [N, M] (transposed):")
print(f"=" * 60)

# Transpose the original
a_T_fp32 = a_fp32.T.contiguous()
a_T_mxfp8 = quantizer.quantize(a_T_fp32)

print(f"\nTransposed matrix: [{N}, {M}]")
print(f"\nRowwise quantization of transpose:")
print(f"  Data shape: {a_T_mxfp8._rowwise_data.shape}")
print(f"  Scale shape: {a_T_mxfp8._rowwise_scale_inv.shape}")

print(f"\nColumnwise quantization of transpose:")
print(f"  Data shape: {a_T_mxfp8._columnwise_data.shape}")
print(f"  Scale shape: {a_T_mxfp8._columnwise_scale_inv.shape}")

# Compare with original columnwise
print(f"\nHypothesis: Original's columnwise quantization is like rowwise of transpose")
print(f"  Original columnwise data: {a_mxfp8._columnwise_data.shape}")
print(f"  Transpose rowwise data: {a_T_mxfp8._rowwise_data.shape}")

print(f"\n  Original columnwise scale: {a_mxfp8._columnwise_scale_inv.shape}")
print(f"  Transpose rowwise scale: {a_T_mxfp8._rowwise_scale_inv.shape}")
