import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N = 64, 128

# Create a specific pattern to understand quantization
a_fp32 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

# Fill first row with 1.0, second row with 2.0, etc.
for i in range(M):
    a_fp32[i, :] = float(i + 1)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)

print(f"Original: [{M}, {N}]")
print(f"First column: {a_fp32[:5, 0]}")
print()

# Dequantize and check
a_dequant = a_mxfp8.dequantize()

print(f"Dequantized shape: {a_dequant.shape}")
print(f"First column after dequant: {a_dequant[:5, 0]}")
print(f"Match: {torch.allclose(a_fp32, a_dequant, rtol=0.1)}")
print()

# Check scale shapes
print(f"Rowwise scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"Columnwise scale shape: {a_mxfp8._columnwise_scale_inv.shape}")
print()

# Check what rowwise scale represents
# If rowwise means "one scale per row", we'd expect M scales
# If the shape is [N, M//32] then it's transposed?

print(f"Number of rows (M): {M}")
print(f"Number of row-blocks (M//32): {M//32}")
print(f"Number of columns (N): {N}")
print(f"Number of column-blocks (N//32): {N//32}")
print()

# Hypothesis: rowwise scale might be stored as [N, M//32] (transposed layout)
# Or it could be that "rowwise" means "quantized in row-major order" which is different

# Let's check if scales match expected pattern
print(f"For data [{M}, {N}]:")
print(f"  If rowwise = scales per row: expect [{M}, {N//32}] = [64, 4]")
print(f"    Got: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  If columnwise = scales per column: expect [{M//32}, {N}] = [2, 128]")
print(f"    Got: {a_mxfp8._columnwise_scale_inv.shape}")
