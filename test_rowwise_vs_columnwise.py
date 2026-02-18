import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

K, N = 512, 256

torch.manual_seed(42)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Comparing rowwise vs columnwise quantization")
print("=" * 60)

# Get rowwise and columnwise data
b_rowwise_data = b_mxfp8._rowwise_data
b_columnwise_data = b_mxfp8._columnwise_data

print(f"\nOriginal: {b_fp32.shape}")
print(f"Rowwise data: {b_rowwise_data.shape}")
print(f"Columnwise data: {b_columnwise_data.shape}")

# Check if data is the same
same_elements = (b_rowwise_data == b_columnwise_data).sum().item()
total = b_rowwise_data.numel()
print(f"\nSame FP8 values: {same_elements}/{total} ({100*same_elements/total:.1f}%)")

# Dequantize each separately
# For rowwise: need to apply rowwise scales
# For columnwise: need to apply columnwise scales

# Check what the built-in dequantize() returns
b_dequant = b_mxfp8.dequantize()

print(f"\nBuilt-in dequantize() shape: {b_dequant.shape}")
print(f"Matches original: {torch.allclose(b_dequant, b_fp32, rtol=0.1)}")

max_diff = torch.max(torch.abs(b_dequant - b_fp32)).item()
print(f"Max diff from original: {max_diff:.4f}")

# The issue might be that columnwise data + columnwise scales should give
# the same dequantized result as rowwise data + rowwise scales
# Let's verify the quantization error for both modes

# Check a specific element
i, j = 0, 0
print(f"\nElement [0, 0]:")
print(f"  Original: {b_fp32[i, j].item():.6f}")
print(f"  Dequantized: {b_dequant[i, j].item():.6f}")
print(f"  Rowwise FP8 (uint8): {b_rowwise_data[i, j].item()}")
print(f"  Columnwise FP8 (uint8): {b_columnwise_data[i, j].item()}")
