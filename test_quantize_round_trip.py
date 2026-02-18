import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Simple test
M, K = 128, 512
a_original = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

# Quantize
a_mxfp8 = quantizer.quantize(a_original)

# Dequantize
a_reconstructed = a_mxfp8.dequantize()

# Compare
max_diff = torch.max(torch.abs(a_original - a_reconstructed)).item()
max_val = torch.max(torch.abs(a_original)).item()
rel_error = max_diff / (max_val + 1e-6)

print(f"Quantize-dequantize round trip:")
print(f"  Input range: [{a_original.min().item():.4f}, {a_original.max().item():.4f}]")
print(f"  Reconstructed range: [{a_reconstructed.min().item():.4f}, {a_reconstructed.max().item():.4f}]")
print(f"  Max diff: {max_diff:.6f}")
print(f"  Rel error: {rel_error:.6f}")
print(f"  First few elements:")
print(f"    Original:      {a_original[0, :10]}")
print(f"    Reconstructed: {a_reconstructed[0, :10]}")

# This should be very accurate for MXFP8
if rel_error > 0.01:  # 1% tolerance
    print(f"\n  ⚠ Large quantization error!")
else:
    print(f"\n  ✓ Quantization is accurate")
