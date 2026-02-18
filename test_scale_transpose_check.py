import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N, K = 128, 128, 128

# Create B with shape [K, N]
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("B matrix quantization")
print("=" * 60)

print(f"\nB data shape: {b_mxfp8._rowwise_data.shape}")  # Should be [K, N] = [128, 128]
print(f"B scale shape: {b_mxfp8._rowwise_scale_inv.shape}")  # Is [K, N//32] = [128, 4]

print(f"\nFor kernel:")
print(f"  B data is [K, N] = [{K}, {N}]")
print(f"  Kernel expects b_scale shape: [K//32, N] = [{K//32}, {N}] = [4, 128]")
print(f"  But we have b_scale shape: [K, N//32] = [{K}, {N//32}] = [128, 4]")
print(f"")
print(f"  ⚠ Scale shape is TRANSPOSED!")
print(f"  We need to transpose B's scale from [128, 4] to [4, 128]")

# Check if transposing fixes it
b_scale_transposed = b_mxfp8._rowwise_scale_inv.T
print(f"\nAfter transpose:")
print(f"  B scale shape: {b_scale_transposed.shape}")
print(f"  Matches expected [4, 128]: {b_scale_transposed.shape == (K//32, N)}")
