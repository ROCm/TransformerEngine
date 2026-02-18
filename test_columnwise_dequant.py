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
print("Verify columnwise dequantization")
print("=" * 60)

# Built-in dequantize (probably uses rowwise)
b_dequant_builtin = b_mxfp8.dequantize()

# Manual dequantization of columnwise
# For columnwise: data [K, N], scales [K//32, N]
# Each column has blocks of 32 elements, each block scaled by one scale
b_columnwise_data = b_mxfp8._columnwise_data
b_columnwise_scale = b_mxfp8._columnwise_scale_inv

# Convert to FP8 then FP32
b_fp8_as_fp32 = b_columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)

# Apply scales
# For each column j, for each K-block i:
#   elements [i*32:(i+1)*32, j] are scaled by scale[i, j]
b_dequant_manual = torch.zeros_like(b_fp32)
VEC_SIZE = 32

for j in range(N):
    for i in range(K // VEC_SIZE):
        # Get scale for this block
        scale_e8m0 = b_columnwise_scale[i, j].item()
        scale = 2.0 ** (scale_e8m0 - 127.0)
        # Apply to elements in this block
        b_dequant_manual[i*VEC_SIZE:(i+1)*VEC_SIZE, j] = b_fp8_as_fp32[i*VEC_SIZE:(i+1)*VEC_SIZE, j] * scale

# Compare
print(f"\nOriginal: {b_fp32.shape}")
print(f"Built-in dequant: {b_dequant_builtin.shape}")
print(f"Manual columnwise dequant: {b_dequant_manual.shape}")

# Check if manual matches original
diff_manual = torch.max(torch.abs(b_dequant_manual - b_fp32)).item()
diff_builtin = torch.max(torch.abs(b_dequant_builtin - b_fp32)).item()

print(f"\nMax diff from original:")
print(f"  Built-in: {diff_builtin:.4f}")
print(f"  Manual columnwise: {diff_manual:.4f}")

# The manual should be close to original if columnwise scales match columnwise data
if diff_manual < 1.0:
    print(f"\n✓ Columnwise data and scales match correctly")
else:
    print(f"\n✗ Columnwise data and scales DO NOT match!")

# Check first element
i, j = 0, 0
print(f"\nFirst element [0, 0]:")
print(f"  Original: {b_fp32[i, j].item():.6f}")
print(f"  FP8 value: {b_fp8_as_fp32[i, j].item():.6f}")
print(f"  Scale (E8M0={b_columnwise_scale[0, 0].item()}): {2.0 ** (b_columnwise_scale[0, 0].item() - 127.0):.6f}")
print(f"  Reconstructed: {b_dequant_manual[i, j].item():.6f}")
