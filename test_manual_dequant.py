import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, K = 128, 128

# Simple test: one value per block
a_fp32 = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
for i in range(K // 32):
    a_fp32[:, i*32:(i+1)*32] = float(i + 1)

print("Original data (first row):")
for i in range(4):
    print(f"  Block {i}: {a_fp32[0, i*32].item()}")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)

print(f"\nQuantized data (uint8, first row):")
for i in range(4):
    print(f"  Block {i}: {a_mxfp8._rowwise_data[0, i*32].item()}")

print(f"\nScales (E8M0, first row):")
print(f"  {a_mxfp8._rowwise_scale_inv[0, :]}")

# Manual dequantization
print(f"\nManual dequantization check:")

# Get FP8 dtype
major, minor = torch.cuda.get_device_capability()
fp8_dtype = torch.float8_e4m3fn if (major == 9 and minor >= 5) else torch.float8_e4m3fnuz

# Convert uint8 to FP8
fp8_data = a_mxfp8._rowwise_data.view(fp8_dtype)

# Convert to FP32 to see values
fp8_as_fp32 = fp8_data.to(torch.float32)

print(f"  FP8 data as FP32 (first row):")
for i in range(4):
    print(f"    Block {i}: {fp8_as_fp32[0, i*32].item()}")

# Apply scales manually
# For E8M0: scale = 2^(e8m0 - 127)
# But the tensor is named "_scale_inv", so it might be inverse
# Let's try both interpretations

scales_e8m0 = a_mxfp8._rowwise_scale_inv[0, :]
print(f"\n  Scales for each block:")
for i in range(4):
    e8m0 = scales_e8m0[i].item()
    forward_scale = 2.0 ** (e8m0 - 127)
    inverse_scale = 2.0 ** (127 - e8m0)
    print(f"    Block {i}: E8M0={e8m0}, forward={forward_scale:.6f}, inverse={inverse_scale:.6f}")

# Manual dequant with forward scale
manual_dequant_forward = torch.zeros_like(a_fp32)
for i in range(K // 32):
    e8m0 = scales_e8m0[i].item()
    scale = 2.0 ** (e8m0 - 127)
    manual_dequant_forward[:, i*32:(i+1)*32] = fp8_as_fp32[:, i*32:(i+1)*32] * scale

# Built-in dequant
auto_dequant = a_mxfp8.dequantize()

print(f"\n  Comparison (first row):")
for i in range(4):
    print(f"    Block {i}:")
    print(f"      Original:  {a_fp32[0, i*32].item():.6f}")
    print(f"      Manual:    {manual_dequant_forward[0, i*32].item():.6f}")
    print(f"      Auto:      {auto_dequant[0, i*32].item():.6f}")
    print(f"      Match: {'✓' if abs(manual_dequant_forward[0, i*32] - auto_dequant[0, i*32]) < 0.01 else '✗'}")
