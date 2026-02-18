import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Create test data
M, K = 128, 128
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device) * 2.0  # All 2.0

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)
a_dequant = a_mxfp8.dequantize()

print(f"Original value: 2.0")
print(f"Dequantized: {a_dequant[0, 0].item()}")
print(f"")
print(f"FP8 data (uint8): {a_mxfp8._rowwise_data[0, 0].item()}")
print(f"Scale E8M0: {a_mxfp8._rowwise_scale_inv[0, 0].item()}")
print(f"")

# The name is "_scale_inv" which suggests it's the INVERSE
# Let's check both interpretations:

scale_e8m0 = a_mxfp8._rowwise_scale_inv[0, 0].item()

# Interpretation 1: It's the forward scale (despite the name)
forward_scale = 2.0 ** (scale_e8m0 - 127)
print(f"If '_scale_inv' is forward scale:")
print(f"  scale = 2^({scale_e8m0} - 127) = {forward_scale}")

# Interpretation 2: It's the inverse scale (as the name suggests)
inverse_scale = 2.0 ** (127 - scale_e8m0)
print(f"If '_scale_inv' is inverse scale:")
print(f"  scale_inv = 2^(127 - {scale_e8m0}) = {inverse_scale}")
print(f"  forward_scale = 1/scale_inv = {1/inverse_scale}")

print(f"\nTo get from FP8 to dequantized value:")
# FP8 value 120 represents what in actual FP8?
# We need to know this to verify the formula

# Actually, let's just test empirically
# If dequant = fp8_data * scale, then scale = dequant / fp8_data
# But fp8_data is in FP8 format, need to convert it first

# Let's use the fact that dequantize() works correctly
# and reverse engineer what the scale must be

# Convert FP8 data to float to see what value it represents
fp8_as_float = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn if torch.cuda.get_device_capability()[1] >= 5 else torch.float8_e4m3fnuz).to(torch.float32)
print(f"\nFP8 data interpreted as float: {fp8_as_float[0, 0].item()}")
print(f"Dequantized value: {a_dequant[0, 0].item()}")
print(f"Implied scale: {a_dequant[0, 0].item() / fp8_as_float[0, 0].item()}")

# Now check which interpretation matches
print(f"\nWhich matches?")
print(f"  Forward scale ({forward_scale}): {'✓' if abs(forward_scale - (a_dequant[0, 0].item() / fp8_as_float[0, 0].item())) < 0.001 else '✗'}")
print(f"  Inverse scale (1/{inverse_scale} = {1/inverse_scale}): {'✓' if abs(1/inverse_scale - (a_dequant[0, 0].item() / fp8_as_float[0, 0].item())) < 0.001 else '✗'}")
