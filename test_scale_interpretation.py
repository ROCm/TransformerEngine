import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Create a simple test case with known values
M, K = 128, 128  # Use 128 to avoid padding
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device) * 0.5  # All 0.5

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)

print("Original values: all 0.5")
print(f"Data shape: {a_mxfp8._rowwise_data.shape}")
print(f"Scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"\nFirst few FP8 data values (uint8): {a_mxfp8._rowwise_data[0, :10]}")
print(f"First few scale values (E8M0): {a_mxfp8._rowwise_scale_inv[0, :4]}")

# Dequantize to check
a_dequant = a_mxfp8.dequantize()
print(f"\nDequantized first few values: {a_dequant[0, :10]}")
print(f"Expected: all ~0.5")

# Manual dequantization to understand the formula
# E8M0: scale = 2^(biased_exp - 127)
# But the tensor stores "scale_inv", so maybe it's 1/scale?
scale_e8m0 = a_mxfp8._rowwise_scale_inv[0, 0].item()
print(f"\nFirst scale E8M0 value: {scale_e8m0}")
print(f"  If forward scale: 2^({scale_e8m0} - 127) = 2^{scale_e8m0 - 127} = {2**(scale_e8m0 - 127)}")
print(f"  If inverse scale: 2^(127 - {scale_e8m0}) = 2^{127 - scale_e8m0} = {2**(127 - scale_e8m0)}")

# Check the naming - is it really "inverse"?
data_uint8 = a_mxfp8._rowwise_data[0, 0].item()
dequant_value = a_dequant[0, 0].item()
print(f"\nFirst data point:")
print(f"  FP8 (as uint8): {data_uint8}")
print(f"  Dequantized: {dequant_value}")
print(f"  Original: 0.5")

# Try to figure out the formula
# If data_fp8 * scale = dequant, then scale = dequant / data_fp8
# But we need to interpret data_fp8 as FP8 first...

# Actually, let's check if the scale is per-block
print(f"\nChecking block structure:")
print(f"  Block size: 32")
print(f"  Data values [0:32]: {a_mxfp8._rowwise_data[0, :32].unique()}")
print(f"  Data values [32:64]: {a_mxfp8._rowwise_data[0, 32:64].unique()}")
print(f"  Scale for block 0: {a_mxfp8._rowwise_scale_inv[0, 0].item()}")
print(f"  Scale for block 1: {a_mxfp8._rowwise_scale_inv[0, 1].item()}")
