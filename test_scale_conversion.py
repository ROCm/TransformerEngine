import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N, K = 128, 128, 128
torch.manual_seed(42)

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("Checking E8M0 scale format")
print("=" * 60)

# Check A rowwise scales
a_scale_e8m0 = a_mxfp8._rowwise_scale_inv
print(f"A rowwise scale shape: {a_scale_e8m0.shape}")
print(f"A scale dtype: {a_scale_e8m0.dtype}")
print(f"A scale sample values (E8M0): {a_scale_e8m0[0, :3]}")

# Convert to actual scales
a_scales_float = 2.0 ** (a_scale_e8m0.to(torch.float32) - 127.0)
print(f"A scale as float: {a_scales_float[0, :3]}")

# Check B columnwise scales
b_scale_e8m0 = b_mxfp8._columnwise_scale_inv
print(f"\nB columnwise scale shape: {b_scale_e8m0.shape}")
print(f"B scale dtype: {b_scale_e8m0.dtype}")
print(f"B scale sample values (E8M0): {b_scale_e8m0[0, :3]}")

# Convert to actual scales
b_scales_float = 2.0 ** (b_scale_e8m0.to(torch.float32) - 127.0)
print(f"B scale as float: {b_scales_float[0, :3]}")

# Check scale ranges
print(f"\nScale statistics:")
print(f"A scale E8M0 range: [{a_scale_e8m0.min().item()}, {a_scale_e8m0.max().item()}]")
print(f"B scale E8M0 range: [{b_scale_e8m0.min().item()}, {b_scale_e8m0.max().item()}]")

# tl.dot_scaled expects E8M0 format where:
# scale = 2^(E8M0 - 127)
# So E8M0=127 means scale=1.0
# E8M0=120 means scale=2^-7 = 0.0078125
# E8M0=134 means scale=2^7 = 128

print(f"\nVerifying E8M0 encoding:")
print(f"E8M0=127 → scale = 2^0 = {2.0**(127-127)}")
print(f"E8M0=120 → scale = 2^-7 = {2.0**(120-127)}")
print(f"E8M0=134 → scale = 2^7 = {2.0**(134-127)}")