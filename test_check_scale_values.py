import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, K = 64, 64
a = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a)

print(f"Input: {a.shape}")
print(f"Data: {a_mxfp8._rowwise_data.shape}")
print(f"Scale: {a_mxfp8._rowwise_scale_inv.shape}")

# Check if padded regions are zero or have meaningful values
scale = a_mxfp8._rowwise_scale_inv
print(f"\nScale tensor stats:")
print(f"  Full scale: min={scale.min().item()}, max={scale.max().item()}, mean={scale.float().mean().item():.2f}")
print(f"  First [64, 2] (expected active region):")
print(f"    min={scale[:64, :2].min().item()}, max={scale[:64, :2].max().item()}, mean={scale[:64, :2].float().mean().item():.2f}")
print(f"  Padded rows [64:128, :]:")
print(f"    min={scale[64:, :].min().item()}, max={scale[64:, :].max().item()}, mean={scale[64:, :].float().mean().item():.2f}")
print(f"  Padded cols [:, 2:4]:")
print(f"    min={scale[:, 2:].min().item()}, max={scale[:, 2:].max().item()}, mean={scale[:, 2:].float().mean().item():.2f}")

# Check if scale values in padded region are 127 (neutral E8M0)
neutral_e8m0 = 127
print(f"\nCheck for neutral E8M0 (127) in padded regions:")
print(f"  Padded rows contain 127: {(scale[64:, :] == neutral_e8m0).all().item()}")
print(f"  Padded cols contain 127: {(scale[:, 2:] == neutral_e8m0).all().item()}")
