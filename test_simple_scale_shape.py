import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Use simple dimensions
M, N = 64, 128

a_fp32 = torch.randn((M, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)

print(f"Matrix shape: [{M}, {N}]")
print(f"VEC_SIZE: 32")
print()
print(f"Rowwise scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  Expected for rowwise ([M, N//32]): [{M}, {N//32}] = [{M}, {128//32}] = [64, 4]")
print()
print(f"Columnwise scale: {a_mxfp8._columnwise_scale_inv.shape}")
print(f"  Expected for columnwise ([M//32, N]): [{M//32}, {N}] = [{64//32}, {128}] = [2, 128]")
