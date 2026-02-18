import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N, K = 128, 128, 256
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

print("Testing simplified approach: dequantize in kernel")
print("=" * 60)

# The C++ logic selects:
# - A with transa=False → columnwise
# - B with transb=False → rowwise

# Get the data and scales
a_data = a_mxfp8._columnwise_data  # [M, K]
a_scale = a_mxfp8._columnwise_scale_inv  # [M//32, K]

b_data = b_mxfp8._rowwise_data  # [K, N]
b_scale = b_mxfp8._rowwise_scale_inv  # [K, N//32]

print(f"A data: {a_data.shape}, scale: {a_scale.shape}")
print(f"B data: {b_data.shape}, scale: {b_scale.shape}")

# Manual dequantization to verify
VEC_SIZE = 32

# A columnwise dequantization
a_dequant = torch.zeros_like(a_fp32)
for j in range(K):
    for i in range(M // VEC_SIZE):
        scale = 2.0 ** (a_scale[i, j].item() - 127.0)
        a_dequant[i*VEC_SIZE:(i+1)*VEC_SIZE, j] = (
            a_data[i*VEC_SIZE:(i+1)*VEC_SIZE, j].view(torch.float8_e4m3fn).to(torch.float32) * scale
        )

# B rowwise dequantization
b_dequant = torch.zeros_like(b_fp32)
for i in range(K):
    for j in range(N // VEC_SIZE):
        scale = 2.0 ** (b_scale[i, j].item() - 127.0)
        b_dequant[i, j*VEC_SIZE:(j+1)*VEC_SIZE] = (
            b_data[i, j*VEC_SIZE:(j+1)*VEC_SIZE].view(torch.float8_e4m3fn).to(torch.float32) * scale
        )

# Reference
ref = torch.matmul(a_dequant, b_dequant)

print(f"\nReference computation:")
print(f"ref[0, 0] = {ref[0, 0].item():.4f}")
print(f"ref shape: {ref.shape}")

# The challenge: tl.dot_scaled expects specific scale layouts
# We have:
# - A columnwise: data[M,K], scales[M//32, K]
# - B rowwise: data[K,N], scales[K, N//32]

# tl.dot_scaled expects:
# - A: scales[M, K//32] (one scale per 32 columns)
# - B: scales[K//32, N] (one scale per 32 rows)

print(f"\ntl.dot_scaled scale mismatch:")
print(f"  A has scales[{a_scale.shape[0]}, {a_scale.shape[1]}] (columnwise)")
print(f"  But needs scales[{M}, {K//32}] (rowwise)")
print(f"  B has scales[{b_scale.shape[0]}, {b_scale.shape[1]}] (rowwise)")
print(f"  But needs scales[{K//32}, {N}] (columnwise)")

print(f"\nConclusion: Cannot directly use tl.dot_scaled with these scale layouts")