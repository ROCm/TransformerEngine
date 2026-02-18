import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")
VEC_SIZE = 32

# Create B: [K, N] = [512, 256]
K, N = 512, 256
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

b_mxfp8 = quantizer.quantize(b_fp32)

print("="*60)
print("Original B tensor (rowwise)")
print("="*60)
print(f"B data shape: {b_mxfp8._rowwise_data.shape} = [{K}, {N}]")
print(f"B scale shape: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"  Expected scale shape: [{K}, {N//VEC_SIZE}] = [{K}, {N//32}] = [{K}, {256//32}] = [{K}, 8]")

print("\n" + "="*60)
print("After transpose: B.T")
print("="*60)
b_data_t = b_mxfp8._rowwise_data.T
print(f"B.T data shape: {b_data_t.shape} = [{N}, {K}]")
print(f"If we naively use original scale: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"  But kernel expects scale shape: [{N}, {K//VEC_SIZE}] = [{N}, {K//32}] = [{256}, {512//32}] = [{256}, 16]")
print(f"  Mismatch! Original is [{K}, {N//VEC_SIZE}], need [{N}, {K//VEC_SIZE}]")

print("\n" + "="*60)
print("Solution: Use columnwise data when transpose is needed")
print("="*60)
print(f"B columnwise data: {b_mxfp8._columnwise_data.shape}")
print(f"B columnwise scale: {b_mxfp8._columnwise_scale_inv.shape}")
print("  Columnwise is ALREADY transposed!")
print("  For B [K, N], columnwise stores it as [N, K]")
print(f"  So columnwise scale is [{N//VEC_SIZE}, {K}]... wait let me check")

# Actually print the columnwise scale shape
if b_mxfp8._columnwise_scale_inv is not None:
    print(f"\n  Actual columnwise scale shape: {b_mxfp8._columnwise_scale_inv.shape}")
    print(f"  Expected for [N, K] data: [{N}, {K//VEC_SIZE}] or [{N//VEC_SIZE}, {K}]?")
