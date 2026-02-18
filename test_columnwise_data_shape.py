import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

# Test B from gemm: [K, N] = [512, 256]
K, N = 512, 256
b = torch.randn((K, N), dtype=torch.bfloat16, device=device)
b_mxfp8 = quantizer.quantize(b)

print(f"B shape: [{K}, {N}]")
print(f"\nRowwise:")
print(f"  Data: {b_mxfp8._rowwise_data.shape if b_mxfp8._rowwise_data is not None else None}")
print(f"  Scale: {b_mxfp8._rowwise_scale_inv.shape if b_mxfp8._rowwise_scale_inv is not None else None}")

print(f"\nColumnwise:")
print(f"  Data: {b_mxfp8._columnwise_data.shape if b_mxfp8._columnwise_data is not None else None}")
print(f"  Scale: {b_mxfp8._columnwise_scale_inv.shape if b_mxfp8._columnwise_scale_inv is not None else None}")

# Check if data is actually the same or transposed
if b_mxfp8._rowwise_data is not None and b_mxfp8._columnwise_data is not None:
    same_shape = b_mxfp8._rowwise_data.shape == b_mxfp8._columnwise_data.shape
    print(f"\nData shapes are same: {same_shape}")

    if same_shape:
        # Check if content is different
        diff_count = (b_mxfp8._rowwise_data != b_mxfp8._columnwise_data).sum().item()
        print(f"Different elements: {diff_count}/{b_mxfp8._rowwise_data.numel()}")
