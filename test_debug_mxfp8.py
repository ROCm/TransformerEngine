import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
import transformer_engine_torch as tex

# Test what happens when we transpose MXFP8 data and scales
device = torch.device("cuda")

M, K = 128, 512

# Create and quantize
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
a_mxfp8 = quantizer.quantize(a_fp32)

print(f"Original rowwise data: {a_mxfp8._rowwise_data.shape}, stride: {a_mxfp8._rowwise_data.stride()}")
print(f"Original rowwise scale: {a_mxfp8._rowwise_scale_inv.shape}, stride: {a_mxfp8._rowwise_scale_inv.stride()}")

# Test what .T does
data_t = a_mxfp8._rowwise_data.T
scale_t = a_mxfp8._rowwise_scale_inv.T

print(f"\nAfter .T:")
print(f"Transposed data: {data_t.shape}, stride: {data_t.stride()}, is_contiguous: {data_t.is_contiguous()}")
print(f"Transposed scale: {scale_t.shape}, stride: {scale_t.stride()}, is_contiguous: {scale_t.is_contiguous()}")

# Test flattening
A_flat = a_mxfp8._rowwise_data.reshape(-1, a_mxfp8._rowwise_data.shape[-1])
print(f"\nAfter reshape(-1, last_dim):")
print(f"Flattened data: {A_flat.shape}, stride: {A_flat.stride()}")

# Test .T on flattened
A_flat_t = A_flat.T
print(f"\nAfter .T on flattened:")
print(f"Transposed flattened data: {A_flat_t.shape}, stride: {A_flat_t.stride()}")

# Now test with a tensor that has batch dimensions
print("\n" + "="*60)
print("Testing with batch dimensions:")
a_batch_fp32 = torch.randn((2, M, K), dtype=torch.bfloat16, device=device)
# Note: This will fail because batch dim (2) is not divisible by 32
try:
    a_batch_mxfp8 = quantizer.quantize(a_batch_fp32)
    print(f"Batch MXFP8 data: {a_batch_mxfp8._rowwise_data.shape}")
    print(f"Batch MXFP8 scale: {a_batch_mxfp8._rowwise_scale_inv.shape}")
except AssertionError as e:
    print(f"Cannot quantize with batch dim 2: {e}")

# Try with batch dim divisible by 32
a_batch_fp32 = torch.randn((32, M, K), dtype=torch.bfloat16, device=device)
a_batch_mxfp8 = quantizer.quantize(a_batch_fp32)
print(f"\nBatch=32 MXFP8 data: {a_batch_mxfp8._rowwise_data.shape}")
print(f"Batch=32 MXFP8 scale: {a_batch_mxfp8._rowwise_scale_inv.shape}")

# Test reshape and transpose
data_flat = a_batch_mxfp8._rowwise_data.reshape(-1, a_batch_mxfp8._rowwise_data.shape[-1])
scale_flat = a_batch_mxfp8._rowwise_scale_inv.reshape(-1, a_batch_mxfp8._rowwise_scale_inv.shape[-1])
print(f"\nFlattened batch data: {data_flat.shape}")
print(f"Flattened batch scale: {scale_flat.shape}")

# Check if scale reshaping could cause out-of-bounds access
print(f"\nScale flattening check:")
print(f"  Original scale shape: {a_batch_mxfp8._rowwise_scale_inv.shape}")
print(f"  Original scale numel: {a_batch_mxfp8._rowwise_scale_inv.numel()}")
print(f"  Flattened scale shape: {scale_flat.shape}")
print(f"  Flattened scale numel: {scale_flat.numel()}")
print(f"  Shapes match: {a_batch_mxfp8._rowwise_scale_inv.numel() == scale_flat.numel()}")
