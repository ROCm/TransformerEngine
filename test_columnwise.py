import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)

# Create a [32, 512] tensor
tensor = torch.randn((32, 512), dtype=torch.bfloat16, device=device)
mxfp8 = quantizer.quantize(tensor)

print(f"Original tensor shape: {tensor.shape}")
print(f"\nRowwise:")
print(f"  data: {mxfp8._rowwise_data.shape}, stride: {mxfp8._rowwise_data.stride()}")
print(f"  scale: {mxfp8._rowwise_scale_inv.shape}, stride: {mxfp8._rowwise_scale_inv.stride()}")

if mxfp8._columnwise_data is not None:
    print(f"\nColumnwise:")
    print(f"  data: {mxfp8._columnwise_data.shape}, stride: {mxfp8._columnwise_data.stride()}")
    print(f"  scale: {mxfp8._columnwise_scale_inv.shape}, stride: {mxfp8._columnwise_scale_inv.stride()}")

    # Check if columnwise is actually transposed
    print(f"\nIs columnwise data just a transpose?")
    print(f"  Rowwise data ptr: {mxfp8._rowwise_data.data_ptr()}")
    print(f"  Columnwise data ptr: {mxfp8._columnwise_data.data_ptr()}")
    print(f"  Same storage: {mxfp8._rowwise_data.data_ptr() == mxfp8._columnwise_data.data_ptr()}")

# Now test with .T
print(f"\nAfter .T on rowwise:")
data_t = mxfp8._rowwise_data.T
scale_t = mxfp8._rowwise_scale_inv.T
print(f"  data.T: {data_t.shape}, stride: {data_t.stride()}")
print(f"  scale.T: {scale_t.shape}, stride: {scale_t.stride()}")

print(f"\nExpected for transposed [32, 512] -> [512, 32]:")
print(f"  data: [512, 32]")
print(f"  scale: [512, 32//32] = [512, 1]")
