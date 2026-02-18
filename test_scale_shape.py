import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)

# Test different shapes
shapes = [
    (128, 512),   # weight
    (32, 512),    # input (batch=32)
    (512, 256),   # another test
]

for shape in shapes:
    tensor = torch.randn(shape, dtype=torch.bfloat16, device=device)
    mxfp8 = quantizer.quantize(tensor)

    print(f"\nInput shape: {shape}")
    print(f"  Data shape: {mxfp8._rowwise_data.shape}")
    print(f"  Scale shape: {mxfp8._rowwise_scale_inv.shape}")
    print(f"  Expected scale shape: [{shape[0]}, {shape[1]//32}]")

    # Check if columnwise exists
    if mxfp8._columnwise_data is not None:
        print(f"  Columnwise data: {mxfp8._columnwise_data.shape}")
        print(f"  Columnwise scale: {mxfp8._columnwise_scale_inv.shape}")
