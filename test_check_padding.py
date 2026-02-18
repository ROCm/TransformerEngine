import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Test different sizes
for size in [(64, 64), (128, 512), (256, 256)]:
    M, K = size
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )

    a_mxfp8 = quantizer.quantize(a)

    expected_scale_shape_0 = M
    expected_scale_shape_1 = K // 32

    print(f"\nInput shape: {a.shape}")
    print(f"  Data shape: {a_mxfp8._rowwise_data.shape}")
    print(f"  Scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
    print(f"  Expected scale: [{expected_scale_shape_0}, {expected_scale_shape_1}]")

    if a_mxfp8._rowwise_data.shape != a.shape:
        print(f"  ⚠ Data is padded!")
    if a_mxfp8._rowwise_scale_inv.shape != torch.Size([expected_scale_shape_0, expected_scale_shape_1]):
        print(f"  ⚠ Scale shape mismatch!")
