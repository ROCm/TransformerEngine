import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

print("Testing scale shapes for different matrix dimensions")
print("=" * 60)

test_cases = [
    (64, 128, "Small"),
    (128, 256, "A in gemm"),
    (512, 256, "B in gemm"),
]

for rows, cols, desc in test_cases:
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device=device)
    x_mxfp8 = quantizer.quantize(x)

    print(f"\n{desc}: [{rows}, {cols}]")
    print(f"  Rowwise scale: {x_mxfp8._rowwise_scale_inv.shape}")
    print(f"  Columnwise scale: {x_mxfp8._columnwise_scale_inv.shape}")

    # Try to infer the pattern
    rw_shape = x_mxfp8._rowwise_scale_inv.shape
    cw_shape = x_mxfp8._columnwise_scale_inv.shape

    print(f"  Pattern analysis:")
    print(f"    Rowwise: [{rw_shape[0]}, {rw_shape[1]}]")
    print(f"      = [{rows}, {cols//32}]? {rw_shape == (rows, cols//32)}")
    print(f"      = [{cols}, {rows//32}]? {rw_shape == (cols, rows//32)}")
    print(f"    Columnwise: [{cw_shape[0]}, {cw_shape[1]}]")
    print(f"      = [{rows//32}, {cols}]? {cw_shape == (rows//32, cols)}")
    print(f"      = [{cols//32}, {rows}]? {cw_shape == (cols//32, rows)}")

print("\n" + "=" * 60)
print("Conclusion:")
print("  Rowwise scale: [rows, cols//32] (scales along row direction)")
print("  Columnwise scale: [rows//32, cols] (scales along column direction)")
print("  Both with padding to multiples of [128, ?]")
