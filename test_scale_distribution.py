import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

# Create random data with different characteristics
print("=" * 60)
print("Scale distribution analysis")
print("=" * 60)

for seed, scale_factor in [(42, 0.1), (123, 1.0), (456, 10.0)]:
    torch.manual_seed(seed)
    M, K = 128, 512
    a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * scale_factor

    a_mxfp8 = quantizer.quantize(a_fp32)

    scales = a_mxfp8._rowwise_scale_inv

    print(f"\nSeed {seed}, scale_factor {scale_factor}:")
    print(f"  Data range: [{a_fp32.min().item():.4f}, {a_fp32.max().item():.4f}]")
    print(f"  Scale shape: {scales.shape}")
    print(f"  Scale range: [{scales.min().item()}, {scales.max().item()}]")
    print(f"  Scale unique values: {len(scales.unique())} out of {scales.numel()}")
    print(f"  Scale mean: {scales.float().mean().item():.2f}")
    print(f"  Scale std: {scales.float().std().item():.2f}")

    # Check if scales are mostly the same
    if len(scales.unique()) < 10:
        print(f"  ⚠ Very few unique scales! {scales.unique()}")

    # Compare with dequantized
    a_dequant = a_mxfp8.dequantize()
    quant_error = torch.max(torch.abs(a_fp32 - a_dequant)).item()
    rel_error = quant_error / (torch.max(torch.abs(a_fp32)).item() + 1e-6)
    print(f"  Quantization error: max={quant_error:.4f}, rel={rel_error:.4f}")
