import torch
import os
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")
os.environ["DEBUG_MXFP8_GEMM"] = "1"

# Simple case that should work: small matrices, no padding
M, N, K = 128, 128, 128

# Create simple data where we know the expected result
# A = all 1.0, B = all 1.0, so C = A @ B should be all K=128
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Test: A=ones[128,128] @ B=ones[128,128]")
print("Expected output: all 128.0 (sum of 128 ones)")
print("=" * 60)

# Check quantization
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()
print(f"\nAfter quantize-dequantize:")
print(f"  A: {a_dequant[0, :5]}")
print(f"  B: {b_dequant[0, :5]}")

# Reference
ref = torch.matmul(a_dequant, b_dequant)
print(f"\nReference output (dequantized matmul):")
print(f"  First row: {ref[0, :5]}")
print(f"  Expected: all ~{K}")

# MXFP8 GEMM
output = te_generic_gemm_triton(
    A=a_mxfp8,
    transa=False,
    B=b_mxfp8,
    transb=False,
    D=None,
    quantizer=None,
    output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(),
    bias_type=tex.DType.kBFloat16,
    gelu=False,
    gelu_in=torch.Tensor(),
    grad=False,
    workspace=torch.Tensor(),
    workspaceSize=0,
    accumulate=False,
    use_split_accumulator=False,
    comm_overlap=False,
    comm_type=0,
    extra_output=torch.Tensor(),
    bulk_overlap=False,
)

print(f"\nMXFP8 GEMM output:")
print(f"  First row: {output[0][0, :5]}")
print(f"  Range: [{output[0].min().item():.2f}, {output[0].max().item():.2f}]")

print(f"\nComparison:")
print(f"  Max diff: {torch.max(torch.abs(output[0] - ref)).item():.4f}")
print(f"  First element: kernel={output[0][0,0].item():.4f}, ref={ref[0,0].item():.4f}")
