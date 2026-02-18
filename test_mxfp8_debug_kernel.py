import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Small test case
M, N, K = 64, 64, 64

# Create simple test data (small values to avoid overflow)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device) * 0.1

print(f"Input ranges:")
print(f"  A: min={a_fp32.min().item():.4f}, max={a_fp32.max().item():.4f}")
print(f"  B: min={b_fp32.min().item():.4f}, max={b_fp32.max().item():.4f}")

# Quantize
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,  # Only rowwise for debugging
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Dequantize to check quantization quality
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()

print(f"\nQuantization error:")
print(f"  A: max_diff={torch.max(torch.abs(a_fp32 - a_dequant)).item():.4f}")
print(f"  B: max_diff={torch.max(torch.abs(b_fp32 - b_dequant)).item():.4f}")

# Compute reference
ref_output = torch.matmul(a_dequant, b_dequant)
print(f"\nReference output range:")
print(f"  min={ref_output.min().item():.4f}, max={ref_output.max().item():.4f}")

# Compute with MXFP8 kernel
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

print(f"\nMXFP8 GEMM output range:")
print(f"  min={output[0].min().item():.4f}, max={output[0].max().item():.4f}")

# Check for NaN/Inf
if torch.isnan(output[0]).any():
    print("  ✗ Output contains NaN!")
if torch.isinf(output[0]).any():
    print("  ✗ Output contains Inf!")

# Compare
max_diff = torch.max(torch.abs(output[0] - ref_output)).item()
print(f"\nComparison:")
print(f"  max_diff={max_diff:.4f}")
print(f"  First few elements:")
print(f"    Kernel: {output[0][0, :5]}")
print(f"    Ref:    {ref_output[0, :5]}")

# Check scale shapes and values
print(f"\nScale information:")
print(f"  A scale shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  A scale range: {a_mxfp8._rowwise_scale_inv.min().item()} - {a_mxfp8._rowwise_scale_inv.max().item()}")
print(f"  B scale shape: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"  B scale range: {b_mxfp8._rowwise_scale_inv.min().item()} - {b_mxfp8._rowwise_scale_inv.max().item()}")
