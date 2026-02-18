import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

M, N, K = 128, 128, 128

torch.manual_seed(42)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

# Quantize
a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Dequantize
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()

print("=" * 60)
print("Error propagation analysis")
print("=" * 60)

# 1. FP32 reference
ref_fp32 = torch.matmul(a_fp32, b_fp32)
print(f"\n1. FP32 reference:")
print(f"  Output range: [{ref_fp32.min().item():.2f}, {ref_fp32.max().item():.2f}]")
print(f"  Sample: {ref_fp32[0, 0].item():.4f}")

# 2. Dequantized reference (what we compare against)
ref_dequant = torch.matmul(a_dequant, b_dequant)
print(f"\n2. Dequantized reference:")
print(f"  Output range: [{ref_dequant.min().item():.2f}, {ref_dequant.max().item():.2f}]")
print(f"  Sample: {ref_dequant[0, 0].item():.4f}")
diff_vs_fp32 = torch.max(torch.abs(ref_dequant - ref_fp32)).item()
print(f"  Diff vs FP32: {diff_vs_fp32:.4f} ({diff_vs_fp32 / (torch.max(torch.abs(ref_fp32)).item() + 1e-6) * 100:.2f}%)")

# 3. MXFP8 kernel output
out_kernel = te_generic_gemm_triton(
    A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

print(f"\n3. MXFP8 kernel output:")
print(f"  Output range: [{out_kernel.min().item():.2f}, {out_kernel.max().item():.2f}]")
print(f"  Sample: {out_kernel[0, 0].item():.4f}")
diff_vs_dequant = torch.max(torch.abs(out_kernel - ref_dequant)).item()
print(f"  Diff vs dequantized: {diff_vs_dequant:.4f} ({diff_vs_dequant / (torch.max(torch.abs(ref_dequant)).item() + 1e-6) * 100:.2f}%)")
diff_vs_fp32_kernel = torch.max(torch.abs(out_kernel - ref_fp32)).item()
print(f"  Diff vs FP32: {diff_vs_fp32_kernel:.4f} ({diff_vs_fp32_kernel / (torch.max(torch.abs(ref_fp32)).item() + 1e-6) * 100:.2f}%)")

# 4. Check individual elements
print(f"\n4. Element-wise comparison (first 5 elements of row 0):")
print(f"  FP32:      {ref_fp32[0, :5]}")
print(f"  Dequant:   {ref_dequant[0, :5]}")
print(f"  Kernel:    {out_kernel[0, :5]}")

# 5. Check if kernel matches dequant exactly
matches = (out_kernel == ref_dequant).sum().item()
total = out_kernel.numel()
print(f"\n5. Exact matches: {matches} / {total} ({matches/total*100:.1f}%)")

if matches < total:
    # Find where they differ
    diff_mask = (out_kernel != ref_dequant)
    diff_indices = diff_mask.nonzero(as_tuple=False)
    print(f"  First mismatch at {diff_indices[0].tolist()}: kernel={out_kernel[diff_indices[0][0], diff_indices[0][1]].item():.4f}, ref={ref_dequant[diff_indices[0][0], diff_indices[0][1]].item():.4f}")
