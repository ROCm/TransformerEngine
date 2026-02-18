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

print("Testing different patterns:")
print("=" * 60)

# Test 1: All ones (we know this works)
print("\n1. All ones:")
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
output = te_generic_gemm_triton(
    A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

print(f"  Expected: {ref[0, 0].item()}, Got: {output[0, 0].item()}, Diff: {abs(ref[0, 0] - output[0, 0]).item()}")

# Test 2: All twos
print("\n2. All twos:")
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device) * 2
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device) * 2

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
output = te_generic_gemm_triton(
    A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

print(f"  Expected: {ref[0, 0].item()}, Got: {output[0, 0].item()}, Diff: {abs(ref[0, 0] - output[0, 0]).item()}")

# Test 3: First row random, rest ones
print("\n3. First row random, rest all ones:")
a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device)
a_fp32[0, :] = torch.randn(K, dtype=torch.bfloat16, device=device)
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
output = te_generic_gemm_triton(
    A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

print(f"  Row 0 expected: {ref[0, 0].item():.4f}, got: {output[0, 0].item():.4f}, diff: {abs(ref[0, 0] - output[0, 0]).item():.4f}")
print(f"  Row 1 expected: {ref[1, 0].item():.4f}, got: {output[1, 0].item():.4f}, diff: {abs(ref[1, 0] - output[1, 0]).item():.4f}")

# Test 4: fully random
print("\n4. Fully random:")
torch.manual_seed(42)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
output = te_generic_gemm_triton(
    A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

max_diff = torch.max(torch.abs(output - ref)).item()
rel_err = max_diff / (torch.max(torch.abs(ref)).item() + 1e-6)
print(f"  Max diff: {max_diff:.4f}, Rel error: {rel_err:.4f}")
print(f"  Sample: ref[0,0]={ref[0,0].item():.4f}, out[0,0]={output[0,0].item():.4f}")
