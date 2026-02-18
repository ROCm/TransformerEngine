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

print("=" * 60)
print("Test: Variation WITHIN 32-element blocks")
print("=" * 60)

# Test 1: Each block has uniform values (different across blocks)
print("\n1. Uniform within blocks, different across blocks:")
a1_fp32 = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
for block_idx in range(K // 32):
    start = block_idx * 32
    end = start + 32
    a1_fp32[:, start:end] = float(block_idx + 1)
b1_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

a1_mxfp8 = quantizer.quantize(a1_fp32)
b1_mxfp8 = quantizer.quantize(b1_fp32)

ref1 = torch.matmul(a1_mxfp8.dequantize(), b1_mxfp8.dequantize())
out1 = te_generic_gemm_triton(
    A=a1_mxfp8, transa=False, B=b1_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

diff1 = abs(out1[0, 0] - ref1[0, 0]).item()
print(f"  Ref: {ref1[0, 0].item():.2f}, Out: {out1[0, 0].item():.2f}, Diff: {diff1:.4f}")

# Test 2: Variation within each block
print("\n2. Random values within each block:")
torch.manual_seed(42)
a2_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
# Normalize each block to have same magnitude
for block_idx in range(K // 32):
    start = block_idx * 32
    end = start + 32
    block_data = a2_fp32[:, start:end]
    # Normalize to mean=0, std=1 within block
    a2_fp32[:, start:end] = (block_data - block_data.mean()) / (block_data.std() + 1e-6)

b2_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

a2_mxfp8 = quantizer.quantize(a2_fp32)
b2_mxfp8 = quantizer.quantize(b2_fp32)

ref2 = torch.matmul(a2_mxfp8.dequantize(), b2_mxfp8.dequantize())
out2 = te_generic_gemm_triton(
    A=a2_mxfp8, transa=False, B=b2_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

diff2 = abs(out2[0, 0] - ref2[0, 0]).item()
print(f"  Ref: {ref2[0, 0].item():.4f}, Out: {out2[0, 0].item():.4f}, Diff: {diff2:.4f}")

# Test 3: Fully random (no normalization)
print("\n3. Fully random (no block structure):")
torch.manual_seed(123)
a3_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b3_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

a3_mxfp8 = quantizer.quantize(a3_fp32)
b3_mxfp8 = quantizer.quantize(b3_fp32)

ref3 = torch.matmul(a3_mxfp8.dequantize(), b3_mxfp8.dequantize())
out3 = te_generic_gemm_triton(
    A=a3_mxfp8, transa=False, B=b3_mxfp8, transb=False, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

diff3 = torch.max(torch.abs(out3 - ref3)).item()
rel3 = diff3 / (torch.max(torch.abs(ref3)).item() + 1e-6)
print(f"  Max diff: {diff3:.4f}, Rel error: {rel3:.4f}")

# Check quantization error for test 3
print(f"\n4. Quantization quality check for fully random:")
a3_quant_error = torch.max(torch.abs(a3_fp32 - a3_mxfp8.dequantize())).item()
b3_quant_error = torch.max(torch.abs(b3_fp32 - b3_mxfp8.dequantize())).item()
print(f"  A quantization error: {a3_quant_error:.4f}")
print(f"  B quantization error: {b3_quant_error:.4f}")
print(f"  A scale values (first row): {a3_mxfp8._rowwise_scale_inv[0, :]}")
print(f"  B scale values (first row): {b3_mxfp8._rowwise_scale_inv[0, :]}")
