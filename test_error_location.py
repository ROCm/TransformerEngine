import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

M, N, K = 128, 128, 256
torch.manual_seed(42)

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Kernel output
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

# Reference with A rowwise + B columnwise
VEC_SIZE = 32

a_rowwise_data = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
a_rowwise_scale = a_mxfp8._rowwise_scale_inv
a_dequant = torch.zeros_like(a_fp32)
for i in range(M):
    for j in range(K // VEC_SIZE):
        scale = 2.0 ** (a_rowwise_scale[i, j].item() - 127.0)
        a_dequant[i, j*VEC_SIZE:(j+1)*VEC_SIZE] = a_rowwise_data[i, j*VEC_SIZE:(j+1)*VEC_SIZE] * scale

b_columnwise_data = b_mxfp8._columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)
b_columnwise_scale = b_mxfp8._columnwise_scale_inv
b_dequant = torch.zeros_like(b_fp32)
for j in range(N):
    for i in range(K // VEC_SIZE):
        scale = 2.0 ** (b_columnwise_scale[i, j].item() - 127.0)
        b_dequant[i*VEC_SIZE:(i+1)*VEC_SIZE, j] = b_columnwise_data[i*VEC_SIZE:(i+1)*VEC_SIZE, j] * scale

ref = torch.matmul(a_dequant, b_dequant)

# Find largest errors
diff = torch.abs(output - ref)
max_diff = torch.max(diff).item()

print(f"Max difference: {max_diff:.4f}")

# Find location of max error
max_idx = torch.argmax(diff.flatten())
row = max_idx // N
col = max_idx % N

print(f"\nMax error at position [{row}, {col}]:")
print(f"  Output: {output[row, col].item():.4f}")
print(f"  Ref: {ref[row, col].item():.4f}")
print(f"  Diff: {diff[row, col].item():.4f}")

# Check error distribution
errors_above_10 = torch.sum(diff > 10.0).item()
errors_above_5 = torch.sum(diff > 5.0).item()
errors_above_1 = torch.sum(diff > 1.0).item()

print(f"\nError distribution:")
print(f"  Errors > 10: {errors_above_10} / {output.numel()} ({100*errors_above_10/output.numel():.2f}%)")
print(f"  Errors > 5: {errors_above_5} / {output.numel()} ({100*errors_above_5/output.numel():.2f}%)")
print(f"  Errors > 1: {errors_above_1} / {output.numel()} ({100*errors_above_1/output.numel():.2f}%)")

# Check if it's a systematic error
rel_error = diff / (torch.abs(ref) + 1e-6)
max_rel_error = torch.max(rel_error).item()
print(f"\nMax relative error: {max_rel_error:.4f}")