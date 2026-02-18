import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

M, N, K = 128, 128, 128

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

print("=" * 60)
print("Which quantization should the reference use?")
print("=" * 60)

# Kernel uses A rowwise + B columnwise
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

# Reference 1: using built-in dequantize() (probably rowwise for both)
ref1 = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())

# Reference 2: manually dequantize A rowwise + B columnwise
# A rowwise dequantization
a_rowwise_data = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
a_rowwise_scale = a_mxfp8._rowwise_scale_inv
a_dequant = torch.zeros_like(a_fp32)
VEC_SIZE = 32
for i in range(M):
    for j in range(K // VEC_SIZE):
        scale = 2.0 ** (a_rowwise_scale[i, j].item() - 127.0)
        a_dequant[i, j*VEC_SIZE:(j+1)*VEC_SIZE] = a_rowwise_data[i, j*VEC_SIZE:(j+1)*VEC_SIZE] * scale

# B columnwise dequantization
b_columnwise_data = b_mxfp8._columnwise_data.view(torch.float8_e4m3fn).to(torch.float32)
b_columnwise_scale = b_mxfp8._columnwise_scale_inv
b_dequant = torch.zeros_like(b_fp32)
for j in range(N):
    for i in range(K // VEC_SIZE):
        scale = 2.0 ** (b_columnwise_scale[i, j].item() - 127.0)
        b_dequant[i*VEC_SIZE:(i+1)*VEC_SIZE, j] = b_columnwise_data[i*VEC_SIZE:(i+1)*VEC_SIZE, j] * scale

ref2 = torch.matmul(a_dequant, b_dequant)

print(f"\nKernel output[0,0]: {output[0, 0].item():.4f}")
print(f"Ref1 (built-in dequant)[0,0]: {ref1[0, 0].item():.4f}")
print(f"Ref2 (manual A rowwise + B columnwise)[0,0]: {ref2[0, 0].item():.4f}")

diff1 = torch.max(torch.abs(output - ref1)).item()
diff2 = torch.max(torch.abs(output - ref2)).item()

print(f"\nMax diff:")
print(f"  Kernel vs Ref1: {diff1:.4f}")
print(f"  Kernel vs Ref2: {diff2:.4f}")

if diff2 < diff1:
    print(f"\n✓ Kernel matches Ref2 better (A rowwise + B columnwise)")
else:
    print(f"\n? Kernel matches Ref1 better (built-in dequant)")
