import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Simple test case with known values
M, N, K = 128, 128, 128  # Smaller K for simplicity
torch.manual_seed(123)

# Create simpler test data with smaller values
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device) * 0.1

print("Simple test case")
print("=" * 60)

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

# Reference - using original FP32 values
ref_fp32 = torch.matmul(a_fp32, b_fp32)

# Reference with quantized values
VEC_SIZE = 32

# A rowwise dequantization
a_rowwise_data = a_mxfp8._rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
a_rowwise_scale = a_mxfp8._rowwise_scale_inv
a_dequant = torch.zeros_like(a_fp32)
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

ref_quant = torch.matmul(a_dequant, b_dequant)

# Compare
print(f"Output shape: {output.shape}")
print(f"Reference shape: {ref_quant.shape}")

# Check for inf/nan
num_inf = torch.sum(torch.isinf(output)).item()
num_nan = torch.sum(torch.isnan(output)).item()
print(f"\nInf values: {num_inf}, NaN values: {num_nan}")

if num_inf == 0 and num_nan == 0:
    # Compute differences
    diff_quant = torch.max(torch.abs(output - ref_quant)).item()
    diff_fp32 = torch.max(torch.abs(output - ref_fp32)).item()

    print(f"\nMax diff vs quantized ref: {diff_quant:.6f}")
    print(f"Max diff vs FP32 ref: {diff_fp32:.6f}")

    # Sample values
    print(f"\nSample values:")
    for i in range(3):
        for j in range(3):
            print(f"  [{i},{j}] Output={output[i,j].item():.4f}, RefQuant={ref_quant[i,j].item():.4f}, RefFP32={ref_fp32[i,j].item():.4f}")

    # Check relative error
    rel_error = torch.max(torch.abs(output - ref_quant) / (torch.abs(ref_quant) + 1e-6)).item()
    print(f"\nMax relative error: {rel_error:.4f}")

    if diff_quant < 0.1:
        print("\n✓ Excellent accuracy!")
    elif diff_quant < 1.0:
        print("\n✓ Good accuracy")
    elif diff_quant < 5.0:
        print("\n⚠ Moderate accuracy")
    else:
        print("\n✗ Poor accuracy")