import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Use size that doesn't need padding: 128 is multiple of 128, 512//32=16 is multiple of 4
M, N, K = 128, 256, 512

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device) * 0.1

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"A data: {a_mxfp8._rowwise_data.shape}, scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"B data: {b_mxfp8._rowwise_data.shape}, scale: {b_mxfp8._rowwise_scale_inv.shape}")
print(f"Expected A scale: [128, 512//32] = [128, 16]")
print(f"Expected B scale: [512, 256//32] = [512, 8]")

# Compute reference
a_dequant = a_mxfp8.dequantize()
b_dequant = b_mxfp8.dequantize()
ref = torch.matmul(a_dequant, b_dequant)

# Compute with MXFP8
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

max_diff = torch.max(torch.abs(output[0] - ref)).item()
max_val = torch.max(torch.abs(ref)).item()
rel_error = max_diff / (max_val + 1e-6)

print(f"\nResults:")
print(f"  Max diff: {max_diff:.4f}")
print(f"  Rel error: {rel_error:.4f}")
print(f"  First few elements:")
print(f"    Kernel: {output[0][0, :5]}")
print(f"    Ref:    {ref[0, :5]}")
