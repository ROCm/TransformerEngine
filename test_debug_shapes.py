import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton
import os

# Enable debug output
os.environ["DEBUG_MXFP8_GEMM"] = "1"

device = torch.device("cuda")

# Simple case: C = A @ B where A is [M,K], B is [K,N]
M, N, K = 128, 256, 512

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

# Quantize
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("="*60)
print("Input tensors:")
print(f"A: logical={a_mxfp8.shape}, rowwise_data={a_mxfp8._rowwise_data.shape}, rowwise_scale={a_mxfp8._rowwise_scale_inv.shape}")
print(f"B: logical={b_mxfp8.shape}, rowwise_data={b_mxfp8._rowwise_data.shape}, rowwise_scale={b_mxfp8._rowwise_scale_inv.shape}")
print(f"\nExpected output shape: [{M}, {N}] = [128, 256]")
print("="*60)

# Test NN layout (most straightforward: C = A @ B, no transposes)
print("\nTest NN layout (transa=False, transb=False):")
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

print(f"\nActual output shape: {output[0].shape}")
print(f"Expected: [128, 256]")
print(f"Match: {output[0].shape == torch.Size([128, 256])}")
