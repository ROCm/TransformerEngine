import os
import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

# Enable debug mode
os.environ["DEBUG_MXFP8_GEMM"] = "1"

device = torch.device("cuda")

M, N, K = 128, 256, 512
transa, transb = False, False

print("=" * 60)
print(f"Testing MXFP8 kernel call: M={M}, N={N}, K={K}")
print(f"transa={transa}, transb={transb}")
print("=" * 60)

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

print(f"\nOriginal shapes:")
print(f"A: {a_mxfp8.size()}")
print(f"B: {b_mxfp8.size()}")
print(f"Expected output: [{M}, {N}]")

# Call the kernel
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

print(f"\nActual output shape: {output.shape}")

# Check first few values
print(f"\nFirst few output values:")
print(f"output[0, :5] = {output[0, :5]}")
print(f"output[1, :5] = {output[1, :5]}")

if torch.any(torch.isinf(output)):
    print(f"\n✗ Output contains inf!")
    inf_mask = torch.isinf(output)
    num_inf = torch.sum(inf_mask).item()
    print(f"  Number of inf values: {num_inf} / {output.numel()}")
if torch.any(torch.isnan(output)):
    print(f"\n✗ Output contains nan!")