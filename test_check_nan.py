import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

# Test case that showed NaN
M, N, K = 128, 256, 512
transa, transb = False, False

torch.manual_seed(42)
if transa:
    a_shape = (K, M)
else:
    a_shape = (M, K)

if transb:
    b_shape = (N, K)
else:
    b_shape = (K, N)

a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

output = te_generic_gemm_triton(
    A=a_mxfp8, transa=transa, B=b_mxfp8, transb=transb, D=None,
    quantizer=None, output_dtype=tex.DType.kBFloat16,
    bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
    gelu=False, gelu_in=torch.Tensor(), grad=False,
    workspace=torch.Tensor(), workspaceSize=0,
    accumulate=False, use_split_accumulator=False,
    comm_overlap=False, comm_type=0,
    extra_output=torch.Tensor(), bulk_overlap=False,
)[0]

ref = torch.matmul(
    a_mxfp8.dequantize().T if transa else a_mxfp8.dequantize(),
    b_mxfp8.dequantize().T if transb else b_mxfp8.dequantize()
)

print(f"Output contains NaN: {torch.isnan(output).any().item()}")
print(f"Output contains Inf: {torch.isinf(output).any().item()}")
print(f"Ref contains NaN: {torch.isnan(ref).any().item()}")
print(f"Ref contains Inf: {torch.isinf(ref).any().item()}")

if not torch.isnan(output).any():
    max_diff = torch.max(torch.abs(output - ref)).item()
    max_val = torch.max(torch.abs(ref)).item()
    rel_error = max_diff / (max_val + 1e-6)
    print(f"\nMax diff: {max_diff:.4f}")
    print(f"Rel error: {rel_error:.4f}")
    print(f"First few: kernel={output[0, :5]}, ref={ref[0, :5]}")
