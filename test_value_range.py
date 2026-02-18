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

def test_gemm(M, N, K, scale_factor):
    """Test MXFP8 GEMM with different value ranges"""

    a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * scale_factor
    b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device) * scale_factor

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    # Reference
    a_dequant = a_mxfp8.dequantize()
    b_dequant = b_mxfp8.dequantize()
    ref = torch.matmul(a_dequant, b_dequant)

    # MXFP8 GEMM
    output = te_generic_gemm_triton(
        A=a_mxfp8, transa=False, B=b_mxfp8, transb=False, D=None,
        quantizer=None, output_dtype=tex.DType.kBFloat16,
        bias=torch.Tensor(), bias_type=tex.DType.kBFloat16,
        gelu=False, gelu_in=torch.Tensor(), grad=False,
        workspace=torch.Tensor(), workspaceSize=0,
        accumulate=False, use_split_accumulator=False,
        comm_overlap=False, comm_type=0,
        extra_output=torch.Tensor(), bulk_overlap=False,
    )

    max_diff = torch.max(torch.abs(output[0] - ref)).item()
    max_val = torch.max(torch.abs(ref)).item()
    rel_error = max_diff / (max_val + 1e-6)

    return max_diff, rel_error, max_val

print("Testing different value ranges:")
print("=" * 60)

M, N, K = 128, 128, 128

for scale in [0.01, 0.1, 1.0, 10.0]:
    max_diff, rel_error, max_val = test_gemm(M, N, K, scale)
    status = "✓" if rel_error < 0.1 else "✗"
    print(f"{status} Scale {scale:6.2f}: max_val={max_val:8.2f}, max_diff={max_diff:8.4f}, rel_err={rel_error:.4f}")
