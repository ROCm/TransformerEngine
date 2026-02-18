import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

# Test MXFP8 GEMM through the wrapper API
device = torch.device("cuda")

# Create simple tensors (no batch dims)
M, N, K = 128, 256, 512

# Create regular tensors
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

# Quantize to MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

# Create MXFP8 tensors
a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"A MXFP8: {a_mxfp8._rowwise_data.shape}, scale: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"B MXFP8: {b_mxfp8._rowwise_data.shape}, scale: {b_mxfp8._rowwise_scale_inv.shape}")

# Call GEMM (TN layout: output = A @ B, where A and B are not transposed)
# In BLAS column-major convention: C = op(A) @ op(B)
# For NN layout: transa=False, transb=False
try:
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
    print(f"✓ MXFP8 GEMM succeeded! Output shape: {output[0].shape}")
except Exception as e:
    print(f"✗ MXFP8 GEMM failed: {e}")
    import traceback
    traceback.print_exc()
