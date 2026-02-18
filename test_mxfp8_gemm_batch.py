import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

# Test MXFP8 GEMM with batch dimensions (like in Megatron)
device = torch.device("cuda")

# Create tensors WITH batch dims (batch*seq, hidden)
batch_seq = 32  # Must be divisible by 32 for MXFP8
M, K = 128, 512  # hidden dimensions (also divisible by 32)
N = 256

# Linear layer forward pass (TN layout):
# weight: [out_features, in_features] = [M, K]
# input: [batch, in_features] = [batch_seq, K]
# output = input @ weight.T = [batch_seq, M]

# Create weight and input
weight_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
input_fp32 = torch.randn((batch_seq, K), dtype=torch.bfloat16, device=device)

# Quantize to MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

weight_mxfp8 = quantizer.quantize(weight_fp32)
input_mxfp8 = quantizer.quantize(input_fp32)

print(f"Weight MXFP8: {weight_mxfp8._rowwise_data.shape}, scale: {weight_mxfp8._rowwise_scale_inv.shape}")
print(f"Input MXFP8: {input_mxfp8._rowwise_data.shape}, scale: {input_mxfp8._rowwise_scale_inv.shape}")
print(f"Expected output shape: [{batch_seq}, {M}]")

# For linear layer forward: output = input @ weight.T
# In BLAS column-major (TN layout): C = B @ A.T where A=weight, B=input
# This means: transa=False (use weight as-is in column-major = weight.T in row-major)
#             transb=True (transpose input in column-major)
#
# Actually, let me think about this more carefully...
# Megatron uses: general_gemm(weight, input) with some transpose flags

# Let's try TN layout (typical for linear layer fprop)
print("\nTrying TN layout (transa=False, transb=True)...")
try:
    output = te_generic_gemm_triton(
        A=weight_mxfp8,  # [M, K]
        transa=False,
        B=input_mxfp8,    # [batch_seq, K]
        transb=True,
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
