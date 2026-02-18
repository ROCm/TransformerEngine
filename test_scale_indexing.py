import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Create a controlled test where different K-blocks have different values
# This way we can check if the right scales are being applied
M, N, K = 128, 128, 128  # K=128 = 4 blocks of 32

print("=" * 60)
print("Test: Different values in different K-blocks")
print("=" * 60)

# Create A matrix where each 32-element block has a different constant value
# Row 0: [block0=1, block1=2, block2=3, block3=4]
a_fp32 = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
for block_idx in range(K // 32):
    start = block_idx * 32
    end = start + 32
    # Each block gets value (block_idx + 1)
    a_fp32[:, start:end] = float(block_idx + 1)

# B matrix: all ones
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

print(f"\nA matrix structure:")
print(f"  Block 0 (cols 0-31): all {a_fp32[0, 0].item()}")
print(f"  Block 1 (cols 32-63): all {a_fp32[0, 32].item()}")
print(f"  Block 2 (cols 64-95): all {a_fp32[0, 64].item()}")
print(f"  Block 3 (cols 96-127): all {a_fp32[0, 96].item()}")
print(f"B matrix: all ones")

# Quantize
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"\nA scales (first row, should have 4 values for 4 blocks):")
print(f"  Shape: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"  Values: {a_mxfp8._rowwise_scale_inv[0, :]}")

# Dequantize to check quantization
a_dequant = a_mxfp8.dequantize()
print(f"\nA after dequantization (should still be 1,2,3,4 per block):")
print(f"  Block 0: {a_dequant[0, 0].item()}")
print(f"  Block 1: {a_dequant[0, 32].item()}")
print(f"  Block 2: {a_dequant[0, 64].item()}")
print(f"  Block 3: {a_dequant[0, 96].item()}")

# Expected result: A[0,:] @ B[:,0] = 1*32 + 2*32 + 3*32 + 4*32 = 32*(1+2+3+4) = 32*10 = 320
expected = 32 * (1 + 2 + 3 + 4)
print(f"\nExpected output[0,0]: {expected}")

# Reference with dequantized
b_dequant = b_mxfp8.dequantize()
ref = torch.matmul(a_dequant, b_dequant)
print(f"Reference (dequantized matmul): {ref[0, 0].item()}")

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
)[0]

print(f"MXFP8 kernel output: {output[0, 0].item()}")
print(f"\nComparison:")
print(f"  Expected: {expected}")
print(f"  Reference: {ref[0, 0].item()}")
print(f"  Kernel: {output[0, 0].item()}")
print(f"  Match: {'✓' if abs(output[0, 0].item() - expected) < 1.0 else '✗'}")

# Check if all outputs are the same (they should be since B is all ones)
print(f"\nAll outputs in first row should be identical:")
print(f"  Min: {output[0, :].min().item()}, Max: {output[0, :].max().item()}")
print(f"  Unique values: {output[0, :].unique()}")
