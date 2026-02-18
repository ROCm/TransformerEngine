import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

M, N, K = 128, 128, 256  # Use M=128 to avoid scale padding issues
transa, transb = False, False

print("=" * 60)
print(f"Testing MXFP8 accuracy: M={M}, N={N}, K={K}")
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

print(f"Output shape: {output.shape}")

# Check accuracy - use the correct quantization for reference
# The kernel uses A columnwise + B rowwise (after the swap logic)
# Actually, due to swap: kernel uses B rowwise as first operand, A columnwise as second

# Manually dequantize using the correct quantizations
VEC_SIZE = 32

# With reversed selection for tl.dot_scaled:
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

# Reference computation
ref = torch.matmul(a_dequant, b_dequant)

# Check for inf/nan
num_inf = torch.sum(torch.isinf(output)).item()
num_nan = torch.sum(torch.isnan(output)).item()

print(f"\nInf/NaN check:")
print(f"  Inf values: {num_inf} / {output.numel()}")
print(f"  NaN values: {num_nan} / {output.numel()}")

if num_inf > 0 or num_nan > 0:
    # Compute accuracy only on non-inf/nan values
    valid_mask = ~(torch.isinf(output) | torch.isnan(output))
    if torch.any(valid_mask):
        max_diff = torch.max(torch.abs(output[valid_mask] - ref[valid_mask])).item()
        print(f"\nAccuracy (excluding inf/nan):")
        print(f"  Max difference: {max_diff:.4f}")
else:
    # Compute accuracy on all values
    max_diff = torch.max(torch.abs(output - ref)).item()
    rel_error = max_diff / torch.max(torch.abs(ref)).item()

    print(f"\nAccuracy:")
    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Relative error: {rel_error:.4%}")

    # Sample comparison
    print(f"\nSample values:")
    print(f"  Output[0,0] = {output[0, 0].item():.4f}")
    print(f"  Ref[0,0] = {ref[0, 0].item():.4f}")
    print(f"  Output[1,1] = {output[1, 1].item():.4f}")
    print(f"  Ref[1,1] = {ref[1, 1].item():.4f}")

    if max_diff < 1.0:
        print(f"\n✓ Good accuracy!")
    elif max_diff < 10.0:
        print(f"\n⚠ Moderate accuracy")
    else:
        print(f"\n✗ Poor accuracy")