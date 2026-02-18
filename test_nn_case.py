import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Test transA=False, transB=False case (NN layout)
M, N, K = 128, 128, 256
transa, transb = False, False

print("=" * 60)
print(f"Testing transA={transa}, transB={transb} (NN layout)")
print("=" * 60)

torch.manual_seed(42)

# Input shapes
a_shape = (M, K)
b_shape = (K, N)

a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device) * 0.5
b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device) * 0.5

print(f"A shape: {a_shape}")
print(f"B shape: {b_shape}")
print(f"Expected output shape: [{M}, {N}]")

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"\nQuantized tensor shapes:")
print(f"A: {a_mxfp8.size()}")
print(f"B: {b_mxfp8.size()}")

# Run kernel
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

print(f"\nOutput shape: {output.shape}")

# Reference computation
ref_fp32 = torch.matmul(a_fp32, b_fp32)

# Check for inf/nan
num_inf = torch.sum(torch.isinf(output)).item()
num_nan = torch.sum(torch.isnan(output)).item()

print(f"\nInf values: {num_inf}, NaN values: {num_nan}")

if num_inf == 0 and num_nan == 0:
    # Compute accuracy
    diff = torch.abs(output - ref_fp32)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    # Relative error for significant values
    threshold = 0.1
    mask = torch.abs(ref_fp32) > threshold
    if torch.any(mask):
        rel_errors = torch.abs((output[mask] - ref_fp32[mask]) / ref_fp32[mask])
        max_rel_error = torch.max(rel_errors).item()
        mean_rel_error = torch.mean(rel_errors).item()
    else:
        max_rel_error = 0
        mean_rel_error = 0

    print(f"\nAccuracy vs FP32:")
    print(f"  Max absolute error: {max_diff:.4f}")
    print(f"  Mean absolute error: {mean_diff:.4f}")
    print(f"  Max relative error (|ref|>{threshold}): {max_rel_error:.4%}")
    print(f"  Mean relative error (|ref|>{threshold}): {mean_rel_error:.4%}")

    # Sample values
    print(f"\nSample values:")
    for i in range(3):
        for j in range(3):
            print(f"  [{i},{j}] Output={output[i,j].item():.4f}, Ref={ref_fp32[i,j].item():.4f}")

    if max_rel_error < 0.15 and mean_rel_error < 0.05:
        print("\n✓ Good accuracy for MXFP8!")
    elif max_rel_error < 0.25:
        print("\n⚠ Moderate accuracy")
    else:
        print("\n✗ Poor accuracy")
else:
    print("✗ Contains inf/nan values!")