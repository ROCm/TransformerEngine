import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

# Test various sizes that avoid scale padding issues
test_cases = [
    (128, 128, 256),
    (256, 256, 128),
    (512, 128, 256),
]

for M, N, K in test_cases:
    print("=" * 60)
    print(f"Testing M={M}, N={N}, K={K}")
    print("=" * 60)

    torch.manual_seed(42)

    # Create test data with moderate values
    a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5
    b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device) * 0.5

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    # Run kernel
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

    # Reference with FP32
    ref_fp32 = torch.matmul(a_fp32, b_fp32)

    # Check for inf/nan
    num_inf = torch.sum(torch.isinf(output)).item()
    num_nan = torch.sum(torch.isnan(output)).item()

    print(f"Output shape: {output.shape}")
    print(f"Inf values: {num_inf}, NaN values: {num_nan}")

    if num_inf == 0 and num_nan == 0:
        # Compute accuracy metrics
        diff = torch.abs(output - ref_fp32)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        # Relative error for values above threshold
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

        # Quantization error is expected to be around 5-10% for MXFP8
        if max_rel_error < 0.15 and mean_rel_error < 0.05:
            print("  ✓ Good accuracy for MXFP8!")
        elif max_rel_error < 0.25:
            print("  ⚠ Moderate accuracy")
        else:
            print("  ✗ Poor accuracy")
    else:
        print("✗ Contains inf/nan values!")

    print()