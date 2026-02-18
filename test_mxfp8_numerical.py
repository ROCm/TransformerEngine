import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton

device = torch.device("cuda")

def test_mxfp8_gemm_numerical(M, N, K, transa=False, transb=False):
    """Test MXFP8 GEMM with numerical validation"""

    # Create input tensors
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

    # Quantize to MXFP8
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    # Compute MXFP8 GEMM
    output = te_generic_gemm_triton(
        A=a_mxfp8,
        transa=transa,
        B=b_mxfp8,
        transb=transb,
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

    # Compute reference with dequantized tensors
    a_dequant = a_mxfp8.dequantize()
    b_dequant = b_mxfp8.dequantize()

    if transa:
        a_dequant = a_dequant.T
    if transb:
        b_dequant = b_dequant.T

    ref_output = torch.matmul(a_dequant, b_dequant)

    # Check shape
    expected_shape = (M, N)
    if output[0].shape != expected_shape:
        print(f"  ✗ Shape mismatch: got {output[0].shape}, expected {expected_shape}")
        return False

    # Check numerical accuracy
    max_diff = torch.max(torch.abs(output[0].float() - ref_output.float())).item()
    max_val = torch.max(torch.abs(ref_output.float())).item()
    rel_error = max_diff / (max_val + 1e-6)

    # For MXFP8, expect some quantization error
    # Typical tolerance is around 1-5% relative error
    if rel_error > 0.1:  # 10% tolerance
        print(f"  ✗ Large numerical error: max_diff={max_diff:.4f}, rel_error={rel_error:.4f}")
        return False

    print(f"  ✓ M={M}, N={N}, K={K}, transa={transa}, transb={transb}")
    print(f"    Shape: {output[0].shape}, max_diff={max_diff:.4f}, rel_error={rel_error:.4f}")
    return True


print("Testing MXFP8 GEMM numerical correctness")
print("=" * 60)

all_passed = True

# Test different sizes and layouts
test_cases = [
    # (M, N, K, transa, transb)
    (128, 256, 512, False, False),  # NN layout
    (128, 256, 512, True, False),   # TN layout
    (128, 256, 512, False, True),   # NT layout
    (256, 128, 512, False, False),  # Different M, N
    (512, 512, 512, False, False),  # Square
    (64, 64, 64, False, False),     # Small
]

for M, N, K, transa, transb in test_cases:
    try:
        passed = test_mxfp8_gemm_numerical(M, N, K, transa, transb)
        all_passed = all_passed and passed
    except Exception as e:
        print(f"  ✗ M={M}, N={N}, K={K}, transa={transa}, transb={transb}")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

print("=" * 60)
if all_passed:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed")

exit(0 if all_passed else 1)
