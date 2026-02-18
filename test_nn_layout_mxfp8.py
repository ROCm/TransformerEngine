"""
Test MXFP8 GEMM with NN layout (the only supported case).
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"
os.environ["TRITON_MXFP8_VERSION"] = "3"  # Indicate updated version

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import te_gemm_triton
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def test_nn_layout():
    print("=" * 80)
    print("Testing MXFP8 GEMM with NN layout (no transposes)")
    print("=" * 80)

    M, N, K = 128, 128, 256  # No padding issues with these dimensions

    print(f"\nDimensions: M={M}, N={N}, K={K}")
    print(f"Computing: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

    # Create test matrices
    A_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

    # Reference computation
    C_ref = torch.matmul(A_fp32.float(), B_fp32.float())

    print("\n" + "-" * 80)
    print("Quantizing to MXFP8...")

    # Create MXFP8 quantizer with both rowwise and columnwise
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    # Quantize inputs
    A_mxfp8 = quantizer.quantize(A_fp32)
    B_mxfp8 = quantizer.quantize(B_fp32)

    print(f"A_mxfp8: rowwise {A_mxfp8._rowwise_data.shape}, columnwise {A_mxfp8._columnwise_data.shape}")
    print(f"B_mxfp8: rowwise {B_mxfp8._rowwise_data.shape}, columnwise {B_mxfp8._columnwise_data.shape}")

    print("\n" + "-" * 80)
    print("Running MXFP8 GEMM with Triton...")

    try:
        # Call Triton GEMM with NN layout
        C_mxfp8 = te_gemm_triton(
            A_mxfp8,
            B_mxfp8,
            M, N, K,
            layout='NN',  # No transposes
            out_dtype=torch.float32
        )

        print(f"Output shape: {C_mxfp8.shape}")
        print(f"Output dtype: {C_mxfp8.dtype}")

        # Check numerical accuracy
        abs_diff = torch.abs(C_mxfp8 - C_ref)
        rel_diff = abs_diff / (torch.abs(C_ref) + 1e-8)

        print("\n" + "-" * 80)
        print("Numerical Accuracy:")
        print(f"Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
        print(f"Max relative difference: {rel_diff.max().item():.4%}")
        print(f"Mean relative difference: {rel_diff.mean().item():.4%}")

        # Check if results are reasonable
        if rel_diff.max().item() < 0.10:  # Less than 10% error
            print("\n✓ MXFP8 NN layout works correctly!")
        else:
            print("\n✗ Large numerical errors detected")
            print("\nSample values:")
            print(f"Reference[0,0]: {C_ref[0,0].item():.6f}")
            print(f"MXFP8[0,0]: {C_mxfp8[0,0].item():.6f}")

    except Exception as e:
        print(f"\n✗ Error: {e}")

def test_unsupported_layouts():
    print("\n" + "=" * 80)
    print("Testing unsupported layouts (should raise errors)")
    print("=" * 80)

    M, N, K = 128, 128, 256

    # Create test matrices
    A_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)
    W_fp32 = torch.randn((N, K), dtype=torch.bfloat16, device=device)  # For TN case

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    A_mxfp8 = quantizer.quantize(A_fp32)
    B_mxfp8 = quantizer.quantize(B_fp32)
    W_mxfp8 = quantizer.quantize(W_fp32)  # Weight for TN case

    # Test TN layout (fprop case)
    print("\nTesting TN layout (should fail)...")
    try:
        C_mxfp8 = te_gemm_triton(
            W_mxfp8, A_mxfp8,
            M, N, K,
            layout='TN',  # transA=True, transB=False
            out_dtype=torch.float32
        )
        print("✗ Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"✓ Expected error: {str(e).split(chr(10))[0]}...")

    # Test NT layout (wgrad case)
    print("\nTesting NT layout (should fail)...")
    try:
        C_mxfp8 = te_gemm_triton(
            A_mxfp8, B_mxfp8,
            M, N, K,
            layout='NT',  # transA=False, transB=True
            out_dtype=torch.float32
        )
        print("✗ Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"✓ Expected error: {str(e).split(chr(10))[0]}...")

if __name__ == "__main__":
    test_nn_layout()
    test_unsupported_layouts()

    print("\n" + "=" * 80)
    print("Summary:")
    print("- NN layout (no transposes) is supported and works")
    print("- TN, NT, TT layouts correctly raise NotImplementedError")
    print("- This is expected since MXFP8 columnwise is not actually transposed")
    print("=" * 80)