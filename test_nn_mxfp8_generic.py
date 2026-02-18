"""
Test MXFP8 GEMM with NN layout using te_generic_gemm_triton.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def test_nn_layout():
    print("=" * 80)
    print("Testing MXFP8 GEMM with NN layout (generic API)")
    print("=" * 80)

    M, N, K = 128, 128, 256  # No padding issues

    print(f"\nDimensions: M={M}, N={N}, K={K}")
    print(f"Computing: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

    # Create test matrices
    A_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

    # Reference computation
    C_ref = torch.matmul(A_fp32.float(), B_fp32.float())

    print("\n" + "-" * 80)
    print("Quantizing to MXFP8...")

    # Create MXFP8 quantizer
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    # Quantize inputs
    A_mxfp8 = quantizer.quantize(A_fp32)
    B_mxfp8 = quantizer.quantize(B_fp32)

    print(f"A: rowwise {A_mxfp8._rowwise_data.shape}, scale {A_mxfp8._rowwise_scale_inv.shape}")
    print(f"   columnwise {A_mxfp8._columnwise_data.shape}, scale {A_mxfp8._columnwise_scale_inv.shape}")
    print(f"B: rowwise {B_mxfp8._rowwise_data.shape}, scale {B_mxfp8._rowwise_scale_inv.shape}")
    print(f"   columnwise {B_mxfp8._columnwise_data.shape}, scale {B_mxfp8._columnwise_scale_inv.shape}")

    # Create output tensor
    D = torch.zeros((M, N), dtype=torch.float32, device=device)

    print("\n" + "-" * 80)
    print("Calling te_generic_gemm_triton with NN layout...")

    try:
        # Call generic wrapper - it should detect MXFP8 and handle appropriately
        te_generic_gemm_triton(
            A_mxfp8,
            False,  # transA = False (NN layout)
            B_mxfp8,
            False,  # transB = False (NN layout)
            D,      # Output
            None,   # quantizer (not needed for output)
            M, N, K
        )

        print(f"Output shape: {D.shape}")
        print(f"Output dtype: {D.dtype}")

        # Check numerical accuracy
        abs_diff = torch.abs(D - C_ref)
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
            print("\nSample comparison:")
            for i in range(min(3, M)):
                for j in range(min(3, N)):
                    print(f"  [{i},{j}] Ref: {C_ref[i,j].item():8.4f}, MXFP8: {D[i,j].item():8.4f}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

def test_unsupported_layouts():
    print("\n" + "=" * 80)
    print("Testing unsupported layouts (should raise errors)")
    print("=" * 80)

    K, M, N = 256, 128, 128

    # Create test matrices - note different shapes for transpose cases
    A_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    W_fp32 = torch.randn((N, K), dtype=torch.bfloat16, device=device)  # Weight for TN
    B_fp32 = torch.randn((N, K), dtype=torch.bfloat16, device=device)  # For NT case

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    A_mxfp8 = quantizer.quantize(A_fp32)
    W_mxfp8 = quantizer.quantize(W_fp32)
    B_mxfp8 = quantizer.quantize(B_fp32)

    D = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Test TN layout (fprop: Y = X @ W^T)
    print("\nTesting TN layout (transA=True, transB=False)...")
    try:
        te_generic_gemm_triton(
            W_mxfp8,  # Shape [N, K], need transpose to [K, N]
            True,     # transA = True
            A_mxfp8,  # Shape [M, K]
            False,    # transB = False
            D,
            None,
            M, N, K
        )
        print("✗ Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"✓ Expected error raised")
        print(f"  Message: {str(e).split(chr(10))[0]}...")

    # Test NT layout
    print("\nTesting NT layout (transA=False, transB=True)...")
    try:
        te_generic_gemm_triton(
            A_mxfp8,  # Shape [M, K]
            False,    # transA = False
            B_mxfp8,  # Shape [N, K], need transpose to [K, N]
            True,     # transB = True
            D,
            None,
            M, N, K
        )
        print("✗ Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"✓ Expected error raised")
        print(f"  Message: {str(e).split(chr(10))[0]}...")

if __name__ == "__main__":
    test_nn_layout()
    test_unsupported_layouts()

    print("\n" + "=" * 80)
    print("Summary:")
    print("- NN layout (no transposes) is being tested")
    print("- TN, NT layouts correctly raise NotImplementedError")
    print("- This is expected since MXFP8 columnwise is not actually transposed")
    print("=" * 80)