"""
Test MXFP8 GEMM with NN layout using direct te_gemm_triton.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import te_gemm_triton
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def test_nn_layout_direct():
    print("=" * 80)
    print("Testing MXFP8 GEMM with NN layout (direct API)")
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

    print("\n" + "-" * 80)
    print("Selecting data for NN layout:")
    print("- A needs rowwise (scales along K)")
    print("- B needs columnwise (scales along N)")

    # For NN layout: A uses rowwise, B uses columnwise
    A_data = A_mxfp8._rowwise_data
    A_scale = A_mxfp8._rowwise_scale_inv
    B_data = B_mxfp8._columnwise_data
    B_scale = B_mxfp8._columnwise_scale_inv

    print(f"\nSelected A: data {A_data.shape}, scale {A_scale.shape}")
    print(f"Selected B: data {B_data.shape}, scale {B_scale.shape}")

    # Create output tensor
    D = torch.zeros((M, N), dtype=torch.float32, device=device)

    print("\n" + "-" * 80)
    print("Calling te_gemm_triton...")

    try:
        # Call low-level API directly
        te_gemm_triton(
            A_data, A_scale, True, tex.DType.kFloat8E4M3, False,  # A, no transpose
            B_data, B_scale, True, tex.DType.kFloat8E4M3, False,  # B, no transpose
            D,  # Output
            None, torch.float32,  # D_scale_inverse, D_type
            torch.float32,  # Output type
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

if __name__ == "__main__":
    test_nn_layout_direct()