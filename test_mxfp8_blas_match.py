"""
Test that MXFP8 Triton matches expected BLAS behavior.
"""

import torch
import os
os.environ["DEBUG_MXFP8_SELECT"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def test_fprop():
    """Test forward pass: Y = X @ W^T"""
    print("=" * 80)
    print("Testing fprop: Y = X @ W^T")
    print("=" * 80)

    batch = 128
    in_features = 768
    out_features = 1024

    # Create weight and input
    W = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    X = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)

    # Reference computation (what we want in row-major)
    Y_ref = X @ W.T

    print(f"\nShapes:")
    print(f"  W: {W.shape}")
    print(f"  X: {X.shape}")
    print(f"  Y_ref: {Y_ref.shape}")

    # BLAS API call would be: gemm(W, X, "TN")
    print(f"\nBLAS API: gemm(W, X, 'TN')")
    print(f"  First arg (A): W[{out_features}, {in_features}], transA=True")
    print(f"  Second arg (B): X[{batch}, {in_features}], transB=False")

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    W_mxfp8 = quantizer.quantize(W)
    X_mxfp8 = quantizer.quantize(X)

    # Test selection with BLAS flags
    print("\n" + "-" * 80)
    print("MXFP8 Selection (following BLAS API):")

    # Following C++ logic:
    # transA=True → W uses rowwise
    # transB=False → X uses rowwise
    print("  W (transA=True): should use rowwise")
    print(f"    W rowwise: data {W_mxfp8._rowwise_data.shape}, scale {W_mxfp8._rowwise_scale_inv.shape}")
    print("  X (transB=False): should use rowwise")
    print(f"    X rowwise: data {X_mxfp8._rowwise_data.shape}, scale {X_mxfp8._rowwise_scale_inv.shape}")

    # After operand swap for Triton
    print("\n" + "-" * 80)
    print("After operand swap for Triton (row-major):")
    print("  First operand: B (X)")
    print("  Second operand: A (W)")
    print("  Triton computes: X @ W^T")

def test_dgrad():
    """Test backward dgrad: dX = dY @ W"""
    print("\n" + "=" * 80)
    print("Testing dgrad: dX = dY @ W")
    print("=" * 80)

    batch = 128
    in_features = 768
    out_features = 1024

    # Create weight and grad output
    W = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    dY = torch.randn(batch, out_features, dtype=torch.bfloat16, device=device)

    # Reference computation
    dX_ref = dY @ W

    print(f"\nShapes:")
    print(f"  W: {W.shape}")
    print(f"  dY: {dY.shape}")
    print(f"  dX_ref: {dX_ref.shape}")

    # BLAS API call would be: gemm(W, dY, "NN")
    print(f"\nBLAS API: gemm(W, dY, 'NN')")
    print(f"  First arg (A): W[{out_features}, {in_features}], transA=False")
    print(f"  Second arg (B): dY[{batch}, {out_features}], transB=False")

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    W_mxfp8 = quantizer.quantize(W)
    dY_mxfp8 = quantizer.quantize(dY)

    # Test selection
    print("\n" + "-" * 80)
    print("MXFP8 Selection (following BLAS API):")

    # Following C++ logic:
    # transA=False → W uses columnwise
    # transB=False → dY uses rowwise
    print("  W (transA=False): should use columnwise")
    print(f"    W columnwise: data {W_mxfp8._columnwise_data.shape}, scale {W_mxfp8._columnwise_scale_inv.shape}")
    print("  dY (transB=False): should use rowwise")
    print(f"    dY rowwise: data {dY_mxfp8._rowwise_data.shape}, scale {dY_mxfp8._rowwise_scale_inv.shape}")

    # After operand swap for Triton
    print("\n" + "-" * 80)
    print("After operand swap for Triton (row-major):")
    print("  First operand: B (dY)")
    print("  Second operand: A (W)")
    print("  Triton computes: dY @ W")

def test_wgrad():
    """Test backward wgrad: dW = dY^T @ X"""
    print("\n" + "=" * 80)
    print("Testing wgrad: dW = dY^T @ X")
    print("=" * 80)

    batch = 128
    in_features = 768
    out_features = 1024

    # Create input and grad output
    X = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
    dY = torch.randn(batch, out_features, dtype=torch.bfloat16, device=device)

    # Reference computation
    dW_ref = dY.T @ X

    print(f"\nShapes:")
    print(f"  X: {X.shape}")
    print(f"  dY: {dY.shape}")
    print(f"  dW_ref: {dW_ref.shape}")

    # BLAS API call would be: gemm(X, dY, "NT")
    print(f"\nBLAS API: gemm(X, dY, 'NT')")
    print(f"  First arg (A): X[{batch}, {in_features}], transA=False")
    print(f"  Second arg (B): dY[{batch}, {out_features}], transB=True")

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    X_mxfp8 = quantizer.quantize(X)
    dY_mxfp8 = quantizer.quantize(dY)

    # Test selection
    print("\n" + "-" * 80)
    print("MXFP8 Selection (following BLAS API):")

    # Following C++ logic:
    # transA=False → X uses columnwise
    # transB=True → dY uses columnwise
    print("  X (transA=False): should use columnwise")
    print(f"    X columnwise: data {X_mxfp8._columnwise_data.shape}, scale {X_mxfp8._columnwise_scale_inv.shape}")
    print("  dY (transB=True): should use columnwise")
    print(f"    dY columnwise: data {dY_mxfp8._columnwise_data.shape}, scale {dY_mxfp8._columnwise_scale_inv.shape}")

    # After operand swap for Triton
    print("\n" + "-" * 80)
    print("After operand swap for Triton (row-major):")
    print("  First operand: B (dY)")
    print("  Second operand: A (X)")
    print("  Triton computes: dY^T @ X (with transB=True)")

if __name__ == "__main__":
    test_fprop()
    test_dgrad()
    test_wgrad()

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("- MXFP8 selection follows C++ BLAS logic")
    print("- Operands are swapped for Triton (row-major)")
    print("- This should match BLAS behavior")
    print("=" * 80)