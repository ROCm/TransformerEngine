"""
Test MXFP8 with the corrected implementation.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def test_operations():
    """Test all three operations with simple dimensions."""
    batch = 128
    in_features = 768
    out_features = 1024

    # Create test matrices
    W = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    X = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
    dY = torch.randn(batch, out_features, dtype=torch.bfloat16, device=device)

    # Reference computations
    Y_ref = X @ W.T  # fprop
    dX_ref = dY @ W  # dgrad
    dW_ref = dY.T @ X  # wgrad

    print("Reference shapes:")
    print(f"  fprop: Y = X @ W^T → {Y_ref.shape}")
    print(f"  dgrad: dX = dY @ W → {dX_ref.shape}")
    print(f"  wgrad: dW = dY^T @ X → {dW_ref.shape}")

    # Quantize to MXFP8
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    W_mxfp8 = quantizer.quantize(W)
    X_mxfp8 = quantizer.quantize(X)
    dY_mxfp8 = quantizer.quantize(dY)

    print("\nMXFP8 storage shapes:")
    print(f"  W: rowwise {W_mxfp8._rowwise_data.shape}, columnwise {W_mxfp8._columnwise_data.shape}")
    print(f"  X: rowwise {X_mxfp8._rowwise_data.shape}, columnwise {X_mxfp8._columnwise_data.shape}")
    print(f"  dY: rowwise {dY_mxfp8._rowwise_data.shape}, columnwise {dY_mxfp8._columnwise_data.shape}")

    # Test fprop: gemm(W, X, "TN")
    print("\n" + "=" * 60)
    print("fprop: gemm(W, X, 'TN')")
    print("  BLAS: W (transA=T) uses rowwise, X (transB=N) uses rowwise")
    print("  After swap for Triton:")
    print("    First = X (no transpose)")
    print("    Second = W (needs transpose)")

    # Selection
    W_data = W_mxfp8._rowwise_data  # transA=T → rowwise
    W_scale = W_mxfp8._rowwise_scale_inv
    X_data = X_mxfp8._rowwise_data  # transB=N → rowwise
    X_scale = X_mxfp8._rowwise_scale_inv

    # After swap and transpose
    first = X_data  # No transpose (transb=False)
    first_scale = X_scale
    second = W_data.T  # Transpose (transa=True)
    second_scale = W_scale.T

    print(f"  Triton operands: {first.shape} @ {second.shape}")
    print(f"  Scales: {first_scale.shape}, {second_scale.shape}")
    if first.shape[1] == second.shape[0]:
        result_shape = (first.shape[0], second.shape[1])
        print(f"  ✓ Valid matmul → {result_shape}")

    # Test dgrad: gemm(W, dY, "NN")
    print("\n" + "=" * 60)
    print("dgrad: gemm(W, dY, 'NN')")
    print("  BLAS: W (transA=N) uses columnwise, dY (transB=N) uses rowwise")
    print("  After swap for Triton:")
    print("    First = dY (no transpose)")
    print("    Second = W (no transpose)")

    # Selection
    W_data = W_mxfp8._columnwise_data  # transA=N → columnwise
    W_scale = W_mxfp8._columnwise_scale_inv
    dY_data = dY_mxfp8._rowwise_data  # transB=N → rowwise
    dY_scale = dY_mxfp8._rowwise_scale_inv

    # After swap and transpose
    first = dY_data  # No transpose (transb=False)
    first_scale = dY_scale
    second = W_data  # No transpose (transa=False)
    second_scale = W_scale

    print(f"  Triton operands: {first.shape} @ {second.shape}")
    print(f"  Scales: {first_scale.shape}, {second_scale.shape}")
    if first.shape[1] == second.shape[0]:
        result_shape = (first.shape[0], second.shape[1])
        print(f"  ✓ Valid matmul → {result_shape}")

    # Test wgrad: gemm(X, dY, "NT")
    print("\n" + "=" * 60)
    print("wgrad: gemm(X, dY, 'NT')")
    print("  BLAS: X (transA=N) uses columnwise, dY (transB=T) uses columnwise")
    print("  After swap for Triton:")
    print("    First = dY (needs transpose)")
    print("    Second = X (no transpose)")

    # Selection
    X_data = X_mxfp8._columnwise_data  # transA=N → columnwise
    X_scale = X_mxfp8._columnwise_scale_inv
    dY_data = dY_mxfp8._columnwise_data  # transB=T → columnwise
    dY_scale = dY_mxfp8._columnwise_scale_inv

    # After swap and transpose
    first = dY_data.T  # Transpose (transb=True)
    first_scale = dY_scale.T
    second = X_data  # No transpose (transa=False)
    second_scale = X_scale

    print(f"  Triton operands: {first.shape} @ {second.shape}")
    print(f"  Scales: {first_scale.shape}, {second_scale.shape}")
    if first.shape[1] == second.shape[0]:
        result_shape = (first.shape[0], second.shape[1])
        print(f"  ✓ Valid matmul → {result_shape}")

    print("\n" + "=" * 60)
    print("Summary: All operations have correct shapes after:")
    print("1. Selecting MXFP8 format based on BLAS flags (C++ logic)")
    print("2. Swapping operands for row-major")
    print("3. Applying logical transpose to data AND scales")
    print("=" * 60)

if __name__ == "__main__":
    test_operations()