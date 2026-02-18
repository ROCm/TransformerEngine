"""
Comprehensive test of MXFP8 selection and computation.
"""

import torch
import os

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

def analyze_operation(name, A_shape, B_shape, transa, transb, expected_result_shape):
    """Analyze what happens with specific operation."""
    print("=" * 80)
    print(f"{name} Analysis")
    print("=" * 80)

    # Create test matrices
    A = torch.randn(A_shape, dtype=torch.bfloat16, device=device)
    B = torch.randn(B_shape, dtype=torch.bfloat16, device=device)

    # Compute reference based on transpose flags (BLAS semantics)
    A_op = A.T if transa else A
    B_op = B.T if transb else B
    C_ref = A_op @ B_op

    print(f"\nBLAS API: gemm(A, B, trans={'T' if transa else 'N'}{'T' if transb else 'N'})")
    print(f"  A: {A_shape}, transA={transa}")
    print(f"  B: {B_shape}, transB={transb}")
    print(f"  Result: {C_ref.shape}")

    assert C_ref.shape == expected_result_shape, f"Expected {expected_result_shape}, got {C_ref.shape}"

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    A_mxfp8 = quantizer.quantize(A)
    B_mxfp8 = quantizer.quantize(B)

    print("\n" + "-" * 80)
    print("MXFP8 Selection (C++ logic):")

    # C++ selection logic
    if transa:
        print(f"  A (transA=True): uses rowwise")
        A_selected = A_mxfp8._rowwise_data
        A_scale_selected = A_mxfp8._rowwise_scale_inv
    else:
        print(f"  A (transA=False): uses columnwise")
        A_selected = A_mxfp8._columnwise_data
        A_scale_selected = A_mxfp8._columnwise_scale_inv

    if transb:
        print(f"  B (transB=True): uses columnwise")
        B_selected = B_mxfp8._columnwise_data
        B_scale_selected = B_mxfp8._columnwise_scale_inv
    else:
        print(f"  B (transB=False): uses rowwise")
        B_selected = B_mxfp8._rowwise_data
        B_scale_selected = B_mxfp8._rowwise_scale_inv

    print(f"    A selected: data {A_selected.shape}, scale {A_scale_selected.shape}")
    print(f"    B selected: data {B_selected.shape}, scale {B_scale_selected.shape}")

    print("\n" + "-" * 80)
    print("For Triton (after operand swap):")

    # After swap: First=B, Second=A
    # Transposes: transB applies to first, transA to second

    first_data = B_selected.T if transb else B_selected
    first_scale = B_scale_selected.T if transb else B_scale_selected

    second_data = A_selected.T if transa else A_selected
    second_scale = A_scale_selected.T if transa else A_scale_selected

    print(f"  First operand (was B): {first_data.shape}, scale {first_scale.shape}")
    print(f"  Second operand (was A): {second_data.shape}, scale {second_scale.shape}")

    # Verify dimensions match for matmul
    if first_data.shape[1] == second_data.shape[0]:
        print(f"  ✓ Dimensions match for matmul: [{first_data.shape[0]}, {first_data.shape[1]}] @ [{second_data.shape[0]}, {second_data.shape[1]}]")
        result_shape = (first_data.shape[0], second_data.shape[1])
        print(f"  Result would be: {result_shape}")
        if result_shape == C_ref.shape:
            print(f"  ✓ Matches expected result shape!")
        else:
            print(f"  ✗ Does NOT match expected shape {C_ref.shape}")
    else:
        print(f"  ✗ Dimension mismatch!")

def main():
    batch = 128
    in_features = 768
    out_features = 1024

    # Test fprop: Y = X @ W^T
    # BLAS call: gemm(W, X, "TN")
    analyze_operation(
        "fprop (Y = X @ W^T)",
        A_shape=(out_features, in_features),  # W
        B_shape=(batch, in_features),          # X
        transa=True,
        transb=False,
        expected_result_shape=(batch, out_features)
    )

    # Test dgrad: dX = dY @ W
    # BLAS call: gemm(W, dY, "NN")
    analyze_operation(
        "dgrad (dX = dY @ W)",
        A_shape=(out_features, in_features),   # W
        B_shape=(batch, out_features),         # dY
        transa=False,
        transb=False,
        expected_result_shape=(batch, in_features)
    )

    # Test wgrad: dW = dY^T @ X
    # BLAS call: gemm(X, dY, "NT")
    analyze_operation(
        "wgrad (dW = dY^T @ X)",
        A_shape=(batch, in_features),          # X
        B_shape=(batch, out_features),         # dY
        transa=False,
        transb=True,
        expected_result_shape=(out_features, in_features)
    )

if __name__ == "__main__":
    main()