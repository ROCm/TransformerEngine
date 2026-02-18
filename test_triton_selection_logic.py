"""
Determine what MXFP8 data/scale selection Triton needs for each layout.

Triton kernel (row-major) expects:
- First operand: [M, K] with scales [M, K//32] (rowwise)
- Second operand: [K, N] with scales [K//32, N] (columnwise)

tl.dot_scaled expects the scales to match this pattern.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

def analyze_layout(transa, transb, M=128, N=128, K=256):
    print("=" * 60)
    print(f"Layout: transA={transa}, transB={transb}")
    print("=" * 60)

    # Create the input shapes based on transpose flags
    if transa:
        a_shape = (K, M)  # Will be transposed to [M, K]
        print(f"A shape: {a_shape} (will transpose to [{M}, {K}])")
    else:
        a_shape = (M, K)  # Already [M, K]
        print(f"A shape: {a_shape} (no transpose)")

    if transb:
        b_shape = (N, K)  # Will be transposed to [K, N]
        print(f"B shape: {b_shape} (will transpose to [{K}, {N}])")
    else:
        b_shape = (K, N)  # Already [K, N]
        print(f"B shape: {b_shape} (no transpose)")

    # Create and quantize tensors
    torch.manual_seed(42)
    a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
    b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    print(f"\nMXFP8 tensor has:")
    print(f"  A rowwise: {a_mxfp8._rowwise_data.shape}, scales: {a_mxfp8._rowwise_scale_inv.shape}")
    print(f"  A columnwise: {a_mxfp8._columnwise_data.shape}, scales: {a_mxfp8._columnwise_scale_inv.shape}")
    print(f"  B rowwise: {b_mxfp8._rowwise_data.shape}, scales: {b_mxfp8._rowwise_scale_inv.shape}")
    print(f"  B columnwise: {b_mxfp8._columnwise_data.shape}, scales: {b_mxfp8._columnwise_scale_inv.shape}")

    print(f"\nTriton kernel needs:")
    print(f"  First operand: [{M}, {K}] with scales [{M}, {K//32}]")
    print(f"  Second operand: [{K}, {N}] with scales [{K//32}, {N}]")

    print(f"\nDirect selection for Triton (row-major):")

    # For A: Need [M, K] with scales [M, K//32]
    if transa:
        # A is [K, M], need [M, K]
        # A's columnwise is stored as [M, K] - but wait, that's wrong
        # Actually A's columnwise is [K, M] transposed = [M, K] in different layout
        # Let me check the actual storage

        # The columnwise storage transposes dimensions
        if a_mxfp8._columnwise_data.shape == (M, K):
            print(f"  A: Use columnwise (already [{M}, {K}])")
            a_choice = "columnwise"
        elif a_mxfp8._rowwise_data.shape == (K, M):
            print(f"  A: Need to transpose rowwise from [{K}, {M}] to [{M}, {K}]")
            a_choice = "rowwise.T"
        else:
            print(f"  A: Problem! Need [{M}, {K}]")
            a_choice = "?"
    else:
        # A is [M, K], already correct
        if a_mxfp8._rowwise_data.shape == (M, K):
            print(f"  A: Use rowwise (already [{M}, {K}])")
            a_choice = "rowwise"
        else:
            print(f"  A: Problem! Need [{M}, {K}]")
            a_choice = "?"

    # Check scale shape for A
    if a_choice == "rowwise" and a_mxfp8._rowwise_scale_inv.shape == (M, K//32):
        print(f"    ✓ A rowwise scales match: {a_mxfp8._rowwise_scale_inv.shape}")
    elif a_choice == "columnwise" and a_mxfp8._columnwise_scale_inv.shape == (M, K//32):
        print(f"    ✓ A columnwise scales match: {a_mxfp8._columnwise_scale_inv.shape}")
    elif a_choice == "rowwise.T":
        # Check if transposed scales would work
        transposed_scale_shape = (a_mxfp8._rowwise_scale_inv.shape[1], a_mxfp8._rowwise_scale_inv.shape[0])
        if transposed_scale_shape == (K//32, M):
            print(f"    ✗ A rowwise scales after transpose: {transposed_scale_shape} != [{M}, {K//32}]")
        else:
            print(f"    ? A scales: unclear")
    else:
        print(f"    ✗ A scales don't match required [{M}, {K//32}]")

    # For B: Need [K, N] with scales [K//32, N]
    if transb:
        # B is [N, K], need [K, N]
        if b_mxfp8._columnwise_data.shape == (K, N):
            print(f"  B: Use columnwise (already [{K}, {N}])")
            b_choice = "columnwise"
        elif b_mxfp8._rowwise_data.shape == (N, K):
            print(f"  B: Need to transpose rowwise from [{N}, {K}] to [{K}, {N}]")
            b_choice = "rowwise.T"
        else:
            print(f"  B: Problem! Need [{K}, {N}]")
            b_choice = "?"
    else:
        # B is [K, N], already correct
        if b_mxfp8._rowwise_data.shape == (K, N):
            print(f"  B: Use rowwise (already [{K}, {N}])")
            b_choice = "rowwise"
        elif b_mxfp8._columnwise_data.shape == (N, K):
            # Columnwise stores it transposed
            print(f"  B: Use columnwise.T to get [{K}, {N}] from [{N}, {K}]")
            b_choice = "columnwise.T"
        else:
            print(f"  B: Problem! Need [{K}, {N}]")
            b_choice = "?"

    # Check scale shape for B
    if b_choice == "rowwise" and b_mxfp8._rowwise_scale_inv.shape == (K, N//32):
        print(f"    ✗ B rowwise scales: {b_mxfp8._rowwise_scale_inv.shape} != [{K//32}, {N}]")
    elif b_choice == "columnwise" and b_mxfp8._columnwise_scale_inv.shape == (K//32, N):
        print(f"    ✓ B columnwise scales match: {b_mxfp8._columnwise_scale_inv.shape}")
    elif b_choice == "columnwise.T":
        # Columnwise is [N, K] with scales [N//32, K]
        # After transpose: [K, N] with scales... still [N//32, K]?
        print(f"    ✗ B columnwise.T scales: would be [{N//32}, {K}] != [{K//32}, {N}]")
    elif b_choice == "rowwise.T":
        # Rowwise is [N, K] with scales [N, K//32]
        # After transpose: [K, N] with scales... [K//32, N]? No, that's wrong
        print(f"    ✗ B rowwise.T scales: would be wrong after transpose")
    else:
        print(f"    ? B scales: unclear")

    print()

# Test all layouts
analyze_layout(False, False)  # NN
analyze_layout(False, True)   # NT
analyze_layout(True, False)   # TN
# analyze_layout(True, True)  # TT (not supported)