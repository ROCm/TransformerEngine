"""
Find the correct MXFP8 data selection for Triton kernel.

The key insight: tl.dot_scaled expects specific scale patterns that may not
match exactly what MXFP8 provides. We need to find what works.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

def analyze_correct_selection(transa, transb, M=128, N=128, K=256):
    print("=" * 60)
    print(f"Layout: transA={transa}, transB={transb}")
    print("=" * 60)

    # Create the input shapes based on transpose flags
    if transa:
        a_shape = (K, M)
    else:
        a_shape = (M, K)

    if transb:
        b_shape = (N, K)
    else:
        b_shape = (K, N)

    print(f"A shape: {a_shape}")
    print(f"B shape: {b_shape}")

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

    print(f"\nFor Triton row-major kernel:")
    print(f"Need first operand: [{M}, {K}] with scales [{M}, {K//32}]")
    print(f"Need second operand: [{K}, {N}] with scales [{K//32}, {N}]")

    # The insight: For MXFP8, the scales are tied to the data orientation
    # We need to find combinations that give us the right scale patterns

    print(f"\nActual selection needed:")

    # First operand needs [M, K] with rowwise scaling (scales along K)
    if not transa:
        # A is already [M, K]
        print(f"  First operand: A rowwise [{M}, {K}] with scales [{M}, {K//32}]")
        first_data = a_mxfp8._rowwise_data
        first_scale = a_mxfp8._rowwise_scale_inv
    else:
        # A is [K, M], need to get [M, K] somehow
        # Option 1: Use A columnwise if it gives us [M, K]
        if a_mxfp8._columnwise_data.shape == (M, K):
            print(f"  First operand: A columnwise (gives [{M}, {K}])")
            first_data = a_mxfp8._columnwise_data
            first_scale = a_mxfp8._columnwise_scale_inv
        else:
            # Option 2: Transpose A rowwise
            print(f"  First operand: A rowwise.T [{K}, {M}] → [{M}, {K}]")
            first_data = a_mxfp8._rowwise_data.T
            first_scale = a_mxfp8._rowwise_scale_inv.T  # This is problematic!

    # Second operand needs [K, N] with columnwise scaling (scales along K)
    # But MXFP8 columnwise means scales along the first dimension in the transposed storage

    # For B: we need [K, N]
    if not transb:
        # B is already [K, N]
        # B rowwise would give [K, N] but with scales [K, N//32] (wrong!)
        # B columnwise is stored as [N, K] with scales [N//32, K]
        if b_mxfp8._columnwise_data.shape == (N, K):
            print(f"  Second operand: B columnwise.T [{N}, {K}] → [{K}, {N}]")
            print(f"    But scales would be [{N//32}, {K}] (wrong!)")
            second_data = b_mxfp8._columnwise_data.T
            second_scale = b_mxfp8._columnwise_scale_inv  # Wrong shape!
        else:
            print(f"  Second operand: B rowwise [{K}, {N}]")
            print(f"    But scales are [{K}, {N//32}] not [{K//32}, {N}]")
            second_data = b_mxfp8._rowwise_data
            second_scale = b_mxfp8._rowwise_scale_inv  # Wrong shape!
    else:
        # B is [N, K], need [K, N]
        # B columnwise might be stored as [K, N]?
        if b_mxfp8._columnwise_data.shape == (K, N):
            print(f"  Second operand: B columnwise (gives [{K}, {N}])")
            print(f"    Scales are {b_mxfp8._columnwise_scale_inv.shape}")
            second_data = b_mxfp8._columnwise_data
            second_scale = b_mxfp8._columnwise_scale_inv
        else:
            print(f"  Second operand: B rowwise.T [{N}, {K}] → [{K}, {N}]")
            print(f"    But scales transpose doesn't work right")
            second_data = b_mxfp8._rowwise_data.T
            second_scale = b_mxfp8._rowwise_scale_inv.T  # Wrong!

    print(f"\nConclusion:")
    print(f"  The fundamental issue is that tl.dot_scaled expects:")
    print(f"    - First operand with rowwise scaling (blocks along K)")
    print(f"    - Second operand with columnwise scaling (blocks along K)")
    print(f"  But MXFP8 provides:")
    print(f"    - Rowwise: blocks along the last dimension")
    print(f"    - Columnwise: blocks along the first dimension (in transposed storage)")
    print(f"  These don't always align with what tl.dot_scaled needs!")
    print()

# Test all layouts
analyze_correct_selection(False, False)  # NN
analyze_correct_selection(False, True)   # NT
analyze_correct_selection(True, False)   # TN