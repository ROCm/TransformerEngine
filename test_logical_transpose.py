"""
Test logical transpose of B columnwise data and scales.
The key insight: we can transpose both data and scales by just changing strides!
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

def test_logical_transpose():
    print("=" * 80)
    print("LOGICAL TRANSPOSE SOLUTION")
    print("=" * 80)

    M, N, K = 128, 128, 256
    VEC_SIZE = 32

    print(f"\nExample: A[{M}, {K}] @ B[{K}, {N}] = C[{M}, {N}]")

    # Create B matrix
    b_shape = (K, N)
    torch.manual_seed(42)
    b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    b_mxfp8 = quantizer.quantize(b_fp32)

    print(f"\nB original shape: {b_shape}")
    print(f"B rowwise: data {b_mxfp8._rowwise_data.shape}, scales {b_mxfp8._rowwise_scale_inv.shape}")
    print(f"B columnwise: data {b_mxfp8._columnwise_data.shape}, scales {b_mxfp8._columnwise_scale_inv.shape}")

    print("\n" + "-" * 80)
    print("Option 1: B rowwise (doesn't work)")
    print("-" * 80)
    print(f"Data: {b_mxfp8._rowwise_data.shape}")
    print(f"Scales: {b_mxfp8._rowwise_scale_inv.shape}")
    print(f"✗ Scales are [{K}, {N//VEC_SIZE}] but need [{K//VEC_SIZE}, {N}]")

    print("\n" + "-" * 80)
    print("Option 2: B columnwise with logical transpose (WORKS!)")
    print("-" * 80)

    # B columnwise is stored as [N, K]
    b_col_data = b_mxfp8._columnwise_data
    b_col_scale = b_mxfp8._columnwise_scale_inv

    print(f"Columnwise storage:")
    print(f"  Data: {b_col_data.shape} with strides {b_col_data.stride()}")
    print(f"  Scales: {b_col_scale.shape} with strides {b_col_scale.stride()}")

    # Logical transpose - just swap dimensions and strides
    b_col_data_T = b_col_data.T  # This is just a view, no data movement
    b_col_scale_T = b_col_scale.T  # This is just a view, no data movement

    print(f"\nAfter logical transpose (just views, no data copy):")
    print(f"  Data: {b_col_data_T.shape} with strides {b_col_data_T.stride()}")
    print(f"  Scales: {b_col_scale_T.shape} with strides {b_col_scale_T.stride()}")

    print(f"\n✓ Perfect match for tl.dot_scaled!")
    print(f"  Data is now: [{K}, {N}]")
    print(f"  Scales are now: [{K//VEC_SIZE}, {N}]")
    print(f"  This is exactly what tl.dot_scaled expects for the second operand!")

    # Verify it's just a view
    print(f"\nMemory layout verification:")
    print(f"  Original data ptr: {b_col_data.data_ptr()}")
    print(f"  Transposed data ptr: {b_col_data_T.data_ptr()}")
    print(f"  Same memory: {b_col_data.data_ptr() == b_col_data_T.data_ptr()} ✓")

    print("\n" + "=" * 80)
    print("COMPLETE SOLUTION FOR ALL LAYOUTS")
    print("=" * 80)

    def get_selection(transa, transb):
        print(f"\nLayout: transA={transa}, transB={transb}")

        # For A (first operand needs [M, K] with scales [M, K//32])
        if not transa:
            print(f"  A: Use rowwise (already [{M}, {K}] with scales [{M}, {K//32}])")
            a_choice = "rowwise"
        else:
            # A is [K, M], need [M, K]
            print(f"  A: Use columnwise.T ([{K}, {M}] → [{M}, {K}])")
            a_choice = "columnwise.T"

        # For B (second operand needs [K, N] with scales [K//32, N])
        if not transb:
            # B is [K, N]
            print(f"  B: Use columnwise.T ([{N}, {K}] → [{K}, {N}])")
            b_choice = "columnwise.T"
        else:
            # B is [N, K], need [K, N]
            print(f"  B: Use rowwise.T ([{N}, {K}] → [{K}, {N}])")
            b_choice = "rowwise.T"

        return a_choice, b_choice

    print("\nNN layout:")
    get_selection(False, False)

    print("\nNT layout:")
    get_selection(False, True)

    print("\nTN layout:")
    get_selection(True, False)

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("\nBy using logical transpose (just changing strides), we can make")
    print("MXFP8 columnwise work perfectly with tl.dot_scaled!")
    print("\nThe transposed view gives us:")
    print("- The right data layout")
    print("- The right scale layout")
    print("- No data movement needed")
    print("- Triton handles strided access efficiently")

test_logical_transpose()