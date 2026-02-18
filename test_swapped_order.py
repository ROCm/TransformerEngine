"""
What if we compute B @ A instead of A @ B, then transpose the result?
Since (A @ B)^T = B^T @ A^T, we might be able to find a better match.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

def analyze_swapped(transa, transb, M=128, N=128, K=256):
    print("=" * 60)
    print(f"Original: transA={transa}, transB={transb}")
    print(f"Want: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")
    print("=" * 60)

    # Could we compute D = B^T @ A^T instead, then transpose?
    # D[N,M] = B^T[N,K] @ A^T[K,M]
    # Then C = D^T

    print(f"\nAlternative computation:")
    print(f"Compute: D[{N},{M}] = B^T[{N},{K}] @ A^T[{K},{M}]")
    print(f"Then: C = D^T to get [{M},{N}]")

    # For this alternative, Triton kernel would need:
    print(f"\nTriton kernel would need:")
    print(f"  First operand: [{N}, {K}] with scales [{N}, {K//32}]")
    print(f"  Second operand: [{K}, {M}] with scales [{K//32}, {M}]")

    # Create the input shapes based on transpose flags
    if transa:
        a_shape = (K, M)
    else:
        a_shape = (M, K)

    if transb:
        b_shape = (N, K)
    else:
        b_shape = (K, N)

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

    print(f"\nMXFP8 available:")
    print(f"A ({a_shape}):")
    print(f"  rowwise: {a_mxfp8._rowwise_data.shape}, scales {a_mxfp8._rowwise_scale_inv.shape}")
    print(f"  columnwise: {a_mxfp8._columnwise_data.shape}, scales {a_mxfp8._columnwise_scale_inv.shape}")
    print(f"B ({b_shape}):")
    print(f"  rowwise: {b_mxfp8._rowwise_data.shape}, scales {b_mxfp8._rowwise_scale_inv.shape}")
    print(f"  columnwise: {b_mxfp8._columnwise_data.shape}, scales {b_mxfp8._columnwise_scale_inv.shape}")

    print(f"\nSelection for swapped computation:")

    # Need B^T as first operand: [N, K] with scales [N, K//32]
    if transb:
        # B is already [N, K]
        if b_mxfp8._rowwise_data.shape == (N, K):
            print(f"  B^T: Use B rowwise (already [{N}, {K}]) with scales {b_mxfp8._rowwise_scale_inv.shape}")
            if b_mxfp8._rowwise_scale_inv.shape == (N, K//32):
                print(f"    ✓ Scales match!")
    else:
        # B is [K, N], need [N, K]
        print(f"  B^T: Need B transposed...")

    # Need A^T as second operand: [K, M] with scales [K//32, M]
    if transa:
        # A is already [K, M]
        if a_mxfp8._rowwise_data.shape == (K, M):
            print(f"  A^T: Use A rowwise (already [{K}, {M}]) with scales {a_mxfp8._rowwise_scale_inv.shape}")
            if a_mxfp8._rowwise_scale_inv.shape == (K, M//32):
                print(f"    ✗ Scales are [{K}, {M//32}] not [{K//32}, {M}]")
    else:
        # A is [M, K], need [K, M]
        print(f"  A^T: Need A transposed...")

    print()

# Test NT case which is common
analyze_swapped(False, True, 128, 128, 256)