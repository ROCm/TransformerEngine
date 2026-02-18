"""
Pragmatic approach: Just use what we have and see what gives the best results.
For each layout, try different combinations and check accuracy.
"""

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

def find_best_selection(transa, transb, M=128, N=128, K=256):
    print("=" * 60)
    print(f"Layout: transA={transa}, transB={transb}")
    print("=" * 60)

    # Create input shapes
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
    a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device) * 0.1
    b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device) * 0.1

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    a_mxfp8 = quantizer.quantize(a_fp32)
    b_mxfp8 = quantizer.quantize(b_fp32)

    # Compute reference
    a_for_ref = a_fp32.T if transa else a_fp32
    b_for_ref = b_fp32.T if transb else b_fp32
    ref = torch.matmul(a_for_ref, b_for_ref)

    print(f"\nBest selection for Triton:")

    # For MXFP8, we have limited options that make sense
    # The key constraint is that scales must match data dimensions

    # Option 1: Both use rowwise (if shapes allow)
    if not transa and not transb:
        # A[M,K] @ B[K,N]
        # A rowwise: [M,K] with scales [M, K//32]
        # B rowwise: [K,N] with scales [K, N//32]
        print(f"  Option: A rowwise + B rowwise")
        print(f"    A: {a_mxfp8._rowwise_data.shape} with scales {a_mxfp8._rowwise_scale_inv.shape}")
        print(f"    B: {b_mxfp8._rowwise_data.shape} with scales {b_mxfp8._rowwise_scale_inv.shape}")
        print(f"    Note: B scales don't match tl.dot_scaled expectation")

    # Option 2: A rowwise + B columnwise (if available)
    if not transa:
        # A needs [M,K], B needs [K,N]
        # Check if B columnwise can give us [K,N]
        if b_mxfp8._columnwise_data.shape[0] == K:
            # Columnwise is stored transposed, but wrong shape
            pass
        elif not transb and b_mxfp8._columnwise_data.shape == (N, K):
            # B columnwise is [N,K], we need [K,N]
            print(f"  Option: A rowwise + B columnwise.T")
            print(f"    A: {a_mxfp8._rowwise_data.shape} with scales {a_mxfp8._rowwise_scale_inv.shape}")
            print(f"    B.T: [{K},{N}] from columnwise [{N},{K}]")
            print(f"    But B scales would be wrong after transpose")

    # The reality check
    print(f"\nReality:")
    print(f"  tl.dot_scaled has specific requirements that MXFP8 doesn't naturally meet")
    print(f"  We may need to:")
    print(f"  1. Use manual dequantization instead of tl.dot_scaled")
    print(f"  2. Modify the kernel to handle MXFP8's actual scale layouts")
    print(f"  3. Accept that some layouts won't work well with tl.dot_scaled")
    print()

# Test all layouts
find_best_selection(False, False)  # NN
find_best_selection(False, True)   # NT
find_best_selection(True, False)   # TN