"""
Visual example of the scale mismatch with actual numbers.
"""

import torch
import numpy as np

def show_concrete_example():
    print("=" * 80)
    print("CONCRETE EXAMPLE: Why MXFP8 scales don't match tl.dot_scaled")
    print("=" * 80)

    # Tiny example for clarity
    M, N, K = 4, 4, 64  # K=64 so we have 2 blocks of 32
    VEC_SIZE = 32

    print(f"\nTiny example: A[{M}, {K}] @ B[{K}, {N}] = C[{M}, {N}]")
    print(f"K = {K} = 2 blocks of {VEC_SIZE}")

    print("\n" + "-" * 80)
    print("What tl.dot_scaled needs:")
    print("-" * 80)

    print("\nA[4, 64] with scales[4, 2]:")
    print("```")
    print("       block0 (K=0:32)    block1 (K=32:64)")
    print("row 0: [   ...data...  ] [   ...data...  ]")
    print("       scale_A[0,0]       scale_A[0,1]")
    print("")
    print("row 1: [   ...data...  ] [   ...data...  ]")
    print("       scale_A[1,0]       scale_A[1,1]")
    print("")
    print("row 2: [   ...data...  ] [   ...data...  ]")
    print("       scale_A[2,0]       scale_A[2,1]")
    print("")
    print("row 3: [   ...data...  ] [   ...data...  ]")
    print("       scale_A[3,0]       scale_A[3,1]")
    print("```")

    print("\nB[64, 4] with scales[2, 4]:")
    print("```")
    print("        col0    col1    col2    col3")
    print("block0  [....]  [....]  [....]  [....]  (K=0:32)")
    print("scale:  s[0,0]  s[0,1]  s[0,2]  s[0,3]")
    print("")
    print("block1  [....]  [....]  [....]  [....]  (K=32:64)")
    print("scale:  s[1,0]  s[1,1]  s[1,2]  s[1,3]")
    print("```")

    print("\n" + "-" * 80)
    print("What MXFP8 provides:")
    print("-" * 80)

    print("\nOption 1: B with MXFP8 rowwise")
    print("B[64, 4] with scales[64, 0] (4/32 rounds to 0, actually [64, 1] with padding):")
    print("```")
    print("        col0    col1    col2    col3")
    print("row 0:  [----all 4 columns use same scale----]  scale[0,0]")
    print("row 1:  [----all 4 columns use same scale----]  scale[1,0]")
    print("...")
    print("row 63: [----all 4 columns use same scale----]  scale[63,0]")
    print("```")
    print("✗ Each ROW has one scale, but tl.dot_scaled needs each COLUMN to have 2 scales!")

    print("\nOption 2: B with MXFP8 columnwise")
    print("B stored as [4, 64] (transposed!) with scales[4, 2]:")
    print("```")
    print("Original B column 0 is now row 0 in transposed storage:")
    print("       block0 (32 elems)  block1 (32 elems)")
    print("col0:  [   ...data...  ]  [   ...data...  ]")
    print("       scale[0,0]         scale[0,1]")
    print("")
    print("Original B column 1 is now row 1:")
    print("col1:  [   ...data...  ]  [   ...data...  ]")
    print("       scale[1,0]         scale[1,1]")
    print("```")
    print("✗ The data is transposed, and after untransposing, scales don't align right!")

    print("\n" + "-" * 80)
    print("The dot product computation:")
    print("-" * 80)

    print("\nTo compute C[0,0] = A[0,:] @ B[:,0]:")
    print("  = A[0,0:32] @ B[0:32,0] * scale_A[0,0] * scale_B[0,0]")
    print("  + A[0,32:64] @ B[32:64,0] * scale_A[0,1] * scale_B[1,0]")

    print("\nWhat we need:")
    print("  scale_B[0,0] = scale for B[0:32, 0]   (first K-block of column 0)")
    print("  scale_B[1,0] = scale for B[32:64, 0]  (second K-block of column 0)")

    print("\nWhat MXFP8 rowwise gives:")
    print("  scale[0,0] = scale for B[0, 0:4]   (all columns of row 0)")
    print("  scale[1,0] = scale for B[1, 0:4]   (all columns of row 1)")
    print("  ✗ Wrong dimension!")

    print("\nWhat MXFP8 columnwise gives (after untransposing):")
    print("  The transposed storage has the right scale shape [4, 2]")
    print("  But it's for the transposed data [4, 64]")
    print("  After untransposing to get [64, 4], the scales don't follow correctly")
    print("  ✗ Can't just transpose scales independently of data!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nThe core issue:")
    print("1. tl.dot_scaled expects scales along the K dimension for BOTH operands")
    print("2. MXFP8 rowwise scales along the last dimension (works for A, not for B)")
    print("3. MXFP8 columnwise scales along the first dimension of transposed data")
    print("4. There's no direct way to get columnwise scaling along K for B")

    print("\nThis is why the accuracy is poor - we're using incompatible scale layouts!")

show_concrete_example()