#!/usr/bin/env python3
"""
Standalone reproducer for Triton tl.dot_scaled() RHS scale bug on AMD gfx950.

================================================================================
BUG SUMMARY
================================================================================

The Triton compiler generates an incorrect layout for the RHS (B) scale operand
of tl.dot_scaled() when lowering to the v_mfma_scale_f32_32x32x64_f8f6f4
instruction on gfx950. The layout maps N-column index bits to the register
dimension instead of the lane dimension, causing most B scale values to be
silently dropped.

================================================================================
SYMPTOMS
================================================================================

  - Uniform B scales (all same value):       CORRECT results
  - Varying A scales with uniform B:         CORRECT results
  - Varying B scales (per-column or per-K):  WRONG results (50-80% mismatch)
  - The bug follows the operand position (always RHS/2nd operand of dot_scaled)

================================================================================
ROOT CAUSE ANALYSIS
================================================================================

The v_mfma_scale_f32_32x32x64_f8f6f4 instruction on gfx950 performs a
block-scaled FP8 matrix multiply with microscaling (MX) support:

    C[32x32] = scale_A * A[32x64] @ scale_B * B[64x32]

Each wavefront lane provides ONE byte for scale_A and ONE byte for scale_B
via the scale VGPR registers. With 64 lanes per wavefront, this gives exactly
64 scale bytes per operand -- matching the 64 unique scales needed (32 spatial
positions x 2 K-blocks of 32 elements each, with VEC=32).

The Triton compiler must generate a layout that maps each lane to the correct
scale element. Examining the Triton GPU IR (TTGIR) for this kernel reveals:

    tt.dot_scaled %lhs scale %a_scale, %rhs scale %b_scale, %acc
        lhs = e4m3 rhs = e4m3
      : tensor<32x64xf8E4M3FN, #dot_op<{opIdx=0}>>,
        tensor<32x2xi8, #linear>            <-- A scale layout
      * tensor<64x32xf8E4M3FN, #dot_op<{opIdx=1}>>,
        tensor<32x2xi8, #linear2>           <-- B scale layout (BUGGY)

Note: In the current API, both scale tensors use [spatial, K//32] layout:
  a_scale: [M, K//32] = [32, 2]
  b_scale: [N, K//32] = [32, 2]  (NOT transposed -- the API docs say
           "Do NOT transpose rhs_scale")

The two scale layouts are:

  A scale (#linear) -- tensor<32x2xi8>:
    register = []
    lane = [[1,0], [2,0], [4,0], [8,0], [16,0], [0,1]]

    Lane bit 0 (laneId & 1)  -> row += 1    (M dimension)
    Lane bit 1 (laneId & 2)  -> row += 2
    Lane bit 2 (laneId & 4)  -> row += 4
    Lane bit 3 (laneId & 8)  -> row += 8
    Lane bit 4 (laneId & 16) -> row += 16
    Lane bit 5 (laneId & 32) -> col += 1    (K-block dimension)

    Result: 5 lane bits -> 32 M-row indices, 1 lane bit -> 2 K-block indices
    Total: 64 unique per-lane addresses.  CORRECT.

  B scale (#linear2) -- tensor<32x2xi8>:
    register = [[2,0], [4,0], [8,0], [16,0]]
    lane     = [[1,0], [0,0], [0,0], [0,0], [0,0], [0,1]]

    Lane bit 0 (laneId & 1)  -> row += 1    (N dimension, 1 bit only!)
    Lane bit 1 (laneId & 2)  -> [0,0]       NO CONTRIBUTION
    Lane bit 2 (laneId & 4)  -> [0,0]       NO CONTRIBUTION
    Lane bit 3 (laneId & 8)  -> [0,0]       NO CONTRIBUTION
    Lane bit 4 (laneId & 16) -> [0,0]       NO CONTRIBUTION
    Lane bit 5 (laneId & 32) -> col += 1    (K-block dimension)

    Register bit 0 -> row += 2              (N dimension)
    Register bit 1 -> row += 4
    Register bit 2 -> row += 8
    Register bit 3 -> row += 16

    Result: 1 lane bit -> 2 N indices, 1 lane bit -> 2 K-block indices
    Total: only 4 unique per-lane addresses!  WRONG -- should be 64.

    The remaining 4 bits of the N-column index (needed to address 32 rows
    of the [N, K//32] scale tensor) are assigned to the register dimension.
    Registers are thread-private -- all 16 register slots within a lane read
    the same byte from the scale VGPR, so these bits are invisible to the
    hardware.

This is confirmed in the LLVM IR. The B scale address computation reduces to:

    b_scale_addr = b_scale_ptr + (K//32) * (threadId & 1) + (threadId >> 5) & 1

which yields only 4 unique addresses across 64 lanes:
    Lane 0:  b_scale[0, 0]     Lane 32: b_scale[0, 1]
    Lane 1:  b_scale[1, 0]     Lane 33: b_scale[1, 1]
    Lane 2:  b_scale[0, 0]     Lane 34: b_scale[0, 1]   <- same as lane 0/32
    ...                         ...

In contrast, the A scale address computation correctly uses all 6 lane bits:

    a_scale_addr = a_scale_ptr + (K//32) * (threadId & 31) + (threadId >> 5) & 1

yielding 64 unique addresses (32 rows x 2 K-blocks).

================================================================================
FIX
================================================================================

The B scale layout should use lane bits (not register bits) for the N-row
index, similar to how the A scale layout uses lane bits for the M-row index:

  Correct B scale layout should be:
    register = []
    lane = [[1,0], [2,0], [4,0], [8,0], [16,0], [0,1]]

    Lane bit 0 -> row += 1   (N dimension)
    Lane bit 1 -> row += 2
    Lane bit 2 -> row += 4
    Lane bit 3 -> row += 8
    Lane bit 4 -> row += 16
    Lane bit 5 -> col += 1   (K-block, 2 values)

    Result: 5 lane bits -> 32 N-rows, 1 lane bit -> 2 K-blocks = 64 unique.

The bug is in the Triton compiler's layout selection pass for DotScaledOp scale
operands (likely in TritonGPUToLLVM/DotOpToLLVM/MFMA.cpp or in the layout
inference pass that assigns #linear/#linear2 to scale tensors).

================================================================================
ENVIRONMENT
================================================================================

  Tested on: AMD Instinct MI355X (gfx950)
  PyTorch:   2.10.0.dev20251112+rocm7.1
  Triton:    3.6.0+gitb94dfe8c

  Dependencies: torch, triton (no TransformerEngine needed)

Usage:
  python triton_dot_scaled_rhs_scale_bug.py
"""

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def mxfp8_dot_scaled_kernel(
    a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Minimal tl.dot_scaled kernel: C[M,N] = dot_scaled(A[M,K], B[K,N]) with E8M0 scales."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load FP8 data as uint8
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

    # Load E8M0 scales: a_scale[M, K//32], b_scale[N, K//32]
    # Note: In current API, both scales use [spatial_dim, K//scale_factor] layout.
    # rhs_scale is [N, K//32], NOT [K//32, N]. The API docs say "Do NOT transpose rhs_scale".
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_BLOCK_K)
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * SCALE_BLOCK_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * SCALE_BLOCK_K + offs_scale_k[None, :])

    # Block-scaled FP8 matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def compute_reference(a_fp8, b_fp8, a_scale, b_scale, K):
    """Software reference: apply E8M0 block scales and compute matmul in float32.

    a_scale: [M, K//32], b_scale: [N, K//32]
    a_fp8: [M, K], b_fp8: [K, N]
    """
    # a_scale [M, K//32] -> [M, K] by repeating each scale 32 times along K
    a_sf = (2.0 ** (a_scale.float() - 127.0)).repeat_interleave(32, dim=1)[:, :K]
    # b_scale [N, K//32] -> [N, K] -> transpose to [K, N] to match b_fp8 shape
    b_sf = (2.0 ** (b_scale.float() - 127.0)).repeat_interleave(32, dim=1)[:, :K].T
    return torch.matmul(a_fp8.float() * a_sf, b_fp8.float() * b_sf)


def run_kernel(a_fp8, b_fp8, a_scale, b_scale, M, K, N):
    """Run the Triton dot_scaled kernel."""
    c = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
    mxfp8_dot_scaled_kernel[(1,)](
        a_fp8.view(torch.uint8), a_scale, b_fp8.view(torch.uint8), b_scale, c,
        M, N, K, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
    )
    return c


def run_test(label, a_scale, b_scale, a_fp8, b_fp8, M, K, N):
    """Run one test case and report results."""
    c = run_kernel(a_fp8, b_fp8, a_scale, b_scale, M, K, N)
    ref = compute_reference(a_fp8, b_fp8, a_scale, b_scale, K)
    diff = (c.float() - ref.float()).abs()
    mismatch = (diff > 0.05).sum().item() / diff.numel() * 100
    status = "PASS" if mismatch < 1.0 else "FAIL"
    print(f"  [{status}] {label:45s}: max_diff={diff.max().item():.4f}, mismatch@0.05={mismatch:.1f}%")
    return status == "PASS"


def main():
    # Print environment info
    print(f"PyTorch:  {torch.__version__}")
    print(f"Triton:   {triton.__version__}")
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability()
    print(f"Compute:  {major}.{minor}")
    print()

    if major < 9 or (major == 9 and minor < 5):
        print("SKIP: tl.dot_scaled requires gfx950 (compute capability >= 9.5)")
        sys.exit(0)

    M, K, N = 32, 64, 32
    torch.manual_seed(42)
    a_fp8 = (torch.randn(M, K, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    b_fp8 = (torch.randn(K, N, device='cuda') * 0.1).to(torch.float8_e4m3fn)

    all_pass = True

    # =========================================================================
    # Test 1: Both scales uniform -- should always pass (baseline sanity check)
    # =========================================================================
    print("Test group 1: Uniform scales (sanity check)")
    all_pass &= run_test(
        "Both unit (127 = scale 1.0)",
        torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda'),
        torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda'),
        a_fp8, b_fp8, M, K, N,
    )
    all_pass &= run_test(
        "Both scale=2.0 (128)",
        torch.full((M, K//32), 128, dtype=torch.uint8, device='cuda'),
        torch.full((N, K//32), 128, dtype=torch.uint8, device='cuda'),
        a_fp8, b_fp8, M, K, N,
    )

    # =========================================================================
    # Test 2: LHS (A) scales vary, RHS (B) uniform -- should pass
    # =========================================================================
    print("\nTest group 2: LHS (A) scales vary, RHS (B) uniform")
    torch.manual_seed(99)
    a_scale_vary = torch.randint(124, 131, (M, K//32), dtype=torch.uint8, device='cuda')
    b_scale_unit = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
    all_pass &= run_test(
        "A random [124-130], B unit",
        a_scale_vary, b_scale_unit, a_fp8, b_fp8, M, K, N,
    )

    # A varies per M-row
    a_s = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')
    a_s[:16, :] = 126; a_s[16:, :] = 128
    all_pass &= run_test(
        "A varies per M-row, B unit",
        a_s, b_scale_unit, a_fp8, b_fp8, M, K, N,
    )

    # A varies per K-block
    a_s2 = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')
    a_s2[:, 0] = 126; a_s2[:, 1] = 128
    all_pass &= run_test(
        "A varies per K-block, B unit",
        a_s2, b_scale_unit, a_fp8, b_fp8, M, K, N,
    )

    # =========================================================================
    # Test 3: RHS (B) scales vary -- THIS IS THE BUG
    # =========================================================================
    print("\nTest group 3: RHS (B) scales vary (BUG: these should pass but don't)")
    a_scale_unit = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')

    # B varies per N-row (same across K-blocks)
    # b_scale shape is [N, K//32] in current API
    b_s = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
    b_s[0, :] = 128; b_s[1, :] = 129  # just N-rows 0 and 1 differ
    all_pass &= run_test(
        "A unit, B varies per N-row (rows 0,1 differ)",
        a_scale_unit, b_s, a_fp8, b_fp8, M, K, N,
    )

    # B varies per K-block (same across N)
    b_s2 = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
    b_s2[:, 0] = 126; b_s2[:, 1] = 128
    all_pass &= run_test(
        "A unit, B varies per K-block",
        a_scale_unit, b_s2, a_fp8, b_fp8, M, K, N,
    )

    # B random scales
    torch.manual_seed(99)
    b_scale_vary = torch.randint(124, 131, (N, K//32), dtype=torch.uint8, device='cuda')
    all_pass &= run_test(
        "A unit, B random [124-130]",
        a_scale_unit, b_scale_vary, a_fp8, b_fp8, M, K, N,
    )

    # Both random
    all_pass &= run_test(
        "Both random [124-130]",
        a_scale_vary, b_scale_vary, a_fp8, b_fp8, M, K, N,
    )

    # =========================================================================
    # Test 4: Operand swap -- proves bug follows operand position
    # =========================================================================
    print("\nTest group 4: Swapped operands (C^T = B^T @ A^T)")
    b_fp8_T = b_fp8.T.contiguous().to(torch.float8_e4m3fn)  # [N, K]
    a_fp8_T = a_fp8.T.contiguous().to(torch.float8_e4m3fn)  # [K, M]

    # Swapped: LHS=B^T[N,K], RHS=A^T[K,M]
    # LHS scale: [N, K//32], RHS scale: [M, K//32]
    # Original b_scale_vary is [N, K//32] -- use directly as LHS scale
    bt_scale_vary = b_scale_vary  # already [N, K//32]
    at_scale_unit = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')
    all_pass &= run_test(
        "Swapped: orig-B as LHS(vary), orig-A as RHS(unit)",
        bt_scale_vary, at_scale_unit, b_fp8_T, a_fp8_T, N, K, M,
    )

    # Original A's scales as RHS (should break when A is 2nd operand)
    # a_scale_vary is [M, K//32] -- use as RHS scale (shape [M, K//32] is correct
    # since RHS=A^T[K,M] so RHS scale should be [M, K//32])
    bt_scale_unit = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
    at_scale_vary = a_scale_vary  # already [M, K//32]
    all_pass &= run_test(
        "Swapped: orig-B as LHS(unit), orig-A as RHS(vary)",
        bt_scale_unit, at_scale_vary, b_fp8_T, a_fp8_T, N, K, M,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    if all_pass:
        print("ALL TESTS PASSED -- bug may be fixed in this Triton version!")
    else:
        print("FAILURES DETECTED -- tl.dot_scaled() RHS scale bug confirmed.")
        print()
        print("Root cause: The Triton compiler's #linear2 layout for the RHS scale")
        print("operand maps N-row index bits to the register dimension instead of")
        print("the lane dimension. The v_mfma_scale_f32_32x32x64_f8f6f4 instruction")
        print("reads one scale byte per lane, so register-level distinctions are")
        print("invisible to the hardware. Only 4 unique RHS scale values are loaded")
        print("(2 N-rows x 2 K-blocks) instead of the needed 64 (32 x 2).")
        print()
        print("See the docstring at the top of this file for full IR-level analysis.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
