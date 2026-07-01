#!/usr/bin/env python3
"""Direct test of MXFP8 kernel (without full quantization)"""

import torch
import sys
import pytest

print("Testing MXFP8 kernel directly...")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from transformer_engine.pytorch import torch_version
if torch_version() < (2, 10):
    pytest.skip(
        f"MXFP8 Triton kernel requires PyTorch >= 2.10 (found {torch_version()}); "
        "earlier versions hit a tl.dot_scaled() RHS-scale compiler bug producing NaNs.",
        allow_module_level=True,
    )


def _get_e4m3_dtype():
    """Return the E4M3 dtype native to the current architecture.

    gfx950 (compute cap >= 9.5) uses OCP torch.float8_e4m3fn where the bit
    patterns 0x7F/0xFF are NaN; gfx942 and earlier use NANOO
    torch.float8_e4m3fnuz where only 0x80 is NaN. Quantizing through the
    right dtype keeps the uint8 payload free of NaN encodings.
    """
    major, minor = torch.cuda.get_device_capability()
    if major == 9 and minor >= 5:
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def test_mxfp8_kernel_with_simulated_data():
    """Test MXFP8 kernel with simulated FP8 data and E8M0 scales"""
    try:
        # Import our kernel
        from transformer_engine.pytorch.gemm_triton import mxfp8_matmul
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

        print(f"✓ Imports successful")
        print(f"  MXFP8_BLOCK_SCALING_SIZE = {MXFP8_BLOCK_SCALING_SIZE}")

        # Create simple test inputs (simulate MXFP8 format)
        M, N, K = 128, 256, 512
        VEC_SIZE = MXFP8_BLOCK_SCALING_SIZE  # 32

        # Create FP8 data (as uint8) - we'll use random values
        # In real MXFP8, this would be quantized FP8 data
        torch.manual_seed(42)

        # Create FP32 data first
        A_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32) * 0.1
        B_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32) * 0.1

        # Quantize through the architecture-native E4M3 dtype, then view as
        # uint8. An int8→uint8 reinterpret cast can produce 0x7F/0xFF bytes
        # which are NaN encodings under OCP e4m3fn (gfx950), poisoning the
        # whole accumulator; routing through the FP8 dtype avoids that.
        e4m3 = _get_e4m3_dtype()
        A_fp8 = A_fp32.to(e4m3).view(torch.uint8)
        B_fp8 = B_fp32.to(e4m3).view(torch.uint8)

        # Create E8M0 scales (uint8 biased exponents)
        # For testing, use constant scales: exponent = 127 means scale = 2^0 = 1.0
        A_scale = torch.full((M, K // VEC_SIZE), 127, dtype=torch.uint8, device='cuda')
        B_scale = torch.full((K // VEC_SIZE, N), 127, dtype=torch.uint8, device='cuda')

        # Output tensor
        C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        print(f"✓ Created test tensors:")
        print(f"  A_fp8: {A_fp8.shape}, dtype={A_fp8.dtype}")
        print(f"  A_scale: {A_scale.shape}, dtype={A_scale.dtype}")
        print(f"  B_fp8: {B_fp8.shape}, dtype={B_fp8.dtype}")
        print(f"  B_scale: {B_scale.shape}, dtype={B_scale.dtype}")
        print(f"  C: {C.shape}, dtype={C.dtype}")

        # Try calling the wrapper
        print("\nCalling mxfp8_matmul...")
        mxfp8_matmul(
            A_fp8, A_scale,
            B_fp8, B_scale,
            C, M, N, K,
            tex.DType.kFloat8E4M3,  # A format
            tex.DType.kFloat8E4M3   # B format
        )
        print("✓ mxfp8_matmul executed without errors!")
        print(f"  Output shape: {C.shape}")
        print(f"  Output range: [{C.min():.6f}, {C.max():.6f}]")
        print(f"  Output mean: {C.mean():.6f}")

        # Check if output is non-zero (basic sanity)
        assert C.abs().max() > 0, "Output should contain non-zero values"
        print("✓ Output contains non-zero values (kernel produced results)")

        # Check output shape
        assert C.shape == (M, N), f"Expected output shape ({M}, {N}), got {C.shape}"

    except Exception as e:
        pytest.fail(f"Kernel execution failed: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running MXFP8 Kernel Direct Test")
    print("="*60)

    test_mxfp8_kernel_with_simulated_data()

    print("\n" + "="*60)
    print("MXFP8 KERNEL TEST PASSED!")
    print("="*60)
    print("\nNote: This test uses simulated FP8 data (not real quantization).")
    print("For full testing, use actual MXFP8Tensor with proper quantization.")
