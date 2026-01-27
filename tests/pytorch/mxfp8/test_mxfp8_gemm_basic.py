#!/usr/bin/env python3
"""Test MXFP8 GEMM implementation - Basic wrapper and import tests"""

import torch
import sys
import pytest

print("Testing MXFP8 GEMM implementation...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def test_mxfp8_imports():
    """Test that MXFP8 classes can be imported"""
    try:
        from transformer_engine.pytorch.gemm_triton import te_generic_gemm_triton
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
        from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
        print("✓ Successfully imported MXFP8 classes")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_mxfp8_wrapper_regular_tensor():
    """Test MXFP8TensorWrapper with regular tensors"""
    try:
        from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper

        # Create simple test tensor
        A_fp32 = torch.randn(128, 512, device='cuda', dtype=torch.float32)

        # Test wrapping a regular tensor
        wrapper = MXFP8TensorWrapper(A_fp32)
        print(f"✓ MXFP8TensorWrapper created for regular tensor")
        print(f"  - is_mxfp8: {wrapper.is_mxfp8}")
        print(f"  - size: {wrapper.size()}")

        assert wrapper.is_mxfp8 == False, "Regular tensor should not be detected as MXFP8"
        assert wrapper.size() == A_fp32.size(), "Size should match original tensor"

    except Exception as e:
        pytest.fail(f"MXFP8TensorWrapper test failed: {e}")


def test_basic_fp32_gemm():
    """Test basic FP32 GEMM for reference"""
    try:
        M, N, K = 128, 256, 512

        A_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32)

        print(f"✓ Created test tensors: A={A_fp32.shape}, B={B_fp32.shape}")

        # Compute reference
        C_ref = torch.matmul(A_fp32, B_fp32)
        print(f"✓ Computed FP32 reference: C={C_ref.shape}")

        assert C_ref.shape == (M, N), f"Expected shape ({M}, {N}), got {C_ref.shape}"

    except Exception as e:
        pytest.fail(f"Tensor creation failed: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running MXFP8 Basic Tests")
    print("="*60)

    test_mxfp8_imports()
    test_mxfp8_wrapper_regular_tensor()
    test_basic_fp32_gemm()

    print("\n" + "="*60)
    print("BASIC TESTS PASSED!")
    print("="*60)
    print("\nNote: Full MXFP8 GEMM test requires MXFP8Quantizer,")
    print("which may need to be tested separately with actual")
    print("MXFP8Tensor instances.")
