# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Triton MXFP8 GEMM kernel and wrapper tests."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from transformer_engine.pytorch import torch_version

if torch_version() < (2, 10):
    pytest.skip(
        f"MXFP8 Triton kernel requires PyTorch >= 2.10 (found {torch_version()}); "
        "earlier versions hit a tl.dot_scaled() RHS-scale compiler bug producing NaNs.",
        allow_module_level=True,
    )


def test_mxfp8_imports():
    """MXFP8 classes are importable from the expected paths."""
    from transformer_engine.pytorch.triton_kernels.gemm import te_generic_gemm_triton  # noqa: F401
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor  # noqa: F401
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage  # noqa: F401


def test_mxfp8_wrapper_regular_tensor():
    """MXFP8TensorWrapper accepts a plain (non-MXFP8) tensor and reports is_mxfp8=False."""
    from transformer_engine.pytorch.triton_kernels.gemm import MXFP8TensorWrapper

    A_fp32 = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    wrapper = MXFP8TensorWrapper(A_fp32)

    assert wrapper.is_mxfp8 is False
    assert wrapper.size() == A_fp32.size()


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
    """mxfp8_matmul() runs end-to-end on simulated FP8 data + E8M0 scales."""
    from transformer_engine.pytorch.triton_kernels.gemm import mxfp8_matmul
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

    M, N, K = 128, 256, 512
    VEC_SIZE = MXFP8_BLOCK_SCALING_SIZE  # 32

    torch.manual_seed(42)
    A_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32) * 0.1
    B_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32) * 0.1

    # Quantize through the architecture-native E4M3 dtype, then view as uint8.
    # An int8->uint8 reinterpret cast can produce 0x7F/0xFF bytes which are
    # NaN encodings under OCP e4m3fn (gfx950), poisoning the whole
    # accumulator; routing through the FP8 dtype avoids that.
    e4m3 = _get_e4m3_dtype()
    A_fp8 = A_fp32.to(e4m3).view(torch.uint8)
    B_fp8 = B_fp32.to(e4m3).view(torch.uint8)

    # E8M0 scales (uint8 biased exponents). Constant 127 -> scale = 2^0 = 1.0.
    A_scale = torch.full((M, K // VEC_SIZE), 127, dtype=torch.uint8, device='cuda')
    B_scale = torch.full((K // VEC_SIZE, N), 127, dtype=torch.uint8, device='cuda')

    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    mxfp8_matmul(
        A_fp8, A_scale,
        B_fp8, B_scale,
        C, M, N, K,
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E4M3,
    )

    assert C.shape == (M, N)
    assert C.abs().max() > 0, "kernel produced an all-zero output"
