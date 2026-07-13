# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""MXFP8 wrapper sanity checks for the Triton GEMM backend.

Numerical correctness for the MXFP8 kernel is covered end-to-end by
``triton_kernels/test_gemm.py::test_triton_vs_pytorch_mxfp8`` and
``::test_triton_vs_cpp_mxfp8`` (they exercise ``general_gemm`` under
``NVTE_USE_GEMM_TRITON=1`` with a real ``MXFP8Tensor`` and both PyTorch
and C++-backend references).

This file only holds narrow wrapper-layer sanity: imports resolve, and
``MXFP8TensorWrapper`` handles a non-MXFP8 tensor correctly.
"""

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
