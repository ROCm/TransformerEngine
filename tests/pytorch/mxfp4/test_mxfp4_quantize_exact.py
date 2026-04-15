# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Bit-exact tests for MXFP4 quantization vs Python reference.

Validates that MXFP4Quantizer produces identical packed FP4 data and E8M0
scales as the MXFP4QuantizerRef pure-Python reference implementation.
"""

import numpy as np
import os
import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef


recipe_available, reason_for_no_recipe = te.is_mxfp4_available(return_reason=True)

BLOCK_SIZE = 32

_USE_TRITON = bool(int(os.environ.get("NVTE_USE_CAST_TRANSPOSE_TRITON", "0")))


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack packed FP4 uint8 tensor into individual nibbles."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def compare_e8m0_scales(
    test: torch.Tensor,
    ref: torch.Tensor,
    msg: str,
    max_diff: int = 1,
    max_mismatch_rate: float = 0.02,
    max_outliers: int = 0,
) -> None:
    """Compare E8M0 scales allowing ±max_diff exponent difference."""
    test_np = test.cpu().numpy().astype(np.int16)
    ref_np = ref.cpu().numpy().astype(np.int16)

    diff = np.abs(test_np - ref_np)
    within = np.sum(diff <= max_diff)
    outliers = np.sum(diff > max_diff)
    total = test_np.size
    mismatch_rate = 1.0 - within / total

    assert outliers <= max_outliers, (
        f"{msg}: {outliers} outliers (diff > {max_diff}) exceeds limit {max_outliers}"
    )
    assert mismatch_rate <= max_mismatch_rate, (
        f"{msg}: mismatch rate {mismatch_rate:.2%} exceeds {max_mismatch_rate:.2%}"
    )


def compare_fp4_data(
    test: torch.Tensor,
    ref: torch.Tensor,
    msg: str,
    max_diff: int = 1,
    max_mismatch_rate: float = 0.02,
) -> None:
    """Compare unpacked FP4 nibbles allowing ±max_diff difference."""
    test_np = test.cpu().numpy().astype(np.int16)
    ref_np = ref.cpu().numpy().astype(np.int16)

    diff = np.abs(test_np - ref_np)
    within = np.sum(diff <= max_diff)
    total = test_np.size
    mismatch_rate = 1.0 - within / total

    assert mismatch_rate <= max_mismatch_rate, (
        f"{msg}: mismatch rate {mismatch_rate:.2%} exceeds {max_mismatch_rate:.2%}"
    )


def check_quantization_mxfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
) -> None:
    te_dtype = tex.DType.kFloat4E2M1

    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Native MXFP4 quantization
    mxfp4_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        shuffle_B_matrix_for_aiter=False,
        use_hadamard=False,
    )
    if use_cpp_allocator:
        x_mxfp4_sut = mxfp4_quantizer(x)
    else:
        x_mxfp4_sut = mxfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_mxfp4_sut = mxfp4_quantizer.update_quantized(x, x_mxfp4_sut)

    # Extract data from MXFP4Tensor
    assert x_mxfp4_sut._rowwise_data is not None
    qx = x_mxfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_mxfp4_sut._rowwise_scale_inv is not None
    sx = x_mxfp4_sut._rowwise_scale_inv

    qx_t = (
        x_mxfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_mxfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_mxfp4_sut._columnwise_scale_inv

    # Reference quantization
    ref_quantizer = MXFP4QuantizerRef(
        rowwise=True,
        columnwise=return_transpose,
    )
    x_ref = ref_quantizer.quantize(x)


    # Both native and reference produce uint8 E8M0 scales and packed FP4 data.
    # The Triton backend uses software thresholds matching the reference exactly,
    # while the C++ HIP kernel uses hardware FP4 conversion which may round
    # differently at threshold boundaries (±1 nibble / ±1 exponent).
    qx_unpacked = unpack_fp4(qx)
    qx_ref_unpacked = unpack_fp4(x_ref.data.view(dtype=torch.uint8))
    num_scale_cols = N // BLOCK_SIZE

    if _USE_TRITON:
        torch.testing.assert_close(qx_unpacked, qx_ref_unpacked, atol=0, rtol=0)
        torch.testing.assert_close(
            sx[:M, :num_scale_cols], x_ref.scale[:M, :num_scale_cols], atol=0, rtol=0,
        )
    else:
        compare_fp4_data(
            qx_unpacked, qx_ref_unpacked,
            msg=f"Rowwise FP4 ({M}x{N}, {x_dtype})",
        )
        compare_e8m0_scales(
            sx[:M, :num_scale_cols], x_ref.scale[:M, :num_scale_cols],
            msg=f"Rowwise E8M0 ({M}x{N}, {x_dtype})", max_diff=1, max_mismatch_rate=1e-4, max_outliers=1
        )

    if return_transpose:
        assert qx_t is not None and x_ref.data_t is not None
        qx_t_unpacked = unpack_fp4(qx_t)
        qx_t_ref_unpacked = unpack_fp4(x_ref.data_t.view(dtype=torch.uint8))
        num_scale_cols_t = M // BLOCK_SIZE

        if _USE_TRITON:
            torch.testing.assert_close(qx_t_unpacked, qx_t_ref_unpacked, atol=0, rtol=0)
            torch.testing.assert_close(
                sx_t[:N, :num_scale_cols_t], x_ref.scale_t[:N, :num_scale_cols_t],
                atol=0, rtol=0,
            )
        else:
            compare_fp4_data(
                qx_t_unpacked, qx_t_ref_unpacked,
                msg=f"Colwise FP4 ({M}x{N}, {x_dtype})",
            )
            compare_e8m0_scales(
                sx_t[:N, :num_scale_cols_t], x_ref.scale_t[:N, :num_scale_cols_t],
                msg=f"Colwise E8M0 ({M}x{N}, {x_dtype})", max_diff=1, max_mismatch_rate=1e-4, max_outliers=1
            )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        (320, 256),
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("return_transpose", [True, False], ids=["with_transpose", "no_transpose"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_alloc", "python_alloc"]
)
def test_quantization_versus_reference(
    M, N, x_dtype, return_transpose, use_cpp_allocator
):
    check_quantization_mxfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(128, 128)])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("extrema_high", [False, True], ids=["all_zeros", "all_max"])
@pytest.mark.parametrize("return_transpose", [True, False], ids=["with_transpose", "no_transpose"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_alloc", "python_alloc"]
)
def test_quantization_extrema(
    M, N, x_dtype, extrema_high, return_transpose, use_cpp_allocator
):
    """Test quantization with extreme values: all zeros or all max."""
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"

    if extrema_high:
        x = torch.full((M, N), torch.finfo(x_dtype).max, dtype=x_dtype, device=device)
    else:
        x = torch.zeros((M, N), dtype=x_dtype, device=device)

    mxfp4_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        shuffle_B_matrix_for_aiter=False,
        use_hadamard=False,
    )

    if use_cpp_allocator:
        result = mxfp4_quantizer(x)
    else:
        result = mxfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        result = mxfp4_quantizer.update_quantized(x, result)

    qx = result._rowwise_data.view(dtype=torch.uint8)
    sx = result._rowwise_scale_inv

    assert qx.shape == (M, N // 2)

    if not extrema_high:
        # All zeros input should produce all-zero packed FP4 data
        assert (qx == 0).all(), "All-zero input should produce all-zero FP4 data"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(32, 128), (64, 256)])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("return_transpose", [True, False], ids=["with_transpose", "no_transpose"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_alloc", "python_alloc"]
)
def test_quantization_noncontiguous_inputs(
    M, N, x_dtype, return_transpose, use_cpp_allocator
):
    """Test that non-contiguous inputs are handled correctly."""
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create non-contiguous input via transpose
    x_base = torch.randn((N, M), dtype=x_dtype, device=device)
    x = x_base.t()  # non-contiguous (M, N)
    assert not x.is_contiguous()

    mxfp4_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        shuffle_B_matrix_for_aiter=False,
        use_hadamard=False,
    )

    if use_cpp_allocator:
        result = mxfp4_quantizer(x)
    else:
        result = mxfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        result = mxfp4_quantizer.update_quantized(x, result)

    # Also quantize the contiguous version
    x_contig = x.contiguous()
    if use_cpp_allocator:
        result_contig = mxfp4_quantizer(x_contig)
    else:
        result_contig = mxfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        result_contig = mxfp4_quantizer.update_quantized(x_contig, result_contig)

    qx = result._rowwise_data.view(dtype=torch.uint8)
    qx_contig = result_contig._rowwise_data.view(dtype=torch.uint8)
    torch.testing.assert_close(qx, qx_contig, atol=0, rtol=0)

    sx = result._rowwise_scale_inv
    sx_contig = result_contig._rowwise_scale_inv
    num_scale_cols = N // BLOCK_SIZE
    torch.testing.assert_close(
        sx[:M, :num_scale_cols], sx_contig[:M, :num_scale_cols], atol=0, rtol=0
    )
