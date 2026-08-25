# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for ROCm Triton blockwise FP8 quantization and grouped GEMM."""

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch.triton_kernels.common import get_torch_e4m3_type
from transformer_engine.pytorch.triton_kernels.blockwise_quantize import (
    quantize_fp8_blockwise,
    quantize_fp8_blockwise_dual,
    quantize_fp8_blockwise_weight,
    quantize_fp8_blockwise_segment_m,
)
from transformer_engine.pytorch.triton_kernels.blockwise_fp8_grouped_gemm import (
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
)

pytestmark = pytest.mark.skipif(not IS_HIP_EXTENSION, reason="ROCm Triton blockwise kernels only")

BLOCK = 128
FP8_DTYPE = get_torch_e4m3_type()
IN_DTYPES = [torch.bfloat16, torch.float16]


def _cdiv(n, d):
    return (n + d - 1) // d


def _floor_to_pow2(scale: torch.Tensor) -> torch.Tensor:
    # 0xFF800000 as signed int32: keep the exponent, zero the mantissa.
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    return (bits & -8388608).view(torch.float32)


def _group_offs(splits, device="cuda"):
    offs = torch.zeros(len(splits) + 1, dtype=torch.int64, device=device)
    offs[1:] = torch.cumsum(torch.tensor(splits, dtype=torch.int64, device=device), 0)
    return offs


def _ref_rowwise_quantize(x: torch.Tensor, dtype: torch.dtype, block: int = BLOCK, pow2: bool = False):
    m, n = x.shape
    fp8_max = torch.finfo(dtype).max
    nb = _cdiv(n, block)
    x_pad = torch.nn.functional.pad(x.float(), (0, nb * block - n))
    tiles = x_pad.reshape(m, nb, block)
    amax = tiles.abs().amax(dim=-1).clamp_min(1e-4)
    scale = fp8_max / amax
    if pow2:
        scale = _floor_to_pow2(scale)
    q = (tiles * scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(dtype)
    q = q.reshape(m, nb * block)[:, :n].contiguous()
    return q, (1.0 / scale).contiguous()


def _ref_colwise_quantize(x: torch.Tensor, dtype: torch.dtype, block: int = BLOCK, pow2: bool = False):
    m, n = x.shape
    fp8_max = torch.finfo(dtype).max
    mb = _cdiv(m, block)
    x_pad = torch.nn.functional.pad(x.float(), (0, 0, 0, mb * block - m))
    tiles = x_pad.reshape(mb, block, n)
    amax = tiles.abs().amax(dim=1).clamp_min(1e-4)
    scale = fp8_max / amax
    if pow2:
        scale = _floor_to_pow2(scale)
    q = (tiles * scale.unsqueeze(1)).clamp(-fp8_max, fp8_max).to(dtype)
    q = q.reshape(mb * block, n)[:m].contiguous()
    return q, (1.0 / scale).contiguous()


def _ref_weight_quantize(w: torch.Tensor, dtype: torch.dtype, block: int = BLOCK, pow2: bool = False):
    g, m, n = w.shape
    fp8_max = torch.finfo(dtype).max
    mb, nb = _cdiv(m, block), _cdiv(n, block)
    w_pad = torch.nn.functional.pad(w.float(), (0, nb * block - n, 0, mb * block - m))
    tiles = w_pad.reshape(g, mb, block, nb, block)
    amax = tiles.abs().amax(dim=(2, 4)).clamp_min(1e-4)
    scale = fp8_max / amax
    if pow2:
        scale = _floor_to_pow2(scale)
    q = (tiles * scale[:, :, None, :, None]).clamp(-fp8_max, fp8_max).to(dtype)
    q = q.reshape(g, mb * block, nb * block)[:, :m, :n].contiguous()
    return q, (1.0 / scale).contiguous()


def _dequant_rowwise(q, s, n, block=BLOCK):
    return q.float() * s.repeat_interleave(block, dim=1)[:, :n]


def _dequant_colwise(q, s, m, block=BLOCK):
    return q.float() * s.repeat_interleave(block, dim=0)[:m]


@pytest.mark.parametrize("shape", [(128, 256), (200, 128), (256, 384), (64, 192)])
@pytest.mark.parametrize("dtype", IN_DTYPES, ids=str)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("pow2", [False, True])
def test_quantize_fp8_blockwise(shape, dtype, axis, pow2):
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    q, s = quantize_fp8_blockwise(x, FP8_DTYPE, axis=axis, block_size=BLOCK, pow2=pow2)
    if axis == 1:
        q_ref, s_ref = _ref_rowwise_quantize(x, FP8_DTYPE, pow2=pow2)
        dq = _dequant_rowwise(q, s, x.shape[1])
        dq_ref = _dequant_rowwise(q_ref, s_ref, x.shape[1])
    else:
        q_ref, s_ref = _ref_colwise_quantize(x, FP8_DTYPE, pow2=pow2)
        dq = _dequant_colwise(q, s, x.shape[0])
        dq_ref = _dequant_colwise(q_ref, s_ref, x.shape[0])
    torch.testing.assert_close(s, s_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dq, dq_ref, atol=0.10, rtol=0.10)


@pytest.mark.parametrize("shape", [(128, 256), (200, 128)])
@pytest.mark.parametrize("dtype", IN_DTYPES, ids=str)
@pytest.mark.parametrize("pow2", [False, True])
def test_quantize_fp8_blockwise_dual(shape, dtype, pow2):
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    q_row, s_row, q_col, s_col = quantize_fp8_blockwise_dual(x, FP8_DTYPE, block_size=BLOCK, pow2=pow2)
    q_row_ref, s_row_ref = _ref_rowwise_quantize(x, FP8_DTYPE, pow2=pow2)
    q_col_ref, s_col_ref = _ref_colwise_quantize(x, FP8_DTYPE, pow2=pow2)
    torch.testing.assert_close(s_row, s_row_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(s_col, s_col_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        _dequant_rowwise(q_row, s_row, x.shape[1]),
        _dequant_rowwise(q_row_ref, s_row_ref, x.shape[1]),
        atol=0.10,
        rtol=0.10,
    )
    torch.testing.assert_close(
        _dequant_colwise(q_col, s_col, x.shape[0]),
        _dequant_colwise(q_col_ref, s_col_ref, x.shape[0]),
        atol=0.10,
        rtol=0.10,
    )


@pytest.mark.parametrize("shape", [(2, 256, 256), (3, 128, 256), (1, 200, 192), (256, 256)])
@pytest.mark.parametrize("dtype", IN_DTYPES, ids=str)
@pytest.mark.parametrize("pow2", [False, True])
def test_quantize_fp8_blockwise_weight(shape, dtype, pow2):
    w = torch.randn(*shape, dtype=dtype, device="cuda")
    q, s = quantize_fp8_blockwise_weight(w, FP8_DTYPE, block_size=BLOCK, pow2=pow2)
    w3 = w if w.dim() == 3 else w.unsqueeze(0)
    q_ref, s_ref = _ref_weight_quantize(w3, FP8_DTYPE, pow2=pow2)
    if w.dim() == 2:
        q_ref, s_ref = q_ref.squeeze(0), s_ref.squeeze(0)
    torch.testing.assert_close(s, s_ref, atol=1e-5, rtol=1e-4)
    n = w.shape[-1]
    m = w.shape[-2]
    dq = q.float() * (
        s.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)[..., :m, :n]
    )
    dq_ref = q_ref.float() * (
        s_ref.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)[..., :m, :n]
    )
    torch.testing.assert_close(dq, dq_ref, atol=0.10, rtol=0.10)


@pytest.mark.parametrize(
    "m_splits",
    [
        [80, 176],
        [128, 128],
        [0, 256],
        [64, 0, 192],
        [200, 56],
    ],
)
@pytest.mark.parametrize("n", [128, 256, 192])
@pytest.mark.parametrize("dtype", IN_DTYPES, ids=str)
def test_quantize_fp8_blockwise_segment_m(m_splits, n, dtype):
    m = sum(m_splits)
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    group_lens = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
    group_offs = _group_offs(m_splits)
    x_fp8, scales, vk_lens, vk_offs = quantize_fp8_blockwise_segment_m(
        x, FP8_DTYPE, BLOCK, group_lens, group_offs
    )
    expected_lens = [((s + BLOCK - 1) // BLOCK) * BLOCK for s in m_splits]
    expected_offs = [0]
    for length in expected_lens:
        expected_offs.append(expected_offs[-1] + length)
    assert vk_lens.tolist() == expected_lens
    assert vk_offs.tolist() == expected_offs
    cursor = 0
    for valid, padded in zip(m_splits, expected_lens):
        if padded > valid:
            assert torch.all(x_fp8[cursor + valid : cursor + padded].float() == 0)
        cursor += padded
    assert x_fp8.shape[0] >= expected_offs[-1]
    assert scales.dtype == torch.float32


@pytest.mark.parametrize(
    "splits,n,k",
    [
        ([64, 96, 96], 256, 256),
        ([128, 128], 256, 192),
        ([80, 176], 192, 256),
        ([0, 128, 128], 256, 256),
        ([256], 128, 128),
    ],
)
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16], ids=str)
def test_grouped_gemm_fp8_blockwise_matches_dequant_ref(splits, n, k, trans_b, in_dtype, out_dtype):
    g = len(splits)
    m = sum(splits)
    out_n = n if trans_b else k
    if out_n % BLOCK != 0:
        pytest.skip("blockwise grouped GEMM uses 128-wide N tiles")
    a_k = k if trans_b else n
    a = torch.randn(m, a_k, dtype=in_dtype, device="cuda")
    b = torch.randn(g, n, k, dtype=in_dtype, device="cuda")
    a_fp8, a_s = quantize_fp8_blockwise(a, FP8_DTYPE, axis=1, block_size=BLOCK)
    b_fp8, b_s = quantize_fp8_blockwise_weight(b, FP8_DTYPE, block_size=BLOCK)
    offs = _group_offs(splits)

    out = grouped_gemm_fp8_blockwise_triton_kernel(
        a_fp8, b_fp8, a_s, b_s, offs, trans_b=trans_b, out_dtype=out_dtype
    )
    assert out.dtype == out_dtype

    a_dq = _dequant_rowwise(a_fp8, a_s, a_k)
    b_dq = b_fp8.float() * (
        b_s.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)[:, :n, :k]
    )
    ref = torch.zeros(m, out_n, dtype=torch.float32, device="cuda")
    for i in range(g):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        if sl.start == sl.stop:
            continue
        ref[sl] = a_dq[sl] @ (b_dq[i].T if trans_b else b_dq[i])
    torch.testing.assert_close(out.float(), ref, atol=0.15, rtol=0.05)


@pytest.mark.parametrize(
    "splits,n,k",
    [
        ([128, 128], 128, 128),
        ([80, 176], 256, 128),
        ([0, 256], 256, 128),
        ([64, 96, 96], 128, 256),
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=str)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_variable_k_wgrad(splits, n, k, accumulate, in_dtype, out_dtype):
    g = len(splits)
    tokens = sum(splits)
    dy = torch.randn(tokens, n, dtype=in_dtype, device="cuda")
    x = torch.randn(tokens, k, dtype=in_dtype, device="cuda")
    group_lens = torch.tensor(splits, dtype=torch.int64, device="cuda")
    group_offs = _group_offs(splits)
    go_col, go_s, _, vk_offs = quantize_fp8_blockwise_segment_m(
        dy, FP8_DTYPE, BLOCK, group_lens, group_offs
    )
    x_col, x_s, _, _ = quantize_fp8_blockwise_segment_m(
        x, FP8_DTYPE, BLOCK, group_lens, group_offs
    )

    fresh = grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
        go_col, x_col, go_s, x_s, vk_offs, out_dtype=out_dtype, accumulate=False
    )
    assert fresh.dtype == out_dtype
    assert fresh.shape == (g, n, k)

    go_dq = _dequant_colwise(go_col, go_s, go_col.shape[0])
    x_dq = _dequant_colwise(x_col, x_s, x_col.shape[0])
    ref = torch.zeros(g, n, k, dtype=torch.float32, device="cuda")
    for i in range(g):
        sl = slice(int(vk_offs[i]), int(vk_offs[i + 1]))
        if sl.start == sl.stop:
            continue
        ref[i] = go_dq[sl].T @ x_dq[sl]
    torch.testing.assert_close(fresh.float(), ref, atol=0.15, rtol=0.05)

    if accumulate:
        main_grad = torch.randn(g, n, k, dtype=out_dtype, device="cuda")
        expected = main_grad.clone()
        grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
            go_col,
            x_col,
            go_s,
            x_s,
            vk_offs,
            out=main_grad,
            accumulate=True,
        )
        torch.testing.assert_close(main_grad.float(), (expected + fresh).float(), atol=1e-3, rtol=1e-4)
