# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Grouped 1x128 blockwise FP8 restripe: the C++ kernel against a PyTorch reference, bit for bit."""

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

import transformer_engine.pytorch  # noqa: F401  (loads the extension)
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8BlockQuantizer
from transformer_engine.pytorch.utils import (
    get_torch_float8_e4m3_type,
    get_torch_float8_e5m2_type,
    is_fp8_fnuz,
)

if not IS_HIP_EXTENSION:
    pytest.skip("The grouped blockwise restripe is ROCm-only.", allow_module_level=True)

_BLOCK = 128
_splits = [
    [128, 0, 384, 256, 0, 128],
    [0, 0, 1024, 0],
    [512],
    [0, 0, 0],
]


def _quantize(splits, K, fp8_dtype, pow2):
    """Rows spanning ~12 decades: tiles mix row scales far apart (deep underflow on re-encode)."""
    M = sum(splits)
    torch.manual_seed(1234)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    x *= torch.logspace(-8, 4, M, device="cuda")[torch.randperm(M, device="cuda")][:, None]
    x *= torch.logspace(-2, 2, K, device="cuda")[None, :]
    q = Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=1,
        force_pow_2_scales=pow2,
    )
    return q(x.to(torch.bfloat16))


def _floor_pow2(s):
    return (s.view(torch.int32) & -0x800000).view(torch.float32)


def _reference(data, scales, splits, fp8_dtype, columnwise, pow2, direct, epsilon=0.0):
    """Group-contiguous outputs of the restripe, computed with PyTorch ops."""
    if fp8_dtype == tex.DType.kFloat8E4M3:
        torch_fp8, fp8_max = get_torch_float8_e4m3_type(), (240.0 if is_fp8_fnuz() else 448.0)
    else:
        torch_fp8, fp8_max = get_torch_float8_e5m2_type(), 57344.0
    M, K = data.shape
    rs_out = torch.cat([scales[:, off : off + m].reshape(-1) for off, m in _offsets(splits)])
    if not columnwise:
        return rs_out, None, None

    q = data.view(torch_fp8).float()  # [M, K]
    s_row = scales.t().repeat_interleave(_BLOCK, dim=1)[:, :K]  # [M, K], scale of each element
    tiles = M // _BLOCK
    if direct:
        s_tile = scales.t().reshape(tiles, _BLOCK, scales.shape[0]).amax(dim=1)  # [M/128, K/128]
        cs = s_tile.repeat_interleave(_BLOCK, dim=1)[:, :K]
        out = (q * (s_row / cs.repeat_interleave(_BLOCK, dim=0))).to(torch_fp8)
    else:
        x = q * s_row
        amax = x.abs().reshape(tiles, _BLOCK, K).amax(dim=1)  # [M/128, K]
        a = torch.where(amax < epsilon, torch.full_like(amax, epsilon), amax)
        # fp32 division on ROCm is not correctly rounded; the kernel's is (__fdiv_rn, __frcp_rn).
        s = (fp8_max / a.double()).float()
        s = torch.where(~torch.isfinite(a) | (a == 0), torch.ones_like(s), s)
        if pow2:
            s = _floor_pow2(s)
        cs = (1.0 / s.double()).float()
        scaled = x * s.repeat_interleave(_BLOCK, dim=0)
        out = scaled.clamp(-fp8_max, fp8_max).to(torch_fp8)
    out = out.view(torch.uint8)
    col = torch.cat([out[off : off + m].t().reshape(-1) for off, m in _offsets(splits)])
    return rs_out, col, cs


def _offsets(splits):
    off = 0
    for m in splits:
        yield off, m
        off += m


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.numel() == b.numel() and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    )


@pytest.mark.parametrize("splits", _splits)
@pytest.mark.parametrize("K", [128, 1536, 4096])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize(
    "mode", ["scales_only", "direct", "requant_pow2", "requant"], ids=lambda m: m
)
def test_matches_reference(splits, K, fp8_dtype, mode):
    columnwise = mode != "scales_only"
    pow2 = mode != "requant"
    direct = mode == "direct"

    inp = _quantize(splits, K, fp8_dtype, pow2)
    data = inp._rowwise_data.reshape(-1, K)
    scales = inp._rowwise_scale_inv

    ref = _reference(data, scales, splits, fp8_dtype, columnwise, pow2, direct)
    got = tex.fp8_blockwise_1d_rowwise_to_columnwise_grouped(
        data,
        scales,
        splits,
        fp8_dtype,
        columnwise=columnwise,
        epsilon=0.0,
        force_pow_2_scales=pow2,
        direct=direct,
    )
    names = ("rowwise scale_inv", "columnwise data", "columnwise scale_inv")
    for name, r, g in zip(names, ref, got):
        assert _same(r, g), f"{name} differs"


def test_noncontiguous_scales():
    splits, K = [256, 0, 128], 1024
    inp = _quantize(splits, K, tex.DType.kFloat8E4M3, True)
    data = inp._rowwise_data.reshape(-1, K)
    scales = inp._rowwise_scale_inv
    strided = scales.t().contiguous().t()
    assert not strided.is_contiguous()
    kwargs = dict(columnwise=True, epsilon=0.0, force_pow_2_scales=True, direct=True)
    restripe = tex.fp8_blockwise_1d_rowwise_to_columnwise_grouped
    a = restripe(data, scales, splits, tex.DType.kFloat8E4M3, **kwargs)
    b = restripe(data, strided, splits, tex.DType.kFloat8E4M3, **kwargs)
    assert all(_same(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "M, splits",
    [(128 * 257, [128] * 257), (256, [64, 64, 128]), (256, [128])],
    ids=["too_many_groups", "not_multiple_of_128", "wrong_sum"],
)
def test_rejects_bad_splits(M, splits):
    K = 1024
    inp = _quantize([M], K, tex.DType.kFloat8E4M3, True)
    with pytest.raises(RuntimeError):
        tex.fp8_blockwise_1d_rowwise_to_columnwise_grouped(
            inp._rowwise_data.reshape(-1, K),
            inp._rowwise_scale_inv,
            splits,
            tex.DType.kFloat8E4M3,
            columnwise=True,
            epsilon=0.0,
            force_pow_2_scales=True,
            direct=False,
        )
