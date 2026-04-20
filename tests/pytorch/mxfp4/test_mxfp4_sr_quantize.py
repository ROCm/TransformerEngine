# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Stochastic rounding tests for MXFP4 quantization.

Validates that the mean of many SR-quantized-then-dequantized tensors
converges closer to the original than a single round-to-nearest (RN)
quantization. This is the statistical property that makes SR useful
for training.
"""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp4_available(return_reason=True)

BLOCK_SIZE = 32
seed = 12345

_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 (M, N/2) into nibbles (M, N)."""
    repeated = packed.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def dequantize_mxfp4(
    fp4_packed: torch.Tensor,
    scales_u8: torch.Tensor,
    M: int,
    N: int,
) -> torch.Tensor:
    """Dequantize packed MXFP4 data using E8M0 block scales.

    Parameters
    ----------
    fp4_packed : (M, N/2) uint8
    scales_u8  : (>=M, >=N/32) uint8 E8M0 exponents (may be padded)
    """
    num_blocks = N // BLOCK_SIZE
    scales = scales_u8[:M, :num_blocks]
    scale_float = torch.pow(2.0, scales.to(torch.float32) - 127.0)

    nibbles = unpack_fp4(fp4_packed)
    lut = _FP4_LUT.to(device=nibbles.device)
    decoded = lut[nibbles.long()]

    decoded_blocks = decoded.reshape(M, num_blocks, BLOCK_SIZE)
    result = (decoded_blocks * scale_float.unsqueeze(-1)).reshape(M, N)
    return result


def quantize_mxfp4(
    x: torch.Tensor,
    stochastic_rounding: bool,
    use_hadamard: bool = False,
):
    """Quantize using MXFP4Quantizer, return (fp4_packed, scales, fp4_packed_t, scales_t)."""
    quantizer = MXFP4Quantizer(
        rowwise=True,
        columnwise=True,
        stochastic_rounding=stochastic_rounding,
        use_hadamard=use_hadamard,
    )
    result = quantizer(x)

    qx = result._rowwise_data.view(dtype=torch.uint8)
    sx = result._rowwise_scale_inv.view(dtype=torch.uint8)
    qx_t = result._columnwise_data.view(dtype=torch.uint8)
    sx_t = result._columnwise_scale_inv.view(dtype=torch.uint8)
    return qx, sx, qx_t, sx_t


def check_sr_versus_rn(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    use_hadamard: bool,
    n_iters: int = 50,
) -> None:
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device) * 2 - 1

    # Round-to-nearest baseline
    qx_rn, sx_rn, qx_t_rn, sx_t_rn = quantize_mxfp4(
        x, stochastic_rounding=False, use_hadamard=use_hadamard,
    )
    dq_rn = dequantize_mxfp4(qx_rn, sx_rn, M, N)
    error_rn = (dq_rn.float() - x.float())
    rmse_rn = torch.sqrt((error_rn ** 2).mean())

    y = x.t().contiguous()
    dq_t_rn = dequantize_mxfp4(qx_t_rn, sx_t_rn, N, M)
    error_t_rn = (dq_t_rn.float() - y.float())
    rmse_t_rn = torch.sqrt((error_t_rn ** 2).mean())

    # Stochastic rounding: accumulate dequantized results
    sr_accum = torch.zeros(M, N, dtype=torch.float32, device=device)
    sr_t_accum = torch.zeros(N, M, dtype=torch.float32, device=device)

    for _ in range(n_iters):
        qx_sr, sx_sr, qx_t_sr, sx_t_sr = quantize_mxfp4(
            x, stochastic_rounding=True, use_hadamard=use_hadamard,
        )
        sr_accum += dequantize_mxfp4(qx_sr, sx_sr, M, N).float()
        sr_t_accum += dequantize_mxfp4(qx_t_sr, sx_t_sr, N, M).float()

    sr_mean = sr_accum / n_iters
    error_sr = sr_mean - x.float()
    rmse_sr = torch.sqrt((error_sr ** 2).mean())

    sr_t_mean = sr_t_accum / n_iters
    error_t_sr = sr_t_mean - y.float()
    rmse_t_sr = torch.sqrt((error_t_sr ** 2).mean())

    print(f"Rowwise  — RMSE SR: {rmse_sr:.3e} | RMSE RN: {rmse_rn:.3e}")
    print(f"Colwise  — RMSE SR: {rmse_t_sr:.3e} | RMSE RN: {rmse_t_rn:.3e}")

    assert rmse_sr < rmse_rn, (
        f"SR rowwise RMSE ({rmse_sr:.3e}) should be smaller than RN ({rmse_rn:.3e})"
    )
    assert rmse_t_sr < rmse_t_rn, (
        f"SR colwise RMSE ({rmse_t_sr:.3e}) should be smaller than RN ({rmse_t_rn:.3e})"
    )


def check_sr_nondeterminism(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Verify that SR produces different outputs across invocations."""
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    qx1, _, _, _ = quantize_mxfp4(x, stochastic_rounding=True)
    qx2, _, _, _ = quantize_mxfp4(x, stochastic_rounding=True)

    assert not torch.equal(qx1, qx2), (
        "Two SR quantizations of the same input should (almost surely) differ"
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("use_hadamard", [False, True], ids=["no_hadamard", "hadamard"])
def test_sr_versus_rn(M, N, x_dtype, use_hadamard):
    """SR mean over many iterations should have lower RMSE than RN."""
    check_sr_versus_rn(
        x_dtype=x_dtype,
        M=M,
        N=N,
        use_hadamard=use_hadamard,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(256, 256)])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
def test_sr_nondeterminism(M, N, x_dtype):
    """Consecutive SR quantizations must produce different bit patterns."""
    check_sr_nondeterminism(x_dtype=x_dtype, M=M, N=N)
