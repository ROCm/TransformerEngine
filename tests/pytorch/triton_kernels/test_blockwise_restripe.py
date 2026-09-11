# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
import transformer_engine_torch as tex
from test_common import te_compare_results, fill_uniform, get_tolerances


@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
        (256, 256),
        (512, 1024),
        (2048, 4096),
    ],
)
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_blockwise_1d_rowwise_to_columnwise_restripe(shape, in_dtype, fp8_dtype):
    """1x128 pow-2 restripe matches dequant + columnwise-only quantize."""
    torch_out_dtype = te_dtype_to_torch_dtype(fp8_dtype)
    x = fill_uniform(shape, dtype=in_dtype)

    row_q = Float8BlockQuantizer(
        fp8_dtype,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=1,
        force_pow_2_scales=True,
    )
    row_tensor = row_q(x)

    hp = row_tensor.dequantize(dtype=in_dtype)
    col_q = Float8BlockQuantizer(
        fp8_dtype,
        rowwise=False,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=True,
    )
    col_ref = col_q(hp)

    row_tensor.update_usage(rowwise_usage=True, columnwise_usage=True)
    assert row_tensor._columnwise_data is not None
    assert row_tensor._rowwise_data is not None

    atol_fp8, rtol_fp8 = get_tolerances(torch_out_dtype)
    te_compare_results(
        row_tensor._columnwise_data.view(torch_out_dtype),
        col_ref._columnwise_data.view(torch_out_dtype),
        atol_fp8,
        rtol_fp8,
        msg="restriped columnwise data doesn't match dequant+requant",
    )
    te_compare_results(
        row_tensor._columnwise_scale_inv,
        col_ref._columnwise_scale_inv,
        0.0,
        0.0,
        msg="restriped columnwise scale_inv doesn't match",
        use_torch_semantics=True,
    )


def test_blockwise_1d_update_usage_columnwise_only_drops_rowwise():
    x = fill_uniform((256, 256), dtype=torch.bfloat16)
    tensor = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=1,
        force_pow_2_scales=True,
    )(x)
    tensor.update_usage(rowwise_usage=False, columnwise_usage=True)
    assert tensor._rowwise_data is None
    assert tensor._columnwise_data is not None
    assert tensor._columnwise_scale_inv is not None
