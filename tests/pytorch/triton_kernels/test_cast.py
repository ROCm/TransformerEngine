# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch

from transformer_engine.pytorch.triton_kernels.cast import te_quantize_triton
from transformer_engine.pytorch.triton_kernels.cast_transpose import _compute_scale_from_amax_triton
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8CurrentScalingQuantizer
from transformer_engine.pytorch.triton_kernels.common import te_dtype_to_torch_dtype
import transformer_engine_torch as tex
from test_common import te_compare_results, fill_uniform, get_tolerances
from transformer_engine.pytorch.fp8 import autocast
from transformer_engine.common import recipe
from transformer_engine.pytorch.utils import get_torch_float8_e4m3_type, get_torch_float8_e5m2_type

@pytest.mark.parametrize("scaling", ("delayed", "current"))
@pytest.mark.parametrize("shape", 
                         [
                        (16 ),
                        (16000 ),
                        (128, 128),
                        (256, 256),
                        (768, 1024),
                        (256, 65536),
                        (2048, 12288),
                        (65536, 128),
                        (65536, 160),
                        (16384, 1616),
                        (1, 128),
                        (1, 1296),
                        (1, 16),
                        (5, 160),
                        (5, 4, 3, 160),
                        (217, 256)
                        ])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize(scaling, shape, in_dtype, out_dtype):
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    if scaling == "current":
        triton_quantizer = Float8CurrentScalingQuantizer(fp8_dtype=out_dtype, device="cuda")
        tex_quantizer = Float8CurrentScalingQuantizer(fp8_dtype=out_dtype, device="cuda")

        with autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
            quantized_out_triton = te_quantize_triton(input_tensor, quantizer=triton_quantizer)
            quantized_out_tex    = tex.quantize(input_tensor, tex_quantizer)

    elif scaling == "delayed":
        scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
        amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')

        triton_quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)
        tex_quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=out_dtype)

        quantized_out_triton  = te_quantize_triton(input_tensor, quantizer=triton_quantizer)
        quantized_out_tex = tex.quantize(input_tensor, tex_quantizer)

    else:
        raise ValueError(f"unknown scaling method {scaling}")

    torch_out_dtype = te_dtype_to_torch_dtype(out_dtype)
    
    atol_q, rtol_q = get_tolerances(torch_out_dtype)
    te_compare_results(
        quantized_out_triton._data.view(torch_out_dtype),
        quantized_out_tex._data.view(torch_out_dtype),
        atol_q,
        rtol_q,
        lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
        use_torch_semantics=True,
    )
    assert quantized_out_triton._transpose is not None, "Triton transpose is none!" 
    assert quantized_out_tex._transpose is not None, "TEX transpose is none!" 
    te_compare_results(
        quantized_out_triton._transpose.view(torch_out_dtype),
        quantized_out_tex._transpose.view(torch_out_dtype),
        atol_q,
        rtol_q,
        lambda msg: f"triton does not match tex <-> hip\n\n{msg}\n",
        use_torch_semantics=True,
    )
    
    atol_scale, rtol_scale = get_tolerances(torch.float32)
    te_compare_results(
        quantized_out_triton._get_quantizer().scale,
        quantized_out_tex._get_quantizer().scale,
        atol=atol_scale, rtol=rtol_scale,
        msg='Scale results do not match!',
        use_torch_semantics=True
    ),
    te_compare_results(
        quantized_out_triton._get_quantizer().amax,
        quantized_out_tex._get_quantizer().amax,
        atol=atol_scale, rtol=rtol_scale,
        msg='AMAX results do not match!',
        use_torch_semantics=True
    )


@pytest.mark.parametrize("t_shape",
                         [
                        (16 ),
                        (768, 1024),
                        (1, 128),
                        (5, 160),
                        (5, 4, 3, 160),
                        ])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_quantize_bad_transpose(t_shape, fp8_dtype):
    """
    Non-regression test for gh-13121, testing whether te_quantize_triton
    correctly dispatches based off of transpose buffer validity.
    """
    # The input type and shape are arbitrary, but we choose only one so as to
    # avoid unnecessarily expanding the test parameter space.
    in_dtype = torch.float32
    shape = (128, 128)
    input_tensor = fill_uniform(shape, dtype=in_dtype)
    output_tensor = torch.empty(shape, dtype=in_dtype, device='cuda')

    scale_tensor = torch.rand(1, dtype=torch.float32, device='cuda') * 3.0 - 2.0
    amax_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    quantizer = Float8Quantizer(scale=scale_tensor, amax=amax_tensor, fp8_dtype=fp8_dtype)

    quantized_output = quantizer(output_tensor)
    quantized_output._transpose_invalid = True
    quantized_output._transpose = torch.empty(t_shape, device='cuda')

    te_quantize_triton(input_tensor, quantizer=quantizer, output=quantized_output)


@pytest.mark.parametrize("amax_val", (0.0, float('nan'), float('inf'), -float('inf'), 1.0, 1e-8, 123.456))
@pytest.mark.parametrize("force_pow_2_scales", (False, True))
@pytest.mark.parametrize("epsilon", (0.0, 1e-3, 100.0))
@pytest.mark.parametrize("fp8_dtype", (get_torch_float8_e4m3_type(), get_torch_float8_e5m2_type()))
def test_compute_scale_from_amax(amax_val, force_pow_2_scales, epsilon, fp8_dtype):
    max_fp8 = torch.finfo(fp8_dtype).max
    value_for_inf = float(torch.finfo(torch.float32).max)

    amax_list = [torch.tensor(amax_val, dtype=torch.float32, device="cuda")]

    # TEX path - TEX expects lists for (amaxes, scales, inv_scales)
    scale_ref = [torch.empty((), dtype=torch.float32, device="cuda")]
    scale_inv_ref = [torch.empty((), dtype=torch.float32, device="cuda")]

    chunk_size = 2048 * 32  # arbitrary
    overflow_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    tex.multi_tensor_compute_scale_and_scale_inv(
        chunk_size,
        overflow_buf,
        [amax_list, scale_ref, scale_inv_ref],
        max_fp8,
        force_pow_2_scales,
        epsilon,
    )

    # Triton path & comparison
    scale_triton = torch.empty((), dtype=torch.float32, device="cuda")
    scale_inv_triton   = torch.empty((), dtype=torch.float32, device="cuda")
    _compute_scale_from_amax_triton[(1,)](
        amax_list[0], scale_triton, scale_inv_triton,
        float(max_fp8), float(epsilon), float(value_for_inf),
        FORCE_POW_2_SCALES=force_pow_2_scales,
    )

    torch.testing.assert_close(scale_triton, scale_ref[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale_inv_triton, scale_inv_ref[0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", ((1, 1), (7, 13), (256, 257), (1024, 1024), (2048, 4097)))
@pytest.mark.parametrize("in_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("out_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_amax_atomic_vs_two_stage(shape, in_dtype, out_dtype):
    import os
    device = "cuda"
    input_tensor = fill_uniform(shape, dtype=in_dtype)

    quantizer_atomic = Float8CurrentScalingQuantizer(fp8_dtype=out_dtype, device=device)
    quantizer_2stage = Float8CurrentScalingQuantizer(fp8_dtype=out_dtype, device=device)

    env_key = "NVTE_USE_ATOMIC_AMAX"
    old_env_val = os.environ.get(env_key)

    try:
        # atomic amax
        os.environ[env_key] = "1"

        with autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
            out_atomic = te_quantize_triton(input_tensor, quantizer=quantizer_atomic)

        # 2-stage amax
        os.environ[env_key] = "0"

        with autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
            out_2stage = te_quantize_triton(input_tensor, quantizer=quantizer_2stage)

        te_compare_results(
            out_atomic._get_quantizer().amax,
            out_2stage._get_quantizer().amax,
            atol=0.0, rtol=0.0,
            msg='AMAX results do not match!',
            use_torch_semantics=True
        )
    finally:
        # Restore environment
        if old_env_val is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env_val
