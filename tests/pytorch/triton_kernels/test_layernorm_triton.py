# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import os
import types

import numpy as np
import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.layernorm_triton import (
    te_layernorm_bwd_triton,
    te_layernorm_fwd_fp8_noalloc_triton,
)


# Mimics behavior of `fillUniform` from `tests/cpp/test_common.cu`.

rng_seed = 12345
rng = np.random.default_rng(np.random.MT19937(rng_seed))


def fill_uniform(shape, dtype):
    x = rng.uniform(-2.0, 1.0, shape)
    x = x.astype(np.float32)
    x = torch.tensor(x, device="cuda")
    x = x.to(dtype)
    return x


# Mimics behavior of `getTolerances` from `tests/cpp/test_common.cu`.
def get_tolerances(dtype):
    if dtype == torch.float32:
        return 1e-6, 5e-6
    elif dtype == torch.float16:
        return 1e-5, 1e-3
    elif dtype == torch.bfloat16:
        return 1e-5, 1e-2
    elif dtype == torch.float8_e4m3fnuz or dtype == torch.float8_e5m2fnuz:
        return 1e-2, 1e-2
    else:
        raise RuntimeError("Invalid type")


# PyTorch implementation of `compareResults` C++ function from `tests/cpp/test_common.cu`.
# Arguments:
#     t: actual tensor
#     r: expected tensor
def te_compare_results(t, r, atol, rtol, msg):
    assert t.dtype == r.dtype, f"Tensor dtypes don't match: {t.dtype} vs {r.dtype}."
    assert t.shape == r.shape, f"Tensor shapes don't match: {t.shape} vs {r.shape}."
    assert atol > 0, "Absolute tolerance must be positive."
    assert rtol > 0, "Relative tolerance must be positive."
    dtype = t.dtype
    t = t.cpu().to(torch.float32).to(torch.float64)
    r = r.cpu().to(torch.float32).to(torch.float64)
    diff = t - r
    atol_mismatch = torch.abs(diff) > atol
    nonzero_r = r != 0
    rtol_mismatch = torch.full_like(atol_mismatch, False)
    rtol_mismatch[nonzero_r] = torch.abs(diff[nonzero_r] / r[nonzero_r]) > rtol
    mismatch = atol_mismatch & (~nonzero_r | rtol_mismatch)
    has_mismatch = torch.any(mismatch).item()
    # for fp32 the floating point comparison is enough to error out
    if has_mismatch and dtype != torch.float32:
        # check if it is just a failure of round to nearest choosing different side of the real value
        # for non fp32 types
        mean = (t + r) / 2
        eps = 1e-6
        mean_one_plus_eps = mean * (1 + eps)
        mean_one_minus_eps = mean * (1 - eps)
        mean_gte_zero = mean >= 0
        mean_p = torch.where(mean_gte_zero, mean_one_plus_eps, mean_one_minus_eps)
        mean_m = torch.where(mean_gte_zero, mean_one_minus_eps, mean_one_plus_eps)
        cast_mean_p = (
            mean_p.to(torch.float32).to(dtype).to(torch.float32).to(torch.float64)
        )
        cast_mean_m = (
            mean_m.to(torch.float32).to(dtype).to(torch.float32).to(torch.float64)
        )
        min_tr = torch.minimum(t, r)
        max_tr = torch.maximum(t, r)
        round_check = ~((cast_mean_m == min_tr) & (cast_mean_p == max_tr))
        mismatch = mismatch & round_check
        has_mismatch = torch.any(mismatch).item()
    if has_mismatch:
        # TODO: Improve base message, add max absolute and relative differences.
        base_msg = "There are tensor mismatches."
        if isinstance(msg, str):
            msg = f"{msg}\n\n{base_msg}\n"
        elif isinstance(msg, types.LambdaType):
            msg = msg(base_msg)
        else:
            msg = base_msg
        assert False, msg


def compare_results(provider, actual, expected, atol, rtol, msg):
    assert provider in {"torch", "te"}
    if provider == "torch":
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)
    else:
        te_compare_results(actual, expected, atol, rtol, msg)


# Add `i` prefix to identify input type.
def input_dtypes_str(dtypes_str):
    return ["i" + dtype_str for dtype_str in dtypes_str]


# Add `o` prefix to identify output type.
def output_dtypes_str(dtypes_str):
    return ["o" + dtype_str for dtype_str in dtypes_str]


# Convert descriptive type string to PyTorch type.
def str_to_torch_dtype(dtype_str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    ]


# Get size in bytes of given PyTorch type.
def sizeof(dtype):
    return torch.finfo(dtype).bits // 8


def get_te_dtype(dtype):
    return {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
        torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3,
        torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2,
    }[dtype]


test_idtypes_str = input_dtypes_str(["fp32", "fp16"])

test_odtypes_str = output_dtypes_str(["fp32", "bf16", "fp16"])

test_shapes = [
    (2048, 12288),
    (768, 1024),
    (256, 65536),
    (128, 6144),
    (64, 2304),
    (229, 541),
    (71, 3571),
    (29, 17389),
]

all_boolean = [False, True]


@pytest.mark.parametrize("in_dtype", test_idtypes_str)
@pytest.mark.parametrize("out_dtype", test_odtypes_str)
@pytest.mark.parametrize("M, N", test_shapes)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_fwd_bwd_triton(in_dtype, out_dtype, M, N, zero_centered_gamma):
    # Get Torch data types:
    in_dtype = str_to_torch_dtype(in_dtype)
    out_dtype = str_to_torch_dtype(out_dtype)

    # Skip conditions:
    if sizeof(in_dtype) < sizeof(out_dtype):
        pytest.skip("size of input dtype < size of output dtype")
    if (in_dtype == torch.float16 and out_dtype == torch.bfloat16) or (
        in_dtype == torch.bfloat16 and out_dtype == torch.float16
    ):
        pytest.skip("hipified layernorm kernel does not support mixing fp16 and bf16")

    # Generate tensors:
    x = fill_uniform((M, N), in_dtype)
    gamma = fill_uniform(N, in_dtype)
    beta = fill_uniform(N, in_dtype)
    dz = fill_uniform((M, N), in_dtype)

    epsilon = 1e-5

    # Run Triton forward.
    y_triton = torch.empty((M, N), dtype=out_dtype, device="cuda")
    y_triton, mu_triton, rsigma_triton = te_layernorm_fwd_fp8_noalloc_triton(
        x, gamma, beta, epsilon, y_triton, out_dtype, zero_centered_gamma
    )

    # Run Hipified forward reference.
    scale = amax = scale_inv = torch.empty(0, device="cuda")
    fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
    y_hipified = torch.empty((M, N), dtype=out_dtype, device="cuda")
    y_hipified, mu_hipified, rsigma_hipified = tex.layernorm_fwd_fp8_noalloc(
        x,
        gamma,
        beta,
        epsilon,
        scale,
        y_hipified,
        amax,
        scale_inv,
        get_te_dtype(out_dtype),
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )

    # Assert on mu and rsigma:
    atol_stats, _ = get_tolerances(torch.float32)
    rtol_stats = 5e-5
    compare_results(
        "torch",
        mu_triton,
        mu_hipified,
        atol_stats,
        rtol_stats,
        lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        "torch",
        rsigma_triton,
        rsigma_hipified,
        atol_stats,
        rtol_stats,
        lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
    )

    # Assert on y:
    atol, rtol = get_tolerances(out_dtype)
    if out_dtype == torch.float32:
        # everything pass with default fp32 atol=1e-6
        # atol = 5e-7
        pass
    compare_results(
        "torch",
        y_triton,
        y_hipified,
        atol,
        rtol,
        lambda msg: f"y does not match triton <-> hip\n\n{msg}\n",
    )

    # Run Triton backward.
    dx_triton, dgamma_triton, dbeta_triton = te_layernorm_bwd_triton(
        dz,
        x,
        mu_triton,
        rsigma_triton,
        gamma,
        zero_centered_gamma,
    )

    # Run Hipified backward reference.
    dx_hipified, dgamma_hipified, dbeta_hipified = tex.layernorm_bwd(
        dz,
        x,
        mu_hipified,
        rsigma_hipified,
        gamma,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )

    # Assert on dx, dgamma and dbeta:
    atol_bwd = 1.5e-4
    rtol_bwd = 1e-4
    # TE comparison deals with fp16 rounding errors.
    bwd_cmp = "te" if in_dtype == out_dtype == torch.float16 else "torch"
    compare_results(
        bwd_cmp,
        dx_triton,
        dx_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        bwd_cmp,
        dgamma_triton,
        dgamma_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
    )
    compare_results(
        bwd_cmp,
        dbeta_triton,
        dbeta_hipified,
        atol_bwd,
        rtol_bwd,
        lambda msg: f"dbeta does not match triton <-> hip\n\n{msg}\n",
    )
