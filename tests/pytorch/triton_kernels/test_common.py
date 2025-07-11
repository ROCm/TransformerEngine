# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import types

import numpy as np
import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex

from transformer_engine.pytorch.triton_kernels.common import (
    torch_e4m3_type,
    torch_e5m2_type,
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
    elif dtype == torch_e4m3_type or dtype == torch_e5m2_type:
        # TODO: different tolerances for FNUZ and OCP
        return 1e-2, 1e-2
    else:
        raise RuntimeError("Invalid type")


# PyTorch implementation of `compareResults` C++ function from `tests/cpp/test_common.cu`.
# Arguments:
#     t: actual tensor
#     r: expected tensor
# NOTE: DO NOT upcast inputs to fp32 if you are using te_compare for any precision other than fp32
def te_compare_results(t, r, atol, rtol, msg):
    assert t.dtype == r.dtype, f"Tensor dtypes don't match: {t.dtype} vs {r.dtype}."
    assert t.shape == r.shape, f"Tensor shapes don't match: {t.shape} vs {r.shape}."
    assert atol > 0, "Absolute tolerance must be positive."
    assert rtol > 0, "Relative tolerance must be positive."
    dtype = t.dtype
    t_orig = t
    r_orig = r
    t = t.cpu().to(torch.float32).to(torch.float64)
    r = r.cpu().to(torch.float32).to(torch.float64)
    diff = t - r
    atol_mismatch = torch.abs(diff) > atol
    nonzero_r = r != 0
    rtol_mismatch = torch.full_like(atol_mismatch, False)
    rtol_mismatch[nonzero_r] = torch.abs(diff[nonzero_r] / r[nonzero_r]) > rtol
    mismatch = atol_mismatch & (~nonzero_r | rtol_mismatch)
    has_mismatch = torch.any(mismatch).item()

    max_rel_diff = 0.0 # Default to 0.0 if no non-zero reference values
    max_abs_diff = 0.0 
    max_abs_diff_indices = None
    max_rel_diff_indices = None
    if has_mismatch:
        max_abs_diff = torch.max(torch.abs(diff[mismatch])).item()
        max_rel_diff = torch.max(torch.abs(diff[mismatch] / r[mismatch])).item()
        rel_diff = torch.full_like(diff, 0.0) # Initialize with zeros
        abs_diff = torch.full_like(diff, 0.0) # Initialize with zeros
        rel_diff[mismatch] = torch.abs(diff[mismatch] / r[mismatch])
        abs_diff[mismatch] = torch.abs(diff[mismatch])
        max_rel_diff_indices = torch.unravel_index(torch.argmax(rel_diff), rel_diff.shape)
        max_abs_diff_indices = torch.unravel_index(torch.argmax(abs_diff), abs_diff.shape)

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
        num_mismatched_elements = torch.sum(mismatch).item()
        total_elements = t.numel() 
        base_msg = (
            f"There are tensor mismatches.\n"
            f"Number of mismatched rows: {num_mismatched_elements} out of {total_elements} total rows.\n"
            f"Max Absolute Difference among mismatched: {max_abs_diff:.6e} (Tolerance: {atol:.6e}) at index {tuple(max_abs_diff_indices)}\n"
            f"Corresponding values: t={t[max_abs_diff_indices].item()}, r={r[max_abs_diff_indices].item()}\n"
        )
        if max_rel_diff_indices is not None:
             base_msg += (
                f"Max Relative Difference among mismatched: {max_rel_diff:.6e} (Tolerance: {rtol:.6e}) at index {tuple(max_rel_diff_indices)}\n"
                f"Corresponding values: t={t[max_rel_diff_indices].item()}, r={r[max_rel_diff_indices].item()}"
            )
        else:
            base_msg += (
                f"Max Relative Difference among mismatched: {max_rel_diff:.6e} (Tolerance: {rtol:.6e}) (no non-zero reference values)\n"
            )
        if isinstance(msg, str):
            msg = f"{msg}\n\n{base_msg}\n"
        elif isinstance(msg, types.LambdaType):
            msg = msg(base_msg)
        else:
            msg = base_msg
        assert False, msg


# Call PyTorch tensor comparison function or TE tensor comparison function.
def compare_results(provider, actual, expected, atol, rtol, msg):
    assert provider in {"torch", "te"}
    if provider == "torch":
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)
    else:
        te_compare_results(actual, expected, atol, rtol, msg)



# Get size in bytes of a given PyTorch type.
def sizeof(dtype):
    return torch.finfo(dtype).bits // 8


# Add `i` prefix to identify input type.
def input_dtypes_str(dtypes_str):
    return ["i" + dtype_str for dtype_str in dtypes_str]


# Add `o` prefix to identify output type.
def output_dtypes_str(dtypes_str):
    return ["o" + dtype_str for dtype_str in dtypes_str]


# Convert descriptive type string to PyTorch type.
def str_to_torch_dtype(dtype_str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8e4": torch_e4m3_type,
        "fp8e5": torch_e5m2_type,
    }[dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str]

# Common pytest skip conditions:

def skip_in_dtype_gt_out_dtype(in_dtype, out_dtype):
    if sizeof(in_dtype) < sizeof(out_dtype):
        pytest.skip("size of input dtype < size of output dtype")


def skip_mixed_16bit_float_types(in_dtype, out_dtype):
    if (in_dtype == torch.float16 and out_dtype == torch.bfloat16) or (
        in_dtype == torch.bfloat16 and out_dtype == torch.float16
    ):
        pytest.skip("hipified implementation does not support mixing fp16 and bf16")

def IS_FP8(dtype):
    return (dtype == torch_e4m3_type) or (dtype == torch_e5m2_type)