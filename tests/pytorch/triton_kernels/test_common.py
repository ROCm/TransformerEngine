# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from __future__ import annotations

import types

import numpy as np
import torch

from transformer_engine.pytorch import cpp_extensions as tex

from transformer_engine.pytorch.triton_kernels.common import (
    get_torch_e4m3_type,
    get_torch_e5m2_type,
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
    elif dtype == get_torch_e4m3_type() or dtype == get_torch_e5m2_type():
        # TODO: different tolerances for FNUZ and OCP
        return 1e-2, 1e-2
    else:
        raise RuntimeError("Invalid type")

def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        if dtype == tex.DType.kFloat8E4M3:
            return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
        if dtype == tex.DType.kFloat8E5M2:
            return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    raise ValueError(f"Unsupported dtype ({dtype})")


# PyTorch implementation of `compareResults` C++ function from `tests/cpp/test_common.cu`.
# Arguments:
#     t: actual tensor
#     r: expected tensor
# NOTE: DO NOT upcast inputs to fp32 if you are using te_compare for any precision other than fp32
def te_compare_results(t, r, atol, rtol, msg, use_torch_semantics=False):
    assert t.dtype == r.dtype, f"Tensor dtypes don't match: {t.dtype} vs {r.dtype}."
    assert t.shape == r.shape, f"Tensor shapes don't match: {t.shape} vs {r.shape}."
    dtype = t.dtype
    t = t.cpu().to(torch.float32).to(torch.float64)
    r = r.cpu().to(torch.float32).to(torch.float64)

    # If any of the tensors contain NaN we
    if torch.isnan(t).any() or torch.isnan(r).any():
        base_msg = (
            f"NaN values found!\n"
        )

        # Find which tensor has NaNs and at which indices
        if torch.isnan(t).any():
            nan_count = torch.isnan(t).sum()
            nan_indices = torch.where(torch.isnan(t))
            base_msg += f"Tensor 't' has {nan_count} NaN(s) at indices: {nan_indices}\n"

        if torch.isnan(r).any():
            nan_count = torch.isnan(r).sum()
            nan_indices = torch.where(torch.isnan(r))
            base_msg += f"Tensor 'r' has {nan_count} NaN(s) at indices: {nan_indices}\n"

        if isinstance(msg, str):
            msg = f"{msg}\n\n{base_msg}\n"
        elif isinstance(msg, types.LambdaType):
            msg = msg(base_msg)
        else:
            msg = base_msg
        assert False, msg

    diff = t - r
    adiff = torch.abs(diff)
    nonzero_r = r != 0
    rel_diff = torch.where(nonzero_r, torch.abs(diff / r), torch.zeros_like(diff))
    if use_torch_semantics:
        mismatch = adiff > atol + rtol * torch.abs(r)
    else:
        assert atol > 0, "Absolute tolerance must be positive."
        assert rtol > 0, "Relative tolerance must be positive."
        atol_mismatch = adiff > atol
        rtol_mismatch = torch.where(nonzero_r, rel_diff > rtol, torch.full_like(atol_mismatch, False))
        mismatch = atol_mismatch & (~nonzero_r | rtol_mismatch)
    has_mismatch = torch.any(mismatch).item()

    max_rel_diff = 0.0 # Default to 0.0 if no non-zero reference values
    max_abs_diff = 0.0 
    max_abs_diff_indices = None
    max_rel_diff_indices = None

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
        abs_diff = torch.where(mismatch, adiff, 0)
        rel_diff = torch.where(mismatch, rel_diff, 0)
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        max_rel_diff_indices = torch.unravel_index(torch.argmax(rel_diff), rel_diff.shape)
        max_abs_diff_indices = torch.unravel_index(torch.argmax(abs_diff), diff.shape)

        num_mismatched_elements = torch.sum(mismatch).item()
        total_elements = t.numel() 
        base_msg = (
            f"There are tensor mismatches.\n"
            f"Number of mismatched elements: {num_mismatched_elements} out of {total_elements} total elements.\n"
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
        "fp8e4": get_torch_e4m3_type(),
        "fp8e5": get_torch_e5m2_type(),
    }[dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str]


