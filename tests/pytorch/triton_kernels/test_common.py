# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from __future__ import annotations

import types
from typing import Optional
from collections.abc import Iterable
from functools import partial

import numpy as np
import pytest
import torch
import math

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.triton_kernels.common import torch_dtype_to_te_dtype
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch.triton_kernels.norm_common import get_ln_sm_margin
from transformer_engine.pytorch.triton_kernels.rmsnorm import (
    te_rmsnorm_bwd_triton,
    te_rmsnorm_fwd_triton,
)
from transformer_engine.pytorch.triton_kernels.layernorm import (
    te_layernorm_bwd_triton,
    te_layernorm_fwd_triton,
)

from transformer_engine.pytorch.triton_kernels.common import (
    get_torch_e4m3_type,
    get_torch_e5m2_type,
)

# Mimics behavior of `fillUniform` from `tests/cpp/test_common.cu`.

rng_seed = 12345
rng = np.random.default_rng(np.random.MT19937(rng_seed))
norms = ["rms", "layer"]
test_shapes = [
    (2048, 4096),
    (768, 2048),
    (256, 1024),
    (128, 768),
    (64, 512),
    (173, 409),
    (71, 3571),
    (29, 17389),
]
_triton_funcs = {
    {
        "fwd": {
            "rms":te_rmsnorm_fwd_triton,
            "layer":te_layernorm_fwd_triton,
        },
        "bwd": {
            "rms":te_rmsnorm_bwd_triton,
            "layer":te_layernorm_bwd_triton,
        }
    }
}
_hip_funcs = {
    {
        "fwd": {
            "rms":tex.rmsnorm_fwd,
            "layer":tex.layernorm_fwd,
        },
        "bwd": {
            "rms":tex.rmsnorm_bwd,
            "layer":tex.layernorm_bwd,
        }
    }
}


# Add `i` prefix to identify input type.
def input_dtypes_str(dtypes_str):
    return ["i" + dtype_str for dtype_str in dtypes_str]


# Add `o` prefix to identify output type.
def output_dtypes_str(dtypes_str):
    return ["o" + dtype_str for dtype_str in dtypes_str]

test_types_str = ["fp32", "fp16", "bf16"]
test_idtypes_str = input_dtypes_str(test_types_str)
test_odtypes_str = output_dtypes_str(test_types_str + ["fp8e4"])

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
    rel_diff = torch.where(nonzero_r, torch.abs(diff / r), torch.zeros_like(diff))
    rtol_mismatch = torch.where(nonzero_r, rel_diff > rtol, torch.full_like(atol_mismatch, False))
    mismatch = atol_mismatch & (~nonzero_r | rtol_mismatch)
    has_mismatch = torch.any(mismatch).item()

    max_rel_diff = 0.0 # Default to 0.0 if no non-zero reference values
    max_abs_diff = 0.0 
    max_abs_diff_indices = None
    max_rel_diff_indices = None
    
    if has_mismatch:
        max_abs_diff = torch.max(torch.abs(diff)).item()
        max_rel_diff = torch.max(rel_diff).item()
        max_rel_diff_indices = torch.unravel_index(torch.argmax(rel_diff), rel_diff.shape)
        max_abs_diff_indices = torch.unravel_index(torch.argmax(torch.abs(diff)), diff.shape)

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


# Convert descriptive type string to PyTorch type.
def str_to_torch_dtype(dtype_str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8e4": get_torch_e4m3_type(),
        "fp8e5": get_torch_e5m2_type(),
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


# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()


def maybe_skip_quantization(
    quantization: Optional[str],
    *,
    dims: Optional[Iterable[int] | int] = None,
    device: Optional[torch.device | str] = None,
) -> None:

    # Don't skip if there is no quantization
    if quantization is None:
        return

    # Check if quantization scheme is supported
    if quantization == "fp8" and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = (dims,)
        if quantization == "fp8":
            if math.prod(dims[:-1]) % 16 != 0 or dims[-1] % 16 != 0:
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")
        elif quantization == "mxfp8":
            if math.prod(dims[:-1]) % 32 != 0 or dims[-1] % 32 != 0:
                pytest.skip("MXFP8 GEMMs require dims that are divisible by 32")

    # Check if device is supported
    if device is not None and torch.device(device).type != "cuda":
        pytest.skip("Quantization is only supported on CUDA devices")


@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("norm", norms)
class TestNorms:

    @pytest.mark.parametrize("M, N", test_shapes)
    @pytest.mark.parametrize("in_dtype", test_idtypes_str)
    @pytest.mark.parametrize("out_dtype", test_odtypes_str)
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    @pytest.mark.parametrize("quantization", (None, 'fp8', 'mxfp8'))
    @pytest.mark.parametrize("ln_out_mode", (None, "quantized"))
    def test_norm_triton(
        self,
        M, N,
        in_dtype,
        out_dtype,
        zero_centered_gamma,
        quantization,
        columnwise,
        ln_out_mode,
        norm,
    ):
        # We only support 8E4M3 for forward kernels
        fp8_dtype = tex.DType.kFloat8E4M3

        in_dtype = str_to_torch_dtype(in_dtype)
        out_dtype = str_to_torch_dtype(out_dtype)
        te_out_dtype = torch_dtype_to_te_dtype(out_dtype)

        input_tensor = fill_uniform((M, N), in_dtype)
        gamma_tensor = fill_uniform(N, in_dtype)
        bias_tensor = fill_uniform(N, in_dtype)

        self._check_skips(
            quantization=quantization,
            shape=(M, N),
            in_dtype=in_dtype,
            out_dtype=out_dtype
        )
        if quantization is None:
            if columnwise:
                pytest.skip("Columnwise only affects quantized calls.")
            if ln_out_mode == "quantized":
                pytest.skip("Quantized output container only affects quantized calls.")


        epsilon = 1e-5
        fwd_ln_sm_margin = get_ln_sm_margin("FWD")

        quantizer_triton, quantizer_hip = self._make_quantizer(
            quantization=quantization,
            fp8_dtype=fp8_dtype,
            columnwise=columnwise
        )

        args = dict(
            input=input_tensor,
            weight=gamma_tensor,
            eps=epsilon,
            ln_out=(
                quantizer_triton.make_empty(input.shape, dtype=out_dtype)
                if ln_out_mode is not None else None
            ),
            otype=te_out_dtype,
            sm_margin=fwd_ln_sm_margin,
            zero_centered_gamma=zero_centered_gamma
        )
        if norm == "layer":
            args |= dict(bias=bias_tensor)

        triton_fwd_func = _triton_funcs["fwd"][norm]
        hip_fwd_func = _hip_funcs["fwd"][norm]

        # run the triton path
        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(**args)

        # run the reference hipified kernel path
        args["ln_out"] = (
            quantizer_hip.make_empty(input.shape, dtype=out_dtype)
            if ln_out_mode is not None else None
        )
        ln_out_hip, mu_hip, rsigma_hip = hip_fwd_func(**args)

        if ln_out_triton.dtype != out_dtype:
            raise ValueError(f"Expected dtypes to match: {ln_out_triton.dtype} != {out_dtype}")

        self._compare_quantized_tensors(
            out_triton=ln_out_triton,
            out_hip=ln_out_hip,
            quantization=quantization,
            fp8_dtype=fp8_dtype,
        )
        self._compare_quantizers(
            quantizer_triton=quantizer_triton,
            quantizer_hip=quantizer_hip,
            quantization=quantization
        )
        self._compare_non_quantized_outputs(
            rsigma_triton=rsigma_triton,
            rsigma_hip=rsigma_hip,
            mu_triton=mu_triton,
            mu_hip=mu_hip,
            norm=norm
        )

        dz = fill_uniform((M, N), in_dtype)
        bwd_ln_sm_margin = get_ln_sm_margin("BWD")

        triton_bwd_func = _triton_funcs["bwd"][norm]
        hip_bwd_func = _hip_funcs["bwd"][norm]

        args = dict(
            dz=dz,
            x=input_tensor,
            mu=mu_triton,
            rsigma=rsigma_triton,
            gamma=gamma_tensor,
            sm_margin=bwd_ln_sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )
        dx_triton, dgamma_triton, dbeta_triton = triton_bwd_func(
        )

        args["rsigma"] = rsigma_hip
        dx_hip, dgamma_hip, dbeta_hip = hip_bwd_func(
            dz,
            input_tensor,
            mu_hip,
            rsigma_hip,
            gamma_tensor,
            bwd_ln_sm_margin,
            zero_centered_gamma,
        )

        # Assert on dx, dgamma and dbeta:
        self._compare_bwd_tensors(
            dx_triton=dx_triton,
            dx_hip=dx_hip,
            dgamma_triton=dgamma_triton,
            dgamma_hip=dgamma_hip,
            dbeta_triton=dbeta_triton,
            dbeta_hip=dbeta_hip,
            norm=norm
            )

    def test_norm_fwd_triton_clamp(self, columnwise, norm):
        """
        Non-regression test for MLPerf divergence issue. We test to ensure that in
        the case of output values beyond the range of the used FP8 dtype, we clamp
        them appropriately.
        """
        # Arbitrary
        M, N = (128, 128)
        zero_centered_gamma = True
        in_dtype = str_to_torch_dtype("fp32")
        out_dtype = str_to_torch_dtype("fp32")
        te_out_dtype = torch_dtype_to_te_dtype(out_dtype)

        input_tensor = torch.full((M, N), 1, dtype=in_dtype, device="cuda")
        bias_tensor = fill_uniform(N, in_dtype)

        epsilon = 1e-5
        fwd_ln_sm_margin = get_ln_sm_margin("FWD")

        quantization = 'fp8'
        fp8_dtype = tex.DType.kFloat8E4M3
        gamma_tensor = torch.tensor([2**20] + [0]*127, dtype=in_dtype, device="cuda")

        self._check_skips(
            quantization=quantization,
            shape=(M, N),
            in_dtype=in_dtype,
            out_dtype=out_dtype
        )

        quantizer_triton, quantizer_hip = self._make_quantizer(
            quantization=quantization,
            fp8_dtype=fp8_dtype,
            columnwise=columnwise
        )

        args = dict(
            input=input_tensor,
            weight=gamma_tensor,
            eps=epsilon,
            ln_out=None,
            otype=te_out_dtype,
            sm_margin=fwd_ln_sm_margin,
            zero_centered_gamma=zero_centered_gamma
        )
        if norm == "layer":
            args |= dict(bias=bias_tensor)

        triton_fwd_func = _triton_funcs["fwd"][norm]
        hip_fwd_func = _hip_funcs["fwd"][norm]

        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(**args)
        ln_out_hip, mu_hip, rsigma_hip = hip_fwd_func(**args)

        if ln_out_triton.dtype != out_dtype:
            raise ValueError(f"Expected dtypes to match: {ln_out_triton.dtype} != {out_dtype}")

        self._compare_quantized_tensors(
            out_triton=ln_out_triton,
            out_hip=ln_out_hip,
            quantization=quantization,
            fp8_dtype=fp8_dtype,
        )
        self._compare_quantizers(
            quantizer_triton=quantizer_triton,
            quantizer_hip=quantizer_hip,
            quantization=quantization
        )
        self._compare_non_quantized_outputs(
            rsigma_triton=rsigma_triton,
            rsigma_hip=rsigma_hip,
            mu_triton=mu_triton,
            mu_hip=mu_hip,
            norm=norm
        )


    def _compare_quantized_tensors(
        self,
        out_triton, out_hip,
        quantization, fp8_dtype
        ):
        tols = dtype_tols(out_triton.dtype if quantization is None else fp8_dtype)
        _compare_func = partial(compare_results, provider="te", atol=tols["atol"], rtol=tols["rtol"])
        _compare_func(
            out_triton,
            out_hip,
            lambda msg: f"Output does not match triton <-> hip\n\n{msg}\n",
        )
        if quantization == "fp8":
            if not isinstance(out_triton, Float8Tensor):
                raise ValueError(f"Expected a Float8Tensor but got {type(out_triton)} instead.")

            if out_triton._transpose_invalid != out_hip._transpose_invalid:
                msg = "Expected a" 
                msg += "n in" if out_hip._transpose_invalid else ""
                msg += "valid transpose buffer."
                raise ValueError(msg)

            if not out_hip._transpose_invalid:
                _compare_func(
                    out_triton._transpose,
                    out_hip._transpose,
                    lambda msg: f"Output transpose does not match triton <-> hip\n\n{msg}\n",
                )

        elif quantization == "mxfp8":
            if not isinstance(out_triton, MXFP8Tensor):
                raise ValueError(f"Expected a MXFP8Tensor but got {type(out_triton)} instead.")

            if out_hip._rowwise_data is not None:
                _compare_func(
                    out_triton._rowwise_data,
                    out_hip._rowwise_data,
                    lambda msg: f"Output rowwise data does not match triton <-> hip\n\n{msg}\n",
                )
            else:
                assert out_triton._rowwise_data is None, "Expected no rowwise data."

            if out_hip._columnwise_data is not None:
                _compare_func(
                    out_triton._columnwise_data,
                    out_hip._columnwise_data,
                    lambda msg: f"Output columnwise data does not match triton <-> hip\n\n{msg}\n",
                )
            else:
                assert out_triton._columnwise_data is None, "Expected no columnwise data."


        # We use higher precision for the scales
        _compare_func = partial(compare_results, provider="te", atol=1e-6, rtol=5e-5)
        if quantization == "fp8":
            _compare_func(
                out_triton._scale_inv,
                out_hip._scale_inv,
                lambda msg: f"Output scale inverse does not match triton <-> hip\n\n{msg}\n",
            )
        elif quantization == "mxfp8":
            _compare_func(
                out_triton._rowwise_scale_inv,
                out_hip._rowwise_scale_inv,
                lambda msg: f"Output rowwise scale inverse does not match triton <-> hip\n\n{msg}\n",
            )
            _compare_func(
                out_triton._columnwise_scale_inv,
                out_hip._columnwise_scale_inv,
                lambda msg: f"Output columnwise scale inverse does not match triton <-> hip\n\n{msg}\n",
            )


    def _compare_quantizers(
        self,
        quantizer_triton, quantizer_hip,
        quantization
    ):
        _compare_func = partial(compare_results, provider="te", atol=1e-6, rtol=5e-5)

        if quantizer_triton.dtype != quantizer_hip.dtype:
            raise ValueError("Expected matching quantizer dtypes, but got "
                             f"{quantizer_triton.dtype} != {quantizer_hip.dtype}"
                            )
        for usage in ("rowwise_usage", "columnwise_usage"):
            qt_usage = getattr(quantizer_triton, usage)
            qh_usage = getattr(quantizer_triton, usage)
            if qt_usage != qh_usage:
                raise ValueError(f"Expected matching quantizer {usage} but got {qt_usage=} != {qh_usage=}")

        if quantization == "fp8":
            _compare_func(
                quantizer_triton.scale,
                quantizer_hip.scale,
                lambda msg: f"Quantizer scale does not match triton <-> hip\n\n{msg}\n",
            )
            _compare_func(
                quantizer_triton.amax,
                quantizer_hip.amax,
                lambda msg: f"Quantizer amax does not match triton <-> hip\n\n{msg}\n",
            )

    def _compare_non_quantized_outputs(
        self,
        rsigma_triton, rsigma_hip,
        mu_triton, mu_hip,
        norm
    ):
        # We use higher precision for the remaining outputs
        _compare_func = partial(compare_results, provider="te", atol=1e-6, rtol=5e-5)
        _compare_func(
            rsigma_triton,
            rsigma_hip,
            lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            _compare_func(
                mu_triton,
                mu_hip,
                lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
            )

    def _check_skips(self, quantization, shape, in_dtype, out_dtype):
        maybe_skip_quantization(quantization, dims=shape, device="cuda")
        skip_in_dtype_gt_out_dtype(in_dtype, out_dtype)
        skip_mixed_16bit_float_types(in_dtype, out_dtype)

    def _make_quantizer(self, quantization, fp8_dtype, columnwise):
        if quantization == "fp8":
            scale_triton = torch.rand(1, dtype=torch.float32, device='cuda') + 1
            amax_triton = torch.empty([1], dtype=torch.float32, device="cuda")

            scale_hip = scale_triton.clone()
            amax_hip = amax_triton.clone()

            quantizer_triton = Float8Quantizer(scale_triton, amax_triton, fp8_dtype, columnwise=columnwise)
            quantizer_hip = Float8Quantizer(scale_hip, amax_hip, fp8_dtype, columnwise=columnwise)
        elif quantization == "mxfp8":
            quantizer_triton = MXFP8Quantizer(fp8_dtype, columnwise=columnwise)
            quantizer_hip = MXFP8Quantizer(fp8_dtype, columnwise=columnwise)
        else:
            quantizer_triton = None
            quantizer_hip = None
        return quantizer_triton, quantizer_hip

    def _compare_bwd_tensors(
        self,
        dinput_triton, dinput_hip,
        dgamma_triton, dgamma_hip,
        dbeta_triton, dbeta_hip,
        norm
    ):
        _compare_func = partial(compare_results, provider="te", atol=1.5e-4, rtol=1e-4)
        _compare_func(
            dinput_triton,
            dinput_hip,
            lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n",
        )
        _compare_func(
            dgamma_triton,
            dgamma_hip,
            lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            _compare_func(
                dbeta_triton,
                dbeta_hip,
                lambda msg: f"dbeta does not match triton <-> hip\n\n{msg}\n",
            )
