# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from __future__ import annotations

import types
from functools import partial
from itertools import product

import numpy as np
import pytest
import torch

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.triton_kernels.common import (
    torch_dtype_to_te_dtype,
    te_dtype_to_torch_dtype
)
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

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

# Mimics behavior of `fillUniform` from `tests/cpp/test_common.cu`.

rng_seed = 12345
rng = np.random.default_rng(np.random.MT19937(rng_seed))
norms = ["rms", "layer"]
test_shapes_by_norm = (
    (
        tuple(
            product(
                ("rms",), [
                    (2048, 4096),
                    (768, 2048),
                    (256, 1024),
                    (128, 768),
                    (64, 512),
                    (173, 409),
                    (71, 3571),
                    (29, 17389),
                ]
            )
        )
    ) + (
        tuple(
            product(
                ("layer",), [
                    (2048, 12288),
                    (768, 1024),
                    (256, 65536),
                    (128, 6144),
                    (64, 2304),
                    (229, 541),
                    (71, 3571),
                    (29, 17389),
                ]
            )
        )
    )
)
# (quantization, columnwise, ln_out_mode)
test_quantizations = ((None, False, None),)
test_quantizations += tuple(
    product(
        ('fp8', 'mxfp8'),
        (True, False),
        (None, 'quantized')
    )
)

_triton_funcs = {
    "fwd": {
        "rms":te_rmsnorm_fwd_triton,
        "layer":te_layernorm_fwd_triton,
    },
    "bwd": {
        "rms":te_rmsnorm_bwd_triton,
        "layer":te_layernorm_bwd_triton,
    }
}
_hip_funcs = {
    "fwd": {
        "rms":tex.rmsnorm_fwd,
        "layer":tex.layernorm_fwd,
    },
    "bwd": {
        "rms":tex.rmsnorm_bwd,
        "layer":tex.layernorm_bwd,
    }
}

# Add `i` prefix to identify input type.
def input_dtypes_str(dtypes_str):
    return ["i" + dtype_str for dtype_str in dtypes_str]


# Add `o` prefix to identify output type.
def output_dtypes_str(dtypes_str):
    return ["o" + dtype_str for dtype_str in dtypes_str]


def _make_test_dtype_pairs(test_types):
    for i, o in product(test_types, test_types):
        i_type, i_width = i
        o_type, o_width = o
        # We observe a strict inequality since the kernels do not allow for
        # mixed fp16/bf16.
        if i_width > o_width:
            yield ("i"+i_type, "o"+o_type)

test_dtypes_types = [("fp32", 32), ("fp16", 16), ("bf16", 16)]
test_dtype_pairs = list(_make_test_dtype_pairs(test_dtypes_types))


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


def mxfp8_layernorm_fwd_ref(
    input,
    weight,
    bias,
    eps,
    ln_out,
    quantizer,
    otype,
    sm_margin,
    zero_centered_gamma
):
    # Dummy function to serve as a stand-in for a reference HIP implementation
    input = input.to(torch.float32)
    mu = torch.mean(input, dim=1, keepdim=True)
    variance = torch.mean((input - mu).square(), dim=1, keepdim=True)
    inv_var = torch.rsqrt(variance + eps)
    g_tensor = weight + int(zero_centered_gamma)
    x_normed = (input - mu) * inv_var * g_tensor + bias

    assert isinstance(quantizer, MXFP8Quantizer)
    out = (
        quantizer.make_empty(
            input.shape,
            dtype=te_dtype_to_torch_dtype(otype),
            device=input.device
        )
    )
    out = quantizer.quantize(x_normed, out=out)
    return out, mu.squeeze(1), inv_var.squeeze(1)

def mxfp8_rmsnorm_fwd_ref(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    otype,
    sm_margin,
    zero_centered_gamma
):
    # Dummy function to serve as a stand-in for a reference HIP implementation
    norm_x = torch.mean(input * input, dim=1, keepdim=True)
    rsigma = torch.rsqrt(norm_x + eps)
    g_tensor = weight + int(zero_centered_gamma)
    x_normed = input * rsigma * g_tensor

    assert isinstance(quantizer, MXFP8Quantizer)
    out = (
        quantizer.make_empty(
            input.shape,
            dtype=te_dtype_to_torch_dtype(otype),
            device=input.device
        )
    )
    out = quantizer.quantize(x_normed, out=out)
    return out, None, rsigma.squeeze(1)

class TestNorms:

    @pytest.mark.parametrize(
        ("norm", "shape"),
        test_shapes_by_norm,
        ids=(f"{norm}-{s[0]}-{s[1]}" for norm, s in test_shapes_by_norm)
    )
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    @pytest.mark.parametrize(
        ("in_dtype", "out_dtype"),
        test_dtype_pairs,
        ids=(f"{i}-{o}" for i,o in test_dtype_pairs)
    )
    @pytest.mark.parametrize(
        ("quantization", "columnwise", "ln_out_mode"),
        test_quantizations,
        ids=(f"{q}-{c}-{l}" for q,c,l in test_quantizations)
    )
    def test_norm_triton(
        self,
        shape,
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
        M, N = shape
        in_dtype = str_to_torch_dtype(in_dtype)
        out_dtype = str_to_torch_dtype(out_dtype)
        te_out_dtype = torch_dtype_to_te_dtype(out_dtype)

        input_tensor = fill_uniform((M, N), in_dtype)
        gamma_tensor = fill_uniform(N, in_dtype)
        bias_tensor = fill_uniform(N, in_dtype)

        self._check_skips(quantization=quantization, shape=(M, N))

        epsilon = 1e-5

        quantizer_triton, quantizer_hip = self._make_quantizer(
            quantization=quantization,
            fp8_dtype=fp8_dtype,
            columnwise=columnwise
        )

        fwd_args = self._make_fwd_args(
            norm=norm,
            ln_out_mode=ln_out_mode,
            input_tensor=input_tensor,
            gamma_tensor=gamma_tensor,
            bias_tensor=bias_tensor,
            epsilon=epsilon,
            out_dtype=out_dtype,
            te_out_dtype=te_out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            quantizer_triton=quantizer_triton,
            quantizer_hip=quantizer_hip,
        )

        triton_fwd_func = _triton_funcs["fwd"][norm]
        hip_fwd_func = _hip_funcs["fwd"][norm]

        # TODO(micky774): Remove when we have HIP kernels to test against
        if quantization == "mxfp8":
            if norm == "layer":
                hip_fwd_func = mxfp8_layernorm_fwd_ref
            elif norm == "rms":
                hip_fwd_func = mxfp8_rmsnorm_fwd_ref

        # run the triton path
        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(**fwd_args["triton"])

        # run the reference hipified kernel path
        ln_out_hip, mu_hip, rsigma_hip = hip_fwd_func(**fwd_args["hip"])

        self._compare_output_tensors(
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
        self._compare_stat_tensors(
            rsigma_triton=rsigma_triton,
            rsigma_hip=rsigma_hip,
            mu_triton=mu_triton,
            mu_hip=mu_hip,
            norm=norm
        )

        dz = fill_uniform((M, N), in_dtype)

        triton_bwd_func = _triton_funcs["bwd"][norm]
        hip_bwd_func = _hip_funcs["bwd"][norm]

        # Backwards kernels do not support quantization
        if quantization is not None:
            return

        args = self._make_bwd_args(
            norm=norm,
            dz=dz,
            input_tensor=input_tensor,
            rsigma_triton=rsigma_triton,
            rsigma_hip=rsigma_hip,
            mu_triton=mu_triton,
            mu_hip=mu_hip,
            gamma_tensor=gamma_tensor,
            zero_centered_gamma=zero_centered_gamma,

        )
        triton_bwd_outs = triton_bwd_func(*args["triton"])

        if norm == "layer":
            dx_triton, dgamma_triton, dbeta_triton = triton_bwd_outs
        elif norm == "rms":
            dx_triton, dgamma_triton = triton_bwd_outs
            dbeta_triton = None

        hip_bwd_outs = hip_bwd_func(*args["hip"])

        if norm == "layer":
            dx_hip, dgamma_hip, dbeta_hip = hip_bwd_outs
        elif norm == "rms":
            dx_hip, dgamma_hip = hip_bwd_outs
            dbeta_hip = None

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

    @pytest.mark.parametrize("norm", norms)
    @pytest.mark.parametrize("columnwise", [False, True])
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

        quantization = 'fp8'
        fp8_dtype = tex.DType.kFloat8E4M3
        gamma_tensor = torch.tensor([2**20] + [0]*127, dtype=in_dtype, device="cuda")

        self._check_skips(quantization=quantization, shape=(M, N))

        quantizer_triton, quantizer_hip = self._make_quantizer(
            quantization=quantization,
            fp8_dtype=fp8_dtype,
            columnwise=columnwise
        )

        fwd_args = self._make_fwd_args(
            norm=norm,
            ln_out_mode=None,
            input_tensor=input_tensor,
            gamma_tensor=gamma_tensor,
            bias_tensor=bias_tensor,
            epsilon=epsilon,
            out_dtype=out_dtype,
            te_out_dtype=te_out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            quantizer_triton=quantizer_triton,
            quantizer_hip=quantizer_hip,
        )

        triton_fwd_func = _triton_funcs["fwd"][norm]
        hip_fwd_func = _hip_funcs["fwd"][norm]

        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(**fwd_args["triton"])
        ln_out_hip, mu_hip, rsigma_hip = hip_fwd_func(**fwd_args["hip"])

        if ln_out_triton.dtype != out_dtype:
            raise ValueError(f"Expected dtypes to match: {ln_out_triton.dtype} != {out_dtype}")

        self._compare_output_tensors(
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
        self._compare_stat_tensors(
            rsigma_triton=rsigma_triton,
            rsigma_hip=rsigma_hip,
            mu_triton=mu_triton,
            mu_hip=mu_hip,
            norm=norm
        )


    def _compare_output_tensors(
        self,
        out_triton, out_hip,
        quantization, fp8_dtype
        ):
        tols = dtype_tols(out_triton.dtype if quantization is None else fp8_dtype)
        _compare_func = partial(compare_results, provider="te", atol=tols["atol"], rtol=tols["rtol"])

        _compare_func(
            actual=out_triton,
            expected=out_hip,
            msg=lambda msg: f"Output does not match triton <-> hip\n\n{msg}\n",
        )
        # TODO(micky774): Remove when `compare_results` correctly handles NaN values
        _compare_func(
            actual=out_triton.isnan(),
            expected=out_hip.isnan(),
            msg=lambda msg: f"ln_out NaNs do not match triton <-> hip\n\n{msg}\n",
        )

        if quantization == "fp8":
            if not isinstance(out_triton, Float8Tensor):
                raise ValueError(f"Expected a Float8Tensor but got {type(out_triton)} instead.")

            if out_triton._transpose_invalid != out_hip._transpose_invalid:
                msg = "Expected a" 
                msg += "n in" if out_hip._transpose_invalid else " "
                msg += "valid transpose buffer."
                raise ValueError(msg)

            if not out_hip._transpose_invalid:
                # The transpose data are generally uint8 so we must convert
                # them for floating point comparison.
                _compare_func(
                    actual=out_triton._transpose.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    expected=out_hip._transpose.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    msg=lambda msg: f"Output transpose does not match triton <-> hip\n\n{msg}\n",
                )

        elif quantization == "mxfp8":
            if not isinstance(out_triton, MXFP8Tensor):
                raise ValueError(f"Expected a MXFP8Tensor but got {type(out_triton)} instead.")

            # TODO(micky774): Figure out if we need to apply the same view
            # trick to MXFP8 data as we do to FP8 transpose data.
            # I suspect not.
            if out_hip._rowwise_data is not None:
                _compare_func(
                    actual=out_triton,
                    expected=out_hip,
                    msg=lambda msg: f"Output rowwise data does not match triton <-> hip\n\n{msg}\n",
                )
                out_triton._rowwise_data = None
            else:
                assert out_triton._rowwise_data is None, "Expected no rowwise data."

        # We use higher precision for the scales
        _compare_func = partial(compare_results, provider="te", atol=1e-6, rtol=5e-5)
        if quantization == "fp8":
            _compare_func(
                actual=out_triton._scale_inv,
                expected=out_hip._scale_inv,
                msg=lambda msg: f"Output scale inverse does not match triton <-> hip\n\n{msg}\n",
            )
        elif quantization == "mxfp8":
            has_rscale_triton = out_triton._rowwise_scale_inv is not None
            has_rscale_hip = out_hip._rowwise_scale_inv is not None
            if has_rscale_triton != has_rscale_hip:
                msg = "Expected rowwise scale to "
                if has_rscale_hip:
                   msg += "not "
                msg += "be None."
                raise ValueError(msg)
            if has_rscale_triton:
                _compare_func(
                    actual=out_triton._rowwise_scale_inv.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    expected=out_hip._rowwise_scale_inv.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    msg=lambda msg: f"Output rowwise scale inverse does not match triton <-> hip\n\n{msg}\n",
                )

            has_cscale_triton = out_triton._columnwise_scale_inv is not None
            has_cscale_hip = out_hip._columnwise_scale_inv is not None
            if has_cscale_triton != has_cscale_hip:
                msg = "Expected columnwwise scale to "
                if has_cscale_hip:
                   msg += "not "
                msg += "be None."
                raise ValueError(msg)
            if has_cscale_triton:
                _compare_func(
                    actual=out_triton._columnwise_scale_inv.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    expected=out_hip._columnwise_scale_inv.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)),
                    msg=lambda msg: f"Output columnwise scale inverse does not match triton <-> hip\n\n{msg}\n",
                )


    def _compare_quantizers(
        self,
        quantizer_triton, quantizer_hip,
        quantization
    ):
        if quantization is None: return
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
                actual=quantizer_triton.scale,
                expected=quantizer_hip.scale,
                msg=lambda msg: f"Quantizer scale does not match triton <-> hip\n\n{msg}\n",
            )
            _compare_func(
                actual=quantizer_triton.amax,
                expected=quantizer_hip.amax,
                msg=lambda msg: f"Quantizer amax does not match triton <-> hip\n\n{msg}\n",
            )

    def _compare_stat_tensors(
        self,
        rsigma_triton, rsigma_hip,
        mu_triton, mu_hip,
        norm
    ):
        # We use higher precision for the remaining outputs
        _compare_func = partial(compare_results, provider="te", atol=1e-6, rtol=5e-5)

        _compare_func(
            actual=rsigma_triton,
            expected=rsigma_hip,
            msg=lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            _compare_func(
                actual=mu_triton,
                expected=mu_hip,
                msg=lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
            )

    def _check_skips(self, quantization, shape):
        # Check if quantization scheme is supported
        if quantization == "fp8" and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if quantization == "mxfp8":
            if not mxfp8_available:
                pytest.skip(reason_for_no_mxfp8)
            if shape[0] % 32 !=0 or shape[1] % 32 !=0:
                pytest.skip("MXFP8 quantization requires dimensions divisible by 32.")


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
        dx_triton, dx_hip,
        dgamma_triton, dgamma_hip,
        dbeta_triton, dbeta_hip,
        norm
    ):
        _compare_func = partial(compare_results, provider="te", atol=1.5e-4, rtol=1e-4)

        _compare_func(
            actual=dx_triton,
            expected=dx_hip,
            msg=lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n",
        )
        _compare_func(
            actual=dgamma_triton,
            expected=dgamma_hip,
            msg=lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            _compare_func(
                actual=dbeta_triton,
                expected=dbeta_hip,
                msg=lambda msg: f"dbeta does not match triton <-> hip\n\n{msg}\n",
            )

    def _make_bwd_args(self, norm, **kwargs):
        # The HIP implementation requires positional only args
        args = {}
        for provider in ("triton", "hip"):
            _args = (kwargs["dz"], kwargs["input_tensor"])
            if norm == "layer":
                _args += (kwargs[f"mu_{provider}"],)
            _args += (
                kwargs[f"rsigma_{provider}"],
                kwargs["gamma_tensor"],
                get_ln_sm_margin("BWD"),
                kwargs["zero_centered_gamma"],
            )
            args[provider] = _args
        return args

    def _make_fwd_args(self, norm, **kwargs):
        args = {}
        for provider in ("triton", "hip"):
            _args = dict(
                input=kwargs["input_tensor"],
                weight=kwargs["gamma_tensor"],
                eps=kwargs["epsilon"],
                ln_out=(
                    kwargs[f"quantizer_{provider}"].make_empty(
                        kwargs["input_tensor"].shape,
                        dtype=kwargs["out_dtype"]
                    ) if kwargs["ln_out_mode"] is not None else None
                ),
                otype=kwargs["te_out_dtype"],
                sm_margin=get_ln_sm_margin("FWD"),
                zero_centered_gamma=kwargs["zero_centered_gamma"],
                quantizer=kwargs[f"quantizer_{provider}"],
            )
            if norm == "layer":
                _args["bias"] = kwargs["bias_tensor"]

            args[provider] = _args
        return args
