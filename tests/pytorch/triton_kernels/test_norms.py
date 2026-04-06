# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import math
import os
import torch
import pytest
from functools import partial
from itertools import product

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.triton_kernels.utils import get_ln_sm_margin
from transformer_engine.pytorch.triton_kernels.common import (
    torch_dtype_to_te_dtype,
    te_dtype_to_torch_dtype
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch.triton_kernels.norms_common import (
    te_layernorm_bwd_triton,
    te_layernorm_fwd_triton,
    te_rmsnorm_bwd_triton,
    te_rmsnorm_fwd_triton,
)
from test_common import dtype_tols, te_compare_results, str_to_torch_dtype, fill_uniform

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

def _make_test_dtype_pairs(test_types):
    for i, o in product(test_types, test_types):
        i_type, i_width = i
        o_type, o_width = o
        # We observe a strict inequality since the kernels do not allow for
        # mixed fp16/bf16.
        if i_width > o_width:
            yield ("i"+i_type, "o"+o_type)
        elif i_type == o_type:
            yield ("i"+i_type, "o"+o_type)

# Note that these test dtypes are used as in_dtype and out_dtype which do NOT
# determine the underlying fp8 dtype used for quantized representation.
# Instead, out_dtype refers to the "fake" dtype used by the quantized tensors
# when upcasting in non-quantized contexts. 
test_dtypes_types = [("fp32", 32), ("fp16", 16), ("bf16", 16)]
test_dtype_pairs = list(_make_test_dtype_pairs(test_dtypes_types))

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

def layernorm_fwd_ref(
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

def rmsnorm_fwd_ref(
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
    input = input.to(torch.float32)
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


@pytest.fixture
def autotune():
    return bool(int(os.environ.get("NVTE_TEST_TRITON_AUTOTUNE", "0")))

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
        autotune,
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

        self._check_skips(quantization=quantization, shape=(M, N), colwise=columnwise)

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
                hip_fwd_func = layernorm_fwd_ref
            elif norm == "rms":
                hip_fwd_func = rmsnorm_fwd_ref

        # run the triton path
        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(autotune=autotune, **fwd_args["triton"])

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

        # Backwards kernels do not support quantization
        if quantization is not None:
            return

        dz = fill_uniform((M, N), in_dtype)

        triton_bwd_func = _triton_funcs["bwd"][norm]
        hip_bwd_func = _hip_funcs["bwd"][norm]

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
    def test_norm_fwd_triton_clamp(self, columnwise, norm, autotune):
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

        self._check_skips(quantization=quantization, shape=(M, N), colwise=columnwise)

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

        ln_out_triton, mu_triton, rsigma_triton = triton_fwd_func(autotune=autotune, **fwd_args["triton"])
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
        compare_func = partial(te_compare_results, **tols, use_torch_semantics=True)

        dq_out_triton = out_triton.dequantize()
        dq_out_hip = out_hip.dequantize()
        compare_func(
            actual=dq_out_triton,
            expected=dq_out_hip,
            msg=lambda msg: f"Output does not match triton <-> hip\n\n{msg}\n",
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
                compare_func(
                    actual=out_triton._transpose.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)).to(torch.float32),
                    expected=out_hip._transpose.view(te_dtype_to_torch_dtype(out_triton._fp8_dtype)).to(torch.float32),
                    msg=lambda msg: f"Output transpose does not match triton <-> hip\n\n{msg}\n",
                )

        elif quantization == "mxfp8":
            if not isinstance(out_triton, MXFP8Tensor):
                raise ValueError(f"Expected a MXFP8Tensor but got {type(out_triton)} instead.")
            if out_hip._rowwise_data is not None:
                assert out_triton._rowwise_data is not None, "Expected rowwise data."
            else:
                assert out_triton._rowwise_data is None, "Expected no rowwise data."

        # We use higher precision for the scales
        compare_func = partial(te_compare_results, atol=1e-6, rtol=5e-5, use_torch_semantics=True)
        if quantization == "fp8":
            compare_func(
                actual=out_triton._scale_inv,
                expected=out_hip._scale_inv,
                msg=lambda msg: f"Output scale inverse does not match triton <-> hip\n\n{msg}\n",
            )
        elif quantization == "mxfp8":
            has_rscale_triton = out_triton._rowwise_scale_inv is not None
            has_rscale_hip = out_hip._rowwise_scale_inv is not None

            # The scale_inv values may differ slightly, but will still dequantize close enough to 
            # pass the earlier comparisons.
            compare_func = partial(te_compare_results, atol=1, rtol=0, use_torch_semantics=True)

            if has_rscale_triton != has_rscale_hip:
                msg = "Expected rowwise scale to "
                if has_rscale_hip:
                   msg += "not "
                msg += "be None."
                raise ValueError(msg)
            if has_rscale_triton:
                compare_func(
                    actual=out_triton._rowwise_scale_inv,
                    expected=out_hip._rowwise_scale_inv,
                    msg=lambda msg: f"Output rowwise scale inverse does not match triton <-> hip\n\n{msg}\n",
                )

            has_cscale_triton = out_triton._columnwise_scale_inv is not None
            has_cscale_hip = out_hip._columnwise_scale_inv is not None
            if has_cscale_triton != has_cscale_hip:
                msg = "Expected columnwise scale to "
                if has_cscale_hip:
                   msg += "not "
                msg += "be None."
                raise ValueError(msg)
            if has_cscale_triton:
                compare_func(
                    actual=out_triton._columnwise_scale_inv,
                    expected=out_hip._columnwise_scale_inv,
                    msg=lambda msg: f"Output columnwise scale inverse does not match triton <-> hip\n\n{msg}\n",
                )

    def _compare_quantizers(
        self,
        quantizer_triton, quantizer_hip,
        quantization
    ):
        if quantization is None: return
        compare_func = partial(te_compare_results, atol=1e-6, rtol=5e-5, use_torch_semantics=True)

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
            compare_func(
                actual=quantizer_triton.scale,
                expected=quantizer_hip.scale,
                msg=lambda msg: f"Quantizer scale does not match triton <-> hip\n\n{msg}\n",
            )
            compare_func(
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
        compare_func = partial(te_compare_results, atol=1e-6, rtol=5e-5, use_torch_semantics=True)

        compare_func(
            actual=rsigma_triton,
            expected=rsigma_hip,
            msg=lambda msg: f"rsigma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            compare_func(
                actual=mu_triton,
                expected=mu_hip,
                msg=lambda msg: f"mu does not match triton <-> hip\n\n{msg}\n",
            )

    def _check_skips(self, quantization, shape, colwise):
        # Check if quantization scheme is supported
        if quantization == "fp8" and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if quantization == "mxfp8":
            if not mxfp8_available:
                pytest.skip(reason_for_no_mxfp8)
            if shape[0] % 32:
                pytest.skip("MXFP8 quantization requires row dimensions divisible by 32.")
            if colwise and shape[1] % 32:
                pytest.skip("Colwise MXFP8 quantization requires col dimensions divisible by 32.") 


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
        compare_func = partial(te_compare_results, atol=1.5e-4, rtol=1e-4, use_torch_semantics=True)

        compare_func(
            actual=dx_triton,
            expected=dx_hip,
            msg=lambda msg: f"dx does not match triton <-> hip\n\n{msg}\n",
        )
        compare_func(
            actual=dgamma_triton,
            expected=dgamma_hip,
            msg=lambda msg: f"dgamma does not match triton <-> hip\n\n{msg}\n",
        )
        if norm == "layer":
            compare_func(
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
