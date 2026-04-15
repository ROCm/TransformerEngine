# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Lite-native LayerNormMLP: fused normalization + two-layer MLP."""

from typing import Callable, Dict, Optional, Tuple, Union, List

import torch
from torch.nn import Parameter

import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.utils import (
    cast_if_needed,
    get_default_init_method,
    init_method_constant,
)

from .fused_layernorm_linear import _get_normalization_funcs


__all__ = ["LayerNormMLP"]


_GATED_ACTIVATIONS = frozenset({
    "geglu", "qgeglu", "reglu", "sreglu", "swiglu", "clamped_swiglu",
})

# Maps activation name → (forward_fn, backward_fn, fused_dbias_dact_fn_or_None)
_ACT_FUNC_MAP = {
    "gelu":           (tex.gelu,           tex.dgelu,           tex.dbias_dgelu),
    "geglu":          (tex.geglu,          tex.dgeglu,          None),
    "qgelu":          (tex.qgelu,          tex.dqgelu,          tex.dbias_dqgelu),
    "qgeglu":         (tex.qgeglu,         tex.dqgeglu,         None),
    "relu":           (tex.relu,           tex.drelu,           tex.dbias_drelu),
    "reglu":          (tex.reglu,          tex.dreglu,          None),
    "srelu":          (tex.srelu,          tex.dsrelu,          tex.dbias_dsrelu),
    "sreglu":         (tex.sreglu,         tex.dsreglu,         None),
    "silu":           (tex.silu,           tex.dsilu,           tex.dbias_dsilu),
    "swiglu":         (tex.swiglu,         tex.dswiglu,         None),
    "clamped_swiglu": (tex.clamped_swiglu, tex.clamped_dswiglu, None),
}


def _gemm(A, transA, B, transB, bias, grad, quantizer=None, output_dtype=None,  # noqa: E501
          gelu=False, gelu_in=None, accumulate=False):
    """Thin wrapper around _lite generic_gemm with sane defaults."""
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]
    return tex.generic_gemm(
        A, transA, B, transB,
        None,            # D (allocate internally)
        quantizer,
        output_dtype,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,
        torch.empty(0),  # workspace (unused in lite)
        0,               # workspace_size
        accumulate,
        False,           # use_split_accumulator
    )


class _LayerNormMLPLite(torch.autograd.Function):
    """Autograd function for fused LayerNorm + MLP (lite backend)."""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Optional[torch.Tensor],
        fc1_weight: torch.Tensor,
        fc1_bias: Optional[torch.Tensor],
        fc2_weight: torch.Tensor,
        fc2_bias: Optional[torch.Tensor],
        eps: float,
        fp8: bool,
        fc1_input_quantizer: Optional[Quantizer],
        fc1_weight_quantizer: Optional[Quantizer],
        fc2_input_quantizer: Optional[Quantizer],
        fc2_weight_quantizer: Optional[Quantizer],
        fc2_grad_output_quantizer: Optional[Quantizer],
        fc1_grad_output_quantizer: Optional[Quantizer],
        fc1_grad_input_quantizer: Optional[Quantizer],
        fc1_grad_weight_quantizer: Optional[Quantizer],
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        normalization: str,
        zero_centered_gamma: bool,
        activation: str,
        activation_params: Optional[Dict],
        is_grad_enabled: bool,
        module: "LayerNormMLP",
        is_first_microbatch: Optional[bool],
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:

        act_fwd, act_bwd, dbias_dact = _ACT_FUNC_MAP[activation]
        is_gated = activation in _GATED_ACTIVATIONS

        # Reshape input
        hidden_size = fc1_weight.shape[1]
        inp_shape = inp.shape
        inputmat = inp.reshape(-1, hidden_size)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        # Configure norm quantizer
        backward_needs_input = is_grad_enabled and fc1_weight.requires_grad
        if fp8 and fc1_input_quantizer is not None:
            fc1_input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)

        with_quantized_norm = (
            fp8
            and fc1_input_quantizer is not None
            and not return_layernorm_output
        )

        # ---- Normalization ----
        norm_fwd, _ = _get_normalization_funcs(normalization)
        if normalization == "LayerNorm":
            ln_out, mu, rsigma = norm_fwd(
                inputmat, ln_weight, ln_bias, eps, None,
                fc1_input_quantizer if with_quantized_norm else None,
                inputmat.dtype, 0, zero_centered_gamma,
            )
        else:
            ln_out, mu, rsigma = norm_fwd(
                inputmat, ln_weight, eps, None,
                fc1_input_quantizer if with_quantized_norm else None,
                inputmat.dtype, 0, zero_centered_gamma,
            )

        ln_out_return = ln_out if return_layernorm_output else None

        # Quantize norm output if not already fused
        if fp8 and not with_quantized_norm and fc1_input_quantizer is not None:
            ln_out = fc1_input_quantizer(ln_out)

        # ---- Prepare FC1 weight ----
        fc1_weightmat = fc1_weight
        if fp8 and fc1_weight_quantizer is not None:
            fc1_weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            update = is_first_microbatch is None or is_first_microbatch
            fc1_weightmat = module.get_weight_workspace(
                tensor=fc1_weight, quantizer=fc1_weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "fc1_weight"),
                update_workspace=update,
            )
            fc1_weightmat.update_usage(rowwise_usage=True)
        else:
            fc1_weightmat = cast_if_needed(fc1_weightmat, activation_dtype)

        # ---- FC1 GEMM ----
        fc1_bias_cast = cast_if_needed(fc1_bias, activation_dtype) if fc1_bias is not None else None
        out_dtype = TE_DType[activation_dtype] if activation_dtype in TE_DType else None

        fc1_out, _, gelu_input, _ = _gemm(
            fc1_weightmat, True, ln_out, False,
            bias=fc1_bias_cast, grad=False, output_dtype=out_dtype,
        )

        # ---- Activation ----
        act_kwargs = activation_params or {}
        act_out = act_fwd(fc1_out, fc2_input_quantizer if fp8 else None, **act_kwargs)

        # ---- Prepare FC2 weight ----
        fc2_weightmat = fc2_weight
        if fp8 and fc2_weight_quantizer is not None:
            fc2_weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            update = is_first_microbatch is None or is_first_microbatch
            fc2_weightmat = module.get_weight_workspace(
                tensor=fc2_weight, quantizer=fc2_weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "fc2_weight"),
                update_workspace=update,
            )
            fc2_weightmat.update_usage(rowwise_usage=True)
        else:
            fc2_weightmat = cast_if_needed(fc2_weightmat, activation_dtype)

        # ---- FC2 GEMM ----
        fc2_bias_cast = cast_if_needed(fc2_bias, activation_dtype) if fc2_bias is not None else None
        fc2_out, _, _, _ = _gemm(
            fc2_weightmat, True, act_out, False,
            bias=fc2_bias_cast, grad=False, output_dtype=out_dtype,
        )

        out = fc2_out.view(-1, *inp_shape[1:-1], hidden_size)

        # ---- Save for backward ----
        if is_grad_enabled:
            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                ln_weight,
                ln_out,
                fc1_weightmat, fc1_weight, fc1_bias,
                fc1_out,
                act_out,
                fc2_weightmat, fc2_weight, fc2_bias,
                mu, rsigma,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.inp_shape = inp_shape
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.normalization = normalization
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.activation = activation
            ctx.activation_params = activation_params or {}
            ctx.use_fc1_bias = fc1_bias is not None
            ctx.use_fc2_bias = fc2_bias is not None
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = fc1_weight.requires_grad
            ctx.fc1_input_quantizer = fc1_input_quantizer
            ctx.fc2_input_quantizer = fc2_input_quantizer
            ctx.fc2_grad_output_quantizer = fc2_grad_output_quantizer
            ctx.fc1_grad_output_quantizer = fc1_grad_output_quantizer
            ctx.fc1_grad_input_quantizer = fc1_grad_input_quantizer
            ctx.fc1_grad_weight_quantizer = fc1_grad_weight_quantizer
            ctx.return_layernorm_output = return_layernorm_output

        if return_layernorm_output:
            return out, ln_out_return.view(inp_shape)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]

        saved_tensors = ctx.saved_tensors
        (
            inputmat,
            ln_weight,
            ln_out,
            fc1_weightmat, fc1_weight, fc1_bias,
            fc1_out,
            act_out,
            fc2_weightmat, fc2_weight, fc2_bias,
            mu, rsigma,
        ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
        ctx.tensor_objects = None

        act_fwd, act_bwd, dbias_dact = _ACT_FUNC_MAP[ctx.activation]
        out_dtype = TE_DType[ctx.activation_dtype] if ctx.activation_dtype in TE_DType else None
        hidden_size = fc1_weight.shape[1]

        grad_output = grad_output.reshape(-1, hidden_size)
        grad_output = cast_if_needed(grad_output, ctx.activation_dtype)

        # Quantize grad_output for FP8 (rowwise for FC2 dgrad, columnwise for FC2 wgrad)
        if ctx.fp8 and ctx.fc2_grad_output_quantizer is not None:
            ctx.fc2_grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            grad_output = ctx.fc2_grad_output_quantizer(grad_output)

        # ---- FC2 DGRAD: d_act = grad_output @ fc2_weight ----
        d_act, _, _, _ = _gemm(
            fc2_weightmat, False, grad_output, False,
            bias=None, grad=False, output_dtype=out_dtype,
        )

        # ---- FC2 WGRAD: dW2 = grad_output^T @ act_out (NT layout) ----
        dfc2_weight = None
        dfc2_bias = None
        if ctx.requires_wgrad:
            # Re-quantize act_out with columnwise usage for NT wgrad GEMM
            if ctx.fp8 and ctx.fc2_input_quantizer is not None:
                if isinstance(act_out, QuantizedTensorStorage):
                    act_out.update_usage(columnwise_usage=True)
                else:
                    ctx.fc2_input_quantizer.set_usage(rowwise=False, columnwise=True)
                    act_out = ctx.fc2_input_quantizer(act_out)

            # Ensure grad_output has columnwise usage for wgrad
            if ctx.fp8 and ctx.fc2_grad_output_quantizer is not None:
                if isinstance(grad_output, QuantizedTensorStorage):
                    grad_output.update_usage(columnwise_usage=True)

            dfc2_weight, dfc2_bias_grad, _, _ = _gemm(
                act_out, False, grad_output, True,
                bias=fc2_bias if ctx.use_fc2_bias else None,
                grad=True, output_dtype=out_dtype,
            )
            if ctx.use_fc2_bias:
                dfc2_bias = dfc2_bias_grad

        # ---- Activation backward + FC1 bias grad ----
        dfc1_bias = None
        if dbias_dact is not None and ctx.use_fc1_bias:
            # Fused bias gradient + activation backward
            dfc1_out, dfc1_bias = dbias_dact(d_act, fc1_out, None, **ctx.activation_params)
        else:
            # Separate activation backward
            dfc1_out = act_bwd(d_act, fc1_out, None, **ctx.activation_params)
            if ctx.use_fc1_bias:
                dfc1_bias = dfc1_out.reshape(-1, dfc1_out.shape[-1]).sum(dim=0)

        # Quantize dfc1_out (fc1_grad_output) for FC1 GEMMs
        if ctx.fp8 and ctx.fc1_grad_output_quantizer is not None:
            ctx.fc1_grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            dfc1_out = ctx.fc1_grad_output_quantizer(dfc1_out)

        # ---- FC1 DGRAD: d_ln_out = dfc1_out @ fc1_weight ----
        d_ln_out = None
        if ctx.requires_dgrad:
            # Quantize FC1 dgrad output
            dgrad_quantizer = None
            if ctx.fp8 and ctx.fc1_grad_input_quantizer is not None:
                ctx.fc1_grad_input_quantizer.set_usage(rowwise=True, columnwise=False)
                dgrad_quantizer = ctx.fc1_grad_input_quantizer

            d_ln_out, _, _, _ = _gemm(
                fc1_weightmat, False, dfc1_out, False,
                bias=None, grad=False, quantizer=dgrad_quantizer, output_dtype=out_dtype,
            )

        # ---- FC1 WGRAD: dW1 = dfc1_out^T @ ln_out (NT layout) ----
        dfc1_weight = None
        if ctx.requires_wgrad:
            # Re-quantize ln_out with columnwise usage for NT wgrad GEMM
            if ctx.fp8 and ctx.fc1_input_quantizer is not None:
                if isinstance(ln_out, QuantizedTensorStorage):
                    ln_out.update_usage(columnwise_usage=True)
                else:
                    ctx.fc1_input_quantizer.set_usage(rowwise=False, columnwise=True)
                    ln_out = ctx.fc1_input_quantizer(ln_out)

            # Ensure dfc1_out has columnwise usage for wgrad
            if ctx.fp8 and ctx.fc1_grad_output_quantizer is not None:
                if isinstance(dfc1_out, QuantizedTensorStorage):
                    dfc1_out.update_usage(columnwise_usage=True)

            dfc1_weight, _, _, _ = _gemm(
                ln_out, False, dfc1_out, True,
                bias=None, grad=False, quantizer=ctx.fc1_grad_weight_quantizer,
                output_dtype=out_dtype,
            )

        # ---- Norm backward ----
        dgrad = None
        dgamma = None
        dbeta = None
        if ctx.requires_dgrad:
            _, norm_bwd = _get_normalization_funcs(ctx.normalization)
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = norm_bwd(
                    d_ln_out, inputmat, mu, rsigma, ln_weight,
                    0, ctx.zero_centered_gamma,
                )
            else:
                dgrad, dgamma = norm_bwd(
                    d_ln_out, inputmat, rsigma, ln_weight,
                    0, ctx.zero_centered_gamma,
                )
            dgrad = dgrad.view(ctx.inp_shape)

        # Return gradients matching forward signature order
        return (
            dgrad,            # inp
            dgamma,           # ln_weight
            dbeta,            # ln_bias
            dfc1_weight,      # fc1_weight
            dfc1_bias,        # fc1_bias
            dfc2_weight,      # fc2_weight
            dfc2_bias,        # fc2_bias
            None,             # eps
            None,             # fp8
            None,             # fc1_input_quantizer
            None,             # fc1_weight_quantizer
            None,             # fc2_input_quantizer
            None,             # fc2_weight_quantizer
            None,             # fc2_grad_output_quantizer
            None,             # fc1_grad_output_quantizer
            None,             # fc1_grad_input_quantizer
            None,             # fc1_grad_weight_quantizer
            None,             # activation_dtype
            None,             # return_layernorm_output
            None,             # normalization
            None,             # zero_centered_gamma
            None,             # activation
            None,             # activation_params
            None,             # is_grad_enabled
            None,             # module
            None,             # is_first_microbatch
        )


class LayerNormMLP(TransformerEngineBaseModule):
    """Fused LayerNorm + MLP (lite-native, single-node).

    Applies normalization followed by a two-layer MLP:
        y = fc2(act(fc1(norm(x))))

    Parameters
    ----------
    hidden_size : int
        Input and output feature dimension.
    ffn_hidden_size : int
        Intermediate (FC1 output) feature dimension.
    eps : float, default = 1e-5
        Epsilon for normalization stability.
    bias : bool, default = True
        Whether to include bias terms in linear layers.
    normalization : str, default = "LayerNorm"
        Type of normalization: "LayerNorm" or "RMSNorm".
    activation : str, default = "gelu"
        Activation function name. Supports: gelu, geglu, qgelu, qgeglu,
        relu, reglu, srelu, sreglu, silu, swiglu, clamped_swiglu.
    activation_params : dict, optional
        Additional keyword arguments passed to the activation function.
    init_method : callable, optional
        Weight initialization for FC1.
    output_layer_init_method : callable, optional
        Weight initialization for FC2.
    params_dtype : torch.dtype, optional
        Data type for parameters.
    zero_centered_gamma : bool, default = False
        If True, gamma is initialized to zero and used as (1 + gamma).
    return_layernorm_output : bool, default = False
        If True, also return the normalization output.
    device : str or torch.device, default = "cuda"
        Device for parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        activation_params: Optional[Dict] = None,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        return_layernorm_output: bool = False,
        device: Union[torch.device, str] = "cuda",
        # Accepted for API compatibility with full-build LayerNormMLP but
        # ignored in lite mode (no TP/SP/FSDP/userbuffers support):
        return_bias: bool = False,
        sequence_parallel: bool = False,
        tp_group=None,
        tp_size: int = 1,
        set_parallel_mode: bool = False,
        fuse_wgrad_accumulation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.use_bias = bias
        self.normalization = normalization
        assert normalization in ("LayerNorm", "RMSNorm"), "Unsupported normalization type!"
        self.activation = activation
        self.activation_params = activation_params
        assert activation in _ACT_FUNC_MAP, f"Unsupported activation: {activation}"
        self.zero_centered_gamma = zero_centered_gamma
        self.return_layernorm_output = return_layernorm_output

        # No TP/SP in lite
        self.tp_size = 1
        self.sequence_parallel = False
        self.set_tensor_parallel_group(None)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        is_gated = activation in _GATED_ACTIVATIONS
        fc1_output_features = (2 * ffn_hidden_size) if is_gated else ffn_hidden_size

        # ---- Norm parameters ----
        layer_norm_weight = Parameter(
            torch.empty(hidden_size, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not zero_centered_gamma)),
        )
        if normalization != "RMSNorm":
            layer_norm_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias",
                layer_norm_bias,
                init_fn=init_method_constant(0.0),
            )
        else:
            self.layer_norm_bias = None

        # ---- FC1 parameters ----
        self.weight_names = ["fc1_weight", "fc2_weight"]
        self.bias_names = ["fc1_bias", "fc2_bias"]
        self.parameter_split_sizes = [fc1_output_features, hidden_size]

        fc1_weight = Parameter(
            torch.empty(fc1_output_features, hidden_size, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "fc1_weight", fc1_weight,
            init_fn=init_method,
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
        )
        if self.use_bias:
            self.register_parameter(
                "fc1_bias",
                Parameter(torch.empty(fc1_output_features, device=device, dtype=params_dtype)),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        # ---- FC2 parameters ----
        fc2_weight = Parameter(
            torch.empty(hidden_size, ffn_hidden_size, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "fc2_weight", fc2_weight,
            init_fn=output_layer_init_method,
            fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
        )
        if self.use_bias:
            self.register_parameter(
                "fc2_bias",
                Parameter(torch.empty(hidden_size, device=device, dtype=params_dtype)),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()
        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=(device == "meta"))

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        results = []
        for name in self.weight_names:
            w = getattr(self, name)
            if isinstance(w, QuantizedTensor) and self.fp8:
                results.append(w.get_quantized_tensor())
            else:
                results.append(w)
        return results

    def _get_weight_quantizers(self) -> List[Quantizer]:
        if not self.fp8 and not self.fp8_calibration:
            return [None, None]
        q1 = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        q1.internal = True
        q2 = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_WEIGHT]
        q2.internal = True
        return [q1, q2]

    def _get_quantizers(self):
        if not self.fp8:
            return (None,) * 8
        fc1_input_q = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        fc1_input_q.internal = True
        fc1_weight_q, fc2_weight_q = self._get_weight_quantizers()
        fc2_input_q = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_INPUT]
        fc2_input_q.internal = True
        # Backward quantizers
        fc2_grad_output_q = None
        fc1_grad_output_q = None
        fc1_grad_input_q = None
        if torch.is_grad_enabled():
            fc2_grad_output_q = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT2]
            fc2_grad_output_q.internal = True
            fc1_grad_output_q = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            fc1_grad_output_q.internal = True
            fc1_grad_input_q = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
        return (
            fc1_input_q, fc1_weight_q, fc2_input_q, fc2_weight_q,
            fc2_grad_output_q, fc1_grad_output_q, fc1_grad_input_q,
            None,  # fc1_grad_weight_q (not used in full build either)
        )

    def set_meta_tensor(self, fwd: bool, recipe) -> None:
        super().set_meta_tensor(fwd, recipe)
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        elif recipe.float8_block_scaling():
            self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)

    def _customize_quantizers_float8_current_scaling(self, fwd, recipe):
        if fwd:
            for idx in (tex.FP8FwdTensors.GEMM1_INPUT, tex.FP8FwdTensors.GEMM1_WEIGHT,
                        tex.FP8FwdTensors.GEMM2_INPUT, tex.FP8FwdTensors.GEMM2_WEIGHT):
                if idx in self.quantizers["scaling_fwd"]:
                    q = self.quantizers["scaling_fwd"][idx]
                    if hasattr(recipe, 'fp8_quant_fwd_inp'):
                        q.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                        q.amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon

    def _customize_quantizers_float8_blockwise_scaling(self, fwd, recipe):
        pass

    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        with self.prepare_forward(inp, num_gemms=2):
            (
                fc1_input_q, fc1_weight_q,
                fc2_input_q, fc2_weight_q,
                fc2_grad_output_q, fc1_grad_output_q,
                fc1_grad_input_q, fc1_grad_weight_q,
            ) = self._get_quantizers()

            out = _LayerNormMLPLite.apply(
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                self.fc1_bias if self.use_bias else None,
                self.fc2_weight,
                self.fc2_bias if self.use_bias else None,
                self.eps,
                self.fp8,
                fc1_input_q,
                fc1_weight_q,
                fc2_input_q,
                fc2_weight_q,
                fc2_grad_output_q,
                fc1_grad_output_q,
                fc1_grad_input_q,
                fc1_grad_weight_q,
                self.activation_dtype,
                self.return_layernorm_output,
                self.normalization,
                self.zero_centered_gamma,
                self.activation,
                self.activation_params,
                torch.is_grad_enabled(),
                self,
                is_first_microbatch,
            )

        return out
