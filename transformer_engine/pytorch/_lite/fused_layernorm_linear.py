# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Lite-native LayerNormLinear: fused normalization + linear projection."""

from typing import Callable, Optional, Tuple, Union, List

import torch

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


__all__ = ["LayerNormLinear"]


def _get_normalization_funcs(normalization: str):
    """Return (fwd_func, bwd_func) for the given normalization type."""
    if normalization == "RMSNorm":
        return tex.rmsnorm_fwd, tex.rmsnorm_bwd
    elif normalization == "LayerNorm":
        return tex.layernorm_fwd, tex.layernorm_bwd
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")


class _LayerNormLinearLite(torch.autograd.Function):
    """Autograd function for fused LayerNorm + Linear (lite backend)."""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Optional[torch.Tensor],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        fp8: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        normalization: str,
        zero_centered_gamma: bool,
        is_grad_enabled: bool,
        module: "LayerNormLinear",
        is_first_microbatch: Optional[bool],
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:

        # Reshape input
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        inp_shape = inp.shape
        inputmat = inp.reshape(-1, in_features)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        # Configure norm quantizer
        backward_needs_input = is_grad_enabled and weight.requires_grad
        if fp8 and input_quantizer is not None:
            input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)

        # Determine if we can use fused norm+quantize
        with_quantized_norm = (
            fp8
            and input_quantizer is not None
            and not return_layernorm_output
        )

        # Apply normalization
        norm_fwd, _ = _get_normalization_funcs(normalization)
        if normalization == "LayerNorm":
            ln_out, mu, rsigma = norm_fwd(
                inputmat, ln_weight, ln_bias, eps,
                None,  # ln_out (allocate internally)
                input_quantizer if with_quantized_norm else None,
                inputmat.dtype,
                0,  # sm_margin (unused in lite)
                zero_centered_gamma,
            )
        else:  # RMSNorm
            ln_out, mu, rsigma = norm_fwd(
                inputmat, ln_weight, eps,
                None,  # ln_out
                input_quantizer if with_quantized_norm else None,
                inputmat.dtype,
                0,  # sm_margin
                zero_centered_gamma,
            )

        # Save unquantized norm output if needed for return
        ln_out_return = ln_out if return_layernorm_output else None

        # Quantize norm output if not already done via fused kernel
        if fp8 and not with_quantized_norm and input_quantizer is not None:
            ln_out = input_quantizer(ln_out)

        # Prepare weight
        weightmat = weight
        if fp8 and weight_quantizer is not None:
            weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            update_workspace = is_first_microbatch is None or is_first_microbatch
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
            )
            weightmat.update_usage(rowwise_usage=True)
        else:
            weightmat = cast_if_needed(weightmat, activation_dtype)

        # Prepare bias
        gemm_bias = cast_if_needed(bias, activation_dtype) if bias is not None else bias

        # Forward GEMM: y = ln_out @ weight^T + bias
        bias_dtype = TE_DType[torch.bfloat16 if gemm_bias is None else gemm_bias.dtype]
        gemm_out, _, _, _ = tex.generic_gemm(
            weightmat,       # A
            True,            # transA (weight is [out, in], need transpose)
            ln_out,          # B
            False,           # transB
            None,            # D (allocate internally)
            None,            # quantizer
            TE_DType[activation_dtype] if activation_dtype in TE_DType else None,
            gemm_bias,       # bias
            bias_dtype,      # bias_type (actually bias dtype)
            False,           # gelu
            None,            # gelu_in
            False,           # grad
            torch.empty(0),  # workspace (unused in lite)
            0,               # workspace_size
            False,           # accumulate
            False,           # use_split_accumulator
        )

        out = gemm_out.view(-1, *inp_shape[1:-1], out_features)

        # Save tensors for backward
        if is_grad_enabled:
            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat, weightmat, weight, bias, ln_weight, ln_out, mu, rsigma,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.inp_shape = inp_shape
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.normalization = normalization
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.use_bias = bias is not None
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.input_quantizer = input_quantizer
            ctx.weight_quantizer = weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
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
            weightmat,
            weight,
            bias,
            ln_weight,
            ln_out,
            mu,
            rsigma,
        ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
        ctx.tensor_objects = None

        # Prepare grad_output
        grad_output = grad_output.reshape(-1, weight.shape[0])
        grad_output = cast_if_needed(grad_output, ctx.activation_dtype)

        # Quantize grad_output for FP8 backward
        if ctx.fp8 and ctx.grad_output_quantizer is not None:
            ctx.grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            grad_output = ctx.grad_output_quantizer(grad_output)

        # ---- DGRAD: d_ln_out = grad_output @ weight ----
        d_ln_out = None
        if ctx.requires_dgrad:
            bias_dtype = TE_DType[torch.bfloat16]
            d_ln_out, _, _, _ = tex.generic_gemm(
                weightmat,       # A (weight)
                False,           # transA=False → weight^T effect via NN layout
                grad_output,     # B
                False,           # transB
                None,            # D
                None,            # quantizer
                TE_DType[ctx.activation_dtype] if ctx.activation_dtype in TE_DType else None,
                None,            # bias
                bias_dtype,      # bias_type
                False,           # gelu
                None,            # gelu_in
                False,           # grad
                torch.empty(0),  # workspace
                0,               # workspace_size
                False,           # accumulate
                False,           # use_split_accumulator
            )

        # ---- WGRAD: dW = grad_output^T @ ln_out (NT layout) ----
        dweight = None
        dbias = None
        if ctx.requires_wgrad:
            bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]
            dweight, dbias_gemm, _, _ = tex.generic_gemm(
                ln_out,          # A (input for wgrad)
                False,           # transA (N)
                grad_output,     # B (grad output)
                True,            # transB (T) → NT layout
                None,            # D
                None,            # quantizer
                TE_DType[ctx.activation_dtype] if ctx.activation_dtype in TE_DType else None,
                bias if ctx.use_bias else None,  # bias (for grad computation)
                bias_dtype,      # bias_type
                False,           # gelu
                None,            # gelu_in
                True,            # grad (compute bias gradient)
                torch.empty(0),  # workspace
                0,               # workspace_size
                False,           # accumulate
                False,           # use_split_accumulator
            )
            if ctx.use_bias:
                dbias = dbias_gemm

        # ---- Norm backward ----
        dgrad = None
        dgamma = None
        dbeta = None
        if ctx.requires_dgrad:
            _, norm_bwd = _get_normalization_funcs(ctx.normalization)
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = norm_bwd(
                    d_ln_out, inputmat, mu, rsigma, ln_weight,
                    0,  # sm_margin
                    ctx.zero_centered_gamma,
                )
            else:  # RMSNorm
                dgrad, dgamma = norm_bwd(
                    d_ln_out, inputmat, rsigma, ln_weight,
                    0,  # sm_margin
                    ctx.zero_centered_gamma,
                )

            dgrad = dgrad.view(ctx.inp_shape)

        # Return gradients matching forward signature
        return (
            dgrad,           # inp
            dgamma,          # ln_weight
            dbeta,           # ln_bias
            dweight,         # weight
            dbias,           # bias
            None,            # eps
            None,            # fp8
            None,            # input_quantizer
            None,            # weight_quantizer
            None,            # grad_output_quantizer
            None,            # activation_dtype
            None,            # return_layernorm_output
            None,            # normalization
            None,            # zero_centered_gamma
            None,            # is_grad_enabled
            None,            # module
            None,            # is_first_microbatch
        )


class LayerNormLinear(TransformerEngineBaseModule):
    """Fused LayerNorm + Linear (lite-native, single-node).

    Applies normalization followed by a linear transformation:
        y = weight @ norm(x) + bias

    Parameters
    ----------
    in_features : int
        Input feature dimension (also the normalization dimension).
    out_features : int
        Output feature dimension.
    eps : float, default = 1e-5
        Epsilon for normalization stability.
    bias : bool, default = True
        Whether to include a bias term in the linear layer.
    normalization : str, default = "LayerNorm"
        Type of normalization: "LayerNorm" or "RMSNorm".
    init_method : callable, optional
        Weight initialization function.
    params_dtype : torch.dtype, optional
        Data type for parameters (default: current default dtype).
    zero_centered_gamma : bool, default = False
        If True, gamma is initialized to zero and used as (1 + gamma).
    return_layernorm_output : bool, default = False
        If True, also return the normalization output.
    device : str or torch.device, default = "cuda"
        Device for parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        bias: bool = True,
        normalization: str = "LayerNorm",
        init_method: Optional[Callable] = None,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        return_layernorm_output: bool = False,
        device: Union[torch.device, str] = "cuda",
        # Accepted for API compatibility with full-build LayerNormLinear but
        # ignored in lite mode (no TP/SP/FSDP/userbuffers support):
        return_bias: bool = False,
        parallel_mode: Optional[str] = None,
        sequence_parallel: bool = False,
        tp_group=None,
        tp_size: int = 1,
        parameters_split: Optional[Union[tuple, dict]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.use_bias = bias
        self.normalization = normalization
        assert normalization in ("LayerNorm", "RMSNorm"), "Unsupported normalization type!"
        self.zero_centered_gamma = zero_centered_gamma
        self.return_layernorm_output = return_layernorm_output

        # No TP/SP in lite
        self.tp_size = 1
        self.sequence_parallel = False
        self.set_tensor_parallel_group(None)

        if init_method is None:
            init_method = get_default_init_method()

        # Norm parameters
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not zero_centered_gamma)),
        )
        if normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias",
                layer_norm_bias,
                init_fn=init_method_constant(0.0),
            )
        else:
            self.layer_norm_bias = None

        # Linear parameters
        weight_tensor = torch.empty(
            out_features, in_features, device=device, dtype=params_dtype,
        )
        self.weight_names = ["weight"]
        self.bias_names = ["bias"]
        self.parameter_split_sizes = [out_features]

        self.register_parameter(
            "weight",
            torch.nn.Parameter(weight_tensor),
            init_fn=init_method,
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
        )

        if self.use_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, device=device, dtype=params_dtype)
                ),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.bias = torch.Tensor().to(dtype=params_dtype, device=device)

        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()
        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=(device == "meta"))

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        w = getattr(self, "weight")
        if isinstance(w, QuantizedTensor) and self.fp8:
            return [w.get_quantized_tensor()]
        return [w]

    def _get_weight_quantizers(self) -> List[Quantizer]:
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

    def _get_quantizers(self, fp8_output: bool = False):
        if not self.fp8:
            return (None, None, None)
        input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        input_quantizer.internal = True
        (weight_quantizer,) = self._get_weight_quantizers()
        grad_output_quantizer = None
        if torch.is_grad_enabled():
            grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
        return (input_quantizer, weight_quantizer, grad_output_quantizer)

    def set_meta_tensor(self, fwd: bool, recipe) -> None:
        super().set_meta_tensor(fwd, recipe)
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        elif recipe.float8_block_scaling():
            self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)

    def _customize_quantizers_float8_current_scaling(self, fwd, recipe):
        if fwd:
            for idx in (tex.FP8FwdTensors.GEMM1_INPUT, tex.FP8FwdTensors.GEMM1_WEIGHT):
                if idx in self.quantizers["scaling_fwd"]:
                    q = self.quantizers["scaling_fwd"][idx]
                    if hasattr(recipe, 'fp8_quant_fwd_inp'):
                        q.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                        q.amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon

    def _customize_quantizers_float8_blockwise_scaling(self, fwd, recipe):
        pass  # Block scaling quantizers work with defaults

    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        with self.prepare_forward(inp, num_gemms=1):
            (
                input_quantizer,
                weight_quantizer,
                grad_output_quantizer,
            ) = self._get_quantizers()

            out = _LayerNormLinearLite.apply(
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.weight,
                self.bias if self.use_bias else None,
                self.eps,
                self.fp8,
                input_quantizer,
                weight_quantizer,
                grad_output_quantizer,
                self.activation_dtype,
                self.return_layernorm_output,
                self.normalization,
                self.zero_centered_gamma,
                torch.is_grad_enabled(),
                self,
                is_first_microbatch,
            )

        return out
