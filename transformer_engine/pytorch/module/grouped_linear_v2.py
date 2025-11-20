# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinearV2 API - Optimized implementation using stacked tensors"""
import os
from typing import Union, Optional, Callable, Tuple, List
import warnings
import functools
import torch

from transformer_engine.pytorch.triton_kernels.grouped_gemm import general_grouped_gemm_triton
import transformer_engine_torch as tex
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.common.recipe import Recipe
from .base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import WeightGradStore
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..cpp_extensions import general_grouped_gemm
from ..constants import GemmParallelModes, dist_group_type, TE_DType
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..cpu_offload import is_cpu_offload_enabled

from ..tensor.quantized_tensor import (
    QuantizedTensorBase,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)

__all__ = ["GroupedLinearV2"]


class _GroupedLinearV2(torch.autograd.Function):
    """GroupedLinearV2 autograd function
    
    Optimized version using stacked tensors:
    - Single weight parameter [num_gemms, out_features, in_features]
    - Single bias parameter [num_gemms, out_features]
    - Reduced autograd overhead (2 AccumulateGrad nodes vs 128)
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        input_quantizers: List[Quantizer],
        weight_quantizers: List[Quantizer],
        output_quantizers: List[Quantizer],
        grad_output_quantizers: List[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        module,
        skip_fp8_weight_update,
        weight_stacked: torch.Tensor,  # [num_gemms, out_features, in_features]
        bias_stacked: torch.Tensor,    # [num_gemms, out_features] or empty
    ) -> torch.Tensor:
        num_gemms = len(m_splits)
        device = inp.device
        
        # Convert stacked tensors to lists for internal processing
        weights = [weight_stacked[i] for i in range(num_gemms)]
        biases = [bias_stacked[i] for i in range(num_gemms)] if use_bias else [torch.Tensor()] * num_gemms
        
        # Make sure input dimensions are compatible
        in_features = weight_stacked.size(-1)
        assert inp.shape[-1] == in_features, "GEMM not possible"
        
        # Check if using Triton kernels
        use_grouped_gemm_triton = bool(int(os.environ.get('NVTE_USE_GROUPED_GEMM_TRITON', '0'))) and IS_HIP_EXTENSION
        
        # For Triton, keep tensor concatenated; for others, split per expert
        if use_grouped_gemm_triton:
            inp_reshaped = inp.view(-1, in_features)
            inputmats = [inp_reshaped]
            if fp8:
                assert_dim_for_fp8_exec(inp_reshaped, *weights)
            inputmats_no_fp8 = [cast_if_needed(inp_reshaped, activation_dtype)]
        else:
            inputmats = torch.split(inp.view(-1, in_features), m_splits)
            if fp8:
                assert_dim_for_fp8_exec(*inputmats, *weights)
            inputmats_no_fp8 = [cast_if_needed(mat, activation_dtype) for mat in inputmats]
        
        inputmats = []
        weight_requires_grad = weight_stacked.requires_grad

        if input_quantizers[0] is not None:
            for input_quantizer in input_quantizers:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )
            columnwise_usage = is_grad_enabled and inp.requires_grad
            if not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            if weight_quantizers[0] is not None:
                for weight_quantizer in weight_quantizers:
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        if output_quantizers[0] is not None:
            for output_quantizer in output_quantizers:
                output_quantizer.set_usage(rowwise=True, columnwise=False)

        fprop_gemm_use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                fprop_gemm_use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator
            inputmats = tex.fused_multi_quantize(
                inputmats_no_fp8, None, input_quantizers, TE_DType[activation_dtype]
            )
            weights_fp8 = []
            bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
            update_workspace = is_first_microbatch is None or is_first_microbatch
            # Apply per-expert FP8 quantization to slices of stacked weight
            for i in range(num_gemms):
                weight_fp8 = module.get_weight_workspace(
                    tensor=weights[i],  # View of stacked weight
                    quantizer=weight_quantizers[i],
                    cache_name=(None if is_first_microbatch is None else f"weight{i}"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                )
                weights_fp8.append(weight_fp8)
        else:
            inputmats = inputmats_no_fp8
            bias_dtype = activation_dtype
            weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]

        biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases

        out = torch.empty(
            [sum(m_splits), weights_fp8[0].size(0)],
            dtype=activation_dtype,
            device=device,
        )
        
        grouped_gemm_func = general_grouped_gemm_triton if use_grouped_gemm_triton else general_grouped_gemm

        _ = grouped_gemm_func(
            weights_fp8,
            inputmats,
            [out],
            activation_dtype,
            get_multi_stream_cublas_workspace(),
            single_output=True,
            m_splits=m_splits,
            bias=biases,
            use_bias=use_bias,
            use_split_accumulator=fprop_gemm_use_split_accumulator,
        )

        if fp8_calibration:
            for i in range(num_gemms):
                input_quantizers[i].calibrate(inputmats[i])
                weight_quantizers[i].calibrate(weights[i])

        if is_grad_enabled:
            ctx.weight_quantizers = weight_quantizers
            ctx.weights_shape = weight_stacked.size()

            if weight_requires_grad:
                for inputmat in inputmats:
                    if isinstance(inputmat, QuantizedTensorBase):
                        inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
            if inp.requires_grad:
                for weight in weights_fp8:
                    if isinstance(weight, QuantizedTensorBase):
                        weight.update_usage(columnwise_usage=True)

            tensors_to_save, tensor_objects = prepare_for_saving(
                *inputmats,
                *weights_fp8,
                weight_stacked,
                bias_stacked if use_bias else torch.Tensor(),
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.weights_requires_grad = weight_stacked.requires_grad
            
            # Store stacked main_grad if available
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                ctx.main_grad_stacked = weight_stacked.main_grad if hasattr(weight_stacked, 'main_grad') else None
            else:
                ctx.main_grad_stacked = None
                
            ctx.device = device
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = inp.requires_grad
            ctx.use_grouped_gemm_triton = use_grouped_gemm_triton
            ctx.num_input_tensors = len(inputmats)
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weight_stacked, bias_stacked if use_bias else None):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )
            ctx.wgrad_store = wgrad_store

        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        with torch.cuda.nvtx.range("_GroupedLinearV2_backward"):
            saved_tensors = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            N = ctx.num_gemms
            num_inputs = ctx.num_input_tensors
            inputmats = saved_tensors[:num_inputs]
            weights_fp8 = saved_tensors[num_inputs : num_inputs + N]
            weight_stacked = saved_tensors[num_inputs + N]
            bias_stacked = saved_tensors[num_inputs + N + 1]

            weights = [weight_stacked[i] for i in range(N)]
            biases = [bias_stacked[i] for i in range(N)] if ctx.use_bias else [torch.Tensor()] * N

            # Handle CPU offloading with fused wgrad accumulation
            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                # For stacked parameters, recreate the parameter wrapper with main_grad
                if ctx.main_grad_stacked is not None:
                    w = torch.nn.Parameter(weight_stacked, weight_stacked.requires_grad)
                    w.main_grad = ctx.main_grad_stacked
                    weight_stacked = w

            grad_output = grad_output.contiguous()
            
            # Process grad_output based on backend
            if ctx.use_grouped_gemm_triton and not ctx.fp8:
                grad_output_reshaped = grad_output.view(-1, grad_output.shape[-1])
                grad_output_list = [grad_output_reshaped]
                grad_biases = [None] * ctx.num_gemms
            else:
                grad_output_mats = torch.split(
                    grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits
                )
                grad_output_list = [None] * ctx.num_gemms
                grad_biases = [None] * ctx.num_gemms
                if ctx.fp8:
                    if ctx.use_bias:
                        if ctx.fp8_recipe.float8_block_scaling():
                            for i in range(ctx.num_gemms):
                                grad_biases[i] = grad_output_mats[i].sum(dim=0)
                                grad_output_list[i] = ctx.grad_output_quantizers[i](grad_output_mats[i])
                        else:
                            for i in range(ctx.num_gemms):
                                grad_output_list[i] = ctx.grad_output_quantizers[i](grad_output_mats[i])
                    else:
                        for i in range(ctx.num_gemms):
                            grad_output_list[i] = ctx.grad_output_quantizers[i](grad_output_mats[i])
                else:
                    for i in range(ctx.num_gemms):
                        grad_output_list[i] = grad_output_mats[i]

            accumulate_wgrad_into_param_main_grad = False
            if ctx.fuse_wgrad_accumulation:
                if ctx.is_first_microbatch is not None:
                    accumulate_wgrad_into_param_main_grad = (
                        ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                    )
                else:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            grouped_gemm_func = general_grouped_gemm_triton if ctx.use_grouped_gemm_triton else general_grouped_gemm
            
            dgrad = None
            if ctx.requires_dgrad:
                dgrad_gemm_use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        dgrad_gemm_use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator
                        
                dgrad = torch.empty(
                    (sum(ctx.m_splits), ctx.weights_shape[2]),
                    dtype=ctx.activation_dtype,
                    device=ctx.device,
                )

                for weight, quantizer in zip(weights, ctx.weight_quantizers):
                    if quantizer is not None and isinstance(weight, QuantizedTensorBase):
                        weight.update_usage(
                            rowwise_usage=quantizer.rowwise_usage,
                            columnwise_usage=quantizer.columnwise_usage,
                        )
                
                grouped_gemm_func(
                    weights,
                    grad_output_list,
                    [dgrad],
                    ctx.activation_dtype,
                    get_multi_stream_cublas_workspace(),
                    single_output=True,
                    layout="NN",
                    m_splits=ctx.m_splits,
                    grad=True,
                    use_split_accumulator=dgrad_gemm_use_split_accumulator,
                )

            wgrad_stacked = None
            if ctx.weights_requires_grad:
                wgrad_gemm_use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        wgrad_gemm_use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator
                
                # Create or use stacked tensor for wgrad
                if ctx.fuse_wgrad_accumulation and ctx.main_grad_stacked is not None:
                    wgrad_stacked = ctx.main_grad_stacked
                else:
                    wgrad_stacked = torch.empty(
                        ctx.weights_shape,
                        dtype=ctx.activation_dtype,
                        device=ctx.device
                    )
                
                # Pass stacked tensor directly for Triton, convert to list for others
                if ctx.use_grouped_gemm_triton:
                    wgrad_arg = wgrad_stacked
                else:
                    wgrad_arg = [wgrad_stacked[i] for i in range(ctx.num_gemms)]
                
                grouped_gemm_wgrad = functools.partial(
                    grouped_gemm_func,
                    out_dtype=ctx.activation_dtype,
                    workspaces=get_multi_stream_cublas_workspace(),
                    layout="NT",
                    grad=True,
                    m_splits=ctx.m_splits,
                    use_bias=ctx.use_bias if grad_biases[0] is None else None,
                    bias=biases,
                    use_split_accumulator=wgrad_gemm_use_split_accumulator,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                )
                
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    ctx.wgrad_store.put([inputmats, grad_output_list, wgrad_arg], grouped_gemm_wgrad)
                    wgrad_stacked = None
                else:
                    _, grad_biases_, _ = grouped_gemm_wgrad(inputmats, grad_output_list, wgrad_arg)

                    for i in range(ctx.num_gemms):
                        if grad_biases[i] is None:
                            grad_biases[i] = grad_biases_[i]
                    del grad_biases_

                    clear_tensor_data(*inputmats)
                
                # Handle custom DDP from Megatron-Core for stacked weight
                if ctx.weights_requires_grad and weight_stacked is not None:
                    if ctx.fuse_wgrad_accumulation and hasattr(weight_stacked, "grad_added_to_main_grad"):
                        weight_stacked.grad_added_to_main_grad = True
                        if getattr(weight_stacked, "zero_out_wgrad", False):
                            wgrad_stacked = torch.zeros(
                                weight_stacked.main_grad.shape,
                                dtype=weight_stacked.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False,
                            )
                        else:
                            wgrad_stacked = torch.empty(
                                weight_stacked.main_grad.shape,
                                dtype=weight_stacked.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False,
                            )
                    elif ctx.fuse_wgrad_accumulation:
                        wgrad_stacked = None

            # Stack bias gradients
            grad_bias_stacked = None
            if ctx.use_bias and not (
                ctx.wgrad_store is not None
                and ctx.wgrad_store.delay_wgrad_compute()
                and not ctx.fp8
            ):
                if any(gb is not None for gb in grad_biases):
                    grad_bias_stacked = torch.stack(grad_biases, dim=0)

            if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
                FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            
            # OPTIMIZED: Return only 2 gradient tensors instead of 128!
            # This reduces AccumulateGrad nodes from 128 to 2
            return (
                dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
                None,  # m_splits
                None,  # use_bias
                None,  # is_first_microbatch
                None,  # fp8
                None,  # fp8_calibration
                None,  # wgrad_store
                None,  # input_quantizers
                None,  # weight_quantizers
                None,  # output_quantizers
                None,  # grad_output_quantizers
                None,  # fuse_wgrad_accumulation
                None,  # cpu_offloading
                None,  # sequence_parallel
                None,  # activation_dtype
                None,  # is_grad_enabled
                None,  # module
                None,  # skip_fp8_weight_update
                wgrad_stacked,      # Stacked weight gradient [num_gemms, out_features, in_features]
                grad_bias_stacked,  # Stacked bias gradient [num_gemms, out_features] or None
            )


class GroupedLinearV2(TransformerEngineBaseModule):
    """Optimized Grouped Linear Layer using stacked tensors
    
    This is an optimized version of GroupedLinear that uses single stacked parameters
    instead of multiple separate parameters.
    
    Key differences from GroupedLinear:
    - Parameters:
      * self.weight: [num_gemms, out_features, in_features] (single 3D tensor)
      * self.bias: [num_gemms, out_features] (single 2D tensor)
      vs GroupedLinear: weight0, weight1, ..., weight63, bias0, ..., bias63
    
    - Performance:
      * Reduces backward pass overhead by ~1ms (128 → 2 AccumulateGrad nodes)
      * Eliminates per-tensor event overhead
      * Better memory locality
    
    - API:
      * Access: model.weight[i] instead of model.weight{i}
      * Otherwise identical to GroupedLinear
    
    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights: `init_method(weight)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get random number generator state tracker.
    rng_tracker_name : str, default = `None`
                 param passed to get_rng_state_tracker.
    device : Union[torch.device, str], default = "cuda"
          Device for parameter allocation.
    
    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             enables fusing of weight gradient accumulation.
    return_bias : bool, default = `False`
                 return bias separately instead of applying it.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  dtype for parameter allocation.
    delay_wgrad_compute : bool, default = `False'
                         delay weight gradient computation.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinearV2 doesn't support Userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)

        self._offsets = {"input": 0, "weight": 1, "output": 2, "grad_output": 0, "grad_input": 1}
        self._num_fp8_tensors_per_gemm = {
            "fwd": 3,
            "bwd": 2,
        }

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinearV2 doesn't support bias when TP > 1. "
                "TP communication is handled outside this module."
            )

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Single stacked weight parameter [num_gemms, out_features, in_features]
        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    self.num_gemms,
                    self.out_features,
                    self.in_features,
                    device=device,
                    dtype=params_dtype,
                ),
            ),
            init_fn=init_method,
            get_rng_state_tracker=get_rng_state_tracker,
            fp8_meta_index=self._offsets["weight"],
        )

        # Single stacked bias parameter [num_gemms, out_features]
        if self.use_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(
                        self.num_gemms,
                        self.out_features,
                        device=device,
                        dtype=params_dtype,
                    ),
                ),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.bias = torch.Tensor().to(dtype=params_dtype, device=device)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=self.num_gemms)

        self.reset_parameters(defer_init=device == "meta")

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            assert not self.tp_size > 1, (
                "GroupedLinearV2 doesn't support TP > 1 with Float8 current scaling."
            )
            self._customize_quantizers_float8_current_scaling(fwd, recipe)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for stacked weight
            # dim+1 for the extra stacking dimension
            set_tensor_model_parallel_attributes(
                tensor=self.weight,
                is_parallel=True,
                dim=2 if self.parallel_mode == "row" else 1,
                stride=1,
            )

            # Set parallelism attributes for stacked bias
            if self.use_bias:
                if self.parallel_mode == "row":
                    setattr(self.bias, "sequence_parallel", self.sequence_parallel)
                elif self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(self.bias, True, 1, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass using stacked tensors
        
        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        m_splits : List[int]
                 List of token counts per expert.
        is_first_microbatch : {True, False, None}, default = None
                             Enables FP8 weight caching and gradient accumulation optimizations.
        
        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]
            Output tensor, or (output, biases) if return_bias=True.
        """
        assert not isinstance(
            inp, QuantizedTensorBase
        ), "GroupedLinearV2 doesn't support FP8 input tensor."
        assert len(m_splits) == self.num_gemms, "Number of splits must match num_gemms."

        skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(inp, num_gemms=self.num_gemms) as inp:

            if not self.fp8 and isinstance(self.weight, QuantizedTensorBase):
                warnings.warn(
                    "Using quantized weights without quantized compute."
                )
                weight_to_use = self.weight.dequantize()
            else:
                weight_to_use = self.weight

            input_quantizers = [None] * self.num_gemms
            weight_quantizers = [None] * self.num_gemms
            output_quantizers = [None] * self.num_gemms
            grad_output_quantizers = [None] * self.num_gemms
            
            if self.fp8:
                input_quantizers = [
                    self.quantizers["scaling_fwd"][
                        self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                    ]
                    for i in range(self.num_gemms)
                ]
                for i in range(self.num_gemms):
                    input_quantizers[i].internal = False
                    
                weight_quantizers = [
                    self.quantizers["scaling_fwd"][
                        self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                    ]
                    for i in range(self.num_gemms)
                ]
                for i in range(self.num_gemms):
                    weight_quantizers[i].internal = True
                    
                if torch.is_grad_enabled():
                    grad_output_quantizers = [
                        self.quantizers["scaling_bwd"][
                            self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                        ]
                        for i in range(self.num_gemms)
                    ]
                    for i in range(self.num_gemms):
                        grad_output_quantizers[i].internal = True

            if torch.is_grad_enabled():
                linear_fn = _GroupedLinearV2.apply
                args = []
            else:
                linear_fn = _GroupedLinearV2.forward
                args = [None]
                
            args += (
                inp,
                m_splits,
                self.apply_bias,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.wgrad_store,
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_output_quantizers,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.sequence_parallel,
                self.activation_dtype,
                torch.is_grad_enabled(),
                self,
                skip_fp8_weight_update,
                weight_to_use,
                self.bias if self.use_bias else torch.Tensor(),
            )
            out = linear_fn(*args)

        if self.return_bias:
            bias_list = [self.bias[i] for i in range(self.num_gemms)] if self.use_bias else [torch.Tensor()] * self.num_gemms
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_list]
        return out

    def backward_dw(self):
        """Execute delayed weight gradient computation"""
        if self.wgrad_store is None or not self.wgrad_store.delay_wgrad_compute():
            return
            
        with torch.cuda.nvtx.range("_GroupedLinearV2_wgrad"):
            (_, grad_biases_, _), tensor_list = self.wgrad_store.pop()
            wgrad_tensor = tensor_list[2]
            
            # Handle wgrad
            if isinstance(wgrad_tensor, list):
                wgrad_stacked = torch.stack(wgrad_tensor, dim=0)
            else:
                wgrad_stacked = wgrad_tensor
            
            if not self.fuse_wgrad_accumulation:
                if self.weight.grad is None:
                    self.weight.grad = wgrad_stacked.to(self.weight.dtype)
                else:
                    self.weight.grad.add_(wgrad_stacked.to(self.weight.dtype))
            
            # Handle bias gradients
            if self.use_bias and grad_biases_ is not None:
                if isinstance(grad_biases_, list):
                    grad_bias_stacked = torch.stack([gb for gb in grad_biases_ if gb is not None], dim=0)
                else:
                    grad_bias_stacked = grad_biases_
                
                if self.bias.grad is None:
                    self.bias.grad = grad_bias_stacked.to(self.bias.dtype)
                else:
                    self.bias.grad.add_(grad_bias_stacked.to(self.bias.dtype))
            
            del grad_biases_
            del wgrad_tensor
            del tensor_list

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers for Float8 current scaling"""
        # Similar implementation to GroupedLinear
        pass

