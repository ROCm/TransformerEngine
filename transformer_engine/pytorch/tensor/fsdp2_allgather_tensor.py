#!/usr/bin/python3
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
from __future__ import annotations
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import torch.utils._pytree as pytree

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


# A wrapper subclass for stateful FSDP transport
class FSDPAGTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem: torch.Tensor, **kwargs):
        # Build an "empty" wrapper with the same meta as elem
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=elem.device,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        fp8_meta_index: str,
    ):
        #The underlying tensor
        self._data = tensor
        # Which quantizer to use within module.quantizers["scaling_fwd"][idx]
        self._fp8_meta_index = fp8_meta_index

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()
    
    def __repr__(self):
            return (
                f"FSDPAGTensor("
                f"elem={self._data}, "
                f"fp8_meta_index={self._fp8_meta_index})"
            )
    
    def __tensor_flatten__(self):
            """
            Makes some ops (view/as_strided, etc.) and serialization friendlier for wrapper subclasses.
            Return (names_of_inner_tensors, flatten_spec_metadata).
            """
            # We only carry the one inner tensor.
            # We store fp8_meta_index as metadata to reconstruct.
            return ["_data"], (self._fp8_meta_index)

    
    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        fp8_meta_index = flatten_spec
        return FSDPAGTensor(
            inner_tensors["_data"],
            fp8_meta_index=fp8_meta_index,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        # detach
        if func is torch.ops.aten.detach.default:
            t = args[0]
            assert isinstance(t, cls), f"Unexpected detach input type: {type(t)}"
            detached = t._data.detach()
            return cls(detached, fp8_meta_index=t._fp8_meta_index)

        # Unwrap only our subclass; capture shared metadata for rewrapping
        meta: Optional[str] = None

        def unwrap(x):
            nonlocal meta
            if isinstance(x, cls):
                if meta is None:
                    meta = x._fp8_meta_index
                return x._data
            return x

        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(cls, unwrap, (args, kwargs))

        # Run the actual op on internal tensors
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # Rewrap outputs only for ops that need to preserve subclass identity
        if func not in _ops_to_preserve_subclass or meta is None:
            return out

        def rewrap(x):
            if isinstance(x, torch.Tensor):
                return cls(x, fp8_meta_index=meta)
            return x

        out = pytree.tree_map_only(torch.Tensor, rewrap, out)
        return out

    # Must return (list_of_tensors_to_all_gather, user_metadata)
    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Functions FSDP2 calls before all-gather of the
        weights for both forward and backward passes.
        Args:
            mesh (torch.distributed.DeviceMesh): DeviceMesh used by FSDP2
            to shard the weights.
            orig_size (torch.Size): Original size of the weight tensor.
            contiguous_orig_stride (Tuple[int]): Original stride of the weight tensor.
            module (FSDPModule): FSDP module. FSDP wrapped module wrapped using fully_shard
            that contains this tensor.
            mp_policy (MixedPrecisionPolicy): Mixed precision policy used by FSDP2.

        Returns:
            sharded_tensors: Tuple[torch.Tensor, ...]: Tuple of tensors
            that need to be all-gathered.
            metadata: Tuple[Any]: Metadata needed for reconstructing the
            tensor after all-gather.
        """
        # pylint: disable=unused-argument
        # If metadata isn't initialized yet, we can't access the quantizers
        if not module.fp8:
            module_class_name = module.__class__.__name__  
            if "LayerNormMLP" in module_class_name:  
                num_gemms = 2  
            else:  # Linear, LayerNormLinear, etc.  
                num_gemms = 1  

            module.init_fp8_metadata(num_gemms=num_gemms)
        if not module.fp8:
            return (self._data,), (self._data.requires_grad, module)
        # Use the actual data
        base = self._data
        # Access the quantizer using fp8_meta_index
        quantizer = module.quantizers["scaling_fwd"][self._fp8_meta_index]
        if not isinstance(quantizer, MXFP8Quantizer):
            quantizer.set_usage(columnwise=False)
        if isinstance(quantizer, Float8CurrentScalingQuantizer):
            quantizer.with_amax_reduction = True
        sharded_fp8_tensor = quantizer(base)
        if isinstance(quantizer, MXFP8Quantizer):
            rowwise_data = sharded_fp8_tensor._rowwise_data if quantizer.rowwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            rowwise_scale_inv = sharded_fp8_tensor._rowwise_scale_inv if quantizer.rowwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            columnwise_data = sharded_fp8_tensor._columnwise_data if quantizer.columnwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            columnwise_scale_inv = sharded_fp8_tensor._columnwise_scale_inv if quantizer.columnwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            return (rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, ), (base.requires_grad, module)
        return (sharded_fp8_tensor._data,), (base.requires_grad, module)
        
    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        """Functions FSDP2 calls after all-gather of the
        weights for both forward and backward passes.
        Args:
            all_gather_outputs (Tuple[torch.Tensor, ...]): sharded_tensors sent out in fsdp_pre_all_gather from each rank
            are all-gathered and received here as a tuple.
            metadata (Any): metadata sent out in fsdp_pre_all_gather used for reconstructing the tensor.
            param_dtype (torch.dtype): high precision dtype of the tensor.
            out (Optional[torch.Tensor], optional): Preallocated output tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tuple[torch.Tensor, ...]]: Allgathered tensor and tuple of internal tensors.
        """
        (requires_grad, module) = metadata
        if not module.fp8:
            (data,) = all_gather_outputs
            return data, all_gather_outputs
        # Retrieve the same quantizer you used in pre_all_gather
        quantizer = module.quantizers["scaling_fwd"][self._fp8_meta_index]
        shape = None
        if  not isinstance(quantizer, MXFP8Quantizer):
            quantizer.set_usage(columnwise=False)
        if isinstance(quantizer, MXFP8Quantizer):
            (rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv,) = all_gather_outputs
            shape = rowwise_data.shape
        else:
            (data,) = all_gather_outputs
            shape = data.shape

        # Construct a new low precision tensor subclass that will wrap the gathered data
        if out is None:
            out = quantizer.make_empty(shape = shape, dtype=param_dtype, requires_grad=requires_grad)

        if isinstance(quantizer, MXFP8Quantizer):
            out._rowwise_data = rowwise_data
            out._rowwise_scale_inv = rowwise_scale_inv 
            out._columnwise_data = None if columnwise_data.numel() == 0 else columnwise_data
            out._columnwise_scale_inv =  None if columnwise_scale_inv.numel() == 0 else columnwise_scale_inv
        else:
            out._scale_inv = 1 / quantizer.scale
            out._data = data
        return out, all_gather_outputs

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling - unwrap to inner tensor
        
        During checkpointing, save just the underlying high-precision tensor.
        FSDPAGTensor is a transient wrapper for FSDP2 communication - when the
        model is loaded and FSDP2 is re-initialized, parameters get wrapped again.
        """
        # Delegate to the inner tensor's serialization
        return self._data.__reduce_ex__(protocol)