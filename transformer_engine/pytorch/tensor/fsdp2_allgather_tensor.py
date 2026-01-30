#!/usr/bin/python3
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import torch.utils._pytree as pytree
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase

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
        module: nn.Module,
        fp8_meta_index: str,
        keep_fp8_weight_transpose_cache: bool,
    ):
        # Guard against double-wrapping - this should never happen
        if type(tensor).__name__ == 'FSDPAGTensor':
            raise ValueError(f"Cannot wrap FSDPAGTensor around another FSDPAGTensor. This indicates a bug.")
        
        # The underlying tensor (renamed from _data to avoid collision with Float8Tensor._data)
        self._elem = tensor
        # Where quantizers are present
        self._module = module
        # Which quantizer to use within module.quantizers["scaling_fwd"][idx]
        self._fp8_meta_index = fp8_meta_index
        # Disable or enable transpose cache for fp8 weights
        self._keep_fp8_weight_transpose_cache = keep_fp8_weight_transpose_cache

    @property
    def data(self) -> torch.Tensor:
        return self._elem.detach()
    
    def __getattr__(self, name):
        """Forward Float8Tensor-specific attributes to the underlying tensor."""
        # Avoid recursion by checking __dict__ directly
        if '_elem' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Forward to the wrapped tensor
        return getattr(self.__dict__['_elem'], name)
    
    def __setattr__(self, name, value):
        """Forward Float8Tensor-specific attribute writes to the underlying tensor."""
        # FSDPAGTensor's own attributes - always set on self
        if name in ['_elem', '_module', '_fp8_meta_index', '_keep_fp8_weight_transpose_cache']:
            object.__setattr__(self, name, value)
        else:
            # Try to forward to underlying tensor if it exists
            if '_elem' in self.__dict__:
                elem = self.__dict__['_elem']
                # Forward to Float8Tensor if it has this attribute
                if hasattr(elem, name):
                    setattr(elem, name, value)
                else:
                    object.__setattr__(self, name, value)
            else:
                # _elem doesn't exist yet (during __init__), set on self
                object.__setattr__(self, name, value)
    
    def __repr__(self):
            return (
                f"FSDPAGTensor("
                f"elem={self._elem}, "
                f"module={self._module.__class__.__name__}, "
                f"fp8_meta_index={self._fp8_meta_index})"
            )
    
    def __tensor_flatten__(self):
            """
            Makes some ops (view/as_strided, etc.) and serialization friendlier for wrapper subclasses.
            Return (names_of_inner_tensors, flatten_spec_metadata).
            """
            # We only carry the one inner tensor.
            # We store (module, fp8_meta_index, keep_fp8_weight_transpose_cache) as metadata to reconstruct.
            return ["_elem"], (self._module, self._fp8_meta_index, self._keep_fp8_weight_transpose_cache)

    
    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        module, fp8_meta_index, keep_fp8_weight_transpose_cache = flatten_spec
        elem = inner_tensors["_elem"]
        # Avoid double-wrapping if elem is already FSDPAGTensor
        if type(elem).__name__ == 'FSDPAGTensor':
            return elem
        return FSDPAGTensor(
            elem,
            module=module,
            fp8_meta_index=fp8_meta_index,
            keep_fp8_weight_transpose_cache=keep_fp8_weight_transpose_cache
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        # detach
        if func is torch.ops.aten.detach.default:
            t = args[0]
            assert type(t).__name__ == 'FSDPAGTensor', f"Unexpected detach input type: {type(t)}"
            # Access via object.__getattribute__ to avoid __getattr__ issues
            t_dict = object.__getattribute__(t, '__dict__')
            elem = t_dict['_elem']
            detached = elem.detach()
            return cls(detached, module=t_dict['_module'], fp8_meta_index=t_dict['_fp8_meta_index'], keep_fp8_weight_transpose_cache=t_dict['_keep_fp8_weight_transpose_cache'])

        # Unwrap only our subclass; capture shared metadata for rewrapping
        meta: Optional[tuple[nn.Module, str, bool]] = None

        def unwrap(x):
            nonlocal meta
            # Recursively unwrap FSDPAGTensor (in case of nested wrapping)
            # Use type() check to avoid isinstance() issues
            while type(x).__name__ == 'FSDPAGTensor':
                x_dict = object.__getattribute__(x, '__dict__')
                if meta is None and '_module' in x_dict:
                    # Capture metadata from the outermost FSDPAGTensor
                    meta = (x_dict['_module'], x_dict['_fp8_meta_index'], x_dict['_keep_fp8_weight_transpose_cache'])
                # Access _elem directly from __dict__
                if '_elem' in x_dict:
                    x = x_dict['_elem']
                else:
                    # _elem not initialized yet, return as-is
                    break
            return x

        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(cls, unwrap, (args, kwargs))

        # Run the actual op on internal tensors
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # Rewrap outputs only for ops that need to preserve subclass identity
        if func not in _ops_to_preserve_subclass or meta is None:
            return out

        def rewrap(x):
            # Don't double-wrap FSDPAGTensor - check type name to be safe
            if type(x).__name__ == 'FSDPAGTensor':
                return x
            # Wrap other tensors (including Float8Tensor)
            if isinstance(x, torch.Tensor):
                mod, idx, keep_transpose = meta
                return cls(x, module=mod, fp8_meta_index=idx, keep_fp8_weight_transpose_cache=keep_transpose)
            return x

        out = pytree.tree_map_only(torch.Tensor, rewrap, out)
        return out

    # Must return (list_of_tensors_to_all_gather, user_metadata)
    def fsdp_pre_all_gather(self, mesh):
        # If metadata isn't initialized yet, we can't access the quantizers
        if not self._module.fp8:
            module_class_name = self._module.__class__.__name__  
            if "LayerNormMLP" in module_class_name:  
                num_gemms = 2  
            else:  # Linear, LayerNormLinear, etc.  
                num_gemms = 1  

            self._module.init_fp8_metadata(num_gemms=num_gemms)
        if not self._module.fp8 and not self._module.primary_weights_in_fp8:
            return (self._elem,), (self._elem.requires_grad, None)
        # Use the actual data
        base = self._elem
        print("BASE: ", base)
        # Access the quantizer using fp8_meta_index
        if not isinstance(self._elem, QuantizedTensorBase):
            quantizer = self._module.quantizers["scaling_fwd"][self._fp8_meta_index]
            sharded_fp8_tensor = quantizer(self._elem)
        else:
            quantizer = self._elem._get_quantizer()
            sharded_fp8_tensor = self._elem
        if not isinstance(quantizer, MXFP8Quantizer) and not self._keep_fp8_weight_transpose_cache:
            quantizer.set_usage(columnwise=False)
        if isinstance(quantizer, Float8CurrentScalingQuantizer):
            quantizer.with_amax_reduction = True
        if isinstance(quantizer, MXFP8Quantizer):
            rowwise_data = sharded_fp8_tensor._rowwise_data if quantizer.rowwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            rowwise_scale_inv = sharded_fp8_tensor._rowwise_scale_inv if quantizer.rowwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            columnwise_data = sharded_fp8_tensor._columnwise_data if quantizer.columnwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            columnwise_scale_inv = sharded_fp8_tensor._columnwise_scale_inv if quantizer.columnwise_usage else torch.empty(0, dtype=torch.uint8, device=base.device)
            return (rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, ), (base.requires_grad,)
        return (sharded_fp8_tensor._data,), (base.requires_grad,quantizer)
        
    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (requires_grad, quantizer) = metadata
        if not self._module.fp8 and not self._module.primary_weights_in_fp8:
            (data,) = all_gather_outputs
            return data, all_gather_outputs
        # Retrieve the same quantizer you used in pre_all_gather
        shape = None
        if  not isinstance(quantizer, MXFP8Quantizer) and not self._keep_fp8_weight_transpose_cache:
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
