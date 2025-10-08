from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
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
class FSDPAGFloat8Tensor(torch.Tensor):

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
        #The underlying tensor
        self._elem = tensor
        # Where quantizers are present
        self._module = module
        # Which quantizer to use within module.quantizers["scaling_fwd"][idx]
        self._fp8_meta_index = fp8_meta_index
        # Disable or enable transpose cache for fp8 weights
        self._keep_fp8_weight_transpose_cache = keep_fp8_weight_transpose_cache

    
    def __repr__(self):
            return (
                f"FSDPAGFloat8Tensor("
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
        return FSDPAGFloat8Tensor(
            inner_tensors["_elem"],
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
            assert isinstance(t, cls), f"Unexpected detach input type: {type(t)}"
            detached = t._elem.detach()
            return cls(detached, module=t._module, fp8_meta_index=t._fp8_meta_index, keep_fp8_weight_transpose_cache=t._keep_fp8_weight_transpose_cache)

        # Unwrap only our subclass; capture shared metadata for rewrapping
        meta: Optional[tuple[nn.Module, str, bool]] = None

        def unwrap(x):
            nonlocal meta
            if isinstance(x, cls):
                if meta is None:
                    meta = (x._module, x._fp8_meta_index, x._keep_fp8_weight_transpose_cache)
                else:
                    # Require consistency when multiple wrappers are involved in a single op
                    # same_mod = (meta[0] is x._module)
                    same_idx = (meta[1] == x._fp8_meta_index)
                    same_flag = (meta[2] == x._keep_fp8_weight_transpose_cache)
                    assert same_idx and same_flag, (
                        "Mixed FSDPAGFloat8Tensor metadata in one op is not supported"
                    )
                return x._elem
            return x

        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(cls, unwrap, (args, kwargs))

        # Run the actual op on internal tensors
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # Rewrap outputs only for ops that need to preserve subclass identity
        if func not in _ops_to_preserve_subclass or meta is None:
            return out

        def rewrap(x):
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
            self._module.init_fp8_metadata()
        # Use the actual data
        base = self._elem
        # Access the quantizer using fp8_meta_index
        quantizer = self._module.quantizers["scaling_fwd"][self._fp8_meta_index]
        if not self._keep_fp8_weight_transpose_cache:
            quantizer.columnwise_usage=False
        sharded_fp8_tensor = quantizer(base)
        transpose_to_send = sharded_fp8_tensor._transpose if self._keep_fp8_weight_transpose_cache else torch.empty(0, dtype=base.dtype, device=base.device)
        return (sharded_fp8_tensor._data, transpose_to_send,), (base.requires_grad,)
        
    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        # Recompose the Float8Tensor from the wire format
        (data, data_transpose) = all_gather_outputs
        (requires_grad, ) = metadata

        # Retrieve the same quantizer you used in pre_all_gather
        quantizer = self._module.quantizers["scaling_fwd"][self._fp8_meta_index]

        if out is not None:
            # If FSDP provided a pre-allocated output (happens in subsequent iterations),
            # fill in the missing bits and still return the expected values.
            assert isinstance(out, Float8Tensor), f"Unexpected out type: {type(out)}"
            out._scale_inv = 1 / quantizer.scale
            # Depending on FSDP's expected return type, return (materialized_param, aux)
            return out, all_gather_outputs

        # Otherwise, construct a new Float8Tensor that wraps the gathered data
        out_fp8 = Float8Tensor(
            shape=data.shape,
            dtype=param_dtype,                    # or self._elem.dtype
            requires_grad=requires_grad,
            data=data,
            fp8_scale_inv=1 / quantizer.scale,
            fp8_dtype=quantizer.dtype,
            data_transpose=None if data_transpose.numel() == 0 else data_transpose,
            quantizer=quantizer,
        )
        return out_fp8, all_gather_outputs
