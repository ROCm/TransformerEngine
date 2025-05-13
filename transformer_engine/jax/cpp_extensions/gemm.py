# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict, List
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp
from transformer_engine.transformer_engine_jax import get_device_compute_capability

from .base import BasePrimitive, register_primitive

from .misc import is_hip_extension

__all__ = ["gemm", "grouped_gemm"]


num_cublas_streams = 4


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM
    """

    name = "te_grouped_gemm_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(*args, num_gemms, out_dtype, has_bias):
        """
        Args:
            *args: Size num_gemms * 4 or num_gemms * 5 depending on has_bias:
                args[  0         :   num_gemms] are the lhs tensors,
                args[  num_gemms : 2*num_gemms] are the rhs tensors,
                args[2*num_gemms : 3*num_gemms] are the lhs scale_inv tensors,
                args[3*num_gemms : 4*num_gemms] are the rhs scale_inv tensors,
                args[4*num_gemms : 5*num_gemms] are the bias tensors if has_bias is True.
            num_gemms: Number of GEMM operations to perform.
            out_dtype: Data type of the output tensors.
            has_bias: Boolean indicating if bias tensors are provided.

        Returns:
           A tuple of ShapedArray objects of size num_gemms+1:
               ret[0 : num_gemms]: GEMM output tensors,
               ret[num_gemms]:workspace tensor.
        """
        expected_num_args = 5 * num_gemms if has_bias else 4 * num_gemms
        assert (
            len(args) == expected_num_args
        ), f"Expected {expected_num_args} input arguments, but got {len(args)}"
        A_list = args[0:num_gemms]
        B_list = args[num_gemms : 2 * num_gemms]
        # A and B have shapes [1, m, k] and [1, n, k]
        out_list_aval = tuple(
            jax.core.ShapedArray((A.shape[1], B.shape[1]), dtype=out_dtype)
            for A, B in zip(A_list, B_list)
        )
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)
        return (*out_list_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(ctx, *args, num_gemms, out_dtype, has_bias):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            num_gemms=num_gemms,
            has_bias=has_bias,
        )

    @staticmethod
    def impl(*args, num_gemms, out_dtype, has_bias):
        assert GroupedGemmPrimitive.inner_primitive is not None
        out = GroupedGemmPrimitive.inner_primitive.bind(
            *args,
            num_gemms=num_gemms,
            out_dtype=out_dtype,
            has_bias=has_bias,
        )
        return out[:-1]  # out is [out_list, wkspace], only return out_list


register_primitive(GroupedGemmPrimitive)


def _shape_normalization(x, dimension_numbers, already_transposed: bool = False):
    orig_order = list(range(x.ndim))
    contracting_dims, batch_dims = dimension_numbers
    contracting_order = [d for d in orig_order if d in contracting_dims]
    batch_order = [d for d in orig_order if d in batch_dims]
    non_contracting_order = [
        d for d in orig_order if d not in contracting_dims and d not in batch_dims
    ]
    batch_shape = [x.shape[d] for d in batch_order]
    rows_shape = [x.shape[d] for d in non_contracting_order]
    cols_shape = [x.shape[d] for d in contracting_order]
    new_order = batch_order + non_contracting_order + contracting_order
    rows, cols, batches = (
        reduce(operator.mul, rows_shape, 1),
        reduce(operator.mul, cols_shape, 1),
        reduce(operator.mul, batch_shape, 1),
    )
    # Remove this transpose when non-TN dot is supported
    if not already_transposed:
        t = jnp.transpose(x, new_order)
    else:
        t = x
    return jnp.reshape(t, (batches, rows, cols))


def _calculate_remaining_shape(shape, contracting_dims):
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def _jax_gemm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
) -> jnp.ndarray:
    """General matrix multiplication using JAX's dot_general."""
    dim_nums = (contracting_dims, ((), ()))
    return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)


def gemm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
) -> jnp.ndarray:
    """General matrix multiplication.

    Args:
        lhs: First input matrix.
        rhs: Second input matrix.
        contracting_dims: Tuple of two sequences representing the contracting dimensions.
            The first sequence represents the contracting dimensions of the first matrix,
            and the second sequence represents the contracting dimensions of the second matrix.

    Returns:
        The matrix multiplication result.
        Shape: (M, N)
        Dtype: Same as input dtype
    """

    return _jax_gemm(lhs, rhs, contracting_dims)


def grouped_gemm(
    lhs_list: jnp.ndarray,
    rhs_list: jnp.ndarray,
    contracting_dims_list: List[Tuple[Sequence[int], Sequence[int]]],
    bias_list: List[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Grouped GEMM for multiple pairs of tensors."""
    assert (
        len(lhs_list) == len(rhs_list) == len(contracting_dims_list)
    ), "lhs_list, rhs_list, contracting_dims_list must have the same length"

    num_gemms = len(lhs_list)
    lhs_list_ = []
    rhs_list_ = []
    lhs_sinv_list_ = []
    rhs_sinv_list_ = []
    bias_list_ = []
    for i in range(num_gemms):
        lhs = lhs_list[i]
        rhs = rhs_list[i]
        contracting_dims = contracting_dims_list[i]
        dim_nums = (contracting_dims, ((), ()))

        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        out_dtype = lhs.dtype

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_dn = (lhs_contract, lhs_batch)
        rhs_dn = (rhs_contract, rhs_batch)

        lhs_remain_shape = _calculate_remaining_shape(lhs_shape, lhs_contract)
        rhs_remain_shape = _calculate_remaining_shape(rhs_shape, rhs_contract)

        lhs_3d = _shape_normalization(lhs, lhs_dn)
        rhs_3d = _shape_normalization(rhs, rhs_dn)

        # Note: already_transposed doesn't matter for the output shape
        # x.shape = [B, D1, D2]
        # contracting_dims = (2, )    --> output.shape = [1, B * D1, D2]
        # contracting_dims = (0, 1, ) --> output.shape = [1, D2, B * D1]
        # x.shape = [D1, D2]
        # contracting_dims = (1, )    --> output.shape = [1, D1, D2]
        # contracting_dims = (0, )    --> output.shape = [1, D2, D1]
        bm = lhs_remain_shape[0]
        bn = rhs_remain_shape[0]
        kl = lhs_3d.shape[-1]
        kr = rhs_3d.shape[-1]
        assert kl == kr, f"After shape normalization, contracting dim size mismatch: {kl} != {kr}"
        if (bm % 16 != 0) or (bn % 16 != 0) or (kl % 16 != 0):
            print("grouped_gemm input pair {i} has invalid problem shape for lowering: ")
            print(f"m = {bm}, n = {bn}, k = {kl}; ")
            print("cuBLAS requires the problem shapes being multiples of 16")
            assert (bm % 16 == 0) and (bn % 16 == 0) and (kl % 16 == 0)

        lhs_list_.append(lhs_3d)
        rhs_list_.append(rhs_3d)

        lhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
        rhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))

        if bias_list is not None:
            bias_list_.append(bias_list[i])

    out_list = GroupedGemmPrimitive.outer_primitive.bind(
        *lhs_list_,
        *rhs_list_,
        *lhs_sinv_list_,
        *rhs_sinv_list_,
        *bias_list_,
        num_gemms=num_gemms,
        out_dtype=out_dtype,
        has_bias=1 if bias_list is not None else 0,
    )

    return out_list
