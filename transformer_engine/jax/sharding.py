# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Sharding utilities for Transformer Engine in JAX.

This module provides utilities for managing tensor sharding in distributed training,
including support for various parallelism strategies like data parallelism (DP),
tensor parallelism (TP), pipeline parallelism (PP), and full-sharded data
parallelism (FSDP). It includes functions for sharding constraints, mesh management,
and collective operations.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from packaging import version
from typing import Callable, Optional
import warnings

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, get_abstract_mesh
import numpy as np

if version.parse(jax.__version__) < version.parse("0.9.0"):
    _PXLA_THREAD_RESOURCES = pxla.thread_resources

# Axis Names
BATCH_AXES = "nvte_batch"
SEQLEN_AXES = "nvte_seqlen"
SEQLEN_TP_AXES = "nvte_seqlen_tp"
SEQLEN_CP_AXES = "nvte_seqlen_cp"
HEAD_AXES = "nvte_head"
HIDDEN_AXES = "nvte_hidden"
HIDDEN_TP_AXES = "nvte_hidden_tp"
JOINED_AXES = "nvte_joined"
W_NO_SHARD_AXES = "nvte_w_no_shard"
W_FSDP_AXES = "nvte_w_fsdp"
W_TP_AXES = "nvte_w_tp"
W_JOINED_AXES = "nvte_w_joined"


def _get_mesh():
    # Handle Mesh's set via `with mesh:`
    # ROCm: add JAX version guard for all backends
    if version.parse(jax.__version__) < version.parse("0.9.0"):
        mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
        if mesh is not None and not mesh.empty:
            return mesh
    # Handle Mesh's set via `jax.set_mesh(mesh)`
    return jax.sharding.get_abstract_mesh()


def _get_mesh_info(resource: str, mesh: jax.sharding.Mesh):
    assert resource in mesh.axis_names, f"{resource} is not in the axis_names of Mesh {mesh}."
    return mesh.shape[resource], resource


def _validate_mesh_resource_configuration(mesh_resource):
    """Validate that the mesh resource configuration is consistent and conflict-free."""
    is_tp_enabled = (
        mesh_resource.tp_resource is not None and get_mesh_axis_size(mesh_resource.tp_resource) > 1
    )
    is_tpsp_enabled = (
        mesh_resource.tpsp_resource is not None
        and get_mesh_axis_size(mesh_resource.tpsp_resource) > 1
    )

    assert not (is_tp_enabled and is_tpsp_enabled), (
        "Tensor parallelism and tensor sequence parallelism cannot be enabled at the same time."
        f" Got tp_resource={mesh_resource.tp_resource} and"
        f" tpsp_resource={mesh_resource.tpsp_resource}"
    )


def is_mesh_available() -> bool:
    """
    Check if a physical mesh is available.
    """
    mesh = _get_mesh()
    return mesh is not None and not mesh.empty


def get_sharding_map_logic_axis_to_mesh_axis():
    """
    Generate a dict to map logical axes to mesh axes.
    """
    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        # If no mesh is defined, return an empty dict and do not require a MeshResource context to be present
        return {}

    abstract_mesh = get_abstract_mesh()
    if sorted(abstract_mesh.manual_axes) == sorted(mesh.axis_names):
        # If all mesh axes are manual axes, return an empty dict and do not require a MeshResource context to be present
        return {}

    gsr = global_mesh_resource()

    is_tpsp_enabled = gsr.tpsp_resource is not None and get_mesh_axis_size(gsr.tpsp_resource) > 1
    is_fsdp_enabled = gsr.fsdp_resource is not None and get_mesh_axis_size(gsr.fsdp_resource) > 1

    te_logical_axis_to_mesh_axis = {
        BATCH_AXES: gsr.fsdp_resource if is_fsdp_enabled else gsr.dp_resource,
        SEQLEN_AXES: None,
        SEQLEN_TP_AXES: gsr.tpsp_resource,
        SEQLEN_CP_AXES: gsr.cp_resource,
        HEAD_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        HIDDEN_AXES: None,
        HIDDEN_TP_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        JOINED_AXES: None,
        W_NO_SHARD_AXES: None,
        W_FSDP_AXES: gsr.fsdp_resource,
        W_TP_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        W_JOINED_AXES: None,
    }
    return te_logical_axis_to_mesh_axis


def _generate_pspec(logical_axis_names):
    """
    Convert TransformerEngine logical axes (e.g. BATCH_AXES) to a JAX PartitionSpec.
    Note, this method does not support Flax logical axes.

    Args:
        logical_axis_names: TransformerEngine logical axes to convert to a JAX PartitionSpec.
    Returns:
        A JAX PartitionSpec with the mesh axes corresponding to the given TransformerEngine logical axis names
    """
    rules = get_sharding_map_logic_axis_to_mesh_axis()

    mesh_axis_names = [rules.get(name) for name in logical_axis_names]
    pspec = jax.sharding.PartitionSpec(*mesh_axis_names)
    return pspec


def with_sharding_constraint(x: jnp.array, pspec: PartitionSpec):
    """
    A wrapper function to jax.lax.with_sharding_constraint
        1. Does nothing if mesh is empty.
        2. If all mesh axes are manual axes, replaces pspec with all Nones.
        3. Otherwise, strips only the manual axes.
    """
    if pspec is None:
        return x

    mesh = _get_mesh()
    if mesh.empty:
        return x

    # We want to exclude the axes that already used by shard_map and shard_map
    # only sets those in the abstract_mesh, not the physical one
    manual_axis_names = get_abstract_mesh().manual_axes

    # Multiple mesh axes can be mapped to a single shape axis, so we need to unpack and process tuples here too
    def filter_manual_axes(name_or_tuple):
        if isinstance(name_or_tuple, tuple):
            out = tuple(n for n in name_or_tuple if n not in manual_axis_names)
            if len(out) == 0:
                return None
            return out
        if name_or_tuple in manual_axis_names:
            return None
        return name_or_tuple

    cleaned_axis_names = tuple(filter_manual_axes(name_or_tuple) for name_or_tuple in pspec)

    if cleaned_axis_names == (None,) * len(cleaned_axis_names):
        return x

    cleaned_pspec = PartitionSpec(*cleaned_axis_names)

    # ROCm: JAX 0.9 compat (all backends) — when an AbstractMesh is active,
    # jax.lax.with_sharding_constraint requires the input to already carry a
    # NamedSharding. This affects both concrete arrays in eager mode and traced
    # values inside jax.jit whose abstract sharding is not a NamedSharding (e.g.
    # Module.init() traces over a single-device input and JAX propagates the
    # SingleDeviceSharding through the Tracer). In both cases the constraint must
    # be skipped because JAX raises unconditionally.
    # A UserWarning is emitted only for concrete (non-Tracer) arrays so the user
    # gets a visible signal in eager mode; the jit-traced skip is unavoidable and
    # kept silent to avoid spurious warnings from traced code.
    if hasattr(x, "sharding") and not isinstance(x.sharding, jax.sharding.NamedSharding):
        if not isinstance(x, jax.core.Tracer):
            warnings.warn(
                f"with_sharding_constraint: the sharding constraint {cleaned_pspec!r} was not"
                f" applied because the input array carries a {type(x.sharding).__name__} rather"
                " than a NamedSharding. This typically happens in eager mode when arrays have not"
                " yet been placed on a mesh (e.g. during model initialisation). Wrap the call in"
                " jax.jit or ensure the array is on a named mesh before applying sharding"
                " constraints.",
                UserWarning,
                stacklevel=2,
            )
        return x

    return jax.lax.with_sharding_constraint(x, cleaned_pspec)


def with_sharding_constraint_by_logical_axes(
    x: jnp.array, logical_axis_names: Optional[tuple | list]
):
    """
    A wrapper function to flax.linen.with_logical_constraint.

    DEPRECATED USE CASE: If no Flax logical axis rules are available, this function falls back to jax.lax.with_sharding_constraint using a hardcoded logical axis rule table from TE rules, such as BATCH_AXES. This functionality will be removed in the future.

    If logical_axis_names = None, this means no sharding constraint is applied.

    If logical_axis_names = (None, None, ...), this means a sharding constraint is applied and the tensor is replicated across all devices.

    Args:
        x: Input tensor to apply sharding constraint
        logical_axis_names: Logical axis names to apply sharding constraint
    Returns:
        Tensor with sharding constraint applied, or the original tensor if no logical axes are provided.

    """
    if not logical_axis_names:
        return x

    try:
        # Check if Flax logical axis rules are available, if so use them
        import flax

        flax_rules = flax.linen.get_logical_axis_rules()
        if len(flax_rules) > 0:
            pspec = flax.linen.logical_to_mesh_axes(logical_axis_names)
            return with_sharding_constraint(x, pspec)
    except ImportError:
        pass

    warnings.warn(
        "TransformerEngine logical axes, such as BATCH_AXES, SEQLEN_AXES, etc. are deprecated and"
        " will be removed in a future version. Please use Flax logical axes with a"
        " flax.linen.logical_axis_rules context and optionally use"
        " transformer_engine.jax.flax.extend_logical_axis_rules to add BATCH_AXES, etc. to your"
        " rules.",
        DeprecationWarning,
    )

    # If no logical axis rules are available from Flax, fallback to TE's hardcoded logical axis rule table
    assert len(x.shape) == len(logical_axis_names)
    pspec = _generate_pspec(logical_axis_names)
    return with_sharding_constraint(x, pspec)


def get_all_mesh_axes():
    """
    Get all name of mesh axes
    """
    mesh = _get_mesh()
    return mesh.axis_names


def get_padded_spec(spec, ndim):
    """
    Get padded spec for partitioning from arguments' information
    """
    if spec is None:
        return (None,) * ndim
    assert len(spec) <= ndim
    return spec + (None,) * (ndim - len(spec))


def lax_paral_op(
    x: jnp.array, ops: Callable, mesh_resource: str, mesh: jax.sharding.Mesh, **kwargs
):
    """
    A wrapper function to invoke lax.p* operations, like psum.
    """
    if mesh_resource is not None:
        _, resource = _get_mesh_info(mesh_resource, mesh)
        return ops(x, resource, **kwargs)
    return x


def num_of_devices():
    """
    Get total number of detected devices
    """
    return len(jax.devices())


def get_num_devices_in_mesh(mesh=None):
    """
    Get the number of devices in the given mesh.
    If the mesh is None, it would be replaced
    by the global mesh.
    """
    if mesh is None:
        mesh = _get_mesh()
    if mesh.empty:
        return 1
    return np.prod(list(mesh.shape.values()))


def get_mesh_axis_size(axis, mesh=None):
    """
    Get the axis size of the given mesh.
    If the mesh is None, it would be replaced
    by the global mesh.
    """
    if mesh is None:
        mesh = _get_mesh()

    if axis is None:
        return 1

    assert axis in mesh.shape, f"{axis} is not a axis of the given mesh {mesh.shape}"
    return mesh.shape[axis]


def get_mesh_axis_rank(axis: str, mesh=None):
    """
    Gets the local axis rank of the `axis` of the array.
    If the mesh is None the rank is 0.
    """
    if mesh is None:
        return 0
    _, axis_name = _get_mesh_info(axis, mesh)
    return jax.lax.axis_index(axis_name)


def get_mesh_axis_rank_host(axis, mesh) -> int:
    """
    Same as get_mesh_axis_rank(), but return a host value instead of a
    traced device value.
    """
    if axis not in mesh.axis_names:
        raise ValueError(f"Axis {axis} not found in mesh axis names: {mesh.axis_names}")

    axis_index = mesh.axis_names.index(axis)

    # Convert mesh.devices (ndarray of Device objects) to flat list
    devices = mesh.devices
    local_device = jax.devices()[jax.process_index()]  # Pick one device on this host

    # Find index of local_device in mesh.devices
    coords = np.argwhere(devices == local_device)
    if coords.size == 0:
        raise ValueError(f"Local device {local_device} not found in mesh.devices.")
    coords = tuple(coords[0])  # Coordinates in the mesh array

    # Get the mesh rank along the specified axis
    rank = coords[axis_index]
    return int(rank)


@dataclass
class MeshResource:
    """A data container for managing mesh resources in distributed training.

    This class defines the mapping between logical axes and physical mesh axes
    for different types of parallelism in distributed training.

    Attributes:
        dp_resource: Axis name for data parallelism (batch sharding), default is None
        tp_resource: Axis name for tensor parallelism (hidden dimension sharding), default is None
        tpsp_resource: Axis name for tensor sequence parallelism (hidden and sequence sharding), default is None
        fsdp_resource: Axis name for full-sharded data parallelism, default is None
        pp_resource: Axis name for pipeline parallelism (layer sharding), default is None
        cp_resource: Axis name for context parallelism (sequence sharding), default is None
        ep_resource: Axis name for expert parallelism. Dispatch input tokens
            must be sharded on their leading dim by ``ep_resource`` (alone or
            compound with ``dp_resource`` / ``fsdp_resource`` as outer, e.g.
            ``PartitionSpec(("dp", "ep"), None, None)``). Dispatch output
            ``[ep_size, recv_capacity, H]`` is always sharded by ``ep_resource``
            on the leading ``ep_size`` dim.
    """

    dp_resource: str = None
    tp_resource: str = None
    tpsp_resource: str = None
    fsdp_resource: str = None
    pp_resource: str = None
    cp_resource: str = None
    ep_resource: str = None


_GLOBAL_MESH_RESOURCE = None
# ROCm: True once _validate_mesh_resource_configuration has successfully run for the
# current _GLOBAL_MESH_RESOURCE.  Reset to False on every global_shard_guard
# entry so that a new resource is always (re-)validated on first use.
_GLOBAL_MESH_RESOURCE_VALIDATED = False


@contextmanager
def global_shard_guard(resource: MeshResource):
    """Context manager for setting global sharding configuration.

    This context manager allows temporarily setting the global mesh resource
    configuration for sharding operations.

    Args:
        resource: MeshResource instance defining the sharding configuration
    """
    global _GLOBAL_MESH_RESOURCE, _GLOBAL_MESH_RESOURCE_VALIDATED
    old_resources = _GLOBAL_MESH_RESOURCE
    old_validated = _GLOBAL_MESH_RESOURCE_VALIDATED
    try:
        _GLOBAL_MESH_RESOURCE = resource
        # ROCm: JAX 0.9 compat (all backends)
        # Attempt early (eager) validation if a mesh is already active at
        # guard-entry time.  Guard with is_mesh_available() so that callers
        # who enter global_shard_guard before any JAX mesh context is active
        # (e.g. maxtext's transformer_engine_context) do not hit an
        # AssertionError in get_mesh_axis_size() when get_abstract_mesh()
        # returns an empty OrderedDict().
        # Reset the validated flag for the new resource so that
        # global_mesh_resource() re-validates on its first call with an
        # active mesh (lazy validation path, see below).
        _GLOBAL_MESH_RESOURCE_VALIDATED = False
        if resource is not None and is_mesh_available():
            _validate_mesh_resource_configuration(resource)
            _GLOBAL_MESH_RESOURCE_VALIDATED = True
        yield
    finally:
        _GLOBAL_MESH_RESOURCE = old_resources
        _GLOBAL_MESH_RESOURCE_VALIDATED = old_validated


def global_mesh_resource() -> MeshResource:
    """Get the current global mesh resource configuration.

    Returns:
        The current MeshResource instance
    """
    global _GLOBAL_MESH_RESOURCE_VALIDATED
    assert _GLOBAL_MESH_RESOURCE is not None, (
        "Global mesh resource is not set. Please set the MeshResource via a global_shard_guard"
        " context. If you are not using multiple GPUs, you can use an empty MeshResource by"
        " wrapping your program in 'with global_shard_guard(MeshResource()):'"
    )
    # ROCm: JAX 0.9 compat (all backends)
    # Lazy validation: if the mesh was not yet active when global_shard_guard
    # was entered (eager validation skipped), validate here on the first call
    # that actually finds an active mesh.  This covers frameworks like maxtext
    # that set up global_shard_guard before activating the JAX mesh context.
    #
    # The _GLOBAL_MESH_RESOURCE_VALIDATED flag ensures validation runs at most
    # once per global_shard_guard context (reset to False on guard entry,
    # set to True after successful validation):
    #   • After validation: `not _GLOBAL_MESH_RESOURCE_VALIDATED` is False →
    #     only one boolean check per call, faster than the pre-JAX-0.9-compat
    #     code that ran get_mesh_axis_size() unconditionally on every call.
    #   • Inside jit(...).lower(): is_mesh_available() returns False (JAX 0.9
    #     get_abstract_mesh() is empty there) → validation safely skipped.
    if not _GLOBAL_MESH_RESOURCE_VALIDATED and is_mesh_available():
        _validate_mesh_resource_configuration(_GLOBAL_MESH_RESOURCE)
        _GLOBAL_MESH_RESOURCE_VALIDATED = True
    return _GLOBAL_MESH_RESOURCE


def get_active_resource_axis(resource_name: str) -> Optional[str]:
    """Resolve a :class:`MeshResource` attribute to its mesh axis name,
    or return ``None`` if that resource is not active.

    "Active" means all three are true:

    * a physical mesh is set (``is_mesh_available()``),
    * the ``MeshResource`` attribute is non-``None``,
    * the corresponding mesh axis has more than 1 device.

    Mirrors the three-step ``is_X_enabled`` idiom in
    :func:`get_sharding_map_logic_axis_to_mesh_axis` but returns the
    axis name itself (or ``None``) so callers can use it directly in
    collectives / ``shard_map`` specs.

    Args:
        resource_name: Attribute name on :class:`MeshResource`, e.g.
            ``"fsdp_resource"`` or ``"ep_resource"``.

    Returns:
        The mesh axis name when active, else ``None``.
    """
    if not is_mesh_available():
        return None
    if _GLOBAL_MESH_RESOURCE is None:
        return None
    axis = getattr(_GLOBAL_MESH_RESOURCE, resource_name)
    if axis is None or get_mesh_axis_size(axis) <= 1:
        return None
    return axis


def all_reduce_sum_along_dp_fsdp(x: jnp.array, mesh: jax.sharding.Mesh):
    """Perform all-reduce sum operation along data parallelism and FSDP axes.

    Args:
        x: Input tensor to reduce
        mesh: JAX mesh for distributed computation

    Returns:
        Reduced tensor
    """
    x = lax_paral_op(x, jax.lax.psum, global_mesh_resource().dp_resource, mesh)
    return lax_paral_op(x, jax.lax.psum, global_mesh_resource().fsdp_resource, mesh)


def all_reduce_sum_along_dp_fsdp_tpsp(x: jnp.array, mesh: jax.sharding.Mesh):
    """Perform all-reduce sum operation along data parallelism and sequence parallelism axes.

    Args:
        x: Input tensor to reduce
        mesh: JAX mesh for distributed computation

    Returns:
        Reduced tensor
    """
    x = lax_paral_op(x, jax.lax.psum, global_mesh_resource().tpsp_resource, mesh)
    x = lax_paral_op(x, jax.lax.psum, global_mesh_resource().dp_resource, mesh)
    return lax_paral_op(x, jax.lax.psum, global_mesh_resource().fsdp_resource, mesh)


def all_reduce_max_along_all_axes_except_PP(x: jnp.array, mesh: jax.sharding.Mesh):
    """Perform all-reduce max operation along all axes except pipeline parallelism.

    Args:
        x: Input tensor to reduce
        mesh: JAX mesh for distributed computation

    Returns:
        Reduced tensor
    """
    # ROCm: JAX 0.9 compat (all backends)
    # Use mesh.axis_names from the concrete mesh argument rather than calling
    # get_all_mesh_axes() → _get_mesh() → get_abstract_mesh(), which returns
    # empty in JAX 0.9 when called from inside a custom_partitioning sharded_impl.
    for axis in mesh.axis_names:
        if axis != global_mesh_resource().pp_resource:
            x = lax_paral_op(x, jax.lax.pmax, axis, mesh)
    return x


def tpsp_axis_size():
    """
    Get the size of the tensor parallelism axis.
    Return 1 if no TP axis is set.
    """
    return get_mesh_axis_size(global_mesh_resource().tpsp_resource)


def dp_or_fsdp_axis_size():
    """
    Get the size of the data parallelism or FSDP axis.
    Return 1 if no DP/FSDP axis is set.
    """
    dp_size = get_mesh_axis_size(global_mesh_resource().dp_resource)
    fsdp_size = get_mesh_axis_size(global_mesh_resource().fsdp_resource)
    return dp_size if dp_size > 1 else fsdp_size


def ep_axis_size():
    """Get the size of the dispatch/EP axis (ep_resource). Returns 1 if unset."""
    return get_mesh_axis_size(global_mesh_resource().ep_resource)
