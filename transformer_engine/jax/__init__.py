# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import logging
import importlib
import importlib.util
import ctypes
from importlib.metadata import version

from transformer_engine.common import get_te_path, is_package_installed
from transformer_engine.common import _get_sys_extension
from .util import is_hip_extension

_logger = logging.getLogger(__name__)


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    module_name = "transformer_engine_jax"
    te_cuda_vers = "rocm" if is_hip_extension() else "cu12"

    if is_package_installed(module_name):
        assert is_package_installed("transformer_engine"), "Could not find `transformer-engine`."
        assert is_package_installed(
            f"transformer_engine_{te_cuda_vers}"
        ), f"Could not find `transformer-engine-{te_cuda_vers}`."
        assert (
            version(module_name)
            == version("transformer-engine")
            == version(f"transformer-engine-{te_cuda_vers}")
        ), (
            "TransformerEngine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and transformer-engine-{te_cuda_vers}"
            f" v{version(f'transformer-engine-{te_cuda_vers}')}."
            " Install transformer-engine using 'pip install"
            " transformer-engine[jax]==VERSION'"
        )

    if is_package_installed(f"transformer-engine-{te_cuda_vers}"):
        if not is_package_installed(module_name):
            _logger.info(
                "Could not find package %s. Install transformer-engine using "
                "'pip3 install transformer-engine[jax]==VERSION'",
                module_name,
            )

    extension = _get_sys_extension()
    try:
        so_dir = get_te_path() / "transformer_engine"
        so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
    except StopIteration:
        try:
            so_dir = get_te_path() / "transformer_engine" / "wheel_lib"
            so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
        except StopIteration:
            so_dir = get_te_path()
            so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)


_TE_JAX_LIB_CTYPES = _load_library()
from . import flax
from .fp8 import fp8_autocast, update_collections, get_delayed_scaling
from .fp8 import NVTE_FP8_COLLECTION_NAME
from .sharding import MeshResource
from .sharding import MajorShardingType, ShardingResource, ShardingType

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum

MajorShardingType = DeprecatedEnum(
    MajorShardingType, "MajorShardingType is deprecating in the near feature."
)
ShardingType = DeprecatedEnum(ShardingType, "ShardingType is deprecating in the near feature.")
ShardingResource = deprecate_wrapper(
    ShardingResource,
    "ShardingResource is renamed to MeshResource, and will be removed in the near feature.",
)

__all__ = [
    "NVTE_FP8_COLLECTION_NAME",
    "fp8_autocast",
    "update_collections",
    "get_delayed_scaling",
    "MeshResource",
    "MajorShardingType",
    "ShardingResource",
    "ShardingType",
    "flax",
    "praxis",
]
