# This file was modified for portability to AMDGPU
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

from . import flax
from .fp8 import fp8_autocast, update_collections, update_fp8_metas, get_delayed_scaling
from .fp8 import NVTE_FP8_COLLECTION_NAME
from .fp8 import jnp_float8_e4m3_type, jnp_float8_e5m2_type
from .sharding import MeshResource
from .sharding import MajorShardingType, ShardingResource, ShardingType
from .util import is_hip_extension

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum

MajorShardingType = DeprecatedEnum(MajorShardingType,
                                   "MajorShardingType is deprecating in the near feature.")
ShardingType = DeprecatedEnum(ShardingType, "ShardingType is deprecating in the near feature.")
ShardingResource = deprecate_wrapper(
    ShardingResource,
    "ShardingResource is renamed to MeshResource, and will be removed in the near feature.")

__all__ = [
    'NVTE_FP8_COLLECTION_NAME',
    'fp8_autocast',
    'update_collections',
    'update_fp8_metas',
    'get_delayed_scaling',
    'MeshResource',
    'MajorShardingType',
    'ShardingResource',
    'ShardingType',
    'flax',
    'praxis',
]
