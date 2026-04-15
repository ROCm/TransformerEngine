# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Module level PyTorch APIs"""
import os as _os

from .layernorm_linear import LayerNormLinear
from .linear import Linear
from .grouped_linear import GroupedLinear
from .layernorm_mlp import LayerNormMLP
from .layernorm import LayerNorm
from .rmsnorm import RMSNorm
from .fp8_padding import Fp8Padding
from .fp8_unpadding import Fp8Unpadding
from .base import initialize_ub, destroy_ub, UserBufferQuantizationMode

# In lite mode, replace the full-build fused modules with lite-native versions
if _os.environ.get("NVTE_LITE", "0") == "1":
    from .._lite.fused_layernorm_linear import LayerNormLinear  # noqa: F811
    from .._lite.fused_layernorm_mlp import LayerNormMLP  # noqa: F811
