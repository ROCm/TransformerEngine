# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL attention kernels (DeepSeek-V4 sparse MLA, gfx950).

Importing this package does **not** import FlyDSL: the kernel module is imported
lazily inside :func:`sparse_mla_attn_fwd` once availability has been confirmed.
"""

from ..exceptions import FlyDSLUnsupportedError
from .attention_wrappers import sparse_mla_attn_fwd

__all__ = [
    "FlyDSLUnsupportedError",
    "sparse_mla_attn_fwd",
]
