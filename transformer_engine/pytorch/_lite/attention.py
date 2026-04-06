# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention operations -- multi-backend: SDPA, AITER, flash-attn.

TODO Phase 3: Full implementation with QKV format translation.
"""

import torch
import torch.nn.functional as F

from .enums import NVTE_Fused_Attn_Backend


# Try to import AITER
_aiter_available = False
try:
    import aiter
    _aiter_available = True
except ImportError:
    pass

# Try to import flash-attn
_flash_attn_available = False
try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    pass


def get_fused_attn_backend(*args, **kwargs):
    """Get the fused attention backend to use.

    In lite mode, we prefer: AITER > flash-attn > SDPA.
    """
    if _aiter_available:
        return NVTE_Fused_Attn_Backend.NVTE_CK
    if _flash_attn_available:
        return NVTE_Fused_Attn_Backend.NVTE_Flash
    return NVTE_Fused_Attn_Backend.NVTE_SDPA


def fused_attn_fwd(*args, **kwargs):
    """Fused attention forward.

    TODO Phase 3: Full implementation with QKV format translation and
    multi-backend dispatch (SDPA / AITER / flash-attn).
    """
    raise NotImplementedError(
        "Fused attention forward not yet implemented in lite mode. "
        "Use DotProductAttention with the 'unfused' backend as a workaround."
    )


def fused_attn_bwd(*args, **kwargs):
    """Fused attention backward.

    TODO Phase 3: Full implementation.
    """
    raise NotImplementedError(
        "Fused attention backward not yet implemented in lite mode. "
        "Use DotProductAttention with the 'unfused' backend as a workaround."
    )


def fa_prepare_fwd(*args, **kwargs):
    """Prepare QKV for Flash Attention.

    TODO Phase 3: Implement QKV format conversion.
    """
    raise NotImplementedError("fa_prepare_fwd not yet implemented in lite mode.")


def fa_prepare_bwd(*args, **kwargs):
    """Backward of QKV preparation for Flash Attention."""
    raise NotImplementedError("fa_prepare_bwd not yet implemented in lite mode.")


def copy_to_kv_cache(*args, **kwargs):
    """Copy new KV tokens to KV cache.

    TODO Phase 3: Implement as simple tensor copy/index operation.
    """
    raise NotImplementedError("copy_to_kv_cache not yet implemented in lite mode.")


def convert_thd_to_bshd(*args, **kwargs):
    """Convert tensor from THD to BSHD format.

    TODO Phase 3: Implement as PyTorch reshape/pad operations.
    """
    raise NotImplementedError("convert_thd_to_bshd not yet implemented in lite mode.")


def convert_bshd_to_thd(*args, **kwargs):
    """Convert tensor from BSHD to THD format.

    TODO Phase 3: Implement as PyTorch reshape operations.
    """
    raise NotImplementedError("convert_bshd_to_thd not yet implemented in lite mode.")
