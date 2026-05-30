# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""AITER availability detection and common utilities.

AITER is an optional pip dependency providing CK/Triton kernels for AMD GPUs.
All _lite modules should use these functions instead of per-file import checks.
"""

import functools


@functools.lru_cache(maxsize=1)
def is_aiter_available():
    """Check if AITER is installed and importable."""
    try:
        import aiter  # noqa: F401
        return True
    except ImportError:
        return False


def get_aiter():
    """Return the aiter module, or None if not installed."""
    if not is_aiter_available():
        return None
    import aiter
    return aiter


def get_aiter_rope():
    """Return aiter.ops.rope module, or None if not available."""
    if not is_aiter_available():
        return None
    try:
        from aiter.ops import rope
        return rope
    except (ImportError, AttributeError):
        return None
