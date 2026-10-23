# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""
ROCM_PATH initialization for Transformer Engine wheels.
"""

from __future__ import annotations

import os


def initialize() -> None:
    """
    Initialize the ROCM_PATH environment variable.
    """
    if not os.getenv("ROCM_PATH"):
        try:
            from rocm_sdk._devel import get_devel_root
            os.environ["ROCM_PATH"] = str(get_devel_root())
            return
        except ImportError:
            pass

        for _candidate in ("/opt/rocm/core", "/opt/rocm"):
            if os.path.exists(_candidate):
                os.environ["ROCM_PATH"] = _candidate
                break
