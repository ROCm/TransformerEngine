# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL flash attention kernels for gfx942/gfx950/gfx1250."""

# ---------------------------------------------------------------------------
# flydsl 0.3 compatibility.
#
# flydsl 0.3 removed the ``buffer_ops`` and ``vector`` submodules from
# ``flydsl.expr``.  Every kernel in this package imports them via
# ``from flydsl.expr import ..., buffer_ops, vector``.  Register the vendored
# 0.2.4 shims (ported to the 0.3 op API) onto ``flydsl.expr`` here — before any
# kernel submodule is imported — so those import lines keep resolving.
#
# Guarded by ``hasattr`` so this is a no-op on flydsl 0.2, where the real
# submodules are present.
# ---------------------------------------------------------------------------
import sys as _sys

import flydsl.expr as _flydsl_expr

from . import _flydsl03_compat as _compat

if not hasattr(_flydsl_expr, "buffer_ops"):
    _flydsl_expr.buffer_ops = _compat.buffer_ops
    _sys.modules["flydsl.expr.buffer_ops"] = _compat.buffer_ops
if not hasattr(_flydsl_expr, "vector"):
    _flydsl_expr.vector = _compat.vector
    _sys.modules["flydsl.expr.vector"] = _compat.vector

del _sys, _flydsl_expr, _compat
