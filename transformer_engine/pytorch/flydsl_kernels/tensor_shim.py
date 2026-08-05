# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import flydsl.compiler as flyc
from flydsl.expr import arith, buffer_ops, ptrtoint
from flydsl.expr.typing import T


def ptr_rsrc(ptr):
    """Convert an fx.Pointer kernel arg to a buffer resource for buffer_load/store."""
    addr_i64 = arith.index_cast(T.i64, ptrtoint(ptr))
    return buffer_ops.create_buffer_resource_from_addr(addr_i64)


def ptr_arg(t: torch.Tensor):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
    import flydsl.expr as fx

    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)
