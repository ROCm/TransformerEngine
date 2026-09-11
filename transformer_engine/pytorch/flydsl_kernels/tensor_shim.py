# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


def ptr_rsrc(ptr, num_records_bytes=None):
    """Convert an fx.Pointer kernel arg to a V# (``!llvm.ptr<8>``) for buffer DMA.

    Uses FlyDSL ``make_buffer_ptr`` + ``get_buffer_rsrc``. ``num_records_bytes``
    bounds the descriptor for hardware OOB zeros (padding / sentinel rows).
    """
    return fx.rocdl.get_buffer_rsrc(fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes))


def ptr_arg(t: torch.Tensor):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
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
