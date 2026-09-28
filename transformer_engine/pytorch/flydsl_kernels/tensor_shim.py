# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Launch and buffer-descriptor helpers adapted from Primus-Turbo
# (https://github.com/AMD-AGI/Primus-Turbo):
#   primus_turbo/flydsl/utils/gemm_helper.py
#     (run_compiled; buffer-resource construction)
#   primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py
#     (compile-once wgrad launch)

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


def ptr_rsrc(ptr, num_records_bytes=None):
    """Convert an fx.Pointer kernel arg to a V# (``!llvm.ptr<8>``) for buffer DMA.

    Uses FlyDSL ``make_buffer_ptr`` + ``get_buffer_rsrc``. ``num_records_bytes``
    bounds the descriptor for hardware OOB zeros (padding / sentinel rows).
    """
    return fx.rocdl.get_buffer_rsrc(fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes))


class _FakeArg:
    """Sentinel returned by :func:`ptr_arg` for a fake/meta tensor.

    A fake/meta tensor has no real storage, so there is nothing to launch against.
    :func:`_run_compiled` detects this sentinel and skips the launch instead of dispatching
    with a null base pointer (see :func:`is_fake_tensor`).
    """

    __slots__ = ()


_FAKE_ARG = _FakeArg()


def is_fake_tensor(t) -> bool:
    """True for a FakeTensor / meta tensor (``torch.compile`` / meta tracing), which has no
    real ``data_ptr`` to launch against."""
    if getattr(t, "is_meta", False):
        return True
    type_name = type(t).__name__
    module_name = type(t).__module__
    return type_name == "FakeTensor" or "fake_tensor" in module_name


def ptr_arg(t: torch.Tensor):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch.

    Returns the :data:`_FAKE_ARG` sentinel for fake/meta tensors so the launch helpers can
    skip execution rather than dispatch with a null pointer.
    """
    if is_fake_tensor(t):
        return _FAKE_ARG
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.

    Under fake/meta tracing (any operand is a FakeTensor / meta tensor) there is no real
    storage, so skip the launch entirely -- dispatching with the null base pointers that
    ``ptr_arg`` would otherwise produce would launch a real kernel against address 0.
    """
    if any(a is _FAKE_ARG for a in args):
        return
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)
