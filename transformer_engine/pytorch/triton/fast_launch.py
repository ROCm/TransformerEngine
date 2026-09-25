# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from AITER (ROCm/aiter @ 7d2f6a51a,
# aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py), MIT licensed.

"""Launch a Triton kernel without redoing the part that only depends on shapes.

``JITFunction.run`` binds the arguments, specializes each one, builds a cache
key and looks the kernel up on every call -- ~51us per dispatch against ~3us
for the launch itself. For multi-launch pipelines like KDA that host cost
decides which pipeline measures faster. Wrap a kernel once and call it as
before::

    _kernel = fast_launch(_kernel)
    _kernel[grid](arg=..., ..., num_warps=4)

Arguments must be passed by keyword. The cache key pins every non-tensor
argument by value and every tensor by (dtype, 16-byte alignment, int32
addressability), matching what Triton specializes on. This relies on Triton
internals (the JIT binder and ``CompiledKernel.run``); if they are not
available the wrapper falls back to ordinary launches.
"""

from collections import ChainMap, OrderedDict

import torch
from triton.runtime import driver
from triton.runtime.jit import JITFunction

_MAX_INT32 = 2**31 - 1

# Entries per wrapped kernel, past which the least recently used one goes. A
# miss is an ordinary Triton dispatch that still hits Triton's compile cache.
_MAX_ENTRIES = 1024


def _tensor_key(t: torch.Tensor):
    return (
        t.dtype,
        t.data_ptr() % 16 == 0,
        t.untyped_storage().size() <= _MAX_INT32,
    )


class _FastLaunch:
    """A kernel that remembers what it compiled for a given set of shapes."""

    def __init__(self, kernel):
        self._kernel = kernel
        self._cache = OrderedDict()
        jit = kernel
        while not isinstance(jit, JITFunction):
            jit = jit.fn
        self._jit = jit
        self._disabled = False

    def __getitem__(self, grid):
        def launch(**kwargs):
            if self._disabled:
                return self._kernel[grid](**kwargs)
            device = driver.active.get_current_device()
            try:
                key = (
                    device,
                    tuple(kwargs.keys()),
                    tuple(
                        _tensor_key(v) if isinstance(v, torch.Tensor) else v
                        for v in kwargs.values()
                    ),
                )
                entry = self._cache.get(key)
            except TypeError:
                # An argument that cannot be hashed cannot be guarded.
                self._disabled = True
                return self._kernel[grid](**kwargs)

            if entry is None:
                return self._capture(grid, kwargs, key)
            self._cache.move_to_end(key)

            compiled, names, frozen = entry
            g = grid(ChainMap(kwargs, frozen)) if callable(grid) else grid
            # Not hoisted: stream capture redirects work onto its own stream.
            stream = driver.active.get_current_stream(device)
            compiled.run(
                g[0],
                g[1] if len(g) > 1 else 1,
                g[2] if len(g) > 2 else 1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *[kwargs[n] if n in kwargs else frozen[n] for n in names],
            )
            return compiled

        return launch

    def _capture(self, grid, kwargs, key):
        """Launch the ordinary way, and keep what it worked out."""
        seen = []

        def hook(*args, **kw):
            seen.append((args, kw))

        try:
            self._jit.pre_run_hooks.append(hook)
        except AttributeError:
            self._disabled = True
            return self._kernel[grid](**kwargs)
        try:
            compiled = self._kernel[grid](**kwargs)
        finally:
            self._jit.pre_run_hooks.remove(hook)
        if compiled is None or not seen:
            return compiled

        args, kw = seen[-1]
        device = driver.active.get_current_device()
        try:
            binder = self._jit.device_caches[device][4]
            bound_args, _, _ = binder(*args, **kw)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            self._disabled = True
            return compiled

        names = list(bound_args.keys())
        # Arguments the call site does not pass (defaults). The key pins every
        # non-tensor value, so these are constant for this entry.
        frozen = {n: v for n, v in bound_args.items() if n not in kwargs}
        self._cache[key] = (compiled, names, frozen)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return compiled

    def clear(self):
        """Drop every entry."""
        self._cache.clear()


def fast_launch(kernel):
    """Wrap a ``@triton.jit`` / ``@gluon.jit`` kernel for low-overhead keyword launches."""
    return _FastLaunch(kernel)
