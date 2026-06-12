# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Shared helpers for the standalone indexer-kernel benchmarks (torch-only)."""

import time

import torch


def make_kernel_inputs(B, oH, T_t, T_s, H, d_i, dtype, device="cuda", seed=0):
    """Build the projected (Hq, Hk, W_o) the kernels actually consume.

    The original profilers generated Q/K/W and ran the einsum projections
    (C_q, H_q, H_k, W_o) before the kernel. Those projections are plain GEMMs,
    not part of the Triton kernels under test, so here we sample the kernel
    inputs directly.

        Hq:  (B, oH, T_t, H, d_i)
        Hk:  (B, oH, T_s, d_i)
        W_o: (B, oH, T_t, H)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    Hq = torch.randn((B, oH, T_t, H, d_i), dtype=dtype, device=device, generator=g)
    Hk = torch.randn((B, oH, T_s, d_i), dtype=dtype, device=device, generator=g)
    W_o = torch.randn((B, oH, T_t, H), dtype=dtype, device=device, generator=g)
    return Hq, Hk, W_o


def time_fn(fn, n_warmup=15, n_iter=50):
    """Time a no-arg thunk that launches GPU work. Returns seconds/call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter
