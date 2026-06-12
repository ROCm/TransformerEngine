# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Torch launchers that drive the ACTUAL Triton kernels in the source file
``transformer_engine/jax/triton_extensions/indexer.py``.

The profiling scripts can import their launchers from here instead of from the
self-contained ``indexer_kernels`` copy, so the benchmarks exercise the real
source kernels (forward ``_score_reduce_kernel``, backward
``_score_dscores_chunk_kernel``, top-k ``_score_topk_kernel`` /
``_score_topk_single_kernel``).

Loading trick: importing ``transformer_engine.jax.triton_extensions.indexer``
the normal way runs ``transformer_engine/jax/__init__.py``, which loads the TE
core C library -- broken in a namespace-package / JAX-less setup. The Triton
kernels need none of that, so we ``importlib``-load ``indexer.py`` directly,
pre-stubbing the parent packages (and the ``.utils`` relative import, which is
only used by the JAX lowerings, never by a direct kernel launch). ``jax`` itself
must be importable (it is -- the kernels' module defines JAX primitives at load
time), but no JAX device/compute is used here.
"""

import importlib.util
import os
import sys
import types

import torch
import triton


def _load_origin_indexer():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    idx_path = os.path.join(
        repo_root, "transformer_engine", "jax", "triton_extensions", "indexer.py"
    )
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"origin indexer.py not found at {idx_path}")

    # Stub parent packages so the relative `from .utils import ...` resolves
    # WITHOUT running the real transformer_engine.jax.__init__ (broken core lib).
    for name in (
        "transformer_engine",
        "transformer_engine.jax",
        "transformer_engine.jax.triton_extensions",
    ):
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            m = types.ModuleType(name)
            m.__path__ = []  # mark as a package
            sys.modules[name] = m

    util_name = "transformer_engine.jax.triton_extensions.utils"
    if util_name not in sys.modules:
        u = types.ModuleType(util_name)
        # Only the JAX lowerings call this; direct kernel launches don't.
        u.triton_call_lowering = lambda *a, **k: None
        sys.modules[util_name] = u

    mod_name = "transformer_engine.jax.triton_extensions.indexer"
    spec = importlib.util.spec_from_file_location(mod_name, idx_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_src = _load_origin_indexer()

# Re-export the source objects the launchers / profilers reference.
_score_reduce_kernel = _src._score_reduce_kernel                # triton Autotuner
_score_dscores_chunk_kernel = _src._score_dscores_chunk_kernel
_score_topk_kernel = _src._score_topk_kernel
_score_topk_single_kernel = _src._score_topk_single_kernel
_next_pow2 = _src._next_pow2
_HBWD_BLOCK_T = _src._HBWD_BLOCK_T
_HBWD_BLOCK_S = _src._HBWD_BLOCK_S
_BWD_H_CHUNK = _src._BWD_H_CHUNK
_SINGLE_SORT_MAX = _src._SINGLE_SORT_MAX
_SCORE_TOPK_CONFIGS = _src._SCORE_TOPK_CONFIGS
_SINGLE_TOPK_CONFIGS = _src._SINGLE_TOPK_CONFIGS

ORIGIN_FILE = _src.__file__


# --- Forward -----------------------------------------------------------------

def score_reduce(Hq, Hk, W_o, out_dtype=None):
    assert Hq.ndim == 5 and Hk.ndim == 4 and W_o.ndim == 4
    B, oH, T_t, H, d_i = Hq.shape
    T_s = Hk.shape[2]
    if out_dtype is None:
        out_dtype = Hq.dtype
    Hq, Hk, W_o = Hq.contiguous(), Hk.contiguous(), W_o.contiguous()
    O = torch.empty((B, oH, T_t, T_s), dtype=out_dtype, device=Hq.device)

    def grid(meta):
        return (triton.cdiv(T_s, meta["BLOCK_S"]),
                triton.cdiv(T_t, meta["BLOCK_T"]), B * oH)

    _score_reduce_kernel[grid](
        Hq, Hk, W_o, O, B=B, oH=oH, T_t=T_t, T_s=T_s, H=H, d_i=d_i,
    )
    return O


# --- Backward ----------------------------------------------------------------

def bwd_h_chunk(H):
    """H_CHUNK selection identical to the source ``_score_reduce_bwd``."""
    if H % _BWD_H_CHUNK == 0:
        return _BWD_H_CHUNK
    for c in (4, 2):
        if H % c == 0:
            return c
    return 1


def score_dscores_chunk(Hq_chunk, Hk, W_o_chunk, dO):
    B, oH, T, H_CHUNK, d_i = Hq_chunk.shape
    T_s = dO.shape[-1]

    Hq_chunk = Hq_chunk.contiguous()
    Hk = Hk.contiguous()
    W_o_chunk = W_o_chunk.contiguous()
    dO = dO.contiguous()

    dscores_chunk = torch.empty(
        (B, oH, T, H_CHUNK, T_s), dtype=Hq_chunk.dtype, device=Hq_chunk.device)
    dWo_chunk = torch.empty(
        (B, oH, T, H_CHUNK), dtype=Hq_chunk.dtype, device=Hq_chunk.device)

    def grid(meta):
        return ((T + meta["BLOCK_T"] - 1) // meta["BLOCK_T"], B * oH)

    _score_dscores_chunk_kernel[grid](
        Hq_chunk, Hk, W_o_chunk, dO, dscores_chunk, dWo_chunk,
        B=B, oH=oH, T=T, T_s=T_s, H_CHUNK=H_CHUNK, d_i=d_i,
    )
    return dscores_chunk, dWo_chunk


# --- Top-k -------------------------------------------------------------------

def score_topk(Hq, Hk, W_o, k):
    B, oH, T_t, H, d_i = Hq.shape
    T_s = Hk.shape[2]
    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError(f"k must be a positive power of 2; got {k}")
    if k > T_s:
        raise ValueError(f"k={k} must be <= T_s={T_s}")
    S_PAD = _next_pow2(T_s)
    Hq, Hk, W_o = Hq.contiguous(), Hk.contiguous(), W_o.contiguous()
    out = torch.empty((B, oH, T_t, k), dtype=torch.int32, device=Hq.device)

    def grid(meta):
        return (triton.cdiv(T_t, meta["BLOCK_T"]), B * oH)

    kernel = (_score_topk_single_kernel if S_PAD <= _SINGLE_SORT_MAX
              else _score_topk_kernel)
    kernel[grid](
        Hq, Hk, W_o, out,
        B=B, oH=oH, T_t=T_t, T_s=T_s, H=H, d_i=d_i, K=k, S_PAD=S_PAD,
    )
    return out
