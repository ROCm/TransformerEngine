# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Opt-in on-the-fly autotuning for bf16 grouped GEMM (forward + backward).

When ``NVTE_AUTOTUNE_KERNELS=1`` (ROCm only), each bf16 grouped GEMM issued through
``general_grouped_gemm`` -- forward (TN), dgrad (NN), and wgrad (NT) -- picks
between the two C++ backends that share that entry point and semantics
(multi-stream hipBLASLt and CK) by measuring them once per shape+layout (via
``triton.testing.do_bench``) and caching the winner. The pure selection lives in
:mod:`kernel_router`; this module supplies the backend candidates and the
process-global router.

The env vars are intentionally op-agnostic (``NVTE_AUTOTUNE_KERNELS`` /
``NVTE_AUTOTUNE_KERNELS_VERBOSE``) so the same switches govern future autotuned ops.

Scope / safety:

* Forward, dgrad, and wgrad grouped GEMMs are routed; each layout is a distinct
  route key, measured independently.
* Only the two backends reachable through ``general_grouped_gemm`` are
  candidates. The Triton grouped GEMM is excluded (different input layout and
  backward) and stays behind its ``NVTE_USE_GROUPED_GEMM_TRITON`` flag.
* Timing is side-effect-free: candidates are measured into a scratch clone of the
  output with ``accumulate=False``, so re-running them under ``do_bench`` can
  never corrupt a fused ``main_grad`` (wgrad accumulation) or any real output.
  Only the chosen winner is then run once into the real output with the real
  arguments.
* The deferred-wgrad path (``wgrad_store``) is left untouched.
* CK requires ``num_gemms > 1`` and bf16/fp16; otherwise it is unavailable and
  the router falls back to multi-stream hipBLASLt (the guaranteed floor).
* Off by default; ``NVTE_AUTOTUNE_KERNELS_VERBOSE=1`` adds per-call selection logging
  (cache hit/miss, route key, per-backend timings, winner).

"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from .cpp_extensions import general_grouped_gemm
from .kernel_router import AutotuneRouter, RouteKey, make_route_key

_FLOAT16_KEYS = ("torch.bfloat16", "torch.float16")

# Op-agnostic autotune switches, shared by any future autotuned op.
_MASTER_ENV = "NVTE_AUTOTUNE_KERNELS"
_VERBOSE_ENV = "NVTE_AUTOTUNE_KERNELS_VERBOSE"


def _autotune_enabled() -> bool:
    return IS_HIP_EXTENSION and os.getenv(_MASTER_ENV, "0") == "1"


def _verbose() -> bool:
    return os.getenv(_VERBOSE_ENV, "0") == "1"


_last_log: str | None = None


def _log_selection(sel) -> None:
    """Print one line per call: cache hit/miss, the route key, what was tried
    (timings or skip reason), and the winner. Consecutive identical lines are
    collapsed so a steady state (repeated cache hits for one shape) logs once."""
    global _last_log
    key = sel.key
    ks = (
        f"G={key.num_groups} N={key.N} K={key.K} dtype={key.dtype} "
        f"layout={key.layout} size_bin={key.size_bin}"
    )
    if sel.from_cache:
        line = f"[gg-autotune] cache HIT  [{ks}] -> {sel.winner}"
    else:
        tried = []
        for r in sel.reports:
            if not r.available:
                tried.append(f"{r.name}=unavailable")
            elif r.error:
                tried.append(f"{r.name}=ERROR({r.error})")
            elif r.time_ms is None:
                tried.append(f"{r.name}=rejected")
            else:
                tried.append(f"{r.name}={r.time_ms:.4f}ms")
        line = (
            f"[gg-autotune] cache MISS [{ks}] tried: {', '.join(tried)} "
            f"-> selected {sel.winner}"
        )
    if line != _last_log:
        print(line, flush=True)
        _last_log = line


@contextlib.contextmanager
def _env(**overrides):
    """Temporarily set/unset env vars; None removes. The C++ grouped-GEMM
    dispatch reads these per call, so each invocation sets them transiently --
    backward (which does not enter this path) is never affected."""
    prev = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@dataclass
class _GGCall:
    """Everything needed to (re)issue one ``general_grouped_gemm`` call, for any
    of the three grouped GEMMs (fprop TN, dgrad NN, wgrad NT). ``out`` is always
    the list ``general_grouped_gemm`` expects; ``N``/``K`` are the weight dims,
    passed by the caller so the route key is stable per layer across layouts."""

    A: list
    B: list
    out: list
    quantization_params: list
    out_dtype: object
    m_splits: list
    layout: str
    N: int
    K: int
    gemm_kwargs: dict

    @property
    def num_groups(self) -> int:
        return len(self.m_splits)


class _GroupedGemmBackend:
    """A grouped-GEMM backend reached through ``general_grouped_gemm``, selected
    by transiently toggling the CK env vars."""

    def __init__(self, name: str, use_ck: bool):
        self.name = name
        self._use_ck = use_ck

    def available(self, key: RouteKey) -> bool:
        if self._use_ck:
            return key.num_groups > 1 and key.dtype in _FLOAT16_KEYS
        return True  # multi-stream hipBLASLt is the guaranteed-available floor

    def _env_overrides(self):
        if self._use_ck:
            return {"NVTE_USE_CUTLASS_GROUPED_GEMM": "1", "NVTE_USE_CK_GROUPED_GEMM": "1"}
        return {"NVTE_USE_CUTLASS_GROUPED_GEMM": None, "NVTE_USE_CK_GROUPED_GEMM": None}

    def prepare(self, call: _GGCall):
        # Side-effect-free timing: measure into a scratch clone of the output with
        # accumulate off, so repeated do_bench runs never touch the real output or
        # a fused main_grad. Only the winner is run for real via run_real().
        overrides = self._env_overrides()
        scratch = [torch.empty_like(o) for o in call.out]
        timing_kwargs = dict(call.gemm_kwargs)
        timing_kwargs["accumulate"] = False

        def run():
            with _env(**overrides):
                general_grouped_gemm(
                    call.A, call.B, scratch, call.quantization_params, call.out_dtype,
                    m_splits=list(call.m_splits), layout=call.layout, **timing_kwargs,
                )

        return run

    def run_real(self, call: _GGCall):
        with _env(**self._env_overrides()):
            return general_grouped_gemm(
                call.A, call.B, call.out, call.quantization_params, call.out_dtype,
                m_splits=list(call.m_splits), layout=call.layout, **call.gemm_kwargs,
            )


_router: AutotuneRouter | None = None
_backends: dict = {}


def _do_bench_ms(fn) -> float:
    import triton

    try:
        return triton.testing.do_bench(fn, warmup=100, rep=100, return_mode="median")
    except TypeError:
        med, _, _ = triton.testing.do_bench(fn, warmup=100, rep=100, quantiles=[0.5, 0.2, 0.8])
        return med


def _get_router() -> AutotuneRouter:
    global _router, _backends
    if _router is None:
        candidates = [
            _GroupedGemmBackend("hipblaslt", use_ck=False),
            _GroupedGemmBackend("ck", use_ck=True),
        ]
        _backends = {c.name: c for c in candidates}
        # verifier=None: both backends are already validated by TE's grouped-GEMM
        # tests, so no per-call numerics gate (which would need an fp32 reference).
        _router = AutotuneRouter(
            candidates=candidates,
            timer=_do_bench_ms,
            verifier=None,
            default="hipblaslt",
        )
    return _router


_FLOAT16_DTYPES = (torch.bfloat16, torch.float16)


def _eligible(quantization_params, out_dtype) -> bool:
    """True on the autotune-enabled bf16/fp16 C++ path with no quantizers (which
    excludes fp8 and debug, whose quantizers are non-None)."""
    return (
        _autotune_enabled()
        and out_dtype in _FLOAT16_DTYPES
        and all(q is None for q in quantization_params)
    )


def _key_dims(A, out, layout):
    """The weight dims (N, K) for the route key, from the layout-appropriate
    operand: wgrad (NT) writes them into ``out``; fprop/dgrad read them off the
    weight operand ``A``."""
    t = out[0] if layout == "NT" else A[0]
    return t.size(0), t.size(1)


def autotuned_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs):
    """Drop-in for :func:`general_grouped_gemm`.

    When ``NVTE_AUTOTUNE_KERNELS=1`` and the call is on the bf16/fp16 C++ path, it
    selects the fastest backend (multi-stream hipBLASLt vs CK) for this
    shape+layout and runs it. Otherwise -- disabled, CUDA, fp8/debug, or fp32 --
    it delegates to ``general_grouped_gemm`` unchanged, returning its result.
    """
    if not _eligible(quantization_params, out_dtype):
        return general_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs)

    m_splits = kwargs.get("m_splits")
    if m_splits is None:
        return general_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs)

    layout = kwargs.get("layout", "TN")
    N, K = _key_dims(A, out, layout)
    # layout/m_splits are tracked on the call; keep them out of the forwarded kwargs.
    gemm_kwargs = {k: v for k, v in kwargs.items() if k not in ("layout", "m_splits")}
    call = _GGCall(
        A=A,
        B=B,
        out=out,
        quantization_params=quantization_params,
        out_dtype=out_dtype,
        m_splits=list(m_splits),
        layout=layout,
        N=N,
        K=K,
        gemm_kwargs=gemm_kwargs,
    )
    key = make_route_key(len(m_splits), tuple(m_splits), N, K, str(out_dtype), layout)
    sel = _get_router().select(key, call)
    if _verbose():
        _log_selection(sel)
    # select() measures into scratch on a miss and returns only a name on a hit;
    # run the chosen backend once into the real `out` to produce the result.
    return _backends[sel.winner].run_real(call)
