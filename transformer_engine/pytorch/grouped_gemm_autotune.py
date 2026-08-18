# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Opt-in on-the-fly autotuning for grouped GEMM (bf16/fp8/mxfp8, fwd + bwd).

When ``NVTE_AUTOTUNE_KERNELS=1`` (ROCm only), each grouped GEMM issued through
``general_grouped_gemm`` -- forward (TN), dgrad (NN), and wgrad (NT) -- picks the
fastest backend for its shape+layout+in_format by measuring the candidates once (via
``triton.testing.do_bench``) and caching the winner. The candidates are the two
C++ backends that share the entry point and semantics (multi-stream hipBLASLt and
CK), HipKittens (mxfp8 only), plus the Triton grouped GEMM (bf16 only). The pure
selection lives in :mod:`kernel_router`; this module supplies the backend
candidates and the process-global router.

The env vars are intentionally op-agnostic (``NVTE_AUTOTUNE_KERNELS`` /
``NVTE_AUTOTUNE_KERNELS_VERBOSE``) so the same switches govern future autotuned ops.

Scope / safety:

* Forward, dgrad, and wgrad grouped GEMMs are routed; each layout is a distinct
  route key, measured independently.
* The C++ backends are selected by transiently toggling env vars: hipBLASLt
  (multi-stream, all formats, the guaranteed floor), CK (bf16/fp8, num_gemms > 1),
  and, for mxfp8 only, a single cutlass-family candidate that prefers HipKittens
  (``hipkittens``). HipKittens needs 256-aligned expert dims; when they are not,
  the C++ path falls back to CK internally, so that one candidate covers
  HK-when-aligned and CK-otherwise. It is deliberately the *only* mxfp8
  cutlass-family candidate: the C++ HK-enable flag is a process-lifetime static
  frozen on first use, so a second competing env config could not switch backends
  at runtime and might pin the wrong one. Triton is a candidate for bf16 only. The
  separate global ``NVTE_USE_GROUPED_GEMM_TRITON`` flag (which forces Triton for all
  layouts) is independent of and mutually exclusive with this path.
* Only the *input* is quantized (fp8/mxfp8); the GEMM output is bf16 with no
  output quantizer, so repeated measurement never mutates quantizer amax state.
  Output-quantized calls (debug, fp8 output) are excluded and delegated.
* The Triton candidate runs TN/NN directly; NT (wgrad) runs into a fresh 3D packed
  buffer (as the ``use_grouped_gemm_triton`` path allocates) and copies the result
  back into the per-group output list.
* Timing is side-effect-free: candidates are measured into a scratch clone of the
  output with ``accumulate=False``, so re-running them under ``do_bench`` can
  never corrupt a fused ``main_grad`` (wgrad accumulation) or any real output.
  Only the chosen winner is then run once into the real output with the real
  arguments.
* The deferred-wgrad path (``wgrad_store``) is left untouched.
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

# Op-agnostic autotune switches, shared by any future autotuned op.
_MASTER_ENV = "NVTE_AUTOTUNE_KERNELS"
_VERBOSE_ENV = "NVTE_AUTOTUNE_KERNELS_VERBOSE"

# Warm-up iterations before timing, to force JIT/autotune compilation (Triton)
# out of the measured region.
_WARMUP_ITERS = 3


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
        f"G={key.num_groups} N={key.N} K={key.K} in_format={key.in_format} "
        f"layout={key.layout} size_bin={key.size_bin}"
    )
    if sel.from_cache:
        return
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


_ALL_FORMATS = ("bf16", "fp8", "mxfp8")


class _GroupedGemmBackend:
    """A grouped-GEMM backend reached through ``general_grouped_gemm``, selected by
    transiently toggling env vars. Availability is format-gated; ``needs_multi``
    requires num_groups > 1 (the C++ path only takes the grouped fast path with more
    than one group)."""

    def __init__(self, name, env, *, formats, needs_multi=False):
        self.name = name
        self._env = dict(env)
        self._formats = frozenset(formats)
        self._needs_multi = needs_multi

    def available(self, key: RouteKey) -> bool:
        if key.in_format not in self._formats:
            return False
        if self._needs_multi and key.num_groups <= 1:
            return False
        return True

    def prepare(self, call: _GGCall):
        # Side-effect-free timing: measure into a scratch clone of the output with
        # accumulate off, so repeated do_bench runs never touch the real output or
        # a fused main_grad. Only the winner is run for real via run_real().
        env = self._env
        scratch = [torch.empty_like(o) for o in call.out]
        timing_kwargs = dict(call.gemm_kwargs)
        timing_kwargs["accumulate"] = False

        def run():
            with _env(**env):
                general_grouped_gemm(
                    call.A, call.B, scratch, call.quantization_params, call.out_dtype,
                    m_splits=list(call.m_splits), layout=call.layout, **timing_kwargs,
                )

        return run

    def run_real(self, call: _GGCall):
        with _env(**self._env):
            return general_grouped_gemm(
                call.A, call.B, call.out, call.quantization_params, call.out_dtype,
                m_splits=list(call.m_splits), layout=call.layout, **call.gemm_kwargs,
            )


def _triton_grouped_gemm():
    """Lazily import the Triton grouped GEMM; return None if unavailable."""
    try:
        from transformer_engine.pytorch.triton_kernels.grouped_gemm import (
            general_grouped_gemm_triton,
        )

        return general_grouped_gemm_triton
    except Exception:
        return None


class _TritonGroupedGemmBackend:
    """Triton (AITER) grouped GEMM as an autotune candidate.

    Forward (TN) and dgrad (NN) call Triton directly -- it concatenates the
    per-group inputs internally (a real cost in this pre-split path, left inside
    the timing). wgrad (NT) is supported partially: Triton wgrad writes a 3D
    packed output (like the ``use_grouped_gemm_triton`` path allocates), so it is
    run into a fresh 3D buffer and the result copied/accumulated back into the
    per-group output list this path uses. An error on any shape drops it from the
    ranking.
    """

    name = "triton"

    def available(self, key: RouteKey) -> bool:
        return (
            key.in_format == "bf16"
            and key.layout in ("TN", "NN", "NT")
            and _triton_grouped_gemm() is not None
        )

    def _run_into(self, fn, call: _GGCall, out_target):
        # TN/NN: Triton writes the per-group output list directly.
        if call.layout != "NT":
            return fn(
                call.A, call.B, out_target, call.quantization_params, call.out_dtype,
                m_splits=list(call.m_splits), layout=call.layout, **call.gemm_kwargs,
            )
        # NT (wgrad): Triton needs a 3D packed output (G, N, K). Run into a fresh
        # buffer with accumulate off, then copy (or accumulate) back into the
        # per-group targets -- so it works whether they are fresh buffers or fused
        # main_grads, and the copy-back cast bridges any dtype difference.
        num_gemms = len(out_target)
        n, k = out_target[0].shape
        out3d = torch.empty((num_gemms, n, k), dtype=call.out_dtype, device=out_target[0].device)
        kwargs = dict(call.gemm_kwargs)
        accumulate = kwargs.pop("accumulate", False)
        result = fn(
            call.A, call.B, out3d, call.quantization_params, call.out_dtype,
            m_splits=list(call.m_splits), layout="NT", accumulate=False, **kwargs,
        )
        for i, w in enumerate(out_target):
            w.add_(out3d[i]) if accumulate else w.copy_(out3d[i])
        return result

    def prepare(self, call: _GGCall):
        fn = _triton_grouped_gemm()
        assert fn is not None  # guaranteed by available()
        scratch = [torch.empty_like(o) for o in call.out]

        def run():
            self._run_into(fn, call, scratch)

        return run

    def run_real(self, call: _GGCall):
        fn = _triton_grouped_gemm()
        assert fn is not None  # guaranteed by available()
        return self._run_into(fn, call, call.out)


_router: AutotuneRouter | None = None
_backends: dict = {}


def _do_bench_ms(fn) -> float:
    import triton

    # Warm up so first-call JIT/autotune compilation (Triton) is excluded from the
    # timed region -- otherwise a cold compile (seconds) is charged to the first
    # measured shape and permanently mis-rates the backend. do_bench's own warmup
    # then runs on already-compiled kernels. Harmless for the JIT-free C++ backends.
    for _ in range(_WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    try:
        return triton.testing.do_bench(fn, warmup=100, rep=100, return_mode="median")
    except TypeError:
        med, _, _ = triton.testing.do_bench(fn, warmup=100, rep=100, quantiles=[0.5, 0.2, 0.8])
        return med


def _get_router() -> AutotuneRouter:
    global _router, _backends
    if _router is None:
        candidates = [
            _GroupedGemmBackend(
                "hipblaslt",
                {
                    "NVTE_USE_CUTLASS_GROUPED_GEMM": None,
                    "NVTE_USE_CK_GROUPED_GEMM": None,
                    "NVTE_USE_HIPKITTENS_GROUPED_GEMM": None,
                },
                formats=_ALL_FORMATS,
            ),
            _GroupedGemmBackend(
                "ck",
                {
                    "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
                    "NVTE_USE_CK_GROUPED_GEMM": "1",
                    "NVTE_USE_HIPKITTENS_GROUPED_GEMM": None,
                },
                formats=("bf16", "fp8"),
                needs_multi=True,
            ),
            # Sole cutlass-family candidate for mxfp8. HipKittens is preferred (env
            # HK=1) and the C++ path silently falls back to CK when the expert dims
            # are not 256-aligned, so this one candidate covers HK-when-aligned and
            # CK-otherwise. It must be the ONLY mxfp8 cutlass-family candidate: the
            # C++ HK-enable flag is a process-lifetime static frozen on first use, so
            # a second competing env config (e.g. a CK-forcing candidate) would not
            # actually switch backends and could permanently pin the wrong choice.
            _GroupedGemmBackend(
                "hipkittens",
                {
                    "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
                    "NVTE_USE_HIPKITTENS_GROUPED_GEMM": "1",
                    "NVTE_USE_CK_GROUPED_GEMM": None,
                },
                formats=("mxfp8",),
                needs_multi=True,
            ),
            _TritonGroupedGemmBackend(),
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

_quant_cache = None


def _quant_classes():
    """Lazily import the quantizer/tensor classes used to classify the input format."""
    global _quant_cache
    if _quant_cache is None:
        from .tensor import (
            Float8CurrentScalingQuantizer,
            Float8Quantizer,
            MXFP8Quantizer,
            QuantizedTensorStorage,
        )

        _quant_cache = (
            (Float8Quantizer, Float8CurrentScalingQuantizer),
            MXFP8Quantizer,
            QuantizedTensorStorage,
        )
    return _quant_cache


def _in_format(A, out_dtype):
    """Classify the grouped GEMM as ``bf16`` / ``fp8`` / ``mxfp8`` from the input
    operand, or ``None`` if unsupported (fp32, nvfp4, unknown). The GEMM output is
    always bf16/fp16; the input format captures the *input* precision so a bf16 and
    an fp8 GEMM of the same shape do not collide in the cache."""
    if out_dtype not in _FLOAT16_DTYPES:
        return None
    fp8_types, mxfp8_type, storage_type = _quant_classes()
    a0 = A[0]
    if not isinstance(a0, storage_type):
        return "bf16" if getattr(a0, "dtype", None) in _FLOAT16_DTYPES else None
    q = getattr(a0, "_quantizer", None)
    if isinstance(q, mxfp8_type):
        return "mxfp8"
    if isinstance(q, fp8_types):
        return "fp8"
    return None


def _eligible(A, quantization_params, out_dtype):
    """Return the input format (bf16/fp8/mxfp8) if this call should be autotuned,
    else None. Excludes disabled autotune, output-quantized calls (debug or fp8
    output, whose quantizer amax would be corrupted by repeated measurement), and
    unsupported input formats."""
    if not _autotune_enabled():
        return None
    if any(q is not None for q in quantization_params):
        return None
    return _in_format(A, out_dtype)


def _key_dims(A, out, layout):
    """The weight dims (N, K) for the route key, from the layout-appropriate
    operand: wgrad (NT) writes them into ``out``; fprop/dgrad read them off the
    weight operand ``A``. Uses ``.size()`` so it works for quantized inputs too
    (the ``*TensorStorage`` classes expose ``.size()`` but not ``.shape``)."""
    t = out[0] if layout == "NT" else A[0]
    return t.size(0), t.size(1)


def autotuned_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs):
    """Drop-in for :func:`general_grouped_gemm`.

    When ``NVTE_AUTOTUNE_KERNELS=1`` and the call is an autotunable grouped GEMM
    (bf16/fp8/mxfp8 inputs, unquantized output), it selects the fastest available
    backend for this shape+layout+in_format and runs it. Otherwise -- disabled,
    CUDA, debug, fp32, or an unsupported input format -- it delegates to
    ``general_grouped_gemm`` unchanged, returning its result.
    """
    in_format = _eligible(A, quantization_params, out_dtype)
    if in_format is None:
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
    key = make_route_key(len(m_splits), tuple(m_splits), N, K, str(out_dtype), layout, in_format)
    sel = _get_router().select(key, call)
    if _verbose():
        _log_selection(sel)
    # select() measures into scratch on a miss and returns only a name on a hit;
    # run the chosen backend once into the real `out` to produce the result.
    return _backends[sel.winner].run_real(call)
