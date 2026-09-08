# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Opt-in on-the-fly autotuning for dense GEMM (bf16 + mxfp8, forward + backward).

Second consumer of the pure :mod:`kernel_router` (after ``grouped_gemm_autotune``).
When ``NVTE_AUTOTUNE_KERNELS=1`` (ROCm only), each dense GEMM issued through
``general_gemm`` -- forward (TN), dgrad (NN), wgrad (NT) -- picks the fastest
backend for its shape+layout by measuring the candidates once and caching the
winner. Two regimes have a real choice: bf16 races the C++ default (hipBLASLt)
against the Triton kernel (``NVTE_USE_GEMM_TRITON=1``); mxfp8 races HipKittens (the
C++ default for mxfp8) against hipBLASLt (``NVTE_ROCM_USE_HIPBLASLT_MXFP8=1``). Both
toggles are read per call, so -- unlike the grouped mxfp8 path -- there is no
process-static to freeze the choice. fp8/fp32 have a single backend and are left
to ``general_gemm`` unchanged.

The op-agnostic switches (``NVTE_AUTOTUNE_KERNELS`` / ``NVTE_AUTOTUNE_KERNELS_VERBOSE``)
and the low-level runtime glue (env toggling, the do_bench timer, the input-format
classifier) are shared with ``grouped_gemm_autotune``.

Scope / safety:

* Only plain GEMMs are routed: calls with a comm-overlap communicator (``ub``) or
  an output quantizer are delegated to ``general_gemm`` unchanged -- the former
  because measuring would double the collective, the latter because repeated
  measurement would corrupt the output quantizer's amax. (mxfp8 quantizes the
  *inputs*; the output is bf16 with no quantizer, so mxfp8 is amax-safe.)
* Timing is side-effect-free: candidates are measured into a fresh output
  (``out=None``, ``accumulate=False``) so re-running them never touches the real
  output or a fused ``main_grad``. Only the winner is run once with the real args.
* For mxfp8, when HipKittens does not support a shape the C++ path falls back to
  hipBLASLt internally (per call, deterministic), so the ``hipkittens`` candidate
  degrades to a tie with ``hipblaslt`` rather than mis-measuring.
* Unlike grouped GEMM, dense token count (M = seq*mbs) is fixed within a run, so
  the key uses exact M/N/K -- no coarse token bin is needed.
* Off by default; ``NVTE_AUTOTUNE_KERNELS_VERBOSE=1`` adds per-call selection logging.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cpp_extensions import general_gemm
from .kernel_router import AutotuneRouter

# Shared, torch-dependent runtime glue (pending extraction into a common module).
from .grouped_gemm_autotune import (
    _FLOAT16_DTYPES,
    _autotune_enabled,
    _do_bench_ms,
    _env,
    _quant_classes,
    _verbose,
)

_GEMM_TRITON_ENV = "NVTE_USE_GEMM_TRITON"
_HIPBLASLT_MXFP8_ENV = "NVTE_ROCM_USE_HIPBLASLT_MXFP8"


@dataclass(frozen=True)
class GemmKey:
    """GPU-free key a dense-GEMM selection is cached on. M/N/K are exact (dense
    token count is stable within a run), so no coarse binning is needed."""

    m: int
    n: int
    k: int
    out_dtype: str
    layout: str
    in_format: str


def _in_format(operand, out_dtype):
    """Classify the GEMM's input precision as ``bf16`` / ``fp8`` / ``mxfp8`` from one
    operand, or ``None`` if unsupported. Mirrors the grouped classifier but takes a
    single tensor (dense operands are tensors, not lists)."""
    if out_dtype not in _FLOAT16_DTYPES:
        return None
    fp8_types, mxfp8_type, storage_type = _quant_classes()
    if not isinstance(operand, storage_type):
        return "bf16" if getattr(operand, "dtype", None) in _FLOAT16_DTYPES else None
    q = getattr(operand, "_quantizer", None)
    if isinstance(q, mxfp8_type):
        return "mxfp8"
    if isinstance(q, fp8_types):
        return "fp8"
    return None


def _mnk(A, B, layout):
    """Logical GEMM dims from the operands, per TE's column-major BLAS convention
    (see ``triton_kernels/gemm/gemm_wrapper.py``): ``m = A0 if transa else A1``,
    ``k = A1 if transa else A0``, ``n = B1 if transb else B0``."""
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    m = A.size(0) if transa else A.size(1)
    k = A.size(1) if transa else A.size(0)
    n = B.size(1) if transb else B.size(0)
    return m, n, k


_last_log: str | None = None


def _log_selection(sel) -> None:
    """One line per call: cache miss, the key, what was tried, and the winner.
    Consecutive identical lines are collapsed; cache hits are silent."""
    global _last_log
    if sel.from_cache:
        return
    key = sel.key
    ks = (
        f"M={key.m} N={key.n} K={key.k} in_format={key.in_format} "
        f"layout={key.layout} out_dtype={key.out_dtype}"
    )
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
    line = f"[te-autotune] cache MISS [{ks}] tried: {', '.join(tried)} -> selected {sel.winner}"
    if line != _last_log:
        print(line, flush=True)
        _last_log = line


@dataclass
class _GemmCall:
    """Everything needed to (re)issue one ``general_gemm`` call. ``kwargs`` is the
    full original keyword set (out, accumulate, layout, bias, ...)."""

    A: object
    B: object
    kwargs: dict


class _GemmBackend:
    """A dense-GEMM backend reached through ``general_gemm``, selected by transiently
    toggling ``NVTE_USE_GEMM_TRITON``."""

    def __init__(self, name, env, *, formats):
        self.name = name
        self._env = dict(env)
        self._formats = frozenset(formats)

    def available(self, key: GemmKey) -> bool:
        return key.in_format in self._formats

    def prepare(self, call: _GemmCall):
        # Side-effect-free timing: fresh output, accumulate off, no comm-overlap
        # extra output. Only the winner is run for real via run_real().
        env = self._env
        mkwargs = dict(call.kwargs)
        mkwargs["out"] = None
        mkwargs["accumulate"] = False
        mkwargs.pop("extra_output", None)

        def run():
            with _env(**env):
                general_gemm(call.A, call.B, **mkwargs)

        return run

    def run_real(self, call: _GemmCall):
        with _env(**self._env):
            return general_gemm(call.A, call.B, **call.kwargs)


_router: AutotuneRouter | None = None
_backends: dict = {}


def _get_router() -> AutotuneRouter:
    global _router, _backends
    if _router is None:
        # Each candidate sets both toggles explicitly so it selects the same backend
        # regardless of format: bf16 ignores the mxfp8 toggle and vice versa.
        candidates = [
            _GemmBackend(
                "hipblaslt",
                {_GEMM_TRITON_ENV: None, _HIPBLASLT_MXFP8_ENV: "1"},
                formats=("bf16", "mxfp8"),
            ),
            _GemmBackend(
                "triton",
                {_GEMM_TRITON_ENV: "1", _HIPBLASLT_MXFP8_ENV: None},
                formats=("bf16",),
            ),
            _GemmBackend(
                "hipkittens",
                {_GEMM_TRITON_ENV: None, _HIPBLASLT_MXFP8_ENV: None},
                formats=("mxfp8",),
            ),
        ]
        _backends = {c.name: c for c in candidates}
        _router = AutotuneRouter(
            candidates=candidates,
            timer=_do_bench_ms,
            verifier=None,
            default="hipblaslt",
        )
    return _router


def _eligible(A, B, kwargs):
    """Return the input format if this dense GEMM should be autotuned, else None.
    Excludes disabled autotune, comm-overlap (``ub``), and output-quantized calls.
    bf16 (hipBLASLt vs Triton) and mxfp8 (hipBLASLt vs HipKittens) each have two
    backends; other formats have one and are left to ``general_gemm``."""
    if not _autotune_enabled():
        return None
    if kwargs.get("ub") is not None:
        return None
    if kwargs.get("quantization_params") is not None:
        return None
    fmt = _in_format(A, kwargs.get("out_dtype"))
    return fmt if fmt in ("bf16", "mxfp8") else None


def autotuned_gemm(A, B, **kwargs):
    """Drop-in for :func:`general_gemm`.

    With ``NVTE_AUTOTUNE_KERNELS=1`` and an autotunable dense GEMM (bf16 or mxfp8
    inputs, no comm-overlap, unquantized output), selects the fastest backend
    (hipBLASLt / Triton for bf16, hipBLASLt / HipKittens for mxfp8) for this
    shape+layout and runs it. Otherwise delegates to ``general_gemm`` unchanged,
    returning its 4-tuple result.
    """
    fmt = _eligible(A, B, kwargs)
    if fmt is None:
        return general_gemm(A, B, **kwargs)

    layout = kwargs.get("layout", "TN")
    m, n, k = _mnk(A, B, layout)
    key = GemmKey(
        m=m, n=n, k=k, out_dtype=str(kwargs.get("out_dtype")), layout=layout, in_format=fmt
    )
    call = _GemmCall(A=A, B=B, kwargs=kwargs)
    sel = _get_router().select(key, call)
    if _verbose():
        _log_selection(sel)
    # select() measures into scratch on a miss and returns only a name on a hit;
    # run the chosen backend once with the real args to produce the result.
    return _backends[sel.winner].run_real(call)
