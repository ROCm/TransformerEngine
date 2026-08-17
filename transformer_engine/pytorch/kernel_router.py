# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Pure, GPU-free kernel router for per-shape backend selection.

This module holds no torch and no TE imports and does no I/O. It selects a
backend for a :class:`RouteKey` by measuring the available candidates once (via
an *injected* timer) and caching the winner in memory. The timer and the
(optional) numerics verifier are injected, so the selection logic is
unit-testable off-GPU and the router never depends on a specific backend or
timing method.

It is deliberately backend-agnostic: an op wires up its own candidates (each
exposing ``name``/``available``/``prepare``), a timer, and -- optionally -- a
numerics verifier, then asks :meth:`AutotuneRouter.select` for the winner. See
``grouped_gemm_autotune.py`` for the first consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class RouteKey:
    """The GPU-free bundle of parameters a selection is keyed on. No live tensors.

    Structural fields (``num_groups``, ``N``, ``K``, ``dtype``, ``layout``) are
    exact -- they are stable per layer. The token distribution is captured
    *coarsely* via ``total_m_bucket`` and ``imbalance_bucket`` (see
    :func:`make_route_key`) so nearby steps -- whose exact per-expert token counts
    differ every iteration -- share one cached decision instead of forcing a
    re-measure on every call. This trades a little routing precision for reuse;
    the bucket granularity is the knob for that trade.
    """

    num_groups: int
    N: int
    K: int
    dtype: str
    layout: str
    total_m_bucket: int
    imbalance_bucket: int


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _imbalance_bucket(m_splits) -> int:
    """Coefficient-of-variation of the per-group token counts, bucketed:
    0 = near-balanced, 1 = moderate skew, 2 = high skew. Multi-stream backends
    (one GEMM per expert) are the most sensitive to this, so it belongs in the key."""
    n = len(m_splits)
    if n <= 1:
        return 0
    mean = sum(m_splits) / n
    if mean == 0:
        return 0
    var = sum((m - mean) ** 2 for m in m_splits) / n
    cv = (var**0.5) / mean
    if cv < 0.1:
        return 0
    if cv < 0.5:
        return 1
    return 2


def make_route_key(num_groups, m_splits, N, K, dtype, layout) -> RouteKey:
    """Derive a coarse, reusable RouteKey from the raw call parameters. Pure."""
    return RouteKey(
        num_groups=num_groups,
        N=N,
        K=K,
        dtype=dtype,
        layout=layout,
        total_m_bucket=_next_pow2(sum(m_splits)),
        imbalance_bucket=_imbalance_bucket(m_splits),
    )


@runtime_checkable
class Candidate(Protocol):
    name: str

    def available(self, key: RouteKey) -> bool: ...

    def prepare(self, operands: Any) -> Callable[[], Any]:
        """Return a zero-arg closure that runs the backend and returns its output.
        Called once for the numerics gate (if any), then repeatedly for timing."""
        ...


# fn -> milliseconds (lower is better)
Timer = Callable[[Callable[[], Any]], float]
# (output, operands) -> (is_correct, sqnr_db)
Verifier = Callable[[Any, Any], "tuple[bool, float]"]


@dataclass
class CandidateReport:
    name: str
    available: bool
    correct: bool | None = None
    sqnr_db: float | None = None
    time_ms: float | None = None  # None when unavailable / incorrect / errored
    error: str | None = None


@dataclass
class Selection:
    key: RouteKey
    winner: str
    from_cache: bool
    reports: list[CandidateReport] = field(default_factory=list)


class AutotuneRouter:
    """Empirical per-shape selection, measured once and cached in memory.

    On a cache miss the router filters to the available candidates, optionally
    runs each once through the injected verifier (an incorrect candidate is
    dropped), times the survivors with the injected timer, and caches the
    fastest. With no correct/available candidate it falls back to ``default``
    (the guaranteed-available floor). A ``verifier`` of ``None`` skips the
    numerics gate (for use where the candidates are already validated).
    """

    def __init__(
        self,
        candidates: list[Candidate],
        timer: Timer,
        verifier: Verifier | None,
        default: str,
    ):
        names = [c.name for c in candidates]
        if default not in names:
            raise ValueError(f"default {default!r} not among candidates {names}")
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate candidate names: {names}")
        self.candidates = list(candidates)
        self.timer = timer
        self.verifier = verifier
        self.default = default
        self._cache: dict[RouteKey, str] = {}

    def cached(self, key: RouteKey) -> str | None:
        return self._cache.get(key)

    def select(self, key: RouteKey, operands: Any) -> Selection:
        hit = self._cache.get(key)
        if hit is not None:
            return Selection(key, hit, from_cache=True)
        reports = [self._measure(c, key, operands) for c in self.candidates]
        winner = self._pick(reports)
        self._cache[key] = winner
        return Selection(key, winner, from_cache=False, reports=reports)

    def _measure(self, cand: Candidate, key: RouteKey, operands: Any) -> CandidateReport:
        if not cand.available(key):
            return CandidateReport(cand.name, available=False)
        try:
            fn = cand.prepare(operands)
            correct, sqnr = None, None
            if self.verifier is not None:
                output = fn()  # one real run feeds the numerics gate
                correct, sqnr = self.verifier(output, operands)
                if not correct:
                    return CandidateReport(cand.name, True, correct=False, sqnr_db=sqnr)
            t = self.timer(fn)
            return CandidateReport(cand.name, True, correct=correct, sqnr_db=sqnr, time_ms=t)
        except Exception as exc:  # a backend that errors drops out; others still rank
            return CandidateReport(cand.name, True, error=repr(exc))

    def _pick(self, reports: list[CandidateReport]) -> str:
        ranked = [r for r in reports if r.time_ms is not None]
        if not ranked:
            return self.default
        return min(ranked, key=lambda r: r.time_ms).name
