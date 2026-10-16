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
    """The GPU-free bundle a selection is keyed on. No live tensors.

    Structural fields (``num_groups``, ``N``, ``K``, ``out_dtype``, ``layout``,
    ``in_format``) are exact and stable per layer -- ``in_format`` is the input
    number format (bf16/fp8/mxfp8), which the output element type (``out_dtype``)
    alone cannot convey. The token count enters only as a coarse
    ``size_bin`` (small/large, see :func:`make_route_key`): the exact per-step
    token count jitters with dynamic routing, so a fine key would thrash the cache
    and re-measure constantly. A single 2-bin split keeps the cache stable while
    still capturing the one regime where the fastest backend flips with size.
    """

    num_groups: int
    N: int
    K: int
    out_dtype: str
    layout: str
    size_bin: int
    in_format: str


# Coarse small/large split on the average per-group token count. A tunable
# heuristic, not a hard boundary: the router still *measures* both backends per
# bin, so the threshold only needs to sit near the size where the winner changes.
_LARGE_TOKENS_PER_GROUP = 2048


def _size_bin(num_groups: int, m_splits) -> int:
    avg_tokens = sum(m_splits) // max(num_groups, 1)
    return 0 if avg_tokens < _LARGE_TOKENS_PER_GROUP else 1


def make_route_key(num_groups, m_splits, N, K, out_dtype, layout, in_format) -> RouteKey:
    """Derive a coarse, reuse-friendly RouteKey from the raw call parameters. Pure."""
    return RouteKey(
        num_groups=num_groups,
        N=N,
        K=K,
        out_dtype=out_dtype,
        layout=layout,
        size_bin=_size_bin(num_groups, m_splits),
        in_format=in_format,
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
