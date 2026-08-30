# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""TE_ROCM_EXTENSION_API, signature level (coarse): every upstream call site of a seam function
fits at least one pybind overload of the built extension - positional count within the
parameter count, and every keyword name a real parameter. pybind11 docstrings carry the
signatures; overloads are listed "1. f(...)", "2. f(...)". Exact type checking is Stage 3."""
import re

import pytest

SIG_RE = re.compile(r"^\s*(?:\d+\.\s*)?(\w+)\((.*)\)\s*->", re.M)


def overload_params(fn) -> list[list[str]]:
    doc = getattr(fn, "__doc__", "") or ""
    out = []
    for m in SIG_RE.finditer(doc):
        params = []
        depth = 0; cur = ""
        for ch in m.group(2):
            if ch in "[(": depth += 1
            if ch in "])": depth -= 1
            if ch == "," and depth == 0:
                params.append(cur); cur = ""
            else:
                cur += ch
        if cur.strip(): params.append(cur)
        names = [p.split(":")[0].strip().lstrip("*") for p in params if p.strip() and p.strip() not in ("*", "/")]
        out.append(names)
    return out


def is_placeholder(fn) -> bool:
    return "Dummy function" in (getattr(fn, "__doc__", "") or "")


def test_placeholders_are_documented(tex):
    """Every ROCm placeholder registration ('Dummy function for python side annotations') must
    be listed in signature_expected_diff.yaml, and every listed one must still be a placeholder."""
    import yaml
    from pathlib import Path
    doc = yaml.safe_load((Path(__file__).parent / "signature_expected_diff.yaml").read_text()) or {}
    listed = set((doc.get("placeholders") or {}).keys())
    actual = {n for n in dir(tex) if not n.startswith("_") and callable(getattr(tex, n)) and is_placeholder(getattr(tex, n))}
    assert actual == listed, f"placeholders drifted: new={sorted(actual - listed)} gone={sorted(listed - actual)}"


def test_call_sites_fit_a_pybind_overload(tex, upstream_demand):
    problems = []; checked = 0
    for fname, sites in sorted(upstream_demand["calls"].items()):
        fn = getattr(tex, fname, None)
        if fn is None or not callable(fn):
            continue
        if is_placeholder(fn):
            continue   # documented in signature_expected_diff.yaml (test_placeholders_are_documented)
        overloads = overload_params(fn)
        if not overloads:
            continue   # no parsable pybind signature (python-level or submodule); Stage 3
        for npos, kwargs in sites:
            if npos < 0:
                continue   # *args / **kwargs call - shape unknown statically
            ok = any(npos <= len(p) and kwargs <= set(p) for p in overloads)
            checked += 1
            if not ok:
                problems.append(f"{fname}: call with {npos} positional + kwargs {sorted(kwargs)} fits no overload {overloads}")
    assert checked > 50, f"signature check exercised only {checked} call sites - parser regressed?"
    assert not problems, "\n".join(problems[:30])


def test_demanded_functions_are_callables(tex, upstream_demand):
    """Names upstream CALLS must be callables on the extension (a class or function), not data."""
    bad = [f for f in upstream_demand["calls"] if hasattr(tex, f) and not callable(getattr(tex, f))]
    assert not bad, f"called upstream but not callable on the extension: {bad}"
