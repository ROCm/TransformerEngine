# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""The seam is an alias, not a copy (plan P2): identity holds for the module and everything on
it; the rename's one visible effect (__module__) is asserted rather than hidden; enums survive
pickling across the seam."""
import pickle
import sys

import pytest

from conftest import COMPILED, SEAM


def test_alias_identity(tex):
    assert sys.modules[SEAM] is sys.modules[COMPILED]
    assert tex is sys.modules[COMPILED]
    assert tex.__name__ == COMPILED, "the module object is the compiled extension; it keeps its own name"


def test_every_public_attribute_is_the_same_object(tex):
    comp = sys.modules[COMPILED]
    for n in dir(tex):
        if not n.startswith("__"):
            assert getattr(tex, n) is getattr(comp, n), n


def test_pybind_types_report_the_compiled_module(tex):
    """Documented effect of the rename: pybind classes/enums say where they live. Anything
    that pickles these by class path sees the compiled name (checkpoints do not - see
    _extra_state: no pybind objects are pickled)."""
    types = [getattr(tex, n) for n in dir(tex) if isinstance(getattr(tex, n), type)]
    assert types, "no pybind types found on the extension"
    wrong = [t.__name__ for t in types if getattr(t, "__module__", None) != COMPILED]
    assert not wrong, f"__module__ != {COMPILED}: {wrong}"


def test_enum_pickle_round_trip(tex):
    v = tex.DType.kFloat8E4M3
    back = pickle.loads(pickle.dumps(v))
    assert back == v and int(back) == int(v)
    assert type(back) is tex.DType


def test_seam_call_overhead_is_zero_by_identity(tex, upstream_demand):
    """thresholds.yaml: seam_call_overhead = 0, asserted by identity - every function upstream
    calls IS the compiled function; there is no wrapper to time."""
    comp = sys.modules[COMPILED]
    for f in upstream_demand["calls"]:
        if hasattr(tex, f):
            assert getattr(tex, f) is getattr(comp, f), f
