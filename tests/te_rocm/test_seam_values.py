# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""TE_ROCM_EXTENSION_API, enum VALUE level (plan F6 / ABI-002-FAENUM).

A name-only inventory reports NVTE_Fused_Attn_Backend as supplied while its ROCm member set
{AOTriton, CK, No_Backend} differs from upstream's {F16_max512_seqlen, F16_arbitrary_seqlen,
FP8, No_Backend}. This test compares, for every enum upstream references, the MEMBERS upstream
references against the members the built extension actually has. Known gaps live in
enum_expected_diff.yaml so the test is green with the gap documented, and red the moment the
gap changes in either direction.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).parent


def enum_members_of(obj) -> set[str]:
    m = getattr(obj, "__members__", None)
    if m is not None:
        return set(m.keys())
    return {n for n in dir(obj) if not n.startswith("_") and n not in ("name", "value")}


def test_enum_members_match_upstream_except_documented(tex, upstream_demand):
    expected = yaml.safe_load((HERE / "enum_expected_diff.yaml").read_text()) or {}
    problems = []
    for enum_name, demanded in sorted(upstream_demand["enum_members"].items()):
        obj = getattr(tex, enum_name, None)
        if obj is None:
            problems.append(f"{enum_name}: enum not on extension"); continue
        have = enum_members_of(obj)
        # members upstream names that are not real members of upstream's enum are attribute
        # accesses on the enum type (e.g. .value); ignore names that are not UPPER/Enum-like
        missing = {m for m in demanded - have if m[:1].isupper() or m.startswith("NVTE") or m.startswith("k")}
        documented = set((expected.get(enum_name) or {}).get("missing_on_rocm", []))
        if missing != documented:
            problems.append(f"{enum_name}: missing on ROCm {sorted(missing)} != documented {sorted(documented)}")
    assert not problems, "\n".join(problems)


def test_documented_diff_is_still_true(tex):
    """The expected-diff file must not go stale: every documented missing member must really be
    missing, so that when HDR-B2 lands upstream and the enums converge, this test says so."""
    expected = yaml.safe_load((HERE / "enum_expected_diff.yaml").read_text()) or {}
    for enum_name, spec in expected.items():
        obj = getattr(tex, enum_name)
        have = enum_members_of(obj)
        stale = set(spec.get("missing_on_rocm", [])) & have
        assert not stale, f"{enum_name}: {sorted(stale)} documented as missing but now present - update enum_expected_diff.yaml"
        for m in spec.get("present_on_rocm_only", []):
            assert m in have, f"{enum_name}.{m} documented as ROCm-only but absent"


def test_enum_values_are_ints_and_stable(tex):
    """Enum values cross the seam as ints in places (DType in TE_DType maps); pin the ones the
    Python layer depends on so a re-numbering in the backend is caught here, not in a checkpoint."""
    d = tex.DType
    assert int(d.kByte) == 0 and int(d.kFloat32) == 4 and int(d.kFloat8E4M3) == 7, \
        "DType numbering changed - transformer_engine.pytorch.constants.TE_DType and pickled recipes depend on it"
