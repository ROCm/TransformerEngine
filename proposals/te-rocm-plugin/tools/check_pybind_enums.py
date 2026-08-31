#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""ABI-002 drift check (plugin plan S4.5).

The plan asked for pybind_helper.h enum lists to be *generated* from the backend's public
headers. Deliberate deviation: pybind_helper.h is an upstream-shared, hipified header - turning
it into a codegen consumer grows divergence in the most contested file class. This check gives
the same invariant (binding and header CANNOT drift apart unnoticed) with zero divergence:
parse the enum members per #ifdef branch in the public header, parse the .value() lists per
branch in pybind_helper.h, assert per-branch set equality. Runs in governance CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HDR = ROOT / "transformer_engine/common/include/transformer_engine/fused_attn.h"
PYB = ROOT / "transformer_engine/common/util/pybind_helper.h"


def branches(text: str, start_re: str, end_lit: str, member_re: str) -> dict[str, set[str]]:
    """Member sets per preprocessor branch for each region matching start_re..end_lit.
    Branch tracking: an ifndef on a ROCm macro (__HIP_PLATFORM_AMD__ in the public header,
    USE_ROCM in pybind_helper.h) opens the CUDA branch, ifdef opens the ROCm branch,
    #else flips, #endif clears."""
    out: dict[str, set[str]] = {}
    for m in re.finditer(start_re, text):
        state = None
        for pm in re.finditer(r"#\s*(ifndef|ifdef|else|endif)[^\n]*", text[: m.start()]):
            kind, line = pm.group(1), pm.group(0)
            if "__HIP_PLATFORM_AMD__" in line or "USE_ROCM" in line:
                state = "cuda" if kind == "ifndef" else "rocm"
            elif kind == "else" and state:
                state = "rocm" if state == "cuda" else "cuda"
            elif kind == "endif":
                state = None
        region = text[m.start(): text.index(end_lit, m.start())]
        members = set(re.findall(member_re, region))
        if members:
            out.setdefault(state or "unguarded", set()).update(members)
    return out


def main() -> int:
    hdr = branches(HDR.read_text(), r"enum NVTE_Fused_Attn_Backend \{", "};",
                   r"\b(NVTE_[A-Za-z0-9_]+)\s*=")
    pyb = branches(PYB.read_text(), r"enum_<NVTE_Fused_Attn_Backend>", ";",
                   r"\.value\(\"([A-Za-z0-9_]+)\"")
    rc = 0
    for branch in ("cuda", "rocm"):
        h, p = hdr.get(branch, set()), pyb.get(branch, set())
        if h != p:
            print(f"DRIFT [{branch}]: header {sorted(h)} vs pybind {sorted(p)}")
            rc = 1
        else:
            print(f"OK    [{branch}]: {sorted(h)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
