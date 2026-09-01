#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Retired-residue invariant (found the hard way, 2026-09-01).

A P5-retired patch means: the overlay drops that divergence, the fork tree keeps it (pending
e.g. the MI300 decision). ACCEPTABLE only while the fork file's divergence equals exactly the
retired patch - the moment such a file gains NEW divergence, that divergence silently never
reaches the overlay (float8_tensor.py + Stage-6 registry sites was the instance).

Invariant checked here: for every divergent non-C++ file under transformer_engine/ without an
active patch and not fork-only, applying its retired patch to the pinned upstream file must
reproduce the fork file byte-for-byte. Exit 1 otherwise, naming the file and the fix
(revive the patch, or regenerate the retired copy after an intentional residue change).
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROP = Path(__file__).resolve().parents[1]
ROOT = PROP.parent.parent
SUB = "3rdparty/transformer_engine_nvidia"
CXX = re.compile(r"\.(cu|cpp|h|cuh|hpp|c|hip)$")
# build-system / packaging files are governed by BS-001 and S3.4 packaging, not the Python queue
# (same exclusion set as the assembler's is_cxx CXX_NAMES + data files)
NON_QUEUE = {"CMakeLists.txt", "MANIFEST.in", "pyproject.toml"}
NON_QUEUE_SUFFIX = {".version", ".txt", ".toml", ".in", ".json", ".cfg"}


def sh(*a, cwd=ROOT):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd)


def targets(d: Path) -> dict[str, Path]:
    out = {}
    for p in d.glob("*.patch"):
        for l in p.read_text().splitlines():
            if l.startswith("--- a/"):
                out[l[6:]] = p
                break
    return out


def main() -> int:
    active = targets(PROP / "patches")
    retired = targets(PROP / "patches" / "retired")
    up_files = set(sh("git", "ls-tree", "-r", "--name-only", "HEAD", "transformer_engine",
                      cwd=ROOT / SUB).stdout.splitlines())
    bad = []
    for rel in sorted(up_files):
        name = rel.rsplit("/", 1)[-1]
        if CXX.search(rel) or rel in active or name in NON_QUEUE \
                or ("." in name and "." + name.rsplit(".", 1)[-1] in NON_QUEUE_SUFFIX):
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        up = sh("git", "show", f"HEAD:{rel}", cwd=ROOT / SUB).stdout
        fork = p.read_text(errors="replace")
        if up == fork:
            continue
        rp = retired.get(rel)
        if rp is None:
            bad.append((rel, "divergent with NO patch, active or retired"))
            continue
        body = "\n".join(l for l in rp.read_text().splitlines() if not l.startswith("#")) + "\n"
        with tempfile.TemporaryDirectory() as td:
            tgt = Path(td) / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(up)
            (Path(td) / "p.patch").write_text(body)
            ap = subprocess.run(["git", "apply", "--whitespace=nowarn", "p.patch"],
                                capture_output=True, cwd=td)
            if ap.returncode or tgt.read_text() != fork:
                bad.append((rel, f"residue drifted from retired {rp.stem} - revive it or "
                                 f"regenerate the retired copy"))
    for rel, why in bad:
        print(f"  RESIDUE-DRIFT {rel}: {why}")
    print(f"retired-residue check: {len(bad)} violations")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
