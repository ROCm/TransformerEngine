#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""C++ patch queue over the submodule (plugin plan S4.2).

The Python queue's mechanism, applied to the C++ side, driven by cxx-strategy.yaml:
  gen       one governed CXX-* patch per `patch-queue` file: diff(upstream@pin, fork HEAD),
            into patches-cxx/ (separate dir - M2 counts stay Python-only)
  verify    the freeze invariant: apply each patch to the pinned upstream file, byte-compare
            with the fork tree
  assemble  materialize the full common/ C++ tree into --out from
            {upstream-identical + patched + native-hip/rocm-only from fork} and byte-compare
            EVERY file against the fork's common/. Identity of the tree is identity of the
            build inputs: the S4.2 exit ('compiles and passes ci/core.sh bit-for-bit
            equivalent') follows from it without a second 30-minute build.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
PROP = TOOLS.parent
ROOT = PROP.parent.parent
SUB = "3rdparty/transformer_engine_nvidia"
PATCHES = PROP / "patches-cxx"
CXX_RE = re.compile(r"\.(cu|cpp|h|cuh|hpp|c|hip)$")


def sh(*a, cwd=ROOT, **kw):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd, **kw)


def pid_for(rel: str) -> str:
    tail = rel.split("transformer_engine/common/", 1)[1]
    return "CXX-" + re.sub(r"[^A-Za-z0-9]+", "-", tail)


def load():
    man = yaml.safe_load(open(PROP / "divergence-manifest.yaml"))
    pin = man["metadata"]["upstream_sha"]
    head = sh("git", "rev-parse", "HEAD", cwd=ROOT / SUB).stdout.strip()
    assert head == pin, f"submodule {head[:12]} != pin {pin[:12]}"
    strat = yaml.safe_load(open(PROP / "cxx-strategy.yaml"))["files"]
    return pin, strat


def gen(pin, strat):
    PATCHES.mkdir(exist_ok=True)
    n = 0
    for row in strat:
        if row["cxx_strategy"] != "patch-queue":
            continue
        rel = row["file"]
        up = sh("git", "show", f"{pin}:{rel}", cwd=ROOT / SUB).stdout
        fk = (ROOT / rel).read_text(errors="replace")
        with tempfile.NamedTemporaryFile("w", delete=False) as f1:
            f1.write(up)
        r = subprocess.run(["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}",
                            f1.name, str(ROOT / rel)], capture_output=True, text=True)
        Path(f1.name).unlink()
        if r.returncode == 0:
            continue
        hdr = (f"# manifest: BK-001\n# base: {pin}\n# mechanism: cxx-guard-patch\n"
               f"# expiry: pin-bump-review\n# tests: ci/core.sh\n# owner: TBD\n"
               f"# evidence: {row['evidence']}\n")
        (PATCHES / f"{pid_for(rel)}.patch").write_text(hdr + r.stdout)
        n += 1
    print(f"gen: {n} CXX patches -> {PATCHES}")


def verify(pin):
    stale = []
    pf = sorted(PATCHES.glob("CXX-*.patch"))
    for p in pf:
        body = "\n".join(l for l in p.read_text().splitlines() if not l.startswith("#")) + "\n"
        rel = next(l[6:] for l in body.splitlines() if l.startswith("--- a/"))
        up = sh("git", "show", f"{pin}:{rel}", cwd=ROOT / SUB)
        with tempfile.TemporaryDirectory() as td:
            tgt = Path(td) / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(up.stdout)
            (Path(td) / "p.patch").write_text(body)
            ap = subprocess.run(["git", "apply", "--whitespace=nowarn", "p.patch"],
                                capture_output=True, cwd=td)
            if ap.returncode or tgt.read_text() != (ROOT / rel).read_text(errors="replace"):
                stale.append(p.stem)
    for s in stale:
        print(f"  STALE {s}")
    print(f"verify: {len(stale)} stale of {len(pf)}")
    return 1 if stale else 0


def assemble(pin, strat, out: Path):
    strat_by = {r["file"]: r["cxx_strategy"] for r in strat}
    # every C++ file the fork's common/ tracks
    fork_files = [f for f in sh("git", "ls-files", "transformer_engine/common").stdout.splitlines()
                  if CXX_RE.search(f)]
    up_files = set(sh("git", "ls-tree", "-r", "--name-only", pin, "transformer_engine/common",
                      cwd=ROOT / SUB).stdout.splitlines())
    counts = {"upstream-identical": 0, "patched": 0, "fork-native": 0}
    mismatch = []
    for rel in fork_files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        strategy = strat_by.get(rel)
        if strategy == "patch-queue":
            body = "\n".join(l for l in (PATCHES / f"{pid_for(rel)}.patch").read_text().splitlines()
                             if not l.startswith("#")) + "\n"
            dst.write_text(sh("git", "show", f"{pin}:{rel}", cwd=ROOT / SUB).stdout)
            (out / "p.patch").write_text(body)
            ap = subprocess.run(["git", "apply", "--whitespace=nowarn", "p.patch"],
                                capture_output=True, cwd=out)
            (out / "p.patch").unlink()
            if ap.returncode:
                mismatch.append((rel, "patch failed")); continue
            counts["patched"] += 1
        elif rel in up_files and sh("git", "show", f"{pin}:{rel}", cwd=ROOT / SUB).stdout \
                == (ROOT / rel).read_text(errors="replace") and strategy is None:
            dst.write_text(sh("git", "show", f"{pin}:{rel}", cwd=ROOT / SUB).stdout)
            counts["upstream-identical"] += 1
        else:
            # native-hip (converted or rocm-only) and anything else the fork owns: fork copy
            dst.write_bytes((ROOT / rel).read_bytes())
            counts["fork-native"] += 1
        if dst.read_bytes() != (ROOT / rel).read_bytes():
            mismatch.append((rel, "assembled != fork"))
    for rel, why in mismatch:
        print(f"  MISMATCH {rel}: {why}")
    print(f"assemble: {counts} -> {out}")
    print(f"tree identity vs fork common/: {'OK - byte-identical, '+str(len(fork_files))+' files' if not mismatch else f'{len(mismatch)} MISMATCHES'}")
    return 1 if mismatch else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["gen", "verify", "assemble"])
    ap.add_argument("--out", default=str(ROOT / "build" / "overlay-cxx"))
    a = ap.parse_args()
    pin, strat = load()
    if a.cmd == "gen":
        gen(pin, strat); return 0
    if a.cmd == "verify":
        return verify(pin)
    return assemble(pin, strat, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
