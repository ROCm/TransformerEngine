#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""C++ patch fingerprints (plugin plan S4.2). Symbol-level, via universal-ctags line ranges -
honest about limits: no libclang, so ranges are ctags' end-line tracking, good enough to say
WHICH function/struct a hunk lands in and whether a candidate pin moved or rewrote it.

  build                     fingerprint every CXX-* patch at the manifest pin -> fingerprints-cxx.json
  verify --upstream SHA     categorize each patch vs SHA: target-unchanged / target-changed /
                            target-moved / file-gone (the pin-bump repair report)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROP = Path(__file__).resolve().parents[1]
ROOT = PROP.parent.parent
SUB = "3rdparty/transformer_engine_nvidia"
PATCHES = PROP / "patches-cxx"
OUT = PATCHES / "fingerprints-cxx.json"


def sh(*a, cwd=ROOT):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd)


def symbols(content: str, suffix: str) -> list[dict]:
    """ctags symbols with line ranges for one file's content."""
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(content); name = f.name
    r = sh("ctags", "--output-format=json", "--fields=+ne", "--languages=C,C++,CUDA",
           "-o", "-", name)
    Path(name).unlink()
    out = []
    for line in r.stdout.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "line" in j:
            out.append({"name": j["name"], "kind": j.get("kind"), "start": j["line"],
                        "end": j.get("end", j["line"])})
    return out


def patch_hunks(p: Path):
    body = [l for l in p.read_text().splitlines() if not l.startswith("#")]
    rel = next(l[6:] for l in body if l.startswith("--- a/"))
    hunks = [(int(m.group(1)), int(m.group(2) or 1)) for m in
             (re.match(r"@@ -(\d+),?(\d+)? ", l) for l in body) if m]
    return rel, hunks


def hunk_symbols(rel: str, hunks, sha: str) -> list[dict]:
    content = sh("git", "show", f"{sha}:{rel}", cwd=ROOT / SUB).stdout
    if not content:
        return None
    syms = symbols(content, "." + rel.rsplit(".", 1)[1])
    out = []
    for start, count in hunks:
        end = start + count - 1
        hit = [s for s in syms if s["start"] <= end and s["end"] >= start]
        best = sorted(hit, key=lambda s: s["end"] - s["start"])[:1]
        text = "\n".join(content.splitlines()[max(0, start - 1): end])
        out.append({"span": [start, end],
                    "symbol": best[0]["name"] if best else None,
                    "kind": best[0]["kind"] if best else None,
                    "sha256_span": hashlib.sha256(text.encode()).hexdigest()[:16]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--upstream")
    a = ap.parse_args()
    pin = yaml.safe_load(open(PROP / "divergence-manifest.yaml"))["metadata"]["upstream_sha"]

    if a.cmd == "build":
        fp = {}
        for p in sorted(PATCHES.glob("CXX-*.patch")):
            rel, hunks = patch_hunks(p)
            fp[p.stem] = {"file": rel, "hunks": hunk_symbols(rel, hunks, pin)}
        OUT.write_text(json.dumps({"pin": pin, "patches": fp}, indent=1) + "\n")
        n = sum(len(v["hunks"]) for v in fp.values())
        named = sum(1 for v in fp.values() for h in v["hunks"] if h["symbol"])
        print(f"build: {len(fp)} patches, {n} hunks ({named} symbol-attributed) -> {OUT}")
        return 0

    if not a.upstream:
        sys.exit("verify needs --upstream SHA")
    sh("git", "fetch", "--no-tags", "origin", cwd=ROOT / SUB)
    base = json.loads(OUT.read_text())
    from collections import Counter
    cat = Counter(); repairs = []
    for pid, v in base["patches"].items():
        cur = hunk_symbols(v["file"], [(h["span"][0], h["span"][1] - h["span"][0] + 1)
                                       for h in v["hunks"]], a.upstream)
        if cur is None:
            cat["file-gone"] += 1; repairs.append((pid, "file-gone")); continue
        changed = [h1["symbol"] for h1, h2 in zip(v["hunks"], cur)
                   if h1["sha256_span"] != h2["sha256_span"]]
        if not changed:
            cat["target-unchanged"] += 1
        else:
            moved = all(h1["symbol"] == h2["symbol"] and h1["symbol"] for h1, h2 in zip(v["hunks"], cur))
            cat["target-moved" if moved else "target-changed"] += 1
            repairs.append((pid, f"{'moved' if moved else 'changed'}: {sorted(set(filter(None, changed))) or 'unattributed span'}"))
    print(f"queue fingerprinted @ {base['pin'][:12]}; candidate = {a.upstream[:12]}")
    print(json.dumps(dict(cat)))
    for pid, why in repairs[:20]:
        print(f"  REPAIR {pid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
