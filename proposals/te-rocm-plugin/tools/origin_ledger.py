#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Origin ledger generator (plugin plan S4.4). GENERATED artifact - never hand-edit the output.

For every file under transformer_engine/common/ in the fork:
  - upstream ancestor at the pin -> {fork_path, upstream_path, upstream_blob_at_pin,
    cxx_strategy, patch_ids, state}
  - ROCm-only trees / fork-only files -> upstream: null, cxx_strategy: native (by existence)

Regenerated per pin bump; `git diff` of the ledger IS the C++ upstream-intake report.
Usage: origin_ledger.py [--out proposals/te-rocm-plugin/origin-ledger.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
PROP = TOOLS.parent
ROOT = PROP.parent.parent
SUB = "3rdparty/transformer_engine_nvidia"
CXX_RE = re.compile(r"\.(cu|cpp|h|cuh|hpp|c|hip)$")


def sh(*a, cwd=ROOT):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROP / "origin-ledger.json"))
    args = ap.parse_args()

    man = yaml.safe_load(open(PROP / "divergence-manifest.yaml"))
    pin = man["metadata"]["upstream_sha"]
    head = sh("git", "rev-parse", "HEAD", cwd=ROOT / SUB).stdout.strip()
    if head != pin:
        sys.exit(f"submodule HEAD {head[:12]} != manifest pin {pin[:12]}")

    strategy = {}
    strat_file = PROP / "cxx-strategy.yaml"
    for row in yaml.safe_load(open(strat_file))["files"]:
        strategy[row["file"]] = row["cxx_strategy"]

    # upstream blob ids at the pin for common/
    up = {}
    for line in sh("git", "ls-tree", "-r", pin, "transformer_engine/common",
                   cwd=ROOT / SUB).stdout.splitlines():
        meta, path = line.split("\t")
        up[path] = meta.split()[2]

    # patch ids per target file (python patches today; CXX-* join in S4.2)
    patch_ids = {}
    for p in sorted(list((PROP / "patches").glob("*.patch")) + list((PROP / "patches-cxx").glob("*.patch"))):
        for l in p.read_text().splitlines():
            if l.startswith("--- a/"):
                patch_ids.setdefault(l[6:], []).append(p.stem)
                break

    ledger = []
    tracked = sh("git", "ls-files", "transformer_engine/common").stdout.splitlines()
    for f in sorted(tracked):
        if not CXX_RE.search(f) and not f.endswith(".py"):
            continue
        p = ROOT / f
        if not p.exists():
            continue
        entry = {"fork_path": f}
        # hipify-generated names map back to their CUDA source for ancestry
        src = re.sub(r"_hip\.(h|cuh|cpp)$", r".\1", f.replace(".hip", ".cu")) if "_hip." in f or f.endswith(".hip") else f
        if f in up or src in up:
            key = f if f in up else src
            entry["upstream_path"] = key
            entry["upstream_blob_at_pin"] = up[key]
            up_content = sh("git", "show", f"{pin}:{key}", cwd=ROOT / SUB).stdout
            entry["state"] = "identical" if up_content == p.read_text(errors="replace") else "diverged"
            entry["cxx_strategy"] = strategy.get(f) or strategy.get(src) or (
                "vendored-python" if f.endswith(".py") else "identical-at-pin")
        else:
            entry["upstream_path"] = None
            entry["upstream_blob_at_pin"] = None
            entry["state"] = "rocm-only"
            entry["cxx_strategy"] = "native-hip" if CXX_RE.search(f) else "fork-only-python"
        entry["patch_ids"] = patch_ids.get(f, [])
        ledger.append(entry)

    out = {"pin": pin, "generated_by": "tools/origin_ledger.py", "schema": 1,
           "counts": {}, "files": ledger}
    from collections import Counter
    out["counts"] = dict(Counter(e["state"] for e in ledger))
    out["counts_strategy"] = dict(Counter(e["cxx_strategy"] for e in ledger))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"origin-ledger: {len(ledger)} files -> {args.out}")
    print(" states:", out["counts"])
    print(" strategies:", out["counts_strategy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
