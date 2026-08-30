# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Did the tests that "passed without the patch" ever execute the lines the patch adds?

A RETIRE-CANDIDATE from shrink_queue.py only means the mapped tests passed with the patch
removed. If those tests never reach the patched code (skipped configs, unexercised paths), the
verdict is vacuous - the sec 4.4 skip-masking problem applied to our own process. This tool
intersects each patch's ADDED lines (fork line numbers, GNU diff vs the upstream file) with the
lines coverage saw executed on the fork tree, and splits candidates into:

  executed     >= 1 added line ran; the passing tests are evidence  -> retirable (on this arch)
  UNTESTED     0 added lines ran; the passing tests say nothing     -> NOT retirable on this evidence

Usage: patch_coverage.py --coverage <coverage.json> [--results <results.json>] [IDS...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROP = HERE.parent
ROOT = PROP.parent.parent
SUBMODULE = ROOT / "3rdparty/transformer_engine_nvidia"
PATCHES = PROP / "patches"


def target_of(pid: str) -> str:
    for line in (PATCHES / f"{pid}.patch").read_text().splitlines():
        if line.startswith("--- a/"):
            return line[6:]
    raise SystemExit(f"{pid}: no target")


def added_lines(rel: str, base: str) -> set[int]:
    up = subprocess.run(["git", "show", f"{base}:{rel}"], capture_output=True, cwd=SUBMODULE, check=True).stdout
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(up); name = f.name
    p = subprocess.run(["diff", "--unchanged-line-format=", "--old-line-format=", "--new-line-format=%dn|%L",
                        name, str(ROOT / rel)], capture_output=True, text=True)
    Path(name).unlink()
    out = set()
    for raw in p.stdout.splitlines():
        n, _, text = raw.partition("|")
        t = text.strip()
        if t and not t.startswith("#"):          # comments/blank lines are never "executed"
            out.add(int(n))
    return out


def stmt_start_map(rel: str) -> dict[int, int]:
    """line -> first line of its enclosing statement. coverage.py records execution on a
    statement's first line only, so an added line inside a multi-line literal/call must be
    judged by its statement's start (verified on PT-045: 1 added line inside a multi-line
    tuple, tests fail without it, raw coverage said 0%)."""
    import ast
    src = (ROOT / rel).read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    m: dict[int, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and hasattr(node, "end_lineno"):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                # innermost statement wins: only set if unset or this one starts later (nested)
                if ln not in m or node.lineno > m[ln]:
                    m[ln] = node.lineno
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", required=True); ap.add_argument("--results"); ap.add_argument("ids", nargs="*")
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    import yaml
    base = a.base or yaml.safe_load((PROP / "divergence-manifest.yaml").read_text())["metadata"]["upstream_sha"]
    cov = json.load(open(a.coverage))["files"]
    executed = {}
    for f, d in cov.items():
        rel = f.split("transformer_engine/", 1)
        if len(rel) == 2:
            executed["transformer_engine/" + rel[1]] = set(d["executed_lines"])
    results = json.load(open(a.results)) if a.results else {}
    ids = a.ids or sorted(p.stem for p in PATCHES.glob("*.patch") if not p.stem.startswith("JX-"))
    print(f"{'ID':8s} {'shrink verdict':18s} {'added':>5s} {'exec':>5s}  coverage verdict   target")
    summary = {}
    for pid in ids:
        rel = target_of(pid)
        if not rel.endswith(".py"):
            continue
        add = added_lines(rel, base)
        ex = executed.get(rel, set()); smap = stmt_start_map(rel)
        ran = {ln for ln in add if ln in ex or smap.get(ln, ln) in ex}
        frac = len(ran) / len(add) if add else 0.0
        # Module-level guard lines (`if IS_HIP_EXTENSION:`) execute at import no matter what, so
        # ">=1 line ran" is nearly vacuous. Bands: UNTESTED 0 | WEAK <25% | executed >=25%.
        if not add: cv = "n/a"
        elif not ran: cv = "UNTESTED"
        elif frac < 0.25: cv = "WEAK"
        else: cv = "executed"
        sv = results.get(pid, {}).get("status", "-")
        if sv == "RETIRE-CANDIDATE" and cv in ("UNTESTED", "WEAK"):
            cv += " -> retire verdict is thin evidence"
        summary[pid] = {"shrink": sv, "added": len(add), "executed": len(ran), "fraction": round(frac, 2), "coverage": cv, "target": rel}
        print(f"{pid:8s} {sv:18s} {len(add):5d} {len(ran):5d} {frac:5.0%}  {cv:40s} {rel.replace('transformer_engine/', '')}")
    if a.results:
        out = Path(a.results).with_name("coverage_verdicts.json"); out.write_text(json.dumps(summary, indent=1))
        print(f"\nwritten: {out}")
    cands = [v for v in summary.values() if v["shrink"] == "RETIRE-CANDIDATE"]
    n_ok = sum(1 for v in cands if v["coverage"] == "executed")
    n_weak = sum(1 for v in cands if v["coverage"].startswith("WEAK"))
    n_unt = sum(1 for v in cands if v["coverage"].startswith("UNTESTED"))
    print(f"retire candidates: {len(cands)} total = {n_ok} executed (>=25% of added lines ran), "
          f"{n_weak} weak (<25%), {n_unt} untested (0 lines ran)")


if __name__ == "__main__":
    main()
