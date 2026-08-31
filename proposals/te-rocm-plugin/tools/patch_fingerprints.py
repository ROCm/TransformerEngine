# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Target-level applicability fingerprints for the patch queue (plan S3.1, proposal sec 3.4).

For each active patch, record WHICH top-level symbols (functions/classes, or <module-level>) its
hunks touch in the UPSTREAM file at the pin, plus a normalized-AST hash of each such symbol's
upstream source. On a candidate pin bump, `verify --upstream <sha>` classifies every patch by ID:

  target-unchanged   every touched symbol's normalized AST is identical at the candidate ->
                     the patch reapplies (line offsets alone never trip it)
  target-moved       symbols identical but at different lines -> reapplies with offset
  target-changed     >=1 touched symbol's AST changed (or vanished) -> TRIP, with the symbol
                     names, so repair starts at the right place
  file-gone          the upstream file no longer exists

This is the sync-workflow step 3 of proposal sec 5, replacing bare `git apply --check` context
matching. Normalization: AST round-trip via ast.unparse with docstrings dropped, so comment and
formatting churn upstream never trips a patch. Per-patch exact-SHA preconditions remain excluded
by design.

Usage:
  patch_fingerprints.py build                      # fingerprint the active queue at the manifest pin
  patch_fingerprints.py verify --upstream <sha>    # classify every patch against a candidate pin
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

PROP = Path(__file__).resolve().parents[1]
ROOT = PROP.parent.parent
SUBMODULE = ROOT / "3rdparty" / "transformer_engine_nvidia"
PATCHES = PROP / "patches"
FPRINTS = PATCHES / "fingerprints.json"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
MODULE_LEVEL = "<module-level>"


def show(sha: str, rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"{sha}:{rel}"], capture_output=True, text=True, cwd=SUBMODULE)
    return r.stdout if r.returncode == 0 else None


def pin() -> str:
    return yaml.safe_load((PROP / "divergence-manifest.yaml").read_text())["metadata"]["upstream_sha"]


def patch_meta(p: Path) -> tuple[str, list[tuple[int, int]]]:
    """(target rel path, [(old_start, old_len), ...]) from the unified diff."""
    rel = None
    hunks = []
    for line in p.read_text().splitlines():
        if line.startswith("--- a/"):
            rel = line[6:]
        m = HUNK_RE.match(line)
        if m:
            hunks.append((int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1))
    return rel, hunks


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def symbol_table(src: str, path: str) -> dict[str, tuple[int, int, str]]:
    """{symbol: (start, end, normalized-ast-sha)}. Non-Python files get one whole-file entry."""
    if not path.endswith(".py"):
        sha = hashlib.sha256(src.encode()).hexdigest()
        return {MODULE_LEVEL: (1, src.count("\n") + 1, sha)}
    tree = ast.parse(src)
    out = {}
    covered = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.parse(ast.get_source_segment(src, node) or "")
            norm = ast.unparse(_strip_docstrings(seg))
            out[node.name] = (node.lineno, node.end_lineno or node.lineno,
                              hashlib.sha256(norm.encode()).hexdigest())
            covered.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    # everything not inside a def/class = module level (imports, constants, registrations)
    mod_lines = [l for i, l in enumerate(src.splitlines(), 1) if i not in covered and l.strip() and not l.strip().startswith("#")]
    out[MODULE_LEVEL] = (1, src.count("\n") + 1, hashlib.sha256("\n".join(mod_lines).encode()).hexdigest())
    return out


def touched_symbols(hunks, table) -> list[str]:
    names = set()
    for start, length in hunks:
        span = range(start, start + max(length, 1))
        hit = False
        for name, (s, e, _) in table.items():
            if name != MODULE_LEVEL and not (span.stop <= s or span.start > e):
                names.add(name); hit = True
        if not hit:
            names.add(MODULE_LEVEL)
    return sorted(names)


def build():
    base = pin()
    out = {"base": base, "patches": {}}
    for p in sorted(PATCHES.glob("*.patch")):
        rel, hunks = patch_meta(p)
        src = show(base, rel)
        if src is None:
            print(f"  {p.stem}: {rel} not upstream at pin - skipped"); continue
        table = symbol_table(src, rel)
        syms = touched_symbols(hunks, table)
        out["patches"][p.stem] = {"target": rel,
                                  "symbols": {s: table[s][2] for s in syms},
                                  "spans": {s: table[s][:2] for s in syms}}
        print(f"  {p.stem}: {len(syms)} symbol(s) touched in {rel}: {', '.join(syms[:6])}{'...' if len(syms) > 6 else ''}")
    FPRINTS.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"fingerprints for {len(out['patches'])} patches @ {base[:12]} -> {FPRINTS.relative_to(ROOT)}")


def verify(candidate: str):
    fp = json.loads(FPRINTS.read_text())
    print(f"queue fingerprinted @ {fp['base'][:12]}; candidate = {candidate[:12]}")
    counts = {"target-unchanged": 0, "target-moved": 0, "target-changed": 0, "file-gone": 0}
    for pid, rec in sorted(fp["patches"].items()):
        src = show(candidate, rec["target"])
        if src is None:
            counts["file-gone"] += 1
            print(f"  {pid:8s} FILE-GONE          {rec['target']}"); continue
        table = symbol_table(src, rec["target"])
        changed, moved = [], False
        for sym, sha in rec["symbols"].items():
            if sym not in table:
                changed.append(sym + "(vanished)")
            elif table[sym][2] != sha:
                changed.append(sym)
            elif tuple(table[sym][:2]) != tuple(rec["spans"][sym]):
                moved = True
        if changed:
            counts["target-changed"] += 1
            print(f"  {pid:8s} TARGET-CHANGED     {rec['target']}  symbols: {', '.join(changed[:5])}")
        elif moved:
            counts["target-moved"] += 1
        else:
            counts["target-unchanged"] += 1
    print(json.dumps(counts))
    return 0 if counts["target-changed"] == 0 and counts["file-gone"] == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "verify"]); ap.add_argument("--upstream")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        if not a.upstream:
            sys.exit("verify needs --upstream <sha>")
        sys.exit(verify(a.upstream))


if __name__ == "__main__":
    main()
