# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""S3.2: prove (or refuse) late-boundness for runtime-override candidates.

A patch may move from the build tier to the runtime-override tier ONLY if every touched symbol
is late-bound everywhere in the overlay: replacing the symbol on its module after import must be
sufficient. The census walks the whole overlay AST for disqualifying references to each touched
symbol of each candidate patch:

  from-import        `from <target-module> import <symbol>`  - copies the object at import time
  alias              `g = mod.symbol` at module level        - captures the object
  default-arg        `def f(x=mod.symbol)`                   - captures at definition time
  decorator          `@mod.symbol` applied at import         - result baked into the decorated fn
  subclass           `class C(mod.Symbol)`                   - class layout fixed at import

Zero disqualifiers -> census PASSED (still needs the S3.2 swap test on GPU before
`runtime_eligible: proven`). Any disqualifier -> the candidate STAYS build-tier, with the exact
referencing sites recorded as evidence.

Usage: override_census.py [--overlay build/overlay-phaseC] [IDS...]   (default: manifest candidates)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

PROP = Path(__file__).resolve().parents[1]
ROOT = PROP.parent.parent
FPRINTS = PROP / "patches" / "fingerprints.json"


def module_of(rel: str) -> str:
    return rel.removesuffix(".py").replace("/", ".").replace(".__init__", "")


def census_one(overlay: Path, target_rel: str, symbols: list[str]) -> list[str]:
    tmod = module_of(target_rel)
    tshort = tmod.rsplit(".", 1)[-1]
    disq: list[str] = []
    for py in sorted((overlay / "transformer_engine").rglob("*.py")):
        rel = str(py.relative_to(overlay))
        if rel == target_rel:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        # local names that alias the target module (import x.y as z / from pkg import mod)
        mod_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == tmod:
                        mod_aliases.add((a.asname or a.name).split(".")[0] if not a.asname else a.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == tmod or (tmod.endswith("." + (node.module or "")) is False and False)):
                    pass
                if node.module == tmod:
                    for a in node.names:
                        if a.name in symbols:
                            disq.append(f"{rel}:{node.lineno} from-import {a.name}")
                        if a.name == "*":
                            disq.append(f"{rel}:{node.lineno} star-import of {tmod}")
                elif node.module and node.module.rsplit(".", 1)[-1] != tshort:
                    for a in node.names:
                        if a.name == tshort:
                            mod_aliases.add(a.asname or a.name)

        def is_target_attr(n: ast.AST, sym: str) -> bool:
            return (isinstance(n, ast.Attribute) and n.attr == sym and isinstance(n.value, ast.Name)
                    and n.value.id in mod_aliases)

        for node in ast.walk(tree):
            for sym in symbols:
                if sym == "<module-level>":
                    continue
                if isinstance(node, ast.Assign) and is_target_attr(node.value, sym) and \
                        any(isinstance(t, ast.Name) for t in node.targets) and node.col_offset == 0:
                    disq.append(f"{rel}:{node.lineno} module-level alias of {sym}")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dflt in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
                        if is_target_attr(dflt, sym):
                            disq.append(f"{rel}:{node.lineno} default-arg captures {sym}")
                    for dec in node.decorator_list:
                        d = dec.func if isinstance(dec, ast.Call) else dec
                        if is_target_attr(d, sym):
                            disq.append(f"{rel}:{node.lineno} decorator {sym}")
                if isinstance(node, ast.ClassDef):
                    for b in node.bases:
                        if is_target_attr(b, sym):
                            disq.append(f"{rel}:{node.lineno} subclass of {sym}")
    return disq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--overlay", default=str(ROOT / "build" / "overlay-phaseC"))
    ap.add_argument("--json")
    a = ap.parse_args()

    fp = json.loads(FPRINTS.read_text())["patches"]
    manifest = yaml.safe_load((PROP / "divergence-manifest.yaml").read_text())
    # candidates = feature/entry ids with runtime_eligible: candidate whose parent patch is ACTIVE
    cands = {}
    for e in manifest["entries"]:
        pid = e["id"]
        feats = e.get("features") or []
        marks = [f["id"] for f in feats if f.get("runtime_eligible") == "candidate"]
        if e.get("runtime_eligible") == "candidate":
            marks.append(pid)
        if marks and pid in fp:
            cands[pid] = marks
    ids = a.ids or sorted(cands)
    overlay = Path(a.overlay)
    out = {}
    for pid in ids:
        rec = fp.get(pid)
        if rec is None:
            out[pid] = {"verdict": "moot", "why": "patch not in the active queue (retired) - no override needed"}
            print(f"  {pid:8s} MOOT (retired)"); continue
        syms = [s for s in rec["symbols"] if s != "<module-level>"]
        disq = census_one(overlay, rec["target"], syms or ["<module-level>"])
        if "<module-level>" in rec["symbols"] and not syms:
            disq.append("patch touches only module level - inherently import-time")
        out[pid] = {"verdict": "census-passed" if not disq else "build-tier",
                    "symbols": syms, "disqualifiers": disq[:20]}
        print(f"  {pid:8s} {'CENSUS-PASSED (swap test next)' if not disq else 'BUILD-TIER'}  "
              f"symbols={syms[:4]}{' disq: ' + '; '.join(disq[:3]) if disq else ''}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
