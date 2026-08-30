# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Static inventory of the transformer_engine_torch seam.

Answers one question without building anything: does the ROCm extension expose
every name that upstream's Python asks for?

  DEMAND  every `tex.NAME`, `from transformer_engine_torch import NAME`, and
          `transformer_engine_torch.NAME` in upstream Python at the pinned base,
          found by AST walk (no regex on source).
  SUPPLY  every name registered in the fork's pybind sources - m.def("NAME"),
          py::class_<..>(m, "NAME"), py::enum_<..>(m, "NAME"), submodules -
          with #if/#ifdef nesting tracked so CUDA-only registrations are
          reported separately from unconditional ones.

  demand - supply  = names upstream will ask for that ROCm cannot answer.
                     Non-empty means the facade cannot be closed.
  supply - demand  = ROCm extras. Harmless, but they leak through the
                     star-import in cpp_extensions/__init__.py, so __all__
                     fidelity matters.

This is the coarse filter for TE_ROCM_EXTENSION_API (proposal section 3.3).
It is static: it does not check signatures, enum values, or class members.
Those need the built extension.

Usage:
  seam_inventory.py --base <upstream-sha> [--csrc <dir>] [--json out.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SEAM = "transformer_engine_torch"


# ----------------------------------------------------------------- demand ---

class SeamVisitor(ast.NodeVisitor):
    """Collect attribute accesses on the seam module within one file."""

    def __init__(self, path: str):
        self.path = path
        self.aliases: set[str] = set()          # local names bound to the seam
        self.names: dict[str, set[str]] = defaultdict(set)   # name -> forms
        self.star = False

    def visit_Import(self, node: ast.Import):
        for a in node.names:
            if a.name == SEAM:
                self.aliases.add(a.asname or SEAM)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == SEAM:
            for a in node.names:
                if a.name == "*":
                    self.star = True
                else:
                    self.names[a.name].add("from-import")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # tex.NAME  or  transformer_engine_torch.NAME
        base = node.value
        if isinstance(base, ast.Name) and base.id in self.aliases:
            self.names[node.attr].add("attr")
        self.generic_visit(node)


def git_ls(base: str, prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", base, prefix],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.endswith(".py")]


def git_show(base: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{base}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def collect_demand(base: str) -> tuple[dict[str, dict], list[str]]:
    demand: dict[str, dict] = {}
    star_files: list[str] = []
    for path in git_ls(base, "transformer_engine/"):
        try:
            tree = ast.parse(git_show(base, path), filename=path)
        except SyntaxError as e:
            print(f"  ! cannot parse {path}: {e}", file=sys.stderr)
            continue
        v = SeamVisitor(path)
        v.visit(tree)
        if v.star:
            star_files.append(path)
        for name, forms in v.names.items():
            d = demand.setdefault(name, {"files": set(), "forms": set()})
            d["files"].add(path)
            d["forms"] |= forms
    return demand, star_files


CTYPES_RE = re.compile(r"_TE_LIB_CTYPES\.([A-Za-z_]\w*)")


def collect_ctypes_demand(root: Path) -> dict[str, list[str]]:
    """nvte_* symbols the FORK's Python calls on the core .so via ctypes (ABI-001).

    Upstream calls none; these exist only in patched files. They bind by symbol name at load,
    so they appear in no pybind inventory - this scan is what keeps them from being invisible
    to a core-wheel reuse decision. Supply is the core .so's export table (needs a build:
    `nm -D libtransformer_engine*.so | grep ' T nvte_'`), reported here as demand only."""
    out: dict[str, list[str]] = {}
    for p in sorted((root / "transformer_engine").rglob("*.py")):
        for m in CTYPES_RE.finditer(p.read_text(errors="replace")):
            name = m.group(1)
            if name in ("restype", "argtypes"):
                continue
            out.setdefault(name, []).append(str(p.relative_to(root)))
    return out


# ----------------------------------------------------------------- supply ---

DEF_RE = re.compile(r'\b(\w+)\.def\(\s*"([A-Za-z_]\w*)"', re.S)
CLASS_RE = re.compile(r'(?:py|pybind11)::(class_|enum_)<[^;]*?>\s*\(\s*(\w+)\s*,\s*"([A-Za-z_]\w*)"', re.S)
SUBMOD_RE = re.compile(r'(\w+)\.def_submodule\(\s*"([A-Za-z_]\w*)"', re.S)
PP_RE = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')


def collect_supply(csrc: Path) -> dict[str, dict]:
    """Walk pybind sources tracking preprocessor nesting."""
    supply: dict[str, dict] = {}
    files = sorted(p for p in list(csrc.rglob("*.cpp")) + list(csrc.rglob("*.h"))
                   if re.search(r'\b(m|\w+)\.def\(|(py|pybind11)::(class_|enum_)<', p.read_text(errors="replace")))
    for path in files:
        text = path.read_text(errors="replace").replace("\\\n", "\n")
        # All registration forms can span lines (m.def(\n"name", ...)), so
        # scan the whole text and compute the #if guard at each match position.
        for mm in DEF_RE.finditer(text):
            lineno = text.count("\n", 0, mm.start()) + 1
            _register(supply, mm.group(2), "function", path, lineno, _guard_at(text, mm.start()))
        for mm in SUBMOD_RE.finditer(text):
            lineno = text.count("\n", 0, mm.start()) + 1
            _register(supply, mm.group(2), "submodule", path, lineno, _guard_at(text, mm.start()))
        # class_/enum_ can span lines; do a second pass on the whole text.
        for mm in CLASS_RE.finditer(text):
            lineno = text.count("\n", 0, mm.start()) + 1
            guard = _guard_at(text, mm.start())
            kind = "class" if mm.group(1) == "class_" else "enum"
            _register(supply, mm.group(3), kind, path, lineno, guard)
    return supply


def _guard_at(text: str, pos: int) -> str:
    stack: list[str] = []
    for line in text[:pos].splitlines():
        m = PP_RE.match(line)
        if not m: continue
        kind, cond = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"): stack.append(f"{kind} {cond}")
        elif kind == "elif" and stack: stack[-1] = f"elif {cond}"
        elif kind == "else" and stack: stack[-1] = f"else[{stack[-1]}]"
        elif kind == "endif" and stack: stack.pop()
    return " && ".join(x for x in stack if "_H_" not in x)


def _register(supply, name, kind, path, lineno, guard):
    e = supply.setdefault(name, {"kind": kind, "sites": []})
    e["sites"].append({"file": str(path), "line": lineno, "guard": guard})


def rocm_excluded(guard: str) -> bool:
    """True if this guard chain means 'not compiled on ROCm'.

    Walk each frame of the nesting. A frame excludes ROCm if it is
    `ifndef USE_ROCM` / `if !defined(USE_ROCM)` NOT inside an else, or the
    else-branch of `ifdef USE_ROCM`. The else-branch of `ifndef USE_ROCM`
    IS the ROCm branch and is reachable.
    """
    for frame in guard.split(" && "):
        f = frame.replace(" ", "")
        if f.startswith("else["):
            inner = f[5:-1]
            if inner.startswith("ifdefUSE_ROCM") or inner.startswith("ifUSE_ROCM") \
               or inner.startswith("ifdefined(USE_ROCM)"):
                return True          # else of "is ROCm" -> not ROCm
            continue                 # else of "not ROCm" -> IS ROCm, fine
        if f.startswith("ifndefUSE_ROCM") or f.startswith("if!defined(USE_ROCM)") \
           or f.startswith("if!USE_ROCM"):
            return True
    return False


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="upstream SHA to read demand from")
    ap.add_argument("--csrc", action="append",
                    help="pybind source dir; repeatable. Default: pytorch/csrc + common/util")
    ap.add_argument("--json", help="write full inventory here")
    ap.add_argument("--values", action="store_true", help="also list enum members per #if branch (static)")
    args = ap.parse_args()

    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True, check=True).stdout.strip())
    csrc_dirs = [root / d for d in (args.csrc or ["transformer_engine/pytorch/csrc", "transformer_engine/common/util"])]

    demand, star_files = collect_demand(args.base)
    supply = {}
    for d in csrc_dirs:
        for n, e in collect_supply(d).items():
            supply.setdefault(n, {"kind": e["kind"], "sites": []})["sites"].extend(e["sites"])

    supplied_on_rocm = {n for n, e in supply.items()
                        if any(not rocm_excluded(s["guard"]) for s in e["sites"])}
    supplied_cuda_only = {n for n in supply if n not in supplied_on_rocm}

    missing = sorted(n for n in demand if n not in supply)
    cuda_only_demanded = sorted(n for n in demand if n in supplied_cuda_only)
    extras = sorted(n for n in supply if n not in demand)
    gated = {n: sorted({s["guard"] for s in supply[n]["sites"] if s["guard"]})
             for n in demand if n in supply and any(s["guard"] for s in supply[n]["sites"])}

    print(f"upstream base : {args.base}")
    print("pybind sources: " + ", ".join(str(d.relative_to(root)) for d in csrc_dirs))
    print()
    print(f"DEMAND  {len(demand):4d} distinct names referenced by upstream Python")
    print(f"SUPPLY  {len(supply):4d} distinct names registered by the fork's pybind")
    print(f"        {len(supplied_on_rocm):4d} unconditional or ROCm-reachable")
    print(f"        {len(supplied_cuda_only):4d} CUDA-only (never compiled on ROCm)")
    print()

    def show(title, names, detail=None):
        print(f"--- {title}: {len(names)} ---")
        for n in dict.fromkeys(names):
            line = f"  {n}"
            if detail: line += f"    {detail(n)}"
            print(line)
        print()

    show("MISSING (demanded, never registered) -> facade CANNOT close", missing,
         lambda n: f"[{', '.join(sorted(demand[n]['forms']))}] " +
                   ", ".join(sorted(p.replace('transformer_engine/pytorch/', '') for p in demand[n]['files']))[:90])
    show("CUDA-ONLY (demanded, registered only behind a non-ROCm guard)", cuda_only_demanded,
         lambda n: "; ".join(sorted({s['guard'] for s in supply[n]['sites']})))
    show("GATED but ROCm-reachable (demanded, registered under some #if)",
         sorted(gated), lambda n: "; ".join(gated[n]))
    print(f"--- star-import sites (copy the whole facade namespace) : {len(star_files)} ---")
    for f in star_files: print(f"  {f}")
    print()
    show("EXTRAS (ROCm supplies, upstream never asks) - leak via star-import", extras,
         lambda n: supply[n]["kind"])

    early = sorted(n for n, d in demand.items() if "from-import" in d["forms"])
    print(f"--- early-bound names (from-import; copied at import time): {len(early)} ---")
    print("  " + ", ".join(early))
    print()

    if args.values:
        # Static enum VALUE inventory: members registered per enum in the pybind sources, with
        # the #if guard of the enum_<> site. Runtime truth is tests/te_rocm/test_seam_values.py;
        # this is the build-free view for pin-bump triage.
        ENUM_BLOCK = re.compile(r'(?:py|pybind11)::enum_<[^;]*?>\s*\(\s*\w+\s*,\s*"(\w+)"(.*?);', re.S)
        VALUE_RE = re.compile(r'\.value\(\s*"(\w+)"')
        print("--- enum VALUES registered (static, per #if branch) ---")
        for d in csrc_dirs:
            for path in sorted(list(d.rglob("*.cpp")) + list(d.rglob("*.h"))):
                text = path.read_text(errors="replace").replace("\\\n", "\n")
                for mm in ENUM_BLOCK.finditer(text):
                    guard = _guard_at(text, mm.start())
                    members = VALUE_RE.findall(mm.group(2))
                    tag = "  [CUDA-only]" if rocm_excluded(guard) else ""
                    print(f"  {mm.group(1)}{tag}  guard='{guard or '-'}'  {len(members)} members: {', '.join(members)}")
        print()

    ctypes_demand = collect_ctypes_demand(root)
    print(f"--- ctypes demand on the CORE .so from the fork's Python (ABI-001; not pybind, not in any header): {len(ctypes_demand)} ---")
    for n, files in sorted(ctypes_demand.items()):
        print(f"  {n}    {', '.join(sorted(set(files)))}")
    print("  supply check needs a built core lib: nm -D libtransformer_engine*.so | grep ' T nvte_'")
    print()

    verdict = "CLOSED" if not missing and not cuda_only_demanded else "OPEN"
    print(f"VERDICT: facade surface is {verdict}")

    if args.json:
        out = {
            "base": args.base,
            "demand": {n: {"files": sorted(d["files"]), "forms": sorted(d["forms"])} for n, d in demand.items()},
            "supply": supply,
            "missing": missing, "cuda_only_demanded": cuda_only_demanded,
            "extras": extras, "star_import_files": star_files, "verdict": verdict,
        }
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"inventory written: {args.json}")

    sys.exit(0 if verdict == "CLOSED" else 1)


if __name__ == "__main__":
    main()
