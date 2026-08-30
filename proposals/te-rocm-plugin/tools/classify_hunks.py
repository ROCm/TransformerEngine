# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Classify every ADDED Python line of the fork (vs the pinned upstream base) into buckets.

This is the "canonical-v2 classifier" the manifest refers to, written down so the split is
reproducible per sync. Rules are explicit and ordered; the first matching bucket wins.

  header    the AMD portability/copyright/license comment block the fork prepends to a file
  guard     the line is the condition of, or lies inside the body OR else-branch of, an `if`/
            ternary whose condition names a ROCm guard (IS_HIP_EXTENSION, rocm_build(), ...).
            Found by AST on the fork's file, not by regex: both branches count, because the
            else-branch is upstream's code re-indented and shows up as "added" in the diff.
  sidecar   the line references a ROCm-only sidecar module or symbol (MXFP4*, triton_kernels,
            fsdp2_allgather, aiter, hipkittens, rocshmem, ...) and is not already inside a guard
  unmarked  everything else: a semantic edit to upstream code with no ROCm marker around it.
            This is the bucket that matters most - real feature divergence.

Invariant: header + guard + sidecar + unmarked == added_lines (GNU diff, '^>' count), per file.
The v2.3 manifest carried only {sidecar, guard, unmarked}; `header` was folded into one of them.
It is reported separately here because counting boilerplate as "unmarked semantic divergence"
overstates the interesting number.

TWO GRANULARITIES - they answer different questions and differ by ~3x on "unmarked":

  hunk   (canonical-v2 compatible; manifest Appendix A says "hunk-granular") - every added line
         of a diff hunk takes the bucket of the strongest marker found ANYWHERE in that hunk:
         header if all lines are header; else guard if any line is inside a guard construct or
         names a guard; else sidecar if any line matches the sidecar regex; else unmarked.
         Generous: a ROCm-only feature's unguarded config lines count as guard because they
         share a hunk with the guard. Measures "how much added code sits in marked hunks".
  line   (strict) - each added line is bucketed on its own: guard only if it is literally the
         condition/body/else of a guard construct (AST). Measures "how many lines are
         syntactically gated". Everything a feature adds outside its `if` is unmarked.

The manifest records BOTH: `added_class` (hunk, comparable to v2.3) and `added_class_line`.

Base selection and diff tool follow tools/measure_divergence.sh (GNU diff; base = submodule pin).

Usage:
  classify_hunks.py --base <sha> [--json out.json] [--update-manifest divergence-manifest.yaml]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- rules -----

# Names that, appearing in an `if` / ternary condition, make the whole construct a ROCm guard.
GUARD_NAMES = {
    "IS_HIP_EXTENSION", "_IS_HIP_EXTENSION", "is_hip_extension",
    "rocm_build", "te_rocm_build", "is_hip", "is_rocm",
    "is_mi200", "is_mi300", "is_mi308", "is_fp8_fnuz", "use_hipkittens",
    "build_rocm_version", "rocm_path", "rocm_version",
}
# Fallback: any identifier in the condition that contains one of these, case-insensitive.
GUARD_SUBSTR = re.compile(r"rocm|hip|gfx9|fnuz|aiter", re.I)
# Environment-compat conditions (JAX/torch/Triton version branches). canonical-v2 counted these
# as "guarded branches" (verified on JX-002 sharding.py: 74 guard lines, zero ROCm conditions,
# all `version.parse(jax.__version__) < ...`). They gate on the environment like a ROCm guard
# does, and the manifest disposes them as compat-pr / capability. Folded into `guard`.
COMPAT_NAMES = {"version", "__version__", "_TRITON_VERSION", "torch_version", "jax_version",
                "get_device_compute_capability", "is_mesh_available"}
# A ROCm marker in a COMMENT. In hunk mode a hunk carrying one is "marked" (canonical-v2 did
# this: PT-018 constants.py, 30 guard lines whose only ROCm marker is a comment). In line mode
# only the comment line itself is marked; the code around it stays unmarked (strict).
COMMENT_MARK_RE = re.compile(r"#.*\b(ROCm|AMD|AMDGPU|HIP|hip|MI300|MI308|MI200|MI350|gfx9\d\d|fnuz|CDNA|aiter)\b")

# "Sidecar wiring" as the proposal defines it: plumbing for the ROCm-only Python sidecar modules
# (MXFP4 stack, triton_kernels, fsdp2 allgather). Backend NAMES (aiter, ck_fused_attn, aotriton,
# hipkittens, rocshmem) are deliberately NOT sidecar: they appear in backend-selection code, which
# canonical-v2 counted as guard (verified on PT-012: 10 aiter/ck lines were guard in v2.3).
SIDECAR_RE = re.compile(
    r"\b(MXFP4\w*|mxfp4\w*|triton_kernels|fsdp2_allgather\w*|FSDPAGTensor|te_quantize_triton|"
    r"check_mxfp4_support|is_mxfp4_available)\b"
)

HEADER_RE = re.compile(
    r"^\s*#\s*(This file (was|is) (modified|from).*(AMDGPU|aiter|ROCm)|Copyright \(c\) \d{4}(-\d{4})?, Advanced Micro Devices|"
    r"License for AMD contributions|See LICENSE for license information)",
)

# ------------------------------------------------------------- plumbing -----

def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def modified_py(base: str) -> list[str]:
    out = sh("git", "diff", "--name-only", "--diff-filter=M", base, "HEAD", "--",
             "transformer_engine/*.py", "transformer_engine/**/*.py")
    return [l for l in out.splitlines() if l.endswith(".py")]


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def added_lines(base: str, path: str, tmp: Path) -> tuple[list[tuple[int, str]], dict[int, int]]:
    """Returns ([(new_file_lineno, text)], {new_file_lineno: hunk_id}) via GNU diff.
    Line counts match measure_divergence.sh ('^>' count == number of added lines)."""
    up = tmp / "up.py"
    up.write_text(sh("git", "show", f"{base}:{path}"))
    p = subprocess.run(["diff", "--unchanged-line-format=", "--old-line-format=",
                        "--new-line-format=%dn|%L", str(up), path], capture_output=True, text=True)
    out = []
    for raw in p.stdout.splitlines():
        n, _, text = raw.partition("|")
        out.append((int(n), text))
    # Hunk membership from the unified header lines (-U0: hunks are maximal runs of change).
    hunk_of: dict[int, int] = {}
    u = subprocess.run(["diff", "-U0", str(up), path], capture_output=True, text=True).stdout
    hid = 0
    for line in u.splitlines():
        m = HUNK_RE.match(line)
        if not m:
            continue
        hid += 1
        start = int(m.group(1)); count = int(m.group(2)) if m.group(2) is not None else 1
        for ln in range(start, start + count):
            hunk_of[ln] = hid
    return out, hunk_of


# ------------------------------------------------------------ guard AST -----

def _cond_is_guard(node: ast.AST, src: str) -> bool:
    seg = ast.get_source_segment(src, node) or ""
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", seg))
    if names & GUARD_NAMES or names & COMPAT_NAMES:
        return True
    return any(GUARD_SUBSTR.search(n) for n in names)


def guard_ranges(src: str, path: str) -> tuple[set[int], set[int]]:
    """Line numbers inside ROCm guards: (branch_lines, else_lines). Condition lines count as branch."""
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        print(f"  ! {path}: {e}", file=sys.stderr)
        return set(), set()
    branch: set[int] = set()
    orelse: set[int] = set()

    def span(nodes):
        s = set()
        for n in nodes:
            if hasattr(n, "lineno"):
                s.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
        return s

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _cond_is_guard(node.test, src):
            branch.update(range(node.test.lineno, (node.test.end_lineno or node.test.lineno) + 1))
            branch |= span(node.body)
            orelse |= span(node.orelse)
        elif isinstance(node, ast.IfExp) and _cond_is_guard(node.test, src):
            branch.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.Try):
            # `try: from ..triton_kernels import x  except ImportError: ...` is a sidecar import
            # guard; leave to the sidecar regex (it names the module).
            pass
    return branch, orelse


# --------------------------------------------------------------- classify ---

GUARD_TEXT_RE = re.compile(r"\b(" + "|".join(sorted(GUARD_NAMES)) + r")\b")


def _line_bucket(n: int, text: str, branch: set[int], orelse: set[int]) -> str:
    if HEADER_RE.match(text):
        return "header"
    if n in branch:
        return "guard"
    if n in orelse:
        return "guard_else"
    if SIDECAR_RE.search(text):
        return "sidecar"
    if text.lstrip().startswith("#") and COMMENT_MARK_RE.search(text):
        return "guard"          # the marker comment line itself; strict mode marks only this line
    return "unmarked"


def classify_file(base: str, path: str, tmp: Path) -> dict:
    src = Path(path).read_text()
    branch, orelse = guard_ranges(src, path)
    lines, hunk_of = added_lines(base, path, tmp)

    # --- line granularity (strict) ---
    line_counts = {"header": 0, "guard": 0, "guard_else": 0, "sidecar": 0, "unmarked": 0}
    unmarked_lines: list[tuple[int, str]] = []
    per_line = {}
    for n, text in lines:
        b = _line_bucket(n, text, branch, orelse)
        per_line[n] = b
        line_counts[b] += 1
        if b == "unmarked":
            unmarked_lines.append((n, text.rstrip()))

    # --- hunk granularity (canonical-v2 compatible) ---
    hunks: dict[int, list[tuple[int, str]]] = {}
    for n, text in lines:
        hunks.setdefault(hunk_of.get(n, -n), []).append((n, text))
    hunk_counts = {"header": 0, "guard": 0, "sidecar": 0, "unmarked": 0}
    for hl in hunks.values():
        buckets = {per_line[n] for n, _ in hl}
        texts = "\n".join(t for _, t in hl)
        # Precedence sidecar > guard reproduces canonical-v2 (verified on PT-004: v2.3 146/31 vs
        # 106/71 with guard-first - identical sum, so v2.3 ranked sidecar first).
        if buckets == {"header"}:
            b = "header"
        elif "sidecar" in buckets or SIDECAR_RE.search(texts):
            b = "sidecar"
        elif buckets & {"guard", "guard_else"} or GUARD_TEXT_RE.search(texts) or COMMENT_MARK_RE.search(texts):
            b = "guard"         # any ROCm/compat condition OR a ROCm comment anywhere in the hunk
        else:
            b = "unmarked"
        hunk_counts[b] += len(hl)

    total = len(lines)
    assert total == sum(line_counts.values()) == sum(hunk_counts.values())
    return {"path": path, "added": total, "counts": line_counts, "hunk_counts": hunk_counts,
            "hunks": len(hunks), "unmarked_lines": unmarked_lines}


# ------------------------------------------------------- manifest update ----

def update_manifest(manifest: Path, results: dict[str, dict]) -> int:
    """Replace `added_class: REGENERATE` under each matching `path:` with the computed triple.
    Line-based so the file's layout and comments survive. Returns the number of entries updated."""
    lines = manifest.read_text().split("\n")
    updated = 0
    cur_path = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s{4}path:\s*(\S+)\s*$", line)
        if m:
            cur_path = m.group(1)
            continue
        if cur_path in results and re.match(r"^\s{4}added_class:\s*REGENERATE\s*$", line):
            h = results[cur_path]["hunk_counts"]
            c = results[cur_path]["counts"]
            guard_l = c["guard"] + c["guard_else"]
            lines[i] = (f"    added_class: {{header: {h['header']}, sidecar: {h['sidecar']}, "
                        f"guard: {h['guard']}, unmarked: {h['unmarked']}}}"
                        f"   # hunk-granular (canonical-v2 compatible); classify_hunks.py\n"
                        f"    added_class_line: {{header: {c['header']}, sidecar: {c['sidecar']}, "
                        f"guard: {guard_l}, unmarked: {c['unmarked']}}}"
                        f"   # line-granular (strict); guard incl. {c['guard_else']} else-branch lines")
            updated += 1
    text = "\n".join(lines)
    if "added_class: REGENERATE" not in text:
        text = text.replace("added_class_status: MUST_REGENERATE", "added_class_status: CURRENT")
    manifest.write_text(text)
    return updated


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--json")
    ap.add_argument("--update-manifest", metavar="YAML")
    ap.add_argument("--show-unmarked", metavar="PATH", help="print the unmarked lines of one file")
    args = ap.parse_args()

    root = Path(sh("git", "rev-parse", "--show-toplevel").strip())
    import os; os.chdir(root)
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    results = {}
    for path in modified_py(args.base):
        results[path] = classify_file(args.base, path, tmp)

    def layer(p):
        if p.startswith("transformer_engine/pytorch/"): return "pytorch"
        if p.startswith("transformer_engine/jax/"): return "jax"
        return "root/common"

    print(f"base: {args.base}\n")
    print(f"{'':60s} {'':>6s} | {'HUNK (canonical-v2)':^27s} | {'LINE (strict)':^27s}")
    print(f"{'FILE':60s} {'added':>6s} | {'hdr':>4s} {'guard':>6s} {'side':>5s} {'unmk':>5s}   "
          f"| {'hdr':>4s} {'guard':>6s} {'side':>5s} {'unmk':>5s}")
    keys = ("header", "guard", "sidecar", "unmarked")
    totH = dict.fromkeys(keys, 0); totL = dict.fromkeys(keys, 0); tot_added = 0
    per_layer: dict[str, dict] = {}
    for path, r in sorted(results.items(), key=lambda kv: -kv[1]["added"]):
        h = r["hunk_counts"]; c = r["counts"]
        l = {"header": c["header"], "guard": c["guard"] + c["guard_else"], "sidecar": c["sidecar"], "unmarked": c["unmarked"]}
        print(f"{path.replace('transformer_engine/', ''):60s} {r['added']:6d} | {h['header']:4d} {h['guard']:6d} "
              f"{h['sidecar']:5d} {h['unmarked']:5d}   | {l['header']:4d} {l['guard']:6d} {l['sidecar']:5d} {l['unmarked']:5d}")
        tot_added += r["added"]
        L = per_layer.setdefault(layer(path), {"added": 0, "H": dict.fromkeys(keys, 0), "L": dict.fromkeys(keys, 0)})
        L["added"] += r["added"]
        for k in keys:
            totH[k] += h[k]; totL[k] += l[k]; L["H"][k] += h[k]; L["L"][k] += l[k]
    print("-" * 120)
    print(f"{'TOTAL':60s} {tot_added:6d} | {totH['header']:4d} {totH['guard']:6d} {totH['sidecar']:5d} {totH['unmarked']:5d}   "
          f"| {totL['header']:4d} {totL['guard']:6d} {totL['sidecar']:5d} {totL['unmarked']:5d}")
    print("\nper layer:")
    for k, L in per_layer.items():
        H, Ll = L["H"], L["L"]
        print(f"  {k:12s} added={L['added']:5d}  HUNK guard={H['guard']:5d} side={H['sidecar']:4d} unmk={H['unmarked']:5d}"
              f"   LINE guard={Ll['guard']:5d} side={Ll['sidecar']:4d} unmk={Ll['unmarked']:5d}")

    if args.show_unmarked:
        r = results.get(args.show_unmarked)
        if r:
            print(f"\nunmarked lines in {args.show_unmarked}:")
            for n, t in r["unmarked_lines"]: print(f"  {n:5d}  {t[:100]}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {p: {"added": r["added"], "hunks": r["hunks"], "hunk_counts": r["hunk_counts"], "line_counts": r["counts"]}
             for p, r in results.items()}, indent=2))
        print(f"\njson: {args.json}")
    if args.update_manifest:
        n = update_manifest(Path(args.update_manifest), results)
        print(f"manifest entries updated: {n}")


if __name__ == "__main__":
    main()
