# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Assemble the pure-Python overlay: pinned upstream tree + the build-tier patch queue (plan P4).

  overlay = upstream transformer_engine/ (Python + data, no C++) at the submodule pin
          + fork-only files that have no upstream ancestor (triton_kernels, MXFP4 stack, ...)
          + patches/<MANIFEST-ID>.patch applied in dependency order
          + generated _rocm_init.py and symlinks to the built binaries, so the overlay imports
            against the SAME compiled extension the fork uses (EXIT-B isolates the Python layer)

Subcommands
  check   assert submodule pin == manifest upstream_sha; report every patch's applicability
          against a clean upstream tree (git apply --check) and every manifest file entry's
          target presence. Exit 1 on any failure. No output tree.
  build   assemble into --out (default build/overlay); write overlay-manifest.json with the
          bundle hash; py_compile the tree.
  gen     seed patches/<ID>.patch from the fork's CURRENT divergence for one manifest entry
          (or --all). The seed is the whole-file diff; P5's work is to shrink each one
          ('try upstream unchanged first'), split compound entries, and retire.
  diff-fork  compare the assembled overlay's Python against the fork's own tree. With whole-file
          seed patches this must be empty - the assembler's self-test.

Patch format: unified diff (paths a/<path> b/<path>) preceded by a header the assembler
validates:  # manifest: <ID>  # base: <sha>  # mechanism: ...  # expiry: ...  # tests: ...  # owner: ...
A patch whose ID is not in the manifest is refused. Applicability in the prototype is
`git apply --check`; the AST-level fingerprint is Stage 3 (S3.1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROP = HERE.parent
ROOT = PROP.parent.parent                      # repo root
SUBMODULE = "3rdparty/transformer_engine_nvidia"
MANIFEST = PROP / "divergence-manifest.yaml"
PATCHES = PROP / "patches"
CXX_EXT = {".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc", ".hip", ".cmake", ".inc", ".version"}   # .version = linker script
CXX_NAMES = {"CMakeLists.txt"}
HEADER_KEYS = ("manifest", "base", "mechanism", "expiry", "tests", "owner")


def sh(*a, cwd=None, check=True) -> str:
    r = subprocess.run(a, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f"command failed: {' '.join(map(str, a))}\n{r.stderr}")
    return r.stdout


def die(msg):  # noqa
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(1)


# ------------------------------------------------------------------ manifest --

def load_manifest():
    d = yaml.safe_load(MANIFEST.read_text())
    entries = {e["id"]: e for e in d["entries"]}
    return d, entries


def file_entries(entries) -> dict[str, dict]:
    """Manifest entries that are patch targets: a real path under transformer_engine/ that
    exists upstream (CM-004 & friends are ROCm-only copies, not patches)."""
    out = {}
    for eid, e in entries.items():
        p = e.get("path", "")
        if (p.startswith("transformer_engine/") and " " not in p
                and e.get("metric_class") != "migration_volume"
                and not is_cxx(Path(p))):          # C++ targets (ABI-002 pybind_helper.h) belong to the Stage-4 C++ queue
            out[eid] = e
    return out


# ------------------------------------------------------------------ upstream --

def ensure_submodule(expected_sha: str) -> str:
    """Initialize the opt-in submodule explicitly and non-recursively; assert the pin."""
    sm = ROOT / SUBMODULE
    if not (sm / "transformer_engine").exists():
        sh("git", "submodule", "update", "--init", "--checkout", SUBMODULE, cwd=ROOT)
    head = sh("git", "rev-parse", "HEAD", cwd=sm).strip()
    if head != expected_sha:
        die(f"submodule HEAD {head[:12]} != manifest upstream_sha {expected_sha[:12]} - "
            f"move the pin AND the manifest together")
    return head


def is_cxx(rel: Path) -> bool:
    return rel.suffix in CXX_EXT or rel.name in CXX_NAMES


def upstream_files(sha: str) -> list[str]:
    out = sh("git", "ls-tree", "-r", "--name-only", sha, "transformer_engine", cwd=ROOT / SUBMODULE)
    return [l for l in out.splitlines() if not is_cxx(Path(l))]


def fork_only_files(sha: str) -> list[str]:
    """Files under transformer_engine/ in the fork with no upstream ancestor (non-C++)."""
    out = sh("git", "diff", "--name-only", "--diff-filter=A", sha, "HEAD", "--", "transformer_engine/", cwd=ROOT)
    return [l for l in out.splitlines() if not is_cxx(Path(l)) and (ROOT / l).exists()]


def materialize_upstream(sha: str, dest: Path):
    """Copy upstream's non-C++ transformer_engine/ tree at <sha> into dest."""
    files = upstream_files(sha)
    sm = ROOT / SUBMODULE
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.run(["git", "show", f"{sha}:{rel}"], capture_output=True, cwd=sm, check=True).stdout)
    return files


# ------------------------------------------------------------------- patches --

def parse_header(text: str) -> dict:
    hdr = {}
    for line in text.splitlines()[:12]:
        m = re.match(r"^#\s*(\w+):\s*(.*)$", line)
        if m:
            hdr[m.group(1)] = m.group(2).strip()
    missing = [k for k in HEADER_KEYS if k not in hdr]
    if missing:
        die(f"patch header missing {missing}")
    return hdr


def load_patches(entries, base_sha: str, exclude: set[str] = frozenset()) -> list[dict]:
    out = []
    for p in sorted(PATCHES.glob("*.patch")):
        if p.stem in exclude:
            continue                     # --exclude: build "without this patch" (P5 shrink), race-free
        text = p.read_text()
        hdr = parse_header(text)
        pid = hdr["manifest"]
        if pid != p.stem:
            die(f"{p.name}: header id {pid} != filename")
        if pid not in entries:
            die(f"{p.name}: manifest id {pid} not in manifest - refused")
        if hdr.get("tests", "").strip() in ("", "TBD"):
            die(f"{p.name}: '# tests:' must name at least one test (S3.1 governance)")
        if hdr["base"] != base_sha:
            print(f"  note: {p.name} was authored against {hdr['base'][:12]} (pin is {base_sha[:12]}) - certification mode, applicability decides")
        deps = [d for d in (entries[pid].get("dependencies") or []) if d in entries]
        out.append({"id": pid, "path": p, "header": hdr, "deps": deps, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return out


def topo(patches: list[dict]) -> list[dict]:
    ids = {p["id"] for p in patches}
    by = {p["id"]: p for p in patches}
    seen, order, stack = set(), [], set()

    def visit(pid):
        if pid in seen: return
        if pid in stack: die(f"dependency cycle at {pid}")
        stack.add(pid)
        for d in by[pid]["deps"]:
            if d in ids: visit(d)
        stack.discard(pid); seen.add(pid); order.append(by[pid])
    for p in sorted(patches, key=lambda x: x["id"]):
        visit(p["id"])
    return order


def apply_patch(patch: Path, tree: Path, check_only: bool) -> tuple[bool, str]:
    args = ["git", "apply", "--check" if check_only else "--verbose", "-p1", str(patch)]
    if not check_only:
        args = ["git", "apply", "-p1", str(patch)]
    r = subprocess.run(args, capture_output=True, text=True, cwd=tree)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


# ----------------------------------------------------------------------- gen --

def gen_patch(eid: str, entries, base_sha: str) -> Path | None:
    e = entries[eid]
    rel = e["path"]
    up = subprocess.run(["git", "show", f"{base_sha}:{rel}"], capture_output=True, cwd=ROOT / SUBMODULE)
    if up.returncode:
        die(f"{eid}: {rel} does not exist upstream at {base_sha[:12]} - not a patch target")
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(up.stdout); upname = f.name
    r = subprocess.run(["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}", upname, str(ROOT / rel)],
                       capture_output=True, text=True)
    os.unlink(upname)
    if r.returncode == 0:
        print(f"  {eid}: {rel} identical to upstream - no patch (retire as unchanged)")
        return None
    feats = e.get("features") or []
    tests = sorted({t for f in feats for t in (f.get("test_ids") or [])} | set(e.get("test_ids") or []))
    prior = PATCHES / f"{eid}.patch"          # regenerating: keep the curated '# tests:' header
    if not tests and prior.exists():
        m = re.search(r"^# tests: (.+)$", prior.read_text(), re.M)
        if m and m.group(1).strip() != "TBD":
            tests = [t.strip() for t in m.group(1).split(",")]
    tests = tests or ["TBD"]
    expiry = e.get("expiry_condition") or ("see features" if feats else "TBD")
    hdr = (f"# manifest: {eid}\n# base: {base_sha}\n# mechanism: {e.get('disposition')}\n"
           f"# expiry: {expiry}\n# tests: {', '.join(tests)}\n# owner: {e.get('owner', 'none')}\n"
           f"# seeded: {datetime.now(timezone.utc).date()} whole-file diff of the fork vs upstream; "
           f"P5 shrinks/splits this ({len(feats)} feature sub-entries)\n")
    PATCHES.mkdir(exist_ok=True)
    out = PATCHES / f"{eid}.patch"
    out.write_text(hdr + r.stdout)
    added = sum(1 for l in r.stdout.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in r.stdout.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"  {eid}: {out.name}  (+{added}/-{removed})  {rel}")
    return out


# --------------------------------------------------------------------- build --

def build(args, d, entries, base_sha: str, check_only: bool):
    out = Path(args.out).resolve()
    tree = Path(tempfile.mkdtemp(prefix="overlay-check-")) if check_only else out
    if not check_only:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    up_files = materialize_upstream(base_sha, tree)
    print(f"upstream tree : {len(up_files)} non-C++ files from {SUBMODULE}@{base_sha[:12]}")

    copied = fork_only_files(base_sha)
    for rel in copied:
        t = tree / rel; t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, t)
    print(f"fork-only     : {len(copied)} files copied (no upstream ancestor)")

    fe = file_entries(entries)
    present = [eid for eid, e in fe.items() if (tree / e["path"]).exists()]
    absent = [eid for eid in fe if eid not in present]
    print(f"targets       : {len(present)} manifest file entries have their upstream target present"
          + (f"; ABSENT: {absent}" if absent else ""))

    patches = topo(load_patches(entries, base_sha, exclude=set(getattr(args, "exclude", None) or [])))
    if getattr(args, "exclude", None):
        print(f"excluded      : {', '.join(args.exclude)}")
    ok_n, bad = 0, []
    for p in patches:
        ok, msg = apply_patch(p["path"], tree, check_only=check_only)
        if ok:
            ok_n += 1
        else:
            bad.append((p["id"], msg.splitlines()[-1] if msg else "?"))
    print(f"patch queue   : {len(patches)} patches, {ok_n} {'applicable' if check_only else 'applied'}"
          + (f", {len(bad)} TRIPPED" if bad else ""))
    for pid, msg in bad:
        print(f"  TRIPPED {pid}: {msg[:110]}")

    if check_only:
        shutil.rmtree(tree, ignore_errors=True)
        sys.exit(1 if bad or absent else 0)
    if bad:
        die("tripped patches - overlay not built")

    # generated + binaries so the overlay imports against the fork's compiled extension
    gen = ROOT / "transformer_engine" / "_rocm_init.py"
    if gen.exists():
        shutil.copy2(gen, tree / "transformer_engine" / "_rocm_init.py")
    linked = []
    for so in list(ROOT.glob("libtransformer_engine*.so")) + list(ROOT.glob("transformer_engine_rocm_torch*.so")):
        (tree / so.name).symlink_to(so); linked.append(so.name)
    libdir = ROOT / "transformer_engine" / "lib"
    if libdir.is_dir():
        (tree / "transformer_engine" / "lib").symlink_to(libdir, target_is_directory=True); linked.append("transformer_engine/lib/")
    print(f"binaries      : {len(linked)} symlinked ({', '.join(linked)})")

    # compile
    n = 0
    for py in tree.rglob("*.py"):
        py_compile.compile(str(py), doraise=True); n += 1
    print(f"py_compile    : {n} files OK")

    bundle = hashlib.sha256((base_sha + "\n" + "\n".join(f"{p['id']} {p['sha256']}" for p in patches)).encode()).hexdigest()
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream_sha": base_sha, "submodule": SUBMODULE,
        "manifest_version": d["metadata"]["manifest_version"],
        "patches": [{"id": p["id"], "sha256": p["sha256"], "mechanism": p["header"]["mechanism"]} for p in patches],
        "fork_only_files": copied, "upstream_files": len(up_files), "binaries": linked,
        "bundle_hash": bundle,
    }
    (tree / "overlay-manifest.json").write_text(json.dumps(manifest, indent=1))
    # Second copy INSIDE the package: the pure-Python wheel (setup.py NVTE_BUILD_OVERLAY=1,
    # plan S3.4) packages only transformer_engine/, so provenance must live there - a loose
    # site-packages file would be pollution. Diagnostics reads either location.
    (tree / "transformer_engine" / "_overlay_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"bundle hash   : {bundle[:16]}...  -> {tree / 'overlay-manifest.json'} (+ package copy)")
    print(f"\nuse: PYTHONPATH={tree} python -c 'import transformer_engine.pytorch'")


def _tree_files(root: Path, suffixes: set[str] | None) -> dict[str, Path]:
    out = {}
    for p in root.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "_rocm_init.py":
            continue
        rel = str(p.relative_to(root))
        if p.suffix == ".so" or rel.startswith("lib/") or is_cxx(p):
            continue
        if suffixes is None or p.suffix in suffixes:
            out[rel] = p
    return out


def diff_fork(args):
    """Self-test: with whole-file seed patches the overlay's PYTHON must equal the fork's.
    Non-Python data files (MANIFEST.in, pyproject.toml, ...) are packaging inputs the Python queue
    does not govern; they are reported, not counted (S3.4)."""
    out = Path(args.out).resolve()
    ov = _tree_files(out / "transformer_engine", {".py"})
    fk = _tree_files(ROOT / "transformer_engine", {".py"})
    diffs = []
    for rel in sorted(set(ov) | set(fk)):
        if rel not in ov: diffs.append(f"missing in overlay: {rel}")
        elif rel not in fk: diffs.append(f"only in overlay:    {rel}")
        elif ov[rel].read_bytes() != fk[rel].read_bytes(): diffs.append(f"content differs:    {rel}")
    print(f"overlay vs fork PYTHON: {len(ov)} overlay / {len(fk)} fork .py files, {len(diffs)} differences")
    for l in diffs[:40]: print("  " + l)
    # informational: data files
    ovd = _tree_files(out / "transformer_engine", None); fkd = _tree_files(ROOT / "transformer_engine", None)
    data = [rel for rel in sorted(set(ovd) | set(fkd)) if not rel.endswith(".py")
            and (rel not in ovd or rel not in fkd or ovd[rel].read_bytes() != fkd[rel].read_bytes())]
    if data:
        print(f"data files differing (not governed by the Python queue; packaging, S3.4): {len(data)}")
        for rel in data[:12]: print("  " + rel)
    sys.exit(1 if diffs else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "overlay"))
    ap.add_argument("--exclude", action="append", metavar="ID", help="leave this patch out (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("build"); sub.add_parser("diff-fork")
    g = sub.add_parser("gen"); g.add_argument("ids", nargs="*"); g.add_argument("--all", action="store_true")
    g.add_argument("--verify", action="store_true",
                   help="stale check (the in-fork-phase freeze invariant, plan G3/S3.6): regenerate "
                        "each ACTIVE patch to a temp file and compare bodies; a mismatch means the "
                        "fork tree was edited without regenerating the patch. Exit 1 on staleness.")
    args = ap.parse_args()

    d, entries = load_manifest()
    base_sha = d["metadata"]["upstream_sha"]
    if args.cmd == "diff-fork":
        return diff_fork(args)
    head = ensure_submodule(base_sha)
    print(f"pin           : {SUBMODULE}@{head[:12]} == manifest upstream_sha  OK")
    if args.cmd == "gen":
        if args.verify:
            # Staleness = the true invariant, tested directly: applying the checked-in patch to
            # the pinned upstream file must reproduce the fork tree byte-for-byte. (Comparing
            # diff texts is unsound - Myers and difflib may split hunks differently for the
            # same edit.) A mismatch means the fork tree was edited without `gen`.
            stale = []
            patch_files = sorted(PATCHES.glob("*.patch"))
            for p in patch_files:
                if args.ids and p.stem not in args.ids:
                    continue
                body = "\n".join(l for l in p.read_text().splitlines() if not l.startswith("#")) + "\n"
                rel = next((l[6:] for l in body.splitlines() if l.startswith("--- a/")), None)
                up = subprocess.run(["git", "show", f"{base_sha}:{rel}"], capture_output=True,
                                    cwd=ROOT / SUBMODULE)
                if up.returncode or not (ROOT / rel).exists():
                    stale.append((p.stem, f"{rel}: target missing")); continue
                with tempfile.TemporaryDirectory() as td:
                    tgt = Path(td) / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
                    tgt.write_bytes(up.stdout)
                    (Path(td) / "p.patch").write_text(body)
                    ap_ = subprocess.run(["git", "apply", "--whitespace=nowarn", "p.patch"],
                                         capture_output=True, cwd=td)
                    if ap_.returncode:
                        stale.append((p.stem, f"{rel}: does not apply at pin")); continue
                    if tgt.read_bytes() != (ROOT / rel).read_bytes():
                        stale.append((p.stem, f"{rel}: patched upstream != fork tree"))
            for pid, why in stale:
                print(f"  STALE {pid}: {why} - regenerate with `gen {pid}`")
            print(f"stale-check: {len(stale)} stale of {len(patch_files)} active")
            sys.exit(1 if stale else 0)
        ids = list(file_entries(entries)) if args.all else args.ids
        if not ids: die("gen: give manifest ids or --all")
        for eid in ids:
            gen_patch(eid, entries, base_sha)
        return
    build(args, d, entries, base_sha, check_only=(args.cmd == "check"))


if __name__ == "__main__":
    main()
