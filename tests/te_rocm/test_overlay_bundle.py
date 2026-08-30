# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Overlay bundle conformance (plan P6 / proposal sec 3.4): when running from an assembled
overlay, the recorded bundle hash must match the pin + the active patch queue, every applied
patch must be a live manifest entry, and the pin must be the submodule's HEAD. Skipped when
running from the fork's own tree."""
import hashlib
import json
import subprocess

import pytest
import yaml

from conftest import MANIFEST, REPO, SUBMODULE, overlay_root


@pytest.fixture(scope="module")
def overlay():
    root = overlay_root()
    if root is None:
        pytest.skip("not running from an assembled overlay")
    return root, json.loads((root / "overlay-manifest.json").read_text())


def test_pin_matches_submodule_and_manifest(overlay):
    _, om = overlay
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=SUBMODULE).stdout.strip()
    msha = yaml.safe_load(MANIFEST.read_text())["metadata"]["upstream_sha"]
    assert om["upstream_sha"] == head == msha, (om["upstream_sha"], head, msha)


def test_bundle_hash_recomputes(overlay):
    _, om = overlay
    patches_dir = REPO / "proposals" / "te-rocm-plugin" / "patches"
    parts = []
    for p in om["patches"]:
        f = patches_dir / f"{p['id']}.patch"
        assert f.exists(), f"applied patch {p['id']} is not in the active queue (retired or removed?)"
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert sha == p["sha256"], f"{p['id']}: patch file changed since the overlay was built"
        parts.append(f"{p['id']} {sha}")
    recomputed = hashlib.sha256((om["upstream_sha"] + "\n" + "\n".join(parts)).encode()).hexdigest()
    assert recomputed == om["bundle_hash"]


def test_applied_patches_are_live_manifest_entries(overlay):
    _, om = overlay
    entries = {e["id"]: e for e in yaml.safe_load(MANIFEST.read_text())["entries"]}
    for p in om["patches"]:
        e = entries.get(p["id"])
        assert e is not None, f"{p['id']}: not in the manifest"
        st = str(e.get("p5_status", ""))
        assert not st.startswith("retired"), f"{p['id']}: applied but manifest says {st}"


def test_no_stray_edits_in_the_overlay(overlay):
    """The overlay must be exactly upstream + queue: rebuild to a temp dir and compare Python."""
    root, om = overlay
    import tempfile, filecmp
    tmp = tempfile.mkdtemp(prefix="overlay-verify-")
    # rebuild with the SAME patch set the overlay records: every active patch not applied is excluded
    applied = {p["id"] for p in om["patches"]}
    active = {p.stem for p in (REPO / "proposals/te-rocm-plugin/patches").glob("*.patch")}
    excl = []
    for pid in sorted(active - applied):
        excl += ["--exclude", pid]
    r = subprocess.run(["python3", str(REPO / "proposals/te-rocm-plugin/tools/assemble_overlay.py"), "--out", tmp, *excl, "build"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    diffs = []
    for py in (root / "transformer_engine").rglob("*.py"):
        rel = py.relative_to(root)
        other = tmp / rel if False else __import__("pathlib").Path(tmp) / rel
        if rel.name == "_rocm_init.py":
            continue
        if not other.exists() or not filecmp.cmp(py, other, shallow=False):
            diffs.append(str(rel))
    assert not diffs, f"overlay Python differs from a clean rebuild (in-place edit?): {diffs[:10]}"
